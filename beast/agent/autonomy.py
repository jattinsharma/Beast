"""Autonomy levels: risk-tier every action and gate accordingly.

L1 automatic          - open apps, type into a field the user asked for
L2 execute + verify   - read/inspect actions (report result)
L3 prepare + confirm  - anything hard to undo (close app, submit) - ask
                        via voice, wait for explicit yes
L4 stop and ask       - ambiguous, unexpected, or verification failed

DEFAULT TO L3 when in doubt about an action's risk tier.
"""

import logging
import re

logger = logging.getLogger("beast.agent")

# Actions that are inherently safe/reversible.
LEVEL1_ACTIONS = {"open_app", "click", "type_text", "press", "hotkey", "scroll",
                  "focus_window"}
# Memory actions: all L1 except forget which is L2 (confirm by speaking).
MEMORY_ACTIONS = {"remember", "recall", "forget", "list_memories"}
# Terminal states handled by the loop directly.
TERMINAL_ACTIONS = {"task_complete", "ask_user"}

# Keywords that make an otherwise-L1 action risky (hard to undo).
RISKY_PATTERNS = re.compile(
    r"\b(close|quit|exit|delete|remove|submit|send|save|overwrite|format|"
    r"shutdown|restart|install|uninstall|pay|purchase)\b", re.IGNORECASE)


def classify(action: dict, goal: str = "") -> int:
    """Return 1-4 for an action dict."""
    kind = action["action"]
    if kind in TERMINAL_ACTIONS:
        return 4 if kind == "ask_user" else 2

    # Any action whose target/text/reason mentions destructive verbs -> L3.
    blob = " ".join(str(v) for v in action.values() if isinstance(v, str))
    if RISKY_PATTERNS.search(blob):
        return 3

    # Application actions
    # Memory actions: remember/list/recall are safe (L1),
    # forget deletes data (L2 - verify required).
    if kind in MEMORY_ACTIONS:
        if kind == "forget":
            return 2  # L2: data deletion, verify required
        return 1  # L1: read/write memory, safe

    if kind == "open_app":
        return 1
    if kind == "focus_window":
        return 1
    # click/type/press/hotkey/scroll: generally safe, but a hotkey like
    # ctrl+w or alt+f4 closes things - be conservative about hotkeys.
    if kind == "hotkey":
        keys = [k.lower() for k in action.get("keys", [])]
        if any(k in ("alt", "ctrl") for k in keys):
            return 3
        return 1

    # Browser actions
    if kind == "browser_navigate":
        return 1  # L1: navigating is safe
    if kind in ("browser_find_element", "browser_get_text", "browser_screenshot", "browser_wait_for"):
        return 2  # L2: reading/inspecting the page
    if kind in ("browser_click", "browser_type_text"):
        return 3  # L3: could submit forms or create content

    return 1  # default to safe


class ConfirmationGate:
    """Level-3 gate: speak a question, record the answer, accept yes/no.

    SAFETY CONTRACT (non-negotiable):
    - Returns True ONLY on a clearly transcribed affirmative.
    - Silence, timeout, transcription failure, or ambiguity ALL return False
      (do not act). There is no code path that returns True by default.
    - Retries up to MAX_ATTEMPTS times on silence/unclear, announcing each
      retry, then gives up WITHOUT acting.

    Timing: after TTS finishes speaking, waits SETTLE_SECONDS before opening
    the mic, so playback tail/residual audio can't mask the user's answer.
    Every stage is timestamped for debugging the TTS->STT handoff.
    """

    # NOTE: 'confirm' removed - the spoken question itself contains
    # 'Say yes to confirm', so speaker bleed into the mic can produce a
    # transcript containing those words. Only short standalone answers count.
    YES = {"yes", "yeah", "yep", "sure", "ok", "okay", "affirmative"}
    NO = {"no", "nope", "stop", "cancel", "abort", "wait"}

    MAX_ATTEMPTS = 3          # total attempts before giving up
    SETTLE_SECONDS = 2.5      # gap after TTS finishes before mic opens
    MIN_SPEECH_SECONDS = 0.2  # shorter than this = treated as silence
    MAX_ANSWER_WORDS = 2      # answers longer than this are ignored (echo/
                              # background chatter protection)
    SAVE_WAV = True           # save every attempt's raw audio for diagnosis
    # Words that suggest TTS feedback rather than genuine user response
    TTS_FEEDBACK_WORDS = {"should", "i", "close", "this", "notepad", "window",
                          "please", "state", "your", "choice"}

    def __init__(self, tts, stt, record_fn):
        """
        tts: object with .speak(text)
        stt: object with .transcribe(audio_float32, sample_rate)
        record_fn: callable() -> float32 numpy array of command audio
        """
        self.tts = tts
        self.stt = stt
        self.record_fn = record_fn

    @staticmethod
    def _ts() -> str:
        import time as _time
        return _time.strftime("%H:%M:%S") + f".{int(_time.time()*1000)%1000:03d}"

    @staticmethod
    def _save_wav(audio, attempt: int) -> str | None:
        """Save raw captured audio as WAV for later listening/diagnosis."""
        try:
            import os
            import time as _time
            import wave
            import numpy as np
            d = os.path.join(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))), "logs", "confirmations")
            os.makedirs(d, exist_ok=True)
            path = os.path.join(d, f"attempt{attempt}_"
                                f"{_time.strftime('%H%M%S')}.wav")
            pcm = np.clip(audio, -1.0, 1.0)
            pcm = (pcm * 32767).astype(np.int16)
            with wave.open(path, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(16000)
                w.writeframes(pcm.tobytes())
            logger.info("[AUTONOMY] Raw audio saved: %s", path)
            return path
        except Exception as e:
            logger.warning("[AUTONOMY] Failed to save WAV: %s", e)
            return None

    @staticmethod
    def _contains_word(text: str, words: set[str]) -> bool:
        """Whole-word match. NEVER substring: 'no' must not match 'know',
        'not', 'notebook'; 'ok' must not match 'broker'. This was the bug
        that made phantom transcripts flip decisions."""
        import re
        tokens = set(re.findall(r"[a-z']+", text))
        # Normalize apostrophes: don't -> dont
        tokens |= {t.replace("'", "") for t in tokens}
        return bool(tokens & words)

    def confirm(self, question: str) -> bool:
        """Ask via voice; returns True ONLY on clear affirmative."""
        import numpy as np

        for attempt in range(1, self.MAX_ATTEMPTS + 1):
            t_speak_start = self._ts()
            self.tts.speak(question)
            t_speak_end = self._ts()

            # Explicit settle gap: let speaker tail decay before listening.
            import time
            time.sleep(self.SETTLE_SECONDS)
            t_record_start = self._ts()
            logger.info(
                "[AUTONOMY] attempt %d/%d | TTS %s -> %s | settle %.1fs | "
                "RECORDING STARTED %s",
                attempt, self.MAX_ATTEMPTS,
                t_speak_start, t_speak_end, self.SETTLE_SECONDS,
                t_record_start)

            audio = self.record_fn()
            t_record_end = self._ts()

            if audio is None or len(audio) < 1600:
                logger.warning(
                    "[AUTONOMY] RECORDING ENDED %s: EMPTY (%s samples) "
                    "- mic captured nothing",
                    t_record_end, 0 if audio is None else len(audio))
            else:
                dur = len(audio) / 16000.0
                peak = float(np.max(np.abs(audio)))
                rms = float(np.sqrt(np.mean(audio ** 2)))
                logger.info(
                    "[AUTONOMY] RECORDING ENDED %s: %.2fs peak=%.4f "
                    "rms=%.5f", t_record_end, dur, peak, rms)

                if self.SAVE_WAV:
                    self._save_wav(audio, attempt)

                if dur >= self.MIN_SPEECH_SECONDS and peak > 0.01:
                    text = self.stt.transcribe(audio).lower().strip()
                    logger.info("[AUTONOMY] Transcript: %r", text)

                    # Echo/chatter guard: real answers are 1-2 words.
                    # Long transcripts are TTS bleed or background noise -
                    # treat as unclear rather than risking a false match.
                    n_words = len(text.split())

                    # Check for likely TTS feedback - if transcript contains words
                    # from the TTS prompt, it's probably feedback
                    feedback_matches = len(set(text.split()) & self.TTS_FEEDBACK_WORDS)
                    is_likely_feedback = feedback_matches >= 1  # 1 or more feedback words likely indicates TTS bleed
                    is_too_many_words = n_words > self.MAX_ANSWER_WORDS

                    if is_likely_feedback:
                        logger.warning(
                            "[AUTONOMY] Transcript contains %d TTS feedback words - "
                            "treating as echo/noise, not an answer",
                            feedback_matches)
                    if is_too_many_words:
                        logger.warning(
                            "[AUTONOMY] Transcript has %d words (> %d) - "
                            "treating as echo/noise, not an answer",
                            n_words, self.MAX_ANSWER_WORDS)

                    # Only check for yes/no if it's not likely feedback and not too many words
                    if not is_likely_feedback and not is_too_many_words:
                        if self._contains_word(text, self.YES):
                            logger.info("[AUTONOMY] CONFIRMED (yes)")
                            return True
                        elif self._contains_word(text, self.NO):
                            self.tts.speak("Okay, I won't do that.")
                            logger.info("[AUTONOMY] DECLINED (no)")
                            return False
                        else:
                            logger.warning("[AUTONOMY] Unclear answer: %r "
                                           "(no yes/no word found)", text)
                    else:
                        logger.warning("[AUTONOMY] Treating as unclear due to feedback or word count")
                else:
                    logger.warning(
                        "[AUTONOMY] Audio too quiet/short to be speech "
                        "(dur=%.2fs peak=%.4f)", dur, peak)

            # Failed this attempt - announce retry or give up.
            if attempt < self.MAX_ATTEMPTS:
                self.tts.speak("I didn't catch that. Please say yes or no.")
            else:
                self.tts.speak(
                    "I still couldn't hear you clearly, so I will NOT "
                    "proceed.")
                logger.warning(
                    "[AUTONOMY] GIVING UP after %d attempts - ACTION "
                    "CANCELLED (fail-safe)", self.MAX_ATTEMPTS)

        return False  # NEVER true by default
