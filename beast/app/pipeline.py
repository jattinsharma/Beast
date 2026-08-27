"""Voice pipeline for BEAST: STT (faster-whisper) -> LLM (router) -> TTS (Piper).

Each component is lazily loaded on first use and cached with an idle timeout
so back-to-back commands avoid reload cost while idle memory stays low.
"""

import glob
import logging
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Idle timeout before unloading models (seconds)
IDLE_TIMEOUT = 300  # 5 minutes

from .llm_router import ModelRouter


class WhisperSTT:
    """Speech-to-text via faster-whisper, int8 on CPU."""

    def __init__(self):
        self._model = None

    def _ensure_loaded(self):
        if self._model is None:
            logger.info("Loading Whisper model (base, int8, CPU)...")
            from faster_whisper import WhisperModel
            self._model = WhisperModel("base", device="cpu", compute_type="int8")
            logger.info("Whisper loaded")

    def transcribe(self, audio_float32: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe float32 mono audio. Returns text ('' if nothing heard)."""
        self._ensure_loaded()
        segments, info = self._model.transcribe(audio_float32, language="en", beam_size=5)
        return " ".join(seg.text.strip() for seg in segments).strip()


class PiperTTS:
    """Text-to-speech via Piper, played through speakers."""

    def __init__(self):
        self._voice = None
        base = Path(__file__).parent.parent / "piper_voices" / "en_US-lessac-medium.onnx"
        self.voice_path = str(base)

    def _ensure_loaded(self):
        if self._voice is None:
            logger.info(f"Loading Piper voice {self.voice_path}...")
            from piper import PiperVoice
            self._voice = PiperVoice.load(self.voice_path)
            logger.info("Piper loaded")

    def speak(self, text: str):
        """Synthesize and play speech (blocking until playback finishes)."""
        self._ensure_loaded()
        import sounddevice as sd
        for chunk in self._voice.synthesize(text):
            audio = np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16)
            sd.play(audio, samplerate=chunk.sample_rate)
            sd.wait()


# Max conversation turns before forced reset (each turn = user + assistant)
MAX_CONVERSATION_TURNS = 10


class VoicePipeline:
    """Orchestrates record -> STT -> LLM (with routing) -> TTS with lazy model caching."""

    def __init__(self, sample_rate: int = 16000, memory_manager=None):
        self.sample_rate = sample_rate
        self.stt = WhisperSTT()
        self.llm_router = ModelRouter()
        self.tts = PiperTTS()
        self._memory = memory_manager
        self._last_used = time.time()
        self._unload_timer = None
        self._lock = threading.Lock()

        # Conversation history: list of {role, content} dicts.
        # Survives across wake word triggers so Beast remembers context.
        # Cleared on idle timeout.
        self._history: list[dict] = []

    def process_command(self, audio_float32: np.ndarray) -> str:
        """Full chain: transcribe, get response (with model routing), speak it. Returns response text."""
        with self._lock:
            try:
                command = self.stt.transcribe(audio_float32, self.sample_rate)
                logger.info(f"[PIPELINE] Transcribed: {command!r}")
                if not command or len(command) < 2:
                    response = "I didn't catch that."
                else:
                    try:
                        # Route the command to the appropriate LLM
                        llm = self.llm_router.route(command)
                        # Build memory context for the LLM
                        memory_ctx = self._memory.context_for_llm() if self._memory else ""
                        # Pass conversation history for multi-turn context
                        response = llm.respond(
                            command,
                            memory_context=memory_ctx,
                            history=self._history,
                        )
                        # Record this exchange in conversation history
                        self._history.append({"role": "user", "content": command})
                        self._history.append({"role": "assistant", "content": response})
                        # Trim to max turns (each turn = 2 messages)
                        max_msgs = MAX_CONVERSATION_TURNS * 2
                        if len(self._history) > max_msgs:
                            self._history = self._history[-max_msgs:]
                    except Exception as e:
                        logger.error(f"[PIPELINE] LLM error: {e}")
                        response = "Sorry, I had trouble processing that."
                logger.info(f"[PIPELINE] Response: {response!r}")
                self.tts.speak(response)
                return response
            finally:
                self._touch()

    def _touch(self):
        """Reset the idle-unload timer."""
        self._last_used = time.time()
        if self._unload_timer is not None:
            self._unload_timer.cancel()
        self._unload_timer = threading.Timer(IDLE_TIMEOUT, self._unload_all)
        self._unload_timer.daemon = True
        self._unload_timer.start()

    def _unload_all(self):
        """Free model memory AND clear conversation history after idle timeout."""
        logger.info("[PIPELINE] Idle timeout - unloading models and clearing history")
        self.stt._model = None
        # Unload all routed models
        for model in getattr(self.llm_router, "_loaded_models", {}).values():
            model._llm = None
        self.tts._voice = None
        # Clear conversation history - fresh start after long idle
        self._history.clear()
        logger.info("[PIPELINE] Conversation history cleared")

    def clear_history(self):
        """Manually clear conversation history."""
        self._history.clear()
        logger.info("[PIPELINE] Conversation history cleared manually")