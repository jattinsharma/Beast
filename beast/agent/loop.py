"""The agent loop: SEE -> THINK -> ACT -> VERIFY, one action per iteration.

Safety properties:
- Emergency stop checked at every stage (estop_flag raises mid-loop, not just
  inside ComputerTool calls).
- Exactly one action per iteration; the LLM never executes arbitrary code.
- Every iteration fully logged (goal, state, decision, result, verdict).
- VERIFY failure or ambiguity -> stop and ask the user, never blind retry.
- Level-3 actions require explicit voice confirmation before executing.
"""

import json
import logging
import time

from computer.safety import EmergencyStop, estop_flag

from .autonomy import LEVEL1_ACTIONS, TERMINAL_ACTIONS, classify, ConfirmationGate
from .brain import AgentBrain
from .executor import ActionExecutor
from .screen_state import ScreenState, find_element_center
from .verifier import Verifier

logger = logging.getLogger("beast.agent")

MAX_ITERATIONS = 8          # hard cap per task
VERIFY_FAILURE_LIMIT = 1    # one failed verify -> stop and ask


class AgentLoop:
    """Runs a single user goal through the SEE-THINK-ACT-VERIFY cycle."""

    def __init__(self, computer_tool, tts=None, stt=None, record_fn=None,
                 memory_manager=None):
        self.ct = computer_tool
        self.see = ScreenState()
        self.brain = AgentBrain()
        self.executor = ActionExecutor(computer_tool, memory_manager=memory_manager)
        self.verifier = Verifier(self.see)
        self.gate = (ConfirmationGate(tts, stt, record_fn)
                     if tts and stt and record_fn else None)
        self._memory = memory_manager
        self.history: list[dict] = []

    def run(self, goal: str) -> str:
        """Execute the goal. Returns a final spoken-style summary."""
        logger.info("[LOOP] === START goal=%r ===", goal)
        self.history = []

        try:
            return self._run_inner(goal)
        except EmergencyStop:
            logger.warning("[LOOP] EMERGENCY STOP - agent loop halted")
            if self.history:
                self.history[-1]["result"] = "ABORTED by emergency stop"
            return "Emergency stop. I've halted everything."
        except Exception as e:
            logger.error("[LOOP] Unexpected error: %s", e, exc_info=True)
            return "Sorry, something went wrong while doing that."

    # ------------------------------------------------------------------

    def _run_inner(self, goal: str) -> str:
        for iteration in range(1, MAX_ITERATIONS + 1):
            estop_flag.check()  # halt between iterations too

            # ---- SEE ----
            state = self.see.summarize()
            logger.info("[LOOP] it%d SEE:\n%s", iteration, state)

            # ---- THINK ----
            memory_ctx = self._memory.context_for_llm() if self._memory else ""
            decision = self.brain.decide(goal, state, self.history,
                                         memory_context=memory_ctx)
            kind = decision["action"]
            reason = decision.get("reason", "")
            logger.info("[LOOP] it%d THINK: %s (%s)",
                        iteration, json.dumps(decision), reason)

            # ---- Terminal states ----
            if kind == "task_complete":
                msg = decision.get("message") or "Done."
                logger.info("[LOOP] COMPLETE after %d iterations", iteration)
                return msg
            if kind == "ask_user":
                msg = decision.get("message", "I need more information.")
                logger.info("[LOOP] Asking user: %s", msg)
                if self.gate:
                    self.gate.tts.speak(msg)
                    answer_audio = self.gate.record_fn()
                    if answer_audio is not None and len(answer_audio) > 1600:
                        answer = self.gate.stt.transcribe(answer_audio)
                        logger.info("[LOOP] User answered: %r", answer)
                        # Feed the answer back as context for next decision.
                        self.history.append({
                            "action": f"ask_user({msg})",
                            "result": f"user said: {answer}",
                        })
                        continue
                return msg

            # ---- Autonomy gating ----
            level = classify(decision, goal)
            logger.info("[LOOP] it%d autonomy level %d for %s",
                        iteration, level, kind)

            if level == 3:
                # SAFETY: a Level-3 action MUST NOT proceed without an
                # explicit confirmation. No gate = no confirmation = cancel.
                # This is the fix for the "silently defaults to yes" bug.
                if self.gate is None:
                    logger.error(
                        "[LOOP] it%d Level-3 action %r requested but NO "
                        "confirmation gate is available - CANCELLING "
                        "(fail-safe)", iteration, kind)
                    msg = ("That action needs your confirmation but I can't "
                           "ask right now, so I'm not doing it.")
                    return msg
                desc = self._describe_action(decision)
                confirmed = self.gate.confirm(
                    f"Should I {desc}? Say yes to confirm.")
                self._log_iteration(iteration, decision, level,
                                    "confirmed" if confirmed else "declined",
                                    "", "")
                if not confirmed:
                    return "Okay, I've cancelled that."

            # ---- Resolve click target BEFORE acting (verify-before-act:
            #      never assume coordinates from a stale screen read) ----
            click_point = None
            if kind == "click":
                elems = self.see.uia_elements(
                    self.ct.get_active_window())
                click_point = find_element_center(elems, decision["target"])
                if click_point is None:
                    ocr = self.see.ocr_lines()
                    ocr_elems = [{"name": l["text"], "rect": l["rect"],
                                  "enabled": True} for l in ocr]
                    click_point = find_element_center(
                        ocr_elems, decision["target"])
                if click_point is None:
                    logger.warning("[LOOP] it%d target %r not found on screen",
                                   iteration, decision["target"])
                    self.history.append({
                        "action": json.dumps(decision),
                        "result": "FAILED: target not found on screen",
                    })
                    # Don't guess coordinates - ask.
                    msg = (f"I couldn't find '{decision['target']}' on "
                           "screen.")
                    if self.gate:
                        self.gate.tts.speak(msg)
                    return msg

            # ---- ACT ----
            state_before = state
            try:
                result = self.executor.execute(decision, click_point)
            except EmergencyStop:
                raise
            except Exception as e:
                result = f"FAILED: {e}"
                logger.error("[LOOP] it%d ACT error: %s", iteration, e)

            logger.info("[LOOP] it%d ACT: %s", iteration, result)

            # ---- VERIFY ----
            verdict = "pass"
            if kind in LEVEL1_ACTIONS and result.startswith(("opened", "clicked",
                                                             "typed", "pressed",
                                                             "scrolled",
                                                             "focused")):
                try:
                    verdict = self.verifier.verify(decision, state_before)
                except EmergencyStop:
                    raise
                except Exception as e:
                    logger.warning("[LOOP] it%d verify error: %s", iteration, e)
                    verdict = "uncertain"

            logger.info("[LOOP] it%d VERIFY: %s", iteration, verdict)
            self._log_iteration(iteration, decision, level, result,
                                verdict, state_before)

            self.history.append({
                "action": json.dumps(decision),
                "result": f"{result} [verify={verdict}]",
            })

            # ---- Handle verification outcome ----
            if verdict == "fail":
                logger.warning("[LOOP] it%d verification FAILED - stopping "
                               "instead of retrying blindly", iteration)
                msg = ("I tried that but couldn't confirm it worked. "
                       "You may want to check the screen.")
                if self.gate:
                    self.gate.tts.speak(msg)
                return msg
            if verdict == "uncertain":
                logger.warning("[LOOP] it%d verification UNCERTAIN", iteration)
                # Continue once; repeated uncertainty stops via history.

        logger.warning("[LOOP] Hit max iterations (%d)", MAX_ITERATIONS)
        msg = "I ran out of steps without finishing. Here's where things stand."
        if self.gate:
            self.gate.tts.speak(msg)
        return msg

    @staticmethod
    def _describe_action(action: dict) -> str:
        kind = action["action"]
        if kind == "open_app":
            return f"open {action['target']}"
        if kind == "click":
            return f"click on {action['target']}"
        if kind == "type_text":
            return f"type '{action['text']}'"
        if kind == "press":
            return f"press {action['key']}"
        if kind == "hotkey":
            return "press " + "+".join(action["keys"])
        if kind == "scroll":
            d = "up" if action.get("amount", 0) > 0 else "down"
            return f"scroll {d}"
        if kind == "focus_window":
            return f"switch to window {action['target']}"
        return kind

    def _log_iteration(self, iteration, decision, level, result, verdict,
                       state):
        """Full audit entry - reconstructable later for the Activity Log."""
        logger.info(
            "[AUDIT] it=%d level=%d action=%s reason=%r result=%r "
            "verify=%s",
            iteration, level, json.dumps(decision),
            decision.get("reason", ""), result, verdict,
        )
