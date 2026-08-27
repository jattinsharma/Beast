"""VERIFY layer: after each action, re-check state for the expected effect.

Never blindly retries on failure - returns a verdict the loop acts on:
  "pass"          expected change observed
  "fail"          change clearly did NOT happen
  "uncertain"     can't tell (ambiguous) -> loop should stop and ask
"""

import logging
import time

from .screen_state import normalize_for_compare

logger = logging.getLogger("beast.agent")

# How long to wait for UI to settle before verifying.
VERIFY_SETTLE_SECONDS = 1.0


class Verifier:
    """Expected-effect checks per action type."""

    def __init__(self, screen_state):
        self.see = screen_state

    def verify(self, action: dict, state_before: str) -> str:
        """Re-capture state and compare against what this action should do."""
        time.sleep(VERIFY_SETTLE_SECONDS)
        kind = action["action"]

        if kind == "open_app":
            return self._verify_open_app(action["target"])
        if kind == "focus_window":
            return self._verify_focus(action["target"])
        if kind == "click":
            return self._verify_click(state_before)
        if kind == "type_text":
            return self._verify_typed(action["text"])
        if kind in ("press", "hotkey", "scroll"):
            # Effects are context-dependent; treat as uncertain unless the
            # loop has a specific expectation (kept simple for M4 scope).
            return "uncertain"
        if kind in ("task_complete", "ask_user"):
            return "pass"
        return "uncertain"

    def _verify_open_app(self, app_name: str) -> str:
        from computer.computer_tool import ComputerTool
        active = ComputerTool.get_active_window() or ""
        if normalize_for_compare(app_name) in normalize_for_compare(active):
            logger.info("[VERIFY] open_app OK: active window %r matches %r",
                        active, app_name)
            return "pass"
        # Also accept any visible window whose title contains the app name.
        try:
            titles = ComputerTool.list_windows()
            if any(normalize_for_compare(app_name)
                   in normalize_for_compare(t) for t in titles):
                logger.info("[VERIFY] open_app OK: window list contains %r",
                            app_name)
                return "pass"
        except Exception:
            pass
        logger.warning("[VERIFY] open_app FAIL: %r not in active window %r",
                       app_name, active)
        return "fail"

    def _verify_focus(self, target: str) -> str:
        from computer.computer_tool import ComputerTool
        active = ComputerTool.get_active_window() or ""
        if normalize_for_compare(target) in normalize_for_compare(active):
            return "pass"
        return "fail"

    def _verify_click(self, state_before: str) -> str:
        """A click's expected effect is task-specific; compare fresh state.

        If the screen state changed at all, report pass (something happened);
        if identical, fail (nothing happened); OCR/UIA errors -> uncertain.
        """
        try:
            state_after = self.see.summarize()
        except Exception as e:
            logger.warning("[VERIFY] click: could not re-read state: %s", e)
            return "uncertain"

        if state_after.strip() != state_before.strip():
            logger.info("[VERIFY] click: screen state changed")
            return "pass"
        logger.warning("[VERIFY] click: no observable change")
        return "fail"

    def _verify_typed(self, text: str) -> str:
        """Check the typed text appears in the active window via UIA/OCR."""
        needle = normalize_for_compare(text)
        if not needle:
            return "uncertain"

        from computer.computer_tool import ComputerTool
        active = ComputerTool.get_active_window()

        elems = self.see.uia_elements(active)
        joined = normalize_for_compare(" ".join(e["name"] for e in elems))
        if needle in joined:
            logger.info("[VERIFY] type_text OK: text found via UIA")
            return "pass"

        ocr = self.see.ocr_lines()
        joined_ocr = normalize_for_compare(" ".join(l["text"] for l in ocr))
        if needle in joined_ocr:
            logger.info("[VERIFY] type_text OK: text found via OCR")
            return "pass"

        logger.warning("[VERIFY] type_text: typed text not found on screen")
        return "fail"
