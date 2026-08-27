"""Safety infrastructure for the Computer Tool.

Every action goes through :func:`log_action` (audit trail) and can be halted
by the emergency stop (:class:`EmergencyStop`).

Two emergency-stop mechanisms:
1. pyautogui's built-in corner failsafe - slam the mouse to the top-left
   corner of the primary screen and any in-flight pyautogui call raises
   ``pyautogui.FailSafeException`` immediately.
2. An explicit ``estop()`` flag that multi-step actions (drag, type) check
   between steps, so long sequences abort mid-way rather than only at the
   next library call boundary.
"""

import logging
import threading
import time
from functools import wraps

logger = logging.getLogger("beast.computer")

# ---------------------------------------------------------------------------
# Emergency stop flag
# ---------------------------------------------------------------------------

class EmergencyStop(Exception):
    """Raised when an action is aborted via estop()."""


class _EStopFlag:
    """Thread-safe boolean flag checked between steps of long actions."""

    def __init__(self):
        self._event = threading.Event()

    def trigger(self):
        self._event.set()
        logger.warning("EMERGENCY STOP triggered - all pending actions aborted")

    def clear(self):
        self._event.clear()

    @property
    def is_stopped(self) -> bool:
        return self._event.is_set()

    def check(self):
        """Raise EmergencyStop if the flag is set."""
        if self._event.is_set():
            raise EmergencyStop("Action aborted by emergency stop")


# Global singleton so any code path can halt everything.
estop_flag = _EStopFlag()


def estop():
    """Trigger the emergency stop. Halts all in-progress/queued actions."""
    estop_flag.trigger()


def reset_estop():
    """Clear the emergency stop (must be called before resuming)."""
    estop_flag.clear()
    logger.info("Emergency stop cleared")


# ---------------------------------------------------------------------------
# Action logging decorator
# ---------------------------------------------------------------------------

def log_action(func):
    """Decorator: logs every ComputerTool action with args and duration.

    This is the seed of the Activity Log feature and the first place to look
    when something went to the wrong window.
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        arg_str = ", ".join(
            [repr(a) for a in args] + [f"{k}={v!r}" for k, v in kwargs.items()]
        )
        t0 = time.perf_counter()
        try:
            result = func(self, *args, **kwargs)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.info("%s(%s) -> ok (%.0f ms)", func.__name__, arg_str, elapsed_ms)
            return result
        except Exception as e:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.error(
                "%s(%s) -> FAILED after %.0f ms: %s",
                func.__name__, arg_str, elapsed_ms, e,
            )
            raise

    return wrapper


# ---------------------------------------------------------------------------
# Guarded step helper for multi-step actions
# ---------------------------------------------------------------------------

def guarded_step(estop_check, func, *args, **kwargs):
    """Run one step of a multi-step action, checking estop first."""
    estop_check.check()
    return func(*args, **kwargs)
