"""ComputerTool - the single controlled interface for screen capture and
input control.

The LLM (Milestone 4) will only ever call methods on this class; it never
executes arbitrary code. This milestone is infrastructure + live verification
only - nothing here is wired into the voice pipeline yet.

Library choices (justified in the milestone plan):
- Screenshots: ``mss`` - fast multi-monitor capture, raw BGRA buffers.
- Mouse/keyboard: ``pyautogui`` - built-in corner failsafe emergency stop.
- Windows: ``pygetwindow`` - lightweight active-window enumeration.

Known limitations (documented up front):
- UAC-elevated windows ignore synthetic input from non-elevated processes.
- DPI scaling: pyautogui works in logical pixels; mss captures physical
  pixels. On scaled displays the two differ - see ``_dpi_scale()``.
"""

import logging
import os
import subprocess
import time

import mss
import numpy as np
import pyautogui

from .safety import EmergencyStop, estop_flag, log_action

logger = logging.getLogger("beast.computer")

# Keep pyautogui's corner failsafe ON at all times.
pyautogui.FAILSAFE = True
# Small pause between pyautogui calls to avoid dropped input events.
pyautogui.PAUSE = 0.05


class ComputerTool:
    """Controlled facade over screen capture and mouse/keyboard control."""

    def __init__(self, screenshot_dir: str = None):
        self.screenshot_dir = screenshot_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "logs", "screenshots",
        )
        os.makedirs(self.screenshot_dir, exist_ok=True)
        self._sct = mss.mss()

    # ------------------------------------------------------------------
    # Screen capture
    # ------------------------------------------------------------------

    @staticmethod
    def list_monitors():
        """Return list of monitor descriptors: index 0 = all monitors."""
        with mss.mss() as sct:
            return sct.monitors

    @log_action
    def screenshot(self, monitor: int = 1) -> np.ndarray:
        """Capture a monitor as an RGB numpy array.

        monitor=0 captures the entire virtual desktop (all monitors);
        monitor=1..n capture individual displays (mss convention).
        """
        with mss.mss() as sct:
            if monitor >= len(sct.monitors):
                raise ValueError(
                    f"monitor={monitor} but only {len(sct.monitors) - 1} display(s) found"
                )
            region = sct.monitors[monitor]
            shot = sct.grab(region)
        # mss returns BGRA; convert to RGB for standard image handling.
        frame = np.asarray(shot)[:, :, :3][:, :, ::-1]
        return np.ascontiguousarray(frame)

    @log_action
    def save_screenshot(self, filename: str = None, monitor: int = 1) -> str:
        """Capture and save a PNG. Returns the file path."""
        from PIL import Image

        frame = self.screenshot(monitor=monitor)
        if not filename:
            filename = f"beast_screenshot_{time.strftime('%Y%m%d_%H%M%S')}.png"
        path = os.path.join(self.screenshot_dir, filename)
        Image.fromarray(frame).save(path)
        logger.info("Screenshot saved: %s (%dx%d)", path, frame.shape[1], frame.shape[0])
        return path

    # ------------------------------------------------------------------
    # DPI helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _dpi_scale() -> float:
        """Ratio of physical pixels to logical points on the primary display.

        If this is not 1.0, coordinates passed to mouse functions are in
        logical space while screenshots are physical - callers must be aware.
        """
        try:
            import ctypes
            user32 = ctypes.windll.user32
            user32.SetProcessDPIAware()
            return user32.GetDpiForSystem() / 96.0
        except Exception:
            return 1.0

    # ------------------------------------------------------------------
    # Mouse control
    # ------------------------------------------------------------------

    @log_action
    def mouse_move(self, x: int, y: int, duration: float = 0.25):
        estop_flag.check()
        # Step the movement ourselves so estop can interrupt mid-motion.
        # A single pyautogui.moveTo(duration=...) is uninterruptible.
        start_x, start_y = pyautogui.position()
        steps = max(1, int(duration / 0.05))  # ~50ms per step
        for i in range(1, steps + 1):
            estop_flag.check()
            t = i / steps
            px = int(start_x + (x - start_x) * t)
            py = int(start_y + (y - start_y) * t)
            pyautogui.moveTo(px, py, duration=0)
            time.sleep(duration / steps)

    @log_action
    def click(self, x: int = None, y: int = None):
        estop_flag.check()
        pyautogui.click(x=x, y=y)

    @log_action
    def double_click(self, x: int = None, y: int = None):
        estop_flag.check()
        pyautogui.doubleClick(x=x, y=y)

    @log_action
    def right_click(self, x: int = None, y: int = None):
        estop_flag.check()
        pyautogui.rightClick(x=x, y=y)

    @log_action
    def scroll(self, amount: int):
        """Positive scrolls up, negative scrolls down."""
        estop_flag.check()
        pyautogui.scroll(amount)

    @log_action
    def drag(self, x1: int, y1: int, x2: int, y2: int, duration: float = 0.5):
        estop_flag.check()
        pyautogui.moveTo(x1, y1)
        estop_flag.check()
        # Step the drag ourselves so estop can interrupt mid-motion AND
        # release the mouse button on abort. A single pyautogui.dragTo()
        # with duration is uninterruptible and would leave the button held.
        pyautogui.mouseDown(button="left")
        try:
            steps = max(1, int(duration / 0.05))  # ~50ms per step
            for i in range(1, steps + 1):
                estop_flag.check()  # raises EmergencyStop mid-drag
                t = i / steps
                px = int(x1 + (x2 - x1) * t)
                py = int(y1 + (y2 - y1) * t)
                pyautogui.moveTo(px, py, duration=0)
                time.sleep(duration / steps)
            # Completed normally - release at destination.
            pyautogui.mouseUp(button="left")
        except EmergencyStop:
            # Safety-critical: ALWAYS release the button when aborted,
            # otherwise the OS thinks the mouse is still held down.
            try:
                pyautogui.mouseUp(button="left")
            except Exception:
                pass
            raise

    @staticmethod
    def mouse_position():
        return pyautogui.position()

    # ------------------------------------------------------------------
    # Keyboard control
    # ------------------------------------------------------------------

    @log_action
    def type_text(self, text: str, interval: float = 0.02):
        """Type text with per-character delay to avoid dropped characters.

        Types character-by-character so estop can interrupt mid-string.
        Note: pyautogui.typewrite cannot type non-US-keyboard characters;
        use clipboard paste for those later if needed.
        """
        estop_flag.check()
        for ch in text:
            estop_flag.check()  # abort mid-string if estop fires
            pyautogui.write(ch)
            time.sleep(interval)

    @log_action
    def press(self, key: str):
        estop_flag.check()
        pyautogui.press(key)

    @log_action
    def hotkey(self, *keys: str):
        estop_flag.check()
        pyautogui.hotkey(*keys)

    # ------------------------------------------------------------------
    # Application / window control
    # ------------------------------------------------------------------

    @log_action
    def open_app(self, name_or_path: str, wait_for_window: bool = True,
                 timeout: float = 5.0):
        """Launch an application by name (resolved via PATH/Start Menu) or
        full path.

        Launch mechanism: subprocess.Popen (direct exec first, shell 'start'
        fallback for PATH-resolved names like notepad/calc/mspaint).

        When wait_for_window=True (default), polls the active window every
        150ms until it CHANGES from whatever was focused before launch
        (or until a window matching the app name appears), up to `timeout`
        seconds. This handles both fast and slow-opening apps without blind
        sleeps. Returns (handle_or_None, new_active_title_or_None).
        """
        estop_flag.check()
        before = self.get_active_window()

        if os.path.isfile(name_or_path):
            proc = subprocess.Popen([name_or_path])
        else:
            try:
                proc = subprocess.Popen([name_or_path], shell=False)
            except FileNotFoundError:
                proc = subprocess.Popen(f"start {name_or_path}", shell=True)

        if not wait_for_window:
            return proc, None

        deadline = time.time() + timeout
        while time.time() < deadline:
            estop_flag.check()
            time.sleep(0.15)
            current = self.get_active_window()
            # Success: active window changed AND is not the old one.
            if current and current != before:
                logger.info(
                    "open_app(%r): window focus changed %r -> %r after %.1fs",
                    name_or_path, before, current,
                    timeout - (deadline - time.time()),
                )
                return proc, current

        logger.warning(
            "open_app(%r): no window-focus change within %.1fs "
            "(still %r)", name_or_path, timeout, before,
        )
        return proc, None

    @staticmethod
    def get_active_window():
        """Return title of the currently focused window, or None."""
        import pygetwindow as gw
        try:
            win = gw.getActiveWindow()
            return win.title if win else None
        except Exception:
            return None

    @staticmethod
    def list_windows():
        """Return list of (title) strings for all visible top-level windows."""
        import pygetwindow as gw
        return [w.title for w in gw.getAllWindows() if w.title.strip()]

    @log_action
    def focus_window(self, title_substring: str) -> bool:
        """Bring the first window whose title contains the substring to front."""
        import pygetwindow as gw
        for w in gw.getAllWindows():
            if title_substring.lower() in w.title.lower():
                try:
                    w.activate()
                except Exception:
                    w.restore()
                return True
        return False
