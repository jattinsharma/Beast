"""SEE layer: screen understanding via the escalation ladder.

Priority order (cheapest, most reliable source of truth first):
1. Windows UI Automation tree (pywinauto, backend="uia") - structured element
   info (name, control type, clickable rect) for the active window.
2. Windows built-in OCR (winocr) - on-screen text with positions when UIA
   doesn't expose what's needed (canvas/webview content).
3. (Reserved seam for a VLM in a later milestone - deliberately not built yet.)

All functions are read-only and never act; they only describe state.
"""

import logging
import re

logger = logging.getLogger("beast.agent")

# Max elements reported from a UIA walk - keeps LLM context bounded.
MAX_UIA_ELEMENTS = 60
# Max OCR lines reported.
MAX_OCR_LINES = 40


class ScreenState:
    """Read-only screen understanding. UIA first, OCR fallback."""

    def __init__(self):
        self._desktop = None

    # ------------------------------------------------------------------
    # Layer 1: Windows UI Automation tree
    # ------------------------------------------------------------------

    def _get_desktop(self):
        if self._desktop is None:
            from pywinauto import Desktop
            self._desktop = Desktop(backend="uia")
        return self._desktop

    def uia_elements(self, window_title_substring: str = None) -> list[dict]:
        """Walk the UIA tree of the active (or named) window.

        Returns a list of element dicts:
          {"name", "type", "rect": [l, t, r, b], "enabled"}
        Rects are in SCREEN coordinates (physical pixels), ready to click.
        Returns [] if no window matches or UIA fails.
        """
        try:
            desktop = self._get_desktop()
            windows = desktop.windows()
            target = None
            for w in windows:
                title = w.window_text() or ""
                if not title.strip():
                    continue
                if window_title_substring is None or \
                        window_title_substring.lower() in title.lower():
                    target = w
                    break
            if target is None:
                logger.info("[SEE] No matching window for %r",
                            window_title_substring)
                return []

            elements = []
            self._walk(target, elements, depth=0)
            logger.info("[SEE] UIA found %d elements in %r",
                        len(elements), target.window_text())
            return elements[:MAX_UIA_ELEMENTS]
        except Exception as e:
            logger.warning("[SEE] UIA walk failed: %s", e)
            return []

    def _walk(self, wrapper, out: list, depth: int):
        """Depth-first walk collecting interactive-looking elements."""
        if depth > 8 or len(out) >= MAX_UIA_ELEMENTS * 2:
            return
        try:
            info = wrapper.element_info
            name = (info.name or "").strip()
            ctype = info.control_type or ""
            rect = wrapper.rectangle()
            enabled = bool(info.enabled)

            # Keep only elements that are plausibly actionable or informative.
            interesting = (
                ctype in ("Button", "MenuItem", "Edit", "ComboBox", "CheckBox",
                          "RadioButton", "TabItem", "Hyperlink", "ListItem",
                          "Document", "Pane")
                and (name or ctype == "Edit")
            )
            if interesting and rect.width() > 0 and rect.height() > 0:
                out.append({
                    "name": name[:80],
                    "type": ctype,
                    "rect": [rect.left, rect.top, rect.right, rect.bottom],
                    "enabled": enabled,
                })

            for child in wrapper.children():
                self._walk(child, out, depth + 1)
                if len(out) >= MAX_UIA_ELEMENTS * 2:
                    break
        except Exception:
            # Some elements raise on access (protected/system); skip them.
            pass

    # ------------------------------------------------------------------
    # Layer 2: Windows built-in OCR
    # ------------------------------------------------------------------

    def ocr_lines(self, frame=None) -> list[dict]:
        """OCR the given RGB numpy frame (or capture monitor 1).

        Returns [{"text", "rect": [l, t, r, b]}] sorted top-to-bottom.
        """
        # Avoid asyncio loop conflicts with winocr
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                logger.warning("[SEE] Asyncio loop detected, skipping OCR to avoid conflict")
                return []
        except:
            pass

        try:
            import winocr
        except ImportError:
            logger.warning("[SEE] winocr not available")
            return []

        if frame is None:
            from computer.computer_tool import ComputerTool
            frame = ComputerTool().screenshot(monitor=1)

        try:
            import PIL.Image
            img = PIL.Image.fromarray(frame)
            # Use synchronous OCR
            result = winocr.recognize_pil_sync(img, lang="en-US")
        except Exception as e:
            logger.warning(f"[SEE] OCR failed: {e}")
            return []

        # winocr exposes per-WORD bounding rects; a line's rect is the union
        # of its words' rects.
        lines = []
        for line in result['lines']:
            text = line['text'].strip()
            if not text:
                continue
            l = t = None
            rgt = bot = 0
            for w in line.get('words', []):
                wr = w['bounding_rect']
                x2, y2 = wr['x'] + wr['width'], wr['y'] + wr['height']
                l = wr['x'] if l is None else min(l, wr['x'])
                t = wr['y'] if t is None else min(t, wr['y'])
                rgt, bot = max(rgt, x2), max(bot, y2)
            rect = [int(l or 0), int(t or 0), int(rgt), int(bot)]
            lines.append({"text": text[:120], "rect": rect})
        lines.sort(key=lambda l: l["rect"][1])
        logger.info("[SEE] OCR found %d text lines", len(lines))
        return lines[:MAX_OCR_LINES]

    # ------------------------------------------------------------------
    # Combined state summary for the THINK step
    # ------------------------------------------------------------------

    def summarize(self, goal_hint: str = None) -> str:
        """Build a compact textual state description for the LLM.

        Escalation ladder in practice: active window title always included;
        UIA elements when available; OCR lines as fallback/supplement.
        """
        from computer.computer_tool import ComputerTool
        active = ComputerTool.get_active_window() or "(none)"

        parts = [f"Active window: {active}"]

        elems = self.uia_elements(active)
        if elems:
            parts.append("UIA elements (name | type | center x,y):")
            for el in elems:
                cx = (el["rect"][0] + el["rect"][2]) // 2
                cy = (el["rect"][1] + el["rect"][3]) // 2
                state = "" if el["enabled"] else " [disabled]"
                parts.append(f"- {el['name']} | {el['type']} | {cx},{cy}{state}")
        else:
            # UIA gave nothing useful -> escalate to OCR.
            ocr = self.ocr_lines()
            if ocr:
                parts.append("Screen text (OCR, text | center x,y):")
                for ln in ocr:
                    cx = (ln["rect"][0] + ln["rect"][2]) // 2
                    cy = (ln["rect"][1] + ln["rect"][3]) // 2
                    parts.append(f"- {ln['text']} | {cx},{cy}")
            else:
                parts.append("(no UIA elements or OCR text detected)")

        return "\n".join(parts)


def find_element_center(elements: list[dict], query: str) -> tuple[int, int] | None:
    """Find an element whose name contains the query (case-insensitive).

    Returns (center_x, center_y) or None. Prefers exact match, then prefix,
    then substring - so 'File' matches the File menu before 'Profile'.
    """
    q = query.lower().strip()
    if not q:
        return None

    candidates = [e for e in elements if e.get("enabled", True)]
    exact = [e for e in candidates if e["name"].lower() == q]
    starts = [e for e in candidates if e["name"].lower().startswith(q)]
    contains = [e for e in candidates if q in e["name"].lower()]

    for pool in (exact, starts, contains):
        if pool:
            best = min(pool, key=lambda e: len(e["name"]))
            r = best["rect"]
            return ((r[0] + r[2]) // 2, (r[1] + r[3]) // 2)
    return None


def normalize_for_compare(text: str) -> str:
    """Lowercase, strip punctuation/whitespace - for VERIFY comparisons."""
    return re.sub(r"[^a-z0-9]+", "", text.lower())