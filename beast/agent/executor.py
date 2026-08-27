"""ACT layer: execute exactly ONE structured action via ComputerTool or BrowserTool.

The LLM never touches automation libraries directly - only these whitelisted mappings
from validated action dicts to tested ComputerTool and BrowserTool methods.
Every call inherits the respective tool's estop checks and audit logging.
"""

import logging
from computer.browser_tool import BrowserTool, TimeoutError

logger = logging.getLogger("beast.agent")


class ActionExecutor:
    """Maps a validated action dict to a single ComputerTool, BrowserTool, or MemoryManager call."""

    def __init__(self, computer_tool, memory_manager=None):
        self.ct = computer_tool
        self._browser_tool = None  # Lazy initialized when needed
        self._memory = memory_manager

    def execute(self, action: dict, click_point: tuple[int, int] = None) -> str:
        """Run one action. Returns a short human-readable result string.

        click_point: resolved screen coordinates for 'click' actions
        (found by the loop from UIA/OCR before calling here).
        """
        kind = action["action"]

        if kind == "open_app":
            _, title = self.ct.open_app(action["target"])
            return f"opened {action['target']}" + (
                f" (window: {title})" if title else " (no window detected yet)")

        if kind == "click":
            if click_point is None:
                return "FAILED: no clickable location found for target"
            x, y = click_point
            self.ct.click(x=x, y=y)
            return f"clicked at {x},{y}"

        if kind == "type_text":
            self.ct.type_text(action["text"])
            return f"typed {len(action['text'])} chars"

        if kind == "press":
            self.ct.press(action["key"])
            return f"pressed {action['key']}"

        if kind == "hotkey":
            keys = "+".join(action["keys"])
            self.ct.hotkey(*action["keys"])
            return f"pressed hotkey {keys}"

        if kind == "scroll":
            self.ct.scroll(action["amount"])
            direction = "up" if action["amount"] > 0 else "down"
            return f"scrolled {direction} by {abs(action['amount'])}"

        if kind == "focus_window":
            ok = self.ct.focus_window(action["target"])
            return ("focused window" if ok
                    else f"FAILED: no window matching {action['target']!r}")

        # Browser actions
        if kind.startswith("browser_"):
            # Initialize browser tool on first use
            if self._browser_tool is None:
                self._browser_tool = BrowserTool(headed=True, slow_mo=0)  # Default to headed, no slowdown
                self._browser_tool.start_session()

            try:
                if kind == "browser_navigate":
                    return self._browser_tool.navigate(action["url"])
                elif kind == "browser_find_element":
                    return self._browser_tool.find_element(action["target"])
                elif kind == "browser_click":
                    return self._browser_tool.click(action["target"])
                elif kind == "browser_type_text":
                    return self._browser_tool.type_text(action["target"], action["text"])
                elif kind == "browser_get_text":
                    return self._browser_tool.get_text(action["target"])
                elif kind == "browser_screenshot":
                    full_page = action.get("full_page", False)
                    return self._browser_tool.screenshot(full_page=full_page)
                elif kind == "browser_wait_for":
                    condition = action["condition"]
                    timeout = action.get("timeout", 5000)
                    return self._browser_tool.wait_for(condition, timeout=timeout)
                else:
                    return f"UNKNOWN BROWSER ACTION: {kind}"
            except Exception as e:
                logger.error(f"Browser action {kind} failed: {e}")
                return f"FAILED: {e}"

        # Memory actions
        if kind == "remember":
            if self._memory is None:
                return "FAILED: memory system not available"
            return self._memory.remember(
                key=action["key"],
                value=action["value"],
                category=action.get("category", "personal"),
                tags=action.get("tags", ""),
                source=action.get("source"),
                confidence=action.get("confidence", 1.0),
            )

        if kind == "recall":
            if self._memory is None:
                return "FAILED: memory system not available"
            return self._memory.recall(action["key"])

        if kind == "forget":
            if self._memory is None:
                return "FAILED: memory system not available"
            return self._memory.forget(action["key"])

        if kind == "list_memories":
            if self._memory is None:
                return "FAILED: memory system not available"
            memories = self._memory.store.list_all(limit=20)
            if not memories:
                return "No memories stored yet."
            parts = [f"{m['key']}: {m['value']}" for m in memories]
            return "Memories: " + "; ".join(parts) + "."

        # task_complete / ask_user are terminal states handled by the loop.
        if kind in ("task_complete", "ask_user"):
            return action.get("message", "")

        return f"UNKNOWN ACTION: {kind}"