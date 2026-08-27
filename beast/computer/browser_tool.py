"""BrowserTool - the single controlled interface for browser automation via Playwright.

This tool provides DOM-based browser control as an alternative to ComputerTool's
pixel/keyboard approach. It follows the same safety patterns: estop checks,
action logging, and controlled interface design.

Milestone 5: Browser Automation
- Headed (visible) browser for verifiable testing
- Chromium as default (standard, well-tested)
- Accessible-role-based locators preferred over CSS selectors
- Session lifecycle management
- Integrated with existing autonomy levels and confirmation gate
"""

import logging
import os
import time
from typing import Optional, List, Dict, Any
from pathlib import Path

logger = logging.getLogger("beast.browser")

try:
    from playwright.sync_api import sync_playwright, Browser, Page, Locator, TimeoutError as PlaywrightTimeoutError
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    logger.warning("Playwright not available - BrowserTool will not function")

from .safety import EmergencyStop, estop_flag, log_action


class BrowserTool:
    """Controlled facade over Playwright browser automation.

    Provides DOM-based actions for web tasks:
    - navigate(url) - go to a URL
    - find_element(description_or_selector) - locate element by text/role/label
    - click(element), type_text(element, text), get_text(element)
    - screenshot() - capture page state
    - wait_for(condition) - handle async loading
    """

    def __init__(self, headed: bool = True, slow_mo: int = 0):
        """
        Initialize BrowserTool.

        Args:
            headed: If True, show browser window (default True for verifiable testing)
            slow_mo: Slow down actions by specified milliseconds (useful for debugging)
        """
        if not PLAYWRIGHT_AVAILABLE:
            raise RuntimeError("Playwright is not installed. Install with: pip install playwright")

        self.headed = headed
        self.slow_mo = slow_mo
        self._playwright = None
        self._browser: Optional[Browser] = None
        self._page: Optional[Page] = None
        self._context = None

        # Session tracking
        self._session_start_time = None
        self._actions_performed = 0

        logger.info(f"BrowserTool initialized (headed={headed}, slow_mo={slow_mo})")

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    @log_action
    def start_session(self) -> None:
        """Start a browser session. Must be called before other actions."""
        estop_flag.check()

        # Check if we are in an asyncio event loop
        try:
            import asyncio
            loop = asyncio.get_running_loop()
            error_msg = ("It looks like you are using Playwright Sync API inside the asyncio loop.\n"
                        "Please use the Async API instead.")
            logger.error(f"Asyncio loop detected: {loop} (id={id(loop)}, _thread={getattr(loop, '_thread', None)})")
            raise RuntimeError(error_msg)
        except RuntimeError as e:
            # If not in an asyncio loop, get_running_loop raises RuntimeError - this is expected
            # But if it's our error about asyncio loop, re-raise it
            if "Playwright Sync API inside the asyncio loop" in str(e):
                raise
            pass

        if self._browser is not None:
            logger.warning("Browser session already started")
            return

        self._session_start_time = time.time()
        logger.info("Starting Playwright browser session...")

        self._playwright = sync_playwright().start()

        # Launch browser
        launch_options = {
            "headless": not self.headed,
            "slow_mo": self.slow_mo,
        }

        try:
            self._browser = self._playwright.chromium.launch(**launch_options)
            self._context = self._browser.new_context()
            self._page = self._context.new_page()

            logger.info(f"Browser session started (headed={self.headed})")

        except Exception as e:
            self._cleanup()
            raise RuntimeError(f"Failed to start browser: {e}") from e

    @log_action
    def close_session(self) -> None:
        """Close the browser session and clean up resources."""
        estop_flag.check()

        if self._browser is None:
            logger.warning("No browser session to close")
            return

        logger.info("Closing browser session...")
        self._cleanup()
        logger.info("Browser session closed")

    def _cleanup(self) -> None:
        """Internal cleanup of browser resources."""
        try:
            if self._page:
                self._page.close()
            if self._context:
                self._context.close()
            if self._browser:
                self._browser.close()
            if self._playwright:
                self._playwright.stop()
        except Exception as e:
            logger.warning(f"Error during browser cleanup: {e}")
        finally:
            self._page = None
            self._context = None
            self._browser = None
            self._playwright = None

    def __del__(self):
        """Ensure cleanup on deletion."""
        try:
            self.close_session()
        except:
            pass

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def is_session_active(self) -> bool:
        """Check if browser session is active."""
        return self._browser is not None and self._page is not None

    # ------------------------------------------------------------------
    # Core browser actions
    # ------------------------------------------------------------------

    @log_action
    def navigate(self, url: str) -> str:
        """Navigate to a URL.

        Args:
            url: The URL to navigate to

        Returns:
            Status message
        """
        estop_flag.check()
        self._ensure_session()

        logger.info(f"Navigating to: {url}")
        self._page.goto(url, wait_until="domcontentloaded")
        self._actions_performed += 1

        return f"navigated to {url}"

    @log_action
    def find_element(self, description: str) -> str:
        """Find an element using accessible role-based selectors (preferred) or fallback strategies.

        Args:
            description: Description of element to find (text content, role, label, etc.)

        Returns:
            Element handle or identifier for use with other actions
        """
        estop_flag.check()
        self._ensure_session()

        logger.info(f"Finding element: {description}")

        # Special handling for common search box descriptions
        if description.lower() in ["search", "google search", "search box"]:
            # Try to find the Google search box specifically
            try:
                # Look for textarea/input with aria-label="Search" or title="Search"
                search_box = self._page.locator('textarea[title="Search"], input[title="Search"], [aria-label="Search"]')
                if search_box.count() > 0:
                    logger.info(f"Found Google search box by specific attributes")
                    return f"css=textarea[title='Search'], input[title='Search'], [aria-label='Search'] >> nth=0"
            except Exception:
                pass
            # Fallback to general search box finding
            try:
                search_box = self._page.locator('input[type="search"], textarea[aria-label*="search" i], input[aria-label*="search" i]')
                if search_box.count() > 0:
                    logger.info(f"Found search box by type/aria-label")
                    return f"css=input[type='search'], textarea[aria-label*='search' i], input[aria-label*='search' i] >> nth=0"
            except Exception:
                pass

        # Strategy 1: Try to find by text content (exact match)
        try:
            element = self._page.get_by_text(description, exact=True)
            if element.count() > 0:
                logger.info(f"Found element by exact text: {description}")
                return f"text={description}"
        except Exception:
            pass

        # Strategy 2: Try to find by label (for form elements) - be more specific
        try:
            element = self._page.get_by_label(description)
            if element.count() > 0:
                # If multiple elements match the label, try to be more specific
                if element.count() == 1:
                    logger.info(f"Found element by label: {description}")
                    return f"label={description}"
                else:
                    # Try to narrow down to common input types
                    input_element = element.locator("input, textarea, select").first
                    if input_element.count() > 0:
                        logger.info(f"Found input element by label: {description}")
                        return f"label={description} >> input, textarea, select >> nth=0"
                    # If that doesn't work, fall back to first match but only if it's an input-like element
                    first_element = element.first
                    if first_element.locator("input, textarea, select").count() > 0:
                        logger.info(f"Found input element by label (first match): {description}")
                        return f"label={description} >> nth=0"
                    else:
                        logger.warning(f"First match for label '{description}' is not an input-like element, trying other strategies")
        except Exception:
            pass

        # Strategy 3: Try to find by placeholder
        try:
            element = self._page.get_by_placeholder(description)
            if element.count() > 0:
                # If multiple elements match the placeholder, try to be more specific
                if element.count() == 1:
                    logger.info(f"Found element by placeholder: {description}")
                    return f"placeholder={description}"
                else:
                    # Try to narrow down to common input types
                    input_element = element.locator("input, textarea").first
                    if input_element.count() > 0:
                        logger.info(f"Found input element by placeholder: {description}")
                        return f"placeholder={description} >> input, textarea >> nth=0"
                    # If that doesn't work, fall back to first match but only if it's an input-like element
                    first_element = element.first
                    if first_element.locator("input, textarea").count() > 0:
                        logger.info(f"Found input element by placeholder (first match): {description}")
                        return f"placeholder={description} >> nth=0"
                    else:
                        logger.warning(f"First match for placeholder '{description}' is not an input-like element, trying other strategies")
        except Exception:
            pass

        # Strategy 4: Try to find by text content (exact match) but for specific roles
        # This helps with cases where we want a button with specific text
        try:
            # Look for buttons, links, etc. with exact text match
            for role in ["button", "link"]:
                try:
                    element = self._page.get_by_role(role, name=description, exact=True)
                    if element.count() > 0:
                        logger.info(f"Found {role} by exact text: {description}")
                        return f"role={role}:name={description}"
                except Exception:
                    continue
        except Exception:
            pass

        # Strategy 5: Try to find by text content (partial match) - but be more specific
        try:
            # Instead of using get_by_text() which can match many elements,
            # try to be more specific by combining with role or other attributes
            text_elements = self._page.get_by_text(description, exact=False)
            if text_elements.count() > 0:
                # If there are too many matches, try to narrow it down
                if text_elements.count() <= 3:  # Reasonable number of matches
                    logger.info(f"Found element by partial text: {description} ({text_elements.count()} matches)")
                    return f"text partial={description}"
                else:
                    # Try to narrow down to interactive elements
                    interactive = text_elements.locator("button, link, input, textarea, [role='button'], [role='link']")
                    if interactive.count() > 0 and interactive.count() <= 3:
                        logger.info(f"Found interactive element by partial text: {description} ({interactive.count()} matches)")
                        return f"text partial={description} >> button, link, input, textarea, [role='button'], [role='link'] >> nth=0"
                    logger.warning(f"Too many matches for partial text '{description}' ({text_elements.count()} matches), trying other strategies")
        except Exception:
            pass

        # Strategy 6: Try to find by role (button, link, etc.) - infer role from description
        try:
            # Try common roles
            for role in ["button", "link", "input", "textbox", "combobox"]:
                try:
                    element = self._page.get_by_role(role, name=description)
                    if element.count() > 0:
                        logger.info(f"Found element by role {role} with name: {description}")
                        return f"role={role}:name={description}"
                except Exception:
                    continue
        except Exception:
            pass

        # Strategy 7: Fallback to CSS selector (less preferred but functional)
        try:
            element = self._page.locator(description)
            if element.count() > 0:
                logger.info(f"Found element by CSS selector: {description}")
                return f"css={description}"
        except Exception:
            pass

        raise ValueError(f"Could not find element matching description: {description}")

    @log_action
    def click(self, element_identifier: str) -> str:
        """Click an element found by find_element.

        Args:
            element_identifier: String returned by find_element() identifying the element

        Returns:
            Status message
        """
        estop_flag.check()
        self._ensure_session()

        logger.info(f"Clicking element: {element_identifier}")

        # Parse the element identifier and click appropriately
        locator = self._build_locator_from_identifier(element_identifier)
        locator.click()

        self._actions_performed += 1
        return f"clicked {element_identifier}"

    @log_action
    def type_text(self, element_identifier: str, text: str) -> str:
        """Type text into an element found by find_element.

        Args:
            element_identifier: String returned by find_element() identifying the element
            text: Text to type into the element

        Returns:
            Status message
        """
        estop_flag.check()
        self._ensure_session()

        logger.info(f"Typing into {element_identifier}: {text[:50]}{'...' if len(text) > 50 else ''}")

        # Parse the element identifier and type appropriately
        locator = self._build_locator_from_identifier(element_identifier)

        # Clear the element first
        locator.clear()

        # Type character-by-character so estop can interrupt mid-string
        for ch in text:
            estop_flag.check()  # Check for emergency stop between each character
            locator.type(ch)
            # Small delay to make typing more human-like and interruptible
            time.sleep(0.02)

        self._actions_performed += 1
        return f"typed {len(text)} chars into {element_identifier}"

    @log_action
    def get_text(self, element_identifier: str) -> str:
        """Get text content from an element found by find_element.

        Args:
            element_identifier: String returned by find_element() identifying the element

        Returns:
            Text content of the element
        """
        estop_flag.check()
        self._ensure_session()

        logger.info(f"Getting text from: {element_identifier}")

        # Parse the element identifier and get text
        locator = self._build_locator_from_identifier(element_identifier)
        text = locator.text_content() or ""

        self._actions_performed += 1
        return text.strip()

    @log_action
    def screenshot(self, full_page: bool = False) -> str:
        """Capture a screenshot of the current page.

        Args:
            full_page: If True, capture full scrollable page; otherwise viewport only

        Returns:
            Path to saved screenshot file
        """
        estop_flag.check()
        self._ensure_session()

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"browser_screenshot_{timestamp}.png"
        screenshot_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "logs", "browser_screenshots"
        )
        os.makedirs(screenshot_dir, exist_ok=True)
        path = os.path.join(screenshot_dir, filename)

        logger.info(f"Taking screenshot (full_page={full_page}): {path}")
        self._page.screenshot(path=path, full_page=full_page)
        self._actions_performed += 1

        return path

    @log_action
    def wait_for(self, condition: str, timeout: int = 5000) -> str:
        """Wait for a condition to be true on the page.

        Args:
            condition: Condition to wait for (e.g., "domcontentloaded", "load", "networkidle")
            timeout: Timeout in milliseconds

        Returns:
            Status message
        """
        estop_flag.check()
        self._ensure_session()

        logger.info(f"Waiting for condition: {condition} (timeout={timeout}ms)")

        try:
            self._page.wait_for_load_state(condition, timeout=timeout)
            self._actions_performed += 1
            return f"waited for {condition}"
        except PlaywrightTimeoutError:
            raise TimeoutError(f"Timeout waiting for condition: {condition}")

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _ensure_session(self) -> None:
        """Ensure browser session is active, raise exception if not."""
        if not self.is_session_active():
            raise RuntimeError(
                "Browser session not active. Call start_session() before performing actions."
            )
        estop_flag.check()

    def _build_locator_from_identifier(self, identifier: str) -> Any:
        """Build a Playwright locator from the element identifier string.

        This reverses the process in find_element() to create a locator for actions.
        """
        # Handle compound identifiers with chaining (e.g., "label=Text >> role=button")
        if " >> " in identifier:
            parts = identifier.split(" >> ")
            # Start with the base locator
            locator = self._build_locator_from_identifier(parts[0])
            # Apply each chained selector
            for part in parts[1:]:
                if part.startswith("nth="):
                    index = int(part[4:])  # Remove "nth=" prefix
                    locator = locator.nth(index)
                elif part == "first":
                    locator = locator.first
                elif part == "last":
                    locator = locator.last
                else:
                    # Try to parse as a locator specifier (e.g., "role=button", "input, textarea")
                    try:
                        # This is a bit hacky but should work for common cases
                        chained_locator = self._page.locator(part)
                        locator = locator.locator(chained_locator)
                    except Exception:
                        # If that fails, try as a simple CSS selector
                        locator = locator.locator(part)
            return locator

        # Handle simple identifiers
        if identifier.startswith("text="):
            text = identifier[5:]  # Remove "text=" prefix
            return self._page.get_by_text(text, exact=True)
        elif identifier.startswith("text partial="):
            text = identifier[15:]  # Remove "text partial=" prefix
            return self._page.get_by_text(text).first
        elif identifier.startswith("label="):
            text = identifier[6:]  # Remove "label=" prefix
            return self._page.get_by_label(text).first
        elif identifier.startswith("placeholder="):
            text = identifier[12:]  # Remove "placeholder=" prefix
            return self._page.get_by_placeholder(text).first
        elif identifier.startswith("role="):
            # Format: role=ROLE:name=NAME
            parts = identifier.split(":name=")
            if len(parts) == 2:
                role_part = parts[0]  # "role=ROLE"
                name_part = parts[1]  # "NAME"
                if role_part.startswith("role="):
                    role = role_part[5:]  # Remove "role=" prefix
                    return self._page.get_by_role(role, name=name_part)
            # Fallback: try to parse as just role
            role = identifier[5:]  # Remove "role=" prefix
            return self._page.get_by_role(role)
        elif identifier.startswith("css="):
            selector = identifier[4:]  # Remove "css=" prefix
            return self._page.locator(selector)
        else:
            # If we don't recognize the format, try as CSS selector as fallback
            logger.warning(f"Unrecognized element identifier format: {identifier}, trying as CSS selector")
            return self._page.locator(identifier)


class TimeoutError(Exception):
    """Raised when a wait_for operation times out."""
    pass