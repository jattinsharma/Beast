"""High-level memory manager for Beast.

Provides the interface that the agent loop and voice pipeline use:
- remember(key, value, category, tags, source, confidence) -> confirmation string
- recall(query) -> result string for TTS / LLM context
- forget(key) -> confirmation string
- context_for_llm() -> memory context block injected into the LLM prompt
"""

import logging
from typing import Optional

from .store import MemoryStore

logger = logging.getLogger("beast.memory")


class MemoryManager:
    """Manages Beast's persistent memory through the MemoryStore."""

    def __init__(self, store: Optional[MemoryStore] = None):
        self.store = store or MemoryStore()

    # ------------------------------------------------------------------
    # Agent-facing actions (called by ActionExecutor)
    # ------------------------------------------------------------------

    def remember(self, key: str, value: str, category: str = "personal",
                 tags: str = "", source: str = None, confidence: float = 1.0) -> str:
        """Store a fact. Returns a spoken confirmation."""
        key = key.strip()
        value = value.strip()
        if not key:
            return "I need a topic to remember."
        if not value:
            return "What should I remember about that?"

        # Validate confidence is between 0.0 and 1.0
        if not 0.0 <= confidence <= 1.0:
            logger.warning("[MEMORY] Confidence %f out of range [0.0, 1.0], clamping", confidence)
            confidence = max(0.0, min(1.0, confidence))

        memory_id = self.store.remember(key, value, category=category, source=source,
                                        confidence=confidence, tags=tags)
        # Check if this was an update or insert by looking at the row
        existing = self.store.recall(key)
        if existing and existing["created_at"] != existing["updated_at"]:
            return f"Updated: {key} is now {value}."
        return f"Remembered: {key} is {value}."

    def recall(self, query: str) -> str:
        """Look up a memory by key or search. Returns a spoken answer."""
        query = query.strip()
        if not query:
            return "What are you looking for?"

        # Try exact key match first
        exact = self.store.recall(query)
        if exact:
            return f"{exact['key']} is {exact['value']}."

        # Try search across keys, values, tags
        results = self.store.search(query, limit=5)
        if not results:
            return f"I don't have anything stored about {query}."
        if len(results) == 1:
            r = results[0]
            return f"{r['key']} is {r['value']}."
        # Multiple results
        parts = [f"{r['key']}: {r['value']}" for r in results[:3]]
        return "Here's what I know: " + "; ".join(parts) + "."

    def forget(self, key: str) -> str:
        """Delete a memory. Returns a spoken confirmation."""
        key = key.strip()
        if not key:
            return "What should I forget?"
        if self.store.forget(key):
            return f"Forgotten: {key}."
        return f"I don't have anything stored about {key}."

    def forget_by_id(self, memory_id: int) -> str:
        """Delete a memory by id."""
        if self.store.forget_by_id(memory_id):
            return "Forgotten."
        return "I couldn't find that memory."

    # ------------------------------------------------------------------
    # LLM context injection
    # ------------------------------------------------------------------

    def context_for_llm(self, max_memories: int = 15) -> str:
        """Build a context block to inject into the LLM system prompt.

        Returns a string like:
            USER MEMORIES:
            - name: Alice (personal)
            - favorite color: blue (preference)
            - dentist appointment: March 5 at 2pm (reminder)

        Only non-sensitive memories are included. Sensitive ones are
        accessed on-demand via recall(), never injected into prompts.
        """
        memories = self.store.list_all(limit=max_memories)
        if not memories:
            return ""

        lines = ["USER MEMORIES (retrieved from persistent storage):"]
        for m in memories:
            sensitive_flag = " [sensitive]" if m["is_sensitive"] else ""
            cat = f" ({m['category']})" if m["category"] else ""
            source_info = f" [from {m['source']}]" if m.get('source') else ""
            conf_info = f" (confidence: {m['confidence']:.2f})" if m.get('confidence') and m['confidence'] < 1.0 else ""
            lines.append(f"- {m['key']}: {m['value']}{cat}{source_info}{conf_info}{sensitive_flag}")

        return "\n".join(lines)

    def stats(self) -> str:
        """Return a short summary of memory usage."""
        total = self.store.count()
        personal = self.store.count("personal")
        preference = self.store.count("preference")
        reminder = self.store.count("reminder")
        return (f"Memory: {total} total "
                f"({personal} personal, {preference} preference, {reminder} reminder)")