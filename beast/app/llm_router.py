"""
Model routing system for Beast voice assistant.
Supports routing queries to different models based on complexity.
"""

import glob
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class BaseLLM(ABC):
    """Abstract base class for LLM implementations."""

    @abstractmethod
    def _ensure_loaded(self):
        """Load the model if not already loaded."""
        pass

    @abstractmethod
    def respond(self, command: str, memory_context: str = "",
                history: list[dict] | None = None) -> str:
        """Generate a response for the given command."""
        pass


class LlamaLLM(BaseLLM):
    """Llama-based LLM implementation."""

    def __init__(self, model_spec: str = "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"):
        self._llm = None
        self.model_spec = model_spec
        # These would be configured based on the model spec
        self.n_ctx = 4096  # Increased context window
        self.verbose = False

    def _ensure_loaded(self):
        if self._llm is None:
            # Resolve the model_spec to a path in huggingface cache
            if self.model_spec == "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf":
                base_path = r"C:\Users\HP\.cache\huggingface\hub\models--bartowski--Meta-Llama-3.1-8B-Instruct-GGUF"
                matches = glob.glob(
                    base_path + r"\snapshots\*\*" + self.model_spec
                )
                if not matches:
                    # Fallback to direct lookup
                    matches = glob.glob(
                        base_path + r"\snapshots\*" + self.model_spec
                    )
                if not matches:
                    raise FileNotFoundError(f"Llama GGUF not found: {self.model_spec}")
                path = matches[0]
            else:
                # For other models, we'd have different resolution logic
                # This is a placeholder for future implementation
                raise NotImplementedError(
                    f"Model resolution not implemented for {self.model_spec}"
                )

            logger.info(f"Loading LLM from {path}...")
            from llama_cpp import Llama

            self._llm = Llama(
                model_path=path, n_ctx=self.n_ctx, verbose=self.verbose,
                n_gpu_layers=0  # Set to appropriate value if GPU available
            )
            logger.info("LLM loaded")

    def respond(self, command: str, memory_context: str = "",
                 history: list[dict] | None = None) -> str:
        """Get a spoken-style response for the user's command.

        Args:
            command: The user's current transcribed command.
            memory_context: Optional memory context block from MemoryManager.
            history: Optional conversation history (list of {role, content} dicts).
        """
        self._ensure_loaded()
        now = datetime.now().strftime("%A, %B %d %Y at %I:%M %p")
        system_prompt = self._get_system_prompt().format(now=now)
        if memory_context:
            system_prompt += "\n\n" + memory_context
        messages = [{"role": "system", "content": system_prompt}]
        # Inject conversation history for multi-turn context
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": command})
        out = self._llm.create_chat_completion(messages=messages, max_tokens=250, temperature=0.7)
        text = out["choices"][0]["message"]["content"].strip()

        # Additional cleanup: remove leading/trailing whitespace and newlines
        text = text.strip()

        return text

    def _get_system_prompt(self) -> str:
        """Get the system prompt for this LLM."""
        return (
            "You are Beast, a concise voice assistant on the user's Windows PC. "
            "Answer briefly and naturally, as if speaking aloud. No markdown, no lists "
            "unless asked. Keep responses under 2 sentences when possible. "
            "If you don't know personal information about the user, say 'I don't "
            "know that yet. Tell me and I'll remember it if you want.' Never invent "
            "or hallucinate personal facts. The current date and time is {now}."
        )


class PhiLLM(BaseLLM):
    """Phi-based LLM implementation for future use."""

    def __init__(self, model_spec: str = "Phi-3-mini-4k-instruct-q4.gguf"):
        self._llm = None
        self.model_spec = model_spec
        self.n_ctx = 4096
        self.verbose = False

    def _ensure_loaded(self):
        if self._llm is None:
            # Placeholder for Phi model loading logic
            logger.info(f"Loading Phi LLM from {self.model_spec}...")
            # Implementation would go here when Phi model is available
            raise NotImplementedError(
                f"Phi model loading not yet implemented. "
                f"Please download {self.model_spec} and implement loading logic."
            )

    def respond(self, command: str, memory_context: str = "",
                 history: list[dict] | None = None) -> str:
        """Generate a response for the given command."""
        self._ensure_loaded()
        # Implementation would go here
        return "Phi model not yet implemented"

    def _get_system_prompt(self) -> str:
        """Get the system prompt for this LLM."""
        return (
            "You are Beast, a concise voice assistant on the user's Windows PC. "
            "Answer briefly and naturally, as if speaking aloud. "
            "The current date and time is {now}."
        )


class QwenLLM(BaseLLM):
    """Qwen-based LLM implementation (backward compatibility)."""

    def __init__(self, model_spec: str = "Qwen_Qwen3-1.7B-Q4_K_M.gguf"):
        self._llm = None
        self.model_spec = model_spec
        # These would be configured based on the model spec
        self.n_ctx = 2048
        self.verbose = False

    def _ensure_loaded(self):
        if self._llm is None:
            # In a real implementation, this would resolve the model_spec to a path
            # For now, we'll keep the existing logic for backward compatibility
            if self.model_spec == "Qwen_Qwen3-1.7B-Q4_K_M.gguf":
                matches = glob.glob(
                    r"C:\Users\HP\.cache\huggingface\hub\models--bartowski--Qwen_Qwen3-1.7B-GGUF"
                    r"\snapshots\*\Qwen_Qwen3-1.7B-Q4_K_M.gguf"
                )
                if not matches:
                    raise FileNotFoundError("Qwen GGUF not found")
                path = matches[0]
            else:
                # For other models, we'd have different resolution logic
                # This is a placeholder for future implementation
                raise NotImplementedError(
                    f"Model resolution not implemented for {self.model_spec}"
                )

            logger.info(f"Loading LLM from {path}...")
            from llama_cpp import Llama

            self._llm = Llama(
                model_path=path, n_ctx=self.n_ctx, verbose=self.verbose
            )
            logger.info("LLM loaded")

    def respond(self, command: str, memory_context: str = "",
                 history: list[dict] | None = None) -> str:
        """Get a spoken-style response for the user's command.

        Args:
            command: The user's current transcribed command.
            memory_context: Optional memory context block from MemoryManager.
            history: Optional conversation history (list of {role, content} dicts).
        """
        self._ensure_loaded()
        now = datetime.now().strftime("%A, %B %d %Y at %I:%M %p")
        system_prompt = self._get_system_prompt().format(now=now)
        if memory_context:
            system_prompt += "\n\n" + memory_context
        messages = [{"role": "system", "content": system_prompt}]
        # Inject conversation history for multi-turn context
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": command + " /no_think"})
        out = self._llm.create_chat_completion(messages=messages, max_tokens=200)
        text = out["choices"][0]["message"]["content"].strip()
        # Strip any leaked thinking block
        if "######" in text:
            text = text.split("######", 1)[1].strip()
        elif text.startswith("<"):
            # Handle XML-style thinking blocks by finding the closing tag
            # and extracting content after it
            if ">" in text:
                # Find the first closing angle bracket and take everything after it
                close_idx = text.find(">")
                if close_idx != -1:
                    text = text[close_idx + 1:].strip()
                else:
                    text = ""  # Malformed XML, fallback to empty
            else:
                text = ""  # No closing tag found, fallback to empty

        # Additional cleanup: remove leading/trailing whitespace and newlines
        text = text.strip()

        return text

    def _get_system_prompt(self) -> str:
        """Get the system prompt for this LLM."""
        return (
            "You are Beast, a concise voice assistant on the user's Windows PC. "
            "Answer briefly and naturally, as if speaking aloud. No markdown, no lists "
            "unless asked. Keep responses under 2 sentences when possible. "
            "If you don't know personal information about the user, say 'I don't "
            "know that yet. Tell me and I'll remember it if you want.' Never invent "
            "or hallucinate personal facts. The current date and time is {now}."
        )


class ModelRouter:
    """Routes queries to appropriate LLMs based on complexity."""

    def __init__(self):
        # Initialize with different model specs for routing
        self.models = {
            "small": LlamaLLM("Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"),  # Fast, efficient
            "medium": LlamaLLM("Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"),  # Same for now, can differentiate later
            "large": LlamaLLM("Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"),  # Same for now
        }
        # Future: Could specify different quantizations or models here
        self._loaded_models = {}  # Cache for lazy loading

    def _get_model(self, size: str) -> BaseLLM:
        """Get a model by size, loading it if necessary."""
        if size not in self._loaded_models:
            if size in self.models:
                self._loaded_models[size] = self.models[size]
            else:
                # Default to small if size not found
                self._loaded_models[size] = self.models["small"]
        return self._loaded_models[size]

    def route(self, command: str) -> BaseLLM:
        """
        Route a command to the appropriate model based on complexity.

        For initial implementation, we'll use simple heuristics:
        - Very short commands -> small model
        - Medium length with specific keywords -> medium model
        - Long commands or complex requests -> large model
        """
        command_lower = command.lower().strip()
        word_count = len(command_lower.split())

        # Simple routing logic for initial implementation
        if word_count <= 3:
            # Very short commands like "open chrome", "play music"
            return self._get_model("small")
        elif any(
            keyword in command_lower
            for keyword in ["improve", "analyze", "explain", "help me with", "create", "write", "code"]
        ):
            # Keywords suggesting more complex tasks
            if word_count > 10:
                return self._get_model("large")
            else:
                return self._get_model("medium")
        elif word_count > 15:
            # Long commands likely need more capable model
            return self._get_model("large")
        else:
            # Default to medium for normal commands
            return self._get_model("medium")


# Backwards compatibility wrapper
class QwenRouter(QwenLLM):
    """Backwards compatibility wrapper for existing QwenRouter usage."""

    def __init__(self):
        super().__init__("Qwen_Qwen3-1.7B-Q4_K_M.gguf")