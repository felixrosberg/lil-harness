"""The interface every engine (fake, vLLM, Ollama, Hugging Face, hosted) implements."""

from __future__ import annotations

from typing import Protocol

from lil_harness.tools.registry import ToolSpec
from lil_harness.types import AssistantTurn, Message


class ChatBackend(Protocol):
    async def complete(self, messages: list[Message], tools: list[ToolSpec]) -> AssistantTurn:
        """Send the conversation and available tools to the model; return its next turn."""
        ...
