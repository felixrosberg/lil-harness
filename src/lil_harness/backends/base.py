"""The interface every engine (fake, vLLM, Ollama, Hugging Face, hosted) implements."""

from __future__ import annotations

from typing import Protocol

from lil_harness.tools.registry import ToolSpec
from lil_harness.types import AssistantTurn, Message


class ChatBackend(Protocol):
    """Anything with this `complete` method is a backend; no inheritance needed."""

    async def complete(self, messages: list[Message], tools: list[ToolSpec]) -> AssistantTurn:
        """Send the conversation and available tools to the model; return its next turn.

        Args:
            messages: The whole conversation so far, oldest first.
            tools: The tools the model may call this turn.

        Returns:
            The model's reply: text, tool calls, or both.
        """
        ...
