"""Core message types shared by every part of the harness.

These are deliberately our own small dataclasses rather than OpenAI SDK objects.
Each backend converts to and from its engine's wire format at the boundary, so the
agent loop and the tools never depend on any one engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class ToolCall:
    """A request from the model to run one tool."""

    id: str
    name: str
    arguments: dict[str, Any]
    # Exactly what the model emitted, before JSON decoding. Most tool-calling bugs
    # are only visible here, so we always keep it.
    raw_arguments: str = ""
    # Set by a backend when `raw_arguments` could not be decoded as JSON. The tool is
    # then not run, and this message goes back to the model instead.
    arguments_error: str | None = None


@dataclass
class ToolResult:
    """The outcome of running a tool, sent back to the model as a `tool` message."""

    tool_call_id: str
    name: str
    content: str
    is_error: bool = False


@dataclass
class Message:
    """One entry in the conversation history."""

    role: Role
    content: str | None = None
    # Set on assistant messages that request tools.
    tool_calls: list[ToolCall] = field(default_factory=list)
    # Set on tool messages: which call this is the answer to.
    tool_call_id: str | None = None
    name: str | None = None
    # The model's thinking before this assistant message, if the server returned it.
    # Kept for inspection only: backends don't send it back to the model.
    reasoning: str | None = None

    @classmethod
    def system(cls, content: str) -> Message:
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str) -> Message:
        return cls(role="user", content=content)

    @classmethod
    def tool(cls, result: ToolResult) -> Message:
        return cls(
            role="tool",
            content=result.content,
            tool_call_id=result.tool_call_id,
            name=result.name,
        )


@dataclass
class AssistantTurn:
    """What a backend returns for one model call."""

    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    reasoning: str | None = None

    def to_message(self) -> Message:
        return Message(
            role="assistant",
            content=self.content,
            tool_calls=list(self.tool_calls),
            reasoning=self.reasoning,
        )
