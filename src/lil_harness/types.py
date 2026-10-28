"""Core message types shared by every part of the harness.

These are deliberately our own small dataclasses rather than OpenAI SDK objects.
Each backend converts to and from its engine's wire format at the boundary, so the
agent loop and the tools never depend on any one engine.

The flow: a backend returns an `AssistantTurn`. The loop stores it as a `Message`,
runs each `ToolCall` in it through the registry, which gives back a `ToolResult`, and
stores each result as a `tool` `Message`. Then the backend gets the whole history again.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

Role = Literal["system", "user", "assistant", "tool"]

# Names for the plain types that show up everywhere, so a signature says what it holds.
ToolName: TypeAlias = str
"""A tool's name, as the model calls it, e.g. "read_file"."""
ToolArgs: TypeAlias = dict[str, Any]
"""A tool call's arguments: argument name -> value, e.g. {"path": "a.py", "offset": 1}."""


@dataclass
class ToolCall:
    """A request from the model to run one tool.

    Attributes:
        id: The call's id, chosen by the server. The matching tool message must carry
            the same id, so the model knows which result answers which call.
        name: The tool the model wants to run.
        arguments: The decoded arguments. Empty if they couldn't be decoded.
        raw_arguments: Exactly what the model emitted, before JSON decoding. Most
            tool-calling bugs are only visible here, so we always keep it.
        arguments_error: Set by a backend when `raw_arguments` could not be decoded as
            JSON. The tool is then not run, and this goes back to the model instead.
    """

    id: str
    name: ToolName
    arguments: ToolArgs
    raw_arguments: str = ""
    arguments_error: str | None = None


@dataclass
class ToolResult:
    """The outcome of running a tool, sent back to the model as a `tool` message.

    Attributes:
        tool_call_id: The id of the `ToolCall` this answers.
        name: The tool's name.
        content: The output, or an error message starting with "Error:".
        is_error: True if the call failed in any way (bad arguments, permission
            denied, the tool raised).
    """

    tool_call_id: str
    name: ToolName
    content: str
    is_error: bool = False


@dataclass
class Message:
    """One entry in the conversation history.

    Attributes:
        role: Who wrote it: "system", "user", "assistant" or "tool".
        content: The text. None on assistant messages that only call tools.
        tool_calls: Set on assistant messages that request tools.
        tool_call_id: Set on tool messages: which call this is the answer to.
        name: Set on tool messages: the tool's name.
        reasoning: The model's thinking before this assistant message, if the server
            returned it. Kept for inspection only: backends don't send it back.
    """

    role: Role
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None
    reasoning: str | None = None

    @classmethod
    def system(cls, content: str) -> Message:
        """Make a system message (instructions for the model).

        Args:
            content: The instructions.

        Returns:
            A message with role "system".
        """
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str) -> Message:
        """Make a user message.

        Args:
            content: What the user says.

        Returns:
            A message with role "user".
        """
        return cls(role="user", content=content)

    @classmethod
    def tool(cls, result: ToolResult) -> Message:
        """Turn a tool's result into the message the model will see.

        Args:
            result: The result from `ToolRegistry.execute`.

        Returns:
            A message with role "tool", linked to its call by `tool_call_id`.
        """
        return cls(
            role="tool",
            content=result.content,
            tool_call_id=result.tool_call_id,
            name=result.name,
        )


@dataclass
class Usage:
    """How many tokens one model call used, as the server counted them.

    Attributes:
        prompt_tokens: The tokens the model read: the whole conversation sent, plus the
            tool definitions. Compare it with the context length to see how full it is.
        completion_tokens: The tokens it wrote in reply.
    """

    prompt_tokens: int
    completion_tokens: int


@dataclass
class AssistantTurn:
    """What a backend returns for one model call.

    Attributes:
        content: The model's text, if any.
        tool_calls: The tools it wants run. Empty means this is its final answer.
        finish_reason: Why generation stopped, as the server reported it, for example
            "stop", "tool_calls" or "length" (hit the token limit).
        reasoning: The model's thinking, if the server returned it.
        usage: Token counts for this call, if the server reported them.
    """

    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    reasoning: str | None = None
    usage: Usage | None = None

    def to_message(self) -> Message:
        """Turn this turn into an assistant message for the history.

        Returns:
            A message with role "assistant" and the same content, calls and reasoning.
        """
        return Message(
            role="assistant",
            content=self.content,
            tool_calls=list(self.tool_calls),
            reasoning=self.reasoning,
        )
