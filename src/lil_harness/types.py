"""Core message types shared by every part of the harness.

These are deliberately our own small dataclasses rather than OpenAI SDK objects.
Each backend converts to and from its engine's wire format at the boundary, so the
agent loop and the tools never depend on any one engine.

The flow: a backend returns an `AssistantTurn`. The loop stores it as a `Message`,
runs each `ToolCall` in it through the registry, which gives back a `ToolResult`, and
stores each result as a `tool` `Message`. Then the backend gets the whole history again.

Images ride along on messages: a user message can carry the pictures it's about, and a
tool can return an `Image` or a `ToolOutput` with text and images, which end up on its
`ToolResult` and `tool` message. The backend decides how they reach the model.
"""

from __future__ import annotations

import base64
import mimetypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeAlias

Role = Literal["system", "user", "assistant", "tool"]

# Names for the plain types that show up everywhere, so a signature says what it holds.
ToolName: TypeAlias = str
"""A tool's name, as the model calls it, e.g. "read_file"."""
ToolArgs: TypeAlias = dict[str, Any]
"""A tool call's arguments: argument name -> value, e.g. {"path": "a.py", "offset": 1}."""


IMAGE_TYPES = ("image/png", "image/jpeg", "image/gif", "image/webp")
"""The formats vision models' servers accept."""


@dataclass(frozen=True)
class Image:
    """An image for the model to see: the encoded bytes and their format.

    Attributes:
        data: The file's bytes, e.g. a PNG or JPEG.
        mime: The format, e.g. "image/png".
    """

    data: bytes = field(repr=False)
    mime: str

    @classmethod
    def from_path(cls, path: str | Path) -> Image:
        """Read an image file as it is. To shrink a big one first, use
        `lil_harness.tools.media.load_image`.

        Args:
            path: A PNG, JPEG, GIF or WebP file.

        Returns:
            The image, its format guessed from the file name.

        Raises:
            ValueError: If the file name doesn't look like one of those formats.
        """
        mime, _ = mimetypes.guess_type(str(path))
        if mime not in IMAGE_TYPES:
            raise ValueError(f"{path} isn't a PNG, JPEG, GIF or WebP image")
        return cls(Path(path).read_bytes(), mime)

    def to_data_url(self) -> str:
        """Returns: the image as a data: URL, the way the OpenAI API takes images."""
        return f"data:{self.mime};base64,{base64.b64encode(self.data).decode()}"


@dataclass
class ToolOutput:
    """What a tool returns when it has images to show, besides text.

    Attributes:
        text: What the model reads, e.g. "cat.jpg: 4000x3000 pixels, shown at 1024x768."
        images: What it sees.
    """

    text: str
    images: list[Image] = field(default_factory=list)


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
        images: Images the tool returned, for the model to see.
    """

    tool_call_id: str
    name: ToolName
    content: str
    is_error: bool = False
    images: list[Image] = field(default_factory=list)


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
        images: Images on user and tool messages, for the model to see.
    """

    role: Role
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None
    reasoning: str | None = None
    images: list[Image] = field(default_factory=list)

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
    def user(cls, content: str, images: list[Image] | None = None) -> Message:
        """Make a user message.

        Args:
            content: What the user says.
            images: Pictures to go with it, e.g. from `load_image`.

        Returns:
            A message with role "user".
        """
        return cls(role="user", content=content, images=list(images or []))

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
            images=list(result.images),
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
