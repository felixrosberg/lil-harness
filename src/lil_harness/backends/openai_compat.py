"""Backend for any server that speaks the OpenAI `/v1/chat/completions` API.

That covers hosted providers (OpenRouter, Together, Groq, ...) and local engines
(vLLM, Ollama, llama.cpp server). This file is the only place that knows the OpenAI
wire format: it converts our types to request JSON and the response back.
"""

from __future__ import annotations

import json
import os
from typing import Any, TypeAlias

from openai import AsyncOpenAI

from lil_harness.tools.registry import ToolSpec
from lil_harness.types import AssistantTurn, Message, ToolArgs, ToolCall

# The OpenAI wire format is plain JSON. These names say which piece of it a dict is.
OpenAIRequest: TypeAlias = dict[str, Any]
"""The body of POST /v1/chat/completions: model, messages, tools, temperature, ..."""
OpenAIMessage: TypeAlias = dict[str, Any]
"""One entry of the request's "messages", like {"role": "user", "content": "Hi"}."""
OpenAITool: TypeAlias = dict[str, Any]
"""One entry of the request's "tools": {"type": "function", "function": {...}}."""


class OpenAICompatBackend:
    """Talks to one model on one OpenAI-compatible server."""

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str | None = None,
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        extra_body: dict[str, Any] | None = None,
        client: AsyncOpenAI | None = None,
    ) -> None:
        """Set up the client. Nothing is sent until `complete` is called.

        Args:
            model: The model name as the server knows it, e.g. "qwen3:4b".
            base_url: The server's API root, ending in /v1.
            api_key: The key for hosted providers. Local servers ignore it.
            temperature: Sampling temperature. 0 makes runs as repeatable as the
                server allows, which is what you want for tool calls and evals.
            max_tokens: Cap on the reply's length. None lets the server decide.
            extra_body: Engine-specific request fields (vLLM's chat_template_kwargs,
                Ollama's reasoning_effort, ...) that the SDK passes through untouched.
            client: A ready-made client, so tests can plug in a fake transport.
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_body = extra_body or {}
        # Local servers usually ignore the key, but the SDK requires one.
        self.client = client or AsyncOpenAI(base_url=base_url, api_key=api_key or "EMPTY")

    @classmethod
    def from_env(cls, **kwargs: Any) -> OpenAICompatBackend:
        """Build from the LIL_BASE_URL, LIL_MODEL and LIL_API_KEY environment variables.

        Args:
            **kwargs: Passed on to the constructor, e.g. temperature.

        Returns:
            A backend for that server and model.

        Raises:
            RuntimeError: If LIL_BASE_URL or LIL_MODEL isn't set.
        """
        missing = [var for var in ("LIL_BASE_URL", "LIL_MODEL") if not os.environ.get(var)]
        if missing:
            raise RuntimeError(f"Set these environment variables first: {', '.join(missing)}")
        return cls(
            model=os.environ["LIL_MODEL"],
            base_url=os.environ["LIL_BASE_URL"],
            api_key=os.environ.get("LIL_API_KEY"),
            **kwargs,
        )

    async def complete(self, messages: list[Message], tools: list[ToolSpec]) -> AssistantTurn:
        """Send one chat completion request and convert the reply.

        Args:
            messages: The conversation so far.
            tools: The tools the model may call.

        Returns:
            The model's turn, with tool calls parsed by the server and our own checks
            for broken arguments (see `tool_call_from_openai`).

        Raises:
            openai.APIConnectionError: If the server can't be reached.
            openai.APIStatusError: If the server rejects the request.
        """
        request: OpenAIRequest = {
            "model": self.model,
            "messages": [message_to_openai(message) for message in messages],
            "temperature": self.temperature,
        }
        # Some servers reject an empty tools list, so only send it when there are tools.
        if tools:
            request["tools"] = [tool_to_openai(tool) for tool in tools]
        if self.max_tokens is not None:
            request["max_tokens"] = self.max_tokens
        if self.extra_body:
            request["extra_body"] = self.extra_body

        response = await self.client.chat.completions.create(**request)
        choice = response.choices[0]
        return AssistantTurn(
            content=choice.message.content,
            tool_calls=[tool_call_from_openai(call) for call in choice.message.tool_calls or []],
            finish_reason=choice.finish_reason,
            reasoning=reasoning_from_openai(choice.message),
        )


def reasoning_from_openai(message: Any) -> str | None:
    """The model's thinking, which isn't part of the official OpenAI format.

    Servers that separate it from the answer put it in an extra field, and the name
    varies: vLLM uses `reasoning` (older versions `reasoning_content`), other servers
    may use `thinking`. The SDK keeps unknown fields in `model_extra`.

    Args:
        message: The SDK's message object from the response.

    Returns:
        The thinking text, or None if the server didn't send any.
    """
    extra = message.model_extra or {}
    for key in ("reasoning", "reasoning_content", "thinking"):
        if extra.get(key):
            return extra[key]
    return None


def tool_to_openai(spec: ToolSpec) -> OpenAITool:
    """Describe one tool in OpenAI's format.

    Args:
        spec: The tool from the registry.

    Returns:
        A {"type": "function", "function": {...}} dict, which the server renders into
        the prompt through the model's chat template.
    """
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.parameters,
        },
    }


def message_to_openai(message: Message) -> OpenAIMessage:
    """Convert one of our messages to OpenAI's format.

    Args:
        message: A message from the history.

    Returns:
        The request dict for that message. Reasoning is left out on purpose: models
        are trained to see their past answers, not their past thinking.
    """
    if message.role == "tool":
        return {
            "role": "tool",
            "tool_call_id": message.tool_call_id,
            "content": message.content or "",
        }

    out: OpenAIMessage = {"role": message.role, "content": message.content}
    if message.tool_calls:
        out["tool_calls"] = [
            {
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.name,
                    # Send back exactly what the model produced, so its own history
                    # isn't silently "fixed" for it.
                    "arguments": tool_call.raw_arguments or json.dumps(tool_call.arguments),
                },
            }
            for tool_call in message.tool_calls
        ]
    return out


def tool_call_from_openai(call: Any) -> ToolCall:
    """Convert one tool call from the response, keeping broken arguments visible.

    Args:
        call: The SDK's tool call object.

    Returns:
        A ToolCall. If the arguments aren't a JSON object, `arguments` is empty and
        `arguments_error` says why, so the registry can tell the model.
    """
    raw = call.function.arguments or ""
    arguments: ToolArgs = {}
    error: str | None = None
    try:
        # Some models send "" for a tool with no parameters.
        decoded = json.loads(raw) if raw.strip() else {}
        if isinstance(decoded, dict):
            arguments = decoded
        else:
            error = f"expected a JSON object, got {type(decoded).__name__}"
    except json.JSONDecodeError as e:
        error = str(e)
    return ToolCall(
        id=call.id,
        name=call.function.name,
        arguments=arguments,
        raw_arguments=raw,
        arguments_error=error,
    )
