"""Backend for any server that speaks the OpenAI `/v1/chat/completions` API.

That covers hosted providers (OpenRouter, Together, Groq, ...) and local engines
(vLLM, Ollama, llama.cpp server). This file is the only place that knows the OpenAI
wire format: it converts our types to request JSON and the response back.
"""

from __future__ import annotations

import json
import os
from typing import Any

from openai import AsyncOpenAI

from lil_harness.tools.registry import ToolSpec
from lil_harness.types import AssistantTurn, Message, ToolCall


class OpenAICompatBackend:
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
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        # Engine-specific request fields (vLLM's guided decoding, OpenRouter's provider
        # routing, ...) that the OpenAI SDK passes through untouched.
        self.extra_body = extra_body or {}
        # Local servers usually ignore the key, but the SDK requires one.
        self.client = client or AsyncOpenAI(base_url=base_url, api_key=api_key or "EMPTY")

    @classmethod
    def from_env(cls, **kwargs: Any) -> OpenAICompatBackend:
        """Build from LIL_BASE_URL, LIL_MODEL and LIL_API_KEY."""
        missing = [v for v in ("LIL_BASE_URL", "LIL_MODEL") if not os.environ.get(v)]
        if missing:
            raise RuntimeError(f"Set these environment variables first: {', '.join(missing)}")
        return cls(
            model=os.environ["LIL_MODEL"],
            base_url=os.environ["LIL_BASE_URL"],
            api_key=os.environ.get("LIL_API_KEY"),
            **kwargs,
        )

    async def complete(self, messages: list[Message], tools: list[ToolSpec]) -> AssistantTurn:
        request: dict[str, Any] = {
            "model": self.model,
            "messages": [message_to_openai(m) for m in messages],
            "temperature": self.temperature,
        }
        # Some servers reject an empty tools list, so only send it when there are tools.
        if tools:
            request["tools"] = [tool_to_openai(t) for t in tools]
        if self.max_tokens is not None:
            request["max_tokens"] = self.max_tokens
        if self.extra_body:
            request["extra_body"] = self.extra_body

        response = await self.client.chat.completions.create(**request)
        choice = response.choices[0]
        return AssistantTurn(
            content=choice.message.content,
            tool_calls=[tool_call_from_openai(c) for c in choice.message.tool_calls or []],
            finish_reason=choice.finish_reason,
            reasoning=reasoning_from_openai(choice.message),
        )


def reasoning_from_openai(message: Any) -> str | None:
    """The model's thinking, which isn't part of the official OpenAI format.

    Servers that separate it from the answer put it in an extra field, and the name
    varies: vLLM uses `reasoning` (older versions `reasoning_content`), other servers
    may use `thinking`. The SDK keeps unknown fields in `model_extra`.
    """
    extra = message.model_extra or {}
    for key in ("reasoning", "reasoning_content", "thinking"):
        if extra.get(key):
            return extra[key]
    return None


def tool_to_openai(spec: ToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.parameters,
        },
    }


def message_to_openai(m: Message) -> dict[str, Any]:
    if m.role == "tool":
        return {"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content or ""}

    out: dict[str, Any] = {"role": m.role, "content": m.content}
    if m.tool_calls:
        out["tool_calls"] = [
            {
                "id": c.id,
                "type": "function",
                "function": {
                    "name": c.name,
                    # Send back exactly what the model produced, so its own history
                    # isn't silently "fixed" for it.
                    "arguments": c.raw_arguments or json.dumps(c.arguments),
                },
            }
            for c in m.tool_calls
        ]
    return out


def tool_call_from_openai(call: Any) -> ToolCall:
    raw = call.function.arguments or ""
    arguments: dict[str, Any] = {}
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
