"""Backend for a local vLLM server.

vLLM speaks the OpenAI API, so this is a thin layer over `OpenAICompatBackend` that adds
what's specific to vLLM: local defaults, finding out which model the server is running,
and vLLM's extra request options.

The server itself is started separately (see `scripts/serve_vllm.sh`), because vLLM only
runs on Linux. On Windows it runs inside WSL2, while this code can run on either side.
"""

from __future__ import annotations

from typing import Any

from lil_harness.backends.openai_compat import OpenAICompatBackend
from lil_harness.tools.registry import ToolSpec
from lil_harness.types import AssistantTurn, Message

DEFAULT_BASE_URL = "http://localhost:8000/v1"


class VLLMBackend(OpenAICompatBackend):
    """An OpenAICompatBackend with vLLM's address, model detection and a thinking switch."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        *,
        enable_thinking: bool = False,
        **kwargs: Any,
    ) -> None:
        """Set up a backend for the model a vLLM server is running.

        Args:
            model: The name the server was started with. Leave it out to ask the
                server on the first call.
            base_url: Where the vLLM server listens.
            enable_thinking: For hybrid reasoning models like Qwen3. Off by default: on
                a small GPU, thinking costs a lot of tokens and rarely helps simple
                tool calls.
            **kwargs: Passed on to OpenAICompatBackend, e.g. temperature.
        """
        extra_body = dict(kwargs.pop("extra_body", None) or {})
        # vLLM passes chat_template_kwargs into the model's chat template. Qwen3's
        # template reads `enable_thinking`; templates that don't know it ignore it.
        extra_body.setdefault("chat_template_kwargs", {"enable_thinking": enable_thinking})
        super().__init__(model or "", base_url, extra_body=extra_body, **kwargs)

    async def served_model(self) -> str:
        """Ask the server which model it is running (GET /v1/models).

        Returns:
            The first model's id. A vLLM server serves one model.

        Raises:
            RuntimeError: If the server lists no models.
        """
        page = await self.client.models.list()
        if not page.data:
            raise RuntimeError("The vLLM server reports no models")
        return page.data[0].id

    async def complete(self, messages: list[Message], tools: list[ToolSpec]) -> AssistantTurn:
        """Like OpenAICompatBackend.complete, but finds the model name first if needed.

        Args:
            messages: The conversation so far.
            tools: The tools the model may call.

        Returns:
            The model's turn.
        """
        if not self.model:
            self.model = await self.served_model()
        return await super().complete(messages, tools)
