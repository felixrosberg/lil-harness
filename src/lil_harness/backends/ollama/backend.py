"""Backend for a local Ollama server.

Ollama runs natively on Windows, macOS and Linux and exposes an OpenAI-compatible API,
so this is a thin layer over `OpenAICompatBackend` with Ollama's defaults. Get a model
first with `ollama pull qwen3:4b`.
"""

from __future__ import annotations

from typing import Any

from lil_harness.backends.openai_compat import OpenAICompatBackend

DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_MODEL = "qwen3:4b"


class OllamaBackend(OpenAICompatBackend):
    """An OpenAICompatBackend with Ollama's address, a default model and a thinking switch."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        *,
        enable_thinking: bool = False,
        **kwargs: Any,
    ) -> None:
        """Set up a backend for a model served by Ollama.

        Args:
            model: An Ollama tag you have pulled. Unlike vLLM, one Ollama server can
                serve any pulled model, so the name is always needed.
            base_url: Where Ollama listens.
            enable_thinking: For hybrid reasoning models like Qwen3. Off by default: on
                a small GPU, thinking costs a lot of tokens and rarely helps simple
                tool calls.
            **kwargs: Passed on to OpenAICompatBackend, e.g. temperature.
        """
        extra_body = dict(kwargs.pop("extra_body", None) or {})
        if not enable_thinking:
            # Ollama maps reasoning_effort "none" to "no thinking" for models whose
            # thinking is a simple on/off switch, like Qwen3.
            extra_body.setdefault("reasoning_effort", "none")
        super().__init__(model, base_url, extra_body=extra_body, **kwargs)
