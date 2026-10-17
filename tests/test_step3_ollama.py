"""Step 3: the Ollama backend, tested against a fake Ollama server (no GPU needed)."""

import json
from typing import Literal

import httpx2 as httpx
import pytest
from openai import AsyncOpenAI

from lil_harness.agent.loop import run_agent
from lil_harness.backends.ollama import OllamaBackend
from lil_harness.tools.registry import ToolRegistry
from lil_harness.types import Message


def fake_ollama(**backend_kwargs) -> tuple[OllamaBackend, list[dict]]:
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 0,
                "model": "qwen3:4b",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hello"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    client = AsyncOpenAI(
        base_url="http://localhost:11434/v1",
        api_key="EMPTY",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    return OllamaBackend(client=client, **backend_kwargs), bodies


def test_defaults_point_at_a_local_server():
    backend = OllamaBackend()
    assert backend.model == "qwen3:4b"
    assert str(backend.client.base_url) == "http://localhost:11434/v1/"


@pytest.mark.parametrize(("think", "expected"), [(False, "none"), (True, None)])
async def test_thinking_switch(think, expected):
    backend, bodies = fake_ollama(enable_thinking=think)

    await run_agent(backend, ToolRegistry(), [Message.user("hi")])

    assert bodies[0].get("reasoning_effort") == expected


@pytest.mark.ollama
async def test_real_ollama_server():
    """Run with: uv run pytest -m ollama (needs Ollama running and qwen3:4b pulled)."""
    reg = ToolRegistry()

    @reg.tool
    def calculate(a: float, b: float, op: Literal["add", "subtract", "multiply", "divide"]):
        """Do basic arithmetic on two numbers. Use this instead of doing math in your head."""
        return {"add": a + b, "subtract": a - b, "multiply": a * b, "divide": a / b}[op]

    result = await run_agent(OllamaBackend(), reg, [Message.user("What is 1234 * 5678?")])

    assert result.stop_reason == "final_answer"
    assert any(m.role == "tool" for m in result.messages), "the model never called the tool"
    assert "7006652" in (result.answer or "").replace(",", "").replace(" ", "")
