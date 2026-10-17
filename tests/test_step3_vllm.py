"""Step 3: the vLLM backend, tested against a fake vLLM server (no GPU needed)."""

import json
import os
from typing import Literal

import httpx2 as httpx
import pytest
from openai import AsyncOpenAI

from lil_harness.agent.loop import run_agent
from lil_harness.backends.vllm import VLLMBackend
from lil_harness.tools.registry import ToolRegistry
from lil_harness.types import Message


def fake_vllm(**backend_kwargs) -> tuple[VLLMBackend, list[httpx.Request]]:
    """A VLLMBackend talking to an in-process fake that serves Qwen/Qwen3-4B-AWQ."""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [
                        {
                            "id": "Qwen/Qwen3-4B-AWQ",
                            "object": "model",
                            "created": 0,
                            "owned_by": "vllm",
                        }
                    ],
                },
            )
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 0,
                "model": "Qwen/Qwen3-4B-AWQ",
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
        base_url="http://localhost:8000/v1",
        api_key="EMPTY",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    return VLLMBackend(client=client, **backend_kwargs), requests


async def test_asks_the_server_which_model_it_runs():
    backend, requests = fake_vllm()

    await run_agent(backend, ToolRegistry(), [Message.user("hi")])
    await run_agent(backend, ToolRegistry(), [Message.user("hi again")])

    assert backend.model == "Qwen/Qwen3-4B-AWQ"
    paths = [r.url.path for r in requests]
    # The model list is only fetched once.
    assert paths == ["/v1/models", "/v1/chat/completions", "/v1/chat/completions"]
    assert json.loads(requests[1].content)["model"] == "Qwen/Qwen3-4B-AWQ"


async def test_explicit_model_skips_the_lookup():
    backend, requests = fake_vllm(model="my-model")

    await run_agent(backend, ToolRegistry(), [Message.user("hi")])

    assert [r.url.path for r in requests] == ["/v1/chat/completions"]


@pytest.mark.parametrize("think", [False, True])
async def test_thinking_switch_is_sent_to_the_chat_template(think):
    backend, requests = fake_vllm(model="m", enable_thinking=think)

    await run_agent(backend, ToolRegistry(), [Message.user("hi")])

    body = json.loads(requests[0].content)
    assert body["chat_template_kwargs"] == {"enable_thinking": think}


def test_defaults_point_at_a_local_server():
    backend = VLLMBackend()
    assert str(backend.client.base_url) == "http://localhost:8000/v1/"


@pytest.mark.vllm
async def test_real_vllm_server():
    """Run with: uv run pytest -m vllm (needs scripts/serve_vllm.sh running)."""
    reg = ToolRegistry()

    @reg.tool
    def calculate(a: float, b: float, op: Literal["add", "subtract", "multiply", "divide"]):
        """Do basic arithmetic on two numbers. Use this instead of doing math in your head."""
        return {"add": a + b, "subtract": a - b, "multiply": a * b, "divide": a / b}[op]

    backend = VLLMBackend(base_url=os.environ.get("LIL_VLLM_URL", "http://localhost:8000/v1"))
    result = await run_agent(backend, reg, [Message.user("What is 1234 * 5678?")])

    assert result.stop_reason == "final_answer"
    assert any(m.role == "tool" for m in result.messages), "the model never called the tool"
    assert "7006652" in (result.answer or "").replace(",", "").replace(" ", "")
