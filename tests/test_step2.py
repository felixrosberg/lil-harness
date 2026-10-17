"""Step 2: the OpenAI-compatible backend, tested against a fake HTTP server.

`httpx2.MockTransport` (httpx2 is the HTTP library the OpenAI SDK uses) answers
the SDK's requests in-process, so these tests check the real JSON on the wire without
any network or API key.
"""

import json
import os
from typing import Literal

import httpx2 as httpx
import pytest
from openai import AsyncOpenAI

from lil_harness.agent.loop import run_agent
from lil_harness.backends.openai_compat import OpenAICompatBackend
from lil_harness.tools.registry import ToolRegistry
from lil_harness.types import Message


@pytest.fixture
def registry() -> ToolRegistry:
    reg = ToolRegistry()

    @reg.tool
    def calculate(a: float, b: float, op: Literal["add", "subtract", "multiply", "divide"]):
        """Do basic arithmetic on two numbers. Use this instead of doing math in your head."""
        return {"add": a + b, "subtract": a - b, "multiply": a * b, "divide": a / b}[op]

    return reg


def completion(content=None, tool_calls=None) -> dict:
    """A minimal /v1/chat/completions response body."""
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content, "tool_calls": tool_calls},
                "finish_reason": "tool_calls" if tool_calls else "stop",
            }
        ],
    }


def tool_call(arguments: str, id: str = "call_1", name: str = "calculate") -> dict:
    return {"id": id, "type": "function", "function": {"name": name, "arguments": arguments}}


def fake_server(*responses: dict) -> tuple[OpenAICompatBackend, list[dict]]:
    """A backend whose HTTP requests are answered by `responses`, in order.

    Returns the backend and a list that collects every request body it sent.
    """
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=responses[len(requests) - 1])

    client = AsyncOpenAI(
        base_url="http://test/v1",
        api_key="test",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    return OpenAICompatBackend(model="test-model", base_url="unused", client=client), requests


async def test_round_trip_over_the_wire(registry: ToolRegistry):
    backend, requests = fake_server(
        completion(tool_calls=[tool_call('{"a": 17, "b": 25, "op": "multiply"}')]),
        completion(content="17 times 25 is 425."),
    )

    result = await run_agent(backend, registry, [Message.user("What is 17 * 25?")])

    assert result.answer == "17 times 25 is 425."
    first, second = requests

    # The tool schema is sent in OpenAI's format.
    assert first["model"] == "test-model"
    assert first["temperature"] == 0.0
    assert first["tools"][0]["type"] == "function"
    assert first["tools"][0]["function"]["name"] == "calculate"
    assert first["tools"][0]["function"]["parameters"]["required"] == ["a", "b", "op"]

    # The second request carries the model's call and our result, linked by id.
    assistant, tool = second["messages"][1:]
    assert assistant["tool_calls"][0]["id"] == "call_1"
    assert assistant["tool_calls"][0]["function"]["arguments"] == (
        '{"a": 17, "b": 25, "op": "multiply"}'
    )
    assert tool == {"role": "tool", "tool_call_id": "call_1", "content": "425.0"}


async def test_no_tools_key_when_there_are_no_tools():
    backend, requests = fake_server(completion(content="hi"))

    await run_agent(backend, ToolRegistry(), [Message.user("hello")])

    assert "tools" not in requests[0]


async def test_invalid_json_arguments_go_back_to_the_model(registry: ToolRegistry):
    backend, requests = fake_server(
        completion(tool_calls=[tool_call('{"a": 17, "b": 25, "op": "multiply"')]),  # no }
        completion(content="Sorry, retrying."),
    )

    await run_agent(backend, registry, [Message.user("What is 17 * 25?")])

    assistant, tool = requests[1]["messages"][1:]
    # The model's broken JSON is sent back as-is, so it can see its own mistake.
    assert assistant["tool_calls"][0]["function"]["arguments"].endswith('"multiply"')
    assert tool["content"].startswith("Error: Arguments for 'calculate' were not valid JSON")


@pytest.mark.parametrize(("raw", "args"), [("", {}), ("{}", {})])
async def test_empty_arguments_are_an_empty_object(raw, args):
    reg = ToolRegistry()

    @reg.tool
    def ping() -> str:
        """Check that tools work."""
        return "pong"

    backend, requests = fake_server(
        completion(tool_calls=[tool_call(raw, name="ping")]), completion(content="done")
    )

    await run_agent(backend, reg, [Message.user("ping")])

    assert requests[1]["messages"][-1]["content"] == "pong"


@pytest.mark.hosted
async def test_real_hosted_model(registry: ToolRegistry):
    """Run with: uv run pytest -m hosted (needs the LIL_* environment variables)."""
    if not os.environ.get("LIL_MODEL"):
        pytest.skip("LIL_MODEL is not set")
    backend = OpenAICompatBackend.from_env()

    result = await run_agent(backend, registry, [Message.user("What is 1234 * 5678?")])

    assert result.stop_reason == "final_answer"
    assert any(m.role == "tool" for m in result.messages), "the model never called the tool"
    assert "7006652" in (result.answer or "").replace(",", "").replace(" ", "")


@pytest.mark.parametrize("field", ["reasoning", "reasoning_content", "thinking"])
async def test_thinking_is_kept_but_not_sent_back(field):
    response = completion(content="hi")
    response["choices"][0]["message"][field] = "The user greets me."
    backend, requests = fake_server(response, completion(content="bye"))

    first = await run_agent(backend, ToolRegistry(), [Message.user("hello")])
    assert first.messages[-1].reasoning == "The user greets me."

    # On the next request, the earlier thinking is not part of the history we send.
    await run_agent(backend, ToolRegistry(), [*first.messages, Message.user("bye")])
    assert all("reasoning" not in m and "thinking" not in m for m in requests[1]["messages"])
