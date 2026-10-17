from typing import Literal

import pytest

from lil_harness.agent.loop import run_agent
from lil_harness.backends.fake import FakeBackend
from lil_harness.tools.registry import ToolRegistry
from lil_harness.types import AssistantTurn, Message, ToolCall


@pytest.fixture
def registry() -> ToolRegistry:
    reg = ToolRegistry()

    @reg.tool
    def calculate(a: float, b: float, op: Literal["add", "subtract", "multiply", "divide"]):
        """Do basic arithmetic on two numbers. Use this instead of doing math in your head."""
        if op == "add":
            return a + b
        if op == "subtract":
            return a - b
        if op == "multiply":
            return a * b
        return a / b

    return reg


def call(name: str, id: str = "call_1", **arguments) -> ToolCall:
    return ToolCall(id=id, name=name, arguments=arguments)


def test_schema_is_generated_from_type_hints(registry: ToolRegistry):
    (spec,) = registry.specs()
    assert spec.name == "calculate"
    assert spec.description.startswith("Do basic arithmetic")
    assert spec.parameters == {
        "type": "object",
        "properties": {
            "a": {"type": "number"},
            "b": {"type": "number"},
            "op": {"type": "string", "enum": ["add", "subtract", "multiply", "divide"]},
        },
        "required": ["a", "b", "op"],
    }


async def test_model_calls_calculator_then_answers(registry: ToolRegistry):
    backend = FakeBackend(
        script=[
            AssistantTurn(tool_calls=[call("calculate", a=17, b=25, op="multiply")]),
            AssistantTurn(content="17 times 25 is 425."),
        ]
    )

    result = await run_agent(backend, registry, [Message.user("What is 17 * 25?")])

    assert result.stop_reason == "final_answer"
    assert result.answer == "17 times 25 is 425."
    assert result.steps == 2
    # The second model call saw the tool result in its history.
    tool_msg = backend.calls[1].messages[-1]
    assert tool_msg.role == "tool"
    assert tool_msg.tool_call_id == "call_1"
    assert tool_msg.content == "425.0"


async def test_parallel_calls_keep_their_order(registry: ToolRegistry):
    backend = FakeBackend(
        script=[
            AssistantTurn(
                tool_calls=[
                    call("calculate", id="c1", a=1, b=2, op="add"),
                    call("calculate", id="c2", a=10, b=4, op="subtract"),
                ]
            ),
            AssistantTurn(content="3 and 6"),
        ]
    )

    result = await run_agent(backend, registry, [Message.user("1+2 and 10-4?")])

    tool_msgs = [m for m in result.messages if m.role == "tool"]
    assert [(m.tool_call_id, m.content) for m in tool_msgs] == [("c1", "3.0"), ("c2", "6.0")]


@pytest.mark.parametrize(
    ("bad_call", "expected"),
    [
        (call("calculator", a=1, b=2, op="add"), "Unknown tool 'calculator'"),
        (call("calculate", a=1, op="add"), "b: Field required"),
        (call("calculate", a=1, b=2, op="power"), "op: Input should be"),
        (call("calculate", a=1, b=0, op="divide"), "ZeroDivisionError"),
    ],
)
async def test_errors_go_back_to_the_model(registry: ToolRegistry, bad_call, expected):
    backend = FakeBackend(
        script=[AssistantTurn(tool_calls=[bad_call]), AssistantTurn(content="ok")]
    )

    result = await run_agent(backend, registry, [Message.user("hi")])

    tool_msg = result.messages[2]
    assert tool_msg.role == "tool"
    assert tool_msg.content.startswith("Error:")
    assert expected in tool_msg.content
    assert result.stop_reason == "final_answer"


async def test_stops_at_max_steps(registry: ToolRegistry):
    looping = AssistantTurn(tool_calls=[call("calculate", a=1, b=1, op="add")])
    backend = FakeBackend(script=[looping] * 5)

    result = await run_agent(backend, registry, [Message.user("hi")], max_steps=3)

    assert result.stop_reason == "max_steps"
    assert result.answer is None
    assert len(backend.calls) == 3


async def test_async_tools_are_awaited():
    reg = ToolRegistry()

    @reg.tool
    async def echo(text: str) -> str:
        """Repeat the text back."""
        return text

    result = await reg.execute(call("echo", text="hello"))
    assert result.content == "hello"
    assert not result.is_error
