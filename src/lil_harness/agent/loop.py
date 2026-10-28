"""The agent loop: ask the model, run the tools it asks for, repeat until it answers."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal, TypeAlias

from lil_harness.backends.base import ChatBackend
from lil_harness.tools.registry import ToolRegistry
from lil_harness.types import AssistantTurn, Message, ToolCall, ToolResult

StopReason = Literal["final_answer", "max_steps"]


@dataclass
class TurnEvent:
    """The model has replied. Sent before any of its tool calls run.

    Attributes:
        step: Which model call this was, counting from 1.
        turn: The reply: text, tool calls, or both. No tool calls means it's the answer.
    """

    step: int
    turn: AssistantTurn


@dataclass
class ToolEvent:
    """One tool call has finished. Sent as each one finishes, so not always in call order.

    Attributes:
        step: The model call that asked for this tool.
        call: What the model asked for.
        result: What goes back to the model.
    """

    step: int
    call: ToolCall
    result: ToolResult


StepEvent: TypeAlias = TurnEvent | ToolEvent
"""Something that happened during a run, for showing progress as it happens."""

OnStep: TypeAlias = Callable[[StepEvent], Awaitable[None] | None]
"""A function, plain or async, that `run_agent` calls with each `StepEvent`."""


@dataclass
class RunResult:
    """How an agent run ended.

    Attributes:
        answer: The model's final text, or None if it ran out of steps.
        messages: The full history: the messages you passed in, then every assistant
            and tool message the run added.
        steps: How many times the model was called.
        stop_reason: "final_answer" if the model replied without tool calls,
            "max_steps" if it was still calling tools when the budget ran out.
    """

    answer: str | None
    messages: list[Message]
    steps: int
    stop_reason: StopReason


async def run_agent(
    backend: ChatBackend,
    registry: ToolRegistry,
    messages: list[Message],
    *,
    max_steps: int = 10,
    on_step: OnStep | None = None,
) -> RunResult:
    """Run until the model replies without tool calls, or `max_steps` model calls are used.

    Args:
        backend: The model to talk to.
        registry: The tools the model may call. Its `specs()` are sent with every call.
        messages: The conversation so far, usually a system and a user message. It is
            copied, not modified.
        max_steps: The most model calls to make. Stops a model that keeps calling
            tools forever.
        on_step: Called with a `TurnEvent` after each model reply and a `ToolEvent`
            as each tool call finishes. Use it to show a run live, in a terminal or
            a GUI. The run waits for it, so keep it quick. If it raises, the run stops
            with that error.

    Returns:
        The final answer, the full history, and why the run stopped.
    """
    history = list(messages)
    tools = registry.specs()

    async def emit(event: StepEvent) -> None:
        if on_step is None:
            return
        outcome = on_step(event)
        if inspect.isawaitable(outcome):
            await outcome

    async def execute(step: int, tool_call: ToolCall) -> ToolResult:
        result = await registry.execute(tool_call)
        await emit(ToolEvent(step, tool_call, result))
        return result

    for step in range(1, max_steps + 1):
        turn = await backend.complete(history, tools)
        history.append(turn.to_message())
        await emit(TurnEvent(step, turn))

        if not turn.tool_calls:
            return RunResult(turn.content, history, step, "final_answer")

        # Independent tool calls in one turn run concurrently; results keep call order.
        results = await asyncio.gather(*(execute(step, tool_call) for tool_call in turn.tool_calls))
        history.extend(Message.tool(result) for result in results)

    return RunResult(None, history, max_steps, "max_steps")
