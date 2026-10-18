"""The agent loop: ask the model, run the tools it asks for, repeat until it answers."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Literal

from lil_harness.backends.base import ChatBackend
from lil_harness.tools.registry import ToolRegistry
from lil_harness.types import Message

StopReason = Literal["final_answer", "max_steps"]


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
) -> RunResult:
    """Run until the model replies without tool calls, or `max_steps` model calls are used.

    Args:
        backend: The model to talk to.
        registry: The tools the model may call. Its `specs()` are sent with every call.
        messages: The conversation so far, usually a system and a user message. It is
            copied, not modified.
        max_steps: The most model calls to make. Stops a model that keeps calling
            tools forever.

    Returns:
        The final answer, the full history, and why the run stopped.
    """
    history = list(messages)
    tools = registry.specs()

    for step in range(1, max_steps + 1):
        turn = await backend.complete(history, tools)
        history.append(turn.to_message())

        if not turn.tool_calls:
            return RunResult(turn.content, history, step, "final_answer")

        # Independent tool calls in one turn run concurrently; results keep call order.
        results = await asyncio.gather(*(registry.execute(c) for c in turn.tool_calls))
        history.extend(Message.tool(r) for r in results)

    return RunResult(None, history, max_steps, "max_steps")
