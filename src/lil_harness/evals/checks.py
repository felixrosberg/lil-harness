"""Grade one agent run against a case's expectations."""

from __future__ import annotations

import re
from dataclasses import dataclass

from lil_harness.agent.loop import RunResult
from lil_harness.evals.cases import Expect


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""


def normalize(text: str) -> str:
    """Lowercase, and drop thousands separators so "4,267,635,650" matches "4267635650"."""
    return re.sub(r"(?<=\d)[,\s](?=\d{3}\b)", "", text).lower()


def grade(result: RunResult, expect: Expect) -> list[Check]:
    assistant_turns = [m for m in result.messages if m.role == "assistant"]
    calls = [c for m in assistant_turns for c in m.tool_calls]
    called = [c.name for c in calls]
    tool_errors = [
        m.content
        for m in result.messages
        if m.role == "tool" and (m.content or "").startswith("Error:")
    ]
    answer = normalize(result.answer or "")

    checks = [
        Check(
            "finished",
            result.stop_reason == "final_answer",
            f"stopped because of {result.stop_reason}",
        )
    ]
    for name in expect.tools_called:
        checks.append(Check(f"called {name}", name in called, f"called: {called or 'nothing'}"))
    if expect.no_tool_calls:
        checks.append(Check("no tool calls", not calls, f"called: {called}"))
    if expect.min_tool_calls is not None:
        checks.append(
            Check(
                f"at least {expect.min_tool_calls} tool calls",
                len(calls) >= expect.min_tool_calls,
                f"made {len(calls)}",
            )
        )
    if expect.parallel:
        turns_with_calls = sum(1 for m in assistant_turns if m.tool_calls)
        checks.append(
            Check(
                "calls in one turn",
                turns_with_calls == 1,
                f"tool calls spread over {turns_with_calls} turns",
            )
        )
    if expect.no_tool_errors:
        checks.append(Check("no tool errors", not tool_errors, "; ".join(tool_errors)[:200]))
    for text in expect.answer_contains:
        checks.append(
            Check(
                f"answer contains {text!r}", normalize(text) in answer, (result.answer or "")[:200]
            )
        )
    if expect.answer_matches is not None:
        checks.append(
            Check(
                f"answer matches /{expect.answer_matches}/",
                re.search(expect.answer_matches, result.answer or "", re.IGNORECASE) is not None,
                (result.answer or "")[:200],
            )
        )
    if expect.max_steps is not None:
        checks.append(
            Check(
                f"at most {expect.max_steps} steps",
                result.steps <= expect.max_steps,
                f"took {result.steps}",
            )
        )
    return checks
