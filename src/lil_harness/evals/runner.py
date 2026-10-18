"""Run eval cases against a backend, concurrently, and summarize the results."""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field

from lil_harness.agent.loop import run_agent
from lil_harness.backends.base import ChatBackend
from lil_harness.evals.cases import Case
from lil_harness.evals.checks import Check, grade
from lil_harness.tools.builtin import default_registry
from lil_harness.types import Message

SYSTEM_PROMPT = "You are a helpful assistant. Use the tools when they help."


@dataclass
class RunRecord:
    """One run of one case. Saved as a line of JSON, so keep it plain data."""

    case_id: str
    repeat: int
    tags: list[str]
    passed: bool
    checks: list[Check] = field(default_factory=list)
    answer: str | None = None
    # (tool name, exact arguments the model wrote, whether the call failed)
    tool_calls: list[tuple[str, str, bool]] = field(default_factory=list)
    steps: int = 0
    seconds: float = 0.0
    # Tools a good run should call; [] means "none". None means the case doesn't say,
    # so the run is left out of the tool selection scores.
    expected_tools: list[str] | None = None
    # Set when the run itself blew up (server down, timeout), not when a check failed.
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def expected_tools(case: Case) -> list[str] | None:
    if case.expect.tools_called:
        return list(case.expect.tools_called)
    if case.expect.no_tool_calls:
        return []
    return None


async def run_case(case: Case, backend: ChatBackend, repeat: int, max_steps: int) -> RunRecord:
    registry = default_registry(case.tools)
    messages = [Message.system(SYSTEM_PROMPT), Message.user(case.prompt)]
    start = time.perf_counter()
    try:
        result = await run_agent(backend, registry, messages, max_steps=max_steps)
    except Exception as e:  # noqa: BLE001 - one broken run must not stop the suite
        return RunRecord(
            case.id, repeat, case.tags, passed=False,
            seconds=time.perf_counter() - start, error=f"{type(e).__name__}: {e}",
            expected_tools=expected_tools(case),
        )  # fmt: skip
    seconds = time.perf_counter() - start

    checks = grade(result, case.expect)
    results_by_id = {m.tool_call_id: m for m in result.messages if m.role == "tool"}
    tool_calls = [
        (
            c.name,
            c.raw_arguments,
            (results_by_id[c.id].content or "").startswith("Error:")
            if c.id in results_by_id
            else True,
        )
        for m in result.messages
        for c in m.tool_calls
    ]
    return RunRecord(
        case_id=case.id,
        repeat=repeat,
        tags=case.tags,
        passed=all(c.passed for c in checks),
        checks=checks,
        answer=result.answer,
        tool_calls=tool_calls,
        steps=result.steps,
        seconds=seconds,
        expected_tools=expected_tools(case),
    )


async def run_suite(
    cases: list[Case],
    backend: ChatBackend,
    *,
    repeats: int = 1,
    concurrency: int = 4,
    max_steps: int = 6,
) -> list[RunRecord]:
    """Run every case `repeats` times, with at most `concurrency` runs in flight.

    Models aren't fully deterministic even at temperature 0, so repeats show how
    reliable a pass is. The semaphore keeps us from flooding a small local server.
    """
    semaphore = asyncio.Semaphore(concurrency)

    async def limited(case: Case, repeat: int) -> RunRecord:
        async with semaphore:
            return await run_case(case, backend, repeat, max_steps)

    return list(await asyncio.gather(*(limited(c, r) for c in cases for r in range(repeats))))


@dataclass
class ToolScore:
    """Did the model pick this tool when it should, and only then?

    Counted per run: a run that calls the tool twice still counts once.
    """

    tp: int = 0  # expected and called
    fp: int = 0  # called but not expected
    fn: int = 0  # expected but not called

    @property
    def precision(self) -> float | None:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else None

    @property
    def recall(self) -> float | None:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else None

    @property
    def f1(self) -> float | None:
        p, r = self.precision, self.recall
        if p is None or r is None:
            return None
        return 2 * p * r / (p + r) if p + r else 0.0


@dataclass
class Summary:
    runs: int
    passed: int
    by_case: dict[str, tuple[int, int]]  # case id -> (passed, runs)
    by_tag: dict[str, tuple[int, int]]
    tool_calls: int
    tool_errors: int
    avg_steps: float
    avg_seconds: float
    crashed: int
    tool_scores: dict[str, ToolScore]
    # Cases that passed every one of their repeats.
    reliable_cases: int


def summarize(records: list[RunRecord]) -> Summary:
    by_case: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    by_tag: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in records:
        by_case[r.case_id][0] += r.passed
        by_case[r.case_id][1] += 1
        for tag in r.tags:
            by_tag[tag][0] += r.passed
            by_tag[tag][1] += 1
    calls = [c for r in records for c in r.tool_calls]
    n = len(records) or 1

    tool_scores: dict[str, ToolScore] = defaultdict(ToolScore)
    for r in records:
        if r.expected_tools is None or r.error:
            continue
        expected = set(r.expected_tools)
        called = {c[0] for c in r.tool_calls}
        for name in expected & called:
            tool_scores[name].tp += 1
        for name in called - expected:
            tool_scores[name].fp += 1
        for name in expected - called:
            tool_scores[name].fn += 1

    return Summary(
        runs=len(records),
        passed=sum(r.passed for r in records),
        by_case={k: (v[0], v[1]) for k, v in by_case.items()},
        by_tag={k: (v[0], v[1]) for k, v in sorted(by_tag.items())},
        tool_calls=len(calls),
        tool_errors=sum(1 for c in calls if c[2]),
        avg_steps=sum(r.steps for r in records) / n,
        avg_seconds=sum(r.seconds for r in records) / n,
        crashed=sum(1 for r in records if r.error),
        tool_scores=dict(sorted(tool_scores.items())),
        reliable_cases=sum(1 for ok, total in by_case.values() if ok == total),
    )
