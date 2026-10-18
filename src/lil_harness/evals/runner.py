"""Run eval cases against a backend, concurrently, and summarize the results."""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import TypeAlias

from lil_harness.agent.loop import run_agent
from lil_harness.backends.base import ChatBackend
from lil_harness.evals.cases import Case
from lil_harness.evals.checks import Check, grade
from lil_harness.tools.builtin import default_registry
from lil_harness.types import Message, ToolName

SYSTEM_PROMPT = "You are a helpful assistant. Use the tools when they help."

CaseName: TypeAlias = str
"""A case's `name` from its YAML file, e.g. "multiply_basic"."""
CaseTag: TypeAlias = str
"""A case tag, e.g. "calculator" (from the file name) or "parallel"."""


@dataclass
class CallRecord:
    """One tool call made during a run, as saved in the results.

    Attributes:
        name: The tool the model called.
        raw_arguments: Exactly what the model wrote as arguments.
        failed: Whether the call ended in an error result.
    """

    name: ToolName
    raw_arguments: str
    failed: bool


@dataclass
class PassCount:
    """How many runs of a case (or of all cases with a tag) passed.

    Attributes:
        passed: Runs where every check passed.
        runs: All runs.
    """

    passed: int = 0
    runs: int = 0

    @property
    def all_passed(self) -> bool:
        """Returns: True if every run passed (and there was at least one)."""
        return self.runs > 0 and self.passed == self.runs

    def add(self, passed: bool) -> None:
        """Count one more run.

        Args:
            passed: Whether that run passed.
        """
        self.passed += passed
        self.runs += 1


@dataclass
class RunRecord:
    """One run of one case. Saved as a line of JSON, so keep it plain data.

    Attributes:
        case_id: Which case this was.
        repeat: Which repeat (0, 1, ...) when a case runs several times.
        tags: The case's tags, for the per-tag scores.
        passed: True if every check passed.
        checks: Every check and its outcome.
        answer: The model's final answer.
        tool_calls: Every tool call the model made, in order.
        steps: How many model calls the run took.
        seconds: Wall-clock time for the run.
        expected_tools: Tools a good run should call; [] means "none". None means the
            case doesn't say, so the run is left out of the tool selection scores.
        error: Set when the run itself blew up (server down, timeout), not when a
            check failed.
    """

    case_id: str
    repeat: int
    tags: list[str]
    passed: bool
    checks: list[Check] = field(default_factory=list)
    answer: str | None = None
    tool_calls: list[CallRecord] = field(default_factory=list)
    steps: int = 0
    seconds: float = 0.0
    expected_tools: list[ToolName] | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        """Returns: the record as plain dicts and lists, ready for json.dumps."""
        return asdict(self)


def expected_tools(case: Case) -> list[ToolName] | None:
    """Which tools a case expects, for the tool selection scores.

    Args:
        case: The eval case.

    Returns:
        Its `tools_called`, [] for a `no_tool_calls` case, or None if it says neither.
    """
    if case.expect.tools_called:
        return list(case.expect.tools_called)
    if case.expect.no_tool_calls:
        return []
    return None


async def run_case(case: Case, backend: ChatBackend, repeat: int, max_steps: int) -> RunRecord:
    """Run one case once, and grade it.

    Args:
        case: What to ask and what to expect.
        backend: The model under test.
        repeat: The repeat number, stored in the record.
        max_steps: The step budget for the agent loop.

    Returns:
        The graded record. Never raises: a crash is recorded in `error`.
    """
    registry = default_registry(case.tools)
    messages = [Message.system(SYSTEM_PROMPT), Message.user(case.prompt)]
    start = time.perf_counter()
    try:
        result = await run_agent(backend, registry, messages, max_steps=max_steps)
    except Exception as e:  # noqa: BLE001 - one broken run must not stop the suite
        return RunRecord(
            case.name, repeat, case.tags, passed=False,
            seconds=time.perf_counter() - start, error=f"{type(e).__name__}: {e}",
            expected_tools=expected_tools(case),
        )  # fmt: skip
    seconds = time.perf_counter() - start

    checks = grade(result, case.expect)
    results_by_id = {
        message.tool_call_id: message for message in result.messages if message.role == "tool"
    }
    tool_calls = [
        CallRecord(
            name=tool_call.name,
            raw_arguments=tool_call.raw_arguments,
            # A call with no result at all (the run stopped first) counts as failed.
            failed=(results_by_id[tool_call.id].content or "").startswith("Error:")
            if tool_call.id in results_by_id
            else True,
        )
        for message in result.messages
        for tool_call in message.tool_calls
    ]
    return RunRecord(
        case_id=case.name,
        repeat=repeat,
        tags=case.tags,
        passed=all(check.passed for check in checks),
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

    Args:
        cases: The cases to run.
        backend: The model under test.
        repeats: How many times to run each case.
        concurrency: The most runs in flight at once.
        max_steps: The step budget for each run.

    Returns:
        One record per run, in case order, then repeat order.
    """
    semaphore = asyncio.Semaphore(concurrency)

    async def limited(case: Case, repeat: int) -> RunRecord:
        """Run one case once, waiting for a free slot first."""
        async with semaphore:
            return await run_case(case, backend, repeat, max_steps)

    return list(
        await asyncio.gather(
            *(limited(case, repeat) for case in cases for repeat in range(repeats))
        )
    )


@dataclass
class ToolScore:
    """Did the model pick this tool when it should, and only then?

    Counted per run: a run that calls the tool twice still counts once.

    Attributes:
        tp: True positives: expected and called.
        fp: False positives: called but not expected.
        fn: False negatives: expected but not called.
    """

    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float | None:
        """Returns: tp / (tp + fp), or None if the tool was never called."""
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else None

    @property
    def recall(self) -> float | None:
        """Returns: tp / (tp + fn), or None if no case expected the tool."""
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else None

    @property
    def f1(self) -> float | None:
        """Returns: the harmonic mean of precision and recall, or None if either is."""
        p, r = self.precision, self.recall
        if p is None or r is None:
            return None
        return 2 * p * r / (p + r) if p + r else 0.0


@dataclass
class Summary:
    """The numbers in the report.

    Attributes:
        runs: Total runs.
        passed: Runs where every check passed.
        by_case: Pass counts per case, in case order.
        by_tag: Pass counts per tag, sorted by tag.
        tool_calls: Tool calls made across all runs.
        tool_errors: Of those, how many failed.
        avg_steps: Mean model calls per run.
        avg_seconds: Mean wall-clock seconds per run.
        crashed: Runs that raised instead of finishing.
        tool_scores: Precision/recall counts per tool, sorted by name.
        reliable_cases: Cases that passed every one of their repeats.
    """

    runs: int
    passed: int
    by_case: dict[CaseName, PassCount]
    by_tag: dict[CaseTag, PassCount]
    tool_calls: int
    tool_errors: int
    avg_steps: float
    avg_seconds: float
    crashed: int
    tool_scores: dict[ToolName, ToolScore]
    reliable_cases: int


def summarize(records: list[RunRecord]) -> Summary:
    """Add up the records into the report's numbers.

    Args:
        records: The runs from `run_suite`.

    Returns:
        The totals, per case, per tag and per tool.
    """
    # defaultdict creates an empty PassCount the first time a case or tag shows up.
    by_case: dict[CaseName, PassCount] = defaultdict(PassCount)
    by_tag: dict[CaseTag, PassCount] = defaultdict(PassCount)
    for record in records:
        by_case[record.case_id].add(record.passed)
        for tag in record.tags:
            by_tag[tag].add(record.passed)
    calls = [tool_call for record in records for tool_call in record.tool_calls]
    num_records = len(records) or 1

    tool_scores: dict[ToolName, ToolScore] = defaultdict(ToolScore)
    for record in records:
        if record.expected_tools is None or record.error:
            continue
        expected = set(record.expected_tools)
        called = {tool_call.name for tool_call in record.tool_calls}
        for name in expected & called:
            tool_scores[name].tp += 1
        for name in called - expected:
            tool_scores[name].fp += 1
        for name in expected - called:
            tool_scores[name].fn += 1

    return Summary(
        runs=len(records),
        passed=sum(record.passed for record in records),
        by_case=dict(by_case),
        by_tag=dict(sorted(by_tag.items())),
        tool_calls=len(calls),
        tool_errors=sum(1 for call in calls if call.failed),
        avg_steps=sum(record.steps for record in records) / num_records,
        avg_seconds=sum(record.seconds for record in records) / num_records,
        crashed=sum(1 for record in records if record.error),
        tool_scores=dict(sorted(tool_scores.items())),
        reliable_cases=sum(1 for count in by_case.values() if count.all_passed),
    )
