"""Step 4: the eval suite, run against a fake model so the grading itself is tested."""

from pathlib import Path

import pytest

from lil_harness.agent.loop import RunResult
from lil_harness.evals.cases import Case, Expect, load_cases
from lil_harness.evals.checks import grade, normalize
from lil_harness.evals.runner import RunRecord, run_suite, summarize
from lil_harness.tools.registry import ToolSpec
from lil_harness.types import AssistantTurn, Message, ToolCall

CASES_DIR = Path(__file__).parent.parent / "evals" / "cases"


def test_the_shipped_cases_load():
    cases = load_cases(CASES_DIR)
    assert len(cases) >= 10
    # Each file name becomes a tag.
    assert {"calculator", "no_tools", "time"} <= {t for c in cases for t in c.tags}


def test_typos_in_a_case_are_rejected():
    with pytest.raises(ValueError):
        Case.model_validate({"id": "x", "prompt": "hi", "expect": {"answer_contain": ["a"]}})


@pytest.mark.parametrize(
    ("text", "expected"),
    [("4,267,635,650", "4267635650"), ("111 563", "111563"), ("Paris", "paris"), ("1, 2", "1, 2")],
)
def test_normalize(text, expected):
    assert normalize(text) == expected


def run_result(*turns: AssistantTurn, tool_contents: list[str] = ()) -> RunResult:
    """Build a finished run: each turn's calls are answered with the next tool_contents."""
    messages = [Message.user("q")]
    contents = iter(tool_contents)
    for turn in turns:
        messages.append(turn.to_message())
        for c in turn.tool_calls:
            messages.append(Message(role="tool", content=next(contents, "ok"), tool_call_id=c.id))
    return RunResult(turns[-1].content, messages, len(turns), "final_answer")


def calc(id: str = "c1") -> ToolCall:
    return ToolCall(id=id, name="calculate", arguments={})


def failed(checks) -> list[str]:
    return [c.name for c in checks if not c.passed]


def test_grade_passing_run():
    result = run_result(AssistantTurn(tool_calls=[calc()]), AssistantTurn(content="It is 111,563."))
    expect = Expect(tools_called=["calculate"], answer_contains=["111563"], max_steps=3)
    assert failed(grade(result, expect)) == []


def test_grade_catches_each_problem():
    result = run_result(
        AssistantTurn(tool_calls=[calc("c1")]),
        AssistantTurn(tool_calls=[calc("c2")]),
        AssistantTurn(content="Maybe 42?"),
        tool_contents=["Error: Invalid arguments", "3"],
    )
    expect = Expect(
        tools_called=["current_time_utc"],
        min_tool_calls=3,
        parallel=True,
        answer_contains=["84"],
        answer_matches=r"\b84\b",
        max_steps=2,
    )
    assert failed(grade(result, expect)) == [
        "called current_time_utc",
        "at least 3 tool calls",
        "calls in one turn",
        "no tool errors",
        "answer contains '84'",
        r"answer matches /\b84\b/",
        "at most 2 steps",
    ]


def test_grade_no_tool_calls():
    used_tool = run_result(AssistantTurn(tool_calls=[calc()]), AssistantTurn(content="Paris"))
    assert failed(grade(used_tool, Expect(no_tool_calls=True))) == ["no tool calls"]


class ScriptedByPrompt:
    """A fake model that answers each question with its own script, so runs can be concurrent."""

    def __init__(self, scripts: dict[str, list[AssistantTurn]]):
        self.scripts = scripts

    async def complete(self, messages: list[Message], tools: list[ToolSpec]) -> AssistantTurn:
        prompt = next(m.content for m in messages if m.role == "user")
        turns_so_far = sum(1 for m in messages if m.role == "assistant")
        return self.scripts[prompt][turns_so_far]


async def test_run_suite_end_to_end():
    cases = [
        Case(
            id="good",
            prompt="3*4?",
            expect=Expect(tools_called=["calculate"], answer_contains=["12"]),
        ),
        Case(id="lazy", prompt="5*5?", expect=Expect(tools_called=["calculate"])),
        Case(id="boom", prompt="crash", expect=Expect()),
    ]
    backend = ScriptedByPrompt(
        {
            "3*4?": [
                AssistantTurn(
                    tool_calls=[
                        ToolCall(
                            id="1",
                            name="calculate",
                            arguments={"a": 3, "b": 4, "op": "multiply"},
                            raw_arguments='{"a": 3, "b": 4, "op": "multiply"}',
                        )
                    ]
                ),
                AssistantTurn(content="12"),
            ],
            "5*5?": [AssistantTurn(content="25")],  # answers without the tool
            # "crash" has no script, so the fake raises KeyError like a broken server.
        }
    )

    records = await run_suite(cases, backend, repeats=2, concurrency=3)
    s = summarize(records)

    assert s.by_case == {"good": (2, 2), "lazy": (0, 2), "boom": (0, 2)}
    assert s.crashed == 2
    assert s.tool_calls == 2 and s.tool_errors == 0
    good = next(r for r in records if r.case_id == "good")
    assert good.tool_calls == [("calculate", '{"a": 3, "b": 4, "op": "multiply"}', False)]
    # "good" called it twice (2 tp), "lazy" skipped it twice (2 fn), "boom" crashed and
    # is left out.
    calc = s.tool_scores["calculate"]
    assert (calc.tp, calc.fp, calc.fn) == (2, 0, 2)
    assert calc.precision == 1.0 and calc.recall == 0.5
    assert s.reliable_cases == 1


def record(case_id, expected, called, passed=True):
    return RunRecord(
        case_id, 0, [], passed,
        tool_calls=[(name, "{}", False) for name in called], expected_tools=expected,
    )  # fmt: skip


def test_tool_selection_scores():
    s = summarize(
        [
            record("a", ["calculate"], ["calculate", "calculate"]),  # tp once per run
            record("b", ["calculate"], ["current_time_utc"]),  # fn calc, fp time
            record("c", [], ["calculate"]),  # should have answered directly: fp
            record("d", None, ["calculate"]),  # case doesn't say: not scored
        ]
    )
    calc, time = s.tool_scores["calculate"], s.tool_scores["current_time_utc"]
    assert (calc.tp, calc.fp, calc.fn) == (1, 1, 1)
    assert calc.precision == calc.recall == calc.f1 == 0.5
    assert (time.tp, time.fp, time.fn) == (0, 1, 0)
    assert time.precision == 0.0 and time.recall is None and time.f1 is None


def test_reliable_cases_needs_every_repeat():
    s = summarize(
        [
            record("steady", None, [], passed=True),
            record("steady", None, [], passed=True),
            record("flaky", None, [], passed=True),
            record("flaky", None, [], passed=False),
        ]
    )
    assert s.reliable_cases == 1
