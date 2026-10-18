"""Eval cases: what to ask the model, and what a good run looks like.

Cases live in YAML files under `evals/cases/`, one file per tool or theme. Each file
holds a list of cases; see `evals/cases/calculator.yaml` for examples.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class Expect(BaseModel):
    """Checks for one case. Every check that is set must pass.

    Each field's description (below) says what it checks; `checks.grade` turns them
    into pass/fail results.
    """

    # Reject typos like `answer_contain:` instead of silently ignoring them.
    model_config = ConfigDict(extra="forbid")

    tools_called: list[str] = Field(
        default_factory=list, description="Each of these tools must be called at least once."
    )
    no_tool_calls: bool = Field(False, description="The model must answer without any tools.")
    min_tool_calls: int | None = Field(None, description="At least this many tool calls in total.")
    parallel: bool = Field(False, description="All tool calls must be in a single turn.")
    no_tool_errors: bool = Field(
        True, description="No tool call may fail (bad arguments, unknown tool, crash)."
    )
    answer_contains: list[str] = Field(
        default_factory=list,
        description="Each string must appear in the final answer (case-insensitive, "
        "thousands separators ignored).",
    )
    answer_matches: str | None = Field(None, description="A regex the final answer must match.")
    max_steps: int | None = Field(None, description="At most this many model calls.")


class Case(BaseModel):
    """One eval case, as written in a YAML file.

    Attributes:
        name: A unique name, shown in the report.
        prompt: The user message the model gets.
        tools: Which built-in tools to offer. None offers all of them.
        tags: Labels for grouping scores; the file name is added automatically.
        expect: What a good run looks like.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    prompt: str
    tools: list[str] | None = Field(None, description="Tools to offer. Default: all built-ins.")
    tags: list[str] = Field(default_factory=list)
    expect: Expect = Field(default_factory=Expect)


def load_cases(directory: Path) -> list[Case]:
    """Load every case from the .yaml files in `directory`, in file and list order.

    Args:
        directory: The folder with the case files, usually evals/cases.

    Returns:
        All cases, each tagged with its file's name.

    Raises:
        ValueError: If a file isn't a list, or two cases share a name.
        pydantic.ValidationError: If a case has a typo or a wrong type.
    """
    cases: list[Case] = []
    for path in sorted(directory.glob("*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
        if not isinstance(raw, list):
            raise ValueError(f"{path}: expected a list of cases")
        for item in raw:
            case = Case.model_validate(item)
            # The file name becomes a tag, so `--tag calculator` selects that file.
            if path.stem not in case.tags:
                case.tags.append(path.stem)
            cases.append(case)

    case_names = [case.name for case in cases]
    duplicates = sorted({name for name in case_names if case_names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate case names: {', '.join(duplicates)}")
    return cases
