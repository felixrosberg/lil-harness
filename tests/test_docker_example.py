"""The Docker demo's run script (examples/docker/run_task.py), run here without Docker."""

import importlib.util
import json
import sys
from pathlib import Path

from lil_harness.backends.fake import FakeBackend
from lil_harness.permissions import PermissionPolicy
from lil_harness.types import AssistantTurn, ToolCall

EXAMPLE = Path(__file__).parent.parent / "examples" / "docker" / "run_task.py"
SAMPLE = Path(__file__).parent.parent / "examples" / "sample_project"

spec = importlib.util.spec_from_file_location("run_task", EXAMPLE)
run_task = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_task)


def test_seed_fills_an_empty_workspace_only(tmp_path):
    workspace = tmp_path / "workspace"

    assert run_task.seed_workspace(SAMPLE, workspace)
    (workspace / "stats.py").write_text("changed")
    assert not run_task.seed_workspace(SAMPLE, workspace)

    assert (workspace / "stats.py").read_text() == "changed"
    assert (workspace / "test_stats.py").exists()


def test_check_reports_pass_and_fail(tmp_path):
    passed = run_task.run_check(f'"{sys.executable}" -c "print(1)"', tmp_path)
    failed = run_task.run_check(f'"{sys.executable}" -c "raise SystemExit(3)"', tmp_path)

    assert (passed["passed"], passed["exit_code"], passed["output"]) == (True, 0, "1\n")
    assert (failed["passed"], failed["exit_code"]) == (False, 3)


async def test_a_run_fixes_the_sample_and_its_record_is_json(tmp_path):
    workspace = tmp_path / "workspace"
    run_task.seed_workspace(SAMPLE, workspace)
    edit = ToolCall(
        id="1",
        name="edit_file",
        arguments={"path": "stats.py", "old_text": "(len(values) - 1)", "new_text": "len(values)"},
    )
    backend = FakeBackend([AssistantTurn(tool_calls=[edit]), AssistantTurn(content="Fixed.")])

    result = await run_task.run_task(
        backend, "Fix the bug.", workspace, PermissionPolicy.preset("auto")
    )
    check = run_task.run_check(f'"{sys.executable}" test_stats.py', workspace)
    record = run_task.make_record("Fix the bug.", "fake", result, 1.0, check)

    assert check["passed"]
    assert (record["stop_reason"], record["steps"], record["answer"]) == (
        "final_answer",
        2,
        "Fixed.",
    )
    assert json.loads(json.dumps(record))["messages"][2]["tool_calls"][0]["name"] == "edit_file"
