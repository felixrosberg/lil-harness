"""Step 5: the permission policy decides which tool calls run, ask, or are refused."""

from pathlib import Path

import pytest

from lil_harness.permissions import Approval, PermissionPolicy, describe_call
from lil_harness.tools.coding import CodingTools
from lil_harness.types import ToolCall


class ScriptedApprover:
    """Answers approval questions from a list, and remembers what it was asked."""

    def __init__(self, *answers: Approval):
        self.answers = list(answers)
        self.asked: list[tuple[str, dict]] = []

    async def __call__(self, tool, risk, args):
        self.asked.append((tool, args))
        return self.answers.pop(0)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    (tmp_path / "a.txt").write_text("one\n")
    return tmp_path


async def run(registry, name, **arguments):
    result = await registry.execute(ToolCall(id="1", name=name, arguments=arguments))
    return result.content, result.is_error


def names(registry):
    return {s.name for s in registry.specs()}


def test_readonly_hides_write_and_execute_tools(workspace):
    registry = CodingTools(workspace).registry(PermissionPolicy.preset("readonly"))
    assert names(registry) == {"list_files", "grep", "read_file"}


async def test_denied_tools_are_refused_even_if_called_by_name(workspace):
    registry = CodingTools(workspace).registry(PermissionPolicy.preset("readonly"))
    content, is_error = await run(registry, "write_file", path="b.txt", content="x")
    assert is_error and content.startswith("Error: Permission denied.")
    assert not (workspace / "b.txt").exists()


async def test_without_an_approver_asking_means_no(workspace):
    registry = CodingTools(workspace).registry()  # "default" preset, nobody to ask
    _, is_error = await run(registry, "read_file", path="a.txt")
    assert not is_error
    content, is_error = await run(registry, "edit_file", path="a.txt", old_text="one", new_text="1")
    assert is_error and "nobody is available to approve" in content
    assert (workspace / "a.txt").read_text() == "one\n"


async def test_yes_runs_and_no_sends_the_note_to_the_model(workspace):
    approver = ScriptedApprover(Approval("yes"), Approval("no", "use 'uno' instead"))
    registry = CodingTools(workspace).registry(
        PermissionPolicy.preset("default", approver=approver)
    )

    _, is_error = await run(registry, "edit_file", path="a.txt", old_text="one", new_text="1")
    assert not is_error
    content, is_error = await run(registry, "edit_file", path="a.txt", old_text="1", new_text="2")
    assert is_error and "declined" in content and "use 'uno' instead" in content
    assert (workspace / "a.txt").read_text() == "1\n"
    assert [tool for tool, _ in approver.asked] == ["edit_file", "edit_file"]


async def test_always_for_a_tool_stops_asking_about_that_tool(workspace):
    approver = ScriptedApprover(Approval("always"))
    registry = CodingTools(workspace).registry(
        PermissionPolicy.preset("default", approver=approver)
    )
    await run(registry, "write_file", path="b.txt", content="x")
    await run(registry, "write_file", path="c.txt", content="y")
    assert len(approver.asked) == 1 and (workspace / "c.txt").exists()


async def test_commands_need_an_exact_allowlist_match():
    approver = ScriptedApprover(Approval("no"), Approval("no"))
    policy = PermissionPolicy.preset("default", approver=approver, allow_commands={"git status"})
    allowed = await policy.check("run_command", "execute", {"command": "git  status "})
    assert allowed.allowed and not approver.asked  # whitespace doesn't matter

    for sneaky in ["git status && del *.*", "git status; rm -rf /"]:
        decision = await policy.check("run_command", "execute", {"command": sneaky})
        assert not decision.allowed
    assert [args["command"] for _, args in approver.asked] == [
        "git status && del *.*",
        "git status; rm -rf /",
    ]


async def test_always_for_a_command_means_only_that_command():
    approver = ScriptedApprover(Approval("always"), Approval("no"))
    policy = PermissionPolicy.preset("default", approver=approver)
    for command in ["pytest", "pytest", "pytest -x"]:
        await policy.check("run_command", "execute", {"command": command})
    assert [args["command"] for _, args in approver.asked] == ["pytest", "pytest -x"]


async def test_deny_patterns_win_even_in_auto():
    policy = PermissionPolicy.preset("auto", deny_patterns=[r"git\s+push"])
    decision = await policy.check("run_command", "execute", {"command": "git  push origin"})
    assert not decision.allowed and "blocked" in decision.message
    assert (await policy.check("run_command", "execute", {"command": "git pull"})).allowed


def test_policy_file(tmp_path):
    path = tmp_path / "rules.toml"
    path.write_text(
        'preset = "default"\n'
        '[modes]\nwrite = "allow"\n'
        "[commands]\nallow = [\"git status\"]\ndeny = ['rm\\s+-rf']\n"
    )
    policy = PermissionPolicy.from_file(path)
    assert policy.modes == {"read": "allow", "write": "allow", "execute": "ask"}
    assert policy.allow_commands == {"git status"}
    assert policy.deny_patterns == [r"rm\s+-rf"]
    assert PermissionPolicy.from_file(path, preset="readonly").modes["execute"] == "deny"


def test_the_example_policy_file_loads():
    example = Path(__file__).parent.parent / "permissions.example.toml"
    assert "python test_stats.py" in PermissionPolicy.from_file(example).allow_commands


def test_policy_file_rejects_typos(tmp_path):
    path = tmp_path / "rules.toml"
    path.write_text('[modes]\nwirte = "allow"\n')
    with pytest.raises(ValueError, match="bad mode"):
        PermissionPolicy.from_file(path)


def test_describe_call_shows_a_diff():
    text = describe_call("edit_file", {"path": "a.py", "old_text": "x = 1", "new_text": "x = 2"})
    assert text == "edit a.py:\n- x = 1\n+ x = 2"
    assert describe_call("run_command", {"command": "ls"}) == "$ ls"


async def test_unlabelled_tools_count_as_execute():
    from lil_harness.tools.registry import ToolRegistry

    registry = ToolRegistry(PermissionPolicy.preset("readonly"))

    @registry.tool
    def mystery() -> str:
        """Does something."""
        return "done"

    assert registry.specs() == []
