"""Step 5: the coding tools, called the way the model calls them (through the registry)."""

import shutil
import sys
from pathlib import Path

import pytest

from lil_harness.agent.loop import run_agent
from lil_harness.backends.fake import FakeBackend
from lil_harness.permissions import PermissionPolicy
from lil_harness.tools.coding import CodingTools, clip
from lil_harness.types import AssistantTurn, Message, ToolCall

SAMPLE = Path(__file__).parent.parent / "examples" / "sample_project"


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("def add(a, b):\n    return a - b\n")
    (tmp_path / "notes.txt").write_text("hello\nworld\nhello again\n")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "junk.py").write_text("hello\n")
    return tmp_path


def auto() -> PermissionPolicy:
    # These tests are about what the tools do; test_step5_permissions.py covers asking.
    return PermissionPolicy.preset("auto")


async def call(tools: CodingTools, name: str, **arguments) -> tuple[str, bool]:
    registry = tools.registry(auto())
    result = await registry.execute(ToolCall(id="1", name=name, arguments=arguments))
    return result.content, result.is_error


@pytest.mark.parametrize("path", ["../outside.txt", "/etc/passwd", "src/../../x"])
async def test_paths_outside_the_workspace_are_refused(workspace, path):
    content, is_error = await call(CodingTools(workspace), "read_file", path=path)
    assert is_error and "outside the workspace" in content


async def test_read_file_numbers_lines_and_pages(workspace):
    tools = CodingTools(workspace)
    content, _ = await call(tools, "read_file", path="notes.txt")
    assert content == "1| hello\n2| world\n3| hello again"

    content, _ = await call(tools, "read_file", path="notes.txt", offset=2, limit=1)
    assert content.startswith("2| world")
    assert "(lines 2-2 of 3; call again with offset=3 for more)" in content


async def test_read_file_errors_say_what_to_do(workspace):
    content, is_error = await call(CodingTools(workspace), "read_file", path="nope.py")
    assert is_error and "list_files" in content
    # A ToolError's message goes to the model as-is, without an exception type name.
    assert content.startswith("Error: 'nope.py' doesn't exist")


async def test_write_file_creates_folders(workspace):
    tools = CodingTools(workspace)
    content, _ = await call(tools, "write_file", path="new/dir/x.py", content="x = 1\n")
    assert content == "Created new/dir/x.py (1 lines)."
    assert (workspace / "new" / "dir" / "x.py").read_text() == "x = 1\n"


async def test_edit_file_replaces_unique_text(workspace):
    tools = CodingTools(workspace)
    content, is_error = await call(
        tools, "edit_file", path="src/app.py", old_text="a - b", new_text="a + b"
    )
    assert not is_error
    assert (workspace / "src" / "app.py").read_text() == "def add(a, b):\n    return a + b\n"
    assert "2|     return a + b" in content  # shows the edited region


async def test_edit_file_refuses_missing_or_ambiguous_text(workspace):
    tools = CodingTools(workspace)
    content, is_error = await call(
        tools, "edit_file", path="notes.txt", old_text="goodbye", new_text="x"
    )
    assert is_error and "not found" in content

    content, is_error = await call(
        tools, "edit_file", path="notes.txt", old_text="hello", new_text="hi"
    )
    assert is_error and "appears 2 times" in content
    assert (workspace / "notes.txt").read_text() == "hello\nworld\nhello again\n"

    content, is_error = await call(
        tools, "edit_file", path="notes.txt", old_text="hello", new_text="hi", replace_all=True
    )
    assert not is_error and "2 places" in content
    assert (workspace / "notes.txt").read_text() == "hi\nworld\nhi again\n"


async def test_edit_file_keeps_windows_line_endings(workspace):
    (workspace / "win.txt").write_bytes(b"one\r\ntwo\r\nthree\r\n")
    _, is_error = await call(
        CodingTools(workspace), "edit_file", path="win.txt", old_text="one\ntwo", new_text="1\n2"
    )
    assert not is_error
    assert (workspace / "win.txt").read_bytes() == b"1\r\n2\r\nthree\r\n"


async def test_list_files_skips_venv(workspace):
    content, _ = await call(CodingTools(workspace), "list_files")
    assert content.splitlines() == ["notes.txt", "src/app.py"]
    content, _ = await call(CodingTools(workspace), "list_files", pattern="**/*.md")
    assert content.startswith("No files match")


async def test_grep(workspace):
    tools = CodingTools(workspace)
    content, _ = await call(tools, "grep", pattern="hel+o")
    assert content.splitlines() == ["notes.txt:1: hello", "notes.txt:3: hello again"]
    content, _ = await call(tools, "grep", pattern="RETURN", ignore_case=True)
    assert content == "src/app.py:2: return a - b"
    content, is_error = await call(tools, "grep", pattern="(")
    assert is_error and "not a valid regular expression" in content


async def test_protected_files_can_be_read_but_not_changed(workspace):
    tools = CodingTools(workspace, protected=[workspace / "notes.txt"])
    content, is_error = await call(tools, "read_file", path="notes.txt")
    assert not is_error
    content, is_error = await call(tools, "write_file", path="notes.txt", content="x")
    assert is_error and "protected" in content
    content, is_error = await call(
        tools, "edit_file", path="notes.txt", old_text="world", new_text="x"
    )
    assert is_error and "protected" in content


async def test_run_command(workspace):
    tools = CodingTools(workspace)
    py = f'"{sys.executable}"'
    content, _ = await call(tools, "run_command", command=f"{py} -c \"print('hi')\"")
    assert content == "exit code 0\nhi"
    content, _ = await call(tools, "run_command", command=f'{py} -c "import sys; sys.exit(3)"')
    assert content.startswith("exit code 3")
    content, is_error = await call(
        tools,
        "run_command",
        command=f'{py} -c "import time; time.sleep(10)"',
        timeout_seconds=1,
    )
    assert is_error and "still running after 1s" in content


def test_clip_keeps_both_ends():
    text = "START" + "x" * 1000 + "END"
    clipped = clip(text, 100)
    assert clipped.startswith("START") and clipped.endswith("END")
    assert "characters cut" in clipped and len(clipped) < 200


async def test_fake_model_fixes_the_sample_project(tmp_path):
    """The whole cycle on the demo project, with a scripted model."""
    project = shutil.copytree(SAMPLE, tmp_path / "sample_project")
    tools = CodingTools(project)

    def turn(name, **arguments):
        return AssistantTurn(tool_calls=[ToolCall(id=name, name=name, arguments=arguments)])

    backend = FakeBackend(
        [
            turn("read_file", path="stats.py"),
            turn(
                "edit_file",
                path="stats.py",
                old_text="/ (len(values) - 1)",
                new_text="/ len(values)",
            ),
            turn("run_command", command=f'"{sys.executable}" test_stats.py'),
            AssistantTurn(content="mean() divided by n - 1; it now divides by n."),
        ]
    )
    result = await run_agent(
        backend, tools.registry(auto()), [Message.user("Fix the failing test.")], max_steps=6
    )
    assert result.stop_reason == "final_answer"
    test_output = [m.content for m in result.messages if m.role == "tool"][-1]
    assert test_output == "exit code 0\nall tests passed"
