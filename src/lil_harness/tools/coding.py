"""Coding tools: read, search and edit files in one folder, and run commands there.

    tools = CodingTools("path/to/project")
    registry = tools.registry(PermissionPolicy.preset("default", approver=...))

Two rules shape every tool here:

- **Stay in the workspace.** Every path is resolved against the workspace folder, and
  anything that ends up outside it (`../..`, an absolute path, a symlink out) is
  refused. `run_command` is the exception: a shell can reach anything your user
  account can. That's what the permission policy is for (see permissions.py).
- **Keep outputs small.** A small model's context fills up fast, and one 3,000-line
  file would push out the task itself. So reads are paged, listings and searches are
  capped, and long command output keeps only its start and end. Each cut says how to
  get the rest, so the model can ask for it.
"""

from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path

from lil_harness.permissions import PermissionPolicy
from lil_harness.tools.registry import ToolError, ToolRegistry

# Folders that are almost never what the model is looking for, and can be huge.
IGNORED_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache",
                ".ruff_cache", ".mypy_cache", "dist", "build"}  # fmt: skip

SHELL = "cmd.exe" if os.name == "nt" else "sh"


def clip(text: str, max_chars: int, hint: str = "") -> str:
    """Cut `text` to about `max_chars`, keeping the start and the end.

    The end of command output is usually the important part (the error, the test
    summary), so both ends survive and the middle goes.
    """
    if len(text) <= max_chars:
        return text
    head = text[: max_chars // 3]
    tail = text[-(max_chars - len(head)) :]
    cut = len(text) - len(head) - len(tail)
    return f"{head}\n[... {cut} characters cut{'. ' + hint if hint else ''} ...]\n{tail}"


class CodingTools:
    def __init__(
        self,
        root: str | Path,
        *,
        max_output_chars: int = 6000,
        read_limit: int = 200,
        protected: list[str | Path] = (),
    ) -> None:
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise ValueError(f"Workspace {self.root} is not a folder")
        # Files the model may read but never change, like the permission policy file
        # itself: otherwise it could edit its own rules.
        self.protected = {Path(p).resolve() for p in protected}
        self.max_output_chars = max_output_chars
        self.read_limit = read_limit

    def registry(self, policy: PermissionPolicy | None = None) -> ToolRegistry:
        """A registry with the tools bound to this workspace.

        Without a policy you get the "default" preset with nobody to ask, so the model
        can look around but every write and command is refused. Pass a policy with an
        approver to let a human decide, or the "auto" preset inside a sandbox.
        """
        registry = ToolRegistry(policy or PermissionPolicy.preset("default"))
        for fn in (self.list_files, self.grep, self.read_file):
            registry.tool(fn, risk="read")
        for fn in (self.write_file, self.edit_file):
            registry.tool(fn, risk="write")
        registry.tool(
            self.run_command,
            risk="execute",
            description=f"Run a shell command in the workspace folder with {SHELL} and "
            "return its exit code and output. Use it to run tests, scripts or git.",
        )
        return registry

    # ---- helpers -------------------------------------------------------------

    def _resolve(self, path: str) -> Path:
        target = (self.root / path).resolve()
        if not target.is_relative_to(self.root):
            raise ToolError(
                f"{path!r} is outside the workspace. Use paths relative to the workspace "
                "folder, like 'src/main.py'."
            )
        return target

    def _writable(self, path: str) -> Path:
        target = self._resolve(path)
        if target in self.protected:
            raise ToolError(f"{path!r} is protected and can't be changed.")
        return target

    def _rel(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix() or "."

    def _read_text(self, path: str) -> tuple[Path, str]:
        target = self._resolve(path)
        if target.is_dir():
            raise ToolError(f"{path!r} is a folder. Use list_files to see what's in it.")
        if not target.exists():
            raise ToolError(f"{path!r} doesn't exist. Use list_files to find the right path.")
        try:
            # newline="" keeps Windows \r\n line endings as they are, so edits keep them.
            with target.open(encoding="utf-8", newline="") as f:
                return target, f.read()
        except UnicodeDecodeError:
            raise ToolError(f"{path!r} is not a UTF-8 text file.") from None

    def _files(self, path: str, pattern: str) -> list[Path]:
        base = self._resolve(path)
        if not base.is_dir():
            raise ToolError(f"{path!r} is not a folder.")
        return sorted(
            p
            for p in base.glob(pattern)
            if p.is_file()
            and not IGNORED_DIRS.intersection(p.relative_to(self.root).parts)
            and p.resolve().is_relative_to(self.root)
        )

    # ---- the tools -------------------------------------------------------------

    def list_files(self, pattern: str = "**/*", path: str = ".") -> str:
        """List files in the workspace that match a glob pattern, like '**/*.py' or 'src/*'.

        Folders such as .git, .venv and node_modules are skipped.
        """
        limit = 200
        files = self._files(path, pattern)
        if not files:
            return f"No files match {pattern!r} in {path!r}."
        lines = [self._rel(p) for p in files[:limit]]
        if len(files) > limit:
            lines.append(f"[... {len(files) - limit} more. Use a narrower pattern ...]")
        return "\n".join(lines)

    def grep(
        self, pattern: str, path: str = ".", file_pattern: str = "**/*", ignore_case: bool = False
    ) -> str:
        """Search file contents with a regular expression. Returns 'file:line: text' matches.

        Narrow the search with path (a folder) or file_pattern (a glob like '**/*.py').
        """
        try:
            regex = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
        except re.error as e:
            raise ToolError(f"{pattern!r} is not a valid regular expression: {e}") from None
        limit = 100
        matches: list[str] = []
        for file in self._files(path, file_pattern):
            if file.stat().st_size > 1_000_000:
                continue
            try:
                lines = file.read_text(encoding="utf-8").splitlines()
            except UnicodeDecodeError:
                continue
            for number, line in enumerate(lines, 1):
                if regex.search(line):
                    matches.append(f"{self._rel(file)}:{number}: {line.strip()[:200]}")
                    if len(matches) > limit:
                        return "\n".join(matches[:limit]) + (
                            f"\n[... more than {limit} matches. Narrow the pattern or path ...]"
                        )
        return "\n".join(matches) or f"No matches for {pattern!r}."

    def read_file(self, path: str, offset: int = 1, limit: int | None = None) -> str:
        """Read a text file. Lines come numbered like '  12| text'; the numbers and the '| '
        are not part of the file.

        Long files are returned in pages: offset is the first line to read (starting at 1)
        and limit how many lines.
        """
        _, text = self._read_text(path)
        lines = text.splitlines()
        limit = limit or self.read_limit
        start = max(offset, 1)
        page = lines[start - 1 : start - 1 + limit]
        if not page:
            return f"{path!r} has {len(lines)} lines, so there is nothing at line {start}."
        width = len(str(start + len(page) - 1))
        body = "\n".join(
            f"{n:>{width}}| {line[:500]}{' [line cut]' if len(line) > 500 else ''}"
            for n, line in enumerate(page, start)
        )
        end = start + len(page) - 1
        if start > 1 or end < len(lines):
            body += f"\n(lines {start}-{end} of {len(lines)}"
            body += f"; call again with offset={end + 1} for more)" if end < len(lines) else ")"
        return clip(body, self.max_output_chars, "Read fewer lines with limit")

    def write_file(self, path: str, content: str) -> str:
        """Create a file, or replace a file's whole content. Creates missing folders.

        To change part of an existing file, use edit_file instead.
        """
        target = self._writable(path)
        if target.is_dir():
            raise ToolError(f"{path!r} is a folder.")
        target.parent.mkdir(parents=True, exist_ok=True)
        existed = target.exists()
        with target.open("w", encoding="utf-8", newline="") as f:
            f.write(content)
        lines = len(content.splitlines())
        return f"{'Replaced' if existed else 'Created'} {self._rel(target)} ({lines} lines)."

    def edit_file(self, path: str, old_text: str, new_text: str, replace_all: bool = False) -> str:
        """Replace an exact piece of text in a file with new text.

        old_text must match the file exactly, including indentation, and appear exactly
        once, unless replace_all is true. Read the file first, and include a few
        surrounding lines if the text you want to change is not unique.
        """
        self._writable(path)
        target, text = self._read_text(path)
        if old_text == new_text:
            raise ToolError("old_text and new_text are the same, so there is nothing to change.")
        if not old_text:
            raise ToolError("old_text is empty. To create or overwrite a file, use write_file.")
        # The model writes \n, but a file saved on Windows may use \r\n.
        if old_text not in text and "\r\n" in text:
            old_text = old_text.replace("\r\n", "\n").replace("\n", "\r\n")
            new_text = new_text.replace("\r\n", "\n").replace("\n", "\r\n")
        count = text.count(old_text)
        if count == 0:
            raise ToolError(
                f"old_text was not found in {path!r}. It must match exactly, including "
                "spaces and indentation, and without the line numbers from read_file. "
                "Read the file again and copy the text."
            )
        if count > 1 and not replace_all:
            raise ToolError(
                f"old_text appears {count} times in {path!r}. Include more surrounding lines "
                "so it is unique, or set replace_all to true."
            )
        with target.open("w", encoding="utf-8", newline="") as f:
            f.write(text.replace(old_text, new_text))

        # Show the edited region, so the model can check the result without a re-read.
        before = text[: text.index(old_text)]
        first = before.count("\n") + 1
        shown = self.read_file(
            path, offset=max(first - 3, 1), limit=new_text.count("\n") + 7
        ).split("\n(lines")[0]
        times = f"{count} places" if count > 1 else "1 place"
        return f"Edited {self._rel(target)} in {times}. Around the first edit:\n{shown}"

    async def run_command(self, command: str, timeout_seconds: int = 60) -> str:
        # Its description is set in registry(), because it names the shell.
        timeout_seconds = min(max(timeout_seconds, 1), 600)
        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=self.root,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout_seconds)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            raise ToolError(
                f"The command was still running after {timeout_seconds}s, so it was stopped. "
                "If it needs longer, raise timeout_seconds; if it waits for input, it can't "
                "be run here."
            ) from None
        output = out.decode("utf-8", errors="replace").strip() or "(no output)"
        return f"exit code {proc.returncode}\n" + clip(output, self.max_output_chars)
