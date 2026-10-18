"""Coding tools: read, search and edit files in one folder, and run commands there.

    tools = CodingTools("path/to/project")
    registry = tools.registry(PermissionPolicy.preset("default", approver=...))

The tools are methods, so they can share the workspace settings. Their docstrings
are what the model reads: the summary becomes the tool's description and each `Args:`
entry that argument's description (see registry.py). `Returns:` is for you.

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

    Args:
        text: The text to shorten.
        max_chars: Roughly how long the result may be (the cut notice adds a little).
        hint: Added to the cut notice, to tell the model how to get the rest.

    Returns:
        `text` unchanged if it fits, otherwise its first third and the rest from its
        end, with a notice of how much was cut in between.
    """
    if len(text) <= max_chars:
        return text
    head = text[: max_chars // 3]
    tail = text[-(max_chars - len(head)) :]
    cut = len(text) - len(head) - len(tail)
    return f"{head}\n[... {cut} characters cut{'. ' + hint if hint else ''} ...]\n{tail}"


class CodingTools:
    """The coding tools, bound to one workspace folder."""

    def __init__(
        self,
        root: str | Path,
        *,
        max_output_chars: int = 6000,
        read_limit: int = 200,
        protected: list[str | Path] = (),
    ) -> None:
        """Set up the tools for one folder.

        Args:
            root: The workspace folder. Every path the model gives is relative to it.
            max_output_chars: The most characters any one tool result may have.
            read_limit: How many lines read_file returns when the model doesn't say.
            protected: Files the model may read but never change, like the permission
                policy file itself: otherwise it could edit its own rules.

        Raises:
            ValueError: If `root` isn't an existing folder.
        """
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise ValueError(f"Workspace {self.root} is not a folder")
        self.protected = {Path(path).resolve() for path in protected}
        self.max_output_chars = max_output_chars
        self.read_limit = read_limit

    def registry(self, policy: PermissionPolicy | None = None) -> ToolRegistry:
        """A registry with the tools bound to this workspace.

        Without a policy you get the "default" preset with nobody to ask, so the model
        can look around but every write and command is refused. Pass a policy with an
        approver to let a human decide, or the "auto" preset inside a sandbox.

        Args:
            policy: Decides which calls run, ask, or are refused.

        Returns:
            A new registry with all six tools. Tools the policy denies outright are
            registered but not offered to the model.
        """
        registry = ToolRegistry(policy or PermissionPolicy.preset("default"))
        for tool in (self.list_files, self.grep, self.read_file):
            registry.tool(tool, risk="read")
        for tool in (self.write_file, self.edit_file):
            registry.tool(tool, risk="write")
        registry.tool(
            self.run_command,
            risk="execute",
            description=f"Run a shell command in the workspace folder with {SHELL} and "
            "return its exit code and output. Use it to run tests, scripts or git.",
        )
        return registry

    # ---- helpers -------------------------------------------------------------

    def _resolve(self, path: str) -> Path:
        """Turn a path from the model into an absolute path inside the workspace.

        Args:
            path: A path relative to the workspace.

        Returns:
            The resolved absolute path (symlinks followed).

        Raises:
            ToolError: If the path ends up outside the workspace.
        """
        target = (self.root / path).resolve()
        if not target.is_relative_to(self.root):
            raise ToolError(
                f"{path!r} is outside the workspace. Use paths relative to the workspace "
                "folder, like 'src/main.py'."
            )
        return target

    def _writable(self, path: str) -> Path:
        """Like `_resolve`, but also refuses protected files.

        Raises:
            ToolError: If the path is outside the workspace or protected.
        """
        target = self._resolve(path)
        if target in self.protected:
            raise ToolError(f"{path!r} is protected and can't be changed.")
        return target

    def _rel(self, path: Path) -> str:
        """Returns: `path` relative to the workspace, with forward slashes."""
        return path.relative_to(self.root).as_posix() or "."

    def _read_text(self, path: str) -> tuple[Path, str]:
        """Read a whole text file, keeping its line endings.

        Args:
            path: A path relative to the workspace.

        Returns:
            The resolved path and the file's text.

        Raises:
            ToolError: If it's a folder, missing, or not UTF-8 text.
        """
        target = self._resolve(path)
        if target.is_dir():
            raise ToolError(f"{path!r} is a folder. Use list_files to see what's in it.")
        if not target.exists():
            raise ToolError(f"{path!r} doesn't exist. Use list_files to find the right path.")
        try:
            # newline="" keeps Windows \r\n line endings as they are, so edits keep them.
            with target.open(encoding="utf-8", newline="") as file:
                return target, file.read()
        except UnicodeDecodeError:
            raise ToolError(f"{path!r} is not a UTF-8 text file.") from None

    def _files(self, path: str, pattern: str) -> list[Path]:
        """Find files under `path` matching a glob, skipping IGNORED_DIRS.

        Args:
            path: A folder relative to the workspace.
            pattern: A glob like "**/*.py", relative to that folder.

        Returns:
            The matching files, sorted, all inside the workspace.

        Raises:
            ToolError: If `path` isn't a folder in the workspace.
        """
        base = self._resolve(path)
        if not base.is_dir():
            raise ToolError(f"{path!r} is not a folder.")
        return sorted(
            file
            for file in base.glob(pattern)
            if file.is_file()
            and not IGNORED_DIRS.intersection(file.relative_to(self.root).parts)
            and file.resolve().is_relative_to(self.root)
        )

    # ---- the tools -------------------------------------------------------------

    def list_files(self, pattern: str = "**/*", path: str = ".") -> str:
        """List files in the workspace that match a glob pattern.

        Folders such as .git, .venv and node_modules are skipped.

        Args:
            pattern: A glob pattern, like '**/*.py' or 'src/*'. '**' means any depth.
            path: The folder to search in, relative to the workspace.

        Returns:
            One path per line, at most 200, or a message that nothing matched.
        """
        limit = 200
        files = self._files(path, pattern)
        if not files:
            return f"No files match {pattern!r} in {path!r}."
        lines = [self._rel(file) for file in files[:limit]]
        if len(files) > limit:
            lines.append(f"[... {len(files) - limit} more. Use a narrower pattern ...]")
        return "\n".join(lines)

    def grep(
        self, pattern: str, path: str = ".", file_pattern: str = "**/*", ignore_case: bool = False
    ) -> str:
        """Search file contents with a regular expression.

        Args:
            pattern: A Python regular expression, matched against each line.
            path: The folder to search in, relative to the workspace.
            file_pattern: Only search files matching this glob, like '**/*.py'.
            ignore_case: Match upper and lower case alike.

        Returns:
            Matches as 'file:line: text', at most 100, or a message that none matched.
            Binary and very large files are skipped.
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

        Args:
            path: The file, relative to the workspace.
            offset: The first line to read, starting at 1.
            limit: How many lines to read. Defaults to 200.

        Returns:
            The numbered lines. If the file has more, a last line says which offset to
            use for the next page.
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

        Args:
            path: The file, relative to the workspace.
            content: The complete new content of the file.

        Returns:
            Whether the file was created or replaced, and its line count.
        """
        target = self._writable(path)
        if target.is_dir():
            raise ToolError(f"{path!r} is a folder.")
        target.parent.mkdir(parents=True, exist_ok=True)
        existed = target.exists()
        with target.open("w", encoding="utf-8", newline="") as file:
            file.write(content)
        lines = len(content.splitlines())
        return f"{'Replaced' if existed else 'Created'} {self._rel(target)} ({lines} lines)."

    def edit_file(self, path: str, old_text: str, new_text: str, replace_all: bool = False) -> str:
        """Replace an exact piece of text in a file with new text.

        Read the file first, and include a few surrounding lines if the text you want
        to change is not unique.

        Args:
            path: The file, relative to the workspace.
            old_text: The exact text to replace, including indentation. It must appear
                exactly once, unless replace_all is true.
            new_text: The text to put in its place.
            replace_all: Replace every occurrence instead of requiring exactly one.

        Returns:
            How many places changed, and the lines around the first change, numbered.
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
        with target.open("w", encoding="utf-8", newline="") as file:
            file.write(text.replace(old_text, new_text))

        # Show the edited region, so the model can check the result without a re-read.
        before = text[: text.index(old_text)]
        first = before.count("\n") + 1
        shown = self.read_file(
            path, offset=max(first - 3, 1), limit=new_text.count("\n") + 7
        ).split("\n(lines")[0]
        times = f"{count} places" if count > 1 else "1 place"
        return f"Edited {self._rel(target)} in {times}. Around the first edit:\n{shown}"

    async def run_command(self, command: str, timeout_seconds: int = 60) -> str:
        """Run a shell command in the workspace folder.

        The model sees a different summary, set in registry(), because it names the
        shell. These Args are still its argument descriptions.

        Args:
            command: The command line to run.
            timeout_seconds: Stop the command after this many seconds (at most 600).

        Returns:
            "exit code N" and the combined stdout and stderr, clipped at both ends.
        """
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
