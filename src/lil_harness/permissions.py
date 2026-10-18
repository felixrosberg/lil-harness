"""Decide whether a tool call may run: allow it, ask a human, or refuse it.

Every tool has a risk level, set when it's registered:

- `read`: only looks at things (read_file, grep, calculate).
- `write`: changes files (write_file, edit_file).
- `execute`: runs arbitrary code (run_command). Also the default for tools that
  don't say, because an unlabelled tool should get the most caution, not the least.

A `PermissionPolicy` maps each level to `allow`, `ask` or `deny`, and adds rules for
shell commands:

- A command is allowed without asking only if it *exactly* matches an allowlisted
  command (after collapsing whitespace). No prefixes or wildcards, because
  `git status && del *.*` starts with `git status`.
- A command that matches a deny pattern is refused outright.
- Everything else falls back to the level's mode, which is usually `ask`.

A tool whose level is `deny` isn't offered to the model at all (see
`ToolRegistry.specs`), so the model doesn't waste turns on calls that can only fail.

None of this is a sandbox. It decides what the harness lets through; it can't stop a
command you approved from doing something you didn't expect. For "free rein, nobody
watching", use the `auto` preset inside a container that only holds the workspace.
"""

from __future__ import annotations

import asyncio
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, TypeAlias

from lil_harness.types import ToolArgs, ToolName

Risk = Literal["read", "write", "execute"]
Mode = Literal["allow", "ask", "deny"]
Answer = Literal["yes", "no", "always"]

Modes: TypeAlias = dict[Risk, Mode]
"""What to do at each risk level, e.g. {"read": "allow", "write": "ask", "execute": "ask"}."""

PRESETS: dict[str, Modes] = {
    # Strict in the sense that even reads needs permission.
    "strict": {"read": "ask", "write": "ask", "execute": "ask"},
    # Look, don't touch: write and execute tools aren't even offered.
    "readonly": {"read": "allow", "write": "deny", "execute": "deny"},
    # A human approves every change and every command.
    "default": {"read": "allow", "write": "ask", "execute": "ask"},
    # Nobody approves anything. Only for a sandbox you can throw away.
    "auto": {"read": "allow", "write": "allow", "execute": "allow"},
}


@dataclass
class Approval:
    """A human's answer to "may this call run?".

    Attributes:
        answer: "yes", "no", or "always" (yes, and don't ask again; see `check`).
        note: Optional message from the human to the model, e.g. "use pytest".
    """

    answer: Answer
    note: str = ""


class Approver(Protocol):
    """Anything that can be awaited like this asks a human about one tool call."""

    async def __call__(self, tool: ToolName, risk: Risk, args: ToolArgs) -> Approval:
        """Ask whether this call may run.

        Args:
            tool: The tool's name.
            risk: Its risk level.
            args: The validated arguments, e.g. {"command": "pytest"}.

        Returns:
            The human's answer.
        """
        ...


@dataclass
class Decision:
    """The policy's verdict on one call.

    Attributes:
        allowed: Whether the call may run.
        message: Why it was refused, worded for the model.
    """

    allowed: bool
    message: str = ""


def normalize_command(command: str) -> str:
    """Collapse runs of whitespace, so "git  status " matches "git status".

    Args:
        command: A command line.

    Returns:
        The command with single spaces and no leading or trailing whitespace.
    """
    return " ".join(command.split())


@dataclass
class PermissionPolicy:
    """Rules for which tool calls run, ask a human, or are refused.

    Usually made with `preset()` or `from_file()` rather than directly.

    Attributes:
        modes: Risk level -> "allow", "ask" or "deny".
        allow_commands: Commands that run without asking, matched exactly.
        deny_patterns: Regexes; a command matching any of them is always refused.
        approver: Who answers the "ask"s. None means every "ask" is refused.
        always_tools: Tools the human answered "always" to, for this session.
    """

    modes: Modes = field(default_factory=lambda: dict(PRESETS["default"]))
    allow_commands: set[str] = field(default_factory=set)
    deny_patterns: list[str] = field(default_factory=list)
    approver: Approver | None = None
    always_tools: set[ToolName] = field(default_factory=set)

    def __post_init__(self) -> None:
        """Normalize the allowlist and set up the lock that keeps prompts one at a time."""
        self.allow_commands = {normalize_command(c) for c in self.allow_commands}
        # Tool calls run concurrently, but a human answers one question at a time.
        self._ask_lock = asyncio.Lock()

    @classmethod
    def preset(cls, name: str, **kwargs: Any) -> PermissionPolicy:
        """Make a policy from one of the PRESETS.

        Args:
            name: "strict", "readonly", "default" or "auto".
            **kwargs: Other fields, e.g. approver or allow_commands.

        Returns:
            A new policy.

        Raises:
            ValueError: If there's no preset with that name.
        """
        if name not in PRESETS:
            raise ValueError(f"Unknown preset {name!r}. Choose from: {', '.join(PRESETS)}")
        return cls(modes=dict(PRESETS[name]), **kwargs)

    @classmethod
    def from_file(cls, path: Path, *, preset: str | None = None, **kwargs: Any) -> PermissionPolicy:
        """Load a TOML policy file (see permissions.example.toml).

        Args:
            path: The TOML file.
            preset: Overrides the file's preset, e.g. from a --permissions flag. The
                file's mode overrides and command rules still apply on top.
            **kwargs: Other fields, e.g. approver.

        Returns:
            A new policy.

        Raises:
            ValueError: If the file has unknown keys, a bad mode, or a bad preset.
            re.error: If a deny pattern isn't a valid regex. Better at startup than on
                the first command.
        """
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        unknown = set(data) - {"preset", "modes", "commands"}
        if unknown:
            raise ValueError(f"{path}: unknown keys {', '.join(sorted(unknown))}")
        policy = cls.preset(preset or data.get("preset", "default"), **kwargs)
        for risk, mode in data.get("modes", {}).items():
            if risk not in PRESETS["default"] or mode not in ("allow", "ask", "deny"):
                raise ValueError(f"{path}: bad mode {risk} = {mode!r}")
            policy.modes[risk] = mode
        commands = data.get("commands", {})
        policy.allow_commands |= {normalize_command(c) for c in commands.get("allow", [])}
        policy.deny_patterns += list(commands.get("deny", []))
        for pattern in policy.deny_patterns:
            re.compile(pattern)
        return policy

    def offers(self, risk: Risk) -> bool:
        """Whether tools of this risk level should be offered to the model at all.

        Args:
            risk: A risk level.

        Returns:
            False if the level is "deny", since those calls could only fail.
        """
        return self.modes[risk] != "deny"

    async def check(self, tool: ToolName, risk: Risk, args: ToolArgs) -> Decision:
        """Decide whether one tool call may run, asking the approver if needed.

        The order: deny patterns, then the exact-match allowlist (for commands) or an
        earlier "always" (for other tools), then the risk level's mode.

        Args:
            tool: The tool's name.
            risk: Its risk level.
            args: The validated arguments. For execute tools, `args["command"]` is
                what the command rules look at.

        Returns:
            Allowed, or refused with a message for the model.
        """
        mode = self.modes[risk]
        command = args.get("command") if risk == "execute" else None

        if isinstance(command, str):
            for pattern in self.deny_patterns:
                if re.search(pattern, command):
                    return Decision(
                        False,
                        f"This command is blocked by the permission rules (it matches "
                        f"/{pattern}/). Don't retry it; find another way or ask the user.",
                    )
            if normalize_command(command) in self.allow_commands:
                return Decision(True)
        elif tool in self.always_tools:
            return Decision(True)

        if mode == "allow":
            return Decision(True)
        if mode == "deny":
            return Decision(False, f"{tool} is not allowed in this session.")
        if self.approver is None:
            return Decision(
                False,
                f"{tool} needs a human's approval, and nobody is available to approve it "
                "in this run. Do without it, or explain in your answer what you would do.",
            )

        async with self._ask_lock:
            approval = await self.approver(tool, risk, args)
        if approval.answer == "always":
            # For commands, "always" means this exact command, not every command.
            if isinstance(command, str):
                self.allow_commands.add(normalize_command(command))
            else:
                self.always_tools.add(tool)
        if approval.answer in ("yes", "always"):
            return Decision(True)
        message = "The user declined this call."
        if approval.note:
            message += f" Their note: {approval.note}"
        return Decision(False, message + " Don't repeat the same call; adjust your approach.")


def describe_call(tool: ToolName, args: ToolArgs, max_lines: int = 20) -> str:
    """What the human sees before approving: the command, or a small diff of the change.

    Args:
        tool: The tool's name.
        args: The call's validated arguments.
        max_lines: Cut the description after this many lines.

    Returns:
        "$ command" for commands, "- old / + new" lines for edits, the start of the
        content for writes, and name(args) for anything else.
    """
    if "command" in args:
        return f"$ {args['command']}"
    if "old_text" in args and "new_text" in args:
        lines = [f"edit {args.get('path')}:"]
        lines += [f"- {line}" for line in str(args["old_text"]).splitlines()]
        lines += [f"+ {line}" for line in str(args["new_text"]).splitlines()]
    elif "content" in args:
        content = str(args["content"]).splitlines()
        lines = [f"write {args.get('path')} ({len(content)} lines):"]
        lines += [f"  {line}" for line in content]
    else:
        lines = [f"{tool}({', '.join(f'{k}={v!r}' for k, v in args.items())})"]
    if len(lines) > max_lines:
        lines = lines[:max_lines] + [f"  ... {len(lines) - max_lines} more lines"]
    return "\n".join(lines)


async def terminal_approver(tool: ToolName, risk: Risk, args: ToolArgs) -> Approval:
    """Ask in the terminal: [y]es, [n]o (with an optional note for the model), [a]lways.

    Pressing Enter means no, so a stray keypress can't approve anything.

    Args:
        tool: The tool's name.
        risk: Its risk level, shown in the question.
        args: The call's validated arguments, shown with `describe_call`.

    Returns:
        The human's answer.
    """
    import click

    def ask() -> Approval:
        """The blocking part: print the question and read the answer."""
        click.echo()
        click.secho(f"The model wants to {risk} with {tool}:", fg="yellow", bold=True)
        click.echo(describe_call(tool, args))
        always = "this exact command" if "command" in args else f"{tool}, for this session"
        choice = click.prompt(
            f"Allow? [y]es / [n]o / [a]lways ({always})",
            type=click.Choice(["y", "n", "a"]),
            show_choices=False,
            default="n",
        )
        if choice == "n":
            note = click.prompt("Tell the model why (optional)", default="", show_default=False)
            return Approval("no", note)
        return Approval("always" if choice == "a" else "yes")

    # click.prompt blocks, so run it in a thread and keep the event loop free.
    return await asyncio.to_thread(ask)
