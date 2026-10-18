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
from typing import Any, Literal, Protocol

Risk = Literal["read", "write", "execute"]
Mode = Literal["allow", "ask", "deny"]
Answer = Literal["yes", "no", "always"]

PRESETS: dict[str, dict[Risk, Mode]] = {
    # Look, don't touch: write and execute tools aren't even offered.
    "readonly": {"read": "allow", "write": "deny", "execute": "deny"},
    # A human approves every change and every command.
    "default": {"read": "allow", "write": "ask", "execute": "ask"},
    # Nobody approves anything. Only for a sandbox you can throw away.
    "auto": {"read": "allow", "write": "allow", "execute": "allow"},
}


@dataclass
class Approval:
    answer: Answer
    note: str = ""  # optional message from the human to the model, e.g. "use pytest"


class Approver(Protocol):
    """Asks a human about one tool call. `args` are the validated arguments."""

    async def __call__(self, tool: str, risk: Risk, args: dict[str, Any]) -> Approval: ...


@dataclass
class Decision:
    allowed: bool
    message: str = ""  # why it was refused, worded for the model


def normalize_command(command: str) -> str:
    return " ".join(command.split())


@dataclass
class PermissionPolicy:
    modes: dict[Risk, Mode] = field(default_factory=lambda: dict(PRESETS["default"]))
    allow_commands: set[str] = field(default_factory=set)
    deny_patterns: list[str] = field(default_factory=list)
    approver: Approver | None = None
    # Tools the human answered "always" to, for this session.
    always_tools: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.allow_commands = {normalize_command(c) for c in self.allow_commands}
        # Tool calls run concurrently, but a human answers one question at a time.
        self._ask_lock = asyncio.Lock()

    @classmethod
    def preset(cls, name: str, **kwargs: Any) -> PermissionPolicy:
        if name not in PRESETS:
            raise ValueError(f"Unknown preset {name!r}. Choose from: {', '.join(PRESETS)}")
        return cls(modes=dict(PRESETS[name]), **kwargs)

    @classmethod
    def from_file(cls, path: Path, *, preset: str | None = None, **kwargs: Any) -> PermissionPolicy:
        """Load a TOML policy file (see permissions.example.toml). `preset` overrides the file's."""
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
            re.compile(pattern)  # fail at startup, not on the first command
        return policy

    def offers(self, risk: Risk) -> bool:
        return self.modes[risk] != "deny"

    async def check(self, tool: str, risk: Risk, args: dict[str, Any]) -> Decision:
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


def describe_call(tool: str, args: dict[str, Any], max_lines: int = 20) -> str:
    """What the human sees before approving: the command, or a small diff of the change."""
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


async def terminal_approver(tool: str, risk: Risk, args: dict[str, Any]) -> Approval:
    """Ask in the terminal: [y]es, [n]o (with an optional note for the model), [a]lways."""
    import click

    def ask() -> Approval:
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
