"""A scripted backend for tests: no model, no GPU, fully deterministic."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

from lil_harness.tools.registry import ToolSpec
from lil_harness.types import AssistantTurn, Message


@dataclass
class FakeCall:
    """What the fake backend received on one call, for assertions in tests."""

    messages: list[Message]
    tool_names: list[str]


@dataclass
class FakeBackend:
    """Returns the scripted turns in order, and records every request it gets."""

    script: list[AssistantTurn]
    calls: list[FakeCall] = field(default_factory=list)

    async def complete(self, messages: list[Message], tools: list[ToolSpec]) -> AssistantTurn:
        # Deep copy so later changes to the history don't rewrite what we recorded.
        self.calls.append(FakeCall(copy.deepcopy(messages), [t.name for t in tools]))
        if len(self.calls) > len(self.script):
            raise RuntimeError(f"FakeBackend script ran out after {len(self.script)} turns")
        return self.script[len(self.calls) - 1]
