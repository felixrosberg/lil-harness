"""A scripted backend for tests: no model, no GPU, fully deterministic."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

from lil_harness.tools.registry import ToolSpec
from lil_harness.types import AssistantTurn, Message


@dataclass
class FakeCall:
    """What the fake backend received on one call, for assertions in tests.

    Attributes:
        messages: A deep copy of the history it was sent.
        tool_names: The names of the tools it was offered.
    """

    messages: list[Message]
    tool_names: list[str]


@dataclass
class FakeBackend:
    """Returns the scripted turns in order, and records every request it gets.

    Attributes:
        script: The turns to return, one per call.
        calls: Filled in as it's called: what each call received.
    """

    script: list[AssistantTurn]
    calls: list[FakeCall] = field(default_factory=list)

    async def complete(self, messages: list[Message], tools: list[ToolSpec]) -> AssistantTurn:
        """Record the request and return the next scripted turn.

        Args:
            messages: The conversation so far.
            tools: The tools on offer.

        Returns:
            The next turn from `script`.

        Raises:
            RuntimeError: If the model is called more times than the script has turns.
        """
        # Deep copy so later changes to the history don't rewrite what we recorded.
        self.calls.append(FakeCall(copy.deepcopy(messages), [tool.name for tool in tools]))
        if len(self.calls) > len(self.script):
            raise RuntimeError(f"FakeBackend script ran out after {len(self.script)} turns")
        return self.script[len(self.calls) - 1]
