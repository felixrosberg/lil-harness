"""The harness's own tools, shared by the examples and the eval suite.

Add new tools here (or in a new module) and register them in `default_registry`, so
every example and eval can use them.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from lil_harness.tools.registry import ToolRegistry


def calculate(a: float, b: float, op: Literal["add", "subtract", "multiply", "divide"]):
    """Do basic arithmetic on two numbers. Use this instead of doing math in your head."""
    return {"add": a + b, "subtract": a - b, "multiply": a * b, "divide": a / b}[op]


def current_time_utc() -> str:
    """Get the current date and time in UTC, in ISO 8601 format."""
    return datetime.now(UTC).isoformat(timespec="seconds")


ALL_TOOLS = [calculate, current_time_utc]


def default_registry(names: list[str] | None = None) -> ToolRegistry:
    """A registry with the built-in tools, or only the ones named."""
    registry = ToolRegistry()
    for fn in ALL_TOOLS:
        if names is None or fn.__name__ in names:
            registry.tool(fn)
    if names is not None:
        unknown = set(names) - {fn.__name__ for fn in ALL_TOOLS}
        if unknown:
            raise ValueError(f"Unknown tools: {', '.join(sorted(unknown))}")
    return registry
