"""The harness's own tools, shared by the examples and the eval suite.

Add new tools here (or in a new module) and register them in `default_registry`, so
every example and eval can use them.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from lil_harness.tools.registry import ToolRegistry


def calculate(a: float, b: float, op: Literal["add", "subtract", "multiply", "divide"]) -> float:
    """Do basic arithmetic on two numbers. Use this instead of doing math in your head.

    Args:
        a: The first number.
        b: The second number.
        op: Which operation to apply, as `a op b`.

    Returns:
        The result. Dividing by zero raises ZeroDivisionError, which the registry
        turns into an error message for the model.
    """
    return {"add": a + b, "subtract": a - b, "multiply": a * b, "divide": a / b}[op]


def current_time_utc() -> str:
    """Get the current date and time in UTC, in ISO 8601 format.

    Returns:
        A timestamp like "2026-10-18T12:00:00+00:00".
    """
    return datetime.now(UTC).isoformat(timespec="seconds")


ALL_TOOLS = [calculate, current_time_utc]


def default_registry(names: list[str] | None = None) -> ToolRegistry:
    """A registry with the built-in tools, or only the ones named.

    Args:
        names: Tool names to include. None includes them all.

    Returns:
        A new registry with no permission policy, since these tools only read.

    Raises:
        ValueError: If a name isn't one of the built-in tools.
    """
    registry = ToolRegistry()
    for tool in ALL_TOOLS:
        if names is None or tool.__name__ in names:
            registry.tool(tool, risk="read")
    if names is not None:
        unknown = set(names) - {tool.__name__ for tool in ALL_TOOLS}
        if unknown:
            raise ValueError(f"Unknown tools: {', '.join(sorted(unknown))}")
    return registry
