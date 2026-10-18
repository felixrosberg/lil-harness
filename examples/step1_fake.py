"""Step 1 demo: one full tool-use round trip with a scripted fake model.

Run with: uv run python examples/step1_fake.py
"""

import asyncio
from typing import Literal

from lil_harness.agent.loop import run_agent
from lil_harness.backends.fake import FakeBackend
from lil_harness.tools.registry import ToolRegistry
from lil_harness.types import AssistantTurn, Message, ToolCall

registry = ToolRegistry()


@registry.tool
def calculate(
    a: float, b: float, op: Literal["add", "subtract", "multiply", "divide"]
) -> float | None:
    """Do basic arithmetic on two numbers. Use this instead of doing math in your head.

    Args:
        a: The first number.
        b: The second number.
        op: Which operation to apply, as `a op b`.

    Returns:
        The result, or None when dividing by zero.
    """
    return {"add": a + b, "subtract": a - b, "multiply": a * b, "divide": a / b if b else None}[op]


async def main() -> None:
    """Print the tool's schema, then run one scripted round trip and print the history."""
    print("Schema the model would see:")
    for spec in registry.specs():
        print(f"  {spec.name}: {spec.parameters}\n")

    backend = FakeBackend(
        script=[
            AssistantTurn(
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="calculate",
                        arguments={"a": 17, "b": 25, "op": "multiply"},
                    )
                ]
            ),
            AssistantTurn(content="17 times 25 is 425."),
        ]
    )
    result = await run_agent(backend, registry, [Message.user("What is 17 * 25?")])

    for m in result.messages:
        if m.tool_calls:
            for c in m.tool_calls:
                print(f"[{m.role}] calls {c.name}({c.arguments})")
        else:
            print(f"[{m.role}] {m.content}")
    print(f"\nstop_reason={result.stop_reason} steps={result.steps}")


if __name__ == "__main__":
    asyncio.run(main())
