"""Step 2 demo: the same agent loop, now talking to a real model.

Set LIL_BASE_URL, LIL_MODEL and LIL_API_KEY (see the README), then run:
    uv run python examples/step2_hosted.py "What is 1234 * 5678, and what time is it in UTC?"
"""

import asyncio
import sys
from datetime import UTC, datetime
from typing import Literal

from lil_harness.agent.loop import run_agent
from lil_harness.backends.openai_compat import OpenAICompatBackend
from lil_harness.tools.registry import ToolRegistry
from lil_harness.types import Message

registry = ToolRegistry()


@registry.tool
def calculate(a: float, b: float, op: Literal["add", "subtract", "multiply", "divide"]):
    """Do basic arithmetic on two numbers. Use this instead of doing math in your head."""
    return {"add": a + b, "subtract": a - b, "multiply": a * b, "divide": a / b}[op]


@registry.tool
def current_time_utc() -> str:
    """Get the current date and time in UTC, in ISO 8601 format."""
    return datetime.now(UTC).isoformat(timespec="seconds")


async def main() -> None:
    question = " ".join(sys.argv[1:]) or "What is 1234 * 5678, and what time is it in UTC?"
    backend = OpenAICompatBackend.from_env()
    print(f"model: {backend.model}\n")

    result = await run_agent(
        backend,
        registry,
        [
            Message.system("You are a helpful assistant. Use the tools when they help."),
            Message.user(question),
        ],
    )

    for m in result.messages[1:]:
        if m.tool_calls:
            for c in m.tool_calls:
                # raw_arguments is exactly what the model emitted.
                print(f"[{m.role}] calls {c.name}({c.raw_arguments})")
        else:
            print(f"[{m.role}] {m.content}")
    print(f"\nstop_reason={result.stop_reason} steps={result.steps}")


if __name__ == "__main__":
    asyncio.run(main())
