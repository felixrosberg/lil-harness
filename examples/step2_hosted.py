"""Step 2 demo: the same agent loop, now talking to a real model.

Set LIL_BASE_URL, LIL_MODEL and LIL_API_KEY (see the README), then run:
    uv run python examples/step2_hosted.py "What is 1234 * 5678, and what time is it in UTC?"
"""

import asyncio
import sys

from lil_harness.agent.loop import run_agent
from lil_harness.backends.openai_compat import OpenAICompatBackend
from lil_harness.tools.builtin import default_registry
from lil_harness.types import Message

registry = default_registry()


async def main() -> None:
    """Ask the question from the command line to the model in LIL_MODEL, and print the run."""
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

    for message in result.messages[1:]:
        if message.tool_calls:
            for tool_call in message.tool_calls:
                # raw_arguments is exactly what the model emitted.
                print(f"[{message.role}] calls {tool_call.name}({tool_call.raw_arguments})")
        else:
            print(f"[{message.role}] {message.content}")
    print(f"\nstop_reason={result.stop_reason} steps={result.steps}")


if __name__ == "__main__":
    asyncio.run(main())
