# lil-harness

A small agent harness library for running tool-using agents on local, open-weight LLMs. Made primarily for my own use cases, for fun, and get slightly more familiar with setting up harnesses.

You write tools as plain Python functions, choose what the agent is allowed to do, point it at a model server (Ollama, vLLM, or any OpenAI-compatible API), and let it work through a task on its own. It's built for small models on a single consumer GPU, so it keeps tool outputs and images small, and it reports errors back to the model so the model can correct itself.

## What it's for

- **Small automated coding tasks.** Fix a failing test, make a refactor, write docs, all inside a sandboxed container.
- **Questions over your own documents and code.** Search with RAG, answer with `file:line` citations.
- **Batch jobs.** Run the same task on many inputs, each in a fresh conversation, and keep a JSON record of every run.
- **Labelling data without a human in the loop.** Look at images or video, extract fields, and write labels to a file.

## What's included

| Module | What it gives you |
| --- | --- |
| `tools.registry` | `ToolRegistry`: turns functions into tools. Builds the JSON schema from type hints and docstrings, validates arguments, and returns errors to the model. |
| `permissions` | `PermissionPolicy`: allow, ask or deny each call by risk level (`read`, `write`, `execute`). Presets are `readonly`, `default` (asks a human), `strict` and `auto`. |
| `agent.loop` | `run_agent`: calls the model, runs the tools it asks for (several at once), and repeats until it gives an answer. |
| `backends` | Ollama, vLLM and any OpenAI-compatible server, plus a scripted `FakeBackend` for tests. |
| `tools.coding` | `CodingTools`: list, grep, read, write and edit files in one folder, and run commands. |
| `tools.media` | `MediaTools`: view images, zoom in, read video info, see an overview grid of frames, and look at single frames. |
| `rag` | Chunking, embeddings, hybrid keyword and semantic search, optional reranking, and a `search_docs` tool. |
| `evals` | YAML test cases for tool calling, and the `lil-eval` and `lil-rag-eval` commands. |

## Install

You need Python 3.11+ and [uv](https://docs.astral.sh/uv/).

To use it in your own project:

```bash
uv add "lil-harness[media] @ git+https://github.com/felixrosberg/lil-harness"
```

To work on lil-harness itself:

```bash
git clone https://github.com/felixrosberg/lil-harness
cd lil-harness
uv sync --extra media
uv run pytest
```

The optional extras are:
- `media`: Pillow and PyAV, for the image and video tools.
- `rerank`: sentence-transformers, for the cross-encoder reranker.

## Serve a model

Any server with an OpenAI-compatible `/v1/chat/completions` endpoint and tool calling works.

**Ollama** is the easiest option, and it runs natively on Windows, macOS and Linux:

```bash
ollama pull qwen3.5:4b
ollama serve
```

Use a model that lists `tools` under Capabilities in `ollama show <model>`. For images it also needs `vision`. Ollama's default context is short, so set `OLLAMA_CONTEXT_LENGTH=8192` before `ollama serve`.

**vLLM** runs on Linux or WSL2, or in Docker:

```bash
docker run --rm --gpus all -p 8000:8000 --shm-size 4gb \
  -v hf-cache:/root/.cache/huggingface \
  vllm/vllm-openai:v0.20.2 Qwen/Qwen3-4B-AWQ \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --dtype half --max-model-len 8192 --gpu-memory-utilization 0.85
```

`scripts/serve_vllm.sh` starts the same server without Docker, and it explains each flag.

To point the harness at a server, set:

```bash
LIL_BASE_URL=http://localhost:11434/v1   # Ollama; vLLM is http://localhost:8000/v1
LIL_MODEL=qwen3.5:4b
LIL_API_KEY=...                          # only for hosted APIs
```

## Example: your own tools

A tool is a function with type hints and a docstring. The docstring's summary becomes the tool's description, and each `Args:` entry becomes that argument's description. The model sees exactly that, so write them for the model.

```python
import json

from lil_harness.permissions import PermissionPolicy
from lil_harness.tools.registry import ToolError, ToolRegistry

PRICES = {"A-1042": 19.99, "B-2001": 4.50}

registry = ToolRegistry(PermissionPolicy.preset("default"))


@registry.tool(risk="read")
def lookup_price(sku: str) -> dict:
    """Look up a product's price in euros.

    Args:
        sku: The product code, like "A-1042".
    """
    if sku not in PRICES:
        # A ToolError goes back to the model as advice, so it can try again.
        raise ToolError(f"No product {sku!r}. Known codes: {', '.join(PRICES)}.")
    return {"sku": sku, "price": PRICES[sku]}


@registry.tool(risk="write")
def save_result(name: str, data: dict) -> str:
    """Save a result as a JSON file in results/.

    Args:
        name: The file name, without .json.
        data: What to save.
    """
    with open(f"results/{name}.json", "w", encoding="utf-8") as out:
        json.dump(data, out, indent=2)
    return f"Saved results/{name}.json"
```

The `risk` label tells the policy how careful to be. `readonly` doesn't even offer `write` and `execute` tools to the model. `default` asks a human before each one, through `terminal_approver` or your own approver function. `auto` runs everything, so use it only somewhere nothing important can break, such as a container or a copy of the data.

You can add the built-in toolsets to the same registry, or start from one of them:

```python
from lil_harness.tools.coding import CodingTools
from lil_harness.tools.media import MediaTools

registry = CodingTools("workspace").registry(PermissionPolicy.preset("auto"))
MediaTools("workspace").register(registry)
registry.tool(lookup_price, risk="read")
```

A tool can also show the model images. To do that, return an `Image` or a `ToolOutput(text, images)` from `lil_harness.types`.

## Example: run a task

```python
import asyncio

from lil_harness.agent.loop import run_agent
from lil_harness.backends.openai_compat import OpenAICompatBackend
from lil_harness.types import Message

backend = OpenAICompatBackend.from_env(max_images=3)  # reads LIL_BASE_URL and LIL_MODEL
messages = [
    Message.system("You answer pricing questions. Look prices up; never guess."),
    Message.user("What do A-1042 and B-2001 cost together? Save the answer as 'total'."),
]
result = asyncio.run(run_agent(backend, registry, messages, max_steps=20))

print(result.stop_reason, result.steps)  # "final_answer" or "max_steps"
print(result.answer)
```

`result.messages` holds the whole conversation, including every tool call and its result.

## Example: label a folder of images

For batch work, give each item its own short conversation. The context stays small, and one bad item can't derail the rest.

```python
import asyncio
import json
from pathlib import Path

from lil_harness.agent.loop import run_agent
from lil_harness.backends.openai_compat import OpenAICompatBackend
from lil_harness.tools.media import MediaTools, load_image
from lil_harness.types import Message

PROMPT = 'Label {file}. Reply with only JSON: {{"animal": "cat" | "dog" | "other"}}'


async def main(folder: Path) -> None:
    backend = OpenAICompatBackend.from_env(max_images=3)
    media = MediaTools(folder)  # view_image, zoom_image, video_overview, ...
    with open(folder / "labels.jsonl", "w", encoding="utf-8") as out:
        for photo in sorted(folder.glob("*.jpg")):
            messages = [
                Message.system("You label images. Zoom in if a detail is too small."),
                Message.user(PROMPT.format(file=photo.name), images=[load_image(photo)]),
            ]
            result = await run_agent(backend, media.registry(), messages, max_steps=5)
            out.write(json.dumps({"file": photo.name, "answer": result.answer}) + "\n")


asyncio.run(main(Path("photos")))
```

`examples/step7_media.py` is a complete version of this, with JSON parsing and scoring against known answers.

## Deploy a task

The idea is to package your script and its tools in an image, run the model server next to it, and give the agent only the folders it should touch.

`Dockerfile`:

```dockerfile
FROM python:3.12-slim
RUN pip install --no-cache-dir \
    "lil-harness[media] @ https://github.com/felixrosberg/lil-harness/archive/refs/heads/main.zip"
COPY my_task.py /app/my_task.py
RUN useradd --create-home agent
USER agent
ENTRYPOINT ["python", "/app/my_task.py"]
```

`compose.yaml`:

```yaml
services:
  ollama:
    image: ollama/ollama
    volumes:
      - ollama-models:/root/.ollama
    environment:
      OLLAMA_CONTEXT_LENGTH: "8192"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

  agent:
    build: .
    environment:
      LIL_BASE_URL: http://ollama:11434/v1
      LIL_MODEL: qwen3.5:4b
    volumes:
      - ./data:/data        # the only folder the agent can change
    read_only: true
    cap_drop: [ALL]

volumes:
  ollama-models:
```

Then start the server, pull the model once, and run the task:

```bash
docker compose up -d ollama
docker compose exec ollama ollama pull qwen3.5:4b
docker compose run --rm agent
```

`examples/docker/` is a fuller version. It has no internet access for the agent, resource limits, a check command that verifies the result, a JSON record of every run, and a vLLM option. See [examples/docker/README.md](examples/docker/README.md).

## More examples

| File | What it shows |
| --- | --- |
| `examples/step1_fake.py` | The agent loop with a scripted model; no GPU needed. |
| `examples/step2_hosted.py` | A hosted OpenAI-compatible API. |
| `examples/step3_local.py` | A local model on Ollama or vLLM. |
| `examples/step5_coding.py` | The coding tools fixing a bug, asking before each change. |
| `examples/step6_rag.py` | Questions over a folder, answered with search and citations. |
| `examples/step7_media.py` | Labelling images and asking about a video. |
| `examples/docker/` | A sandboxed deployment with a model server. |

## Evaluate

```bash
uv run lil-eval                # tool-calling cases in evals/cases/
uv run lil-rag-eval -v         # retrieval quality: recall@5, MRR, recall@20
```

Use these to compare models, prompts and settings on your own cases before trusting a setup to run unattended.

## License

MIT
