# Docker sandbox demo

lil-harness is a library: in your own project you install it, write a short script that sets up the tools and the agent, and run that in a container you control. This folder is one example of that, used for testing and showing what the harness can do. It's not part of the library.

```
┌──────────── internal network "sandbox" (no internet) ────────────┐
│                                                                  │
│   agent  ──── HTTP /v1/chat/completions ────►  ollama (GPU)  ────┼──► internet
│   run_task.py                                                    │    (for ollama pull)
│   /workspace  ← ./workspace (rw)                                 │
│   /output     ← ./output    (rw)                                 │
└──────────────────────────────────────────────────────────────────┘
```

The agent runs with the `auto` permission preset: there's nobody to approve anything, and the container is what keeps it in. It can do anything inside `/workspace`, but:

- **No internet.** The `sandbox` network is `internal`, so the only thing the agent can reach is the model server. A `pip install` or `curl` from the agent fails.
- **Nothing outside two folders.** Only `./workspace` and `./output` are mounted. The rest of the filesystem is read-only.
- **No privileges.** It runs as an ordinary user, with all Linux capabilities dropped and no way to gain new ones.
- **Limits.** 2 GB of memory, 2 CPUs and 256 processes, so a runaway loop can't take your machine down.

## Run it (Windows)

You need [Docker Desktop](https://docs.docker.com/desktop/) with the WSL2 backend, and an NVIDIA driver recent enough for GPU support in WSL2.

```powershell
cd examples\docker
mkdir workspace, output
docker compose up -d ollama
docker compose exec ollama ollama pull qwen3.5:4b
docker compose run --rm agent
```

The model is downloaded again into the container's own volume, since it can't see the models your Windows Ollama already has. It's stored in a volume, so this happens once.

The last command builds the image the first time, copies `examples/sample_project` (the `mean()` bug from Step 5) into `workspace\`, runs the agent, then runs `python test_stats.py` as a check. You'll see every tool call. Afterwards:

- `workspace\stats.py` holds the agent's fix.
- `output\run.json` records the task, model, stop reason, steps, time, the check's result and the whole conversation.
- The exit code is 1 if the check failed. That's what an eval runner or CI job would look at.

To run it again from scratch, empty `workspace\` (`Remove-Item -Recurse workspace\*`). If the workspace isn't empty, the sample isn't copied and the agent carries on with what's there.

On Linux, create `workspace` and `output` yourself before the first run, as above. Otherwise Docker creates them owned by root and the agent can't write to them.

## Your own task

Anything after `agent` replaces the demo's arguments:

```powershell
docker compose run --rm agent "Add type hints to every function in stats.py"
docker compose run --rm agent --check "python -m pytest -q" "Make the tests pass"
$env:LIL_MODEL="qwen3:8b"; docker compose run --rm agent "..."
```

To work on your own project, change the `./workspace` mount in `compose.yaml` to point at a copy of it, not the original: with `auto`, the agent changes files without asking.

## The files

| File | What it is |
|---|---|
| `run_task.py` | The script that runs in the container. It copies the seed into the workspace if it's empty, builds the coding tools with the `auto` preset, runs the agent, runs the check, and writes `run.json`. The model comes from `LIL_BASE_URL` and `LIL_MODEL`, so the same image works with Ollama, vLLM or a hosted API. |
| `Dockerfile` | Python 3.12, git, lil-harness and pytest, running as user `agent`. In your own project, the `pip install` line is the one you change. |
| `Dockerfile.dockerignore` | Sends only what the Dockerfile copies into the build, so builds are quick and nothing private ends up in the image. |
| `compose.yaml` | The two services, the two networks and the locks described above. Every setting has a comment. |

## What this doesn't protect against

A container shares your machine's kernel, so it's a strong fence, not a perfect one. A kernel exploit could get out. For a model you run on your own tasks that's a reasonable trade. For untrusted code from the internet, people use a VM, or gVisor (`runtime: runsc`) as a stricter runtime.

The agent can also send anything in `/workspace` to the model server. That's fine for your own Ollama, but think about it before pointing `LIL_BASE_URL` at a hosted API.
