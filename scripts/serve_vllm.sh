#!/usr/bin/env bash
# Start a vLLM server with tool calling turned on. Run this inside WSL2 (or Linux),
# with the vLLM environment activated:
#
#   source ~/vllm-env/bin/activate
#   bash scripts/serve_vllm.sh qwen3-4b
#
# Each profile below is tuned for an 8 GB Turing card (RTX 2080 Super). Extra arguments
# are passed straight to `vllm serve`, e.g. `bash scripts/serve_vllm.sh qwen3-4b --enforce-eager`.
set -euo pipefail

profile="${1:-qwen3-4b}"
shift || true

case "$profile" in
  qwen3-4b)
    # 4-bit AWQ weights are about 2.5 GB, which leaves room for the KV cache.
    model="Qwen/Qwen3-4B-AWQ"
    ;;
  qwen3-1.7b)
    # Unquantized and tiny: a fallback if the 4B model doesn't fit or AWQ misbehaves.
    model="Qwen/Qwen3-1.7B"
    ;;
  *)
    echo "Unknown profile '$profile'. Choose: qwen3-4b, qwen3-1.7b" >&2
    exit 1
    ;;
esac

# What each flag is for:
#   --enable-auto-tool-choice  let the model decide when to call tools
#   --tool-call-parser hermes  Qwen writes tool calls as <tool_call>{json}</tool_call>;
#                              this parser turns them into OpenAI-style tool_calls
#   --reasoning-parser qwen3   if thinking is on, move <think>...</think> out of the answer
#   --dtype half               Turing GPUs have no bfloat16, so use float16
#   --max-model-len 8192       short context so the KV cache fits in 8 GB
#   --gpu-memory-utilization   below 0.9 because Windows also uses this GPU for the display
exec vllm serve "$model" \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --reasoning-parser qwen3 \
  --dtype half \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --host 0.0.0.0 \
  --port 8000 \
  "$@"
