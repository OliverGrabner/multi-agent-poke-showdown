#!/usr/bin/env bash
# One run as one Slurm job: start the model server on this node, wait until it answers, check the
# connection, play the battles, then stop the server.
# Submit with: scripts/hprc/submit.sh MODEL [pokerl llm arguments...]
# Example:     scripts/hprc/submit.sh qwen3.8-27b --side-a qwen3.8-27b --side-b maxpower \
#                  --mode-a free --label baseline-free --pairs 50 --workers 24
set -euo pipefail
umask 077
: "${SLURM_JOB_ID:?Submit this script through Slurm: scripts/hprc/submit.sh}"
cd "${SLURM_SUBMIT_DIR:?}"
source scripts/hprc/common.sh
printf '%s job started on %s\n' "$(date +%T)" "$(hostname)"
model=${1:?Give the model to serve, e.g. qwen3.8-27b}
shift

case "$model" in
  qwen3.8-27b)
    repo=Qwen/Qwen3.8-27B
    url_var=QWEN_BASE_URL
    parsers=(--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder)
    ;;
  gemma-4-31b)
    repo=google/gemma-4-31B-it
    url_var=GEMMA_BASE_URL
    parsers=(--reasoning-parser gemma4 --enable-auto-tool-choice --tool-call-parser gemma4)
    ;;
  *)
    printf 'Unknown model %s\n' "$model" >&2
    exit 2
    ;;
esac

# FlashInfer compiles kernels during warm-up and needs a CUDA toolchain (as in the pruning project).
module load GCC/13.3.0 CUDA/13.0.0
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
export HF_HUB_OFFLINE=1  # weights were downloaded by setup.sh

# Unpack the vLLM environment to local disk: importing it straight from scratch took ~30 minutes.
local_env="${TMPDIR:-/tmp}/pokerl-vllm-${SLURM_JOB_ID}"
mkdir -p "$local_env"
tar -xf "$VLLM_ARCHIVE" -C "$local_env"
vllm_python="$local_env/.venv-vllm/bin/python"
export PATH="$local_env/.venv-vllm/bin:$PATH"
printf '%s vLLM environment unpacked to %s\n' "$(date +%T)" "$local_env"

# A private key for this job's server; it never leaves the node.
export VLLM_API_KEY="$(.venv/bin/python -c 'import secrets; print(secrets.token_urlsafe(32))')"
port=8000
gpus=$(nvidia-smi --list-gpus | wc -l)
nvidia-smi > "runs/vllm-${SLURM_JOB_ID}-gpus.txt"

"$vllm_python" -m vllm.entrypoints.openai.api_server \
  --model "$repo" --served-model-name "$repo" \
  --host 127.0.0.1 --port "$port" \
  --dtype bfloat16 --tensor-parallel-size "$gpus" \
  --max-model-len 131072 --max-num-seqs 64 --gpu-memory-utilization 0.95 \
  --language-model-only --generation-config vllm "${parsers[@]}" \
  > "runs/vllm-${SLURM_JOB_ID}.log" 2>&1 &
server=$!
trap 'kill "$server" 2>/dev/null || true; rm -rf "$local_env"' EXIT

# Wait up to 90 minutes for the server (loading weights and compiling kernels takes a while).
ready=false
for _ in $(seq 540); do
  if curl -fs -o /dev/null -H "Authorization: Bearer $VLLM_API_KEY" "http://127.0.0.1:$port/v1/models"; then
    ready=true
    break
  fi
  if ! kill -0 "$server" 2>/dev/null; then
    printf 'The model server stopped; see runs/vllm-%s.log\n' "$SLURM_JOB_ID" >&2
    exit 1
  fi
  sleep 10
done
if [[ $ready != true ]]; then
  printf 'The model server was not ready after 90 minutes; see runs/vllm-%s.log\n' "$SLURM_JOB_ID" >&2
  exit 1
fi
printf '%s model server ready\n' "$(date +%T)"
export "$url_var=http://127.0.0.1:$port/v1"

.venv/bin/pokerl model-check --model "$model"
if [[ $# -gt 0 ]]; then
  .venv/bin/pokerl llm "$@"
fi
