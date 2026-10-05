#!/usr/bin/env bash
# Serve one model with vLLM on this node until stopped. Started by run.sh, never on its own.
#   scripts/hprc/serve.sh MODEL HOST PORT
# HOST is 127.0.0.1 for a server on the battle node, 0.0.0.0 for one the battle node reaches over
# the cluster network (it still needs the job's VLLM_API_KEY).
set -euo pipefail
umask 077
cd "${SLURM_SUBMIT_DIR:?}"
source scripts/hprc/common.sh
model=${1:?model}
host=${2:?host}
port=${3:?port}
model_info "$model"

# FlashInfer compiles kernels during warm-up and needs a CUDA toolchain (as in the pruning project).
module load GCC/13.3.0 CUDA/13.0.0
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
export HF_HUB_OFFLINE=1  # weights were downloaded by setup.sh

# Unpack the vLLM environment to local disk: importing it straight from scratch took ~30 minutes.
local_env="${TMPDIR:-/tmp}/pokerl-vllm-${SLURM_JOB_ID}-${model}"
trap 'rm -rf "$local_env"' EXIT
mkdir -p "$local_env"
tar -xf "$VLLM_ARCHIVE" -C "$local_env"
export PATH="$local_env/.venv-vllm/bin:$PATH"
printf '%s %s: vLLM environment unpacked on %s\n' "$(date +%T)" "$model" "$(hostname)"

"$local_env/.venv-vllm/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$repo" --served-model-name "$repo" \
  --host "$host" --port "$port" \
  --dtype bfloat16 --tensor-parallel-size "$gpus" \
  --max-model-len 262144 --max-num-seqs 64 --gpu-memory-utilization 0.95 \
  --language-model-only --generation-config vllm "${parsers[@]}" \
  > "runs/vllm-${SLURM_JOB_ID}-${model}.log" 2>&1
