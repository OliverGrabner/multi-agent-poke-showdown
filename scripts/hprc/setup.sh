#!/usr/bin/env bash
# One-time setup on Grace (submit with: scripts/hprc/submit.sh setup).
# Installs uv, the project's Python environment, a separate vLLM environment, a portable Node.js
# with the Showdown simulator, and the model weights; then runs the test suite as a check.
set -euo pipefail
umask 077
: "${SLURM_JOB_ID:?Submit this script through Slurm: scripts/hprc/submit.sh setup}"
cd "${SLURM_SUBMIT_DIR:?}"
module load WebProxy
source scripts/hprc/common.sh
source configs/private/hprc.env

NODE_VERSION=22.14.0
mkdir -p .tools

if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$PWD/.tools" INSTALLER_NO_MODIFY_PATH=1 sh
fi

uv venv --allow-existing --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"

# vLLM gets its own environment; managed Python ships the headers Triton needs.
uv venv --allow-existing --python 3.12 --seed .venv-vllm
uv pip install --python .venv-vllm/bin/python "vllm${VLLM_VERSION:+==$VLLM_VERSION}"
.venv-vllm/bin/python -m pip freeze > "runs/setup-${SLURM_JOB_ID}-packages.txt"

if [[ ! -x .tools/node/bin/node ]]; then
  curl -LsSf "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-x64.tar.xz" | tar -xJ -C .tools
  mv ".tools/node-v${NODE_VERSION}-linux-x64" .tools/node
fi
(cd bridge && npm ci --no-audit --no-fund)

download() {
  .venv-vllm/bin/python -c 'import sys; from huggingface_hub import snapshot_download; snapshot_download(sys.argv[1])' "$1"
}
download Qwen/Qwen3.8-27B
# The second family is optional; its weights may need a license accepted on Hugging Face first.
download google/gemma-4-31B-it || printf 'Warning: could not download google/gemma-4-31B-it\n' >&2

.venv/bin/python -m pytest -q
printf 'Setup complete: %s\n' "$SLURM_JOB_ID"
