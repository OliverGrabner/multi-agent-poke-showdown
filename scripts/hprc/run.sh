#!/usr/bin/env bash
# One run as one Slurm job: start the model server(s), wait until they answer, check the
# connections, play the battles, then stop the servers.
# Submit with: scripts/hprc/submit.sh MODEL [pokerl llm arguments...]
# Example:     scripts/hprc/submit.sh qwen3.8-27b --side-a qwen3.8-27b --side-b maxpower \
#                  --mode-a free --label baseline-free --pairs 50 --workers 24
# With SECOND_MODEL set (submit.sh then asks for two nodes), that model is served on the second
# node, for battles between two different models.
set -euo pipefail
umask 077
: "${SLURM_JOB_ID:?Submit this script through Slurm: scripts/hprc/submit.sh}"
cd "${SLURM_SUBMIT_DIR:?}"
source scripts/hprc/common.sh
source configs/private/hprc.env
printf '%s job started on %s\n' "$(date +%T)" "$(hostname)"
model=${1:?Give the model to serve, e.g. qwen3.8-27b}
shift

# A private key for this job's servers; it never leaves the job's nodes.
export VLLM_API_KEY="$(.venv/bin/python -c 'import secrets; print(secrets.token_urlsafe(32))')"
nvidia-smi > "runs/vllm-${SLURM_JOB_ID}-gpus.txt"

servers=()
trap 'kill "${servers[@]}" 2>/dev/null || true' EXIT
scripts/hprc/serve.sh "$model" 127.0.0.1 8000 &
servers+=($!)
model_info "$model"
export "$url_var=http://127.0.0.1:8000/v1"
urls=("http://127.0.0.1:8000/v1")
checks=("$model")

if [[ -n ${SECOND_MODEL:-} ]]; then
  second_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sed -n 2p)
  : "${second_node:?SECOND_MODEL needs a two-node job: submit through scripts/hprc/submit.sh}"
  srun --nodes=1 --ntasks=1 --relative=1 --gres="$HPRC_GPU_GRES" --cpus-per-task=8 --mem=0 \
    scripts/hprc/serve.sh "$SECOND_MODEL" 0.0.0.0 8001 &
  servers+=($!)
  model_info "$SECOND_MODEL"
  export "$url_var=http://$second_node:8001/v1"
  export NO_PROXY="$NO_PROXY,$second_node" no_proxy="$NO_PROXY,$second_node"
  urls+=("http://$second_node:8001/v1")
  checks+=("$SECOND_MODEL")
fi

# Wait up to 90 minutes for every server (loading weights and compiling kernels takes a while).
for url in "${urls[@]}"; do
  ready=false
  for _ in $(seq 540); do
    if curl -fs -o /dev/null -H "Authorization: Bearer $VLLM_API_KEY" "$url/models"; then
      ready=true
      break
    fi
    for server in "${servers[@]}"; do
      if ! kill -0 "$server" 2>/dev/null; then
        printf 'A model server stopped; see runs/vllm-%s-*.log\n' "$SLURM_JOB_ID" >&2
        exit 1
      fi
    done
    sleep 10
  done
  if [[ $ready != true ]]; then
    printf 'No model server at %s after 90 minutes; see runs/vllm-%s-*.log\n' "$url" "$SLURM_JOB_ID" >&2
    exit 1
  fi
  printf '%s model server ready at %s\n' "$(date +%T)" "$url"
done

for check in "${checks[@]}"; do
  .venv/bin/pokerl model-check --model "$check"
done
if [[ $# -gt 0 ]]; then
  .venv/bin/pokerl llm "$@"
fi
