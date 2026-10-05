#!/usr/bin/env bash
# Submit the one-time setup, or one run, using the allocation in configs/private/hprc.env.
#   scripts/hprc/submit.sh setup
#   scripts/hprc/submit.sh MODEL [pokerl llm arguments...]     (see run.sh)
# Prints the job ID. Set AFTER=JOBID to start only once that job has finished successfully, and
# TIME=HH:MM:SS to ask for less time than the default (short jobs often start sooner), and
# SECOND_MODEL=name to serve a second model on a second node (see run.sh).
set -euo pipefail
source configs/private/hprc.env
HPRC_TIME=${TIME:-$HPRC_TIME}
mkdir -p runs
common=(--account="$HPRC_ACCOUNT" --ntasks=1 --output=runs/slurm-%j.out)
if [[ -n ${AFTER:-} ]]; then
  common+=(--dependency="afterok:$AFTER" --kill-on-invalid-dep=yes)
fi

if [[ ${1:-} == setup ]]; then
  job=$(sbatch --parsable "${common[@]}" --nodes=1 --partition="$HPRC_CPU_PARTITION" --cpus-per-task=8 \
    --mem=48G --time=04:00:00 --job-name=pokerl-setup scripts/hprc/setup.sh)
else
  nodes=1
  if [[ -n ${SECOND_MODEL:-} ]]; then
    nodes=2  # the second model gets its own GPU node
  fi
  job=$(sbatch --parsable "${common[@]}" --nodes="$nodes" --partition="$HPRC_GPU_PARTITION" --gres="$HPRC_GPU_GRES" \
    --cpus-per-task="$HPRC_CPUS" --mem="$HPRC_MEM" --time="$HPRC_TIME" --job-name=pokerl \
    scripts/hprc/run.sh "$@")
fi
# Grace's sbatch wrapper can report success even when submission fails.
if [[ ! $job =~ ^[0-9]+$ ]]; then
  printf 'Slurm did not return a job ID: %s\n' "$job" >&2
  exit 1
fi
printf '%s\n' "$job"
printf 'Submitted job %s; follow it with: tail -f runs/slurm-%s.out\n' "$job" "$job" >&2
