#!/usr/bin/env bash
# Submit the one-time setup, or one run, using the allocation in configs/private/hprc.env.
#   scripts/hprc/submit.sh setup
#   scripts/hprc/submit.sh MODEL [pokerl llm arguments...]     (see run.sh)
# Prints the job ID. Set AFTER=JOBID to start only once that job has finished successfully.
set -euo pipefail
source configs/private/hprc.env
mkdir -p runs
common=(--account="$HPRC_ACCOUNT" --nodes=1 --ntasks=1 --output=runs/slurm-%j.out)
if [[ -n ${AFTER:-} ]]; then
  common+=(--dependency="afterok:$AFTER" --kill-on-invalid-dep=yes)
fi

if [[ ${1:-} == setup ]]; then
  job=$(sbatch --parsable "${common[@]}" --partition="$HPRC_CPU_PARTITION" --cpus-per-task=8 \
    --mem=48G --time=04:00:00 --job-name=pokerl-setup scripts/hprc/setup.sh)
else
  job=$(sbatch --parsable "${common[@]}" --partition="$HPRC_GPU_PARTITION" --gres="$HPRC_GPU_GRES" \
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
