#!/usr/bin/env bash
#SBATCH --job-name=avrw-scandiff
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-7%4
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --gpus-per-node=h100:1
#SBATCH --signal=B:TERM@120
#SBATCH --no-requeue
#SBATCH --open-mode=append

set -euo pipefail

: "${AVRW_PLAN_DIR:?Set AVRW_PLAN_DIR to a materialized AVRW plan directory}"
repository_root="${AVRW_REPOSITORY_ROOT:-${SLURM_SUBMIT_DIR:?}}"
avrw_python="${AVRW_PYTHON:-python}"

export PYTHONUNBUFFERED=1
cd "${repository_root}"
exec "${avrw_python}" -m activevision_workbench slurm-run-shard \
  --plan "${AVRW_PLAN_DIR}" \
  --shard-index "${SLURM_ARRAY_TASK_ID}"
