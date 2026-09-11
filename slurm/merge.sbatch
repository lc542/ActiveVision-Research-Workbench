#!/usr/bin/env bash
#SBATCH --job-name=avrw-merge
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --no-requeue
#SBATCH --open-mode=append

set -euo pipefail

: "${AVRW_PLAN_DIR:?Set AVRW_PLAN_DIR to a materialized AVRW plan directory}"
: "${AVRW_ARRAY_JOB_ID:?Set AVRW_ARRAY_JOB_ID to the submitted array job ID}"
repository_root="${AVRW_REPOSITORY_ROOT:-${SLURM_SUBMIT_DIR:?}}"
avrw_python="${AVRW_PYTHON:-python}"

export PYTHONUNBUFFERED=1
cd "${repository_root}"
exec "${avrw_python}" -m activevision_workbench slurm-merge \
  --plan "${AVRW_PLAN_DIR}" \
  --replace-matching-plan
