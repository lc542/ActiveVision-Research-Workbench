#!/usr/bin/env bash
#SBATCH --job-name=avrw-final-evaluate
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-5%6
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --signal=B:TERM@120
#SBATCH --no-requeue
#SBATCH --open-mode=append

set -euo pipefail

: "${AVRW_BENCHMARK_ROOT:?Set AVRW_BENCHMARK_ROOT to the prepared benchmark root}"
repository_root="${AVRW_REPOSITORY_ROOT:-${SLURM_SUBMIT_DIR:?}}"
avrw_python="${AVRW_PYTHON:-python}"

experiments=(
  scandiff-osie
  tpp-gaze-osie
  individualscanpath-osie
  gazexplain-osie
  scandiff-coco-search18
  gazexplain-coco-search18
)
experiment="${experiments[${SLURM_ARRAY_TASK_ID:?}]}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${repository_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${repository_root}"
exec "${avrw_python}" -m activevision_workbench evaluate \
  --run "${AVRW_BENCHMARK_ROOT}/runs/final-v1-${experiment}" \
  --protocol "${AVRW_BENCHMARK_ROOT}/configs/evaluation/${experiment}.json"
