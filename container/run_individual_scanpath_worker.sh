#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
image="${AVRW_INDIVIDUAL_SCANPATH_SIF:-${script_dir}/individualscanpath-cu121-py310.sif}"

if ! command -v apptainer >/dev/null 2>&1; then
    if command -v module >/dev/null 2>&1; then
        module load apptainer/1.4.5
    fi
fi
command -v apptainer >/dev/null 2>&1 || {
    echo "apptainer is unavailable; load the site module first" >&2
    exit 2
}
[[ -f "${image}" ]] || {
    echo "IndividualScanpath image does not exist: ${image}" >&2
    exit 2
}

export APPTAINERENV_PYTHONPATH="${PYTHONPATH:-}"
export APPTAINERENV_PYTHONUNBUFFERED=1
export APPTAINERENV_PYTHONDONTWRITEBYTECODE=1
exec apptainer exec --nv "${image}" python3 "$@"
