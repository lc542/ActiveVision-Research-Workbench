#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
definition="${script_dir}/tppgaze-cu121-py310.def"
image="${1:-${script_dir}/tppgaze-cu121-py310.sif}"

if [[ -e "${image}" ]]; then
    echo "Refusing to overwrite existing image: ${image}" >&2
    exit 2
fi

if ! command -v apptainer >/dev/null 2>&1; then
    if command -v module >/dev/null 2>&1; then
        module load apptainer/1.4.5
    fi
fi
command -v apptainer >/dev/null 2>&1 || {
    echo "apptainer is unavailable; load the site module first" >&2
    exit 2
}

apptainer build "${image}" "${definition}"
