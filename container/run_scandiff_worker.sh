#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
image="${AVRW_SCANDIFF_SIF:-${script_dir}/scandiff-cu121-py310.sif}"

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
    echo "ScanDiff image does not exist: ${image}" >&2
    exit 2
}

export APPTAINERENV_PYTHONPATH="${PYTHONPATH:-}"
export APPTAINERENV_PYTHONUNBUFFERED=1
export APPTAINERENV_HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export APPTAINERENV_SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export APPTAINERENV_CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
exec apptainer exec --nv "${image}" python3 "$@"
