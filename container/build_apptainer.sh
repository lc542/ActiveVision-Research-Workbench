#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RECIPE="${RECIPE:-${SCRIPT_DIR}/activevision-workbench.def}"
OUTPUT="${OUTPUT:-${SCRIPT_DIR}/activevision-workbench-py311.sif}"
BUILD_KIND="sif"
USE_FAKEROOT=1
FORCE=0

usage() {
  cat <<'EOF'
Build the ActiveVision Research Workbench Apptainer image.

Usage:
  container/build_apptainer.sh [options]

Options:
  --output PATH    Output image path.
  --recipe PATH    Apptainer definition file.
  --sandbox        Build a writable sandbox instead of a SIF.
  --no-fakeroot    Do not pass --fakeroot to apptainer build.
  --force          Explicitly overwrite an existing output.
  --help           Show this message.

The image contains code and dependencies only. Bind-mount datasets and run
directories at runtime; they are never copied into the image.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output)
      [[ $# -ge 2 ]] || { echo "--output requires a path" >&2; exit 2; }
      OUTPUT="$2"
      shift 2
      ;;
    --recipe)
      [[ $# -ge 2 ]] || { echo "--recipe requires a path" >&2; exit 2; }
      RECIPE="$2"
      shift 2
      ;;
    --sandbox)
      BUILD_KIND="sandbox"
      shift
      ;;
    --no-fakeroot)
      USE_FAKEROOT=0
      shift
      ;;
    --force)
      FORCE=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -f "${RECIPE}" ]]; then
  echo "Recipe not found: ${RECIPE}" >&2
  exit 1
fi

if [[ -e "${OUTPUT}" && "${FORCE}" -ne 1 ]]; then
  echo "Output already exists; pass --force to replace it: ${OUTPUT}" >&2
  exit 1
fi

if ! command -v apptainer >/dev/null 2>&1; then
  if command -v module >/dev/null 2>&1; then
    module load "${AVRW_APPTAINER_MODULE:-apptainer/1.4.5}"
  fi
fi

if ! command -v apptainer >/dev/null 2>&1; then
  echo "Could not find apptainer on PATH." >&2
  exit 1
fi

AVRW_TEMP_BASE="${TMPDIR:-/tmp}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${AVRW_TEMP_BASE}/avrw-cache}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-${AVRW_TEMP_BASE}/avrw-tmp}"
mkdir -p "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}" "$(dirname "${OUTPUT}")"

BUILD_ARGS=(build)
if [[ "${USE_FAKEROOT}" -eq 1 ]]; then
  BUILD_ARGS+=(--fakeroot)
fi
if [[ "${FORCE}" -eq 1 ]]; then
  BUILD_ARGS+=(--force)
fi
if [[ "${BUILD_KIND}" == "sandbox" ]]; then
  BUILD_ARGS+=(--sandbox)
fi

echo "Repository: ${REPO_ROOT}"
echo "Recipe: ${RECIPE}"
echo "Output: ${OUTPUT}"
echo "Cache: ${APPTAINER_CACHEDIR}"
echo "Temporary files: ${APPTAINER_TMPDIR}"

cd "${REPO_ROOT}"
apptainer "${BUILD_ARGS[@]}" "${OUTPUT}" "${RECIPE}"

echo "Container ready: ${OUTPUT}"
