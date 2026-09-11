#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
temporary_root="$(mktemp -d "${TMPDIR:-/tmp}/avrw-clean-check.XXXXXX")"
trap 'rm -rf "${temporary_root}"' EXIT

python_command="${PYTHON:-python3}"
environment_root="${temporary_root}/venv"
"${python_command}" -m venv "${environment_root}"
environment_python="${environment_root}/bin/python"

"${environment_python}" -m pip install -e "${repository_root}"

cd "${temporary_root}"
"${environment_python}" -c "import activevision_workbench; print(activevision_workbench.__version__)"

"${environment_python}" -c 'import json,sys; from pathlib import Path; source=Path(sys.argv[1]); destination=Path(sys.argv[2]); output=Path(sys.argv[3]); data=json.loads(source.read_text(encoding="utf-8")); base=source.parent; data["run_id"]="clean-environment-dummy"; data["output_root"]=str(output); dataset=data["dataset"]; dataset["root"]=str((base/dataset["root"]).resolve()); [item.__setitem__("image_ref",str((base/item["image_ref"]).resolve())) for item in dataset["items"]]; destination.write_text(json.dumps(data),encoding="utf-8")' "${repository_root}/examples/dummy_tiny_dataset_run.yaml" "${temporary_root}/run.json" "${temporary_root}/runs"
"${environment_root}/bin/activevision" run --config "${temporary_root}/run.json"

"${environment_python}" -c 'import json,sys; from pathlib import Path; source=Path(sys.argv[1]); destination=Path(sys.argv[2]); repository=Path(sys.argv[3]); run=Path(sys.argv[4]); data=json.loads(source.read_text(encoding="utf-8")); data["run_directory"]=str(run); data["dataset"]["root"]=str(repository/"tests/fixtures/datasets/tiny_gaze"); data["output_directory"]=str(run/"metrics"); destination.write_text(json.dumps(data),encoding="utf-8")' "${repository_root}/examples/evaluation/dummy_tiny_dataset.yaml" "${temporary_root}/evaluation.json" "${repository_root}" "${temporary_root}/runs/clean-environment-dummy"
"${environment_root}/bin/activevision" evaluate --run "${temporary_root}/runs/clean-environment-dummy" --protocol "${temporary_root}/evaluation.json"
"${environment_root}/bin/activevision" inspect --run "${temporary_root}/runs/clean-environment-dummy" --item-id scene-a --output-dir "${temporary_root}/inspection"

"${environment_python}" "${repository_root}/scripts/check_release.py" --repository "${repository_root}"

cd "${repository_root}"
PYTHONDONTWRITEBYTECODE=1 "${environment_python}" -m unittest discover -s tests -v
