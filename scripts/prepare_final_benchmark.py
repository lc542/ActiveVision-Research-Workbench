from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from activevision_workbench.adapters.gazexplain import (
    gazexplain_run_item_from_dataset_item,
)
from activevision_workbench.adapters.individual_scanpath import (
    individual_scanpath_run_items_from_dataset_item,
)
from activevision_workbench.adapters.scandiff import (
    scandiff_run_item_from_dataset_item,
)
from activevision_workbench.adapters.tpp_gaze import (
    tpp_gaze_run_item_from_dataset_item,
)
from activevision_workbench.datasets import (
    COCOSearch18DatasetAdapter,
    OSIEDatasetAdapter,
)
from activevision_workbench.run_config import RunConfig
from activevision_workbench.slurm import SlurmConfig, create_plan


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_FINGERPRINT_CACHE: dict[Path, str] = {}
OSIE_TEST_ITEM_IDS = (
    "1009",
    "1017",
    "1049",
    "1056",
    "1062",
    "1086",
    "1087",
    "1099",
    "1108",
    "1114",
    "1116",
    "1117",
    "1127",
    "1130",
    "1131",
    "1136",
    "1140",
    "1152",
    "1192",
    "1220",
    "1225",
    "1226",
    "1252",
    "1255",
    "1269",
    "1295",
    "1307",
    "1360",
    "1369",
    "1372",
    "1394",
    "1397",
    "1405",
    "1420",
    "1423",
    "1433",
    "1441",
    "1478",
    "1480",
    "1481",
    "1489",
    "1490",
    "1493",
    "1502",
    "1509",
    "1523",
    "1528",
    "1530",
    "1549",
    "1555",
    "1558",
    "1567",
    "1576",
    "1581",
    "1595",
    "1596",
    "1605",
    "1609",
    "1615",
    "1616",
    "1618",
    "1622",
    "1628",
    "1637",
    "1640",
    "1657",
    "1663",
    "1677",
    "1682",
    "1699",
)
COCO_PILOT_ITEM_ID = "validation:bottle:000000350270"
OSIE_PROMPT = "Question: What do you see in the image?"


@dataclass(frozen=True, slots=True)
class Experiment:
    name: str
    template: str
    upstream: str
    dataset: str
    items: tuple[dict[str, object], ...]
    shard_count: int
    assignment_strategy: str
    cpus: int
    memory: str
    gpu_type: str
    full_time: str
    pilot_time: str
    pilot_item_id: str


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--osie-root", type=Path, required=True)
    parser.add_argument("--coco-root", type=Path, required=True)
    parser.add_argument("--scandiff-root", type=Path, required=True)
    parser.add_argument("--tpp-gaze-root", type=Path, required=True)
    parser.add_argument("--individual-scanpath-root", type=Path, required=True)
    parser.add_argument("--gazexplain-root", type=Path, required=True)
    parser.add_argument("--account", required=True)
    parser.add_argument("--max-parallel-tasks", type=int, default=8)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _fingerprint(path: Path) -> str:
    path = path.resolve()
    cached = _FINGERPRINT_CACHE.get(path)
    if cached is not None:
        return cached
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    fingerprint = f"sha256:{digest.hexdigest()}"
    _FINGERPRINT_CACHE[path] = fingerprint
    return fingerprint


def _selection_fingerprint(dataset: str, split: str, item_ids: tuple[str, ...]) -> str:
    payload = json.dumps(
        {"dataset": dataset, "split": split, "item_ids": list(item_ids)},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _git_head(path: Path) -> str:
    result = subprocess.run(
        ("git", "-C", str(path), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _load_template(relative: str) -> dict[str, object]:
    path = REPOSITORY_ROOT / relative
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _model(
    template: str,
    upstream_root: Path,
    *,
    output_root: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    config = _load_template(template)
    model = config["model"]
    environment = config["environment"]
    if not isinstance(model, dict) or not isinstance(environment, dict):
        raise ValueError(f"{template} has invalid model metadata")
    options = model["options"]
    if not isinstance(options, dict) or len(options) != 1:
        raise ValueError(f"{template} has invalid model options")
    model_id, raw_options = next(iter(options.items()))
    if not isinstance(raw_options, dict):
        raise ValueError(f"{template} has invalid {model_id} options")
    launcher_names = {
        "scandiff": "run_scandiff_worker.sh",
        "tpp_gaze": "run_tppgaze_worker.sh",
        "individualscanpath": "run_individual_scanpath_worker.sh",
        "gazexplain": "run_gazexplain_worker.sh",
    }
    image_names = {
        "scandiff": "scandiff-cu121-py310.sif",
        "tpp_gaze": "tppgaze-cu121-py310.sif",
        "individualscanpath": "individualscanpath-cu121-py310.sif",
        "gazexplain": "gazexplain-cu121-py310.sif",
    }
    checkpoint_paths = {
        "scandiff": {
            "free-viewing": upstream_root
            / "checkpoints"
            / "scandiff_freeview.pth",
            "visual-search": upstream_root
            / "checkpoints"
            / "scandiff_visualsearch.pth",
        },
        "tpp_gaze": {
            "transformer": upstream_root / "checkpoints" / "model_transformer.pth"
        },
        "individualscanpath": {
            "chenlstm-osie": upstream_root
            / "checkpoints"
            / "pretrained_models"
            / "ChenLSTM"
            / "OSIE"
            / "checkpoints"
            / "checkpoint_best.pth"
        },
        "gazexplain": {
            "joint-all": upstream_root
            / "checkpoints"
            / "ALL_runX_baseline"
            / "checkpoints"
            / "ckpt_best"
            / "model.safetensors"
        },
    }
    variant = model["variant"]
    if not isinstance(model_id, str) or not isinstance(variant, str):
        raise ValueError(f"{template} has invalid model identity")
    checkpoint = checkpoint_paths[model_id][variant]
    launcher = REPOSITORY_ROOT / "container" / launcher_names[model_id]
    image = REPOSITORY_ROOT / "container" / image_names[model_id]
    required = (upstream_root, checkpoint, launcher, image)
    if any(not path.exists() for path in required):
        missing = next(path for path in required if not path.exists())
        raise FileNotFoundError(missing)
    expected_commit = model["upstream_commit"]
    if _git_head(upstream_root) != expected_commit:
        raise ValueError(
            f"{model_id} checkout does not match pinned commit {expected_commit}"
        )
    raw_options["upstream_root"] = str(upstream_root)
    raw_options["launcher"] = [str(launcher)]
    if model_id == "scandiff":
        raw_options["task_embeddings_path"] = str(
            upstream_root / "data" / "task_embeddings.npy"
        )
        raw_options["feature_root"] = None
        raw_options["allow_feature_download"] = False
    elif model_id == "tpp_gaze":
        raw_options["model_config_path"] = str(upstream_root / "data" / "config.yaml")
        raw_options["safety_max_fixations"] = 64
    elif model_id == "individualscanpath":
        mapping = (
            REPOSITORY_ROOT
            / "examples"
            / "individual_scanpath"
            / "osie_observers.json"
        )
        raw_options["observer_mapping_path"] = str(mapping)
        raw_options["observer_mapping_fingerprint"] = _fingerprint(mapping)
    elif model_id == "gazexplain":
        hparams = (
            upstream_root / "checkpoints" / "ALL_runX_baseline" / "hparams.json"
        )
        raw_options["hparams_path"] = str(hparams)
        raw_options["hparams_fingerprint"] = _fingerprint(hparams)
    model["checkpoint_path"] = str(checkpoint)
    model["checkpoint_fingerprint"] = _fingerprint(checkpoint)
    environment["image_digest"] = _fingerprint(image)
    config["output_root"] = str(output_root)
    return model, environment


def _run_config(
    experiment: Experiment,
    *,
    upstream_root: Path,
    output_root: Path,
    dataset_root: Path,
    dataset_version: str,
    dataset_split: str,
    dataset_fingerprint: str,
    pilot: bool,
) -> dict[str, object]:
    model, environment = _model(
        experiment.template,
        upstream_root,
        output_root=output_root,
    )
    items = list(copy.deepcopy(experiment.items))
    if pilot:
        pilot_item = next(
            (
                item
                for item in items
                if item["item_id"] == experiment.pilot_item_id
            ),
            None,
        )
        if pilot_item is None:
            raise ValueError(
                f"{experiment.name} pilot item {experiment.pilot_item_id!r} "
                "was not found"
            )
        if experiment.assignment_strategy == "balanced_by_image":
            pilot_image_ref = pilot_item["image_ref"]
            items = [
                item for item in items if item["image_ref"] == pilot_image_ref
            ]
        else:
            items = [pilot_item]
    prefix = "pilot-v1" if pilot else "final-v1"
    config = {
        "schema_version": 1,
        "run_id": f"{prefix}-{experiment.name}",
        "model": model,
        "dataset": {
            "id": experiment.dataset,
            "version": dataset_version,
            "split": dataset_split,
            "root": str(dataset_root),
            "root_fingerprint": dataset_fingerprint,
            "items": items,
            "item_subset": None,
        },
        "request": {
            "task_text": None,
            "target_description": None,
            "observer_id": None,
            "observer_metadata": {},
            "num_samples": 10,
            "base_seed": 2026,
            "seed_strategy": "fixed_per_item",
            "max_fixations": 64 if experiment.name == "tpp-gaze-osie" else 16,
            "time_horizon_s": 2.0 if experiment.name == "tpp-gaze-osie" else None,
        },
        "device": "cuda",
        "output_root": str(output_root),
        "run_policy": "create",
        "tags": [
            "final-benchmark" if not pilot else "final-benchmark-pilot",
            experiment.name,
        ],
        "notes": (
            "Exact ten-sample resource pilot."
            if pilot
            else "Complete compatible final benchmark run."
        ),
        "environment": environment,
    }
    return RunConfig.from_dict(config).to_dict()


def _slurm_config(
    experiment: Experiment,
    *,
    account: str,
    max_parallel_tasks: int,
    repository_root: Path,
    output_root: Path,
    dataset_root: Path,
    upstream_root: Path,
    python: str,
    pilot: bool,
) -> dict[str, object]:
    shard_count = 1 if pilot else experiment.shard_count
    prefix = "pilot" if pilot else "final"
    config = {
        "schema_version": 1,
        "account": account,
        "partition": None,
        "job_name": f"avrw-{prefix}-{experiment.name}",
        "shard_count": shard_count,
        "max_parallel_tasks": (
            1 if pilot else min(max_parallel_tasks, shard_count)
        ),
        "assignment_strategy": experiment.assignment_strategy,
        "resources": {
            "nodes": 1,
            "tasks_per_node": 1,
            "cpus_per_task": experiment.cpus,
            "gpus_per_node": 1,
            "gpu_type": experiment.gpu_type,
            "memory": experiment.memory,
            "time_limit": (
                experiment.pilot_time if pilot else experiment.full_time
            ),
        },
        "environment": {
            "name": "activevision-core-with-isolated-model-worker",
            "modules": [],
            "launcher": [
                "env",
                f"PYTHONPATH={repository_root / 'src'}",
                python,
            ],
            "apptainer_image": None,
            "apptainer_gpu": False,
            "bind_paths": [],
        },
        "working_directory": str(repository_root),
        "dataset_roots": [str(dataset_root)],
        "checkpoint_roots": [str(upstream_root / "checkpoints")],
        "log_directory": str(
            output_root / "logs" / prefix / experiment.name
        ),
        "email": None,
        "mail_types": [],
        "requeue": False,
        "max_retries": 1,
        "signal_seconds": 300,
        "dependency_job_ids": [],
    }
    return SlurmConfig.from_dict(config).to_dict()


def _evaluation_config(
    experiment: Experiment,
    *,
    benchmark_root: Path,
    dataset_root: Path,
    dataset_version: str,
) -> dict[str, object]:
    is_osie = experiment.dataset == "osie"
    run_id = f"final-v1-{experiment.name}"
    return {
        "schema_version": 1,
        "run_directory": str(benchmark_root / "runs" / run_id),
        "dataset": {
            "adapter": "osie" if is_osie else "coco_search18",
            "id": experiment.dataset,
            "version": dataset_version,
            "root": str(dataset_root),
            "split": "all" if is_osie else "validation",
            "task_type": (
                "free_viewing" if is_osie else "visual_search_target_present"
            ),
            "options": (
                {
                    "duration_unit": "milliseconds",
                    "clip_out_of_bounds": False,
                }
                if is_osie
                else {
                    "split_definition": "split1",
                    "duration_unit": "milliseconds",
                    "clip_out_of_bounds": False,
                }
            ),
        },
        "metrics": [
            {"name": "fixation_count", "parameters": {}},
            {"name": "scanpath_length_normalized", "parameters": {}},
            {"name": "timestamp_span_s", "parameters": {}},
            {"name": "total_fixation_duration_s", "parameters": {}},
            {"name": "mean_fixation_duration_s", "parameters": {}},
            {"name": "duration_mae_s", "parameters": {}},
            {
                "name": "tfp_auc",
                "parameters": {
                    "max_fixations": 16,
                    "include_initial_fixation": True,
                    "normalization": "unit_interval",
                },
            },
            {
                "name": "endpoint_pairwise_distance_normalized",
                "parameters": {},
            },
            {
                "name": "endpoint_dispersion_normalized",
                "parameters": {},
            },
            {
                "name": "fixation_density_kde",
                "parameters": {
                    "kernel": "gaussian",
                    "bandwidth_norm": 0.08,
                    "grid_width": 32,
                    "grid_height": 24,
                },
            },
            {
                "name": "auc_judd_fixation_density",
                "parameters": {
                    "kernel": "gaussian",
                    "bandwidth_norm": 0.08,
                    "grid_width": 32,
                    "grid_height": 24,
                    "tie_policy": "average_rank",
                },
            },
            {
                "name": "nss_fixation_density",
                "parameters": {
                    "kernel": "gaussian",
                    "bandwidth_norm": 0.08,
                    "grid_width": 32,
                    "grid_height": 24,
                },
            },
            {"name": "auc_judd", "parameters": {"artifact_format": "dense_json"}},
            {"name": "nss", "parameters": {"artifact_format": "dense_json"}},
        ],
        "sample_aggregation": {
            "policy": "mean_std",
            "standard_deviation": "population",
            "confidence_interval": {"method": "normal", "level": 0.95},
            "best_of_n": {"enabled": False, "metrics": []},
        },
        "observer_aggregation": {
            "policy": "macro",
            "include_micro": False,
            "unknown_observer_policy": "separate",
        },
        "temporal_alignment": {
            "policy": "truncate_to_shorter",
            "unit": "seconds",
        },
        "coordinate_conversion": {
            "policy": "canonical_pixel_center",
            "out_of_bounds": "invalid",
        },
        "output_directory": str(
            benchmark_root / "evaluations" / experiment.name
        ),
        "existing_output": "create",
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    arguments = _arguments()
    roots = {
        "scandiff": arguments.scandiff_root.resolve(),
        "tpp_gaze": arguments.tpp_gaze_root.resolve(),
        "individualscanpath": arguments.individual_scanpath_root.resolve(),
        "gazexplain": arguments.gazexplain_root.resolve(),
    }
    benchmark_root = arguments.output_root.resolve()
    if benchmark_root.exists() and any(benchmark_root.iterdir()):
        if not arguments.overwrite:
            raise FileExistsError(
                f"{benchmark_root} is not empty; use --overwrite to replace configs"
            )
    benchmark_root.mkdir(parents=True, exist_ok=True)
    osie_root = arguments.osie_root.resolve()
    coco_root = arguments.coco_root.resolve()
    osie = OSIEDatasetAdapter(
        osie_root,
        duration_unit="milliseconds",
        clip_out_of_bounds=False,
    )
    coco = COCOSearch18DatasetAdapter(
        coco_root,
        split_definition="split1",
        duration_unit="milliseconds",
        clip_out_of_bounds=False,
    )
    available_osie = set(osie.item_ids(split="all"))
    missing_osie = sorted(set(OSIE_TEST_ITEM_IDS) - available_osie)
    if missing_osie:
        raise ValueError(f"OSIE test selection is missing item {missing_osie[0]}")
    coco_ids = coco.item_ids(split="validation")
    if len(coco_ids) != 326 or COCO_PILOT_ITEM_ID not in coco_ids:
        raise ValueError("COCO-Search18 split1 validation identity changed")
    osie_items = tuple(
        osie.get_item(item_id, split="all") for item_id in OSIE_TEST_ITEM_IDS
    )
    coco_items = tuple(
        coco.get_item(item_id, split="validation") for item_id in coco_ids
    )
    scandiff_osie = tuple(
        scandiff_run_item_from_dataset_item(item, variant="free-viewing")
        for item in osie_items
    )
    tpp_osie = tuple(
        tpp_gaze_run_item_from_dataset_item(item) for item in osie_items
    )
    individual_osie = tuple(
        run_item
        for item in osie_items
        for run_item in individual_scanpath_run_items_from_dataset_item(item)
    )
    gaze_osie = tuple(
        gazexplain_run_item_from_dataset_item(item, task_text=OSIE_PROMPT)
        for item in osie_items
    )
    scandiff_coco = tuple(
        scandiff_run_item_from_dataset_item(item, variant="visual-search")
        for item in coco_items
    )
    gaze_coco = tuple(
        gazexplain_run_item_from_dataset_item(
            item,
            task_text=(
                f"Question: Is there a {item.target_label} in the image? "
                "Answer: yes."
            ),
        )
        for item in coco_items
    )
    experiments = (
        Experiment(
            "scandiff-osie",
            "examples/scandiff/free_viewing.json",
            "scandiff",
            "osie",
            scandiff_osie,
            3,
            "balanced",
            4,
            "32G",
            "h100",
            "00:30:00",
            "00:30:00",
            "1009",
        ),
        Experiment(
            "tpp-gaze-osie",
            "examples/tpp_gaze/multiple_simulations.json",
            "tpp_gaze",
            "osie",
            tpp_osie,
            1,
            "balanced",
            4,
            "24G",
            "h100",
            "00:30:00",
            "00:30:00",
            "1009",
        ),
        Experiment(
            "individualscanpath-osie",
            "examples/individual_scanpath/two_observers.json",
            "individualscanpath",
            "osie",
            individual_osie,
            2,
            "balanced_by_image",
            4,
            "24G",
            "h100",
            "00:30:00",
            "00:30:00",
            "1009:subject-01",
        ),
        Experiment(
            "gazexplain-osie",
            "examples/gazexplain/two_tasks.json",
            "gazexplain",
            "osie",
            gaze_osie,
            2,
            "balanced",
            8,
            "48G",
            "h100",
            "00:30:00",
            "00:30:00",
            "1009",
        ),
        Experiment(
            "scandiff-coco-search18",
            "examples/scandiff/visual_search.json",
            "scandiff",
            "coco_search18",
            scandiff_coco,
            12,
            "balanced",
            4,
            "32G",
            "h100",
            "00:30:00",
            "00:30:00",
            COCO_PILOT_ITEM_ID,
        ),
        Experiment(
            "gazexplain-coco-search18",
            "examples/gazexplain/two_tasks.json",
            "gazexplain",
            "coco_search18",
            gaze_coco,
            5,
            "balanced",
            8,
            "48G",
            "h100",
            "00:30:00",
            "00:30:00",
            COCO_PILOT_ITEM_ID,
        ),
    )
    osie_fingerprint = _selection_fingerprint(
        "osie", "upstream-test", OSIE_TEST_ITEM_IDS
    )
    coco_fingerprint = _selection_fingerprint(
        "coco_search18", "split1-validation", coco_ids
    )
    dataset_info = {
        "osie": (
            osie_root,
            osie.dataset_version,
            "all",
            osie_fingerprint,
        ),
        "coco_search18": (
            coco_root,
            coco.dataset_version,
            "validation",
            coco_fingerprint,
        ),
    }
    summaries = []
    for experiment in experiments:
        dataset_root, version, split, fingerprint = dataset_info[
            experiment.dataset
        ]
        full_run = _run_config(
            experiment,
            upstream_root=roots[experiment.upstream],
            output_root=benchmark_root / "runs",
            dataset_root=dataset_root,
            dataset_version=version,
            dataset_split=split,
            dataset_fingerprint=fingerprint,
            pilot=False,
        )
        pilot_run = _run_config(
            experiment,
            upstream_root=roots[experiment.upstream],
            output_root=benchmark_root / "pilots",
            dataset_root=dataset_root,
            dataset_version=version,
            dataset_split=split,
            dataset_fingerprint=fingerprint,
            pilot=True,
        )
        full_slurm = _slurm_config(
            experiment,
            account=arguments.account,
            max_parallel_tasks=arguments.max_parallel_tasks,
            repository_root=REPOSITORY_ROOT,
            output_root=benchmark_root,
            dataset_root=dataset_root,
            upstream_root=roots[experiment.upstream],
            python=arguments.python,
            pilot=False,
        )
        pilot_slurm = _slurm_config(
            experiment,
            account=arguments.account,
            max_parallel_tasks=arguments.max_parallel_tasks,
            repository_root=REPOSITORY_ROOT,
            output_root=benchmark_root,
            dataset_root=dataset_root,
            upstream_root=roots[experiment.upstream],
            python=arguments.python,
            pilot=True,
        )
        full_run_path = benchmark_root / "configs" / "full" / f"{experiment.name}.run.json"
        full_slurm_path = (
            benchmark_root / "configs" / "full" / f"{experiment.name}.slurm.json"
        )
        pilot_run_path = (
            benchmark_root / "configs" / "pilot" / f"{experiment.name}.run.json"
        )
        pilot_slurm_path = (
            benchmark_root
            / "configs"
            / "pilot"
            / f"{experiment.name}.slurm.json"
        )
        _write_json(full_run_path, full_run)
        _write_json(full_slurm_path, full_slurm)
        _write_json(pilot_run_path, pilot_run)
        _write_json(pilot_slurm_path, pilot_slurm)
        _write_json(
            benchmark_root
            / "configs"
            / "evaluation"
            / f"{experiment.name}.json",
            _evaluation_config(
                experiment,
                benchmark_root=benchmark_root,
                dataset_root=dataset_root,
                dataset_version=version,
            ),
        )
        run = RunConfig.from_file(full_run_path)
        slurm = SlurmConfig.from_file(full_slurm_path)
        plan = create_plan(run, slurm)
        sizes = [len(shard.item_ids) for shard in plan.shards]
        summaries.append(
            {
                "name": experiment.name,
                "dataset": experiment.dataset,
                "model": run.model_id,
                "request_count": len(run.selected_items),
                "prediction_count": len(run.selected_items) * run.num_samples,
                "shard_count": len(plan.shards),
                "nonempty_shard_count": sum(size > 0 for size in sizes),
                "minimum_shard_requests": min(sizes),
                "maximum_shard_requests": max(sizes),
                "assignment_strategy": slurm.assignment_strategy,
                "max_parallel_tasks": slurm.max_parallel_tasks,
                "gpu_type": slurm.resources.gpu_type,
                "time_limit": slurm.resources.time_limit,
                "run_config": str(full_run_path),
                "slurm_config": str(full_slurm_path),
            }
        )
    _write_json(
        benchmark_root / "configs" / "osie-upstream-test-items.json",
        {
            "schema_version": 1,
            "dataset": "osie",
            "split_name": "upstream-test",
            "source": (
                "Pinned IndividualScanpath and GazeXplain OSIE preprocessing "
                "protocol"
            ),
            "item_ids": list(OSIE_TEST_ITEM_IDS),
            "selection_fingerprint": osie_fingerprint,
        },
    )
    manifest = {
        "schema_version": 1,
        "benchmark_id": "activevision-final-v1",
        "sample_seeds": list(range(2026, 2036)),
        "experiments": summaries,
        "total_requests": sum(
            int(summary["request_count"]) for summary in summaries
        ),
        "total_predictions": sum(
            int(summary["prediction_count"]) for summary in summaries
        ),
    }
    _write_json(benchmark_root / "experiment-manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
