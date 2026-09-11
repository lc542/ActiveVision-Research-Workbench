from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import tempfile
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path

from activevision_workbench.contracts import PredictionRecord
from activevision_workbench.errors import (
    SlurmMergeError,
    SlurmPlanError,
    SlurmSubmissionError,
)
from activevision_workbench.registry import AdapterRegistry
from activevision_workbench.run_artifacts import (
    FailureRecord,
    RunDirectory,
    RunStatus,
    initial_manifest,
    utc_now,
)
from activevision_workbench.run_config import RunConfig, RunPolicy
from activevision_workbench.run_engine import RunEngine, RunResult
from activevision_workbench.serialization import dumps_prediction
from activevision_workbench.slurm.config import SlurmConfig
from activevision_workbench.slurm.planning import SlurmPlan, create_plan

CommandRunner = Callable[[tuple[str, ...]], subprocess.CompletedProcess[str]]
RegistryFactory = Callable[[RunConfig], AdapterRegistry]


@dataclass(frozen=True, slots=True)
class SubmissionResult:
    plan_directory: Path
    array_job_id: str | None
    merge_job_id: str | None
    selected_shards: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class RetryPlan:
    plan_directory: Path
    eligible_shards: tuple[int, ...]
    completed_shards: tuple[int, ...]
    exhausted_shards: tuple[int, ...]
    active_shards: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "plan_directory": str(self.plan_directory),
            "eligible_shards": list(self.eligible_shards),
            "completed_shards": list(self.completed_shards),
            "exhausted_shards": list(self.exhausted_shards),
            "active_shards": list(self.active_shards),
        }


class SlurmExecutor:
    def __init__(self, command_runner: CommandRunner | None = None) -> None:
        self.command_runner = command_runner or _run_command

    def dry_run(
        self, run_config: RunConfig, slurm_config: SlurmConfig
    ) -> SlurmPlan:
        plan = create_plan(run_config, slurm_config)
        validate_environment(plan)
        return plan

    def materialize(
        self, run_config: RunConfig, slurm_config: SlurmConfig
    ) -> SlurmPlan:
        plan = create_plan(run_config, slurm_config)
        validate_environment(plan)
        materialize_plan(plan)
        return plan

    def submit(
        self,
        run_config: RunConfig,
        slurm_config: SlurmConfig,
        *,
        resume: bool = False,
        retry_running: bool = False,
    ) -> SubmissionResult:
        expected = create_plan(run_config, slurm_config)
        validate_environment(expected)
        if resume:
            plan = load_plan(expected.plan_directory)
            retry = retry_plan(
                plan.plan_directory, include_running=retry_running
            )
            selected = retry.eligible_shards
            if not selected:
                return SubmissionResult(plan.plan_directory, None, None, ())
            plan = replace(plan, selected_shards=selected, retry=True)
            script_name = _next_retry_script_name(plan.plan_directory)
            script_path = plan.plan_directory / script_name
            _atomic_write_text(script_path, plan.array_script, executable=True)
        else:
            plan = expected
            materialize_plan(plan)
            script_path = plan.plan_directory / "array.sbatch"

        log_directory = Path(slurm_config.log_directory)
        log_directory.mkdir(parents=True, exist_ok=True)
        array_command = (
            "sbatch",
            "--parsable",
            "--output",
            str(log_directory / f"{slurm_config.job_name}-%A_%a.out"),
            "--error",
            str(log_directory / f"{slurm_config.job_name}-%A_%a.err"),
            str(script_path),
        )
        array_job_id = _submit(self.command_runner, array_command)
        _update_plan_submission(
            plan.plan_directory,
            array_job_id=array_job_id,
            merge_job_id=None,
            selected_shards=plan.selected_shards,
        )
        merge_command = (
            "sbatch",
            "--parsable",
            f"--dependency=afterany:{array_job_id}",
            "--output",
            str(log_directory / f"{slurm_config.job_name}-merge-%j.out"),
            "--error",
            str(log_directory / f"{slurm_config.job_name}-merge-%j.err"),
            str(plan.plan_directory / "merge.sbatch"),
        )
        merge_job_id = _submit(self.command_runner, merge_command)
        _update_plan_submission(
            plan.plan_directory,
            array_job_id=array_job_id,
            merge_job_id=merge_job_id,
            selected_shards=plan.selected_shards,
        )
        return SubmissionResult(
            plan.plan_directory,
            array_job_id,
            merge_job_id,
            plan.selected_shards,
        )


def materialize_plan(plan: SlurmPlan) -> None:
    destination = plan.plan_directory
    if destination.exists():
        raise SlurmPlanError(
            f"Slurm plan already exists: {destination}; use submit --resume"
        )
    if plan.final_run_directory.exists():
        raise SlurmPlanError(
            f"final run directory already exists: {plan.final_run_directory}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.creating-", dir=destination.parent
        )
    )
    installed = False
    try:
        (temporary / "shards").mkdir()
        _atomic_write_json(temporary / "plan.json", plan.to_dict())
        _atomic_write_json(temporary / "run-config.json", plan.run_config.to_dict())
        _atomic_write_json(temporary / "slurm-config.json", plan.slurm_config.to_dict())
        _atomic_write_text(
            temporary / "array.sbatch", plan.array_script, executable=True
        )
        _atomic_write_text(
            temporary / "merge.sbatch", plan.merge_script, executable=True
        )
        for shard in plan.shards:
            shard_directory = temporary / "shards" / f"{shard.index:04d}"
            shard_directory.mkdir()
            metadata = {
                **shard.to_dict(),
                "plan_hash": plan.plan_hash,
                "base_configuration_hash": plan.run_config.configuration_hash,
                "output_root": str(
                    destination / "shards" / f"{shard.index:04d}" / "runs"
                ),
            }
            _atomic_write_json(shard_directory / "shard.json", metadata)
            _atomic_write_json(
                shard_directory / "status.json",
                {
                    "schema_version": 1,
                    "shard_index": shard.index,
                    "state": "pending",
                    "attempts": 0,
                    "updated_at": utc_now(),
                    "job_id": None,
                    "array_job_id": None,
                    "array_task_id": None,
                    "error": None,
                },
            )
        os.replace(temporary, destination)
        installed = True
    finally:
        if not installed and temporary.exists():
            shutil.rmtree(temporary)


def load_plan(path: str | Path) -> SlurmPlan:
    directory = Path(path)
    state = _read_json(directory / "plan.json")
    run_config = RunConfig.from_dict(
        _read_json(directory / "run-config.json"), base_dir=directory
    )
    slurm_config = SlurmConfig.from_dict(
        _read_json(directory / "slurm-config.json"), base_dir=directory
    )
    expected = create_plan(run_config, slurm_config)
    if expected.plan_hash != state.get("plan_hash"):
        raise SlurmPlanError("stored Slurm plan hash does not match its configurations")
    if directory.resolve(strict=False) != expected.plan_directory.resolve(strict=False):
        raise SlurmPlanError("stored Slurm plan is not at its deterministic path")
    selected_raw = state.get("selected_shards")
    if not isinstance(selected_raw, list) or any(
        type(value) is not int for value in selected_raw
    ):
        raise SlurmPlanError("stored Slurm plan has invalid selected_shards")
    return replace(
        expected,
        selected_shards=tuple(selected_raw),
        retry=bool(state.get("retry", False)),
    )


def validate_environment(plan: SlurmPlan) -> None:
    config = plan.slurm_config
    working = Path(config.working_directory)
    if not working.is_dir():
        raise SlurmPlanError(f"working directory does not exist: {working}")
    for label, roots in (
        ("dataset root", config.dataset_roots),
        ("checkpoint root", config.checkpoint_roots),
        ("environment bind path", config.environment.bind_paths),
    ):
        for value in roots:
            if not Path(value).exists():
                raise SlurmPlanError(f"{label} does not exist: {value}")
    _validate_declared_root(
        plan.run_config.dataset_root,
        config.dataset_roots,
        "run dataset root",
    )
    _validate_declared_root(
        plan.run_config.checkpoint_path,
        config.checkpoint_roots,
        "run checkpoint",
    )
    image = config.environment.apptainer_image
    if image is not None and not Path(image).is_file():
        raise SlurmPlanError(f"Apptainer image does not exist: {image}")
    if image is None and not config.environment.modules:
        executable = config.environment.launcher[0]
        if "/" in executable:
            available = Path(executable).is_file() and os.access(executable, os.X_OK)
        else:
            available = shutil.which(executable) is not None
        if not available:
            raise SlurmPlanError(f"environment launcher is unavailable: {executable}")


def _validate_declared_root(
    reference: str | None,
    roots: tuple[str, ...],
    label: str,
) -> None:
    if reference is None or "://" in reference or not roots:
        return
    resolved = Path(reference).resolve(strict=False)
    if not any(
        resolved.is_relative_to(Path(root).resolve(strict=False))
        for root in roots
    ):
        raise SlurmPlanError(
            f"{label} is outside the corresponding declared Slurm roots: {reference}"
        )


def retry_plan(path: str | Path, *, include_running: bool = False) -> RetryPlan:
    plan = load_plan(path)
    eligible: list[int] = []
    completed: list[int] = []
    exhausted: list[int] = []
    active: list[int] = []
    limit = 1 + plan.slurm_config.max_retries
    for shard in plan.shards:
        status = _read_status(plan.plan_directory, shard.index)
        state = status.get("state")
        attempts = status.get("attempts")
        if type(attempts) is not int or attempts < 0:
            raise SlurmPlanError(f"shard {shard.index} has invalid attempt count")
        if state == "completed":
            completed.append(shard.index)
        elif state == "running":
            if include_running and attempts < limit:
                eligible.append(shard.index)
            else:
                active.append(shard.index)
        elif attempts >= limit:
            exhausted.append(shard.index)
        else:
            eligible.append(shard.index)
    return RetryPlan(
        plan.plan_directory,
        tuple(eligible),
        tuple(completed),
        tuple(exhausted),
        tuple(active),
    )


def run_shard(
    path: str | Path,
    shard_index: int,
    registry_factory: RegistryFactory,
    *,
    repository_root: str | Path | None = None,
) -> RunResult | None:
    plan = load_plan(path)
    if shard_index < 0 or shard_index >= len(plan.shards):
        raise SlurmPlanError(f"shard index {shard_index} is outside the plan")
    shard = plan.shards[shard_index]
    status = _read_status(plan.plan_directory, shard_index)
    attempts = status.get("attempts")
    if type(attempts) is not int:
        raise SlurmPlanError(f"shard {shard_index} has invalid status")
    limit = 1 + plan.slurm_config.max_retries
    if attempts >= limit and status.get("state") != "completed":
        raise SlurmPlanError(
            f"shard {shard_index} exhausted its {limit} permitted attempt(s)"
        )
    status.update(
        {
            "state": "running",
            "attempts": attempts + 1,
            "updated_at": utc_now(),
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "error": None,
        }
    )
    _write_status(plan.plan_directory, shard_index, status)
    if not shard.item_ids:
        status.update({"state": "completed", "updated_at": utc_now()})
        _write_status(plan.plan_directory, shard_index, status)
        return None
    shard_root = plan.plan_directory / "shards" / f"{shard_index:04d}" / "runs"
    config = replace(
        plan.run_config,
        output_root=str(shard_root),
        run_policy=RunPolicy.CREATE,
    )
    destination = shard_root / str(config.run_id)
    try:
        registry = registry_factory(config)
        result = RunEngine(
            registry,
            repository_root=repository_root or plan.slurm_config.working_directory,
        ).run(
            config,
            resume=destination.exists(),
            retry_failures=destination.exists(),
            item_ids=shard.item_ids,
        )
        state = {
            RunStatus.COMPLETED: "completed",
            RunStatus.COMPLETED_WITH_ITEM_FAILURES: "completed_with_item_failures",
            RunStatus.INTERRUPTED: "interrupted",
            RunStatus.FAILED: "failed",
        }.get(result.status, "failed")
        _annotate_shard_manifest(plan, shard_index, result.run_directory)
        status.update({"state": state, "updated_at": utc_now()})
        _write_status(plan.plan_directory, shard_index, status)
        return result
    except BaseException as error:
        status.update(
            {
                "state": (
                    "interrupted"
                    if isinstance(error, KeyboardInterrupt)
                    else "failed"
                ),
                "updated_at": utc_now(),
                "error": (
                    f"{type(error).__module__}.{type(error).__qualname__}: {error}"
                ),
            }
        )
        _write_status(plan.plan_directory, shard_index, status)
        raise


def install_term_handler() -> Callable[[int, object], None] | int | None:
    previous = signal.getsignal(signal.SIGTERM)

    def interrupt(signum: int, frame: object) -> None:
        del signum, frame
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    return previous


def restore_term_handler(previous: Callable[[int, object], None] | int | None) -> None:
    signal.signal(signal.SIGTERM, previous)


def merge_shards(
    path: str | Path,
    registry: AdapterRegistry,
    *,
    replace_matching_plan: bool = False,
    repository_root: str | Path | None = None,
) -> Path:
    plan = load_plan(path)
    base = plan.run_config
    descriptor = registry.describe(base.model_id, base.model_variant)
    expected_requests = base.requests()
    expected = {
        (request.request_id, sample_index): request
        for request in expected_requests
        for sample_index in range(request.num_samples)
    }
    predictions: list[PredictionRecord] = []
    failures: list[FailureRecord] = []
    shard_manifests: list[Mapping[str, object]] = []
    states: dict[int, str] = {}
    for shard in plan.shards:
        status = _read_status(plan.plan_directory, shard.index)
        state = status.get("state")
        if type(state) is not str:
            raise SlurmMergeError(f"shard {shard.index} has invalid state")
        states[shard.index] = state
        run_path = (
            plan.plan_directory
            / "shards"
            / f"{shard.index:04d}"
            / "runs"
            / str(base.run_id)
        )
        if not run_path.exists():
            continue
        directory = RunDirectory.open(run_path)
        shard_manifest = directory.read_manifest()
        _validate_shard_manifest(plan, shard.index, shard.item_ids, shard_manifest)
        shard_manifests.append(shard_manifest)
        predictions.extend(directory.read_predictions(recover_trailing_record=True))
        failures.extend(
            replace(
                record,
                traceback_log_reference=(
                    f"slurm/shards/{shard.index:04d}/run/"
                    f"{record.traceback_log_reference}"
                ),
            )
            for record in directory.read_failures(recover_trailing_record=True)
        )

    keys: set[tuple[str, int]] = set()
    sample_ids: set[str] = set()
    for record in predictions:
        key = (record.request_id, record.sample_index)
        request = expected.get(key)
        if request is None or request.item_id != record.item_id:
            raise SlurmMergeError(
                f"prediction {record.sample_id!r} does not belong to the base run"
            )
        if key in keys or record.sample_id in sample_ids:
            raise SlurmMergeError(
                "duplicate prediction request/sample identity across shards"
            )
        keys.add(key)
        sample_ids.add(record.sample_id)
    request_ids = {request.request_id: request for request in expected_requests}
    for record in failures:
        request = request_ids.get(record.request_id)
        if request is None or request.item_id != record.item_id:
            raise SlurmMergeError("failure record does not belong to the base run")

    complete_requests = {
        request.request_id
        for request in expected_requests
        if all(
            (request.request_id, index) in keys
            for index in range(request.num_samples)
        )
    }
    all_shards_completed = all(state == "completed" for state in states.values())
    complete = all_shards_completed and keys == set(expected)
    status = RunStatus.COMPLETED if complete else RunStatus.COMPLETED_WITH_ITEM_FAILURES
    manifest = initial_manifest(
        base,
        descriptor,
        repository_root=repository_root or plan.slurm_config.working_directory,
    )
    manifest.update(
        {
            "started_at": _earliest_started_at(shard_manifests),
            "ended_at": utc_now(),
            "status": status.value,
            "counts": {
                "total": len(expected_requests),
                "succeeded": len(complete_requests),
                "failed": len(expected_requests) - len(complete_requests),
                "skipped": 0,
            },
        }
    )
    actual: dict[str, list[int]] = {}
    for record in predictions:
        actual.setdefault(record.request_id, []).append(record.seed)
    seeds = manifest.get("seeds")
    if isinstance(seeds, dict):
        seeds["actual_per_request"] = {
            request_id: sorted(values) for request_id, values in sorted(actual.items())
        }
    plan_state = _read_json(plan.plan_directory / "plan.json")
    array_job_id = plan_state.get("array_job_id") or os.environ.get(
        "AVRW_ARRAY_JOB_ID"
    )
    merge_job_id = plan_state.get("merge_job_id") or os.environ.get(
        "SLURM_JOB_ID"
    )
    plan_state["array_job_id"] = array_job_id
    plan_state["merge_job_id"] = merge_job_id
    _atomic_write_json(plan.plan_directory / "plan.json", plan_state)
    manifest["slurm"] = {
        "job_id": merge_job_id,
        "array_job_id": array_job_id,
        "array_task_id": None,
        "partition": plan.slurm_config.partition,
        "node_list": os.environ.get("SLURM_JOB_NODELIST"),
        "plan_hash": plan.plan_hash,
        "plan_directory": str(plan.plan_directory),
        "shard_count": len(plan.shards),
        "shard_states": {str(index): value for index, value in sorted(states.items())},
        "submission_attempts": plan_state.get("submission_attempts", 0),
        "submissions": plan_state.get("submissions", []),
    }
    warnings = manifest.get("warnings")
    if isinstance(warnings, list):
        warnings.append(
            "Per-shard native artifacts remain in the preserved Slurm plan directory."
        )
        if not complete:
            warnings.append(
                "Finalization is partial; inspect slurm.shard_states and "
                "per-shard evidence."
            )
    predictions.sort(
        key=lambda record: (
            record.request_id,
            record.sample_index,
            record.sample_id,
        )
    )
    failures.sort(
        key=lambda record: (
            record.request_id,
            record.attempt,
            record.occurred_at,
        )
    )
    _install_merged_run(
        plan,
        manifest,
        predictions,
        failures,
        replace_matching_plan=replace_matching_plan,
    )
    plan_state["final_run_status"] = status.value
    plan_state["finalized_at"] = utc_now()
    _atomic_write_json(plan.plan_directory / "plan.json", plan_state)
    return plan.final_run_directory


def _install_merged_run(
    plan: SlurmPlan,
    manifest: Mapping[str, object],
    predictions: list[PredictionRecord],
    failures: list[FailureRecord],
    *,
    replace_matching_plan: bool,
) -> None:
    destination = plan.final_run_directory
    output_root = destination.parent
    output_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.merging-", dir=output_root
        )
    )
    installed = False
    backup: Path | None = None
    try:
        for name in RunDirectory.REQUIRED_DIRECTORIES:
            (temporary / name).mkdir()
        _atomic_write_json(
            temporary / RunDirectory.CONFIG_NAME, plan.run_config.to_dict()
        )
        _atomic_write_json(temporary / RunDirectory.MANIFEST_NAME, manifest)
        _atomic_write_text(
            temporary / RunDirectory.PREDICTIONS_NAME,
            "".join(f"{dumps_prediction(record)}\n" for record in predictions),
        )
        _atomic_write_text(
            temporary / RunDirectory.FAILURES_NAME,
            "".join(
                json.dumps(
                    record.to_dict(),
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n"
                for record in failures
            ),
        )
        _atomic_write_text(
            temporary / "logs" / "events.jsonl",
            json.dumps(
                {
                    "timestamp": utc_now(),
                    "level": "INFO",
                    "event": "slurm_merge_finished",
                    "message": "Slurm shard artifacts were finalized.",
                    "plan_hash": plan.plan_hash,
                },
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n",
        )
        _copy_evidence(plan.plan_directory, temporary / "slurm")
        if destination.exists():
            if not replace_matching_plan:
                raise SlurmMergeError(
                    f"final run directory already exists: {destination}"
                )
            existing = _read_json(destination / "manifest.json")
            slurm = existing.get("slurm")
            if (
                not isinstance(slurm, Mapping)
                or slurm.get("plan_hash") != plan.plan_hash
            ):
                raise SlurmMergeError(
                    "refusing to replace a final run from a different Slurm plan"
                )
            backups = plan.plan_directory / "finalization-backups"
            backups.mkdir(exist_ok=True)
            backup = backups / f"{destination.name}-{uuid.uuid4().hex}"
            os.replace(destination, backup)
        try:
            os.replace(temporary, destination)
        except Exception:
            if backup is not None and not destination.exists():
                os.replace(backup, destination)
                backup = None
            raise
        installed = True
    finally:
        if not installed and temporary.exists():
            shutil.rmtree(temporary)


def _copy_evidence(source: Path, destination: Path) -> None:
    names = ["plan.json", "run-config.json", "slurm-config.json", "merge.sbatch"]
    names.extend(path.name for path in sorted(source.glob("array*.sbatch")))
    for name in names:
        path = source / name
        if path.exists():
            shutil.copy2(path, destination / name)
    output_shards = destination / "shards"
    output_shards.mkdir()
    for shard_directory in sorted((source / "shards").iterdir()):
        target = output_shards / shard_directory.name
        target.mkdir()
        for name in ("shard.json", "status.json"):
            path = shard_directory / name
            if path.exists():
                shutil.copy2(path, target / name)
        run_root = shard_directory / "runs"
        if not run_root.is_dir():
            continue
        for run_directory in run_root.iterdir():
            run_target = target / "run"
            run_target.mkdir(exist_ok=True)
            for name in ("config.yaml", "manifest.json", "failures.jsonl"):
                path = run_directory / name
                if path.exists():
                    shutil.copy2(path, run_target / name)
            logs = run_directory / "logs"
            if logs.is_dir():
                shutil.copytree(logs, run_target / "logs", symlinks=True)


def _validate_shard_manifest(
    plan: SlurmPlan,
    shard_index: int,
    item_ids: tuple[str, ...],
    manifest: Mapping[str, object],
) -> None:
    if (
        manifest.get("schema_version") != 1
        or manifest.get("run_id") != plan.run_config.run_id
    ):
        raise SlurmMergeError(f"shard {shard_index} has incompatible run identity")
    model = manifest.get("model")
    dataset = manifest.get("dataset")
    if not isinstance(model, Mapping) or model.get("id") != plan.run_config.model_id:
        raise SlurmMergeError(f"shard {shard_index} has incompatible model identity")
    if (
        not isinstance(dataset, Mapping)
        or dataset.get("id") != plan.run_config.dataset_id
    ):
        raise SlurmMergeError(f"shard {shard_index} has incompatible dataset identity")
    selected_item_ids = (
        dataset.get("selected_item_ids") if isinstance(dataset, Mapping) else None
    )
    if (
        not isinstance(selected_item_ids, list)
        or any(type(item_id) is not str for item_id in selected_item_ids)
        or sorted(selected_item_ids) != sorted(item_ids)
    ):
        raise SlurmMergeError(f"shard {shard_index} has incompatible item membership")
    slurm = manifest.get("slurm")
    if not isinstance(slurm, Mapping) or slurm.get("plan_hash") != plan.plan_hash:
        raise SlurmMergeError(f"shard {shard_index} is missing matching plan metadata")
    if slurm.get("base_configuration_hash") != plan.run_config.configuration_hash:
        raise SlurmMergeError(
            f"shard {shard_index} has incompatible base configuration"
        )
    expected_config = replace(
        plan.run_config,
        output_root=str(
            plan.plan_directory / "shards" / f"{shard_index:04d}" / "runs"
        ),
        run_policy=RunPolicy.CREATE,
    )
    if manifest.get("configuration_hash") != expected_config.configuration_hash:
        raise SlurmMergeError(
            f"shard {shard_index} has incompatible execution configuration"
        )
    resolved = manifest.get("resolved_configuration")
    try:
        resolved_config = RunConfig.from_dict(resolved, base_dir=plan.plan_directory)
    except Exception as error:
        raise SlurmMergeError(
            f"shard {shard_index} has invalid resolved configuration: {error}"
        ) from error
    if resolved_config.configuration_hash != expected_config.configuration_hash:
        raise SlurmMergeError(
            f"shard {shard_index} resolved configuration was modified"
        )


def _annotate_shard_manifest(plan: SlurmPlan, index: int, path: Path) -> None:
    directory = RunDirectory.open(path)
    manifest = directory.read_manifest()
    slurm = manifest.get("slurm")
    if not isinstance(slurm, dict):
        slurm = {}
        manifest["slurm"] = slurm
    slurm.update(
        {
            "plan_hash": plan.plan_hash,
            "base_configuration_hash": plan.run_config.configuration_hash,
            "shard_index": index,
            "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        }
    )
    directory.write_manifest(manifest)


def _earliest_started_at(manifests: list[Mapping[str, object]]) -> str | None:
    values = [
        value
        for manifest in manifests
        if type(value := manifest.get("started_at")) is str
    ]
    return min(values) if values else None


def _read_status(directory: Path, index: int) -> dict[str, object]:
    return _read_json(directory / "shards" / f"{index:04d}" / "status.json")


def _write_status(directory: Path, index: int, status: Mapping[str, object]) -> None:
    _atomic_write_json(directory / "shards" / f"{index:04d}" / "status.json", status)


def _next_retry_script_name(directory: Path) -> str:
    existing = tuple(directory.glob("array.retry-*.sbatch"))
    return f"array.retry-{len(existing) + 1}.sbatch"


def _update_plan_submission(
    directory: Path,
    *,
    array_job_id: str,
    merge_job_id: str | None,
    selected_shards: tuple[int, ...],
) -> None:
    state = _read_json(directory / "plan.json")
    history = state.get("submissions")
    if not isinstance(history, list):
        history = []
        state["submissions"] = history
    if merge_job_id is None:
        attempts = state.get("submission_attempts")
        attempt = (attempts if type(attempts) is int else 0) + 1
        state["submission_attempts"] = attempt
        history.append(
            {
                "attempt": attempt,
                "array_job_id": array_job_id,
                "merge_job_id": None,
                "selected_shards": list(selected_shards),
                "submitted_at": utc_now(),
            }
        )
    elif history and isinstance(history[-1], dict):
        history[-1]["merge_job_id"] = merge_job_id
    state["array_job_id"] = array_job_id
    if merge_job_id is not None:
        state["merge_job_id"] = merge_job_id
    state["submitted_at"] = utc_now()
    _atomic_write_json(directory / "plan.json", state)


def _submit(runner: CommandRunner, command: tuple[str, ...]) -> str:
    try:
        result = runner(command)
    except OSError as error:
        raise SlurmSubmissionError(f"could not execute sbatch: {error}") from error
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or "unknown error"
        raise SlurmSubmissionError(f"sbatch failed: {message}")
    value = result.stdout.strip().split(";", 1)[0]
    if not value.isdigit():
        raise SlurmSubmissionError(
            f"sbatch returned invalid job ID {result.stdout.strip()!r}"
        )
    return value


def _run_command(command: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=False, capture_output=True, text=True)


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SlurmPlanError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise SlurmPlanError(f"{path} must contain a JSON object")
    return value


def _atomic_write_json(path: Path, value: object) -> None:
    _atomic_write_text(
        path,
        json.dumps(value, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
    )


def _atomic_write_text(path: Path, value: str, *, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        if executable:
            temporary.chmod(0o755)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
