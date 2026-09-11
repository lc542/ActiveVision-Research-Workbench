
from __future__ import annotations

import importlib
import random
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from activevision_workbench.adapters import ModelAdapter
from activevision_workbench.contracts import InferenceRequest, PredictionRecord
from activevision_workbench.errors import RunExistsError, RunResumeError
from activevision_workbench.registry import AdapterDescriptor, AdapterRegistry
from activevision_workbench.run_artifacts import (
    FailureRecord,
    RunDirectory,
    RunStatus,
    StructuredRunLogger,
    initial_manifest,
    utc_now,
)
from activevision_workbench.run_config import RunConfig, RunPolicy


@dataclass(frozen=True, slots=True)
class DryRunSummary:

    run_id: str
    run_directory: Path
    total: int
    will_run: int
    skipped_completed: int
    skipped_failed: int
    resume: bool

    def to_dict(self) -> dict[str, object]:

        return {
            "run_id": self.run_id,
            "run_directory": str(self.run_directory),
            "total": self.total,
            "will_run": self.will_run,
            "skipped_completed": self.skipped_completed,
            "skipped_failed": self.skipped_failed,
            "resume": self.resume,
        }


@dataclass(frozen=True, slots=True)
class RunResult:

    run_id: str
    run_directory: Path
    status: RunStatus
    total: int
    succeeded: int
    failed: int
    skipped: int


@dataclass(frozen=True, slots=True)
class _Decision:
    to_run: tuple[InferenceRequest, ...]
    skipped_completed: tuple[InferenceRequest, ...]
    skipped_failed: tuple[InferenceRequest, ...]


class RunEngine:

    def __init__(
        self,
        registry: AdapterRegistry,
        *,
        repository_root: str | Path | None = None,
        console: Callable[[str], None] | None = print,
    ) -> None:
        self.registry = registry
        self.repository_root = (
            Path.cwd() if repository_root is None else Path(repository_root)
        )
        self.console = console

    def dry_run(
        self,
        config: RunConfig,
        *,
        resume: bool = False,
        overwrite: bool = False,
        retry_failures: bool = False,
        item_ids: Sequence[str] | None = None,
    ) -> DryRunSummary:

        resolved, resume_mode, overwrite_mode = self._resolve_mode(
            config, resume=resume, overwrite=overwrite
        )
        descriptor, adapter, requests = self._validated_adapter(
            resolved, item_ids=item_ids
        )
        del descriptor
        try:
            destination = Path(resolved.output_root) / _required_run_id(resolved)
            if resume_mode:
                directory = RunDirectory.open(destination)
                predictions, failures = self._resume_records(
                    directory, resolved, requests, recover=False
                )
                decision = _decide(
                    requests,
                    predictions,
                    failures,
                    retry_failures=retry_failures,
                )
            else:
                if destination.exists() and not overwrite_mode:
                    raise RunExistsError(
                        f"run directory already exists: {destination}; use resume "
                        "or explicit overwrite"
                    )
                decision = _Decision(requests, (), ())
            return DryRunSummary(
                run_id=_required_run_id(resolved),
                run_directory=destination,
                total=len(requests),
                will_run=len(decision.to_run),
                skipped_completed=len(decision.skipped_completed),
                skipped_failed=len(decision.skipped_failed),
                resume=resume_mode,
            )
        finally:
            adapter.close()

    def run(
        self,
        config: RunConfig,
        *,
        resume: bool = False,
        overwrite: bool = False,
        retry_failures: bool = False,
        item_ids: Sequence[str] | None = None,
    ) -> RunResult:

        resolved, resume_mode, overwrite_mode = self._resolve_mode(
            config, resume=resume, overwrite=overwrite
        )
        descriptor, adapter, requests = self._validated_adapter(
            resolved, item_ids=item_ids
        )
        directory: RunDirectory | None = None
        logger: StructuredRunLogger | None = None
        manifest: dict[str, object] | None = None
        try:
            destination = Path(resolved.output_root) / _required_run_id(resolved)
            if resume_mode:
                directory = RunDirectory.open(destination)
                predictions, failures = self._resume_records(
                    directory, resolved, requests, recover=True
                )
                manifest = directory.read_manifest()
            else:
                manifest = initial_manifest(
                    resolved,
                    descriptor,
                    repository_root=self.repository_root,
                    selected_item_ids=tuple(request.item_id for request in requests),
                )
                directory = RunDirectory.create(
                    resolved,
                    manifest,
                    overwrite=overwrite_mode,
                )
                predictions = ()
                failures = ()

            logger = StructuredRunLogger(directory, console=self.console)
            decision = _decide(
                requests,
                predictions,
                failures,
                retry_failures=retry_failures,
            )
            prediction_keys = {
                (record.request_id, record.sample_index) for record in predictions
            }
            attempt_counts = _failure_attempts(failures)

            _set_running_manifest(
                manifest,
                resume=resume_mode,
                decision=decision,
            )
            _record_actual_seeds(manifest, predictions)
            _set_counts(manifest, requests, prediction_keys, failures, decision)
            directory.write_manifest(manifest)
            logger.log(
                "INFO",
                "run_started" if not resume_mode else "run_resumed",
                (
                    f"Run {_required_run_id(resolved)}: "
                    f"{len(decision.to_run)} item(s) to execute, "
                    f"{len(decision.skipped_completed)} completed item(s) skipped, "
                    f"{len(decision.skipped_failed)} failed item(s) held."
                ),
                run_id=_required_run_id(resolved),
                decision={
                    "will_run": len(decision.to_run),
                    "skipped_completed": len(decision.skipped_completed),
                    "skipped_failed": len(decision.skipped_failed),
                },
            )

            current_failures = list(failures)
            try:
                for request in decision.to_run:
                    seed_components, seed_warnings = _seed_components(
                        request.base_seed
                    )
                    _merge_seed_components(manifest, seed_components)
                    for warning in seed_warnings:
                        _add_warning(manifest, warning)
                        logger.log("WARNING", "seed_warning", warning)
                    logger.log(
                        "INFO",
                        "item_started",
                        f"Running item {request.item_id}.",
                        request_id=request.request_id,
                        item_id=request.item_id,
                        base_seed=request.base_seed,
                    )
                    try:
                        produced = adapter.predict(
                            request, run_id=_required_run_id(resolved)
                        )
                    except Exception as error:
                        attempt = attempt_counts.get(request.request_id, 0) + 1
                        attempt_counts[request.request_id] = attempt
                        failure = directory.append_failure(
                            request, error, attempt=attempt
                        )
                        current_failures.append(failure)
                        logger.log(
                            "ERROR",
                            "item_failed",
                            f"Item {request.item_id} failed: {error}",
                            request_id=request.request_id,
                            item_id=request.item_id,
                            exception_type=failure.exception_type,
                            traceback_log_reference=(
                                failure.traceback_log_reference
                            ),
                        )
                    else:
                        missing = tuple(
                            record
                            for record in produced
                            if (record.request_id, record.sample_index)
                            not in prediction_keys
                        )
                        if missing:
                            directory.append_predictions(missing)
                        prediction_keys.update(
                            (record.request_id, record.sample_index)
                            for record in missing
                        )
                        _record_actual_seeds(manifest, produced)
                        logger.log(
                            "INFO",
                            "item_completed",
                            f"Completed item {request.item_id}.",
                            request_id=request.request_id,
                            item_id=request.item_id,
                            samples_written=len(missing),
                        )
                    _set_counts(
                        manifest,
                        requests,
                        prediction_keys,
                        tuple(current_failures),
                        decision,
                    )
                    directory.write_manifest(manifest)
            except KeyboardInterrupt:
                _finish_manifest(manifest, RunStatus.INTERRUPTED)
                _set_counts(
                    manifest,
                    requests,
                    prediction_keys,
                    tuple(current_failures),
                    decision,
                )
                directory.write_manifest(manifest)
                logger.log(
                    "WARNING",
                    "run_interrupted",
                    "Run interrupted; completed JSONL records remain resumable.",
                )
                return _result_from_manifest(directory, manifest)

            succeeded = _completed_request_ids(requests, prediction_keys)
            failed = _current_failed_request_ids(
                requests, prediction_keys, tuple(current_failures)
            )
            status = (
                RunStatus.COMPLETED_WITH_ITEM_FAILURES
                if failed
                else RunStatus.COMPLETED
            )
            _finish_manifest(manifest, status)
            _set_counts(
                manifest,
                requests,
                prediction_keys,
                tuple(current_failures),
                decision,
            )
            directory.write_manifest(manifest)
            logger.log(
                "INFO",
                "run_finished",
                (
                    f"Run finished with status {status.value}: "
                    f"{len(succeeded)} succeeded, {len(failed)} failed."
                ),
                status=status.value,
            )
            return _result_from_manifest(directory, manifest)
        except Exception as error:
            if directory is not None and manifest is not None:
                try:
                    _add_warning(
                        manifest,
                        f"Run-level failure: {type(error).__name__}: {error}",
                    )
                    _finish_manifest(manifest, RunStatus.FAILED)
                    directory.write_manifest(manifest)
                    if logger is not None:
                        logger.log(
                            "ERROR",
                            "run_failed",
                            f"Run failed: {error}",
                            exception_type=(
                                f"{type(error).__module__}."
                                f"{type(error).__qualname__}"
                            ),
                        )
                except Exception:
                    pass
            raise
        finally:
            adapter.close()

    def _resolve_mode(
        self,
        config: RunConfig,
        *,
        resume: bool,
        overwrite: bool,
    ) -> tuple[RunConfig, bool, bool]:
        if resume and overwrite:
            raise RunResumeError("resume and overwrite are mutually exclusive")
        if resume:
            resume_mode, overwrite_mode = True, False
        elif overwrite:
            resume_mode, overwrite_mode = False, True
        else:
            resume_mode = config.run_policy is RunPolicy.RESUME
            overwrite_mode = config.run_policy is RunPolicy.OVERWRITE
        if config.run_id is None:
            if resume_mode:
                raise RunResumeError(
                    "resume requires a concrete run_id; use the resolved "
                    "config.yaml stored in the run directory"
                )
            config = config.with_run_id(_new_run_id())
        return config, resume_mode, overwrite_mode

    def _validated_adapter(
        self,
        config: RunConfig,
        *,
        item_ids: Sequence[str] | None = None,
    ) -> tuple[AdapterDescriptor, ModelAdapter, tuple[InferenceRequest, ...]]:
        descriptor = self.registry.describe(config.model_id, config.model_variant)
        requests = _filter_requests(config.requests(), item_ids)
        adapter = self.registry.create(config.model_id, config.model_variant)
        try:
            for request in requests:
                adapter.validate_request(request)
        except Exception:
            adapter.close()
            raise
        return descriptor, adapter, requests

    def _resume_records(
        self,
        directory: RunDirectory,
        config: RunConfig,
        requests: tuple[InferenceRequest, ...],
        *,
        recover: bool,
    ) -> tuple[tuple[PredictionRecord, ...], tuple[FailureRecord, ...]]:
        manifest = directory.read_manifest()
        if manifest.get("schema_version") != 1:
            raise RunResumeError("resume requires manifest schema_version 1")
        if manifest.get("run_id") != config.run_id:
            raise RunResumeError("resume run_id does not match the stored manifest")
        stored_hash = manifest.get("configuration_hash")
        if stored_hash != config.configuration_hash:
            raise RunResumeError(
                "resume configuration is incompatible: configuration hash "
                f"changed from {stored_hash!r} to {config.configuration_hash!r}"
            )
        predictions = directory.read_predictions(recover_trailing_record=recover)
        failures = directory.read_failures(recover_trailing_record=recover)
        _validate_resume_predictions(config, requests, predictions)
        _validate_resume_failures(requests, failures)
        return predictions, failures


def _decide(
    requests: tuple[InferenceRequest, ...],
    predictions: tuple[PredictionRecord, ...],
    failures: tuple[FailureRecord, ...],
    *,
    retry_failures: bool,
) -> _Decision:
    keys = {(record.request_id, record.sample_index) for record in predictions}
    failed_ids = {record.request_id for record in failures}
    to_run: list[InferenceRequest] = []
    skipped_completed: list[InferenceRequest] = []
    skipped_failed: list[InferenceRequest] = []
    for request in requests:
        expected = {
            (request.request_id, sample_index)
            for sample_index in range(request.num_samples)
        }
        if expected.issubset(keys):
            skipped_completed.append(request)
        elif request.request_id in failed_ids and not retry_failures:
            skipped_failed.append(request)
        else:
            to_run.append(request)
    return _Decision(
        tuple(to_run), tuple(skipped_completed), tuple(skipped_failed)
    )


def _validate_resume_predictions(
    config: RunConfig,
    requests: tuple[InferenceRequest, ...],
    predictions: tuple[PredictionRecord, ...],
) -> None:
    request_by_id = {request.request_id: request for request in requests}
    keys: set[tuple[str, int]] = set()
    sample_ids: set[str] = set()
    for line_index, record in enumerate(predictions, start=1):
        if record.run_id != config.run_id:
            raise RunResumeError(
                f"predictions.jsonl line {line_index} has run_id "
                f"'{record.run_id}', expected '{config.run_id}'"
            )
        request = request_by_id.get(record.request_id)
        if request is None:
            raise RunResumeError(
                f"predictions.jsonl line {line_index} references an unknown request"
            )
        if record.sample_index >= request.num_samples:
            raise RunResumeError(
                f"predictions.jsonl line {line_index} has out-of-range sample_index"
            )
        expected_fields = {
            "model_id": (record.model.model_id, request.model_id),
            "model_variant": (record.model.model_variant, request.model_variant),
            "dataset_id": (record.dataset_id, request.dataset_id),
            "dataset_split": (record.dataset_split, request.dataset_split),
            "item_id": (record.item_id, request.item_id),
            "observer_id": (record.observer_id, request.observer_id),
            "task_text": (record.task_text, request.task_text),
            "target_description": (
                record.target_description,
                request.target_description,
            ),
        }
        mismatch = next(
            (
                field
                for field, (actual, expected) in expected_fields.items()
                if actual != expected
            ),
            None,
        )
        if mismatch is not None:
            raise RunResumeError(
                f"predictions.jsonl line {line_index} has incompatible {mismatch}"
            )
        key = (record.request_id, record.sample_index)
        if key in keys or record.sample_id in sample_ids:
            raise RunResumeError(
                "predictions.jsonl contains duplicate request/sample identity"
            )
        keys.add(key)
        sample_ids.add(record.sample_id)


def _validate_resume_failures(
    requests: tuple[InferenceRequest, ...],
    failures: tuple[FailureRecord, ...],
) -> None:
    items_by_request = {
        request.request_id: request.item_id for request in requests
    }
    for line_index, failure in enumerate(failures, start=1):
        expected_item = items_by_request.get(failure.request_id)
        if expected_item is None:
            raise RunResumeError(
                f"failures.jsonl line {line_index} references an unknown request"
            )
        if failure.item_id != expected_item:
            raise RunResumeError(
                f"failures.jsonl line {line_index} has incompatible item_id"
            )


def _set_running_manifest(
    manifest: dict[str, object],
    *,
    resume: bool,
    decision: _Decision,
) -> None:
    manifest["status"] = RunStatus.RUNNING.value
    if manifest.get("started_at") is None:
        manifest["started_at"] = utc_now()
    manifest["ended_at"] = None
    resume_data = _mapping_field(manifest, "resume")
    attempts = resume_data.get("attempts", 0)
    if type(attempts) is not int:
        attempts = 0
    if resume:
        attempts += 1
    resume_data["attempts"] = attempts
    resume_data["last_decision"] = {
        "timestamp": utc_now(),
        "resume": resume,
        "will_run": len(decision.to_run),
        "skipped_completed": len(decision.skipped_completed),
        "skipped_failed": len(decision.skipped_failed),
        "request_ids_to_run": [request.request_id for request in decision.to_run],
    }


def _finish_manifest(manifest: dict[str, object], status: RunStatus) -> None:
    manifest["status"] = status.value
    manifest["ended_at"] = utc_now()


def _set_counts(
    manifest: dict[str, object],
    requests: tuple[InferenceRequest, ...],
    prediction_keys: set[tuple[str, int]],
    failures: tuple[FailureRecord, ...],
    decision: _Decision,
) -> None:
    succeeded = _completed_request_ids(requests, prediction_keys)
    failed = _current_failed_request_ids(requests, prediction_keys, failures)
    counts = _mapping_field(manifest, "counts")
    counts.update(
        {
            "total": len(requests),
            "succeeded": len(succeeded),
            "failed": len(failed),
            "skipped": len(decision.skipped_failed),
        }
    )


def _completed_request_ids(
    requests: tuple[InferenceRequest, ...],
    keys: set[tuple[str, int]],
) -> set[str]:
    return {
        request.request_id
        for request in requests
        if all(
            (request.request_id, index) in keys
            for index in range(request.num_samples)
        )
    }


def _current_failed_request_ids(
    requests: tuple[InferenceRequest, ...],
    keys: set[tuple[str, int]],
    failures: tuple[FailureRecord, ...],
) -> set[str]:
    completed = _completed_request_ids(requests, keys)
    selected = {request.request_id for request in requests}
    return {
        record.request_id
        for record in failures
        if record.request_id in selected and record.request_id not in completed
    }


def _record_actual_seeds(
    manifest: dict[str, object], records: tuple[PredictionRecord, ...]
) -> None:
    seeds = _mapping_field(manifest, "seeds")
    actual = seeds.get("actual_per_request")
    if not isinstance(actual, dict):
        actual = {}
        seeds["actual_per_request"] = actual
    grouped: dict[str, list[tuple[int, int]]] = {}
    for record in records:
        grouped.setdefault(record.request_id, []).append(
            (record.sample_index, record.seed)
        )
    for request_id, values in grouped.items():
        actual[request_id] = [seed for _, seed in sorted(values)]


def _merge_seed_components(
    manifest: dict[str, object], components: tuple[str, ...]
) -> None:
    seeds = _mapping_field(manifest, "seeds")
    current = seeds.get("seeded_components")
    if not isinstance(current, list):
        current = []
    seeds["seeded_components"] = sorted(set(current).union(components))


def _seed_components(seed: int) -> tuple[tuple[str, ...], tuple[str, ...]]:
    seeded = ["python.random"]
    warnings: list[str] = []
    random.seed(seed)
    for module_name in ("numpy", "torch"):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        except Exception as error:
            warnings.append(f"Could not import {module_name} for seeding: {error}")
            continue
        try:
            if module_name == "numpy":
                module.random.seed(seed % (2**32))
                seeded.append("numpy.random")
            else:
                module.manual_seed(seed)
                seeded.append("torch")
                cuda = getattr(module, "cuda", None)
                if cuda is not None and callable(
                    getattr(cuda, "manual_seed_all", None)
                ):
                    cuda.manual_seed_all(seed)
                    seeded.append("torch.cuda")
        except Exception as error:
            warnings.append(f"Could not seed {module_name}: {error}")
    return tuple(seeded), tuple(warnings)


def _failure_attempts(failures: tuple[FailureRecord, ...]) -> dict[str, int]:
    attempts: dict[str, int] = {}
    for failure in failures:
        attempts[failure.request_id] = max(
            attempts.get(failure.request_id, 0), failure.attempt
        )
    return attempts


def _mapping_field(
    manifest: dict[str, object], field: str
) -> dict[str, object]:
    value = manifest.get(field)
    if not isinstance(value, dict):
        value = {}
        manifest[field] = value
    return value


def _add_warning(manifest: dict[str, object], warning: str) -> None:
    value = manifest.get("warnings")
    warnings = value if isinstance(value, list) else []
    if warning not in warnings:
        warnings.append(warning)
    manifest["warnings"] = warnings


def _result_from_manifest(
    directory: RunDirectory, manifest: Mapping[str, object]
) -> RunResult:
    status = RunStatus(str(manifest["status"]))
    counts = manifest.get("counts")
    if not isinstance(counts, Mapping):  # pragma: no cover - engine owns it
        raise RunResumeError("manifest counts are unavailable")
    return RunResult(
        run_id=str(manifest["run_id"]),
        run_directory=directory.path,
        status=status,
        total=int(counts["total"]),
        succeeded=int(counts["succeeded"]),
        failed=int(counts["failed"]),
        skipped=int(counts["skipped"]),
    )


def _required_run_id(config: RunConfig) -> str:
    if config.run_id is None:  # pragma: no cover - resolved before use
        raise RunResumeError("run ID has not been resolved")
    return config.run_id


def _filter_requests(
    requests: tuple[InferenceRequest, ...],
    item_ids: Sequence[str] | None,
) -> tuple[InferenceRequest, ...]:
    if item_ids is None:
        return requests
    selected = tuple(item_ids)
    if len(selected) != len(set(selected)):
        raise RunResumeError("shard item IDs must not contain duplicates")
    available = {request.item_id for request in requests}
    unknown = next((item_id for item_id in selected if item_id not in available), None)
    if unknown is not None:
        raise RunResumeError(f"shard references unknown item ID '{unknown}'")
    wanted = set(selected)
    return tuple(request for request in requests if request.item_id in wanted)


def _new_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"run-{timestamp}-{uuid.uuid4().hex[:12]}"
