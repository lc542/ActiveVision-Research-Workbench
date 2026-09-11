from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
import shutil
import statistics
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist

from activevision_workbench.contracts import CapabilitySet, PredictionRecord
from activevision_workbench.datasets import (
    COCOSearch18DatasetAdapter,
    DatasetAdapter,
    DatasetItem,
    GroundTruthScanpath,
    JsonDatasetAdapter,
    OSIEDatasetAdapter,
)
from activevision_workbench.errors import (
    ContractValidationError,
    EvaluationConfigurationError,
    EvaluationDataError,
    EvaluationOutputError,
    RunResumeError,
)
from activevision_workbench.evaluation.config import (
    EvaluationProtocol,
    ExistingOutputPolicy,
)
from activevision_workbench.evaluation.metrics import (
    MetricContext,
    MetricDefinition,
    MetricDirection,
    MetricInvalidData,
    MetricNotApplicable,
    MetricRegistry,
    MetricScope,
    MetricStatus,
    MetricValue,
    create_default_metric_registry,
)
from activevision_workbench.run_artifacts import RunDirectory


SAMPLE_FIELDS = (
    "schema_version",
    "run_id",
    "dataset_id",
    "dataset_version",
    "dataset_split",
    "item_id",
    "request_id",
    "sample_id",
    "sample_index",
    "seed",
    "observer_id",
    "observer_group",
    "task_text",
    "target_description",
    "metric_name",
    "metric_version",
    "metric_kind",
    "status",
    "value",
    "unit",
    "ground_truth_scanpath_ids",
    "aggregation",
    "message",
    "artifact_reference",
    "explanation_text",
    "native_artifacts",
    "stopping_reason",
)

ITEM_FIELDS = (
    "schema_version",
    "run_id",
    "dataset_id",
    "dataset_version",
    "dataset_split",
    "item_id",
    "request_id",
    "observer_id",
    "observer_group",
    "task_text",
    "target_description",
    "metric_name",
    "metric_version",
    "metric_kind",
    "status",
    "mean",
    "std",
    "count",
    "ci_lower",
    "ci_upper",
    "unit",
    "aggregation",
    "source_sample_ids",
    "message",
    "artifact_reference",
)

OBSERVER_FIELDS = (
    "schema_version",
    "run_id",
    "dataset_id",
    "dataset_version",
    "dataset_split",
    "observer_id",
    "observer_group",
    "task_text",
    "metric_name",
    "metric_version",
    "metric_kind",
    "status",
    "mean",
    "std",
    "count_items",
    "unit",
    "aggregation",
    "source_item_keys",
    "message",
)

OUTPUT_FILES = (
    "protocol.yaml",
    "metric_versions.json",
    "per_item.csv",
    "per_sample.csv",
    "per_observer.csv",
    "summary.json",
    "warnings.jsonl",
)


@dataclass(frozen=True, slots=True)
class EvaluationPlan:
    run_directory: Path
    output_directory: Path
    protocol_hash: str
    run_id: str
    prediction_count: int
    item_count: int
    metrics: tuple[Mapping[str, object], ...]
    existing_output: ExistingOutputPolicy

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "run_directory": str(self.run_directory),
            "output_directory": str(self.output_directory),
            "protocol_hash": self.protocol_hash,
            "run_id": self.run_id,
            "prediction_count": self.prediction_count,
            "item_count": self.item_count,
            "metrics": [dict(metric) for metric in self.metrics],
            "existing_output": self.existing_output.value,
            "dry_run": True,
        }


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    run_id: str
    output_directory: Path
    status: str
    sample_row_count: int
    item_row_count: int
    observer_row_count: int
    warning_count: int


@dataclass(frozen=True, slots=True)
class _PreparedEvaluation:
    run_directory: RunDirectory
    manifest: Mapping[str, object]
    capabilities: CapabilitySet
    predictions: tuple[PredictionRecord, ...]
    dataset: DatasetAdapter
    items: Mapping[str, DatasetItem]
    metrics: tuple[tuple[MetricDefinition, Mapping[str, object]], ...]


class EvaluationEngine:
    def __init__(self, registry: MetricRegistry | None = None) -> None:
        self.registry = create_default_metric_registry() if registry is None else registry

    def dry_run(self, protocol: EvaluationProtocol) -> EvaluationPlan:
        prepared = self._prepare(protocol)
        plans = []
        for definition, parameters in prepared.metrics:
            counts = Counter()
            if definition.scope is MetricScope.SAMPLE:
                for prediction in prepared.predictions:
                    item = prepared.items[prediction.item_id]
                    truth = _matching_ground_truth(prediction, item)
                    status, _ = _eligibility(
                        definition,
                        (prediction,),
                        item,
                        truth,
                        prepared.capabilities,
                        protocol,
                    )
                    counts[status.value] += 1
            else:
                for predictions in _prediction_groups(prepared.predictions).values():
                    item = prepared.items[predictions[0].item_id]
                    truth = _matching_ground_truth(predictions[0], item)
                    status, _ = _eligibility(
                        definition,
                        predictions,
                        item,
                        truth,
                        prepared.capabilities,
                        protocol,
                    )
                    counts[status.value] += 1
            overall = _overall_status(counts)
            plans.append(
                {
                    "name": definition.name,
                    "implementation_version": definition.implementation_version,
                    "scope": definition.scope.value,
                    "kind": definition.kind.value,
                    "status": overall.value,
                    "eligibility_counts": dict(sorted(counts.items())),
                    "resolved_parameters": dict(parameters),
                }
            )
        return EvaluationPlan(
            run_directory=prepared.run_directory.path,
            output_directory=Path(protocol.output_directory),
            protocol_hash=protocol.protocol_hash,
            run_id=_manifest_string(prepared.manifest, "run_id"),
            prediction_count=len(prepared.predictions),
            item_count=len(prepared.items),
            metrics=tuple(plans),
            existing_output=protocol.existing_output,
        )

    def evaluate(self, protocol: EvaluationProtocol) -> EvaluationResult:
        prepared = self._prepare(protocol)
        output = Path(protocol.output_directory)
        existing_rows, existing_warnings = _prepare_output(output, protocol)
        warnings: list[dict[str, object]] = list(existing_warnings)
        if prepared.manifest.get("status") == "completed_with_item_failures":
            warnings.append(
                {
                    "schema_version": 1,
                    "code": "source_run_item_failures",
                    "metric_name": None,
                    "item_id": None,
                    "sample_ids": [],
                    "message": "source run completed with item failures; only existing predictions were evaluated",
                }
            )
        sample_rows = dict(existing_rows)
        run_id = _manifest_string(prepared.manifest, "run_id")

        for prediction in prepared.predictions:
            item = prepared.items[prediction.item_id]
            truth = _matching_ground_truth(prediction, item)
            for definition, parameters in prepared.metrics:
                if definition.scope is not MetricScope.SAMPLE:
                    continue
                key = (prediction.request_id, prediction.sample_id, definition.name)
                if key in sample_rows:
                    continue
                status, message = _eligibility(
                    definition,
                    (prediction,),
                    item,
                    truth,
                    prepared.capabilities,
                    protocol,
                )
                value = None
                artifact = None
                unit = definition.unit
                if status is MetricStatus.APPLICABLE:
                    context = MetricContext(
                        predictions=(prediction,),
                        item=item,
                        ground_truth=truth,
                        run_directory=prepared.run_directory.path,
                        output_directory=output,
                        protocol=protocol,
                        parameters=parameters,
                        artifact_stem=_artifact_stem(
                            prediction.item_id,
                            prediction.request_id,
                            prediction.observer_id,
                            prediction.task_text,
                        ),
                    )
                    status, message, metric_value = _compute_metric(
                        definition, context, warnings
                    )
                    if metric_value is not None:
                        value = metric_value.value
                        artifact = metric_value.artifact_reference
                        unit = metric_value.unit
                row = _sample_row(
                    prediction,
                    item,
                    truth,
                    definition,
                    status,
                    value,
                    unit,
                    message,
                    artifact,
                )
                sample_rows[key] = row

        ordered_samples = tuple(
            sample_rows[key] for key in sorted(sample_rows, key=_sample_key_sort)
        )
        item_rows = list(_aggregate_sample_rows(ordered_samples, protocol))
        direct_item_rows = []
        for predictions in _prediction_groups(prepared.predictions).values():
            prediction = predictions[0]
            item = prepared.items[prediction.item_id]
            truth = _matching_ground_truth(prediction, item)
            for definition, parameters in prepared.metrics:
                if definition.scope is not MetricScope.ITEM:
                    continue
                status, message = _eligibility(
                    definition,
                    predictions,
                    item,
                    truth,
                    prepared.capabilities,
                    protocol,
                )
                value = None
                artifact = None
                unit = definition.unit
                if status is MetricStatus.APPLICABLE:
                    context = MetricContext(
                        predictions=predictions,
                        item=item,
                        ground_truth=truth,
                        run_directory=prepared.run_directory.path,
                        output_directory=output,
                        protocol=protocol,
                        parameters=parameters,
                        artifact_stem=_artifact_stem(
                            prediction.item_id,
                            prediction.request_id,
                            prediction.observer_id,
                            prediction.task_text,
                        ),
                    )
                    status, message, metric_value = _compute_metric(
                        definition, context, warnings
                    )
                    if metric_value is not None:
                        value = metric_value.value
                        artifact = metric_value.artifact_reference
                        unit = metric_value.unit
                direct_item_rows.append(
                    _direct_item_row(
                        predictions,
                        item,
                        definition,
                        status,
                        value,
                        unit,
                        message,
                        artifact,
                        protocol,
                    )
                )
        item_rows.extend(direct_item_rows)
        ordered_items = tuple(sorted(item_rows, key=_item_row_sort))
        observer_rows = _aggregate_observer_rows(ordered_items, protocol)
        warnings = list(_deduplicated_warnings(warnings))
        summary = _build_summary(
            protocol,
            prepared,
            ordered_samples,
            ordered_items,
            observer_rows,
            warnings,
        )
        metric_versions = {
            "schema_version": 1,
            "metrics": [
                definition.to_dict(resolved_parameters=parameters)
                for definition, parameters in prepared.metrics
            ],
        }
        _atomic_json(output / "protocol.yaml", protocol.to_dict())
        _atomic_json(output / "metric_versions.json", metric_versions)
        _atomic_csv(output / "per_sample.csv", SAMPLE_FIELDS, ordered_samples)
        _atomic_csv(output / "per_item.csv", ITEM_FIELDS, ordered_items)
        _atomic_csv(output / "per_observer.csv", OBSERVER_FIELDS, observer_rows)
        _atomic_json(output / "summary.json", summary)
        _atomic_jsonl(output / "warnings.jsonl", warnings)
        status = str(summary["status"])
        return EvaluationResult(
            run_id=run_id,
            output_directory=output,
            status=status,
            sample_row_count=len(ordered_samples),
            item_row_count=len(ordered_items),
            observer_row_count=len(observer_rows),
            warning_count=len(warnings),
        )

    def _prepare(self, protocol: EvaluationProtocol) -> _PreparedEvaluation:
        if not isinstance(protocol, EvaluationProtocol):
            raise EvaluationConfigurationError(
                "protocol must be an EvaluationProtocol"
            )
        try:
            run_directory = RunDirectory.open(protocol.run_directory)
            manifest = run_directory.read_manifest()
        except RunResumeError as error:
            raise EvaluationDataError(str(error)) from error
        status = manifest.get("status")
        if status not in {"completed", "completed_with_item_failures"}:
            raise EvaluationDataError(
                f"evaluation requires a completed run; manifest status is {status!r}"
            )
        capabilities = _manifest_capabilities(manifest)
        predictions = tuple(
            sorted(
                run_directory.read_predictions(),
                key=lambda record: (
                    record.item_id,
                    record.request_id,
                    record.sample_index,
                    record.sample_id,
                ),
            )
        )
        if not predictions:
            raise EvaluationDataError("completed run contains no prediction records")
        run_id = _manifest_string(manifest, "run_id")
        for prediction in predictions:
            if prediction.run_id != run_id:
                raise EvaluationDataError(
                    f"prediction '{prediction.sample_id}' run_id does not match manifest"
                )
        dataset = _dataset_adapter(protocol)
        metadata = dataset.metadata
        if metadata.dataset_id != protocol.dataset.dataset_id:
            raise EvaluationDataError(
                "ground-truth adapter dataset ID does not match the protocol"
            )
        if metadata.dataset_version != protocol.dataset.version:
            raise EvaluationDataError(
                "ground-truth adapter version does not match the protocol"
            )
        if protocol.dataset.split not in metadata.available_splits:
            raise EvaluationDataError(
                f"ground-truth split '{protocol.dataset.split}' is unavailable"
            )
        _validate_manifest_dataset(manifest, protocol)
        items = {}
        representatives = {}
        for prediction in predictions:
            representatives.setdefault(prediction.item_id, prediction)
        for item_id, prediction in sorted(representatives.items()):
            items[item_id] = _load_ground_truth_item(
                dataset,
                prediction,
                split=protocol.dataset.split,
            )
        resolved_metrics = []
        names = {request.name for request in protocol.metrics}
        best = protocol.sample_aggregation.best_of_n
        missing_best = sorted(set(best.metrics) - names)
        if missing_best:
            raise EvaluationConfigurationError(
                f"best-of-N metric '{missing_best[0]}' is not requested"
            )
        for request in protocol.metrics:
            definition = self.registry.get(request.name)
            parameters = definition.resolve_parameters(request.parameters)
            if request.name in best.metrics and definition.direction is MetricDirection.INTERPRET_ONLY:
                raise EvaluationConfigurationError(
                    f"best-of-N metric '{request.name}' has no better direction"
                )
            resolved_metrics.append((definition, parameters))
        return _PreparedEvaluation(
            run_directory=run_directory,
            manifest=manifest,
            capabilities=capabilities,
            predictions=predictions,
            dataset=dataset,
            items=items,
            metrics=tuple(resolved_metrics),
        )


def _dataset_adapter(protocol: EvaluationProtocol) -> DatasetAdapter:
    config = protocol.dataset
    options = _plain_options(config.options)
    if config.adapter == "json":
        _check_options(options, {"annotation_name"}, config.adapter)
        return JsonDatasetAdapter(config.root, **options)
    if config.adapter == "osie":
        _check_options(
            options,
            {"split_definition_path", "duration_unit", "clip_out_of_bounds"},
            config.adapter,
        )
        if "split_definition_path" in options:
            options["split_definition_path"] = _dataset_relative_path(
                config.root, options["split_definition_path"], "split_definition_path"
            )
        return OSIEDatasetAdapter(
            config.root,
            dataset_version=config.version,
            **options,
        )
    _check_options(
        options,
        {"split_definition", "duration_unit", "clip_out_of_bounds"},
        config.adapter,
    )
    return COCOSearch18DatasetAdapter(
        config.root,
        dataset_version=config.version,
        **options,
    )


def _load_ground_truth_item(
    dataset: DatasetAdapter,
    prediction: PredictionRecord,
    *,
    split: str,
) -> DatasetItem:
    candidates = [prediction.item_id]
    observer = prediction.observer_id
    if observer is not None and prediction.item_id.endswith(f":{observer}"):
        source_id = prediction.item_id[: -(len(observer) + 1)]
        if source_id:
            candidates.append(source_id)
    errors = []
    for candidate in candidates:
        try:
            return dataset.get_item(candidate, split=split)
        except Exception as error:
            errors.append(str(error))
    raise EvaluationDataError(
        f"cannot load ground truth for run item '{prediction.item_id}': "
        + "; ".join(errors)
    )


def _plain_options(options: Mapping[str, object]) -> dict[str, object]:
    result = {}
    for key, value in options.items():
        if isinstance(value, tuple):
            result[key] = list(value)
        elif isinstance(value, Mapping):
            result[key] = _plain_options(value)
        else:
            result[key] = value
    return result


def _check_options(
    options: Mapping[str, object], allowed: set[str], adapter: str
) -> None:
    unknown = sorted(set(options) - allowed)
    if unknown:
        raise EvaluationConfigurationError(
            f"dataset adapter '{adapter}' option '{unknown[0]}' is unknown"
        )


def _dataset_relative_path(root: str, value: object, field_name: str) -> str:
    if type(value) is not str or not value:
        raise EvaluationConfigurationError(
            f"dataset.options.{field_name} must be a non-empty string"
        )
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(root) / path
    return str(path.resolve(strict=False))


def _manifest_capabilities(manifest: Mapping[str, object]) -> CapabilitySet:
    model = manifest.get("model")
    if not isinstance(model, Mapping):
        raise EvaluationDataError("run manifest model metadata is missing")
    raw = model.get("capabilities")
    try:
        return CapabilitySet.from_dict(raw)
    except ContractValidationError as error:
        raise EvaluationDataError(
            f"run manifest capabilities are invalid: {error}"
        ) from error


def _validate_manifest_dataset(
    manifest: Mapping[str, object], protocol: EvaluationProtocol
) -> None:
    dataset = manifest.get("dataset")
    if not isinstance(dataset, Mapping):
        raise EvaluationDataError("run manifest dataset metadata is missing")
    expected = {
        "id": protocol.dataset.dataset_id,
        "version": protocol.dataset.version,
        "split": protocol.dataset.split,
    }
    for field, value in expected.items():
        if dataset.get(field) != value:
            raise EvaluationDataError(
                f"run manifest dataset {field} does not match evaluation protocol"
            )


def _manifest_string(manifest: Mapping[str, object], field_name: str) -> str:
    value = manifest.get(field_name)
    if type(value) is not str or value == "":
        raise EvaluationDataError(
            f"run manifest field '{field_name}' must be a non-empty string"
        )
    return value


def _prediction_groups(
    predictions: Sequence[PredictionRecord],
) -> Mapping[tuple[str, str, str, str], tuple[PredictionRecord, ...]]:
    groups = defaultdict(list)
    for prediction in predictions:
        key = (
            prediction.item_id,
            prediction.request_id,
            prediction.observer_id or "",
            prediction.task_text or "",
        )
        groups[key].append(prediction)
    return {
        key: tuple(sorted(values, key=lambda value: (value.sample_index, value.sample_id)))
        for key, values in sorted(groups.items())
    }


def _matching_ground_truth(
    prediction: PredictionRecord,
    item: DatasetItem,
) -> tuple[GroundTruthScanpath, ...]:
    scanpaths = item.scanpaths
    if prediction.target_description is not None and item.target_label is not None:
        requested_target = prediction.target_description.strip().casefold()
        item_target = item.target_label.strip().casefold()
        if requested_target != item_target:
            return ()
        targeted = tuple(
            scanpath
            for scanpath in scanpaths
            if scanpath.target_label is None
            or scanpath.target_label.strip().casefold() == requested_target
        )
        if targeted:
            scanpaths = targeted
    elif prediction.task_text is not None:
        task_values = {
            value
            for value in (
                item.task_text,
                *(scanpath.task_text for scanpath in scanpaths),
            )
            if value is not None
        }
        if len(task_values) > 1:
            scanpaths = tuple(
                scanpath
                for scanpath in scanpaths
                if (
                    scanpath.task_text
                    if scanpath.task_text is not None
                    else item.task_text
                )
                == prediction.task_text
            )
    if prediction.observer_id is not None:
        scanpaths = tuple(
            scanpath
            for scanpath in scanpaths
            if scanpath.observer_id == prediction.observer_id
        )
    return tuple(sorted(scanpaths, key=lambda value: value.scanpath_id))


def _task_type(
    protocol: EvaluationProtocol,
    prediction: PredictionRecord,
    item: DatasetItem,
) -> str:
    configured = protocol.dataset.task_type
    if configured != "auto":
        return configured
    if item.target_boxes:
        present = item.target_metadata.get("target_present")
        if present is not False:
            return "visual_search_target_present"
    if prediction.task_text is not None:
        return "instruction_conditioned"
    return "free_viewing"


def _eligibility(
    definition: MetricDefinition,
    predictions: tuple[PredictionRecord, ...],
    item: DatasetItem,
    truth: tuple[GroundTruthScanpath, ...],
    capabilities: CapabilitySet,
    protocol: EvaluationProtocol,
) -> tuple[MetricStatus, str]:
    missing_capabilities = sorted(
        capability.value
        for capability in definition.compatible_capabilities
        if not capabilities.supports(capability)
    )
    if missing_capabilities:
        return (
            MetricStatus.NOT_APPLICABLE,
            "model capability is absent: " + ", ".join(missing_capabilities),
        )
    task = _task_type(protocol, predictions[0], item)
    if "any" not in definition.supported_task_types and task not in definition.supported_task_types:
        return (
            MetricStatus.NOT_APPLICABLE,
            f"metric does not support task type '{task}'",
        )
    missing_dependencies = tuple(
        name
        for name in definition.optional_dependencies
        if not _dependency_available(name)
    )
    if missing_dependencies:
        return (
            MetricStatus.UNAVAILABLE,
            "missing metric dependency: " + ", ".join(missing_dependencies),
        )
    for field_name in definition.required_prediction_fields:
        status = _prediction_field_status(field_name, predictions)
        if status is not None:
            return status
    for field_name in definition.required_ground_truth_fields:
        status = _ground_truth_field_status(field_name, item, truth)
        if status is not None:
            return status
    return MetricStatus.APPLICABLE, ""


def _dependency_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _prediction_field_status(
    field_name: str,
    predictions: tuple[PredictionRecord, ...],
) -> tuple[MetricStatus, str] | None:
    if field_name == "scanpath_record":
        return None
    if field_name == "fixations":
        if any(not prediction.fixations for prediction in predictions):
            return MetricStatus.INVALID, "prediction fixations are absent"
        return None
    if field_name == "timestamps":
        if any(
            not prediction.fixations
            or any(fixation.timestamp_s is None for fixation in prediction.fixations)
            for prediction in predictions
        ):
            return MetricStatus.INVALID, "complete prediction timestamps are absent"
        return None
    if field_name == "durations":
        if any(
            not prediction.fixations
            or any(fixation.duration_s is None for fixation in prediction.fixations)
            for prediction in predictions
        ):
            return MetricStatus.INVALID, "complete prediction durations are absent"
        return None
    if field_name == "saliency_artifact":
        if any(prediction.saliency_artifact is None for prediction in predictions):
            return MetricStatus.INVALID, "prediction saliency artifact is absent"
        return None
    if field_name == "multiple_nonempty_scanpaths":
        if len(predictions) < 2:
            return MetricStatus.NOT_APPLICABLE, "fewer than two samples are present"
        if any(not prediction.fixations for prediction in predictions):
            return MetricStatus.INVALID, "one or more sample endpoints are absent"
        return None
    return MetricStatus.INVALID, f"unknown required prediction field '{field_name}'"


def _ground_truth_field_status(
    field_name: str,
    item: DatasetItem,
    truth: tuple[GroundTruthScanpath, ...],
) -> tuple[MetricStatus, str] | None:
    if field_name == "target_boxes":
        if not item.target_boxes:
            return MetricStatus.NOT_APPLICABLE, "target-present boxes are absent"
        return None
    if field_name == "fixations":
        if not truth or any(not scanpath.fixations for scanpath in truth):
            return MetricStatus.INVALID, "matching ground-truth fixations are absent"
        return None
    if field_name == "timestamps":
        if not truth or any(
            any(fixation.timestamp_s is None for fixation in scanpath.fixations)
            for scanpath in truth
        ):
            return MetricStatus.NOT_APPLICABLE, "ground-truth timing is absent"
        return None
    if field_name == "durations":
        if not truth or any(
            any(fixation.duration_s is None for fixation in scanpath.fixations)
            for scanpath in truth
        ):
            return MetricStatus.NOT_APPLICABLE, "ground-truth timing is absent"
        return None
    return MetricStatus.INVALID, f"unknown required ground-truth field '{field_name}'"


def _overall_status(counts: Counter) -> MetricStatus:
    for status in (
        MetricStatus.APPLICABLE,
        MetricStatus.ERROR,
        MetricStatus.INVALID,
        MetricStatus.UNAVAILABLE,
        MetricStatus.NOT_APPLICABLE,
    ):
        if counts[status.value]:
            return status
    return MetricStatus.NOT_APPLICABLE


def _compute_metric(
    definition: MetricDefinition,
    context: MetricContext,
    warnings: list[dict[str, object]],
) -> tuple[MetricStatus, str, MetricValue | None]:
    try:
        value = definition.evaluator(context)
    except MetricNotApplicable as error:
        return MetricStatus.NOT_APPLICABLE, str(error), None
    except MetricInvalidData as error:
        return MetricStatus.INVALID, str(error), None
    except Exception as error:
        message = f"{type(error).__name__}: {error}"
        warnings.append(
            {
                "schema_version": 1,
                "code": "metric_computation_failed",
                "metric_name": definition.name,
                "item_id": context.item.item_id,
                "sample_ids": sorted(
                    prediction.sample_id for prediction in context.predictions
                ),
                "message": message,
            }
        )
        return MetricStatus.ERROR, message, None
    for warning in value.warnings:
        warnings.append(
            {
                "schema_version": 1,
                "code": "metric_preprocessing_warning",
                "metric_name": definition.name,
                "item_id": context.item.item_id,
                "sample_ids": sorted(
                    prediction.sample_id for prediction in context.predictions
                ),
                "message": warning,
            }
        )
    return MetricStatus.APPLICABLE, "", value


def _sample_row(
    prediction: PredictionRecord,
    item: DatasetItem,
    truth: tuple[GroundTruthScanpath, ...],
    definition: MetricDefinition,
    status: MetricStatus,
    value: float | None,
    unit: str,
    message: str,
    artifact: str | None,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "run_id": prediction.run_id,
        "dataset_id": prediction.dataset_id,
        "dataset_version": item.dataset_version,
        "dataset_split": prediction.dataset_split,
        "item_id": prediction.item_id,
        "request_id": prediction.request_id,
        "sample_id": prediction.sample_id,
        "sample_index": prediction.sample_index,
        "seed": prediction.seed,
        "observer_id": prediction.observer_id or "",
        "observer_group": _observer_group(prediction, item),
        "task_text": prediction.task_text if prediction.task_text is not None else "",
        "target_description": (
            prediction.target_description
            if prediction.target_description is not None
            else ""
        ),
        "metric_name": definition.name,
        "metric_version": definition.implementation_version,
        "metric_kind": definition.kind.value,
        "status": status.value,
        "value": "" if value is None else _number_text(value),
        "unit": unit,
        "ground_truth_scanpath_ids": _json_text(
            [scanpath.scanpath_id for scanpath in truth]
        ),
        "aggregation": (
            "none" if not truth else "macro_matching_ground_truth_scanpaths"
        ),
        "message": message,
        "artifact_reference": artifact or "",
        "explanation_text": (
            prediction.explanation_text
            if prediction.explanation_text is not None
            else ""
        ),
        "native_artifacts": _json_text(
            [artifact.to_dict() for artifact in prediction.native_artifacts]
        ),
        "stopping_reason": (
            ""
            if prediction.stopping_reason is None
            else prediction.stopping_reason.value
        ),
    }


def _observer_group(prediction: PredictionRecord, item: DatasetItem) -> str:
    if prediction.observer_id is None:
        return "anonymous"
    if prediction.observer_id in item.observer_ids:
        return "known"
    return "unknown"


def _aggregate_sample_rows(
    rows: Sequence[Mapping[str, object]],
    protocol: EvaluationProtocol,
) -> tuple[dict[str, object], ...]:
    groups = defaultdict(list)
    for row in rows:
        key = tuple(
            str(row[field])
            for field in (
                "run_id",
                "dataset_id",
                "dataset_version",
                "dataset_split",
                "item_id",
                "request_id",
                "observer_id",
                "observer_group",
                "task_text",
                "target_description",
                "metric_name",
                "metric_version",
                "metric_kind",
                "unit",
            )
        )
        groups[key].append(row)
    output = []
    for key, current in sorted(groups.items()):
        values = [
            float(row["value"])
            for row in current
            if row["status"] == MetricStatus.APPLICABLE.value
            and str(row["value"]) != ""
        ]
        status, message = _aggregate_status(current, values)
        mean, std, lower, upper = _statistics(values, protocol)
        output.append(
            {
                "schema_version": 1,
                "run_id": key[0],
                "dataset_id": key[1],
                "dataset_version": key[2],
                "dataset_split": key[3],
                "item_id": key[4],
                "request_id": key[5],
                "observer_id": key[6],
                "observer_group": key[7],
                "task_text": key[8],
                "target_description": key[9],
                "metric_name": key[10],
                "metric_version": key[11],
                "metric_kind": key[12],
                "status": status.value,
                "mean": _optional_number_text(mean),
                "std": _optional_number_text(std),
                "count": len(values),
                "ci_lower": _optional_number_text(lower),
                "ci_upper": _optional_number_text(upper),
                "unit": key[13],
                "aggregation": "mean_std_across_preserved_samples",
                "source_sample_ids": _json_text(
                    sorted(str(row["sample_id"]) for row in current)
                ),
                "message": message,
                "artifact_reference": "",
            }
        )
    return tuple(output)


def _aggregate_status(
    rows: Sequence[Mapping[str, object]], values: Sequence[float]
) -> tuple[MetricStatus, str]:
    if values:
        return MetricStatus.APPLICABLE, ""
    statuses = Counter(str(row["status"]) for row in rows)
    messages = sorted({str(row["message"]) for row in rows if row["message"]})
    return _overall_status(statuses), "; ".join(messages)


def _statistics(
    values: Sequence[float], protocol: EvaluationProtocol
) -> tuple[float | None, float | None, float | None, float | None]:
    if not values:
        return None, None, None, None
    mean = statistics.fmean(values)
    sample = protocol.sample_aggregation.standard_deviation == "sample"
    if sample and len(values) < 2:
        std = None
    elif sample:
        std = statistics.stdev(values)
    else:
        std = statistics.pstdev(values)
    ci = protocol.sample_aggregation.confidence_interval
    if ci.method == "none" or std is None:
        return mean, std, None, None
    z_score = NormalDist().inv_cdf((1.0 + ci.level) / 2.0)
    half_width = z_score * std / math.sqrt(len(values))
    return mean, std, mean - half_width, mean + half_width


def _direct_item_row(
    predictions: tuple[PredictionRecord, ...],
    item: DatasetItem,
    definition: MetricDefinition,
    status: MetricStatus,
    value: float | None,
    unit: str,
    message: str,
    artifact: str | None,
    protocol: EvaluationProtocol,
) -> dict[str, object]:
    prediction = predictions[0]
    values = [] if value is None else [value]
    mean, std, lower, upper = _statistics(values, protocol)
    return {
        "schema_version": 1,
        "run_id": prediction.run_id,
        "dataset_id": prediction.dataset_id,
        "dataset_version": item.dataset_version,
        "dataset_split": prediction.dataset_split,
        "item_id": prediction.item_id,
        "request_id": prediction.request_id,
        "observer_id": prediction.observer_id or "",
        "observer_group": _observer_group(prediction, item),
        "task_text": prediction.task_text or "",
        "target_description": prediction.target_description or "",
        "metric_name": definition.name,
        "metric_version": definition.implementation_version,
        "metric_kind": definition.kind.value,
        "status": status.value,
        "mean": _optional_number_text(mean),
        "std": _optional_number_text(std),
        "count": len(predictions) if status is MetricStatus.APPLICABLE else 0,
        "ci_lower": _optional_number_text(lower),
        "ci_upper": _optional_number_text(upper),
        "unit": unit,
        "aggregation": "item_level_across_preserved_samples",
        "source_sample_ids": _json_text(
            sorted(prediction.sample_id for prediction in predictions)
        ),
        "message": message,
        "artifact_reference": artifact or "",
    }


def _aggregate_observer_rows(
    rows: Sequence[Mapping[str, object]],
    protocol: EvaluationProtocol,
) -> tuple[dict[str, object], ...]:
    groups = defaultdict(list)
    for row in rows:
        key = tuple(
            str(row[field])
            for field in (
                "run_id",
                "dataset_id",
                "dataset_version",
                "dataset_split",
                "observer_id",
                "observer_group",
                "task_text",
                "metric_name",
                "metric_version",
                "metric_kind",
                "unit",
            )
        )
        groups[key].append(row)
    output = []
    for key, current in sorted(groups.items()):
        values = [
            float(row["mean"])
            for row in current
            if row["status"] == MetricStatus.APPLICABLE.value
            and str(row["mean"]) != ""
        ]
        status, message = _aggregate_status(current, values)
        mean, std, _, _ = _statistics(values, protocol)
        output.append(
            {
                "schema_version": 1,
                "run_id": key[0],
                "dataset_id": key[1],
                "dataset_version": key[2],
                "dataset_split": key[3],
                "observer_id": key[4],
                "observer_group": key[5],
                "task_text": key[6],
                "metric_name": key[7],
                "metric_version": key[8],
                "metric_kind": key[9],
                "status": status.value,
                "mean": _optional_number_text(mean),
                "std": _optional_number_text(std),
                "count_items": len(values),
                "unit": key[10],
                "aggregation": "macro_across_items",
                "source_item_keys": _json_text(
                    sorted(
                        {
                            (str(row["item_id"]), str(row["request_id"]))
                            for row in current
                        }
                    )
                ),
                "message": message,
            }
        )
    return tuple(output)


def _build_summary(
    protocol: EvaluationProtocol,
    prepared: _PreparedEvaluation,
    sample_rows: Sequence[Mapping[str, object]],
    item_rows: Sequence[Mapping[str, object]],
    observer_rows: Sequence[Mapping[str, object]],
    warnings: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    groups = defaultdict(list)
    for row in observer_rows:
        key = (
            str(row["task_text"]),
            str(row["observer_group"]),
            str(row["metric_name"]),
            str(row["metric_version"]),
            str(row["metric_kind"]),
            str(row["unit"]),
        )
        groups[key].append(row)
    summaries = []
    for key, current in sorted(groups.items()):
        values = [
            float(row["mean"])
            for row in current
            if row["status"] == MetricStatus.APPLICABLE.value
            and str(row["mean"]) != ""
        ]
        status, message = _aggregate_status(current, values)
        mean, std, _, _ = _statistics(values, protocol)
        summaries.append(
            {
                "task_text": key[0],
                "observer_group": key[1],
                "metric_name": key[2],
                "metric_version": key[3],
                "metric_kind": key[4],
                "status": status.value,
                "mean": mean,
                "std": std,
                "count_observers": len(values),
                "unit": key[5],
                "aggregation": "macro_across_observers",
                "source_observers": sorted(
                    {
                        str(row["observer_id"])
                        for row in current
                    }
                ),
                "message": message,
            }
        )
    micro = []
    if protocol.observer_aggregation.include_micro:
        micro = _micro_summaries(item_rows, protocol)
    best = _best_of_n(sample_rows, prepared, protocol)
    status_counts = Counter(str(row["status"]) for row in sample_rows)
    status_counts.update(str(row["status"]) for row in item_rows)
    errors = status_counts[MetricStatus.ERROR.value]
    manifest_status = prepared.manifest.get("status")
    explanations = [
        {
            "sample_id": row["sample_id"],
            "item_id": row["item_id"],
            "task_text": row["task_text"],
            "explanation_text": row["explanation_text"],
            "native_artifacts": json.loads(str(row["native_artifacts"])),
        }
        for row in sample_rows
        if str(row["explanation_text"]) != ""
        and str(row["metric_name"])
        == _first_sample_metric_name(prepared.metrics)
    ]
    return {
        "schema_version": 1,
        "protocol_hash": protocol.protocol_hash,
        "run_id": _manifest_string(prepared.manifest, "run_id"),
        "status": "completed_with_metric_errors" if errors else "completed",
        "source_run_status": manifest_status,
        "input_prediction_count": len(prepared.predictions),
        "evaluated_item_count": len(prepared.items),
        "sample_row_count": len(sample_rows),
        "item_row_count": len(item_rows),
        "observer_row_count": len(observer_rows),
        "metric_status_counts": dict(sorted(status_counts.items())),
        "sample_aggregation": protocol.sample_aggregation.to_dict(),
        "observer_aggregation": protocol.observer_aggregation.to_dict(),
        "temporal_alignment": protocol.temporal_alignment.to_dict(),
        "coordinate_conversion": protocol.coordinate_conversion.to_dict(),
        "groups": summaries,
        "micro_groups": micro,
        "secondary_best_of_n": best,
        "text_evaluation": {
            "status": "not_applicable",
            "reason": "no official text metric and reference text were configured",
            "preserved_outputs": explanations,
        },
        "ground_truth_manifest": prepared.dataset.create_manifest().to_dict(),
        "warnings_count": len(warnings),
    }


def _micro_summaries(
    item_rows: Sequence[Mapping[str, object]], protocol: EvaluationProtocol
) -> list[dict[str, object]]:
    groups = defaultdict(list)
    for row in item_rows:
        if row["status"] != MetricStatus.APPLICABLE.value or str(row["mean"]) == "":
            continue
        key = (
            str(row["task_text"]),
            str(row["observer_group"]),
            str(row["metric_name"]),
            str(row["metric_version"]),
            str(row["unit"]),
        )
        groups[key].append(float(row["mean"]))
    output = []
    for key, values in sorted(groups.items()):
        mean, std, _, _ = _statistics(values, protocol)
        output.append(
            {
                "task_text": key[0],
                "observer_group": key[1],
                "metric_name": key[2],
                "metric_version": key[3],
                "status": "applicable",
                "mean": mean,
                "std": std,
                "count_items": len(values),
                "unit": key[4],
                "aggregation": "pooled_micro_across_items",
            }
        )
    return output


def _best_of_n(
    sample_rows: Sequence[Mapping[str, object]],
    prepared: _PreparedEvaluation,
    protocol: EvaluationProtocol,
) -> dict[str, object]:
    config = protocol.sample_aggregation.best_of_n
    if not config.enabled:
        return {
            "enabled": False,
            "label": "secondary_best_of_n",
            "results": [],
        }
    definitions = {definition.name: definition for definition, _ in prepared.metrics}
    groups = defaultdict(list)
    for row in sample_rows:
        if row["metric_name"] not in config.metrics:
            continue
        if row["status"] != MetricStatus.APPLICABLE.value or str(row["value"]) == "":
            continue
        key = (
            str(row["item_id"]),
            str(row["request_id"]),
            str(row["observer_id"]),
            str(row["task_text"]),
            str(row["metric_name"]),
        )
        groups[key].append(row)
    results = []
    for key, current in sorted(groups.items()):
        direction = definitions[key[4]].direction
        chooser = max if direction is MetricDirection.HIGHER else min
        selected = chooser(current, key=lambda row: float(row["value"]))
        results.append(
            {
                "label": "secondary_best_of_n",
                "item_id": key[0],
                "request_id": key[1],
                "observer_id": key[2],
                "task_text": key[3],
                "metric_name": key[4],
                "sample_id": selected["sample_id"],
                "value": float(selected["value"]),
                "unit": selected["unit"],
            }
        )
    return {"enabled": True, "label": "secondary_best_of_n", "results": results}


def _first_sample_metric_name(
    metrics: Sequence[tuple[MetricDefinition, Mapping[str, object]]]
) -> str:
    for definition, _ in metrics:
        if definition.scope is MetricScope.SAMPLE:
            return definition.name
    return ""


def _prepare_output(
    output: Path, protocol: EvaluationProtocol
) -> tuple[
    dict[tuple[str, str, str], dict[str, object]],
    tuple[dict[str, object], ...],
]:
    output.mkdir(parents=True, exist_ok=True)
    children = tuple(output.iterdir())
    policy = protocol.existing_output
    if policy is ExistingOutputPolicy.CREATE and children:
        raise EvaluationOutputError(
            f"evaluation output is not empty: {output}; use overwrite or resume"
        )
    if policy is ExistingOutputPolicy.OVERWRITE:
        for name in OUTPUT_FILES:
            path = output / name
            if path.is_file():
                path.unlink()
        artifacts = output / "artifacts"
        if artifacts.exists():
            if not artifacts.is_dir():
                raise EvaluationOutputError(
                    f"evaluation artifact path is not a directory: {artifacts}"
                )
            shutil.rmtree(artifacts)
    existing = {}
    warnings = ()
    if policy is ExistingOutputPolicy.RESUME:
        protocol_path = output / "protocol.yaml"
        if not protocol_path.is_file():
            raise EvaluationOutputError(
                f"resume requires an existing protocol: {protocol_path}"
            )
        stored = EvaluationProtocol.from_file(protocol_path)
        if stored.protocol_hash != protocol.protocol_hash:
            raise EvaluationOutputError(
                "resume protocol does not match the existing scientific protocol"
            )
        sample_path = output / "per_sample.csv"
        if sample_path.is_file():
            for row in _read_csv(sample_path, SAMPLE_FIELDS):
                key = (
                    str(row["request_id"]),
                    str(row["sample_id"]),
                    str(row["metric_name"]),
                )
                if key in existing:
                    raise EvaluationOutputError(
                        f"duplicate existing per-sample row: {key}"
                    )
                existing[key] = row
        warning_path = output / "warnings.jsonl"
        if warning_path.is_file():
            warnings = _read_warning_jsonl(warning_path)
    (output / "artifacts").mkdir(exist_ok=True)
    return existing, warnings


def _read_csv(
    path: Path, fields: Sequence[str]
) -> tuple[dict[str, object], ...]:
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != tuple(fields):
                raise EvaluationOutputError(
                    f"existing CSV schema does not match: {path}"
                )
            return tuple(dict(row) for row in reader)
    except OSError as error:
        raise EvaluationOutputError(f"cannot read existing CSV '{path}': {error}")


def _read_warning_jsonl(path: Path) -> tuple[dict[str, object], ...]:
    records = []
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if line.strip() == "":
                    raise EvaluationOutputError(
                        f"{path}: line {line_number}: blank warning record"
                    )
                value = json.loads(line, parse_constant=_bad_json_constant)
                if not isinstance(value, dict):
                    raise EvaluationOutputError(
                        f"{path}: line {line_number}: warning must be an object"
                    )
                records.append(value)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise EvaluationOutputError(
            f"cannot read existing warnings '{path}': {error}"
        ) from error
    return tuple(records)


def _atomic_csv(
    path: Path,
    fields: Sequence[str],
    rows: Sequence[Mapping[str, object]],
) -> None:
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary_name = stream.name
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass


def _atomic_json(path: Path, value: object) -> None:
    _atomic_text(
        path,
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    text = "".join(
        json.dumps(
            row,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
        for row in sorted(rows, key=_warning_sort)
    )
    _atomic_text(path, text)


def _atomic_text(path: Path, text: str) -> None:
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary_name = stream.name
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass


def _artifact_stem(
    item_id: str,
    request_id: str,
    observer_id: str | None,
    task_text: str | None,
) -> str:
    raw = "\x1f".join((item_id, request_id, observer_id or "", task_text or ""))
    readable = "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in item_id
    ).strip(".")[:48]
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    return f"{readable or 'item'}-{digest}"


def _json_text(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _number_text(value: float) -> str:
    return format(value, ".17g")


def _optional_number_text(value: float | None) -> str:
    return "" if value is None else _number_text(value)


def _sample_key_sort(key: tuple[str, str, str]) -> tuple[str, str, str]:
    return key


def _item_row_sort(row: Mapping[str, object]) -> tuple[str, ...]:
    return tuple(
        str(row[field])
        for field in (
            "item_id",
            "request_id",
            "observer_group",
            "observer_id",
            "task_text",
            "metric_name",
        )
    )


def _warning_sort(row: Mapping[str, object]) -> tuple[str, ...]:
    return (
        str(row.get("metric_name") or ""),
        str(row.get("item_id") or ""),
        _json_text(row.get("sample_ids", [])),
        str(row.get("code") or ""),
        str(row.get("message") or ""),
    )


def _deduplicated_warnings(
    rows: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    unique = {}
    for row in rows:
        key = _json_text(row)
        unique[key] = dict(row)
    return tuple(unique[key] for key in sorted(unique))


def _bad_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON value '{value}'")
