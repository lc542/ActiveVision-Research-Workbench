from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType

from activevision_workbench.contracts import Capability, FixationEvent, PredictionRecord
from activevision_workbench.datasets import DatasetItem, GroundTruthScanpath
from activevision_workbench.errors import (
    EvaluationConfigurationError,
    EvaluationDataError,
)
from activevision_workbench.evaluation.config import EvaluationProtocol


class MetricStatus(str, Enum):
    APPLICABLE = "applicable"
    NOT_APPLICABLE = "not_applicable"
    UNAVAILABLE = "unavailable"
    INVALID = "invalid"
    ERROR = "error"


class MetricScope(str, Enum):
    SAMPLE = "sample"
    ITEM = "item"


class MetricKind(str, Enum):
    SCIENTIFIC = "scientific_metric"
    DIAGNOSTIC = "diagnostic"


class MetricDirection(str, Enum):
    HIGHER = "higher_is_better"
    LOWER = "lower_is_better"
    INTERPRET_ONLY = "interpret_only"


class MetricInvalidData(EvaluationDataError):
    pass


class MetricNotApplicable(EvaluationDataError):
    pass


@dataclass(frozen=True, slots=True)
class ParameterSpec:
    value_type: str
    required: bool = False
    default: object = None
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[object, ...] = ()

    def __post_init__(self) -> None:
        if self.value_type not in {"integer", "number", "boolean", "string"}:
            raise ValueError(f"unsupported parameter type '{self.value_type}'")
        if type(self.required) is not bool:
            raise ValueError("parameter required flag must be boolean")

    def validate(self, value: object, field_name: str) -> object:
        if self.value_type == "integer":
            if type(value) is not int:
                raise EvaluationConfigurationError(
                    f"{field_name} must be an integer"
                )
            numeric = float(value)
        elif self.value_type == "number":
            if type(value) not in {int, float}:
                raise EvaluationConfigurationError(f"{field_name} must be a number")
            numeric = float(value)
            if not math.isfinite(numeric):
                raise EvaluationConfigurationError(f"{field_name} must be finite")
            value = numeric
        elif self.value_type == "boolean":
            if type(value) is not bool:
                raise EvaluationConfigurationError(
                    f"{field_name} must be a boolean"
                )
            numeric = None
        else:
            if type(value) is not str or value == "":
                raise EvaluationConfigurationError(
                    f"{field_name} must be a non-empty string"
                )
            numeric = None
        if numeric is not None:
            if self.minimum is not None and numeric < self.minimum:
                raise EvaluationConfigurationError(
                    f"{field_name} must be >= {self.minimum}"
                )
            if self.maximum is not None and numeric > self.maximum:
                raise EvaluationConfigurationError(
                    f"{field_name} must be <= {self.maximum}"
                )
        if self.choices and value not in self.choices:
            options = ", ".join(repr(choice) for choice in self.choices)
            raise EvaluationConfigurationError(
                f"{field_name} must be one of: {options}"
            )
        return value

    def to_dict(self) -> dict[str, object]:
        return {
            "type": self.value_type,
            "required": self.required,
            "default": self.default,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "choices": list(self.choices),
        }


@dataclass(frozen=True, slots=True)
class MetricValue:
    value: float | None
    unit: str
    artifact_reference: str | None = None
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.value is not None and not math.isfinite(self.value):
            raise MetricInvalidData("metric returned a non-finite value")


@dataclass(frozen=True, slots=True)
class MetricContext:
    predictions: tuple[PredictionRecord, ...]
    item: DatasetItem
    ground_truth: tuple[GroundTruthScanpath, ...]
    run_directory: Path
    output_directory: Path
    protocol: EvaluationProtocol
    parameters: Mapping[str, object]
    artifact_stem: str


MetricEvaluator = Callable[[MetricContext], MetricValue]


@dataclass(frozen=True, slots=True)
class MetricDefinition:
    name: str
    implementation_version: str
    kind: MetricKind
    scope: MetricScope
    required_prediction_fields: tuple[str, ...]
    required_ground_truth_fields: tuple[str, ...]
    supported_task_types: frozenset[str]
    compatible_capabilities: frozenset[Capability]
    parameter_schema: Mapping[str, ParameterSpec]
    aggregation_semantics: str
    direction: MetricDirection
    unit: str
    references: tuple[str, ...]
    evaluator: MetricEvaluator = field(repr=False, compare=False)
    optional_dependencies: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name or not self.implementation_version:
            raise ValueError("metric name and implementation version are required")
        object.__setattr__(
            self,
            "parameter_schema",
            MappingProxyType(dict(self.parameter_schema)),
        )

    def resolve_parameters(self, raw: Mapping[str, object]) -> Mapping[str, object]:
        unknown = sorted(set(raw) - set(self.parameter_schema))
        if unknown:
            raise EvaluationConfigurationError(
                f"metric '{self.name}' parameter '{unknown[0]}' is unknown"
            )
        resolved: dict[str, object] = {}
        for name, spec in self.parameter_schema.items():
            if name in raw:
                value = raw[name]
            elif spec.required:
                raise EvaluationConfigurationError(
                    f"metric '{self.name}' parameter '{name}' is required"
                )
            else:
                value = spec.default
            resolved[name] = spec.validate(
                value,
                f"metrics.{self.name}.parameters.{name}",
            )
        return MappingProxyType(resolved)

    def to_dict(
        self,
        *,
        resolved_parameters: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        return {
            "name": self.name,
            "implementation_version": self.implementation_version,
            "kind": self.kind.value,
            "scope": self.scope.value,
            "required_prediction_fields": list(self.required_prediction_fields),
            "required_ground_truth_fields": list(
                self.required_ground_truth_fields
            ),
            "supported_task_types": sorted(self.supported_task_types),
            "compatible_model_capabilities": sorted(
                capability.value for capability in self.compatible_capabilities
            ),
            "parameter_schema": {
                name: spec.to_dict()
                for name, spec in sorted(self.parameter_schema.items())
            },
            "resolved_parameters": (
                None
                if resolved_parameters is None
                else dict(resolved_parameters)
            ),
            "aggregation_semantics": self.aggregation_semantics,
            "direction": self.direction.value,
            "unit": self.unit,
            "references": list(self.references),
            "optional_dependencies": list(self.optional_dependencies),
        }


class MetricRegistry:
    def __init__(self) -> None:
        self._definitions: dict[str, MetricDefinition] = {}

    def register(self, definition: MetricDefinition) -> None:
        if not isinstance(definition, MetricDefinition):
            raise TypeError("definition must be a MetricDefinition")
        if definition.name in self._definitions:
            raise EvaluationConfigurationError(
                f"metric '{definition.name}' is already registered"
            )
        self._definitions[definition.name] = definition

    def get(self, name: str) -> MetricDefinition:
        try:
            return self._definitions[name]
        except KeyError as error:
            available = ", ".join(sorted(self._definitions))
            raise EvaluationConfigurationError(
                f"unknown metric '{name}'; available: {available}"
            ) from error

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._definitions))


def create_default_metric_registry() -> MetricRegistry:
    registry = MetricRegistry()
    scanpath_capability = frozenset({Capability.PRODUCES_SCANPATHS})
    registry.register(
        MetricDefinition(
            name="fixation_count",
            implementation_version="1.0.0",
            kind=MetricKind.DIAGNOSTIC,
            scope=MetricScope.SAMPLE,
            required_prediction_fields=("scanpath_record",),
            required_ground_truth_fields=(),
            supported_task_types=frozenset({"any"}),
            compatible_capabilities=scanpath_capability,
            parameter_schema={},
            aggregation_semantics="one count per preserved prediction sample",
            direction=MetricDirection.INTERPRET_ONLY,
            unit="fixations",
            references=(),
            evaluator=_fixation_count,
        )
    )
    registry.register(
        MetricDefinition(
            name="scanpath_length_normalized",
            implementation_version="1.0.0",
            kind=MetricKind.DIAGNOSTIC,
            scope=MetricScope.SAMPLE,
            required_prediction_fields=("fixations",),
            required_ground_truth_fields=(),
            supported_task_types=frozenset({"any"}),
            compatible_capabilities=scanpath_capability,
            parameter_schema={},
            aggregation_semantics=(
                "sum of consecutive Euclidean saccade lengths divided by image diagonal"
            ),
            direction=MetricDirection.INTERPRET_ONLY,
            unit="image_diagonals",
            references=(),
            evaluator=_scanpath_length_normalized,
        )
    )
    registry.register(
        MetricDefinition(
            name="timestamp_span_s",
            implementation_version="1.0.0",
            kind=MetricKind.DIAGNOSTIC,
            scope=MetricScope.SAMPLE,
            required_prediction_fields=("timestamps",),
            required_ground_truth_fields=(),
            supported_task_types=frozenset({"any"}),
            compatible_capabilities=frozenset(
                {Capability.PRODUCES_SCANPATHS, Capability.CONTINUOUS_TIME_OUTPUT}
            ),
            parameter_schema={},
            aggregation_semantics="last timestamp minus first timestamp, in seconds",
            direction=MetricDirection.INTERPRET_ONLY,
            unit="seconds",
            references=(),
            evaluator=_timestamp_span,
        )
    )
    registry.register(
        MetricDefinition(
            name="total_fixation_duration_s",
            implementation_version="1.0.0",
            kind=MetricKind.DIAGNOSTIC,
            scope=MetricScope.SAMPLE,
            required_prediction_fields=("durations",),
            required_ground_truth_fields=(),
            supported_task_types=frozenset({"any"}),
            compatible_capabilities=frozenset(
                {
                    Capability.PRODUCES_SCANPATHS,
                    Capability.FIXATION_DURATION_OUTPUT,
                }
            ),
            parameter_schema={},
            aggregation_semantics="sum of model-emitted fixation durations only",
            direction=MetricDirection.INTERPRET_ONLY,
            unit="seconds",
            references=(),
            evaluator=_total_duration,
        )
    )
    registry.register(
        MetricDefinition(
            name="mean_fixation_duration_s",
            implementation_version="1.0.0",
            kind=MetricKind.DIAGNOSTIC,
            scope=MetricScope.SAMPLE,
            required_prediction_fields=("durations",),
            required_ground_truth_fields=(),
            supported_task_types=frozenset({"any"}),
            compatible_capabilities=frozenset(
                {
                    Capability.PRODUCES_SCANPATHS,
                    Capability.FIXATION_DURATION_OUTPUT,
                }
            ),
            parameter_schema={},
            aggregation_semantics="arithmetic mean of model-emitted fixation durations",
            direction=MetricDirection.INTERPRET_ONLY,
            unit="seconds",
            references=(),
            evaluator=_mean_duration,
        )
    )
    registry.register(
        MetricDefinition(
            name="duration_mae_s",
            implementation_version="1.0.0",
            kind=MetricKind.DIAGNOSTIC,
            scope=MetricScope.SAMPLE,
            required_prediction_fields=("durations",),
            required_ground_truth_fields=("durations",),
            supported_task_types=frozenset({"any"}),
            compatible_capabilities=frozenset(
                {
                    Capability.PRODUCES_SCANPATHS,
                    Capability.FIXATION_DURATION_OUTPUT,
                }
            ),
            parameter_schema={},
            aggregation_semantics=(
                "mean absolute duration error after the configured explicit index alignment"
            ),
            direction=MetricDirection.LOWER,
            unit="seconds",
            references=(),
            evaluator=_duration_mae,
        )
    )
    registry.register(
        MetricDefinition(
            name="tfp_auc",
            implementation_version="1.0.0",
            kind=MetricKind.SCIENTIFIC,
            scope=MetricScope.SAMPLE,
            required_prediction_fields=("fixations",),
            required_ground_truth_fields=("target_boxes",),
            supported_task_types=frozenset({"visual_search_target_present"}),
            compatible_capabilities=scanpath_capability,
            parameter_schema={
                "max_fixations": ParameterSpec("integer", required=True, minimum=1),
                "include_initial_fixation": ParameterSpec(
                    "boolean", required=True
                ),
                "normalization": ParameterSpec(
                    "string",
                    required=True,
                    choices=("raw_trapezoid", "unit_interval"),
                ),
            },
            aggregation_semantics=(
                "trapezoidal area of the cumulative target-hit curve; averaging "
                "sample values equals the empirical TFP curve area"
            ),
            direction=MetricDirection.HIGHER,
            unit="fixation_steps",
            references=(
                "Yang et al., Predicting Goal-directed Human Attention Using Inverse Reinforcement Learning, CVPR 2020",
                "Chen et al., COCO-Search18 fixation dataset, Scientific Reports 2021",
            ),
            evaluator=_tfp_auc,
        )
    )
    saliency_capability = frozenset({Capability.PRODUCES_SALIENCY_MAPS})
    registry.register(
        MetricDefinition(
            name="auc_judd",
            implementation_version="1.0.0",
            kind=MetricKind.SCIENTIFIC,
            scope=MetricScope.SAMPLE,
            required_prediction_fields=("saliency_artifact",),
            required_ground_truth_fields=("fixations",),
            supported_task_types=frozenset({"any"}),
            compatible_capabilities=saliency_capability,
            parameter_schema={
                "artifact_format": ParameterSpec(
                    "string", default="dense_json", choices=("dense_json",)
                ),
                "tie_policy": ParameterSpec(
                    "string", default="average_rank", choices=("average_rank",)
                ),
            },
            aggregation_semantics=(
                "deterministic tie-aware ROC AUC at ground-truth fixation pixels"
            ),
            direction=MetricDirection.HIGHER,
            unit="unit_interval",
            references=(
                "Judd et al., Learning to Predict Where Humans Look, ICCV 2009",
            ),
            evaluator=_auc_judd,
        )
    )
    registry.register(
        MetricDefinition(
            name="nss",
            implementation_version="1.0.0",
            kind=MetricKind.SCIENTIFIC,
            scope=MetricScope.SAMPLE,
            required_prediction_fields=("saliency_artifact",),
            required_ground_truth_fields=("fixations",),
            supported_task_types=frozenset({"any"}),
            compatible_capabilities=saliency_capability,
            parameter_schema={
                "artifact_format": ParameterSpec(
                    "string", default="dense_json", choices=("dense_json",)
                )
            },
            aggregation_semantics=(
                "mean z-scored saliency at ground-truth fixation pixels"
            ),
            direction=MetricDirection.HIGHER,
            unit="z_score",
            references=(
                "Peters et al., Components of bottom-up gaze allocation in natural images, Vision Research 2005",
            ),
            evaluator=_nss,
        )
    )
    registry.register(
        MetricDefinition(
            name="auc_judd_fixation_density",
            implementation_version="1.0.0",
            kind=MetricKind.SCIENTIFIC,
            scope=MetricScope.ITEM,
            required_prediction_fields=("fixations",),
            required_ground_truth_fields=("fixations",),
            supported_task_types=frozenset({"any"}),
            compatible_capabilities=scanpath_capability,
            parameter_schema={
                **_fixation_density_parameter_schema(),
                "tie_policy": ParameterSpec(
                    "string", default="average_rank", choices=("average_rank",)
                ),
            },
            aggregation_semantics=(
                "Judd ROC AUC over an equal-fixation-weight Gaussian density "
                "formed from every preserved prediction sample for one request"
            ),
            direction=MetricDirection.HIGHER,
            unit="unit_interval",
            references=(
                "Judd et al., Learning to Predict Where Humans Look, ICCV 2009",
                "ScanDiff official scanpath-derived saliency evaluation",
                "GazeXplain official scanpath-derived saliency evaluation",
            ),
            evaluator=_auc_judd_fixation_density,
        )
    )
    registry.register(
        MetricDefinition(
            name="nss_fixation_density",
            implementation_version="1.0.0",
            kind=MetricKind.SCIENTIFIC,
            scope=MetricScope.ITEM,
            required_prediction_fields=("fixations",),
            required_ground_truth_fields=("fixations",),
            supported_task_types=frozenset({"any"}),
            compatible_capabilities=scanpath_capability,
            parameter_schema=_fixation_density_parameter_schema(),
            aggregation_semantics=(
                "NSS over an equal-fixation-weight Gaussian density formed from "
                "every preserved prediction sample for one request"
            ),
            direction=MetricDirection.HIGHER,
            unit="z_score",
            references=(
                "Peters et al., Components of bottom-up gaze allocation in natural images, Vision Research 2005",
                "ScanDiff official scanpath-derived saliency evaluation",
                "GazeXplain official scanpath-derived saliency evaluation",
            ),
            evaluator=_nss_fixation_density,
        )
    )
    registry.register(
        MetricDefinition(
            name="endpoint_pairwise_distance_normalized",
            implementation_version="1.0.0",
            kind=MetricKind.DIAGNOSTIC,
            scope=MetricScope.ITEM,
            required_prediction_fields=("multiple_nonempty_scanpaths",),
            required_ground_truth_fields=(),
            supported_task_types=frozenset({"any"}),
            compatible_capabilities=scanpath_capability,
            parameter_schema={},
            aggregation_semantics=(
                "mean pairwise endpoint distance across samples divided by image diagonal"
            ),
            direction=MetricDirection.INTERPRET_ONLY,
            unit="image_diagonals",
            references=(),
            evaluator=_endpoint_pairwise_distance,
        )
    )
    registry.register(
        MetricDefinition(
            name="endpoint_dispersion_normalized",
            implementation_version="1.0.0",
            kind=MetricKind.DIAGNOSTIC,
            scope=MetricScope.ITEM,
            required_prediction_fields=("multiple_nonempty_scanpaths",),
            required_ground_truth_fields=(),
            supported_task_types=frozenset({"any"}),
            compatible_capabilities=scanpath_capability,
            parameter_schema={},
            aggregation_semantics=(
                "root-mean-square endpoint distance from the sample centroid divided by image diagonal"
            ),
            direction=MetricDirection.INTERPRET_ONLY,
            unit="image_diagonals",
            references=(),
            evaluator=_endpoint_dispersion,
        )
    )
    registry.register(
        MetricDefinition(
            name="fixation_density_kde",
            implementation_version="1.0.0",
            kind=MetricKind.DIAGNOSTIC,
            scope=MetricScope.ITEM,
            required_prediction_fields=("fixations",),
            required_ground_truth_fields=(),
            supported_task_types=frozenset({"any"}),
            compatible_capabilities=scanpath_capability,
            parameter_schema=_fixation_density_parameter_schema(),
            aggregation_semantics=(
                "equal-weight Gaussian KDE over every fixation in every preserved sample"
            ),
            direction=MetricDirection.INTERPRET_ONLY,
            unit="probability_density_artifact",
            references=(),
            evaluator=_fixation_density_kde,
        )
    )
    return registry


def _fixation_density_parameter_schema() -> dict[str, ParameterSpec]:
    return {
        "kernel": ParameterSpec(
            "string", required=True, choices=("gaussian",)
        ),
        "bandwidth_norm": ParameterSpec(
            "number", required=True, minimum=1e-12, maximum=1.0
        ),
        "grid_width": ParameterSpec(
            "integer", required=True, minimum=2, maximum=512
        ),
        "grid_height": ParameterSpec(
            "integer", required=True, minimum=2, maximum=512
        ),
    }


def _one_prediction(context: MetricContext) -> PredictionRecord:
    if len(context.predictions) != 1:
        raise MetricInvalidData("sample metric requires exactly one prediction")
    return context.predictions[0]


def _fixation_count(context: MetricContext) -> MetricValue:
    return MetricValue(
        value=float(len(_one_prediction(context).fixations)),
        unit="fixations",
    )


def _scanpath_length_normalized(context: MetricContext) -> MetricValue:
    prediction = _one_prediction(context)
    points, warnings = _pixel_coordinates(prediction.fixations, context)
    if not points:
        raise MetricInvalidData("prediction has no fixations")
    length = sum(
        math.hypot(second[0] - first[0], second[1] - first[1])
        for first, second in zip(points, points[1:])
    )
    return MetricValue(
        value=length / _image_diagonal(context.item),
        unit="image_diagonals",
        warnings=warnings,
    )


def _timestamp_span(context: MetricContext) -> MetricValue:
    prediction = _one_prediction(context)
    timestamps = tuple(fixation.timestamp_s for fixation in prediction.fixations)
    if not timestamps or any(value is None for value in timestamps):
        raise MetricInvalidData("complete prediction timestamps are required")
    values = tuple(float(value) for value in timestamps if value is not None)
    return MetricValue(value=values[-1] - values[0], unit="seconds")


def _durations(prediction: PredictionRecord) -> tuple[float, ...]:
    values = tuple(fixation.duration_s for fixation in prediction.fixations)
    if not values or any(value is None for value in values):
        raise MetricInvalidData("complete prediction durations are required")
    return tuple(float(value) for value in values if value is not None)


def _total_duration(context: MetricContext) -> MetricValue:
    return MetricValue(
        value=sum(_durations(_one_prediction(context))),
        unit="seconds",
    )


def _mean_duration(context: MetricContext) -> MetricValue:
    values = _durations(_one_prediction(context))
    return MetricValue(value=sum(values) / len(values), unit="seconds")


def _duration_mae(context: MetricContext) -> MetricValue:
    predicted = _durations(_one_prediction(context))
    if not context.ground_truth:
        raise MetricNotApplicable("matching ground-truth scanpaths are unavailable")
    errors: list[float] = []
    for scanpath in context.ground_truth:
        raw = tuple(fixation.duration_s for fixation in scanpath.fixations)
        if any(value is None for value in raw):
            raise MetricNotApplicable(
                "matching ground truth does not contain complete durations"
            )
        truth = tuple(float(value) for value in raw if value is not None)
        if context.protocol.temporal_alignment.policy == "strict_index":
            if len(predicted) != len(truth):
                raise MetricInvalidData(
                    "strict_index duration alignment requires equal sequence lengths"
                )
            length = len(predicted)
        else:
            length = min(len(predicted), len(truth))
        if length == 0:
            raise MetricInvalidData("duration alignment produced no paired events")
        errors.extend(
            abs(predicted[index] - truth[index]) for index in range(length)
        )
    return MetricValue(value=sum(errors) / len(errors), unit="seconds")


def _tfp_auc(context: MetricContext) -> MetricValue:
    prediction = _one_prediction(context)
    points, warnings = _pixel_coordinates(prediction.fixations, context)
    if not context.item.target_boxes:
        raise MetricNotApplicable("target-present bounding boxes are required")
    if not bool(context.parameters["include_initial_fixation"]):
        points = points[1:]
    maximum = int(context.parameters["max_fixations"])
    curve = [0.0]
    found = False
    for step in range(maximum):
        if step < len(points) and _inside_any_box(points[step], context.item):
            found = True
        curve.append(1.0 if found else 0.0)
    area = sum(
        (curve[index] + curve[index + 1]) / 2.0
        for index in range(maximum)
    )
    unit = "fixation_steps"
    if context.parameters["normalization"] == "unit_interval":
        area /= maximum
        unit = "unit_interval"
    return MetricValue(value=area, unit=unit, warnings=warnings)


def _nss(context: MetricContext) -> MetricValue:
    values, width, height = _load_saliency_map(context)
    fixation_indices, warnings = _ground_truth_pixel_indices(
        context, width=width, height=height
    )
    return MetricValue(
        value=_nss_score(values, fixation_indices),
        unit="z_score",
        warnings=warnings,
    )


def _nss_score(
    values: Sequence[float], fixation_indices: Sequence[int]
) -> float:
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    if variance == 0.0:
        raise MetricInvalidData("NSS is undefined for a constant saliency map")
    scale = math.sqrt(variance)
    score = sum((values[index] - mean) / scale for index in fixation_indices)
    return score / len(fixation_indices)


def _auc_judd(context: MetricContext) -> MetricValue:
    values, width, height = _load_saliency_map(context)
    positives, warnings = _ground_truth_pixel_indices(
        context, width=width, height=height
    )
    return MetricValue(
        value=_auc_judd_score(values, positives),
        unit="unit_interval",
        warnings=warnings,
    )


def _auc_judd_score(values: Sequence[float], positives: Sequence[int]) -> float:
    positive_set = set(positives)
    if len(positive_set) == len(values):
        raise MetricInvalidData("AUC requires at least one non-fixated pixel")
    ranked = sorted((value, index) for index, value in enumerate(values))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ranked):
        end = start + 1
        while end < len(ranked) and ranked[end][0] == ranked[start][0]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        for _, index in ranked[start:end]:
            ranks[index] = average_rank
        start = end
    positive_count = len(positive_set)
    negative_count = len(values) - positive_count
    rank_sum = sum(ranks[index] for index in positive_set)
    auc = (
        rank_sum - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)
    return auc


def _auc_judd_fixation_density(context: MetricContext) -> MetricValue:
    values, width, height, density_warnings, fixation_count = (
        _fixation_density_values(context)
    )
    positives, truth_warnings = _ground_truth_grid_indices(
        context, width=width, height=height
    )
    artifact = _write_derived_density_artifact(
        context,
        values=values,
        width=width,
        height=height,
        fixation_count=fixation_count,
    )
    return MetricValue(
        value=_auc_judd_score(values, positives),
        unit="unit_interval",
        artifact_reference=artifact,
        warnings=tuple(sorted(set(density_warnings + truth_warnings))),
    )


def _nss_fixation_density(context: MetricContext) -> MetricValue:
    values, width, height, density_warnings, fixation_count = (
        _fixation_density_values(context)
    )
    positives, truth_warnings = _ground_truth_grid_indices(
        context, width=width, height=height
    )
    artifact = _write_derived_density_artifact(
        context,
        values=values,
        width=width,
        height=height,
        fixation_count=fixation_count,
    )
    return MetricValue(
        value=_nss_score(values, positives),
        unit="z_score",
        artifact_reference=artifact,
        warnings=tuple(sorted(set(density_warnings + truth_warnings))),
    )


def _endpoint_pairwise_distance(context: MetricContext) -> MetricValue:
    endpoints, warnings = _endpoints(context)
    distances = [
        math.hypot(second[0] - first[0], second[1] - first[1])
        for index, first in enumerate(endpoints)
        for second in endpoints[index + 1 :]
    ]
    return MetricValue(
        value=(sum(distances) / len(distances)) / _image_diagonal(context.item),
        unit="image_diagonals",
        warnings=warnings,
    )


def _endpoint_dispersion(context: MetricContext) -> MetricValue:
    endpoints, warnings = _endpoints(context)
    center_x = sum(point[0] for point in endpoints) / len(endpoints)
    center_y = sum(point[1] for point in endpoints) / len(endpoints)
    squared = sum(
        (point[0] - center_x) ** 2 + (point[1] - center_y) ** 2
        for point in endpoints
    ) / len(endpoints)
    return MetricValue(
        value=math.sqrt(squared) / _image_diagonal(context.item),
        unit="image_diagonals",
        warnings=warnings,
    )


def _fixation_density_kde(context: MetricContext) -> MetricValue:
    values, width, height, warnings, fixation_count = _fixation_density_values(
        context
    )
    density = _density_rows(values, width=width, height=height)
    artifact_name = f"{context.artifact_stem}.fixation_density_kde.json"
    artifact_path = context.output_directory / "artifacts" / artifact_name
    _atomic_json(
        artifact_path,
        _density_artifact_payload(
            context,
            metric_name="fixation_density_kde",
            metric_version="1.0.0",
            width=width,
            height=height,
            fixation_count=fixation_count,
            density=density,
        ),
    )
    return MetricValue(
        value=None,
        unit="probability_density_artifact",
        artifact_reference=f"artifacts/{artifact_name}",
        warnings=warnings,
    )


def _fixation_density_values(
    context: MetricContext,
) -> tuple[tuple[float, ...], int, int, tuple[str, ...], int]:
    normalized: list[tuple[float, float]] = []
    warnings: list[str] = []
    for prediction in context.predictions:
        points, current = _normalized_coordinates(prediction.fixations, context)
        normalized.extend(points)
        warnings.extend(current)
    if not normalized:
        raise MetricInvalidData("KDE requires at least one predicted fixation")
    width = int(context.parameters["grid_width"])
    height = int(context.parameters["grid_height"])
    bandwidth = float(context.parameters["bandwidth_norm"])
    density = _gaussian_density_grid(tuple(normalized), width, height, bandwidth)
    return (
        density,
        width,
        height,
        tuple(sorted(set(warnings))),
        len(normalized),
    )


@lru_cache(maxsize=8)
def _gaussian_density_grid(
    normalized: tuple[tuple[float, float], ...],
    width: int,
    height: int,
    bandwidth: float,
) -> tuple[float, ...]:
    inverse = 1.0 / (2.0 * bandwidth * bandwidth)
    density: list[float] = []
    total = 0.0
    for row in range(height):
        y = row / (height - 1)
        for column in range(width):
            x = column / (width - 1)
            value = sum(
                math.exp(-((x - px) ** 2 + (y - py) ** 2) * inverse)
                for px, py in normalized
            )
            density.append(value)
            total += value
    if total == 0.0:
        raise MetricInvalidData("KDE normalization sum is zero")
    return tuple(value / total for value in density)


def _density_rows(
    values: Sequence[float], *, width: int, height: int
) -> list[list[float]]:
    return [
        [float(value) for value in values[row * width : (row + 1) * width]]
        for row in range(height)
    ]


def _density_artifact_payload(
    context: MetricContext,
    *,
    metric_name: str,
    metric_version: str,
    width: int,
    height: int,
    fixation_count: int,
    density: list[list[float]],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "metric_name": metric_name,
        "metric_version": metric_version,
        "source": "all_fixations_from_all_preserved_prediction_samples",
        "kernel": "gaussian",
        "bandwidth_norm": float(context.parameters["bandwidth_norm"]),
        "coordinate_space": "normalized_pixel_center",
        "grid_width": width,
        "grid_height": height,
        "sample_ids": sorted(
            prediction.sample_id for prediction in context.predictions
        ),
        "sample_count": len(context.predictions),
        "fixation_count": fixation_count,
        "normalization": "discrete_grid_sum_1",
        "density": density,
    }


def _write_derived_density_artifact(
    context: MetricContext,
    *,
    values: Sequence[float],
    width: int,
    height: int,
    fixation_count: int,
) -> str:
    bandwidth = float(context.parameters["bandwidth_norm"])
    bandwidth_token = (
        format(bandwidth, ".12g").replace("-", "m").replace(".", "p")
    )
    artifact_name = (
        f"{context.artifact_stem}.derived_fixation_density."
        f"{width}x{height}.bw-{bandwidth_token}.json"
    )
    artifact_path = context.output_directory / "artifacts" / artifact_name
    _atomic_json(
        artifact_path,
        _density_artifact_payload(
            context,
            metric_name="derived_fixation_density",
            metric_version="1.0.0",
            width=width,
            height=height,
            fixation_count=fixation_count,
            density=_density_rows(values, width=width, height=height),
        ),
    )
    return f"artifacts/{artifact_name}"


def _pixel_coordinates(
    fixations: Sequence[FixationEvent],
    context: MetricContext,
) -> tuple[tuple[tuple[float, float], ...], tuple[str, ...]]:
    points = []
    warnings = []
    width = context.item.image_width
    height = context.item.image_height
    for index, fixation in enumerate(fixations):
        if fixation.x_px is not None and fixation.y_px is not None:
            x = fixation.x_px
            y = fixation.y_px
        elif fixation.x_norm is not None and fixation.y_norm is not None:
            x = fixation.x_norm * (width - 1)
            y = fixation.y_norm * (height - 1)
        else:
            raise MetricInvalidData(
                f"fixation {index} has no complete coordinate pair"
            )
        if x < 0.0 or x > width - 1 or y < 0.0 or y > height - 1:
            if context.protocol.coordinate_conversion.out_of_bounds == "invalid":
                raise MetricInvalidData(
                    f"fixation {index} is outside the canonical image lattice"
                )
            x = min(max(x, 0.0), width - 1)
            y = min(max(y, 0.0), height - 1)
            warnings.append(
                f"fixation {index} was clipped by the configured coordinate policy"
            )
        points.append((x, y))
    return tuple(points), tuple(sorted(set(warnings)))


def _normalized_coordinates(
    fixations: Sequence[FixationEvent],
    context: MetricContext,
) -> tuple[tuple[tuple[float, float], ...], tuple[str, ...]]:
    pixels, warnings = _pixel_coordinates(fixations, context)
    return (
        tuple(
            (
                x / (context.item.image_width - 1),
                y / (context.item.image_height - 1),
            )
            for x, y in pixels
        ),
        warnings,
    )


def _inside_any_box(point: tuple[float, float], item: DatasetItem) -> bool:
    x, y = point
    return any(
        box.x_px <= x < box.x_px + box.width_px
        and box.y_px <= y < box.y_px + box.height_px
        for box in item.target_boxes
    )


def _image_diagonal(item: DatasetItem) -> float:
    return math.hypot(item.image_width - 1, item.image_height - 1)


def _endpoints(
    context: MetricContext,
) -> tuple[tuple[tuple[float, float], ...], tuple[str, ...]]:
    if len(context.predictions) < 2:
        raise MetricNotApplicable(
            "at least two preserved samples are required for diversity"
        )
    endpoints = []
    warnings = []
    for prediction in context.predictions:
        points, current = _pixel_coordinates(prediction.fixations, context)
        if not points:
            raise MetricInvalidData(
                f"sample '{prediction.sample_id}' has no endpoint"
            )
        endpoints.append(points[-1])
        warnings.extend(current)
    return tuple(endpoints), tuple(sorted(set(warnings)))


def _load_saliency_map(
    context: MetricContext,
) -> tuple[tuple[float, ...], int, int]:
    prediction = _one_prediction(context)
    artifact = prediction.saliency_artifact
    if artifact is None:
        raise MetricInvalidData("prediction has no saliency artifact")
    if "://" in artifact.uri:
        raise MetricInvalidData("saliency metric requires a local artifact path")
    path = Path(artifact.uri)
    if not path.is_absolute():
        path = context.run_directory / path
    try:
        raw = json.loads(path.read_text(encoding="utf-8"), parse_constant=_bad_json)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise MetricInvalidData(f"cannot load dense saliency JSON '{path}': {error}")
    if isinstance(raw, Mapping):
        if set(raw) != {"width", "height", "values"}:
            raise MetricInvalidData(
                "dense saliency JSON object must contain width, height, and values"
            )
        width = raw["width"]
        height = raw["height"]
        rows = raw["values"]
    else:
        width = context.item.image_width
        height = context.item.image_height
        rows = raw
    if type(width) is not int or type(height) is not int:
        raise MetricInvalidData("saliency width and height must be integers")
    if width != context.item.image_width or height != context.item.image_height:
        raise MetricInvalidData(
            "saliency map dimensions do not match the source image dimensions"
        )
    if not isinstance(rows, list) or len(rows) != height:
        raise MetricInvalidData("saliency values must have exactly height rows")
    values = []
    for row_index, row in enumerate(rows):
        if not isinstance(row, list) or len(row) != width:
            raise MetricInvalidData(
                f"saliency row {row_index} must contain exactly width values"
            )
        for column, value in enumerate(row):
            if type(value) not in {int, float} or not math.isfinite(float(value)):
                raise MetricInvalidData(
                    f"saliency value [{row_index}][{column}] must be finite"
                )
            values.append(float(value))
    return tuple(values), width, height


def _ground_truth_pixel_indices(
    context: MetricContext,
    *,
    width: int,
    height: int,
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    if not context.ground_truth:
        raise MetricInvalidData("matching ground-truth fixations are required")
    indices = set()
    warnings = []
    for scanpath in context.ground_truth:
        points, current = _pixel_coordinates(scanpath.fixations, context)
        warnings.extend(current)
        for x, y in points:
            column = min(width - 1, int(math.floor(x + 0.5)))
            row = min(height - 1, int(math.floor(y + 0.5)))
            indices.add(row * width + column)
    if not indices:
        raise MetricInvalidData("matching ground truth contains no fixations")
    return tuple(sorted(indices)), tuple(sorted(set(warnings)))


def _ground_truth_grid_indices(
    context: MetricContext,
    *,
    width: int,
    height: int,
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    if not context.ground_truth:
        raise MetricInvalidData("matching ground-truth fixations are required")
    indices = set()
    warnings = []
    for scanpath in context.ground_truth:
        points, current = _normalized_coordinates(scanpath.fixations, context)
        warnings.extend(current)
        for x, y in points:
            column = min(width - 1, int(math.floor(x * (width - 1) + 0.5)))
            row = min(height - 1, int(math.floor(y * (height - 1) + 0.5)))
            indices.add(row * width + column)
    if not indices:
        raise MetricInvalidData("matching ground truth contains no fixations")
    return tuple(sorted(indices)), tuple(sorted(set(warnings)))


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
            json.dump(
                value,
                stream,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            stream.write("\n")
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


def _bad_json(value: str) -> None:
    raise ValueError(f"non-standard JSON value '{value}'")
