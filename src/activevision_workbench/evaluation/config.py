from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import ClassVar

from activevision_workbench.contracts._validation import (
    JsonMapping,
    freeze_json_mapping,
    thaw_json_value,
)
from activevision_workbench.errors import EvaluationConfigurationError


class ExistingOutputPolicy(str, Enum):
    CREATE = "create"
    OVERWRITE = "overwrite"
    RESUME = "resume"


@dataclass(frozen=True, slots=True)
class MetricRequest:
    name: str
    parameters: JsonMapping = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _string(self.name, "metric.name"))
        object.__setattr__(
            self,
            "parameters",
            _json_mapping(self.parameters, f"metrics.{self.name}.parameters"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "parameters": thaw_json_value(self.parameters),
        }


@dataclass(frozen=True, slots=True)
class GroundTruthConfig:
    adapter: str
    dataset_id: str
    version: str
    root: str
    split: str
    task_type: str = "auto"
    options: JsonMapping = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("adapter", "dataset_id", "version", "root", "split"):
            object.__setattr__(
                self,
                name,
                _string(getattr(self, name), f"dataset.{name}"),
            )
        if self.adapter not in {"json", "osie", "coco_search18"}:
            raise EvaluationConfigurationError(
                "dataset.adapter must be 'json', 'osie', or 'coco_search18'"
            )
        if self.task_type not in {
            "auto",
            "free_viewing",
            "visual_search_target_present",
            "instruction_conditioned",
        }:
            raise EvaluationConfigurationError(
                "dataset.task_type must be 'auto', 'free_viewing', "
                "'visual_search_target_present', or 'instruction_conditioned'"
            )
        object.__setattr__(
            self,
            "options",
            _json_mapping(self.options, "dataset.options"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "adapter": self.adapter,
            "id": self.dataset_id,
            "version": self.version,
            "root": self.root,
            "split": self.split,
            "task_type": self.task_type,
            "options": thaw_json_value(self.options),
        }


@dataclass(frozen=True, slots=True)
class ConfidenceIntervalConfig:
    method: str = "normal"
    level: float = 0.95

    def __post_init__(self) -> None:
        if self.method not in {"normal", "none"}:
            raise EvaluationConfigurationError(
                "sample_aggregation.confidence_interval.method must be "
                "'normal' or 'none'"
            )
        level = _number(
            self.level,
            "sample_aggregation.confidence_interval.level",
            minimum=0.0,
            maximum=1.0,
        )
        if self.method == "normal" and level in {0.0, 1.0}:
            raise EvaluationConfigurationError(
                "normal confidence interval level must be strictly between 0 and 1"
            )
        object.__setattr__(self, "level", level)

    def to_dict(self) -> dict[str, object]:
        return {"method": self.method, "level": self.level}


@dataclass(frozen=True, slots=True)
class BestOfNConfig:
    enabled: bool = False
    metrics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise EvaluationConfigurationError(
                "sample_aggregation.best_of_n.enabled must be a boolean"
            )
        object.__setattr__(
            self,
            "metrics",
            _string_tuple(self.metrics, "sample_aggregation.best_of_n.metrics"),
        )
        if self.enabled and not self.metrics:
            raise EvaluationConfigurationError(
                "enabled best-of-N requires at least one explicitly named metric"
            )

    def to_dict(self) -> dict[str, object]:
        return {"enabled": self.enabled, "metrics": list(self.metrics)}


@dataclass(frozen=True, slots=True)
class SampleAggregationConfig:
    policy: str = "mean_std"
    standard_deviation: str = "population"
    confidence_interval: ConfidenceIntervalConfig = field(
        default_factory=ConfidenceIntervalConfig
    )
    best_of_n: BestOfNConfig = field(default_factory=BestOfNConfig)

    def __post_init__(self) -> None:
        if self.policy != "mean_std":
            raise EvaluationConfigurationError(
                "sample_aggregation.policy must be 'mean_std'"
            )
        if self.standard_deviation not in {"population", "sample"}:
            raise EvaluationConfigurationError(
                "sample_aggregation.standard_deviation must be 'population' "
                "or 'sample'"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "policy": self.policy,
            "standard_deviation": self.standard_deviation,
            "confidence_interval": self.confidence_interval.to_dict(),
            "best_of_n": self.best_of_n.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ObserverAggregationConfig:
    policy: str = "macro"
    include_micro: bool = False
    unknown_observer_policy: str = "separate"

    def __post_init__(self) -> None:
        if self.policy != "macro":
            raise EvaluationConfigurationError(
                "observer_aggregation.policy must be 'macro'"
            )
        if type(self.include_micro) is not bool:
            raise EvaluationConfigurationError(
                "observer_aggregation.include_micro must be a boolean"
            )
        if self.unknown_observer_policy != "separate":
            raise EvaluationConfigurationError(
                "observer_aggregation.unknown_observer_policy must be 'separate'"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "policy": self.policy,
            "include_micro": self.include_micro,
            "unknown_observer_policy": self.unknown_observer_policy,
        }


@dataclass(frozen=True, slots=True)
class TemporalAlignmentConfig:
    policy: str = "strict_index"
    unit: str = "seconds"

    def __post_init__(self) -> None:
        if self.policy not in {"strict_index", "truncate_to_shorter"}:
            raise EvaluationConfigurationError(
                "temporal_alignment.policy must be 'strict_index' or "
                "'truncate_to_shorter'"
            )
        if self.unit != "seconds":
            raise EvaluationConfigurationError(
                "temporal_alignment.unit must be 'seconds'"
            )

    def to_dict(self) -> dict[str, object]:
        return {"policy": self.policy, "unit": self.unit}


@dataclass(frozen=True, slots=True)
class CoordinateConversionConfig:
    policy: str = "canonical_pixel_center"
    out_of_bounds: str = "invalid"

    def __post_init__(self) -> None:
        if self.policy != "canonical_pixel_center":
            raise EvaluationConfigurationError(
                "coordinate_conversion.policy must be 'canonical_pixel_center'"
            )
        if self.out_of_bounds not in {"invalid", "clip"}:
            raise EvaluationConfigurationError(
                "coordinate_conversion.out_of_bounds must be 'invalid' or 'clip'"
            )

    def to_dict(self) -> dict[str, object]:
        return {"policy": self.policy, "out_of_bounds": self.out_of_bounds}


@dataclass(frozen=True, slots=True)
class EvaluationProtocol:
    CURRENT_SCHEMA_VERSION: ClassVar[int] = 1

    schema_version: int
    run_directory: str
    dataset: GroundTruthConfig
    metrics: tuple[MetricRequest, ...]
    sample_aggregation: SampleAggregationConfig
    observer_aggregation: ObserverAggregationConfig
    temporal_alignment: TemporalAlignmentConfig
    coordinate_conversion: CoordinateConversionConfig
    output_directory: str
    existing_output: ExistingOutputPolicy

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise EvaluationConfigurationError("schema_version must be integer 1")
        for name in ("run_directory", "output_directory"):
            object.__setattr__(
                self,
                name,
                _string(getattr(self, name), name),
            )
        if not isinstance(self.dataset, GroundTruthConfig):
            raise EvaluationConfigurationError(
                "dataset must be a GroundTruthConfig"
            )
        if not isinstance(self.metrics, Sequence) or isinstance(
            self.metrics, (str, bytes, bytearray)
        ):
            raise EvaluationConfigurationError("metrics must be a sequence")
        metrics = tuple(self.metrics)
        if not metrics:
            raise EvaluationConfigurationError("metrics must not be empty")
        if any(not isinstance(metric, MetricRequest) for metric in metrics):
            raise EvaluationConfigurationError(
                "metrics must contain MetricRequest values"
            )
        names = [metric.name for metric in metrics]
        if len(names) != len(set(names)):
            raise EvaluationConfigurationError(
                "metrics must not contain duplicate stable names"
            )
        object.__setattr__(self, "metrics", metrics)
        expected_types = {
            "sample_aggregation": SampleAggregationConfig,
            "observer_aggregation": ObserverAggregationConfig,
            "temporal_alignment": TemporalAlignmentConfig,
            "coordinate_conversion": CoordinateConversionConfig,
        }
        for name, expected in expected_types.items():
            if not isinstance(getattr(self, name), expected):
                raise EvaluationConfigurationError(
                    f"{name} must be a {expected.__name__}"
                )
        if not isinstance(self.existing_output, ExistingOutputPolicy):
            raise EvaluationConfigurationError(
                "existing_output must be an ExistingOutputPolicy"
            )

    @classmethod
    def from_file(cls, path: str | Path) -> "EvaluationProtocol":
        source = Path(path)
        try:
            text = source.read_text(encoding="utf-8")
        except OSError as error:
            raise EvaluationConfigurationError(
                f"cannot read evaluation protocol '{source}': {error}"
            ) from error
        data = _load_json_or_yaml(text, source)
        return cls.from_dict(data, base_dir=source.resolve().parent)

    @classmethod
    def from_dict(
        cls,
        data: object,
        *,
        base_dir: str | Path | None = None,
    ) -> "EvaluationProtocol":
        root = Path.cwd() if base_dir is None else Path(base_dir)
        top = _mapping(
            data,
            "EvaluationProtocol",
            {
                "schema_version",
                "run_directory",
                "dataset",
                "metrics",
                "sample_aggregation",
                "observer_aggregation",
                "temporal_alignment",
                "coordinate_conversion",
                "output_directory",
                "existing_output",
            },
        )
        dataset_raw = _mapping(
            top["dataset"],
            "dataset",
            {"adapter", "id", "version", "root", "split", "task_type", "options"},
        )
        dataset = GroundTruthConfig(
            adapter=dataset_raw["adapter"],  # type: ignore[arg-type]
            dataset_id=dataset_raw["id"],  # type: ignore[arg-type]
            version=dataset_raw["version"],  # type: ignore[arg-type]
            root=_resolved_path(dataset_raw["root"], root, "dataset.root"),
            split=dataset_raw["split"],  # type: ignore[arg-type]
            task_type=dataset_raw["task_type"],  # type: ignore[arg-type]
            options=dataset_raw["options"],  # type: ignore[arg-type]
        )
        raw_metrics = top["metrics"]
        if not isinstance(raw_metrics, list):
            raise EvaluationConfigurationError("metrics must be a JSON array")
        metrics = []
        for index, raw_metric in enumerate(raw_metrics):
            fields = _mapping(
                raw_metric,
                f"metrics[{index}]",
                {"name", "parameters"},
            )
            metrics.append(
                MetricRequest(
                    name=fields["name"],  # type: ignore[arg-type]
                    parameters=fields["parameters"],  # type: ignore[arg-type]
                )
            )

        sample_raw = _mapping(
            top["sample_aggregation"],
            "sample_aggregation",
            {"policy", "standard_deviation", "confidence_interval", "best_of_n"},
        )
        ci_raw = _mapping(
            sample_raw["confidence_interval"],
            "sample_aggregation.confidence_interval",
            {"method", "level"},
        )
        best_raw = _mapping(
            sample_raw["best_of_n"],
            "sample_aggregation.best_of_n",
            {"enabled", "metrics"},
        )
        sample = SampleAggregationConfig(
            policy=sample_raw["policy"],  # type: ignore[arg-type]
            standard_deviation=sample_raw["standard_deviation"],  # type: ignore[arg-type]
            confidence_interval=ConfidenceIntervalConfig(
                method=ci_raw["method"],  # type: ignore[arg-type]
                level=ci_raw["level"],  # type: ignore[arg-type]
            ),
            best_of_n=BestOfNConfig(
                enabled=best_raw["enabled"],  # type: ignore[arg-type]
                metrics=_raw_string_tuple(
                    best_raw["metrics"],
                    "sample_aggregation.best_of_n.metrics",
                ),
            ),
        )
        observer_raw = _mapping(
            top["observer_aggregation"],
            "observer_aggregation",
            {"policy", "include_micro", "unknown_observer_policy"},
        )
        observer = ObserverAggregationConfig(
            policy=observer_raw["policy"],  # type: ignore[arg-type]
            include_micro=observer_raw["include_micro"],  # type: ignore[arg-type]
            unknown_observer_policy=observer_raw[
                "unknown_observer_policy"
            ],  # type: ignore[arg-type]
        )
        temporal_raw = _mapping(
            top["temporal_alignment"],
            "temporal_alignment",
            {"policy", "unit"},
        )
        temporal = TemporalAlignmentConfig(
            policy=temporal_raw["policy"],  # type: ignore[arg-type]
            unit=temporal_raw["unit"],  # type: ignore[arg-type]
        )
        coordinate_raw = _mapping(
            top["coordinate_conversion"],
            "coordinate_conversion",
            {"policy", "out_of_bounds"},
        )
        coordinates = CoordinateConversionConfig(
            policy=coordinate_raw["policy"],  # type: ignore[arg-type]
            out_of_bounds=coordinate_raw["out_of_bounds"],  # type: ignore[arg-type]
        )
        try:
            existing = ExistingOutputPolicy(top["existing_output"])
        except (TypeError, ValueError) as error:
            raise EvaluationConfigurationError(
                "existing_output must be 'create', 'overwrite', or 'resume'"
            ) from error
        return cls(
            schema_version=top["schema_version"],  # type: ignore[arg-type]
            run_directory=_resolved_path(
                top["run_directory"], root, "run_directory"
            ),
            dataset=dataset,
            metrics=tuple(metrics),
            sample_aggregation=sample,
            observer_aggregation=observer,
            temporal_alignment=temporal,
            coordinate_conversion=coordinates,
            output_directory=_resolved_path(
                top["output_directory"], root, "output_directory"
            ),
            existing_output=existing,
        )

    def with_overrides(
        self,
        *,
        run_directory: str | Path | None = None,
        existing_output: ExistingOutputPolicy | None = None,
    ) -> "EvaluationProtocol":
        run = (
            self.run_directory
            if run_directory is None
            else str(Path(run_directory).expanduser().resolve(strict=False))
        )
        policy = self.existing_output if existing_output is None else existing_output
        return replace(self, run_directory=run, existing_output=policy)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "run_directory": self.run_directory,
            "dataset": self.dataset.to_dict(),
            "metrics": [metric.to_dict() for metric in self.metrics],
            "sample_aggregation": self.sample_aggregation.to_dict(),
            "observer_aggregation": self.observer_aggregation.to_dict(),
            "temporal_alignment": self.temporal_alignment.to_dict(),
            "coordinate_conversion": self.coordinate_conversion.to_dict(),
            "output_directory": self.output_directory,
            "existing_output": self.existing_output.value,
        }

    @property
    def protocol_hash(self) -> str:
        data = self.to_dict()
        data.pop("existing_output")
        encoded = json.dumps(
            data,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _load_json_or_yaml(text: str, source: Path) -> object:
    try:
        return json.loads(text, parse_constant=_reject_constant)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore[import-not-found]
        except ImportError as error:
            raise EvaluationConfigurationError(
                f"{source}: protocol is not strict JSON; install PyYAML only "
                "for block-style YAML"
            ) from error
        try:
            return yaml.safe_load(text)
        except Exception as error:
            raise EvaluationConfigurationError(
                f"{source}: invalid evaluation protocol: {error}"
            ) from error


def _reject_constant(value: str) -> None:
    raise EvaluationConfigurationError(
        f"non-standard JSON numeric constant '{value}'"
    )


def _mapping(
    value: object,
    field_name: str,
    expected: set[str],
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise EvaluationConfigurationError(f"{field_name} must be an object")
    if any(type(key) is not str for key in value):
        raise EvaluationConfigurationError(
            f"{field_name} field names must be strings"
        )
    keys = set(value)
    missing = sorted(expected - keys)
    unknown = sorted(keys - expected)
    if missing:
        raise EvaluationConfigurationError(
            f"{field_name}.{missing[0]} is required"
        )
    if unknown:
        raise EvaluationConfigurationError(
            f"{field_name}.{unknown[0]} is unknown"
        )
    return value  # type: ignore[return-value]


def _string(value: object, field_name: str) -> str:
    if type(value) is not str or value == "":
        raise EvaluationConfigurationError(
            f"{field_name} must be a non-empty string"
        )
    return value


def _number(
    value: object,
    field_name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if type(value) not in {int, float}:
        raise EvaluationConfigurationError(f"{field_name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise EvaluationConfigurationError(f"{field_name} must be finite")
    if minimum is not None and result < minimum:
        raise EvaluationConfigurationError(
            f"{field_name} must be >= {minimum}"
        )
    if maximum is not None and result > maximum:
        raise EvaluationConfigurationError(
            f"{field_name} must be <= {maximum}"
        )
    return result


def _json_mapping(value: object, field_name: str) -> JsonMapping:
    try:
        return freeze_json_mapping(value, field_name)
    except Exception as error:
        raise EvaluationConfigurationError(str(error)) from error


def _resolved_path(value: object, base_dir: Path, field_name: str) -> str:
    raw = _string(value, field_name)
    if "://" in raw:
        raise EvaluationConfigurationError(
            f"{field_name} must be an explicit local path"
        )
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return str(path.resolve(strict=False))


def _raw_string_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise EvaluationConfigurationError(f"{field_name} must be a JSON array")
    return _string_tuple(value, field_name)


def _string_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(
        value, (str, bytes, bytearray)
    ):
        raise EvaluationConfigurationError(
            f"{field_name} must be a sequence of strings"
        )
    result = tuple(
        _string(item, f"{field_name}[{index}]")
        for index, item in enumerate(value)
    )
    if len(result) != len(set(result)):
        raise EvaluationConfigurationError(
            f"{field_name} must not contain duplicates"
        )
    return result
