
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar, NoReturn

from activevision_workbench.contracts._validation import (
    JsonMapping,
    freeze_json_mapping,
    optional_finite_number,
    optional_int,
    optional_string,
    require_int,
    require_string,
    thaw_json_value,
    validate_dict_keys,
    validate_schema_version,
)
from activevision_workbench.contracts.capabilities import Capability, CapabilitySet
from activevision_workbench.errors import ContractValidationError


def _raise_nested_error(
    error: ContractValidationError,
    *,
    parent: str,
    nested_type: str,
) -> NoReturn:

    prefix = f"{nested_type}."
    if error.field.startswith(prefix):
        suffix = error.field[len(prefix) :]
        field = f"{parent}.{suffix}"
    elif error.field == nested_type:
        field = parent
    else:
        field = f"{parent}.{error.field}"
    raise ContractValidationError(field, error.reason) from error


@dataclass(frozen=True, slots=True)
class ModelIdentity:

    model_id: str
    model_variant: str
    adapter_version: str
    upstream_version: str | None = None
    checkpoint_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("model_id", "model_variant", "adapter_version"):
            object.__setattr__(
                self,
                name,
                require_string(getattr(self, name), f"ModelIdentity.{name}"),
            )
        for name in ("upstream_version", "checkpoint_id"):
            object.__setattr__(
                self,
                name,
                optional_string(
                    getattr(self, name),
                    f"ModelIdentity.{name}",
                    allow_empty=False,
                ),
            )

    def to_dict(self) -> dict[str, object]:

        return {
            "model_id": self.model_id,
            "model_variant": self.model_variant,
            "adapter_version": self.adapter_version,
            "upstream_version": self.upstream_version,
            "checkpoint_id": self.checkpoint_id,
        }

    @classmethod
    def from_dict(cls, data: object) -> "ModelIdentity":

        names = frozenset(
            {
                "model_id",
                "model_variant",
                "adapter_version",
                "upstream_version",
                "checkpoint_id",
            }
        )
        fields = validate_dict_keys(data, "ModelIdentity", required=names)
        return cls(**{name: fields[name] for name in names})  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class ArtifactReference:

    uri: str
    kind: str
    media_type: str | None = None
    checksum: str | None = None
    metadata: JsonMapping = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "uri",
            require_string(self.uri, "ArtifactReference.uri"),
        )
        object.__setattr__(
            self,
            "kind",
            require_string(self.kind, "ArtifactReference.kind"),
        )
        for name in ("media_type", "checksum"):
            object.__setattr__(
                self,
                name,
                optional_string(
                    getattr(self, name),
                    f"ArtifactReference.{name}",
                    allow_empty=False,
                ),
            )
        object.__setattr__(
            self,
            "metadata",
            freeze_json_mapping(self.metadata, "ArtifactReference.metadata"),
        )

    def to_dict(self) -> dict[str, object]:

        return {
            "uri": self.uri,
            "kind": self.kind,
            "media_type": self.media_type,
            "checksum": self.checksum,
            "metadata": thaw_json_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: object) -> "ArtifactReference":

        names = frozenset({"uri", "kind", "media_type", "checksum", "metadata"})
        fields = validate_dict_keys(data, "ArtifactReference", required=names)
        return cls(**{name: fields[name] for name in names})  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class FixationEvent:

    x_px: float | None = None
    y_px: float | None = None
    x_norm: float | None = None
    y_norm: float | None = None
    sequence_index: int | None = None
    timestamp_s: float | None = None
    duration_s: float | None = None
    confidence: float | None = None
    native_timing: JsonMapping = field(default_factory=dict)
    native_metadata: JsonMapping = field(default_factory=dict)

    def __post_init__(self) -> None:
        x_px = optional_finite_number(self.x_px, "FixationEvent.x_px")
        y_px = optional_finite_number(self.y_px, "FixationEvent.y_px")
        x_norm = optional_finite_number(
            self.x_norm,
            "FixationEvent.x_norm",
            minimum=0.0,
            maximum=1.0,
        )
        y_norm = optional_finite_number(
            self.y_norm,
            "FixationEvent.y_norm",
            minimum=0.0,
            maximum=1.0,
        )
        if (x_px is None) != (y_px is None):
            missing = "y_px" if y_px is None else "x_px"
            raise ContractValidationError(
                f"FixationEvent.{missing}",
                "pixel coordinates must be provided as an x/y pair",
            )
        if (x_norm is None) != (y_norm is None):
            missing = "y_norm" if y_norm is None else "x_norm"
            raise ContractValidationError(
                f"FixationEvent.{missing}",
                "normalized coordinates must be provided as an x/y pair",
            )
        if x_px is None and x_norm is None:
            raise ContractValidationError(
                "FixationEvent.coordinates",
                "at least one pixel or normalized coordinate pair is required",
            )
        object.__setattr__(self, "x_px", x_px)
        object.__setattr__(self, "y_px", y_px)
        object.__setattr__(self, "x_norm", x_norm)
        object.__setattr__(self, "y_norm", y_norm)
        object.__setattr__(
            self,
            "sequence_index",
            optional_int(
                self.sequence_index,
                "FixationEvent.sequence_index",
                minimum=0,
            ),
        )
        object.__setattr__(
            self,
            "timestamp_s",
            optional_finite_number(
                self.timestamp_s,
                "FixationEvent.timestamp_s",
                minimum=0.0,
            ),
        )
        object.__setattr__(
            self,
            "duration_s",
            optional_finite_number(
                self.duration_s,
                "FixationEvent.duration_s",
                minimum=0.0,
            ),
        )
        object.__setattr__(
            self,
            "confidence",
            optional_finite_number(self.confidence, "FixationEvent.confidence"),
        )
        object.__setattr__(
            self,
            "native_timing",
            freeze_json_mapping(self.native_timing, "FixationEvent.native_timing"),
        )
        object.__setattr__(
            self,
            "native_metadata",
            freeze_json_mapping(
                self.native_metadata,
                "FixationEvent.native_metadata",
            ),
        )

    def to_dict(self) -> dict[str, object]:

        return {
            "x_px": self.x_px,
            "y_px": self.y_px,
            "x_norm": self.x_norm,
            "y_norm": self.y_norm,
            "sequence_index": self.sequence_index,
            "timestamp_s": self.timestamp_s,
            "duration_s": self.duration_s,
            "confidence": self.confidence,
            "native_timing": thaw_json_value(self.native_timing),
            "native_metadata": thaw_json_value(self.native_metadata),
        }

    @classmethod
    def from_dict(cls, data: object) -> "FixationEvent":

        names = frozenset(
            {
                "x_px",
                "y_px",
                "x_norm",
                "y_norm",
                "sequence_index",
                "timestamp_s",
                "duration_s",
                "confidence",
                "native_timing",
                "native_metadata",
            }
        )
        fields = validate_dict_keys(data, "FixationEvent", required=names)
        return cls(**{name: fields[name] for name in names})  # type: ignore[arg-type]


class StoppingReason(str, Enum):

    COMPLETED = "completed"
    MAX_FIXATIONS = "max_fixations"
    TIME_HORIZON = "time_horizon"
    MODEL_STOP = "model_stop"
    ERROR = "error"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class PredictionRecord:

    CURRENT_SCHEMA_VERSION: ClassVar[int] = 1

    run_id: str
    request_id: str
    model: ModelIdentity
    dataset_id: str
    dataset_split: str
    item_id: str
    sample_id: str
    sample_index: int
    seed: int
    schema_version: int = CURRENT_SCHEMA_VERSION
    observer_id: str | None = None
    task_text: str | None = None
    target_description: str | None = None
    fixations: tuple[FixationEvent, ...] = ()
    saliency_artifact: ArtifactReference | None = None
    explanation_text: str | None = None
    native_artifacts: tuple[ArtifactReference, ...] = ()
    warnings: tuple[str, ...] = ()
    native_metadata: JsonMapping = field(default_factory=dict)
    stopping_reason: StoppingReason | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "schema_version",
            validate_schema_version(
                self.schema_version,
                "PredictionRecord.schema_version",
            ),
        )
        for name in (
            "run_id",
            "request_id",
            "dataset_id",
            "dataset_split",
            "item_id",
            "sample_id",
        ):
            object.__setattr__(
                self,
                name,
                require_string(getattr(self, name), f"PredictionRecord.{name}"),
            )
        if not isinstance(self.model, ModelIdentity):
            raise ContractValidationError(
                "PredictionRecord.model",
                "must be a ModelIdentity",
            )
        object.__setattr__(
            self,
            "sample_index",
            require_int(
                self.sample_index,
                "PredictionRecord.sample_index",
                minimum=0,
            ),
        )
        object.__setattr__(
            self,
            "seed",
            require_int(self.seed, "PredictionRecord.seed", minimum=0),
        )
        for name in ("observer_id", "task_text", "target_description"):
            object.__setattr__(
                self,
                name,
                optional_string(
                    getattr(self, name),
                    f"PredictionRecord.{name}",
                    allow_empty=name != "observer_id",
                ),
            )
        if not isinstance(self.fixations, Sequence) or isinstance(
            self.fixations,
            (str, bytes, bytearray),
        ):
            raise ContractValidationError(
                "PredictionRecord.fixations",
                "must be a sequence of FixationEvent values",
            )
        fixations = tuple(self.fixations)
        for index, fixation in enumerate(fixations):
            if not isinstance(fixation, FixationEvent):
                raise ContractValidationError(
                    f"PredictionRecord.fixations[{index}]",
                    "must be a FixationEvent",
                )
        object.__setattr__(self, "fixations", fixations)
        self._validate_timestamp_order()

        if self.saliency_artifact is not None and not isinstance(
            self.saliency_artifact,
            ArtifactReference,
        ):
            raise ContractValidationError(
                "PredictionRecord.saliency_artifact",
                "must be an ArtifactReference or null",
            )
        object.__setattr__(
            self,
            "explanation_text",
            optional_string(
                self.explanation_text,
                "PredictionRecord.explanation_text",
                allow_empty=True,
            ),
        )
        if not isinstance(self.native_artifacts, Sequence) or isinstance(
            self.native_artifacts,
            (str, bytes, bytearray),
        ):
            raise ContractValidationError(
                "PredictionRecord.native_artifacts",
                "must be a sequence of ArtifactReference values",
            )
        native_artifacts = tuple(self.native_artifacts)
        for index, artifact in enumerate(native_artifacts):
            if not isinstance(artifact, ArtifactReference):
                raise ContractValidationError(
                    f"PredictionRecord.native_artifacts[{index}]",
                    "must be an ArtifactReference",
                )
        object.__setattr__(self, "native_artifacts", native_artifacts)

        if not isinstance(self.warnings, Sequence) or isinstance(
            self.warnings,
            (str, bytes, bytearray),
        ):
            raise ContractValidationError(
                "PredictionRecord.warnings",
                "must be a sequence of strings",
            )
        warnings = tuple(
            require_string(warning, f"PredictionRecord.warnings[{index}]")
            for index, warning in enumerate(self.warnings)
        )
        object.__setattr__(self, "warnings", warnings)
        object.__setattr__(
            self,
            "native_metadata",
            freeze_json_mapping(
                self.native_metadata,
                "PredictionRecord.native_metadata",
            ),
        )
        if self.stopping_reason is not None and not isinstance(
            self.stopping_reason,
            StoppingReason,
        ):
            raise ContractValidationError(
                "PredictionRecord.stopping_reason",
                "must be a StoppingReason or null",
            )

    def _validate_timestamp_order(self) -> None:
        previous: float | None = None
        for index, fixation in enumerate(self.fixations):
            timestamp = fixation.timestamp_s
            if timestamp is None:
                continue
            if previous is not None and timestamp < previous:
                raise ContractValidationError(
                    f"PredictionRecord.fixations[{index}].timestamp_s",
                    "must be non-decreasing in fixation list order",
                )
            previous = timestamp

    def validate_for_capabilities(self, capabilities: CapabilitySet) -> None:

        if self.fixations and not capabilities.supports(Capability.PRODUCES_SCANPATHS):
            raise ContractValidationError(
                "PredictionRecord.fixations",
                "adapter did not report scanpath output capability",
            )
        if self.saliency_artifact is not None and not capabilities.supports(
            Capability.PRODUCES_SALIENCY_MAPS
        ):
            raise ContractValidationError(
                "PredictionRecord.saliency_artifact",
                "adapter did not report saliency output capability",
            )
        if self.explanation_text is not None and not capabilities.supports(
            Capability.TEXT_EXPLANATION_OUTPUT
        ):
            raise ContractValidationError(
                "PredictionRecord.explanation_text",
                "adapter did not report text output capability",
            )

        timestamps = [fixation.timestamp_s for fixation in self.fixations]
        has_timestamp = any(value is not None for value in timestamps)
        if has_timestamp and not capabilities.supports(
            Capability.CONTINUOUS_TIME_OUTPUT
        ):
            raise ContractValidationError(
                "PredictionRecord.fixations.timestamp_s",
                "adapter did not report continuous-time output capability",
            )
        if (
            capabilities.supports(Capability.CONTINUOUS_TIME_OUTPUT)
            and has_timestamp
            and any(value is None for value in timestamps)
        ):
            raise ContractValidationError(
                "PredictionRecord.fixations.timestamp_s",
                "continuous-time output cannot mix present and missing timestamps",
            )
        has_duration = any(
            fixation.duration_s is not None for fixation in self.fixations
        )
        if has_duration and not capabilities.supports(
            Capability.FIXATION_DURATION_OUTPUT
        ):
            raise ContractValidationError(
                "PredictionRecord.fixations.duration_s",
                "adapter did not report fixation-duration output capability",
            )

    def to_dict(self) -> dict[str, object]:

        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "request_id": self.request_id,
            "model": self.model.to_dict(),
            "dataset_id": self.dataset_id,
            "dataset_split": self.dataset_split,
            "item_id": self.item_id,
            "sample_id": self.sample_id,
            "sample_index": self.sample_index,
            "seed": self.seed,
            "observer_id": self.observer_id,
            "task_text": self.task_text,
            "target_description": self.target_description,
            "fixations": [fixation.to_dict() for fixation in self.fixations],
            "saliency_artifact": (
                None
                if self.saliency_artifact is None
                else self.saliency_artifact.to_dict()
            ),
            "explanation_text": self.explanation_text,
            "native_artifacts": [
                artifact.to_dict() for artifact in self.native_artifacts
            ],
            "warnings": list(self.warnings),
            "native_metadata": thaw_json_value(self.native_metadata),
            "stopping_reason": (
                None if self.stopping_reason is None else self.stopping_reason.value
            ),
        }

    @classmethod
    def from_dict(cls, data: object) -> "PredictionRecord":

        names = frozenset(
            {
                "schema_version",
                "run_id",
                "request_id",
                "model",
                "dataset_id",
                "dataset_split",
                "item_id",
                "sample_id",
                "sample_index",
                "seed",
                "observer_id",
                "task_text",
                "target_description",
                "fixations",
                "saliency_artifact",
                "explanation_text",
                "native_artifacts",
                "warnings",
                "native_metadata",
                "stopping_reason",
            }
        )
        fields = validate_dict_keys(data, "PredictionRecord", required=names)
        raw_fixations = fields["fixations"]
        if not isinstance(raw_fixations, list):
            raise ContractValidationError(
                "PredictionRecord.fixations",
                "must be a JSON array",
            )
        raw_native = fields["native_artifacts"]
        if not isinstance(raw_native, list):
            raise ContractValidationError(
                "PredictionRecord.native_artifacts",
                "must be a JSON array",
            )
        raw_warnings = fields["warnings"]
        if not isinstance(raw_warnings, list):
            raise ContractValidationError(
                "PredictionRecord.warnings",
                "must be a JSON array",
            )
        raw_saliency = fields["saliency_artifact"]
        try:
            model = ModelIdentity.from_dict(fields["model"])
        except ContractValidationError as error:
            _raise_nested_error(
                error,
                parent="PredictionRecord.model",
                nested_type="ModelIdentity",
            )
        fixations: list[FixationEvent] = []
        for index, item in enumerate(raw_fixations):
            try:
                fixations.append(FixationEvent.from_dict(item))
            except ContractValidationError as error:
                _raise_nested_error(
                    error,
                    parent=f"PredictionRecord.fixations[{index}]",
                    nested_type="FixationEvent",
                )
        if raw_saliency is None:
            saliency_artifact = None
        else:
            try:
                saliency_artifact = ArtifactReference.from_dict(raw_saliency)
            except ContractValidationError as error:
                _raise_nested_error(
                    error,
                    parent="PredictionRecord.saliency_artifact",
                    nested_type="ArtifactReference",
                )
        native_artifacts: list[ArtifactReference] = []
        for index, item in enumerate(raw_native):
            try:
                native_artifacts.append(ArtifactReference.from_dict(item))
            except ContractValidationError as error:
                _raise_nested_error(
                    error,
                    parent=f"PredictionRecord.native_artifacts[{index}]",
                    nested_type="ArtifactReference",
                )
        raw_reason = fields["stopping_reason"]
        if raw_reason is None:
            stopping_reason = None
        else:
            reason = require_string(raw_reason, "PredictionRecord.stopping_reason")
            try:
                stopping_reason = StoppingReason(reason)
            except ValueError as error:
                raise ContractValidationError(
                    "PredictionRecord.stopping_reason",
                    f"unknown stopping reason '{reason}'",
                ) from error
        return cls(
            schema_version=fields["schema_version"],  # type: ignore[arg-type]
            run_id=fields["run_id"],  # type: ignore[arg-type]
            request_id=fields["request_id"],  # type: ignore[arg-type]
            model=model,
            dataset_id=fields["dataset_id"],  # type: ignore[arg-type]
            dataset_split=fields["dataset_split"],  # type: ignore[arg-type]
            item_id=fields["item_id"],  # type: ignore[arg-type]
            sample_id=fields["sample_id"],  # type: ignore[arg-type]
            sample_index=fields["sample_index"],  # type: ignore[arg-type]
            seed=fields["seed"],  # type: ignore[arg-type]
            observer_id=fields["observer_id"],  # type: ignore[arg-type]
            task_text=fields["task_text"],  # type: ignore[arg-type]
            target_description=fields["target_description"],  # type: ignore[arg-type]
            fixations=tuple(fixations),
            saliency_artifact=saliency_artifact,
            explanation_text=fields["explanation_text"],  # type: ignore[arg-type]
            native_artifacts=tuple(native_artifacts),
            warnings=tuple(raw_warnings),  # type: ignore[arg-type]
            native_metadata=fields["native_metadata"],  # type: ignore[arg-type]
            stopping_reason=stopping_reason,
        )
