
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import ClassVar

from activevision_workbench.contracts import ArtifactReference, FixationEvent
from activevision_workbench.contracts._validation import (
    JsonMapping,
    freeze_json_mapping,
    optional_string,
    require_finite_number,
    require_int,
    require_string,
    thaw_json_value,
    validate_dict_keys,
    validate_schema_version,
)
from activevision_workbench.errors import (
    ContractValidationError,
    DatasetAnnotationError,
    DatasetItemNotFoundError,
)


@dataclass(frozen=True, slots=True)
class TargetBoundingBox:

    x_px: float
    y_px: float
    width_px: float
    height_px: float
    target_label: str | None = None
    native_metadata: JsonMapping = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("x_px", "y_px"):
            object.__setattr__(
                self,
                name,
                require_finite_number(
                    getattr(self, name), f"TargetBoundingBox.{name}"
                ),
            )
        for name in ("width_px", "height_px"):
            value = require_finite_number(
                getattr(self, name), f"TargetBoundingBox.{name}", minimum=0.0
            )
            if value == 0.0:
                raise ContractValidationError(
                    f"TargetBoundingBox.{name}", "must be > 0"
                )
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "target_label",
            optional_string(
                self.target_label,
                "TargetBoundingBox.target_label",
                allow_empty=False,
            ),
        )
        object.__setattr__(
            self,
            "native_metadata",
            freeze_json_mapping(
                self.native_metadata, "TargetBoundingBox.native_metadata"
            ),
        )

    def to_dict(self) -> dict[str, object]:

        return {
            "x_px": self.x_px,
            "y_px": self.y_px,
            "width_px": self.width_px,
            "height_px": self.height_px,
            "target_label": self.target_label,
            "native_metadata": thaw_json_value(self.native_metadata),
        }

    @classmethod
    def from_dict(cls, data: object) -> "TargetBoundingBox":
        names = frozenset(
            {
                "x_px",
                "y_px",
                "width_px",
                "height_px",
                "target_label",
                "native_metadata",
            }
        )
        values = validate_dict_keys(data, "TargetBoundingBox", required=names)
        return cls(**{name: values[name] for name in names})  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class GroundTruthScanpath:

    scanpath_id: str
    fixations: tuple[FixationEvent, ...]
    observer_id: str | None = None
    observer_metadata: JsonMapping = field(default_factory=dict)
    task_text: str | None = None
    target_label: str | None = None
    target_metadata: JsonMapping = field(default_factory=dict)
    native_annotation: ArtifactReference | None = None
    warnings: tuple[str, ...] = ()
    native_metadata: JsonMapping = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "scanpath_id",
            require_string(self.scanpath_id, "GroundTruthScanpath.scanpath_id"),
        )
        if not isinstance(self.fixations, Sequence) or isinstance(
            self.fixations, (str, bytes, bytearray)
        ):
            raise ContractValidationError(
                "GroundTruthScanpath.fixations",
                "must be a sequence of FixationEvent values",
            )
        fixations = tuple(self.fixations)
        if not fixations:
            raise ContractValidationError(
                "GroundTruthScanpath.fixations", "must not be empty"
            )
        for index, fixation in enumerate(fixations):
            if not isinstance(fixation, FixationEvent):
                raise ContractValidationError(
                    f"GroundTruthScanpath.fixations[{index}]",
                    "must be a FixationEvent",
                )
        object.__setattr__(self, "fixations", fixations)
        for name in ("observer_id", "task_text", "target_label"):
            object.__setattr__(
                self,
                name,
                optional_string(
                    getattr(self, name),
                    f"GroundTruthScanpath.{name}",
                    allow_empty=name == "task_text",
                ),
            )
        for name in ("observer_metadata", "target_metadata", "native_metadata"):
            object.__setattr__(
                self,
                name,
                freeze_json_mapping(
                    getattr(self, name), f"GroundTruthScanpath.{name}"
                ),
            )
        if self.native_annotation is not None and not isinstance(
            self.native_annotation, ArtifactReference
        ):
            raise ContractValidationError(
                "GroundTruthScanpath.native_annotation",
                "must be an ArtifactReference or null",
            )
        object.__setattr__(
            self,
            "warnings",
            _string_tuple(self.warnings, "GroundTruthScanpath.warnings"),
        )

    def to_dict(self) -> dict[str, object]:

        return {
            "scanpath_id": self.scanpath_id,
            "fixations": [fixation.to_dict() for fixation in self.fixations],
            "observer_id": self.observer_id,
            "observer_metadata": thaw_json_value(self.observer_metadata),
            "task_text": self.task_text,
            "target_label": self.target_label,
            "target_metadata": thaw_json_value(self.target_metadata),
            "native_annotation": (
                None
                if self.native_annotation is None
                else self.native_annotation.to_dict()
            ),
            "warnings": list(self.warnings),
            "native_metadata": thaw_json_value(self.native_metadata),
        }

    @classmethod
    def from_dict(cls, data: object) -> "GroundTruthScanpath":
        names = frozenset(
            {
                "scanpath_id",
                "fixations",
                "observer_id",
                "observer_metadata",
                "task_text",
                "target_label",
                "target_metadata",
                "native_annotation",
                "warnings",
                "native_metadata",
            }
        )
        values = validate_dict_keys(data, "GroundTruthScanpath", required=names)
        raw_fixations = values["fixations"]
        if not isinstance(raw_fixations, list):
            raise ContractValidationError(
                "GroundTruthScanpath.fixations", "must be a JSON array"
            )
        fixations = tuple(
            _nested_fixation(value, index)
            for index, value in enumerate(raw_fixations)
        )
        native = values["native_annotation"]
        native_annotation = (
            None if native is None else ArtifactReference.from_dict(native)
        )
        raw_warnings = values["warnings"]
        if not isinstance(raw_warnings, list):
            raise ContractValidationError(
                "GroundTruthScanpath.warnings", "must be a JSON array"
            )
        return cls(
            scanpath_id=values["scanpath_id"],  # type: ignore[arg-type]
            fixations=fixations,
            observer_id=values["observer_id"],  # type: ignore[arg-type]
            observer_metadata=values["observer_metadata"],  # type: ignore[arg-type]
            task_text=values["task_text"],  # type: ignore[arg-type]
            target_label=values["target_label"],  # type: ignore[arg-type]
            target_metadata=values["target_metadata"],  # type: ignore[arg-type]
            native_annotation=native_annotation,
            warnings=tuple(raw_warnings),  # type: ignore[arg-type]
            native_metadata=values["native_metadata"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class DatasetItem:

    CURRENT_SCHEMA_VERSION: ClassVar[int] = 1

    dataset_id: str
    dataset_version: str
    official_split: str
    item_id: str
    image_id: str
    image_ref: str
    image_width: int
    image_height: int
    scanpaths: tuple[GroundTruthScanpath, ...]
    schema_version: int = CURRENT_SCHEMA_VERSION
    task_text: str | None = None
    target_label: str | None = None
    target_metadata: JsonMapping = field(default_factory=dict)
    target_boxes: tuple[TargetBoundingBox, ...] = ()
    target_masks: tuple[ArtifactReference, ...] = ()
    native_annotation_refs: tuple[ArtifactReference, ...] = ()
    warnings: tuple[str, ...] = ()
    native_metadata: JsonMapping = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "schema_version",
            validate_schema_version(
                self.schema_version, "DatasetItem.schema_version"
            ),
        )
        for name in (
            "dataset_id",
            "dataset_version",
            "official_split",
            "item_id",
            "image_id",
            "image_ref",
        ):
            object.__setattr__(
                self,
                name,
                require_string(getattr(self, name), f"DatasetItem.{name}"),
            )
        for name in ("image_width", "image_height"):
            object.__setattr__(
                self,
                name,
                require_int(
                    getattr(self, name), f"DatasetItem.{name}", minimum=2
                ),
            )
        object.__setattr__(
            self,
            "scanpaths",
            _typed_tuple(
                self.scanpaths,
                GroundTruthScanpath,
                "DatasetItem.scanpaths",
                allow_empty=False,
            ),
        )
        for name in ("task_text", "target_label"):
            object.__setattr__(
                self,
                name,
                optional_string(
                    getattr(self, name),
                    f"DatasetItem.{name}",
                    allow_empty=name == "task_text",
                ),
            )
        object.__setattr__(
            self,
            "target_metadata",
            freeze_json_mapping(
                self.target_metadata, "DatasetItem.target_metadata"
            ),
        )
        object.__setattr__(
            self,
            "target_boxes",
            _typed_tuple(
                self.target_boxes,
                TargetBoundingBox,
                "DatasetItem.target_boxes",
            ),
        )
        for name in ("target_masks", "native_annotation_refs"):
            object.__setattr__(
                self,
                name,
                _typed_tuple(
                    getattr(self, name),
                    ArtifactReference,
                    f"DatasetItem.{name}",
                ),
            )
        object.__setattr__(
            self, "warnings", _string_tuple(self.warnings, "DatasetItem.warnings")
        )
        object.__setattr__(
            self,
            "native_metadata",
            freeze_json_mapping(
                self.native_metadata, "DatasetItem.native_metadata"
            ),
        )

    @property
    def observer_ids(self) -> tuple[str, ...]:

        return tuple(
            sorted(
                {
                    scanpath.observer_id
                    for scanpath in self.scanpaths
                    if scanpath.observer_id is not None
                }
            )
        )

    def to_run_item_config(
        self, *, observer_id: str | None = None
    ) -> dict[str, object]:

        candidates = tuple(
            scanpath
            for scanpath in self.scanpaths
            if observer_id is None or scanpath.observer_id == observer_id
        )
        if observer_id is None and len(candidates) != 1:
            raise DatasetAnnotationError(
                "observer_id is required when a dataset item has multiple scanpaths"
            )
        if not candidates:
            raise DatasetItemNotFoundError(
                f"item '{self.item_id}' has no scanpath for observer "
                f"'{observer_id}'"
            )
        if len(candidates) != 1:
            raise DatasetAnnotationError(
                f"item '{self.item_id}' has multiple scanpaths for observer "
                f"'{observer_id}'"
            )
        scanpath = candidates[0]
        return {
            "item_id": self.item_id,
            "image_ref": self.image_ref,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "task_text": (
                scanpath.task_text
                if scanpath.task_text is not None
                else self.task_text
            ),
            "target_description": (
                scanpath.target_label
                if scanpath.target_label is not None
                else self.target_label
            ),
            "observer_id": scanpath.observer_id,
            "observer_metadata": thaw_json_value(scanpath.observer_metadata),
        }

    def to_dict(self) -> dict[str, object]:

        return {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
            "official_split": self.official_split,
            "item_id": self.item_id,
            "image_id": self.image_id,
            "image_ref": self.image_ref,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "scanpaths": [scanpath.to_dict() for scanpath in self.scanpaths],
            "task_text": self.task_text,
            "target_label": self.target_label,
            "target_metadata": thaw_json_value(self.target_metadata),
            "target_boxes": [box.to_dict() for box in self.target_boxes],
            "target_masks": [mask.to_dict() for mask in self.target_masks],
            "native_annotation_refs": [
                reference.to_dict() for reference in self.native_annotation_refs
            ],
            "warnings": list(self.warnings),
            "native_metadata": thaw_json_value(self.native_metadata),
        }

    @classmethod
    def from_dict(cls, data: object) -> "DatasetItem":
        names = frozenset(
            {
                "schema_version",
                "dataset_id",
                "dataset_version",
                "official_split",
                "item_id",
                "image_id",
                "image_ref",
                "image_width",
                "image_height",
                "scanpaths",
                "task_text",
                "target_label",
                "target_metadata",
                "target_boxes",
                "target_masks",
                "native_annotation_refs",
                "warnings",
                "native_metadata",
            }
        )
        values = validate_dict_keys(data, "DatasetItem", required=names)
        arrays = {}
        for name in (
            "scanpaths",
            "target_boxes",
            "target_masks",
            "native_annotation_refs",
            "warnings",
        ):
            value = values[name]
            if not isinstance(value, list):
                raise ContractValidationError(
                    f"DatasetItem.{name}", "must be a JSON array"
                )
            arrays[name] = value
        return cls(
            schema_version=values["schema_version"],  # type: ignore[arg-type]
            dataset_id=values["dataset_id"],  # type: ignore[arg-type]
            dataset_version=values["dataset_version"],  # type: ignore[arg-type]
            official_split=values["official_split"],  # type: ignore[arg-type]
            item_id=values["item_id"],  # type: ignore[arg-type]
            image_id=values["image_id"],  # type: ignore[arg-type]
            image_ref=values["image_ref"],  # type: ignore[arg-type]
            image_width=values["image_width"],  # type: ignore[arg-type]
            image_height=values["image_height"],  # type: ignore[arg-type]
            scanpaths=tuple(
                GroundTruthScanpath.from_dict(value)
                for value in arrays["scanpaths"]
            ),
            task_text=values["task_text"],  # type: ignore[arg-type]
            target_label=values["target_label"],  # type: ignore[arg-type]
            target_metadata=values["target_metadata"],  # type: ignore[arg-type]
            target_boxes=tuple(
                TargetBoundingBox.from_dict(value)
                for value in arrays["target_boxes"]
            ),
            target_masks=tuple(
                ArtifactReference.from_dict(value)
                for value in arrays["target_masks"]
            ),
            native_annotation_refs=tuple(
                ArtifactReference.from_dict(value)
                for value in arrays["native_annotation_refs"]
            ),
            warnings=tuple(arrays["warnings"]),  # type: ignore[arg-type]
            native_metadata=values["native_metadata"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class DatasetMetadata:

    dataset_id: str
    dataset_version: str
    adapter_version: str
    data_root: str
    available_splits: tuple[str, ...]
    item_counts: JsonMapping
    observer_ids: tuple[str, ...] = ()
    target_labels: tuple[str, ...] = ()
    annotation_references: tuple[ArtifactReference, ...] = ()
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("dataset_id", "dataset_version", "adapter_version", "data_root"):
            object.__setattr__(
                self,
                name,
                require_string(getattr(self, name), f"DatasetMetadata.{name}"),
            )
        object.__setattr__(
            self,
            "available_splits",
            _string_tuple(
                self.available_splits, "DatasetMetadata.available_splits"
            ),
        )
        if not self.available_splits:
            raise ContractValidationError(
                "DatasetMetadata.available_splits", "must not be empty"
            )
        object.__setattr__(
            self,
            "item_counts",
            freeze_json_mapping(self.item_counts, "DatasetMetadata.item_counts"),
        )
        for split in self.available_splits:
            count = self.item_counts.get(split)
            require_int(count, f"DatasetMetadata.item_counts.{split}", minimum=0)
        for name in ("observer_ids", "target_labels", "warnings"):
            object.__setattr__(
                self,
                name,
                _string_tuple(getattr(self, name), f"DatasetMetadata.{name}"),
            )
        object.__setattr__(
            self,
            "annotation_references",
            _typed_tuple(
                self.annotation_references,
                ArtifactReference,
                "DatasetMetadata.annotation_references",
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
            "adapter_version": self.adapter_version,
            "data_root": self.data_root,
            "available_splits": list(self.available_splits),
            "item_counts": thaw_json_value(self.item_counts),
            "observer_ids": list(self.observer_ids),
            "target_labels": list(self.target_labels),
            "annotation_references": [
                reference.to_dict() for reference in self.annotation_references
            ],
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, slots=True)
class DatasetManifest:

    dataset_id: str
    dataset_version: str
    configured_root: str
    adapter_version: str
    annotation_fingerprints: JsonMapping
    split_definition_fingerprint: str
    item_counts: JsonMapping
    created_at: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "schema_version",
            validate_schema_version(
                self.schema_version, "DatasetManifest.schema_version"
            ),
        )
        for name in (
            "dataset_id",
            "dataset_version",
            "configured_root",
            "adapter_version",
            "split_definition_fingerprint",
            "created_at",
        ):
            object.__setattr__(
                self,
                name,
                require_string(getattr(self, name), f"DatasetManifest.{name}"),
            )
        for name in ("annotation_fingerprints", "item_counts"):
            object.__setattr__(
                self,
                name,
                freeze_json_mapping(
                    getattr(self, name), f"DatasetManifest.{name}"
                ),
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
            "configured_root": self.configured_root,
            "adapter_version": self.adapter_version,
            "annotation_fingerprints": thaw_json_value(
                self.annotation_fingerprints
            ),
            "split_definition_fingerprint": self.split_definition_fingerprint,
            "item_counts": thaw_json_value(self.item_counts),
            "created_at": self.created_at,
        }


def _nested_fixation(value: object, index: int) -> FixationEvent:
    try:
        return FixationEvent.from_dict(value)
    except ContractValidationError as error:
        prefix = "FixationEvent."
        suffix = (
            error.field[len(prefix) :]
            if error.field.startswith(prefix)
            else error.field
        )
        raise ContractValidationError(
            f"GroundTruthScanpath.fixations[{index}].{suffix}", error.reason
        ) from error


def _typed_tuple(
    value: object,
    expected: type,
    field_name: str,
    *,
    allow_empty: bool = True,
) -> tuple:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ContractValidationError(field_name, "must be a sequence")
    result = tuple(value)
    if not allow_empty and not result:
        raise ContractValidationError(field_name, "must not be empty")
    for index, item in enumerate(result):
        if not isinstance(item, expected):
            raise ContractValidationError(
                f"{field_name}[{index}]", f"must be a {expected.__name__}"
            )
    return result


def _string_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ContractValidationError(field_name, "must be a sequence of strings")
    return tuple(
        require_string(item, f"{field_name}[{index}]")
        for index, item in enumerate(value)
    )
