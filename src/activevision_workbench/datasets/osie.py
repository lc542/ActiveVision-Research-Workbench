
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from activevision_workbench.contracts import ArtifactReference
from activevision_workbench.datasets.base import (
    DatasetAdapter,
    load_json,
    read_image_dimensions,
    relative_reference,
)
from activevision_workbench.datasets.contracts import (
    DatasetItem,
    DatasetMetadata,
    GroundTruthScanpath,
)
from activevision_workbench.datasets.coordinates import source_fixation_from_pixel
from activevision_workbench.errors import (
    DatasetAnnotationError,
    DatasetDependencyError,
)


class OSIEDatasetAdapter(DatasetAdapter):

    dataset_id = "osie"
    adapter_version = "0.1.0"

    def __init__(
        self,
        data_root: str | Path,
        *,
        dataset_version: str = "source-release-unversioned",
        split_definition_path: str | Path | None = None,
        duration_unit: str | None = None,
        clip_out_of_bounds: bool = False,
    ) -> None:
        super().__init__(data_root)
        if type(dataset_version) is not str or not dataset_version:
            raise DatasetAnnotationError("dataset_version must be non-empty")
        if duration_unit not in {None, "milliseconds", "seconds"}:
            raise DatasetAnnotationError(
                "duration_unit must be null, 'milliseconds', or 'seconds'"
            )
        if type(clip_out_of_bounds) is not bool:
            raise DatasetAnnotationError("clip_out_of_bounds must be a boolean")
        self.dataset_version = dataset_version
        self.duration_unit = duration_unit
        self.clip_out_of_bounds = clip_out_of_bounds
        self.fixation_path = self.data_root / "eye" / "fixations.mat"
        self.stimuli_path = self.data_root / "stimuli"
        if not self.fixation_path.is_file():
            raise DatasetAnnotationError(
                f"OSIE fixation annotation is missing: {self.fixation_path}"
            )
        if not self.stimuli_path.is_dir():
            raise DatasetAnnotationError(
                f"OSIE stimuli directory is missing: {self.stimuli_path}"
            )
        images = tuple(sorted(self.stimuli_path.glob("*.jpg"), key=lambda p: p.name))
        if not images:
            raise DatasetAnnotationError("OSIE stimuli directory contains no JPEGs")
        self._images = {path.stem: path for path in images}
        self.split_definition_path = (
            None
            if split_definition_path is None
            else Path(split_definition_path).expanduser().resolve(strict=False)
        )
        self._splits = self._load_splits()
        self._annotations_by_id: dict[str, tuple[int, object]] | None = None
        annotations = [self.fixation_path]
        attributes = self.data_root / "attrs.mat"
        if attributes.is_file():
            annotations.append(attributes)
        if self.split_definition_path is not None:
            annotations.append(self.split_definition_path)
        self._annotation_paths = tuple(annotations)

    @property
    def metadata(self) -> DatasetMetadata:
        warnings = [
            "OSIE subject IDs are derived from one-based subject-array positions.",
            "OSIE MATLAB one-based image coordinates are converted to the "
            "canonical zero-based pixel-center lattice; raw values are retained.",
        ]
        if self.split_definition_path is None:
            warnings.append(
                "The inspected OSIE release contains no official split file; only "
                "the explicit 'all' split is available."
            )
        if self.duration_unit is None:
            warnings.append(
                "fix_duration unit is unspecified by the local annotation; raw "
                "values are preserved and duration_s is null."
            )
        references = tuple(
            ArtifactReference(
                uri=relative_reference(path, self.data_root),
                kind=(
                    "dataset_split_definition"
                    if path == self.split_definition_path
                    else "dataset_annotation"
                ),
            )
            for path in self._annotation_paths
        )
        return DatasetMetadata(
            dataset_id=self.dataset_id,
            dataset_version=self.dataset_version,
            adapter_version=self.adapter_version,
            data_root=str(self.data_root),
            available_splits=tuple(self._splits),
            item_counts={split: len(ids) for split, ids in self._splits.items()},
            observer_ids=tuple(f"subject-{index:02d}" for index in range(1, 16)),
            annotation_references=references,
            warnings=tuple(warnings),
        )

    @property
    def annotation_paths(self) -> tuple[Path, ...]:
        return self._annotation_paths

    def _item_ids_for_split(self, split: str) -> tuple[str, ...]:
        return self._splits[split]

    def _load_item(self, item_id: str, split: str) -> DatasetItem:
        annotations = self._annotations()
        try:
            item_index, annotation = annotations[item_id]
        except KeyError as error:  # pragma: no cover - index validation guards it
            raise DatasetAnnotationError(
                f"OSIE fixation annotation is missing item '{item_id}'"
            ) from error
        image_path = self._images[item_id]
        width, height = read_image_dimensions(image_path)
        subjects = _as_sequence(getattr(annotation, "subjects", None), "subjects")
        scanpaths: list[GroundTruthScanpath] = []
        item_warnings: set[str] = {
            "OSIE observer IDs are derived from source subject-array positions.",
            "OSIE MATLAB one-based image coordinates were converted to the "
            "canonical zero-based pixel-center lattice.",
        }
        if self.duration_unit is None:
            item_warnings.add(
                "OSIE fix_duration unit is source-unspecified; canonical duration_s "
                "is unavailable."
            )
        for subject_index, subject in enumerate(subjects):
            xs = _numeric_sequence(getattr(subject, "fix_x", None), "fix_x")
            ys = _numeric_sequence(getattr(subject, "fix_y", None), "fix_y")
            durations = _numeric_sequence(
                getattr(subject, "fix_duration", None), "fix_duration"
            )
            if not (len(xs) == len(ys) == len(durations)) or not xs:
                raise DatasetAnnotationError(
                    f"OSIE item '{item_id}' subject {subject_index + 1} has "
                    "misaligned or empty fixation arrays"
                )
            fixation_events = []
            scanpath_warnings: set[str] = set()
            for sequence_index, (x_value, y_value, raw_duration) in enumerate(
                zip(xs, ys, durations, strict=True)
            ):
                duration_s = _duration_seconds(raw_duration, self.duration_unit)
                fixation, warning = source_fixation_from_pixel(
                    x_px=x_value - 1.0,
                    y_px=y_value - 1.0,
                    image_width=width,
                    image_height=height,
                    sequence_index=sequence_index,
                    duration_s=duration_s,
                    native_timing={
                        "duration": raw_duration,
                        "unit": self.duration_unit or "source_unspecified",
                        "source_field": "fix_duration",
                    },
                    native_metadata={
                        "source_x_px": x_value,
                        "source_y_px": y_value,
                        "source_coordinate_convention": (
                            "matlab_one_based_pixel_centers"
                        ),
                        "canonical_coordinate_transform": "subtract_one",
                    },
                    clip_out_of_bounds=self.clip_out_of_bounds,
                )
                fixation_events.append(fixation)
                if warning is not None:
                    scanpath_warnings.add(warning)
                    item_warnings.add(warning)
            native_uri = (
                "eye/fixations.mat#"
                f"fixations[{item_index}].subjects[{subject_index}]"
            )
            scanpaths.append(
                GroundTruthScanpath(
                    scanpath_id=f"osie:{item_id}:subject-{subject_index + 1:02d}",
                    observer_id=f"subject-{subject_index + 1:02d}",
                    observer_metadata={
                        "source_subject_index": subject_index,
                        "identity_kind": "derived_array_position",
                    },
                    fixations=tuple(fixation_events),
                    native_annotation=ArtifactReference(
                        uri=native_uri,
                        kind="gaze_annotation",
                        media_type="application/x-matlab-data",
                    ),
                    warnings=tuple(sorted(scanpath_warnings)),
                    native_metadata={"source_image_name": f"{item_id}.jpg"},
                )
            )
        references = [
            ArtifactReference(
                uri=f"eye/fixations.mat#fixations[{item_index}]",
                kind="gaze_annotation",
                media_type="application/x-matlab-data",
            )
        ]
        if (self.data_root / "attrs.mat").is_file():
            references.append(
                ArtifactReference(
                    uri=f"attrs.mat#image_index={item_index}",
                    kind="native_semantic_attributes",
                    media_type="application/x-matlab-data",
                )
            )
        return DatasetItem(
            dataset_id=self.dataset_id,
            dataset_version=self.dataset_version,
            official_split=split,
            item_id=item_id,
            image_id=item_id,
            image_ref=str(image_path),
            image_width=width,
            image_height=height,
            scanpaths=tuple(scanpaths),
            native_annotation_refs=tuple(references),
            warnings=tuple(sorted(item_warnings)),
            native_metadata={
                "source_image_name": f"{item_id}.jpg",
                "source_fixation_item_index": item_index,
            },
        )

    def _annotations(self) -> dict[str, tuple[int, object]]:
        if self._annotations_by_id is not None:
            return self._annotations_by_id
        raw = _load_fixations_mat(self.fixation_path)
        values = _as_sequence(raw, "fixations")
        annotations: dict[str, tuple[int, object]] = {}
        for index, annotation in enumerate(values):
            image_name = getattr(annotation, "img", None)
            if type(image_name) is not str or not image_name:
                raise DatasetAnnotationError(
                    f"OSIE fixations[{index}].img must be a filename"
                )
            item_id = Path(image_name).stem
            if item_id in annotations:
                raise DatasetAnnotationError(
                    f"duplicate OSIE fixation item '{item_id}'"
                )
            annotations[item_id] = (index, annotation)
        if set(annotations) != set(self._images):
            raise DatasetAnnotationError(
                "OSIE image names and fixation annotation names do not match"
            )
        self._annotations_by_id = annotations
        return annotations

    def _load_splits(self) -> dict[str, tuple[str, ...]]:
        image_ids = tuple(sorted(self._images))
        if self.split_definition_path is None:
            return {"all": image_ids}
        if not self.split_definition_path.is_file():
            raise DatasetAnnotationError(
                f"OSIE split definition does not exist: {self.split_definition_path}"
            )
        raw = load_json(self.split_definition_path)
        if not isinstance(raw, Mapping) or set(raw) != {"schema_version", "splits"}:
            raise DatasetAnnotationError(
                "OSIE split definition must contain schema_version and splits"
            )
        if raw["schema_version"] != 1 or not isinstance(raw["splits"], Mapping):
            raise DatasetAnnotationError("invalid OSIE split definition schema")
        splits: dict[str, tuple[str, ...]] = {}
        assigned: set[str] = set()
        for split, values in raw["splits"].items():
            if type(split) is not str or not split or not isinstance(values, list):
                raise DatasetAnnotationError("invalid OSIE split name or item list")
            ids = tuple(sorted(_source_id(value) for value in values))
            if not ids or len(ids) != len(set(ids)):
                raise DatasetAnnotationError(
                    f"OSIE split '{split}' is empty or contains duplicate IDs"
                )
            unknown = set(ids) - set(image_ids)
            overlap = assigned.intersection(ids)
            if unknown or overlap:
                value = sorted(unknown or overlap)[0]
                raise DatasetAnnotationError(
                    f"OSIE split '{split}' has unknown/duplicate item '{value}'"
                )
            assigned.update(ids)
            splits[split] = ids
        return dict(sorted(splits.items()))


def _source_id(value: object) -> str:
    if type(value) is not str or not value:
        raise DatasetAnnotationError("OSIE split item IDs must be strings")
    return Path(value).stem


def _as_sequence(value: object, field: str) -> tuple[object, ...]:
    if value is None:
        raise DatasetAnnotationError(f"OSIE {field} is missing")
    if isinstance(value, (list, tuple)):
        return tuple(value)
    reshape = getattr(value, "reshape", None)
    if callable(reshape):
        try:
            flattened = reshape(-1).tolist()
        except (AttributeError, TypeError, ValueError) as error:
            raise DatasetAnnotationError(
                f"OSIE {field} cannot be flattened"
            ) from error
        return tuple(flattened) if isinstance(flattened, list) else (flattened,)
    return (value,)


def _load_fixations_mat(path: Path) -> object:
    try:
        from scipy.io import loadmat
    except ImportError as error:
        raise DatasetDependencyError(
            "OSIE item loading requires optional dependency 'scipy'; metadata "
            "inspection and manifest creation remain available without it"
        ) from error
    try:
        return loadmat(path, squeeze_me=True, struct_as_record=False)["fixations"]
    except Exception as error:
        raise DatasetAnnotationError(
            f"cannot load OSIE fixation MAT file: {error}"
        ) from error


def _numeric_sequence(value: object, field: str) -> tuple[float, ...]:
    values = _as_sequence(value, field)
    result: list[float] = []
    for index, item in enumerate(values):
        if type(item) not in (int, float) and not hasattr(item, "item"):
            raise DatasetAnnotationError(
                f"OSIE {field}[{index}] must be numeric"
            )
        result.append(float(item))
    return tuple(result)


def _duration_seconds(value: float, unit: str | None) -> float | None:
    if unit is None:
        return None
    return value / 1000.0 if unit == "milliseconds" else value
