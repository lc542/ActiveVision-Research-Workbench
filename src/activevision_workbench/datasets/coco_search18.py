
from __future__ import annotations

from collections.abc import Mapping, Sequence
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
    TargetBoundingBox,
)
from activevision_workbench.datasets.coordinates import source_fixation_from_pixel
from activevision_workbench.errors import DatasetAnnotationError


class COCOSearch18DatasetAdapter(DatasetAdapter):

    dataset_id = "coco_search18"
    adapter_version = "0.1.0"

    def __init__(
        self,
        data_root: str | Path,
        *,
        split_definition: str = "split1",
        dataset_version: str = "1.0-target-present",
        duration_unit: str | None = None,
        clip_out_of_bounds: bool = False,
    ) -> None:
        super().__init__(data_root)
        if split_definition not in {"split1", "split2"}:
            raise DatasetAnnotationError(
                "split_definition must be 'split1' or 'split2'"
            )
        if type(dataset_version) is not str or not dataset_version:
            raise DatasetAnnotationError("dataset_version must be non-empty")
        if duration_unit not in {None, "milliseconds", "seconds"}:
            raise DatasetAnnotationError(
                "duration_unit must be null, 'milliseconds', or 'seconds'"
            )
        if type(clip_out_of_bounds) is not bool:
            raise DatasetAnnotationError("clip_out_of_bounds must be a boolean")
        self.dataset_version = dataset_version
        self.split_definition = split_definition
        self.duration_unit = duration_unit
        self.clip_out_of_bounds = clip_out_of_bounds
        self.images_path = self.data_root / "images"
        if not self.images_path.is_dir():
            raise DatasetAnnotationError(
                f"COCO-Search18 images directory is missing: {self.images_path}"
            )
        self._files = {
            "train": self.data_root
            / f"coco_search18_fixations_TP_train_{split_definition}.json",
            "validation": self.data_root
            / f"coco_search18_fixations_TP_validation_{split_definition}.json",
        }
        for path in self._files.values():
            if not path.is_file():
                raise DatasetAnnotationError(
                    f"COCO-Search18 annotation is missing: {path}"
                )
        self._groups: dict[str, dict[str, tuple[_Trial, ...]]] | None = None
        self._observers: tuple[str, ...] = ()
        self._targets: tuple[str, ...] = ()

    @property
    def metadata(self) -> DatasetMetadata:
        groups = self._index()
        warnings = [
            "COCO-Search18 v1.0 release provides target-present train and "
            "validation gaze only; test fixations are withheld."
        ]
        if self.duration_unit is None:
            warnings.append(
                "T and RT units are not declared by the bundled README; raw values "
                "are preserved and canonical seconds are unavailable."
            )
        references = tuple(
            ArtifactReference(
                uri=relative_reference(path, self.data_root),
                kind="dataset_annotation",
                media_type="application/json",
                metadata={"split_definition": self.split_definition},
            )
            for path in self._files.values()
        )
        return DatasetMetadata(
            dataset_id=self.dataset_id,
            dataset_version=self.dataset_version,
            adapter_version=self.adapter_version,
            data_root=str(self.data_root),
            available_splits=("train", "validation"),
            item_counts={split: len(values) for split, values in groups.items()},
            observer_ids=self._observers,
            target_labels=self._targets,
            annotation_references=references,
            warnings=tuple(warnings),
        )

    @property
    def annotation_paths(self) -> tuple[Path, ...]:
        return tuple(self._files.values())

    def _item_ids_for_split(self, split: str) -> tuple[str, ...]:
        return tuple(self._index()[split])

    def _load_item(self, item_id: str, split: str) -> DatasetItem:
        trials = self._index()[split][item_id]
        first = trials[0].record
        task = _string(first["task"], f"{trials[0].path}:task")
        image_name = _string(first["name"], f"{trials[0].path}:name")
        image_path = self.images_path / task / image_name
        width, height = read_image_dimensions(image_path)
        conditions = {
            _string(trial.record["condition"], "condition")
            for trial in trials
        }
        if len(conditions) != 1:
            raise DatasetAnnotationError(
                f"COCO-Search18 item '{item_id}' mixes target conditions"
            )
        condition = next(iter(conditions))
        boxes: dict[tuple[float, float, float, float], TargetBoundingBox] = {}
        scanpaths: list[GroundTruthScanpath] = []
        item_warnings: set[str] = set()
        if self.duration_unit is None:
            item_warnings.add(
                "COCO-Search18 T/RT units are source-unspecified; canonical "
                "duration_s and reaction-time seconds are unavailable."
            )
        annotation_refs: list[ArtifactReference] = []
        for trial in trials:
            record = trial.record
            bbox = _numeric_list(record["bbox"], "bbox", expected_length=4)
            box_key = tuple(bbox)
            boxes[box_key] = TargetBoundingBox(
                x_px=bbox[0],
                y_px=bbox[1],
                width_px=bbox[2],
                height_px=bbox[3],
                target_label=task,
                native_metadata={
                    "source_format": "xywh",
                    "coordinate_space": "resized_padded_image_pixels",
                },
            )
            xs = _numeric_list(record["X"], "X")
            ys = _numeric_list(record["Y"], "Y")
            durations = _numeric_list(record["T"], "T")
            length = _integer(record["length"], "length", minimum=1)
            if not (len(xs) == len(ys) == len(durations) == length):
                raise DatasetAnnotationError(
                    f"{trial.path.name} record {trial.index} has misaligned "
                    "X/Y/T/length values"
                )
            subject = _integer(record["subject"], "subject", minimum=1)
            fixations = []
            scanpath_warnings: set[str] = set()
            for sequence_index, (x_value, y_value, raw_duration) in enumerate(
                zip(xs, ys, durations, strict=True)
            ):
                duration_s = _duration_seconds(raw_duration, self.duration_unit)
                fixation, warning = source_fixation_from_pixel(
                    x_px=x_value,
                    y_px=y_value,
                    image_width=width,
                    image_height=height,
                    sequence_index=sequence_index,
                    duration_s=duration_s,
                    native_timing={
                        "duration": raw_duration,
                        "unit": self.duration_unit or "source_unspecified",
                        "source_field": "T",
                    },
                    native_metadata={
                        "source_x_px": x_value,
                        "source_y_px": y_value,
                    },
                    clip_out_of_bounds=self.clip_out_of_bounds,
                )
                fixations.append(fixation)
                if warning is not None:
                    scanpath_warnings.add(warning)
                    item_warnings.add(warning)
            uri = f"{relative_reference(trial.path, self.data_root)}#/{trial.index}"
            reference = ArtifactReference(
                uri=uri,
                kind="gaze_annotation",
                media_type="application/json",
            )
            annotation_refs.append(reference)
            scanpaths.append(
                GroundTruthScanpath(
                    scanpath_id=f"{item_id}:subject-{subject:02d}",
                    observer_id=f"subject-{subject:02d}",
                    observer_metadata={"source_subject": subject},
                    task_text=f"search for {task}",
                    target_label=task,
                    target_metadata={
                        "condition": condition,
                        "target_present": condition == "present",
                    },
                    fixations=tuple(fixations),
                    native_annotation=reference,
                    warnings=tuple(sorted(scanpath_warnings)),
                    native_metadata={
                        "correct": _integer(record["correct"], "correct", minimum=0),
                        "reaction_time": _number(record["RT"], "RT"),
                        "reaction_time_unit": self.duration_unit
                        or "source_unspecified",
                        "source_length": length,
                        "source_split": record["split"],
                        "source_condition": condition,
                    },
                )
            )
        return DatasetItem(
            dataset_id=self.dataset_id,
            dataset_version=self.dataset_version,
            official_split=split,
            item_id=item_id,
            image_id=Path(image_name).stem,
            image_ref=str(image_path),
            image_width=width,
            image_height=height,
            scanpaths=tuple(scanpaths),
            task_text=f"search for {task}",
            target_label=task,
            target_metadata={
                "condition": condition,
                "target_present": condition == "present",
                "split_definition": self.split_definition,
            },
            target_boxes=tuple(boxes[key] for key in sorted(boxes)),
            native_annotation_refs=tuple(annotation_refs),
            warnings=tuple(sorted(item_warnings)),
            native_metadata={
                "source_image_name": image_name,
                "source_canvas": {"width": width, "height": height},
            },
        )

    def _index(self) -> dict[str, dict[str, tuple[_Trial, ...]]]:
        if self._groups is not None:
            return self._groups
        groups: dict[str, dict[str, list[_Trial]]] = {
            "train": {},
            "validation": {},
        }
        observers: set[str] = set()
        targets: set[str] = set()
        for split, path in self._files.items():
            raw = load_json(path)
            if not isinstance(raw, list):
                raise DatasetAnnotationError(f"{path} must contain a JSON array")
            source_split = "train" if split == "train" else "valid"
            for index, value in enumerate(raw):
                record = _validate_record(value, path, index, source_split)
                task = record["task"]
                name = record["name"]
                subject = record["subject"]
                assert isinstance(task, str)
                assert isinstance(name, str)
                assert isinstance(subject, int)
                item_id = (
                    f"{split}:{_safe_component(task)}:{Path(name).stem}"
                )
                groups[split].setdefault(item_id, []).append(
                    _Trial(path=path, index=index, record=record)
                )
                observers.add(f"subject-{subject:02d}")
                targets.add(task)
        frozen: dict[str, dict[str, tuple[_Trial, ...]]] = {}
        for split, values in groups.items():
            frozen[split] = {
                item_id: tuple(
                    sorted(
                        trials,
                        key=lambda trial: int(trial.record["subject"]),
                    )
                )
                for item_id, trials in sorted(values.items())
            }
        self._groups = frozen
        self._observers = tuple(sorted(observers))
        self._targets = tuple(sorted(targets))
        return frozen


class _Trial:
    __slots__ = ("path", "index", "record")

    def __init__(self, *, path: Path, index: int, record: dict[str, object]) -> None:
        self.path = path
        self.index = index
        self.record = record


def _validate_record(
    value: object,
    path: Path,
    index: int,
    expected_split: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise DatasetAnnotationError(f"{path.name}[{index}] must be an object")
    required = {
        "name",
        "subject",
        "task",
        "condition",
        "bbox",
        "X",
        "Y",
        "T",
        "length",
        "correct",
        "RT",
        "split",
    }
    missing = sorted(required - set(value))
    if missing:
        raise DatasetAnnotationError(
            f"{path.name}[{index}] missing field '{missing[0]}'"
        )
    record = dict(value)
    _string(record["name"], "name")
    _integer(record["subject"], "subject", minimum=1)
    _string(record["task"], "task")
    _string(record["condition"], "condition")
    _numeric_list(record["bbox"], "bbox", expected_length=4)
    _numeric_list(record["X"], "X")
    _numeric_list(record["Y"], "Y")
    _numeric_list(record["T"], "T")
    _integer(record["length"], "length", minimum=1)
    _integer(record["correct"], "correct", minimum=0)
    _number(record["RT"], "RT")
    if record["split"] != expected_split:
        raise DatasetAnnotationError(
            f"{path.name}[{index}] split is not '{expected_split}'"
        )
    return record


def _numeric_list(
    value: object, field: str, *, expected_length: int | None = None
) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(
        value, (str, bytes, bytearray)
    ):
        raise DatasetAnnotationError(f"{field} must be a numeric array")
    result = tuple(
        _number(item, f"{field}[{index}]")
        for index, item in enumerate(value)
    )
    if expected_length is not None and len(result) != expected_length:
        raise DatasetAnnotationError(
            f"{field} must contain exactly {expected_length} values"
        )
    return result


def _number(value: object, field: str) -> float:
    if type(value) not in (int, float):
        raise DatasetAnnotationError(f"{field} must be numeric")
    result = float(value)
    if not (-float("inf") < result < float("inf")):
        raise DatasetAnnotationError(f"{field} must be finite")
    return result


def _integer(value: object, field: str, *, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise DatasetAnnotationError(f"{field} must be an integer >= {minimum}")
    return value


def _string(value: object, field: str) -> str:
    if type(value) is not str or not value:
        raise DatasetAnnotationError(f"{field} must be a non-empty string")
    return value


def _duration_seconds(value: float, unit: str | None) -> float | None:
    if unit is None:
        return None
    return value / 1000.0 if unit == "milliseconds" else value


def _safe_component(value: str) -> str:
    component = "_".join(value.lower().split())
    if not component or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789_-"
        for character in component
    ):
        raise DatasetAnnotationError(f"unsafe COCO-Search18 task label '{value}'")
    return component
