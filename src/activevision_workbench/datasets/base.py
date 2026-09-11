
from __future__ import annotations

import hashlib
import json
import struct
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from activevision_workbench.datasets.contracts import (
    DatasetItem,
    DatasetManifest,
    DatasetMetadata,
)
from activevision_workbench.errors import (
    DataRootError,
    DatasetAnnotationError,
    DatasetItemNotFoundError,
)


class DatasetAdapter(ABC):

    dataset_id: str
    dataset_version: str
    adapter_version: str

    def __init__(self, data_root: str | Path) -> None:
        raw = str(data_root)
        if "://" in raw:
            raise DataRootError(
                "dataset roots must be explicit local directories; network "
                "references and hidden downloads are not supported"
            )
        root = Path(data_root).expanduser().resolve(strict=False)
        if not root.exists():
            raise DataRootError(f"dataset root does not exist: {root}")
        if not root.is_dir():
            raise DataRootError(f"dataset root is not a directory: {root}")
        self.data_root = root

    @property
    @abstractmethod
    def metadata(self) -> DatasetMetadata:
        pass

    @property
    @abstractmethod
    def annotation_paths(self) -> tuple[Path, ...]:
        pass

    @abstractmethod
    def _item_ids_for_split(self, split: str) -> tuple[str, ...]:
        pass

    @abstractmethod
    def _load_item(self, item_id: str, split: str) -> DatasetItem:
        pass

    def item_ids(self, *, split: str) -> tuple[str, ...]:

        self._validate_split(split)
        ids = self._item_ids_for_split(split)
        if len(ids) != len(set(ids)):
            raise DatasetAnnotationError(
                f"adapter produced duplicate item IDs for split '{split}'"
            )
        return ids

    def iter_items(
        self,
        *,
        split: str,
        item_ids: Sequence[str] | None = None,
        observer_ids: Sequence[str] | None = None,
        tasks: Sequence[str] | None = None,
    ) -> Iterator[DatasetItem]:

        available_ids = self.item_ids(split=split)
        selected_ids = _selection(
            item_ids,
            available_ids,
            field="item_ids",
            missing_error=DatasetItemNotFoundError,
        )
        observers = _optional_filter(observer_ids, "observer_ids")
        task_filter = _optional_filter(tasks, "tasks")
        for item_id in selected_ids:
            item = self._load_item(item_id, split)
            if item.item_id != item_id or item.official_split != split:
                raise DatasetAnnotationError(
                    f"adapter item identity mismatch for '{item_id}' in '{split}'"
                )
            scanpaths = item.scanpaths
            if observers is not None:
                scanpaths = tuple(
                    scanpath
                    for scanpath in scanpaths
                    if scanpath.observer_id in observers
                )
            if task_filter is not None:
                scanpaths = tuple(
                    scanpath
                    for scanpath in scanpaths
                    if _scanpath_matches_task(scanpath, item, task_filter)
                )
            if not scanpaths:
                continue
            yield item if scanpaths == item.scanpaths else replace(
                item, scanpaths=scanpaths
            )

    def get_item(
        self,
        item_id: str,
        *,
        split: str | None = None,
        observer_ids: Sequence[str] | None = None,
        tasks: Sequence[str] | None = None,
    ) -> DatasetItem:

        if type(item_id) is not str or not item_id:
            raise DatasetItemNotFoundError("item_id must be a non-empty string")
        candidate_splits = (
            (split,) if split is not None else self.metadata.available_splits
        )
        containing = tuple(
            candidate
            for candidate in candidate_splits
            if item_id in self.item_ids(split=candidate)
        )
        if not containing:
            raise DatasetItemNotFoundError(
                f"dataset '{self.dataset_id}' has no item '{item_id}'"
            )
        if len(containing) != 1:
            raise DatasetItemNotFoundError(
                f"item '{item_id}' occurs in multiple splits; specify split"
            )
        items = tuple(
            self.iter_items(
                split=containing[0],
                item_ids=(item_id,),
                observer_ids=observer_ids,
                tasks=tasks,
            )
        )
        if not items:
            raise DatasetItemNotFoundError(
                f"item '{item_id}' has no scanpath matching the filters"
            )
        return items[0]

    def create_manifest(self) -> DatasetManifest:

        fingerprints: dict[str, object] = {}
        for path in self.annotation_paths:
            if not path.is_file():
                raise DatasetAnnotationError(
                    f"annotation file does not exist: {path}"
                )
            key = relative_reference(path, self.data_root)
            fingerprints[key] = file_fingerprint(path)
        split_payload = {
            split: list(self.item_ids(split=split))
            for split in self.metadata.available_splits
        }
        encoded = json.dumps(
            split_payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        split_fingerprint = f"sha256:{hashlib.sha256(encoded).hexdigest()}"
        return DatasetManifest(
            dataset_id=self.dataset_id,
            dataset_version=self.dataset_version,
            configured_root=str(self.data_root),
            adapter_version=self.adapter_version,
            annotation_fingerprints=fingerprints,
            split_definition_fingerprint=split_fingerprint,
            item_counts={
                split: len(ids) for split, ids in split_payload.items()
            },
            created_at=datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
        )

    def _validate_split(self, split: object) -> str:
        if type(split) is not str or not split:
            raise DatasetItemNotFoundError("split must be a non-empty string")
        if split not in self.metadata.available_splits:
            choices = ", ".join(self.metadata.available_splits)
            raise DatasetItemNotFoundError(
                f"unknown split '{split}' for '{self.dataset_id}'; "
                f"available: {choices}"
            )
        return split


def load_json(path: Path) -> object:

    try:
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream, parse_constant=_reject_json_constant)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise DatasetAnnotationError(
            f"cannot read annotation '{path}': {error}"
        ) from error


def file_fingerprint(path: Path) -> dict[str, object]:

    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        stat = path.stat()
    except OSError as error:
        raise DatasetAnnotationError(
            f"cannot fingerprint annotation '{path}': {error}"
        ) from error
    return {
        "sha256": digest.hexdigest(),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def relative_reference(path: Path, root: Path) -> str:

    try:
        return path.resolve(strict=False).relative_to(root).as_posix()
    except ValueError:
        return str(path.resolve(strict=False))


def read_image_dimensions(path: Path) -> tuple[int, int]:

    if not path.is_file():
        raise DatasetAnnotationError(f"image file does not exist: {path}")
    try:
        with path.open("rb") as stream:
            signature = stream.read(24)
            if signature.startswith(b"\x89PNG\r\n\x1a\n"):
                if len(signature) < 24:
                    raise DatasetAnnotationError(f"truncated PNG header: {path}")
                width, height = struct.unpack(">II", signature[16:24])
                return _positive_dimensions(width, height, path)
            if signature[:2] != b"\xff\xd8":
                raise DatasetAnnotationError(
                    f"unsupported image format for metadata-only read: {path}"
                )
            stream.seek(2)
            return _read_jpeg_dimensions(stream, path)
    except OSError as error:
        raise DatasetAnnotationError(
            f"cannot read image header '{path}': {error}"
        ) from error


def _read_jpeg_dimensions(stream: object, path: Path) -> tuple[int, int]:
    start_of_frame = {
        0xC0,
        0xC1,
        0xC2,
        0xC3,
        0xC5,
        0xC6,
        0xC7,
        0xC9,
        0xCA,
        0xCB,
        0xCD,
        0xCE,
        0xCF,
    }
    while True:
        prefix = stream.read(1)
        if not prefix:
            raise DatasetAnnotationError(f"JPEG dimensions not found: {path}")
        if prefix != b"\xff":
            continue
        marker_bytes = stream.read(1)
        while marker_bytes == b"\xff":
            marker_bytes = stream.read(1)
        if not marker_bytes:
            raise DatasetAnnotationError(f"truncated JPEG marker: {path}")
        marker = marker_bytes[0]
        if marker in {0x01, *range(0xD0, 0xDA)}:
            continue
        length_data = stream.read(2)
        if len(length_data) != 2:
            raise DatasetAnnotationError(f"truncated JPEG segment: {path}")
        segment_length = int.from_bytes(length_data, "big")
        if segment_length < 2:
            raise DatasetAnnotationError(f"invalid JPEG segment length: {path}")
        if marker in start_of_frame:
            frame = stream.read(5)
            if len(frame) != 5:
                raise DatasetAnnotationError(f"truncated JPEG frame: {path}")
            height = int.from_bytes(frame[1:3], "big")
            width = int.from_bytes(frame[3:5], "big")
            return _positive_dimensions(width, height, path)
        stream.seek(segment_length - 2, 1)


def _positive_dimensions(width: int, height: int, path: Path) -> tuple[int, int]:
    if width < 2 or height < 2:
        raise DatasetAnnotationError(f"invalid image dimensions in '{path}'")
    return width, height


def _selection(
    requested: Sequence[str] | None,
    available: tuple[str, ...],
    *,
    field: str,
    missing_error: type[Exception],
) -> tuple[str, ...]:
    if requested is None:
        return available
    selected = _optional_filter(requested, field)
    assert selected is not None
    unknown = sorted(selected - set(available))
    if unknown:
        raise missing_error(f"unknown {field} value '{unknown[0]}'")
    return tuple(item_id for item_id in available if item_id in selected)


def _optional_filter(
    values: Sequence[str] | None, field: str
) -> frozenset[str] | None:
    if values is None:
        return None
    if isinstance(values, (str, bytes, bytearray)):
        raise DatasetAnnotationError(f"{field} must be a sequence, not a string")
    result: set[str] = set()
    for index, value in enumerate(values):
        if type(value) is not str or not value:
            raise DatasetAnnotationError(
                f"{field}[{index}] must be a non-empty string"
            )
        result.add(value)
    return frozenset(result)


def _scanpath_matches_task(
    scanpath: object, item: DatasetItem, tasks: frozenset[str]
) -> bool:
    values = {
        item.task_text,
        item.target_label,
        getattr(scanpath, "task_text", None),
        getattr(scanpath, "target_label", None),
    }
    return bool(tasks.intersection(value for value in values if value is not None))


def _reject_json_constant(value: str) -> None:
    if value in {"NaN", "Infinity", "-Infinity"}:
        raise ValueError(f"non-finite JSON numeric constant '{value}'")
    raise ValueError(f"unsupported JSON constant '{value}'")
