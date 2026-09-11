
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

from activevision_workbench.contracts import ArtifactReference
from activevision_workbench.contracts._validation import validate_dict_keys
from activevision_workbench.datasets.base import DatasetAdapter, load_json
from activevision_workbench.datasets.contracts import DatasetItem, DatasetMetadata
from activevision_workbench.errors import (
    ContractValidationError,
    DatasetAnnotationError,
)


class JsonDatasetAdapter(DatasetAdapter):

    adapter_version = "0.1.0"

    def __init__(
        self,
        data_root: str | Path,
        *,
        annotation_name: str = "annotations.json",
    ) -> None:
        super().__init__(data_root)
        if type(annotation_name) is not str or not annotation_name:
            raise DatasetAnnotationError("annotation_name must be a non-empty string")
        self.annotation_path = self.data_root / annotation_name
        if not self.annotation_path.is_file():
            raise DatasetAnnotationError(
                f"normalized annotation does not exist: {self.annotation_path}"
            )
        self._items: dict[str, DatasetItem]
        self._splits: dict[str, tuple[str, ...]]
        self._metadata: DatasetMetadata
        self._load_annotation()

    @property
    def metadata(self) -> DatasetMetadata:
        return self._metadata

    @property
    def annotation_paths(self) -> tuple[Path, ...]:
        return (self.annotation_path,)

    def _item_ids_for_split(self, split: str) -> tuple[str, ...]:
        return self._splits[split]

    def _load_item(self, item_id: str, split: str) -> DatasetItem:
        del split
        return self._items[item_id]

    def _load_annotation(self) -> None:
        raw = load_json(self.annotation_path)
        try:
            top = validate_dict_keys(
                raw,
                "JsonDataset",
                required=frozenset(
                    {
                        "schema_version",
                        "dataset_id",
                        "dataset_version",
                        "splits",
                        "items",
                    }
                ),
            )
            if top["schema_version"] != 1:
                raise DatasetAnnotationError(
                    "JsonDataset.schema_version must be integer 1"
                )
            dataset_id = _string(top["dataset_id"], "JsonDataset.dataset_id")
            dataset_version = _string(
                top["dataset_version"], "JsonDataset.dataset_version"
            )
            raw_items = top["items"]
            if not isinstance(raw_items, list) or not raw_items:
                raise DatasetAnnotationError(
                    "JsonDataset.items must be a non-empty JSON array"
                )
            items: dict[str, DatasetItem] = {}
            for index, value in enumerate(raw_items):
                try:
                    item = DatasetItem.from_dict(value)
                except ContractValidationError as error:
                    raise DatasetAnnotationError(
                        f"JsonDataset.items[{index}]: {error}"
                    ) from error
                if (
                    item.dataset_id != dataset_id
                    or item.dataset_version != dataset_version
                ):
                    raise DatasetAnnotationError(
                        f"JsonDataset.items[{index}] dataset identity mismatch"
                    )
                if item.item_id in items:
                    raise DatasetAnnotationError(
                        f"duplicate normalized item ID '{item.item_id}'"
                    )
                image_ref = _local_image_reference(item.image_ref, self.data_root)
                items[item.item_id] = replace(item, image_ref=image_ref)

            raw_splits = top["splits"]
            if not isinstance(raw_splits, Mapping) or not raw_splits:
                raise DatasetAnnotationError(
                    "JsonDataset.splits must be a non-empty JSON object"
                )
            splits: dict[str, tuple[str, ...]] = {}
            assigned: dict[str, str] = {}
            for raw_split, raw_ids in raw_splits.items():
                split = _string(raw_split, "JsonDataset.splits.<name>")
                if not isinstance(raw_ids, list) or not raw_ids:
                    raise DatasetAnnotationError(
                        f"JsonDataset.splits.{split} must be a non-empty array"
                    )
                ids = tuple(
                    _string(item_id, f"JsonDataset.splits.{split}[{index}]")
                    for index, item_id in enumerate(raw_ids)
                )
                if len(ids) != len(set(ids)):
                    raise DatasetAnnotationError(
                        f"JsonDataset.splits.{split} contains duplicate IDs"
                    )
                for item_id in ids:
                    if item_id not in items:
                        raise DatasetAnnotationError(
                            f"split '{split}' references unknown item '{item_id}'"
                        )
                    if items[item_id].official_split != split:
                        raise DatasetAnnotationError(
                            f"item '{item_id}' official_split does not match '{split}'"
                        )
                    if item_id in assigned:
                        raise DatasetAnnotationError(
                            f"item '{item_id}' occurs in multiple splits"
                        )
                    assigned[item_id] = split
                splits[split] = tuple(sorted(ids))
            if set(assigned) != set(items):
                missing = sorted(set(items) - set(assigned))[0]
                raise DatasetAnnotationError(
                    f"item '{missing}' is absent from split definitions"
                )
        except ContractValidationError as error:
            raise DatasetAnnotationError(str(error)) from error

        self.dataset_id = dataset_id
        self.dataset_version = dataset_version
        self._items = items
        self._splits = dict(sorted(splits.items()))
        observers = sorted(
            {
                observer
                for item in items.values()
                for observer in item.observer_ids
            }
        )
        targets = sorted(
            {
                label
                for item in items.values()
                for label in (item.target_label,)
                if label is not None
            }
        )
        reference = ArtifactReference(
            uri=self.annotation_path.relative_to(self.data_root).as_posix(),
            kind="dataset_annotation",
            media_type="application/json",
        )
        self._metadata = DatasetMetadata(
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            adapter_version=self.adapter_version,
            data_root=str(self.data_root),
            available_splits=tuple(self._splits),
            item_counts={split: len(ids) for split, ids in self._splits.items()},
            observer_ids=tuple(observers),
            target_labels=tuple(targets),
            annotation_references=(reference,),
        )


def _local_image_reference(value: str, root: Path) -> str:
    if "://" in value:
        raise DatasetAnnotationError(
            "normalized image_ref must be a local path; network access is disabled"
        )
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    path = path.resolve(strict=False)
    if not path.is_file():
        raise DatasetAnnotationError(f"normalized image_ref does not exist: {path}")
    return str(path)


def _string(value: object, field: str) -> str:
    if type(value) is not str or not value:
        raise DatasetAnnotationError(f"{field} must be a non-empty string")
    return value
