
from __future__ import annotations

from collections.abc import Sequence

from activevision_workbench.datasets.contracts import DatasetItem
from activevision_workbench.errors import ContractValidationError


def individual_scanpath_run_items_from_dataset_item(
    item: DatasetItem,
    *,
    observer_ids: Sequence[str] | None = None,
) -> tuple[dict[str, object], ...]:

    if not isinstance(item, DatasetItem):
        raise ContractValidationError(
            "individual_scanpath_run_items_from_dataset_item.item",
            "must be a DatasetItem",
        )
    available = {scanpath.observer_id: scanpath for scanpath in item.scanpaths}
    if None in available:
        raise ContractValidationError(
            "DatasetItem.scanpaths.observer_id",
            "every IndividualScanpath source scanpath must identify an observer",
        )
    selected = tuple(available) if observer_ids is None else tuple(observer_ids)
    if not selected:
        raise ContractValidationError("observer_ids", "must not be empty")
    if len(selected) != len(set(selected)):
        raise ContractValidationError("observer_ids", "must not contain duplicates")
    records = []
    for observer_id in selected:
        if type(observer_id) is not str or observer_id not in available:
            raise ContractValidationError(
                "observer_ids", f"observer '{observer_id}' is unavailable for this item"
            )
        scanpath = available[observer_id]
        assert scanpath is not None
        records.append(
            {
                "item_id": f"{item.item_id}:{observer_id}",
                "image_ref": item.image_ref,
                "image_width": item.image_width,
                "image_height": item.image_height,
                "task_text": None,
                "target_description": None,
                "observer_id": observer_id,
                "observer_metadata": dict(scanpath.observer_metadata),
            }
        )
    return tuple(records)
