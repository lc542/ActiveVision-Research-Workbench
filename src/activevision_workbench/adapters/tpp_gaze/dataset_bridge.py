
from __future__ import annotations

from activevision_workbench.datasets.contracts import DatasetItem
from activevision_workbench.errors import ContractValidationError


def tpp_gaze_run_item_from_dataset_item(item: DatasetItem) -> dict[str, object]:

    if not isinstance(item, DatasetItem):
        raise ContractValidationError(
            "tpp_gaze_run_item_from_dataset_item.item", "must be a DatasetItem"
        )
    return {
        "item_id": item.item_id,
        "image_ref": item.image_ref,
        "image_width": item.image_width,
        "image_height": item.image_height,
        "task_text": None,
        "target_description": None,
        "observer_id": None,
        "observer_metadata": {},
    }
