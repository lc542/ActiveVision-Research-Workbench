
from __future__ import annotations

from activevision_workbench.datasets.contracts import DatasetItem
from activevision_workbench.errors import ContractValidationError


def gazexplain_run_item_from_dataset_item(
    item: DatasetItem,
    *,
    task_text: str | None = None,
    target_description: str | None = None,
) -> dict[str, object]:

    if not isinstance(item, DatasetItem):
        raise ContractValidationError(
            "gazexplain_run_item_from_dataset_item.item", "must be a DatasetItem"
        )
    resolved_task = item.task_text if task_text is None else task_text
    if type(resolved_task) is not str or not resolved_task:
        raise ContractValidationError(
            "gazexplain_run_item_from_dataset_item.task_text",
            "must be a non-empty string or be present on the DatasetItem",
        )
    resolved_target = (
        item.target_label
        if target_description is None
        else target_description
    )
    if resolved_target is not None and type(resolved_target) is not str:
        raise ContractValidationError(
            "gazexplain_run_item_from_dataset_item.target_description",
            "must be a string or null",
        )
    return {
        "item_id": item.item_id,
        "image_ref": item.image_ref,
        "image_width": item.image_width,
        "image_height": item.image_height,
        "task_text": resolved_task,
        "target_description": resolved_target,
        "observer_id": None,
        "observer_metadata": {},
    }
