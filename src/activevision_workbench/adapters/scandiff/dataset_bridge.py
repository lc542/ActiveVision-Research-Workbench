
from __future__ import annotations

from activevision_workbench.adapters.scandiff.config import (
    SCANDIFF_FREE_VIEWING_VARIANT,
    SCANDIFF_VARIANTS,
    SCANDIFF_VISUAL_SEARCH_VARIANT,
)
from activevision_workbench.datasets.contracts import DatasetItem
from activevision_workbench.errors import ContractValidationError


def scandiff_run_item_from_dataset_item(
    item: DatasetItem,
    *,
    variant: str,
) -> dict[str, object]:

    if not isinstance(item, DatasetItem):
        raise ContractValidationError(
            "scandiff_run_item_from_dataset_item.item",
            "must be a DatasetItem",
        )
    if variant not in SCANDIFF_VARIANTS:
        choices = ", ".join(sorted(SCANDIFF_VARIANTS))
        raise ContractValidationError(
            "scandiff_run_item_from_dataset_item.variant",
            f"must be one of: {choices}",
        )

    task_text: str | None = None
    target_description: str | None = None
    if variant == SCANDIFF_VISUAL_SEARCH_VARIANT:
        labels = {
            value
            for value in (
                item.target_label,
                *(scanpath.target_label for scanpath in item.scanpaths),
            )
            if value is not None
        }
        if len(labels) > 1:
            raise ContractValidationError(
                "DatasetItem.target_label",
                "ScanDiff visual search requires one unambiguous target label",
            )
        if labels:
            target_description = next(iter(labels))
        else:
            tasks = {
                value
                for value in (
                    item.task_text,
                    *(scanpath.task_text for scanpath in item.scanpaths),
                )
                if value is not None
            }
            if len(tasks) != 1:
                raise ContractValidationError(
                    "DatasetItem.task_text",
                    "ScanDiff visual search requires one target label or one "
                    "unambiguous task string",
                )
            task_text = next(iter(tasks))
    else:
        assert variant == SCANDIFF_FREE_VIEWING_VARIANT

    return {
        "item_id": item.item_id,
        "image_ref": item.image_ref,
        "image_width": item.image_width,
        "image_height": item.image_height,
        "task_text": task_text,
        "target_description": target_description,
        "observer_id": None,
        "observer_metadata": {},
    }
