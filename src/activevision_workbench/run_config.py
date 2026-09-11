
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import ClassVar, TypeVar

from activevision_workbench.contracts import InferenceRequest
from activevision_workbench.contracts._validation import (
    JsonMapping,
    freeze_json_mapping,
    freeze_namespaced_options,
    optional_finite_number,
    optional_int,
    optional_string,
    require_int,
    require_string,
    thaw_json_value,
    validate_dict_keys,
    validate_schema_version,
)
from activevision_workbench.errors import (
    ContractValidationError,
    RunConfigurationError,
)

_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
EnumT = TypeVar("EnumT", bound=Enum)


class RunPolicy(str, Enum):

    CREATE = "create"
    OVERWRITE = "overwrite"
    RESUME = "resume"


class SeedStrategy(str, Enum):

    INCREMENT_PER_ITEM = "increment_per_item"
    FIXED_PER_ITEM = "fixed_per_item"


@dataclass(frozen=True, slots=True)
class RunItem:

    item_id: str
    image_ref: str
    image_width: int | None
    image_height: int | None
    task_text: str | None
    target_description: str | None
    observer_id: str | None
    observer_metadata: JsonMapping

    def to_dict(self) -> dict[str, object]:

        return {
            "item_id": self.item_id,
            "image_ref": self.image_ref,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "task_text": self.task_text,
            "target_description": self.target_description,
            "observer_id": self.observer_id,
            "observer_metadata": thaw_json_value(self.observer_metadata),
        }


@dataclass(frozen=True, slots=True)
class RunConfig:

    CURRENT_SCHEMA_VERSION: ClassVar[int] = 1

    schema_version: int
    run_id: str | None
    model_id: str
    model_variant: str
    model_options: JsonMapping
    checkpoint_path: str | None
    checkpoint_fingerprint: str | None
    upstream_code_ref: str | None
    upstream_commit: str | None
    dataset_id: str
    dataset_version: str | None
    dataset_split: str
    dataset_root: str | None
    dataset_root_fingerprint: str | None
    items: tuple[RunItem, ...]
    item_subset: tuple[str, ...] | None
    task_text: str | None
    target_description: str | None
    observer_id: str | None
    observer_metadata: JsonMapping
    num_samples: int
    base_seed: int
    seed_strategy: SeedStrategy
    max_fixations: int | None
    time_horizon_s: float | None
    device: str
    output_root: str
    run_policy: RunPolicy
    tags: tuple[str, ...]
    notes: str | None
    environment_name: str | None
    environment_image_digest: str | None

    @classmethod
    def from_file(cls, path: str | Path) -> "RunConfig":

        source = Path(path)
        try:
            text = source.read_text(encoding="utf-8")
        except OSError as error:
            raise RunConfigurationError(
                f"cannot read run configuration '{source}': {error}"
            ) from error
        data: object
        try:
            data = json.loads(text, parse_constant=_reject_json_constant)
        except json.JSONDecodeError as json_error:
            try:
                import yaml  # type: ignore[import-not-found]
            except ImportError as error:
                raise RunConfigurationError(
                    f"{source}: configuration is not strict JSON. JSON is a valid "
                    "YAML 1.2 subset; install PyYAML only if block-style YAML is "
                    f"required ({json_error.msg} at line {json_error.lineno})."
                ) from error
            try:
                data = yaml.safe_load(text)
            except Exception as error:
                raise RunConfigurationError(
                    f"{source}: invalid YAML configuration: {error}"
                ) from error
        return cls.from_dict(data, base_dir=source.resolve().parent)

    @classmethod
    def from_dict(
        cls,
        data: object,
        *,
        base_dir: str | Path | None = None,
    ) -> "RunConfig":

        root = Path.cwd() if base_dir is None else Path(base_dir)
        try:
            top = validate_dict_keys(
                data,
                "RunConfig",
                required=frozenset(
                    {
                        "schema_version",
                        "run_id",
                        "model",
                        "dataset",
                        "request",
                        "device",
                        "output_root",
                        "run_policy",
                        "tags",
                        "notes",
                        "environment",
                    }
                ),
            )
            schema_version = validate_schema_version(
                top["schema_version"], "RunConfig.schema_version"
            )
            run_id = optional_string(
                top["run_id"], "RunConfig.run_id", allow_empty=False
            )
            if run_id is not None:
                _validate_run_id(run_id)

            model = validate_dict_keys(
                top["model"],
                "RunConfig.model",
                required=frozenset(
                    {
                        "id",
                        "variant",
                        "options",
                        "checkpoint_path",
                        "checkpoint_fingerprint",
                        "upstream_code_ref",
                        "upstream_commit",
                    }
                ),
            )
            model_id = require_string(model["id"], "RunConfig.model.id")
            model_variant = require_string(
                model["variant"], "RunConfig.model.variant"
            )
            model_options = freeze_namespaced_options(
                model["options"],
                "RunConfig.model.options",
                model_id=model_id,
            )
            checkpoint_path = _optional_reference(
                model["checkpoint_path"],
                "RunConfig.model.checkpoint_path",
                root,
                resolve_path=True,
            )
            checkpoint_fingerprint = optional_string(
                model["checkpoint_fingerprint"],
                "RunConfig.model.checkpoint_fingerprint",
                allow_empty=False,
            )
            upstream_code_ref = _optional_reference(
                model["upstream_code_ref"],
                "RunConfig.model.upstream_code_ref",
                root,
                resolve_path=False,
            )
            upstream_commit = optional_string(
                model["upstream_commit"],
                "RunConfig.model.upstream_commit",
                allow_empty=False,
            )

            dataset = validate_dict_keys(
                top["dataset"],
                "RunConfig.dataset",
                required=frozenset(
                    {
                        "id",
                        "version",
                        "split",
                        "root",
                        "root_fingerprint",
                        "items",
                        "item_subset",
                    }
                ),
            )
            dataset_id = require_string(dataset["id"], "RunConfig.dataset.id")
            dataset_version = optional_string(
                dataset["version"],
                "RunConfig.dataset.version",
                allow_empty=False,
            )
            dataset_split = require_string(
                dataset["split"], "RunConfig.dataset.split"
            )
            dataset_root = _optional_reference(
                dataset["root"],
                "RunConfig.dataset.root",
                root,
                resolve_path=True,
            )
            dataset_root_fingerprint = optional_string(
                dataset["root_fingerprint"],
                "RunConfig.dataset.root_fingerprint",
                allow_empty=False,
            )

            request = validate_dict_keys(
                top["request"],
                "RunConfig.request",
                required=frozenset(
                    {
                        "task_text",
                        "target_description",
                        "observer_id",
                        "observer_metadata",
                        "num_samples",
                        "base_seed",
                        "seed_strategy",
                        "max_fixations",
                        "time_horizon_s",
                    }
                ),
            )
            task_text = optional_string(
                request["task_text"], "RunConfig.request.task_text"
            )
            target_description = optional_string(
                request["target_description"],
                "RunConfig.request.target_description",
            )
            observer_id = optional_string(
                request["observer_id"],
                "RunConfig.request.observer_id",
                allow_empty=False,
            )
            observer_metadata = freeze_json_mapping(
                request["observer_metadata"],
                "RunConfig.request.observer_metadata",
            )
            num_samples = require_int(
                request["num_samples"],
                "RunConfig.request.num_samples",
                minimum=1,
            )
            base_seed = require_int(
                request["base_seed"],
                "RunConfig.request.base_seed",
                minimum=0,
            )
            seed_strategy = _enum_value(
                SeedStrategy,
                request["seed_strategy"],
                "RunConfig.request.seed_strategy",
            )
            max_fixations = optional_int(
                request["max_fixations"],
                "RunConfig.request.max_fixations",
                minimum=1,
            )
            time_horizon_s = optional_finite_number(
                request["time_horizon_s"],
                "RunConfig.request.time_horizon_s",
                minimum=0.0,
            )
            if time_horizon_s == 0.0:
                raise ContractValidationError(
                    "RunConfig.request.time_horizon_s", "must be > 0"
                )

            items = _parse_items(
                dataset["items"],
                task_text=task_text,
                target_description=target_description,
                observer_id=observer_id,
                observer_metadata=observer_metadata,
            )
            item_subset = _parse_subset(dataset["item_subset"], items)

            device = require_string(top["device"], "RunConfig.device")
            output_root_raw = require_string(
                top["output_root"], "RunConfig.output_root"
            )
            output_root = str(_resolve_path(output_root_raw, root))
            run_policy = _enum_value(
                RunPolicy, top["run_policy"], "RunConfig.run_policy"
            )
            tags = _parse_string_sequence(top["tags"], "RunConfig.tags")
            if len(tags) != len(set(tags)):
                raise ContractValidationError(
                    "RunConfig.tags", "must not contain duplicates"
                )
            notes = optional_string(top["notes"], "RunConfig.notes")
            environment = validate_dict_keys(
                top["environment"],
                "RunConfig.environment",
                required=frozenset({"name", "image_digest"}),
            )
            environment_name = optional_string(
                environment["name"],
                "RunConfig.environment.name",
                allow_empty=False,
            )
            environment_image_digest = optional_string(
                environment["image_digest"],
                "RunConfig.environment.image_digest",
                allow_empty=False,
            )
        except ContractValidationError as error:
            raise RunConfigurationError(str(error)) from error

        return cls(
            schema_version=schema_version,
            run_id=run_id,
            model_id=model_id,
            model_variant=model_variant,
            model_options=model_options,
            checkpoint_path=checkpoint_path,
            checkpoint_fingerprint=checkpoint_fingerprint,
            upstream_code_ref=upstream_code_ref,
            upstream_commit=upstream_commit,
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            dataset_split=dataset_split,
            dataset_root=dataset_root,
            dataset_root_fingerprint=dataset_root_fingerprint,
            items=items,
            item_subset=item_subset,
            task_text=task_text,
            target_description=target_description,
            observer_id=observer_id,
            observer_metadata=observer_metadata,
            num_samples=num_samples,
            base_seed=base_seed,
            seed_strategy=seed_strategy,
            max_fixations=max_fixations,
            time_horizon_s=time_horizon_s,
            device=device,
            output_root=output_root,
            run_policy=run_policy,
            tags=tags,
            notes=notes,
            environment_name=environment_name,
            environment_image_digest=environment_image_digest,
        )

    @property
    def selected_items(self) -> tuple[RunItem, ...]:

        if self.item_subset is None:
            return self.items
        by_id = {item.item_id: item for item in self.items}
        return tuple(by_id[item_id] for item_id in self.item_subset)

    @property
    def configuration_hash(self) -> str:

        payload = json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return f"sha256:{hashlib.sha256(payload).hexdigest()}"

    def with_run_id(self, run_id: str) -> "RunConfig":

        try:
            value = require_string(run_id, "RunConfig.run_id")
            _validate_run_id(value)
        except ContractValidationError as error:
            raise RunConfigurationError(str(error)) from error
        return replace(self, run_id=value)

    def seed_for_item(self, index: int) -> int:

        if self.seed_strategy is SeedStrategy.FIXED_PER_ITEM:
            return self.base_seed
        return self.base_seed + index * self.num_samples

    def requests(self) -> tuple[InferenceRequest, ...]:

        requests: list[InferenceRequest] = []
        for index, item in enumerate(self.selected_items):
            requests.append(
                InferenceRequest.create(
                    model_id=self.model_id,
                    model_variant=self.model_variant,
                    dataset_id=self.dataset_id,
                    dataset_split=self.dataset_split,
                    item_id=item.item_id,
                    image_ref=item.image_ref,
                    image_width=item.image_width,
                    image_height=item.image_height,
                    task_text=item.task_text,
                    target_description=item.target_description,
                    observer_id=item.observer_id,
                    observer_metadata=item.observer_metadata,
                    num_samples=self.num_samples,
                    base_seed=self.seed_for_item(index),
                    max_fixations=self.max_fixations,
                    time_horizon_s=self.time_horizon_s,
                    model_options=self.model_options,
                )
            )
        return tuple(requests)

    def to_dict(self) -> dict[str, object]:

        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "model": {
                "id": self.model_id,
                "variant": self.model_variant,
                "options": thaw_json_value(self.model_options),
                "checkpoint_path": self.checkpoint_path,
                "checkpoint_fingerprint": self.checkpoint_fingerprint,
                "upstream_code_ref": self.upstream_code_ref,
                "upstream_commit": self.upstream_commit,
            },
            "dataset": {
                "id": self.dataset_id,
                "version": self.dataset_version,
                "split": self.dataset_split,
                "root": self.dataset_root,
                "root_fingerprint": self.dataset_root_fingerprint,
                "items": [item.to_dict() for item in self.items],
                "item_subset": (
                    None if self.item_subset is None else list(self.item_subset)
                ),
            },
            "request": {
                "task_text": self.task_text,
                "target_description": self.target_description,
                "observer_id": self.observer_id,
                "observer_metadata": thaw_json_value(self.observer_metadata),
                "num_samples": self.num_samples,
                "base_seed": self.base_seed,
                "seed_strategy": self.seed_strategy.value,
                "max_fixations": self.max_fixations,
                "time_horizon_s": self.time_horizon_s,
            },
            "device": self.device,
            "output_root": self.output_root,
            "run_policy": self.run_policy.value,
            "tags": list(self.tags),
            "notes": self.notes,
            "environment": {
                "name": self.environment_name,
                "image_digest": self.environment_image_digest,
            },
        }


def _parse_items(
    value: object,
    *,
    task_text: str | None,
    target_description: str | None,
    observer_id: str | None,
    observer_metadata: JsonMapping,
) -> tuple[RunItem, ...]:
    if not isinstance(value, Sequence) or isinstance(
        value, (str, bytes, bytearray)
    ):
        raise ContractValidationError("RunConfig.dataset.items", "must be a list")
    if len(value) == 0:
        raise ContractValidationError(
            "RunConfig.dataset.items", "must contain at least one item"
        )
    items: list[RunItem] = []
    ids: set[str] = set()
    required = frozenset({"item_id", "image_ref", "image_width", "image_height"})
    optional = frozenset(
        {"task_text", "target_description", "observer_id", "observer_metadata"}
    )
    for index, raw in enumerate(value):
        path = f"RunConfig.dataset.items[{index}]"
        fields = validate_dict_keys(raw, path, required=required, optional=optional)
        item_id = require_string(fields["item_id"], f"{path}.item_id")
        if item_id in ids:
            raise ContractValidationError(
                f"{path}.item_id", f"duplicate item ID '{item_id}'"
            )
        ids.add(item_id)
        width = optional_int(fields["image_width"], f"{path}.image_width", minimum=1)
        height = optional_int(
            fields["image_height"], f"{path}.image_height", minimum=1
        )
        if (width is None) != (height is None):
            missing = "image_height" if height is None else "image_width"
            raise ContractValidationError(
                f"{path}.{missing}", "image width and height must be provided together"
            )
        resolved_task = optional_string(
            fields.get("task_text", task_text), f"{path}.task_text"
        )
        resolved_target = optional_string(
            fields.get("target_description", target_description),
            f"{path}.target_description",
        )
        resolved_observer = optional_string(
            fields.get("observer_id", observer_id),
            f"{path}.observer_id",
            allow_empty=False,
        )
        resolved_metadata = freeze_json_mapping(
            fields.get("observer_metadata", observer_metadata),
            f"{path}.observer_metadata",
        )
        items.append(
            RunItem(
                item_id=item_id,
                image_ref=require_string(fields["image_ref"], f"{path}.image_ref"),
                image_width=width,
                image_height=height,
                task_text=resolved_task,
                target_description=resolved_target,
                observer_id=resolved_observer,
                observer_metadata=resolved_metadata,
            )
        )
    return tuple(items)


def _parse_subset(
    value: object, items: tuple[RunItem, ...]
) -> tuple[str, ...] | None:
    if value is None:
        return None
    subset = _parse_string_sequence(value, "RunConfig.dataset.item_subset")
    if not subset:
        raise ContractValidationError(
            "RunConfig.dataset.item_subset", "must not be empty when supplied"
        )
    if len(subset) != len(set(subset)):
        raise ContractValidationError(
            "RunConfig.dataset.item_subset", "must not contain duplicates"
        )
    available = {item.item_id for item in items}
    unknown = next((item_id for item_id in subset if item_id not in available), None)
    if unknown is not None:
        raise ContractValidationError(
            "RunConfig.dataset.item_subset", f"unknown item ID '{unknown}'"
        )
    return subset


def _parse_string_sequence(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(
        value, (str, bytes, bytearray)
    ):
        raise ContractValidationError(field, "must be a list of strings")
    return tuple(
        require_string(item, f"{field}[{index}]")
        for index, item in enumerate(value)
    )


def _enum_value(enum_type: type[EnumT], value: object, field: str) -> EnumT:
    raw = require_string(value, field)
    try:
        return enum_type(raw)
    except ValueError as error:
        choices = ", ".join(member.value for member in enum_type)
        raise ContractValidationError(
            field, f"must be one of: {choices}"
        ) from error


def _validate_run_id(run_id: str) -> None:
    if run_id in {".", ".."} or _RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise ContractValidationError(
            "RunConfig.run_id",
            "must be a safe 1-128 character filename using letters, digits, '.', "
            "'_', or '-'",
        )


def _optional_reference(
    value: object,
    field: str,
    base_dir: Path,
    *,
    resolve_path: bool,
) -> str | None:
    reference = optional_string(value, field, allow_empty=False)
    if reference is None or not resolve_path or "://" in reference:
        return reference
    return str(_resolve_path(reference, base_dir))


def _resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve(strict=False)


def _reject_json_constant(value: str) -> None:
    raise RunConfigurationError(
        f"configuration contains non-standard JSON numeric constant '{value}'"
    )
