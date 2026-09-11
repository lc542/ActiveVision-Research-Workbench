
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from typing import ClassVar

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
from activevision_workbench.contracts.capabilities import (
    Capability,
    CapabilitySet,
    RequestRequirements,
)
from activevision_workbench.errors import ContractValidationError


@dataclass(frozen=True, slots=True)
class InferenceRequest:

    CURRENT_SCHEMA_VERSION: ClassVar[int] = 1

    request_id: str
    model_id: str
    model_variant: str
    dataset_id: str
    dataset_split: str
    item_id: str
    image_ref: str
    schema_version: int = CURRENT_SCHEMA_VERSION
    image_width: int | None = None
    image_height: int | None = None
    task_text: str | None = None
    target_description: str | None = None
    observer_id: str | None = None
    observer_metadata: JsonMapping = field(default_factory=dict)
    num_samples: int = 1
    base_seed: int = 0
    max_fixations: int | None = None
    time_horizon_s: float | None = None
    model_options: JsonMapping = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "schema_version",
            validate_schema_version(
                self.schema_version,
                "InferenceRequest.schema_version",
            ),
        )
        for name in (
            "request_id",
            "model_id",
            "model_variant",
            "dataset_id",
            "dataset_split",
            "item_id",
            "image_ref",
        ):
            object.__setattr__(
                self,
                name,
                require_string(
                    getattr(self, name),
                    f"InferenceRequest.{name}",
                ),
            )

        width = optional_int(
            self.image_width,
            "InferenceRequest.image_width",
            minimum=1,
        )
        height = optional_int(
            self.image_height,
            "InferenceRequest.image_height",
            minimum=1,
        )
        if (width is None) != (height is None):
            missing = "image_height" if height is None else "image_width"
            raise ContractValidationError(
                f"InferenceRequest.{missing}",
                "image width and height must be provided together",
            )
        object.__setattr__(self, "image_width", width)
        object.__setattr__(self, "image_height", height)

        object.__setattr__(
            self,
            "task_text",
            optional_string(
                self.task_text,
                "InferenceRequest.task_text",
                allow_empty=True,
            ),
        )
        object.__setattr__(
            self,
            "target_description",
            optional_string(
                self.target_description,
                "InferenceRequest.target_description",
                allow_empty=True,
            ),
        )
        object.__setattr__(
            self,
            "observer_id",
            optional_string(
                self.observer_id,
                "InferenceRequest.observer_id",
                allow_empty=False,
            ),
        )
        observer_metadata = freeze_json_mapping(
            self.observer_metadata,
            "InferenceRequest.observer_metadata",
        )
        object.__setattr__(self, "observer_metadata", observer_metadata)

        object.__setattr__(
            self,
            "num_samples",
            require_int(
                self.num_samples,
                "InferenceRequest.num_samples",
                minimum=1,
            ),
        )
        object.__setattr__(
            self,
            "base_seed",
            require_int(
                self.base_seed,
                "InferenceRequest.base_seed",
                minimum=0,
            ),
        )
        object.__setattr__(
            self,
            "max_fixations",
            optional_int(
                self.max_fixations,
                "InferenceRequest.max_fixations",
                minimum=1,
            ),
        )
        horizon = optional_finite_number(
            self.time_horizon_s,
            "InferenceRequest.time_horizon_s",
            minimum=0.0,
        )
        if horizon is not None and horizon == 0.0:
            raise ContractValidationError(
                "InferenceRequest.time_horizon_s",
                "must be > 0",
            )
        object.__setattr__(self, "time_horizon_s", horizon)
        object.__setattr__(
            self,
            "model_options",
            freeze_namespaced_options(
                self.model_options,
                "InferenceRequest.model_options",
                model_id=self.model_id,
            ),
        )

    @classmethod
    def create(
        cls,
        *,
        model_id: str,
        model_variant: str,
        dataset_id: str,
        dataset_split: str,
        item_id: str,
        image_ref: str,
        request_id: str | None = None,
        schema_version: int = CURRENT_SCHEMA_VERSION,
        image_width: int | None = None,
        image_height: int | None = None,
        task_text: str | None = None,
        target_description: str | None = None,
        observer_id: str | None = None,
        observer_metadata: object | None = None,
        num_samples: int = 1,
        base_seed: int = 0,
        max_fixations: int | None = None,
        time_horizon_s: float | None = None,
        model_options: object | None = None,
    ) -> "InferenceRequest":

        provisional = cls(
            schema_version=schema_version,
            request_id=request_id if request_id is not None else "pending",
            model_id=model_id,
            model_variant=model_variant,
            dataset_id=dataset_id,
            dataset_split=dataset_split,
            item_id=item_id,
            image_ref=image_ref,
            image_width=image_width,
            image_height=image_height,
            task_text=task_text,
            target_description=target_description,
            observer_id=observer_id,
            observer_metadata=(
                {} if observer_metadata is None else observer_metadata
            ),  # type: ignore[arg-type]
            num_samples=num_samples,
            base_seed=base_seed,
            max_fixations=max_fixations,
            time_horizon_s=time_horizon_s,
            model_options=(
                {} if model_options is None else model_options
            ),  # type: ignore[arg-type]
        )
        if request_id is not None:
            return provisional
        return replace(provisional, request_id=provisional.deterministic_id())

    def deterministic_id(self) -> str:

        payload = self.to_dict()
        del payload["request_id"]
        canonical = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return f"req_{hashlib.sha256(canonical).hexdigest()}"

    def validate_for(
        self,
        capabilities: CapabilitySet,
        requirements: RequestRequirements = RequestRequirements(),
    ) -> None:

        requirements.validate_against(capabilities)
        has_instruction = (
            self.task_text is not None or self.target_description is not None
        )
        if has_instruction and not capabilities.supports(
            Capability.INSTRUCTION_CONDITIONED
        ):
            field = (
                "InferenceRequest.task_text"
                if self.task_text is not None
                else "InferenceRequest.target_description"
            )
            raise ContractValidationError(
                field,
                "adapter does not support instruction-conditioned input",
            )
        if requirements.instruction and not has_instruction:
            raise ContractValidationError(
                "InferenceRequest.task_text",
                "task_text or target_description is required by this adapter variant",
            )
        has_observer = self.observer_id is not None or bool(self.observer_metadata)
        if has_observer and not capabilities.supports(
            Capability.OBSERVER_CONDITIONED
        ):
            field = (
                "InferenceRequest.observer_id"
                if self.observer_id is not None
                else "InferenceRequest.observer_metadata"
            )
            raise ContractValidationError(
                field,
                "adapter does not support observer-conditioned input",
            )
        if requirements.observer and not has_observer:
            raise ContractValidationError(
                "InferenceRequest.observer_id",
                "observer_id or observer_metadata is required by this adapter variant",
            )
        if self.num_samples > 1 and not capabilities.supports(
            Capability.STOCHASTIC_MULTI_SAMPLE
        ):
            raise ContractValidationError(
                "InferenceRequest.num_samples",
                "adapter does not support multi-sample inference",
            )
        if self.time_horizon_s is not None and not capabilities.supports(
            Capability.CONTINUOUS_TIME_OUTPUT
        ):
            raise ContractValidationError(
                "InferenceRequest.time_horizon_s",
                "adapter does not support continuous-time output",
            )

    def to_dict(self) -> dict[str, object]:

        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "model_id": self.model_id,
            "model_variant": self.model_variant,
            "dataset_id": self.dataset_id,
            "dataset_split": self.dataset_split,
            "item_id": self.item_id,
            "image_ref": self.image_ref,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "task_text": self.task_text,
            "target_description": self.target_description,
            "observer_id": self.observer_id,
            "observer_metadata": thaw_json_value(self.observer_metadata),
            "num_samples": self.num_samples,
            "base_seed": self.base_seed,
            "max_fixations": self.max_fixations,
            "time_horizon_s": self.time_horizon_s,
            "model_options": thaw_json_value(self.model_options),
        }

    @classmethod
    def from_dict(cls, data: object) -> "InferenceRequest":

        names = frozenset(
            {
                "schema_version",
                "request_id",
                "model_id",
                "model_variant",
                "dataset_id",
                "dataset_split",
                "item_id",
                "image_ref",
                "image_width",
                "image_height",
                "task_text",
                "target_description",
                "observer_id",
                "observer_metadata",
                "num_samples",
                "base_seed",
                "max_fixations",
                "time_horizon_s",
                "model_options",
            }
        )
        fields = validate_dict_keys(
            data,
            "InferenceRequest",
            required=names,
        )
        return cls(**{name: fields[name] for name in names})  # type: ignore[arg-type]
