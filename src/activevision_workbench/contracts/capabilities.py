
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar

from activevision_workbench.contracts._validation import (
    require_string,
    validate_dict_keys,
    validate_schema_version,
)
from activevision_workbench.errors import ContractValidationError


class Capability(str, Enum):

    PRODUCES_SCANPATHS = "produces_scanpaths"
    PRODUCES_SALIENCY_MAPS = "produces_saliency_maps"
    STOCHASTIC_MULTI_SAMPLE = "stochastic_multi_sample"
    INSTRUCTION_CONDITIONED = "instruction_conditioned"
    OBSERVER_CONDITIONED = "observer_conditioned"
    CONTINUOUS_TIME_OUTPUT = "continuous_time_output"
    FIXATION_DURATION_OUTPUT = "fixation_duration_output"
    TEXT_EXPLANATION_OUTPUT = "text_explanation_output"
    BATCH_INFERENCE = "batch_inference"


@dataclass(frozen=True, slots=True)
class CapabilitySet:

    CURRENT_SCHEMA_VERSION: ClassVar[int] = 1

    values: frozenset[Capability] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if not isinstance(self.values, frozenset):
            raise ContractValidationError(
                "CapabilitySet.values",
                "must be a frozenset of Capability values",
            )
        for value in self.values:
            if not isinstance(value, Capability):
                raise ContractValidationError(
                    "CapabilitySet.values",
                    "must contain only Capability values",
                )

    def supports(self, capability: Capability) -> bool:

        if not isinstance(capability, Capability):
            raise ContractValidationError(
                "capability",
                "must be a Capability value",
            )
        return capability in self.values

    def to_dict(self) -> dict[str, object]:

        return {
            "schema_version": self.CURRENT_SCHEMA_VERSION,
            "values": sorted(capability.value for capability in self.values),
        }

    @classmethod
    def from_dict(cls, data: object) -> "CapabilitySet":

        fields = validate_dict_keys(
            data,
            "CapabilitySet",
            required=frozenset({"schema_version", "values"}),
        )
        validate_schema_version(
            fields["schema_version"],
            "CapabilitySet.schema_version",
        )
        raw_values = fields["values"]
        if not isinstance(raw_values, list):
            raise ContractValidationError(
                "CapabilitySet.values",
                "must be a JSON array",
            )
        capabilities: set[Capability] = set()
        for index, raw_value in enumerate(raw_values):
            value = require_string(
                raw_value,
                f"CapabilitySet.values[{index}]",
            )
            try:
                capabilities.add(Capability(value))
            except ValueError as error:
                raise ContractValidationError(
                    f"CapabilitySet.values[{index}]",
                    f"unknown capability '{value}'",
                ) from error
        return cls(frozenset(capabilities))


@dataclass(frozen=True, slots=True)
class RequestRequirements:

    instruction: bool = False
    observer: bool = False

    def __post_init__(self) -> None:
        if type(self.instruction) is not bool:
            raise ContractValidationError(
                "RequestRequirements.instruction",
                "must be a boolean",
            )
        if type(self.observer) is not bool:
            raise ContractValidationError(
                "RequestRequirements.observer",
                "must be a boolean",
            )

    def validate_against(self, capabilities: CapabilitySet) -> None:

        if self.instruction and not capabilities.supports(
            Capability.INSTRUCTION_CONDITIONED
        ):
            raise ContractValidationError(
                "RequestRequirements.instruction",
                "requires the instruction-conditioned capability",
            )
        if self.observer and not capabilities.supports(
            Capability.OBSERVER_CONDITIONED
        ):
            raise ContractValidationError(
                "RequestRequirements.observer",
                "requires the observer-conditioned capability",
            )
