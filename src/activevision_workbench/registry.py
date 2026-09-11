
from __future__ import annotations

import importlib.util
from collections.abc import Callable, Collection
from dataclasses import dataclass

from activevision_workbench.adapters.base import ModelAdapter
from activevision_workbench.contracts import (
    CapabilitySet,
    ModelIdentity,
    RequestRequirements,
)
from activevision_workbench.contracts._validation import require_string
from activevision_workbench.errors import (
    AdapterUnavailableError,
    DuplicateAdapterError,
    RegistryError,
    UnknownModelError,
    UnsupportedModelError,
)

CHARTERED_MODEL_IDS = frozenset(
    {"individualscanpath", "tpp_gaze", "gazexplain", "scandiff"}
)


@dataclass(frozen=True, slots=True)
class AdapterDescriptor:

    model_id: str
    identity: ModelIdentity
    capabilities: CapabilitySet
    factory: Callable[[], ModelAdapter]
    requirements: RequestRequirements = RequestRequirements()
    optional_dependencies: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "model_id",
            require_string(self.model_id, "AdapterDescriptor.model_id"),
        )
        if not isinstance(self.identity, ModelIdentity):
            raise RegistryError("AdapterDescriptor.identity must be a ModelIdentity")
        if self.identity.model_id != self.model_id:
            raise RegistryError(
                "AdapterDescriptor.identity.model_id must match descriptor model_id"
            )
        if not isinstance(self.capabilities, CapabilitySet):
            raise RegistryError(
                "AdapterDescriptor.capabilities must be a CapabilitySet"
            )
        if not isinstance(self.requirements, RequestRequirements):
            raise RegistryError(
                "AdapterDescriptor.requirements must be RequestRequirements"
            )
        self.requirements.validate_against(self.capabilities)
        if not callable(self.factory):
            raise RegistryError("AdapterDescriptor.factory must be callable")
        if isinstance(self.optional_dependencies, (str, bytes, bytearray)):
            raise RegistryError(
                "AdapterDescriptor.optional_dependencies must be a sequence "
                "of import names, not a string"
            )
        dependencies = tuple(self.optional_dependencies)
        for index, dependency in enumerate(dependencies):
            require_string(
                dependency,
                f"AdapterDescriptor.optional_dependencies[{index}]",
            )
        if len(dependencies) != len(set(dependencies)):
            raise RegistryError(
                "AdapterDescriptor.optional_dependencies must not contain duplicates"
            )
        object.__setattr__(self, "optional_dependencies", dependencies)


class AdapterRegistry:

    def __init__(
        self,
        *,
        allowed_model_ids: Collection[str] = CHARTERED_MODEL_IDS,
    ) -> None:
        if isinstance(allowed_model_ids, (str, bytes, bytearray)):
            raise RegistryError(
                "allowed_model_ids must be a collection of IDs, not a string"
            )
        self._allowed_model_ids = frozenset(
            require_string(value, "allowed_model_ids[]")
            for value in allowed_model_ids
        )
        self._descriptors: dict[tuple[str, str], AdapterDescriptor] = {}

    @property
    def model_ids(self) -> tuple[str, ...]:

        return tuple(sorted({key[0] for key in self._descriptors}))

    def variants(self, model_id: str) -> tuple[str, ...]:

        stable_id = require_string(model_id, "model_id")
        variants = tuple(
            sorted(key[1] for key in self._descriptors if key[0] == stable_id)
        )
        if not variants:
            available = ", ".join(self.model_ids) or "none"
            raise UnknownModelError(
                f"no adapter registered for '{stable_id}'; available: {available}"
            )
        return variants

    def register(self, descriptor: AdapterDescriptor) -> None:

        if not isinstance(descriptor, AdapterDescriptor):
            raise RegistryError("descriptor must be an AdapterDescriptor")
        if descriptor.model_id not in self._allowed_model_ids:
            raise UnsupportedModelError(
                f"model ID '{descriptor.model_id}' is not in this registry's "
                "explicit allow-list"
            )
        key = (descriptor.model_id, descriptor.identity.model_variant)
        if key in self._descriptors:
            raise DuplicateAdapterError(
                f"adapter '{descriptor.model_id}' variant "
                f"'{descriptor.identity.model_variant}' is already registered"
            )
        self._descriptors[key] = descriptor

    def describe(
        self,
        model_id: str,
        model_variant: str | None = None,
    ) -> AdapterDescriptor:

        stable_id = require_string(model_id, "model_id")
        variants = self.variants(stable_id)
        if model_variant is None:
            if len(variants) != 1:
                choices = ", ".join(variants)
                raise RegistryError(
                    f"adapter '{stable_id}' has multiple variants; specify one "
                    f"of: {choices}"
                )
            selected_variant = variants[0]
        else:
            selected_variant = require_string(model_variant, "model_variant")
        try:
            return self._descriptors[(stable_id, selected_variant)]
        except KeyError as error:
            choices = ", ".join(variants)
            raise UnknownModelError(
                f"no adapter registered for '{stable_id}' variant "
                f"'{selected_variant}'; available variants: {choices}"
            ) from error

    def capabilities(
        self,
        model_id: str,
        model_variant: str | None = None,
    ) -> CapabilitySet:

        return self.describe(model_id, model_variant).capabilities

    def create(
        self,
        model_id: str,
        model_variant: str | None = None,
    ) -> ModelAdapter:

        descriptor = self.describe(model_id, model_variant)
        missing: list[str] = []
        for dependency in descriptor.optional_dependencies:
            try:
                available = importlib.util.find_spec(dependency) is not None
            except (ImportError, ModuleNotFoundError):
                available = False
            if not available:
                missing.append(dependency)
        if missing:
            raise AdapterUnavailableError(descriptor.model_id, tuple(missing))

        try:
            adapter = descriptor.factory()
        except ModuleNotFoundError as error:
            if error.name in descriptor.optional_dependencies:
                raise AdapterUnavailableError(
                    descriptor.model_id,
                    (error.name,),
                ) from error
            raise
        if not isinstance(adapter, ModelAdapter):
            raise RegistryError(
                f"factory for '{descriptor.model_id}' did not return a ModelAdapter"
            )
        mismatches: list[str] = []
        if adapter.identity != descriptor.identity:
            mismatches.append("identity")
        if adapter.capabilities != descriptor.capabilities:
            mismatches.append("capabilities")
        if adapter.requirements != descriptor.requirements:
            mismatches.append("requirements")
        if mismatches:
            adapter.close()
            raise RegistryError(
                f"factory for '{descriptor.model_id}' disagrees with descriptor: "
                f"{', '.join(mismatches)}"
            )
        return adapter
