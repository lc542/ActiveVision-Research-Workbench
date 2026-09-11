
from __future__ import annotations

import random
from collections.abc import Iterable, Mapping

from activevision_workbench.adapters.base import ModelAdapter
from activevision_workbench.contracts import (
    ArtifactReference,
    Capability,
    CapabilitySet,
    FixationEvent,
    InferenceRequest,
    ModelIdentity,
    PredictionRecord,
    RequestRequirements,
    StoppingReason,
)
from activevision_workbench.contracts._validation import require_string
from activevision_workbench.errors import ContractValidationError
from activevision_workbench.registry import AdapterDescriptor, AdapterRegistry

DUMMY_MODEL_ID = "dummy"
DUMMY_MODEL_VARIANT = "deterministic-v1"
DUMMY_IDENTITY = ModelIdentity(
    model_id=DUMMY_MODEL_ID,
    model_variant=DUMMY_MODEL_VARIANT,
    adapter_version="0.1.0",
    upstream_version="builtin",
    checkpoint_id="none",
)
DUMMY_CAPABILITIES = CapabilitySet(
    frozenset(
        {
            Capability.PRODUCES_SCANPATHS,
            Capability.STOCHASTIC_MULTI_SAMPLE,
            Capability.INSTRUCTION_CONDITIONED,
            Capability.OBSERVER_CONDITIONED,
            Capability.CONTINUOUS_TIME_OUTPUT,
            Capability.FIXATION_DURATION_OUTPUT,
            Capability.TEXT_EXPLANATION_OUTPUT,
        }
    )
)
DUMMY_REQUIREMENTS = RequestRequirements(instruction=True, observer=True)


class DummyAdapter(ModelAdapter):

    def __init__(self) -> None:
        super().__init__()
        self.prepare_count = 0
        self.close_count = 0

    @property
    def identity(self) -> ModelIdentity:
        return DUMMY_IDENTITY

    @property
    def capabilities(self) -> CapabilitySet:
        return DUMMY_CAPABILITIES

    @property
    def requirements(self) -> RequestRequirements:
        return DUMMY_REQUIREMENTS

    def _validate_request(self, request: InferenceRequest) -> None:
        options = request.model_options.get(DUMMY_MODEL_ID, {})
        if not isinstance(options, Mapping):  # pragma: no cover - contract guards it
            return
        unknown = set(options) - {"explanation_prefix"}
        if unknown:
            option = sorted(unknown)[0]
            raise ContractValidationError(
                f"InferenceRequest.model_options.dummy.{option}",
                "unknown dummy adapter option",
            )
        if "explanation_prefix" in options:
            require_string(
                options["explanation_prefix"],
                "InferenceRequest.model_options.dummy.explanation_prefix",
                allow_empty=True,
            )

    def _prepare(self) -> None:
        self.prepare_count += 1

    def _predict(
        self,
        request: InferenceRequest,
        *,
        run_id: str,
    ) -> Iterable[PredictionRecord]:
        options = request.model_options.get(DUMMY_MODEL_ID, {})
        prefix = "Dummy"
        if isinstance(options, Mapping) and "explanation_prefix" in options:
            configured_prefix = options["explanation_prefix"]
            if type(configured_prefix) is str:
                prefix = configured_prefix

        requested_fixations = request.max_fixations or 3
        maximum_fixations = min(requested_fixations, 3)
        for sample_index in range(request.num_samples):
            seed = request.base_seed + sample_index
            random_source = random.Random(seed)
            fixations: list[FixationEvent] = []
            for sequence_index in range(maximum_fixations):
                timestamp_s = sequence_index * 0.25
                if (
                    request.time_horizon_s is not None
                    and timestamp_s > request.time_horizon_s
                ):
                    break
                x_norm = random_source.random()
                y_norm = random_source.random()
                x_px = None
                y_px = None
                if request.image_width is not None and request.image_height is not None:
                    x_px = x_norm * (request.image_width - 1)
                    y_px = y_norm * (request.image_height - 1)
                fixations.append(
                    FixationEvent(
                        x_px=x_px,
                        y_px=y_px,
                        x_norm=x_norm,
                        y_norm=y_norm,
                        sequence_index=sequence_index,
                        timestamp_s=timestamp_s,
                        duration_s=0.1 + random_source.random() * 0.05,
                        confidence=0.9,
                        native_timing={
                            "value": int(timestamp_s * 1000),
                            "unit": "ms",
                            "meaning": "timestamp",
                        },
                        native_metadata={"source": "dummy"},
                    )
                )

            if (
                request.time_horizon_s is not None
                and len(fixations) < maximum_fixations
            ):
                stopping_reason = StoppingReason.TIME_HORIZON
            elif (
                request.max_fixations is not None
                and len(fixations) >= request.max_fixations
            ):
                stopping_reason = StoppingReason.MAX_FIXATIONS
            else:
                stopping_reason = StoppingReason.COMPLETED
            task = (
                request.task_text
                if request.task_text is not None
                else request.target_description
            )
            yield PredictionRecord(
                run_id=run_id,
                request_id=request.request_id,
                model=self.identity,
                dataset_id=request.dataset_id,
                dataset_split=request.dataset_split,
                item_id=request.item_id,
                sample_id=f"{request.request_id}:sample:{sample_index}",
                sample_index=sample_index,
                seed=seed,
                observer_id=request.observer_id,
                task_text=request.task_text,
                target_description=request.target_description,
                fixations=tuple(fixations),
                explanation_text=f"{prefix} sample {sample_index} for {task!r}",
                native_artifacts=(
                    ArtifactReference(
                        uri=(
                            f"memory://dummy/{request.request_id}/"
                            f"sample-{sample_index}.json"
                        ),
                        kind="model_native_prediction",
                        media_type="application/json",
                        metadata={"sample_index": sample_index},
                    ),
                ),
                native_metadata={
                    "dummy": {
                        "generator": "python_random",
                        "base_seed": request.base_seed,
                    }
                },
                stopping_reason=stopping_reason,
            )

    def _close(self) -> None:
        self.close_count += 1


def dummy_descriptor() -> AdapterDescriptor:

    return AdapterDescriptor(
        model_id=DUMMY_MODEL_ID,
        identity=DUMMY_IDENTITY,
        capabilities=DUMMY_CAPABILITIES,
        requirements=DUMMY_REQUIREMENTS,
        factory=DummyAdapter,
    )


def create_dummy_registry() -> AdapterRegistry:

    registry = AdapterRegistry(allowed_model_ids={DUMMY_MODEL_ID})
    registry.register(dummy_descriptor())
    return registry
