
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from enum import Enum

from activevision_workbench.contracts import (
    CapabilitySet,
    InferenceRequest,
    ModelIdentity,
    PredictionRecord,
    RequestRequirements,
)
from activevision_workbench.contracts._validation import require_string
from activevision_workbench.errors import (
    AdapterLifecycleError,
    AdapterOutputError,
    ContractValidationError,
)


class AdapterState(str, Enum):

    NEW = "new"
    READY = "ready"
    FAILED = "failed"
    CLOSED = "closed"


class ModelAdapter(ABC):

    def __init__(self) -> None:
        self._state = AdapterState.NEW

    @property
    @abstractmethod
    def identity(self) -> ModelIdentity:
        pass

    @property
    @abstractmethod
    def capabilities(self) -> CapabilitySet:
        pass

    @property
    def requirements(self) -> RequestRequirements:

        return RequestRequirements()

    @property
    def state(self) -> AdapterState:

        return self._state

    @property
    def is_prepared(self) -> bool:

        return self._state is AdapterState.READY

    def validate_request(self, request: InferenceRequest) -> None:

        if not isinstance(request, InferenceRequest):
            raise ContractValidationError(
                "request",
                "must be an InferenceRequest",
            )
        if request.model_id != self.identity.model_id:
            raise ContractValidationError(
                "InferenceRequest.model_id",
                f"expected '{self.identity.model_id}'",
            )
        if request.model_variant != self.identity.model_variant:
            raise ContractValidationError(
                "InferenceRequest.model_variant",
                f"expected '{self.identity.model_variant}'",
            )
        request.validate_for(self.capabilities, self.requirements)
        self._validate_request(request)

    def prepare(self) -> None:

        if self._state is AdapterState.CLOSED:
            raise AdapterLifecycleError("cannot prepare a closed adapter")
        if self._state is AdapterState.FAILED:
            raise AdapterLifecycleError(
                "cannot prepare an adapter after preparation failed; close it"
            )
        if self._state is AdapterState.READY:
            return
        try:
            self._prepare()
        except Exception:
            self._state = AdapterState.FAILED
            raise
        self._state = AdapterState.READY

    def predict(
        self,
        request: InferenceRequest,
        *,
        run_id: str,
    ) -> tuple[PredictionRecord, ...]:

        if self._state is AdapterState.CLOSED:
            raise AdapterLifecycleError("cannot predict with a closed adapter")
        validated_run_id = require_string(run_id, "run_id")
        self.validate_request(request)
        self.prepare()
        produced = self._predict(request, run_id=validated_run_id)
        if not isinstance(produced, Iterable):
            raise AdapterOutputError("adapter predict result must be iterable")
        records = tuple(produced)
        self._validate_prediction_records(
            request,
            run_id=validated_run_id,
            records=records,
        )
        return records

    def close(self) -> None:

        if self._state is AdapterState.CLOSED:
            return
        try:
            self._close()
        finally:
            self._state = AdapterState.CLOSED

    def __enter__(self) -> "ModelAdapter":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def _validate_prediction_records(
        self,
        request: InferenceRequest,
        *,
        run_id: str,
        records: tuple[object, ...],
    ) -> None:
        if len(records) != request.num_samples:
            raise AdapterOutputError(
                f"expected {request.num_samples} prediction records, "
                f"received {len(records)}"
            )
        sample_ids: set[str] = set()
        sample_indices: set[int] = set()
        for index, value in enumerate(records):
            if not isinstance(value, PredictionRecord):
                raise AdapterOutputError(
                    f"prediction[{index}] must be a PredictionRecord"
                )
            record = value
            expected_fields = {
                "run_id": run_id,
                "request_id": request.request_id,
                "model": self.identity,
                "dataset_id": request.dataset_id,
                "dataset_split": request.dataset_split,
                "item_id": request.item_id,
                "observer_id": request.observer_id,
                "task_text": request.task_text,
                "target_description": request.target_description,
            }
            for field_name, expected in expected_fields.items():
                actual = getattr(record, field_name)
                if actual != expected:
                    raise AdapterOutputError(
                        f"prediction[{index}].{field_name}: expected "
                        f"{expected!r}, received {actual!r}"
                    )
            if record.sample_id in sample_ids:
                raise AdapterOutputError(
                    f"prediction[{index}].sample_id: duplicate '{record.sample_id}'"
                )
            if record.sample_index in sample_indices:
                raise AdapterOutputError(
                    f"prediction[{index}].sample_index: duplicate "
                    f"{record.sample_index}"
                )
            sample_ids.add(record.sample_id)
            sample_indices.add(record.sample_index)
            try:
                record.validate_for_capabilities(self.capabilities)
            except ContractValidationError as error:
                raise AdapterOutputError(
                    f"prediction[{index}].{error.field}: {error.reason}"
                ) from error
        expected_indices = set(range(request.num_samples))
        if sample_indices != expected_indices:
            raise AdapterOutputError(
                "prediction sample indices must be exactly "
                f"{sorted(expected_indices)}; received {sorted(sample_indices)}"
            )

    def _validate_request(self, request: InferenceRequest) -> None:
        pass

    @abstractmethod
    def _prepare(self) -> None:
        pass

    @abstractmethod
    def _predict(
        self,
        request: InferenceRequest,
        *,
        run_id: str,
    ) -> Iterable[PredictionRecord]:
        pass

    def _close(self) -> None:
        pass
