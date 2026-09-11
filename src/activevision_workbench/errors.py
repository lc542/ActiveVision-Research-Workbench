
from __future__ import annotations


class WorkbenchError(Exception):
    pass


class ContractValidationError(WorkbenchError, ValueError):

    def __init__(self, field: str, reason: str) -> None:
        self.field = field
        self.reason = reason
        super().__init__(f"{field}: {reason}")


class SerializationError(WorkbenchError, ValueError):
    pass


class RunConfigurationError(WorkbenchError, ValueError):
    pass


class RunError(WorkbenchError):
    pass


class RunExistsError(RunError, FileExistsError):
    pass


class RunResumeError(RunError, ValueError):
    pass


class EvaluationError(WorkbenchError):
    pass


class EvaluationConfigurationError(EvaluationError, ValueError):
    pass


class EvaluationDataError(EvaluationError, ValueError):
    pass


class EvaluationOutputError(EvaluationError, OSError):
    pass


class InspectionError(WorkbenchError):
    pass


class InspectionConfigurationError(InspectionError, ValueError):
    pass


class InspectionDataError(InspectionError, ValueError):
    pass


class InspectionOutputError(InspectionError, OSError):
    pass


class SlurmError(WorkbenchError):
    pass


class SlurmConfigurationError(SlurmError, ValueError):
    pass


class SlurmPlanError(SlurmError, ValueError):
    pass


class SlurmSubmissionError(SlurmError, RuntimeError):
    pass


class SlurmMergeError(SlurmError, ValueError):
    pass


class DatasetError(WorkbenchError):
    pass


class DataRootError(DatasetError, FileNotFoundError):
    pass


class DatasetAnnotationError(DatasetError, ValueError):
    pass


class DatasetItemNotFoundError(DatasetError, LookupError):
    pass


class DatasetDependencyError(DatasetError, ImportError):
    pass


class CoordinateOutOfBoundsError(DatasetError, ValueError):
    pass


class AdapterError(WorkbenchError):
    pass


class AdapterLifecycleError(AdapterError):
    pass


class AdapterOutputError(AdapterError):
    pass


class RegistryError(WorkbenchError):
    pass


class DuplicateAdapterError(RegistryError):
    pass


class UnknownModelError(RegistryError, LookupError):
    pass


class UnsupportedModelError(RegistryError, ValueError):
    pass


class AdapterUnavailableError(RegistryError, ImportError):

    def __init__(self, model_id: str, missing_dependencies: tuple[str, ...]) -> None:
        self.model_id = model_id
        self.missing_dependencies = missing_dependencies
        missing = ", ".join(missing_dependencies)
        super().__init__(
            f"adapter '{model_id}' is unavailable; missing optional "
            f"dependencies: {missing}. Install them in the adapter's isolated "
            "environment; the registry will not install packages."
        )
