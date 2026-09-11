
from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from activevision_workbench.contracts._validation import require_string
from activevision_workbench.errors import ContractValidationError
from activevision_workbench.run_config import RunConfig

INDIVIDUAL_SCANPATH_MODEL_ID = "individualscanpath"
INDIVIDUAL_SCANPATH_CHENLSTM_OSIE_VARIANT = "chenlstm-osie"
INDIVIDUAL_SCANPATH_UPSTREAM_URL = "https://github.com/chenxy99/IndividualScanpath.git"
INDIVIDUAL_SCANPATH_UPSTREAM_COMMIT = "50adc583144e910b2a520ca59f865e5581390938"
INDIVIDUAL_SCANPATH_CHENLSTM_OSIE_CHECKPOINT_SHA256 = (
    "ce89bf36d0b49f9cdc73ae408a66085b2286666d8d9b249aa7c126ac001f0e3e"
)
INDIVIDUAL_SCANPATH_DATASET_ID = "osie"
INDIVIDUAL_SCANPATH_SUBJECT_COUNT = 15
INDIVIDUAL_SCANPATH_EMBEDDING_DIM = 128
INDIVIDUAL_SCANPATH_INPUT_WIDTH = 320
INDIVIDUAL_SCANPATH_INPUT_HEIGHT = 240
INDIVIDUAL_SCANPATH_MAP_WIDTH = 40
INDIVIDUAL_SCANPATH_MAP_HEIGHT = 30
INDIVIDUAL_SCANPATH_MODEL_MAX_FIXATIONS = 16
INDIVIDUAL_SCANPATH_NATIVE_ARTIFACT_POLICY = "preserve-json"
INDIVIDUAL_SCANPATH_UNKNOWN_OBSERVER_POLICY = "error"

_OPTION_NAMES = frozenset(
    {
        "upstream_root",
        "observer_mapping_path",
        "observer_mapping_fingerprint",
        "unknown_observer_policy",
        "launcher",
        "worker_timeout_s",
        "startup_timeout_s",
        "min_fixations",
        "model_max_fixations",
        "native_artifact_policy",
    }
)


@dataclass(frozen=True, slots=True)
class IndividualScanpathAdapterConfig:

    variant: str
    upstream_root: Path
    upstream_commit: str
    checkpoint_path: Path
    checkpoint_fingerprint: str | None
    observer_mapping_path: Path
    observer_mapping_fingerprint: str
    unknown_observer_policy: str
    device: str
    output_root: Path
    launcher: tuple[str, ...]
    min_fixations: int = 1
    model_max_fixations: int = INDIVIDUAL_SCANPATH_MODEL_MAX_FIXATIONS
    worker_timeout_s: float = 900.0
    startup_timeout_s: float = 300.0
    native_artifact_policy: str = INDIVIDUAL_SCANPATH_NATIVE_ARTIFACT_POLICY
    worker_module: str = "activevision_workbench.adapters.individual_scanpath.worker"
    fake_worker: bool = False
    fake_behavior: str = "normal"

    def __post_init__(self) -> None:
        if self.variant != INDIVIDUAL_SCANPATH_CHENLSTM_OSIE_VARIANT:
            raise ContractValidationError(
                "IndividualScanpathAdapterConfig.variant",
                f"must equal '{INDIVIDUAL_SCANPATH_CHENLSTM_OSIE_VARIANT}'",
            )
        if self.upstream_commit != INDIVIDUAL_SCANPATH_UPSTREAM_COMMIT:
            raise ContractValidationError(
                "IndividualScanpathAdapterConfig.upstream_commit",
                f"must equal pinned commit {INDIVIDUAL_SCANPATH_UPSTREAM_COMMIT}",
            )
        for name in ("upstream_root", "checkpoint_path", "observer_mapping_path", "output_root"):
            value = getattr(self, name)
            if not isinstance(value, Path) or not value.is_absolute():
                raise ContractValidationError(
                    f"IndividualScanpathAdapterConfig.{name}",
                    "must be an absolute pathlib.Path",
                )
        fingerprint = self.observer_mapping_fingerprint.removeprefix("sha256:")
        if len(fingerprint) != 64 or any(character not in "0123456789abcdefABCDEF" for character in fingerprint):
            raise ContractValidationError(
                "IndividualScanpathAdapterConfig.observer_mapping_fingerprint",
                "must be a SHA-256 fingerprint",
            )
        if self.unknown_observer_policy != INDIVIDUAL_SCANPATH_UNKNOWN_OBSERVER_POLICY:
            raise ContractValidationError(
                "IndividualScanpathAdapterConfig.unknown_observer_policy",
                "must equal 'error'; the pinned upstream has no unknown token or mean-embedding fallback",
            )
        if not self.fake_worker and self.device != "cuda":
            raise ContractValidationError(
                "IndividualScanpathAdapterConfig.device",
                "the verified upstream inference path requires device='cuda'",
            )
        if not self.launcher:
            raise ContractValidationError("IndividualScanpathAdapterConfig.launcher", "must not be empty")
        for index, value in enumerate(self.launcher):
            require_string(value, f"IndividualScanpathAdapterConfig.launcher[{index}]")
        if type(self.model_max_fixations) is not int or not (1 <= self.model_max_fixations <= INDIVIDUAL_SCANPATH_MODEL_MAX_FIXATIONS):
            raise ContractValidationError(
                "IndividualScanpathAdapterConfig.model_max_fixations",
                f"must be an integer from 1 through {INDIVIDUAL_SCANPATH_MODEL_MAX_FIXATIONS}",
            )
        if type(self.min_fixations) is not int or not (0 <= self.min_fixations < self.model_max_fixations):
            raise ContractValidationError(
                "IndividualScanpathAdapterConfig.min_fixations",
                "must be an integer from 0 through model_max_fixations - 1",
            )
        for name in ("worker_timeout_s", "startup_timeout_s"):
            value = getattr(self, name)
            if type(value) not in (int, float) or value <= 0:
                raise ContractValidationError(f"IndividualScanpathAdapterConfig.{name}", "must be a positive number")
        if self.native_artifact_policy != INDIVIDUAL_SCANPATH_NATIVE_ARTIFACT_POLICY:
            raise ContractValidationError(
                "IndividualScanpathAdapterConfig.native_artifact_policy",
                f"must equal '{INDIVIDUAL_SCANPATH_NATIVE_ARTIFACT_POLICY}'",
            )
        require_string(self.worker_module, "IndividualScanpathAdapterConfig.worker_module")
        if self.fake_behavior not in {"normal", "malformed", "timeout", "error"}:
            raise ContractValidationError(
                "IndividualScanpathAdapterConfig.fake_behavior",
                "must be normal, malformed, timeout, or error",
            )

    @classmethod
    def from_run_config(cls, config: RunConfig) -> "IndividualScanpathAdapterConfig":
        if config.model_id != INDIVIDUAL_SCANPATH_MODEL_ID:
            raise ContractValidationError("RunConfig.model.id", f"expected '{INDIVIDUAL_SCANPATH_MODEL_ID}'")
        if config.upstream_commit != INDIVIDUAL_SCANPATH_UPSTREAM_COMMIT:
            raise ContractValidationError("RunConfig.model.upstream_commit", f"must equal pinned commit {INDIVIDUAL_SCANPATH_UPSTREAM_COMMIT}")
        if config.checkpoint_path is None:
            raise ContractValidationError("RunConfig.model.checkpoint_path", "is required")
        raw = config.model_options.get(INDIVIDUAL_SCANPATH_MODEL_ID)
        if not isinstance(raw, Mapping):
            raise ContractValidationError("RunConfig.model.options.individualscanpath", "must be an object")
        unknown = set(raw) - _OPTION_NAMES
        if unknown:
            name = sorted(unknown)[0]
            raise ContractValidationError(f"RunConfig.model.options.individualscanpath.{name}", "unknown IndividualScanpath option")
        return cls(
            variant=config.model_variant,
            upstream_root=_required_path(raw, "upstream_root"),
            upstream_commit=config.upstream_commit,
            checkpoint_path=Path(config.checkpoint_path).resolve(),
            checkpoint_fingerprint=config.checkpoint_fingerprint,
            observer_mapping_path=_required_path(raw, "observer_mapping_path"),
            observer_mapping_fingerprint=require_string(raw.get("observer_mapping_fingerprint"), "RunConfig.model.options.individualscanpath.observer_mapping_fingerprint"),
            unknown_observer_policy=require_string(raw.get("unknown_observer_policy", "error"), "RunConfig.model.options.individualscanpath.unknown_observer_policy"),
            device=config.device,
            output_root=Path(config.output_root).resolve(),
            launcher=_launcher(raw.get("launcher")),
            min_fixations=_integer(raw.get("min_fixations", 1), "min_fixations"),
            model_max_fixations=_integer(raw.get("model_max_fixations", 16), "model_max_fixations"),
            worker_timeout_s=_positive(raw.get("worker_timeout_s", 900), "worker_timeout_s"),
            startup_timeout_s=_positive(raw.get("startup_timeout_s", 300), "startup_timeout_s"),
            native_artifact_policy=require_string(raw.get("native_artifact_policy", INDIVIDUAL_SCANPATH_NATIVE_ARTIFACT_POLICY), "RunConfig.model.options.individualscanpath.native_artifact_policy"),
        )

    @classmethod
    def for_fake(
        cls,
        *,
        upstream_root: str | Path,
        checkpoint_path: str | Path,
        observer_mapping_path: str | Path,
        observer_mapping_fingerprint: str,
        output_root: str | Path,
        worker_timeout_s: float = 5.0,
        fake_behavior: str = "normal",
    ) -> "IndividualScanpathAdapterConfig":
        return cls(
            variant=INDIVIDUAL_SCANPATH_CHENLSTM_OSIE_VARIANT,
            upstream_root=Path(upstream_root).resolve(),
            upstream_commit=INDIVIDUAL_SCANPATH_UPSTREAM_COMMIT,
            checkpoint_path=Path(checkpoint_path).resolve(),
            checkpoint_fingerprint=None,
            observer_mapping_path=Path(observer_mapping_path).resolve(),
            observer_mapping_fingerprint=observer_mapping_fingerprint,
            unknown_observer_policy="error",
            device="cpu",
            output_root=Path(output_root).resolve(),
            launcher=(sys.executable,),
            worker_timeout_s=worker_timeout_s,
            startup_timeout_s=5.0,
            worker_module="activevision_workbench.adapters.individual_scanpath.fake_worker",
            fake_worker=True,
            fake_behavior=fake_behavior,
        )


def adapter_option_names() -> frozenset[str]:
    return _OPTION_NAMES


def _required_path(options: Mapping[object, object], name: str) -> Path:
    value = require_string(options.get(name), f"RunConfig.model.options.individualscanpath.{name}")
    return Path(value).expanduser().resolve()


def _launcher(value: object) -> tuple[str, ...]:
    field = "RunConfig.model.options.individualscanpath.launcher"
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ContractValidationError(field, "must be a non-empty array of strings")
    result = tuple(require_string(item, f"{field}[{index}]") for index, item in enumerate(value))
    if not result:
        raise ContractValidationError(field, "must not be empty")
    return result


def _integer(value: object, name: str) -> int:
    if type(value) is not int:
        raise ContractValidationError(f"RunConfig.model.options.individualscanpath.{name}", "must be an integer")
    return value


def _positive(value: object, name: str) -> float:
    if type(value) not in (int, float) or value <= 0:
        raise ContractValidationError(f"RunConfig.model.options.individualscanpath.{name}", "must be a positive number")
    return float(value)
