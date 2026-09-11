
from __future__ import annotations

import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from activevision_workbench.contracts._validation import require_string
from activevision_workbench.errors import ContractValidationError
from activevision_workbench.run_config import RunConfig

TPP_GAZE_MODEL_ID = "tpp_gaze"
TPP_GAZE_TRANSFORMER_VARIANT = "transformer"
TPP_GAZE_UPSTREAM_URL = "https://github.com/phuselab/tppgaze.git"
TPP_GAZE_UPSTREAM_COMMIT = "59cf572ba679b8e3351da29dbbbd2ff66bcfa552"
TPP_GAZE_TRANSFORMER_CHECKPOINT_SHA256 = (
    "f925a322932438947e27ed407a40649a94275c120f12e4ae201ee2665024ce9d"
)
TPP_GAZE_TRANSFORMER_CONFIG_SHA256 = (
    "e31d4d166733af6ecf70300596ec9eb3d44935de8d59e2284d390a16c5ff8406"
)
TPP_GAZE_NATIVE_ARTIFACT_POLICY = "preserve-json"
TPP_GAZE_NATIVE_TIME_UNITS = {
    "seconds": 1.0,
    "milliseconds": 0.001,
}

_OPTION_NAMES = frozenset(
    {
        "upstream_root",
        "model_config_path",
        "temperature",
        "launcher",
        "worker_timeout_s",
        "startup_timeout_s",
        "native_time_unit",
        "temporal_scale_to_seconds",
        "safety_max_fixations",
        "native_artifact_policy",
    }
)


@dataclass(frozen=True, slots=True)
class TPPGazeAdapterConfig:

    variant: str
    upstream_root: Path
    upstream_commit: str
    checkpoint_path: Path
    checkpoint_fingerprint: str | None
    model_config_path: Path
    device: str
    output_root: Path
    launcher: tuple[str, ...]
    temperature: float | None = None
    native_time_unit: str = "milliseconds"
    temporal_scale_to_seconds: float = 0.001
    safety_max_fixations: int = 512
    worker_timeout_s: float = 900.0
    startup_timeout_s: float = 300.0
    native_artifact_policy: str = TPP_GAZE_NATIVE_ARTIFACT_POLICY
    worker_module: str = "activevision_workbench.adapters.tpp_gaze.worker"
    fake_worker: bool = False
    fake_behavior: str = "normal"

    def __post_init__(self) -> None:
        if self.variant != TPP_GAZE_TRANSFORMER_VARIANT:
            raise ContractValidationError(
                "TPPGazeAdapterConfig.variant",
                f"must equal '{TPP_GAZE_TRANSFORMER_VARIANT}' for the verified config",
            )
        if self.upstream_commit != TPP_GAZE_UPSTREAM_COMMIT:
            raise ContractValidationError(
                "TPPGazeAdapterConfig.upstream_commit",
                f"must equal the pinned commit {TPP_GAZE_UPSTREAM_COMMIT}",
            )
        for name in (
            "upstream_root",
            "checkpoint_path",
            "model_config_path",
            "output_root",
        ):
            value = getattr(self, name)
            if not isinstance(value, Path) or not value.is_absolute():
                raise ContractValidationError(
                    f"TPPGazeAdapterConfig.{name}",
                    "must be an absolute pathlib.Path",
                )
        if self.device not in {"cpu", "cuda"}:
            raise ContractValidationError(
                "TPPGazeAdapterConfig.device", "must be 'cpu' or 'cuda'"
            )
        if not self.launcher:
            raise ContractValidationError(
                "TPPGazeAdapterConfig.launcher", "must not be empty"
            )
        for index, value in enumerate(self.launcher):
            require_string(value, f"TPPGazeAdapterConfig.launcher[{index}]")
        if self.temperature is not None and (
            type(self.temperature) not in (int, float)
            or not math.isfinite(self.temperature)
            or self.temperature <= 0
        ):
            raise ContractValidationError(
                "TPPGazeAdapterConfig.temperature",
                "must be null or a positive finite number",
            )
        expected_scale = TPP_GAZE_NATIVE_TIME_UNITS.get(self.native_time_unit)
        if expected_scale is None:
            choices = ", ".join(sorted(TPP_GAZE_NATIVE_TIME_UNITS))
            raise ContractValidationError(
                "TPPGazeAdapterConfig.native_time_unit",
                f"must be one of: {choices}",
            )
        if type(self.temporal_scale_to_seconds) not in (int, float) or (
            self.temporal_scale_to_seconds <= 0
        ):
            raise ContractValidationError(
                "TPPGazeAdapterConfig.temporal_scale_to_seconds",
                "must be a positive number",
            )
        if float(self.temporal_scale_to_seconds) != expected_scale:
            raise ContractValidationError(
                "TPPGazeAdapterConfig.temporal_scale_to_seconds",
                f"must be {expected_scale:g} for native_time_unit="
                f"'{self.native_time_unit}'",
            )
        if type(self.safety_max_fixations) is not int or (
            self.safety_max_fixations < 1
        ):
            raise ContractValidationError(
                "TPPGazeAdapterConfig.safety_max_fixations",
                "must be a positive integer",
            )
        for name in ("worker_timeout_s", "startup_timeout_s"):
            value = getattr(self, name)
            if type(value) not in (int, float) or value <= 0:
                raise ContractValidationError(
                    f"TPPGazeAdapterConfig.{name}", "must be a positive number"
                )
        if self.native_artifact_policy != TPP_GAZE_NATIVE_ARTIFACT_POLICY:
            raise ContractValidationError(
                "TPPGazeAdapterConfig.native_artifact_policy",
                f"must equal '{TPP_GAZE_NATIVE_ARTIFACT_POLICY}'",
            )
        require_string(self.worker_module, "TPPGazeAdapterConfig.worker_module")
        if self.fake_behavior not in {
            "normal",
            "absolute_only",
            "empty",
            "decreasing_time",
            "negative_time",
            "malformed",
            "timeout",
            "error",
        }:
            raise ContractValidationError(
                "TPPGazeAdapterConfig.fake_behavior",
                "is not a supported fake-worker behavior",
            )

    @classmethod
    def from_run_config(cls, config: RunConfig) -> "TPPGazeAdapterConfig":

        if config.model_id != TPP_GAZE_MODEL_ID:
            raise ContractValidationError(
                "RunConfig.model.id", f"expected '{TPP_GAZE_MODEL_ID}'"
            )
        if config.upstream_commit != TPP_GAZE_UPSTREAM_COMMIT:
            raise ContractValidationError(
                "RunConfig.model.upstream_commit",
                f"must equal pinned TPP-Gaze commit {TPP_GAZE_UPSTREAM_COMMIT}",
            )
        if config.checkpoint_path is None:
            raise ContractValidationError(
                "RunConfig.model.checkpoint_path",
                "is required by the TPP-Gaze adapter",
            )
        raw_options = config.model_options.get(TPP_GAZE_MODEL_ID)
        if not isinstance(raw_options, Mapping):
            raise ContractValidationError(
                "RunConfig.model.options.tpp_gaze",
                "must be an object containing TPP-Gaze worker settings",
            )
        unknown = set(raw_options) - _OPTION_NAMES
        if unknown:
            name = sorted(unknown)[0]
            raise ContractValidationError(
                f"RunConfig.model.options.tpp_gaze.{name}",
                "unknown TPP-Gaze adapter option",
            )
        native_unit = raw_options.get("native_time_unit", "milliseconds")
        native_unit = require_string(
            native_unit, "RunConfig.model.options.tpp_gaze.native_time_unit"
        )
        default_scale = TPP_GAZE_NATIVE_TIME_UNITS.get(native_unit)
        if default_scale is None:
            choices = ", ".join(sorted(TPP_GAZE_NATIVE_TIME_UNITS))
            raise ContractValidationError(
                "RunConfig.model.options.tpp_gaze.native_time_unit",
                f"must be one of: {choices}",
            )
        return cls(
            variant=config.model_variant,
            upstream_root=_required_path(raw_options, "upstream_root"),
            upstream_commit=config.upstream_commit,
            checkpoint_path=Path(config.checkpoint_path).resolve(),
            checkpoint_fingerprint=config.checkpoint_fingerprint,
            model_config_path=_required_path(raw_options, "model_config_path"),
            device=config.device,
            output_root=Path(config.output_root).resolve(),
            launcher=_launcher(raw_options.get("launcher")),
            temperature=_optional_positive_number(
                raw_options.get("temperature"), "temperature"
            ),
            native_time_unit=native_unit,
            temporal_scale_to_seconds=_positive_number(
                raw_options.get("temporal_scale_to_seconds", default_scale),
                "temporal_scale_to_seconds",
            ),
            safety_max_fixations=_positive_integer(
                raw_options.get("safety_max_fixations", 512),
                "safety_max_fixations",
            ),
            worker_timeout_s=_positive_number(
                raw_options.get("worker_timeout_s", 900.0), "worker_timeout_s"
            ),
            startup_timeout_s=_positive_number(
                raw_options.get("startup_timeout_s", 300.0), "startup_timeout_s"
            ),
            native_artifact_policy=require_string(
                raw_options.get(
                    "native_artifact_policy", TPP_GAZE_NATIVE_ARTIFACT_POLICY
                ),
                "RunConfig.model.options.tpp_gaze.native_artifact_policy",
            ),
        )

    @classmethod
    def for_fake(
        cls,
        *,
        upstream_root: str | Path,
        checkpoint_path: str | Path,
        model_config_path: str | Path,
        output_root: str | Path,
        temperature: float | None = None,
        native_time_unit: str = "milliseconds",
        temporal_scale_to_seconds: float = 0.001,
        safety_max_fixations: int = 8,
        worker_timeout_s: float = 5.0,
        fake_behavior: str = "normal",
    ) -> "TPPGazeAdapterConfig":

        return cls(
            variant=TPP_GAZE_TRANSFORMER_VARIANT,
            upstream_root=Path(upstream_root).resolve(),
            upstream_commit=TPP_GAZE_UPSTREAM_COMMIT,
            checkpoint_path=Path(checkpoint_path).resolve(),
            checkpoint_fingerprint=None,
            model_config_path=Path(model_config_path).resolve(),
            device="cpu",
            output_root=Path(output_root).resolve(),
            launcher=(sys.executable,),
            temperature=temperature,
            native_time_unit=native_time_unit,
            temporal_scale_to_seconds=temporal_scale_to_seconds,
            safety_max_fixations=safety_max_fixations,
            worker_timeout_s=worker_timeout_s,
            startup_timeout_s=5.0,
            worker_module="activevision_workbench.adapters.tpp_gaze.fake_worker",
            fake_worker=True,
            fake_behavior=fake_behavior,
        )


def adapter_option_names() -> frozenset[str]:

    return _OPTION_NAMES


def _required_path(options: Mapping[object, object], name: str) -> Path:
    if name not in options:
        raise ContractValidationError(
            f"RunConfig.model.options.tpp_gaze.{name}", "is required"
        )
    value = require_string(
        options[name], f"RunConfig.model.options.tpp_gaze.{name}"
    )
    return Path(value).expanduser().resolve()


def _launcher(value: object) -> tuple[str, ...]:
    field = "RunConfig.model.options.tpp_gaze.launcher"
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ContractValidationError(field, "must be a non-empty array of strings")
    launcher = tuple(
        require_string(item, f"{field}[{index}]")
        for index, item in enumerate(value)
    )
    if not launcher:
        raise ContractValidationError(field, "must not be empty")
    return launcher


def _positive_number(value: object, name: str) -> float:
    field = f"RunConfig.model.options.tpp_gaze.{name}"
    if type(value) not in (int, float) or value <= 0:
        raise ContractValidationError(field, "must be a positive number")
    return float(value)


def _optional_positive_number(value: object, name: str) -> float | None:
    if value is None:
        return None
    field = f"RunConfig.model.options.tpp_gaze.{name}"
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ContractValidationError(
            field, "must be null or a positive finite number"
        )
    return float(value)


def _positive_integer(value: object, name: str) -> int:
    field = f"RunConfig.model.options.tpp_gaze.{name}"
    if type(value) is not int or value < 1:
        raise ContractValidationError(field, "must be a positive integer")
    return value
