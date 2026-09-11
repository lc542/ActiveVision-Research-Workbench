
from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from activevision_workbench.contracts._validation import require_string
from activevision_workbench.errors import ContractValidationError
from activevision_workbench.run_config import RunConfig

SCANDIFF_MODEL_ID = "scandiff"
SCANDIFF_FREE_VIEWING_VARIANT = "free-viewing"
SCANDIFF_VISUAL_SEARCH_VARIANT = "visual-search"
SCANDIFF_VARIANTS = frozenset(
    {SCANDIFF_FREE_VIEWING_VARIANT, SCANDIFF_VISUAL_SEARCH_VARIANT}
)
SCANDIFF_UPSTREAM_URL = "https://github.com/aimagelab/ScanDiff.git"
SCANDIFF_UPSTREAM_COMMIT = "9eff26cf2df140c09b868e7351f02d3797a37d1f"
SCANDIFF_MAX_FIXATIONS = 16
SCANDIFF_NATIVE_ARTIFACT_POLICY = "preserve-json"

_OPTION_NAMES = frozenset(
    {
        "upstream_root",
        "task_embeddings_path",
        "feature_root",
        "launcher",
        "worker_timeout_s",
        "startup_timeout_s",
        "allow_feature_download",
        "native_artifact_policy",
    }
)


@dataclass(frozen=True, slots=True)
class ScanDiffAdapterConfig:

    variant: str
    upstream_root: Path
    upstream_commit: str
    checkpoint_path: Path
    checkpoint_fingerprint: str | None
    task_embeddings_path: Path
    feature_root: Path | None
    device: str
    output_root: Path
    launcher: tuple[str, ...]
    worker_timeout_s: float = 900.0
    startup_timeout_s: float = 300.0
    allow_feature_download: bool = False
    native_artifact_policy: str = SCANDIFF_NATIVE_ARTIFACT_POLICY
    worker_module: str = "activevision_workbench.adapters.scandiff.worker"
    fake_worker: bool = False
    fake_behavior: str = "normal"

    def __post_init__(self) -> None:
        if self.variant not in SCANDIFF_VARIANTS:
            choices = ", ".join(sorted(SCANDIFF_VARIANTS))
            raise ContractValidationError(
                "ScanDiffAdapterConfig.variant",
                f"must be one of: {choices}",
            )
        if self.upstream_commit != SCANDIFF_UPSTREAM_COMMIT:
            raise ContractValidationError(
                "ScanDiffAdapterConfig.upstream_commit",
                f"must equal the pinned commit {SCANDIFF_UPSTREAM_COMMIT}",
            )
        for name in (
            "upstream_root",
            "checkpoint_path",
            "task_embeddings_path",
            "output_root",
        ):
            value = getattr(self, name)
            if not isinstance(value, Path) or not value.is_absolute():
                raise ContractValidationError(
                    f"ScanDiffAdapterConfig.{name}",
                    "must be an absolute pathlib.Path",
                )
        if self.feature_root is not None and (
            not isinstance(self.feature_root, Path)
            or not self.feature_root.is_absolute()
        ):
            raise ContractValidationError(
                "ScanDiffAdapterConfig.feature_root",
                "must be null or an absolute pathlib.Path",
            )
        if not self.fake_worker and self.device != "cuda":
            raise ContractValidationError(
                "ScanDiffAdapterConfig.device",
                "the verified upstream demo requires device='cuda'; select a "
                "physical GPU with CUDA_VISIBLE_DEVICES",
            )
        if not self.launcher:
            raise ContractValidationError(
                "ScanDiffAdapterConfig.launcher", "must not be empty"
            )
        for index, value in enumerate(self.launcher):
            require_string(value, f"ScanDiffAdapterConfig.launcher[{index}]")
        for name in ("worker_timeout_s", "startup_timeout_s"):
            value = getattr(self, name)
            if type(value) not in (int, float) or value <= 0:
                raise ContractValidationError(
                    f"ScanDiffAdapterConfig.{name}", "must be a positive number"
                )
        if type(self.allow_feature_download) is not bool:
            raise ContractValidationError(
                "ScanDiffAdapterConfig.allow_feature_download",
                "must be a boolean",
            )
        if self.native_artifact_policy != SCANDIFF_NATIVE_ARTIFACT_POLICY:
            raise ContractValidationError(
                "ScanDiffAdapterConfig.native_artifact_policy",
                f"must equal '{SCANDIFF_NATIVE_ARTIFACT_POLICY}'",
            )
        require_string(self.worker_module, "ScanDiffAdapterConfig.worker_module")
        if self.fake_behavior not in {"normal", "malformed", "timeout", "error"}:
            raise ContractValidationError(
                "ScanDiffAdapterConfig.fake_behavior",
                "must be normal, malformed, timeout, or error",
            )

    @classmethod
    def from_run_config(cls, config: RunConfig) -> "ScanDiffAdapterConfig":

        if config.model_id != SCANDIFF_MODEL_ID:
            raise ContractValidationError(
                "RunConfig.model.id", f"expected '{SCANDIFF_MODEL_ID}'"
            )
        if config.upstream_commit != SCANDIFF_UPSTREAM_COMMIT:
            raise ContractValidationError(
                "RunConfig.model.upstream_commit",
                f"must equal pinned ScanDiff commit {SCANDIFF_UPSTREAM_COMMIT}",
            )
        if config.checkpoint_path is None:
            raise ContractValidationError(
                "RunConfig.model.checkpoint_path",
                "is required by the ScanDiff adapter",
            )
        raw_options = config.model_options.get(SCANDIFF_MODEL_ID)
        if not isinstance(raw_options, Mapping):
            raise ContractValidationError(
                "RunConfig.model.options.scandiff",
                "must be an object containing ScanDiff process settings",
            )
        unknown = set(raw_options) - _OPTION_NAMES
        if unknown:
            name = sorted(unknown)[0]
            raise ContractValidationError(
                f"RunConfig.model.options.scandiff.{name}",
                "unknown ScanDiff adapter option",
            )
        upstream_root = _required_path(raw_options, "upstream_root")
        embeddings = _required_path(raw_options, "task_embeddings_path")
        feature_root = _optional_path(raw_options, "feature_root")
        launcher = _launcher(raw_options.get("launcher"))
        worker_timeout = _positive_number(
            raw_options.get("worker_timeout_s", 900.0), "worker_timeout_s"
        )
        startup_timeout = _positive_number(
            raw_options.get("startup_timeout_s", 300.0), "startup_timeout_s"
        )
        allow_download = raw_options.get("allow_feature_download", False)
        if type(allow_download) is not bool:
            raise ContractValidationError(
                "RunConfig.model.options.scandiff.allow_feature_download",
                "must be a boolean",
            )
        artifact_policy = raw_options.get(
            "native_artifact_policy", SCANDIFF_NATIVE_ARTIFACT_POLICY
        )
        if type(artifact_policy) is not str:
            raise ContractValidationError(
                "RunConfig.model.options.scandiff.native_artifact_policy",
                "must be a string",
            )
        return cls(
            variant=config.model_variant,
            upstream_root=upstream_root,
            upstream_commit=config.upstream_commit,
            checkpoint_path=Path(config.checkpoint_path).resolve(),
            checkpoint_fingerprint=config.checkpoint_fingerprint,
            task_embeddings_path=embeddings,
            feature_root=feature_root,
            device=config.device,
            output_root=Path(config.output_root).resolve(),
            launcher=launcher,
            worker_timeout_s=worker_timeout,
            startup_timeout_s=startup_timeout,
            allow_feature_download=allow_download,
            native_artifact_policy=artifact_policy,
        )

    @classmethod
    def for_fake(
        cls,
        *,
        variant: str,
        upstream_root: str | Path,
        checkpoint_path: str | Path,
        task_embeddings_path: str | Path,
        output_root: str | Path,
        feature_root: str | Path | None = None,
        worker_timeout_s: float = 5.0,
        fake_behavior: str = "normal",
    ) -> "ScanDiffAdapterConfig":

        return cls(
            variant=variant,
            upstream_root=Path(upstream_root).resolve(),
            upstream_commit=SCANDIFF_UPSTREAM_COMMIT,
            checkpoint_path=Path(checkpoint_path).resolve(),
            checkpoint_fingerprint=None,
            task_embeddings_path=Path(task_embeddings_path).resolve(),
            feature_root=(
                None if feature_root is None else Path(feature_root).resolve()
            ),
            device="cpu",
            output_root=Path(output_root).resolve(),
            launcher=(sys.executable,),
            worker_timeout_s=worker_timeout_s,
            startup_timeout_s=5.0,
            worker_module="activevision_workbench.adapters.scandiff.fake_worker",
            fake_worker=True,
            allow_feature_download=False,
            fake_behavior=fake_behavior,
        )


def adapter_option_names() -> frozenset[str]:

    return _OPTION_NAMES


def _required_path(options: Mapping[object, object], name: str) -> Path:
    if name not in options:
        raise ContractValidationError(
            f"RunConfig.model.options.scandiff.{name}", "is required"
        )
    value = require_string(
        options[name], f"RunConfig.model.options.scandiff.{name}"
    )
    return Path(value).expanduser().resolve()


def _optional_path(options: Mapping[object, object], name: str) -> Path | None:
    value = options.get(name)
    if value is None:
        return None
    text = require_string(value, f"RunConfig.model.options.scandiff.{name}")
    return Path(text).expanduser().resolve()


def _launcher(value: object) -> tuple[str, ...]:
    field = "RunConfig.model.options.scandiff.launcher"
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ContractValidationError(field, "must be a non-empty array of strings")
    result = tuple(
        require_string(item, f"{field}[{index}]")
        for index, item in enumerate(value)
    )
    if not result:
        raise ContractValidationError(field, "must not be empty")
    return result


def _positive_number(value: object, name: str) -> float:
    field = f"RunConfig.model.options.scandiff.{name}"
    if type(value) not in (int, float) or value <= 0:
        raise ContractValidationError(field, "must be a positive number")
    return float(value)
