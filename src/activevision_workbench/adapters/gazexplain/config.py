
from __future__ import annotations

import hashlib
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from activevision_workbench.contracts._validation import require_string
from activevision_workbench.errors import ContractValidationError
from activevision_workbench.run_config import RunConfig

GAZEXPLAIN_MODEL_ID = "gazexplain"
GAZEXPLAIN_JOINT_VARIANT = "joint-all"
GAZEXPLAIN_UPSTREAM_URL = "https://github.com/chenxy99/GazeXplain.git"
GAZEXPLAIN_UPSTREAM_COMMIT = "443f4aaa1c53f6804a3a503cd2842c0b5d225da4"
GAZEXPLAIN_JOINT_CHECKPOINT_SHA256 = (
    "051447807b0b0d20ddb776c1e1e4c27665f90b4dc02801a4f9b6bdb0bdb0280f"
)
GAZEXPLAIN_JOINT_HPARAMS_SHA256 = (
    "1c606c286ec5c73e85ad4c4df34abb53770a21e93c795d3e5f4a35b31666d3a9"
)
GAZEXPLAIN_ROBERTA_REVISION = "e2da8e2f811d1448a5b465c236feacd80ffbac7b"
GAZEXPLAIN_BLIP_REVISION = "82a37760796d32b1411fe092ab5d4e227313294b"
GAZEXPLAIN_MASKRCNN_COCO_SHA256 = (
    "bf2d0c1efbc936eeee2bc95a48e80ebc86b891f61b0106485937fc29f9315fc0"
)
GAZEXPLAIN_MODEL_INPUT_WIDTH = 512
GAZEXPLAIN_MODEL_INPUT_HEIGHT = 384
GAZEXPLAIN_FEATURE_INPUT_WIDTH = 1024
GAZEXPLAIN_FEATURE_INPUT_HEIGHT = 768
GAZEXPLAIN_MAP_WIDTH = 32
GAZEXPLAIN_MAP_HEIGHT = 24
GAZEXPLAIN_MODEL_MAX_FIXATIONS = 16
GAZEXPLAIN_DEFAULT_MAX_GENERATION_LENGTH = 20
GAZEXPLAIN_DEFAULT_NUM_EXPLANATION_BEAMS = 3
GAZEXPLAIN_DEFAULT_MODEL_ASSETS_ROOT = "/opt/avrw/gazexplain-assets"
GAZEXPLAIN_NATIVE_ARTIFACT_POLICY = "preserve-json"

_OPTION_NAMES = frozenset(
    {
        "upstream_root",
        "hparams_path",
        "hparams_fingerprint",
        "launcher",
        "worker_timeout_s",
        "startup_timeout_s",
        "model_max_fixations",
        "max_generation_length",
        "num_explanation_beams",
        "model_assets_root",
        "native_artifact_policy",
    }
)


@dataclass(frozen=True, slots=True)
class GazeXplainAdapterConfig:

    variant: str
    upstream_root: Path
    upstream_commit: str
    checkpoint_path: Path
    checkpoint_fingerprint: str | None
    hparams_path: Path
    hparams_fingerprint: str
    device: str
    output_root: Path
    launcher: tuple[str, ...]
    model_assets_root: str = GAZEXPLAIN_DEFAULT_MODEL_ASSETS_ROOT
    model_max_fixations: int = GAZEXPLAIN_MODEL_MAX_FIXATIONS
    max_generation_length: int = GAZEXPLAIN_DEFAULT_MAX_GENERATION_LENGTH
    num_explanation_beams: int = GAZEXPLAIN_DEFAULT_NUM_EXPLANATION_BEAMS
    worker_timeout_s: float = 1800.0
    startup_timeout_s: float = 600.0
    native_artifact_policy: str = GAZEXPLAIN_NATIVE_ARTIFACT_POLICY
    worker_module: str = "activevision_workbench.adapters.gazexplain.worker"
    fake_worker: bool = False
    fake_behavior: str = "normal"

    def __post_init__(self) -> None:
        if self.variant != GAZEXPLAIN_JOINT_VARIANT:
            raise ContractValidationError(
                "GazeXplainAdapterConfig.variant",
                f"must equal '{GAZEXPLAIN_JOINT_VARIANT}'",
            )
        if self.upstream_commit != GAZEXPLAIN_UPSTREAM_COMMIT:
            raise ContractValidationError(
                "GazeXplainAdapterConfig.upstream_commit",
                f"must equal pinned commit {GAZEXPLAIN_UPSTREAM_COMMIT}",
            )
        for name in (
            "upstream_root",
            "checkpoint_path",
            "hparams_path",
            "output_root",
        ):
            value = getattr(self, name)
            if not isinstance(value, Path) or not value.is_absolute():
                raise ContractValidationError(
                    f"GazeXplainAdapterConfig.{name}",
                    "must be an absolute pathlib.Path",
                )
        _fingerprint(
            self.hparams_fingerprint,
            "GazeXplainAdapterConfig.hparams_fingerprint",
        )
        if self.checkpoint_fingerprint is not None:
            _fingerprint(
                self.checkpoint_fingerprint,
                "GazeXplainAdapterConfig.checkpoint_fingerprint",
            )
        if not self.fake_worker and self.device != "cuda":
            raise ContractValidationError(
                "GazeXplainAdapterConfig.device",
                "the verified upstream inference path requires device='cuda'",
            )
        if not self.launcher:
            raise ContractValidationError(
                "GazeXplainAdapterConfig.launcher", "must not be empty"
            )
        for index, value in enumerate(self.launcher):
            require_string(value, f"GazeXplainAdapterConfig.launcher[{index}]")
        require_string(
            self.model_assets_root, "GazeXplainAdapterConfig.model_assets_root"
        )
        if not Path(self.model_assets_root).is_absolute():
            raise ContractValidationError(
                "GazeXplainAdapterConfig.model_assets_root",
                "must be an absolute path inside the worker environment",
            )
        if self.model_max_fixations != GAZEXPLAIN_MODEL_MAX_FIXATIONS:
            raise ContractValidationError(
                "GazeXplainAdapterConfig.model_max_fixations",
                f"must equal the checkpoint sequence length {GAZEXPLAIN_MODEL_MAX_FIXATIONS}",
            )
        if type(self.max_generation_length) is not int or not (
            3 <= self.max_generation_length <= 128
        ):
            raise ContractValidationError(
                "GazeXplainAdapterConfig.max_generation_length",
                "must be an integer from 3 through 128",
            )
        if type(self.num_explanation_beams) is not int or not (
            1 <= self.num_explanation_beams <= 16
        ):
            raise ContractValidationError(
                "GazeXplainAdapterConfig.num_explanation_beams",
                "must be an integer from 1 through 16",
            )
        for name in ("worker_timeout_s", "startup_timeout_s"):
            value = getattr(self, name)
            if type(value) not in (int, float) or value <= 0:
                raise ContractValidationError(
                    f"GazeXplainAdapterConfig.{name}",
                    "must be a positive number",
                )
        if self.native_artifact_policy != GAZEXPLAIN_NATIVE_ARTIFACT_POLICY:
            raise ContractValidationError(
                "GazeXplainAdapterConfig.native_artifact_policy",
                f"must equal '{GAZEXPLAIN_NATIVE_ARTIFACT_POLICY}'",
            )
        require_string(self.worker_module, "GazeXplainAdapterConfig.worker_module")
        if self.fake_behavior not in {
            "normal",
            "no_explanation",
            "truncated",
            "malformed",
            "malformed_text",
            "timeout",
            "error",
        }:
            raise ContractValidationError(
                "GazeXplainAdapterConfig.fake_behavior",
                "must be normal, no_explanation, truncated, malformed, malformed_text, timeout, or error",
            )

    @classmethod
    def from_run_config(cls, config: RunConfig) -> "GazeXplainAdapterConfig":
        if config.model_id != GAZEXPLAIN_MODEL_ID:
            raise ContractValidationError(
                "RunConfig.model.id", f"expected '{GAZEXPLAIN_MODEL_ID}'"
            )
        if config.upstream_commit != GAZEXPLAIN_UPSTREAM_COMMIT:
            raise ContractValidationError(
                "RunConfig.model.upstream_commit",
                f"must equal pinned commit {GAZEXPLAIN_UPSTREAM_COMMIT}",
            )
        if config.checkpoint_path is None:
            raise ContractValidationError(
                "RunConfig.model.checkpoint_path", "is required"
            )
        raw = config.model_options.get(GAZEXPLAIN_MODEL_ID)
        if not isinstance(raw, Mapping):
            raise ContractValidationError(
                "RunConfig.model.options.gazexplain", "must be an object"
            )
        unknown = set(raw) - _OPTION_NAMES
        if unknown:
            name = sorted(unknown)[0]
            raise ContractValidationError(
                f"RunConfig.model.options.gazexplain.{name}",
                "unknown GazeXplain option",
            )
        return cls(
            variant=config.model_variant,
            upstream_root=_required_path(raw, "upstream_root"),
            upstream_commit=config.upstream_commit,
            checkpoint_path=Path(config.checkpoint_path).expanduser().resolve(),
            checkpoint_fingerprint=config.checkpoint_fingerprint,
            hparams_path=_required_path(raw, "hparams_path"),
            hparams_fingerprint=require_string(
                raw.get("hparams_fingerprint"),
                "RunConfig.model.options.gazexplain.hparams_fingerprint",
            ),
            device=config.device,
            output_root=Path(config.output_root).expanduser().resolve(),
            launcher=_launcher(raw.get("launcher")),
            model_assets_root=require_string(
                raw.get(
                    "model_assets_root", GAZEXPLAIN_DEFAULT_MODEL_ASSETS_ROOT
                ),
                "RunConfig.model.options.gazexplain.model_assets_root",
            ),
            model_max_fixations=_integer(
                raw.get(
                    "model_max_fixations", GAZEXPLAIN_MODEL_MAX_FIXATIONS
                ),
                "model_max_fixations",
            ),
            max_generation_length=_integer(
                raw.get(
                    "max_generation_length",
                    GAZEXPLAIN_DEFAULT_MAX_GENERATION_LENGTH,
                ),
                "max_generation_length",
            ),
            num_explanation_beams=_integer(
                raw.get(
                    "num_explanation_beams",
                    GAZEXPLAIN_DEFAULT_NUM_EXPLANATION_BEAMS,
                ),
                "num_explanation_beams",
            ),
            worker_timeout_s=_positive(
                raw.get("worker_timeout_s", 1800), "worker_timeout_s"
            ),
            startup_timeout_s=_positive(
                raw.get("startup_timeout_s", 600), "startup_timeout_s"
            ),
            native_artifact_policy=require_string(
                raw.get(
                    "native_artifact_policy", GAZEXPLAIN_NATIVE_ARTIFACT_POLICY
                ),
                "RunConfig.model.options.gazexplain.native_artifact_policy",
            ),
        )

    @classmethod
    def for_fake(
        cls,
        *,
        upstream_root: str | Path,
        checkpoint_path: str | Path,
        hparams_path: str | Path,
        output_root: str | Path,
        worker_timeout_s: float = 5.0,
        fake_behavior: str = "normal",
    ) -> "GazeXplainAdapterConfig":
        hparams = Path(hparams_path).resolve()
        fingerprint = _sha256_file(hparams) if hparams.is_file() else "0" * 64
        return cls(
            variant=GAZEXPLAIN_JOINT_VARIANT,
            upstream_root=Path(upstream_root).resolve(),
            upstream_commit=GAZEXPLAIN_UPSTREAM_COMMIT,
            checkpoint_path=Path(checkpoint_path).resolve(),
            checkpoint_fingerprint=None,
            hparams_path=hparams,
            hparams_fingerprint=f"sha256:{fingerprint}",
            device="cpu",
            output_root=Path(output_root).resolve(),
            launcher=(sys.executable,),
            worker_timeout_s=worker_timeout_s,
            startup_timeout_s=5.0,
            worker_module="activevision_workbench.adapters.gazexplain.fake_worker",
            fake_worker=True,
            fake_behavior=fake_behavior,
        )


def adapter_option_names() -> frozenset[str]:
    return _OPTION_NAMES


def _required_path(options: Mapping[object, object], name: str) -> Path:
    value = require_string(
        options.get(name), f"RunConfig.model.options.gazexplain.{name}"
    )
    return Path(value).expanduser().resolve()


def _launcher(value: object) -> tuple[str, ...]:
    field = "RunConfig.model.options.gazexplain.launcher"
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ContractValidationError(field, "must be a non-empty array of strings")
    result = tuple(
        require_string(item, f"{field}[{index}]")
        for index, item in enumerate(value)
    )
    if not result:
        raise ContractValidationError(field, "must not be empty")
    return result


def _integer(value: object, name: str) -> int:
    if type(value) is not int:
        raise ContractValidationError(
            f"RunConfig.model.options.gazexplain.{name}", "must be an integer"
        )
    return value


def _positive(value: object, name: str) -> float:
    if type(value) not in (int, float) or value <= 0:
        raise ContractValidationError(
            f"RunConfig.model.options.gazexplain.{name}",
            "must be a positive number",
        )
    return float(value)


def _fingerprint(value: str, field: str) -> None:
    fingerprint = require_string(value, field).removeprefix("sha256:")
    if len(fingerprint) != 64 or any(
        character not in "0123456789abcdefABCDEF" for character in fingerprint
    ):
        raise ContractValidationError(field, "must be a SHA-256 fingerprint")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
