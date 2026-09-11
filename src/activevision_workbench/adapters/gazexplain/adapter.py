
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import time
import uuid
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import IO, Any

from activevision_workbench.adapters.base import ModelAdapter
from activevision_workbench.adapters.gazexplain.config import (
    GAZEXPLAIN_BLIP_REVISION,
    GAZEXPLAIN_FEATURE_INPUT_HEIGHT,
    GAZEXPLAIN_FEATURE_INPUT_WIDTH,
    GAZEXPLAIN_JOINT_CHECKPOINT_SHA256,
    GAZEXPLAIN_JOINT_HPARAMS_SHA256,
    GAZEXPLAIN_MAP_HEIGHT,
    GAZEXPLAIN_MAP_WIDTH,
    GAZEXPLAIN_MASKRCNN_COCO_SHA256,
    GAZEXPLAIN_MODEL_ID,
    GAZEXPLAIN_MODEL_INPUT_HEIGHT,
    GAZEXPLAIN_MODEL_INPUT_WIDTH,
    GAZEXPLAIN_NATIVE_ARTIFACT_POLICY,
    GAZEXPLAIN_ROBERTA_REVISION,
    GazeXplainAdapterConfig,
    adapter_option_names,
)
from activevision_workbench.adapters.gazexplain.protocol import (
    PROTOCOL_VERSION,
    atomic_write_json,
    read_json,
    sha256_file,
    validate_response,
    worker_error,
)
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
from activevision_workbench.datasets.coordinates import source_fixation_from_pixel
from activevision_workbench.errors import (
    AdapterError,
    AdapterLifecycleError,
    AdapterOutputError,
    ContractValidationError,
)
from activevision_workbench.registry import AdapterDescriptor, AdapterRegistry

GAZEXPLAIN_ADAPTER_VERSION = "0.1.0"
GAZEXPLAIN_TASK_CONVERSION_VERSION = "exact-pass-through-v1"
GAZEXPLAIN_CAPABILITIES = CapabilitySet(
    frozenset(
        {
            Capability.PRODUCES_SCANPATHS,
            Capability.STOCHASTIC_MULTI_SAMPLE,
            Capability.INSTRUCTION_CONDITIONED,
            Capability.FIXATION_DURATION_OUTPUT,
            Capability.TEXT_EXPLANATION_OUTPUT,
        }
    )
)


class GazeXplainWorkerError(AdapterError):
    pass


class GazeXplainAdapter(ModelAdapter):

    def __init__(self, config: GazeXplainAdapterConfig) -> None:
        super().__init__()
        self.config = config
        self._identity = _identity(config)
        self._checkpoint_sha256: str | None = None
        self._hparams_sha256: str | None = None
        self._config_sha256: str | None = None
        self._process: subprocess.Popen[str] | None = None
        self._worker_stdin: IO[str] | None = None
        self._worker_run_id: str | None = None
        self._native_root: Path | None = None
        self._stdout_log: IO[str] | None = None
        self._stderr_log: IO[str] | None = None

    @property
    def identity(self) -> ModelIdentity:
        return self._identity

    @property
    def capabilities(self) -> CapabilitySet:
        return GAZEXPLAIN_CAPABILITIES

    @property
    def requirements(self) -> RequestRequirements:
        return RequestRequirements(instruction=True)

    def _validate_request(self, request: InferenceRequest) -> None:
        if request.task_text is None or not request.task_text:
            raise ContractValidationError(
                "InferenceRequest.task_text",
                "is required and must be non-empty for GazeXplain",
            )
        try:
            request.task_text.encode("utf-8")
        except UnicodeEncodeError as error:
            raise ContractValidationError(
                "InferenceRequest.task_text", "must contain valid Unicode"
            ) from error
        if request.image_width is None or request.image_height is None:
            raise ContractValidationError(
                "InferenceRequest.image_width", "image dimensions are required"
            )
        if request.observer_id is not None or request.observer_metadata:
            raise ContractValidationError(
                "InferenceRequest.observer_id",
                "the verified GazeXplain model does not consume observer identity or attributes",
            )
        if request.time_horizon_s is not None:
            raise ContractValidationError(
                "InferenceRequest.time_horizon_s",
                "GazeXplain stops by EOS or fixation count, not a time horizon",
            )
        if request.max_fixations is not None and (
            request.max_fixations > self.config.model_max_fixations
        ):
            raise ContractValidationError(
                "InferenceRequest.max_fixations",
                f"must not exceed {self.config.model_max_fixations}",
            )
        if request.base_seed + request.num_samples - 1 > 2**32 - 1:
            raise ContractValidationError(
                "InferenceRequest.base_seed",
                "base_seed + num_samples - 1 must fit a uint32 seed",
            )
        options = request.model_options.get(GAZEXPLAIN_MODEL_ID, {})
        if isinstance(options, Mapping):
            unknown = set(options) - adapter_option_names()
            if unknown:
                raise ContractValidationError(
                    f"InferenceRequest.model_options.gazexplain.{sorted(unknown)[0]}",
                    "unknown GazeXplain option; unsupported prompt fields are not ignored",
                )

    def _prepare(self) -> None:
        _require_directory(self.config.upstream_root, "upstream_root")
        _require_file(self.config.checkpoint_path, "checkpoint_path")
        _require_file(self.config.hparams_path, "hparams_path")
        launcher = self.config.launcher[0]
        if os.sep in launcher:
            if not Path(launcher).is_file():
                raise GazeXplainWorkerError(
                    f"GazeXplain launcher does not exist: {launcher}"
                )
        elif shutil.which(launcher) is None:
            raise GazeXplainWorkerError(
                f"GazeXplain launcher is not on PATH: {launcher}"
            )
        if not self.config.fake_worker:
            actual_commit = _upstream_commit(self.config.upstream_root)
            if actual_commit != self.config.upstream_commit:
                raise GazeXplainWorkerError(
                    "GazeXplain checkout commit mismatch: expected "
                    f"{self.config.upstream_commit}, found {actual_commit}"
                )
        self._checkpoint_sha256 = sha256_file(self.config.checkpoint_path)
        self._hparams_sha256 = sha256_file(self.config.hparams_path)
        if not self.config.fake_worker:
            if self._checkpoint_sha256 != GAZEXPLAIN_JOINT_CHECKPOINT_SHA256:
                raise GazeXplainWorkerError(
                    "GazeXplain checkpoint SHA-256 does not match the pinned joint model"
                )
            if self._hparams_sha256 != GAZEXPLAIN_JOINT_HPARAMS_SHA256:
                raise GazeXplainWorkerError(
                    "GazeXplain hparams SHA-256 does not match the pinned joint model"
                )
            _validate_hparams(self.config.hparams_path)
        configured_checkpoint = self.config.checkpoint_fingerprint
        if configured_checkpoint is not None and (
            configured_checkpoint.removeprefix("sha256:").lower()
            != self._checkpoint_sha256
        ):
            raise GazeXplainWorkerError(
                "checkpoint SHA-256 does not match configured fingerprint"
            )
        if (
            self.config.hparams_fingerprint.removeprefix("sha256:").lower()
            != self._hparams_sha256
        ):
            raise GazeXplainWorkerError(
                "hparams SHA-256 does not match configured fingerprint"
            )
        payload = {
            "protocol_version": PROTOCOL_VERSION,
            "adapter_version": GAZEXPLAIN_ADAPTER_VERSION,
            "variant": self.config.variant,
            "upstream_commit": self.config.upstream_commit,
            "checkpoint_sha256": self._checkpoint_sha256,
            "hparams_sha256": self._hparams_sha256,
            "device": self.config.device,
            "model_assets_root": self.config.model_assets_root,
            "model_max_fixations": self.config.model_max_fixations,
            "max_generation_length": self.config.max_generation_length,
            "num_explanation_beams": self.config.num_explanation_beams,
            "task_conversion_version": GAZEXPLAIN_TASK_CONVERSION_VERSION,
            "roberta_revision": GAZEXPLAIN_ROBERTA_REVISION,
            "blip_revision": GAZEXPLAIN_BLIP_REVISION,
            "feature_backbone_sha256": GAZEXPLAIN_MASKRCNN_COCO_SHA256,
            "native_artifact_policy": self.config.native_artifact_policy,
        }
        self._config_sha256 = hashlib.sha256(
            json.dumps(payload, separators=(",", ":"), sort_keys=True).encode(
                "utf-8"
            )
        ).hexdigest()

    def _predict(
        self, request: InferenceRequest, *, run_id: str
    ) -> Iterable[PredictionRecord]:
        self._ensure_worker(run_id)
        assert self._native_root is not None
        assert request.task_text is not None
        assert request.image_width is not None
        assert request.image_height is not None
        request_key = hashlib.sha256(request.request_id.encode("utf-8")).hexdigest()[:20]
        request_root = self._native_root / "requests" / request_key
        request_root.mkdir(parents=True, exist_ok=True)
        command_id = uuid.uuid4().hex
        native_directory = request_root / f"attempt-{command_id}"
        request_path = request_root / f"request-{command_id}.json"
        response_path = request_root / f"response-{command_id}.json"
        effective_max = request.max_fixations or self.config.model_max_fixations
        sample_specs = [
            {
                "sample_id": _sample_id(request.request_id, index),
                "sample_index": index,
                "seed": request.base_seed + index,
            }
            for index in range(request.num_samples)
        ]
        atomic_write_json(
            request_path,
            {
                "schema_version": PROTOCOL_VERSION,
                "request_id": request.request_id,
                "image_ref": request.image_ref,
                "image_width": request.image_width,
                "image_height": request.image_height,
                "resolved_task_text": request.task_text,
                "target_description": request.target_description,
                "sample_specs": sample_specs,
                "max_fixations": effective_max,
                "native_output_dir": str(native_directory.resolve()),
            },
        )
        self._send_command(request_path, response_path)
        response = read_json(response_path, label="GazeXplain worker response")
        error = worker_error(response)
        if error is not None:
            raise GazeXplainWorkerError(f"GazeXplain worker failed: {error}")
        samples, metadata = validate_response(
            response,
            request_id=request.request_id,
            sample_count=request.num_samples,
        )
        self._validate_metadata(request, metadata)
        return tuple(
            self._record(
                request,
                run_id=run_id,
                sample=sample,
                metadata=metadata,
                native_directory=native_directory,
                effective_max=effective_max,
            )
            for sample in samples
        )

    def _validate_metadata(
        self, request: InferenceRequest, metadata: dict[str, Any]
    ) -> None:
        expected = {
            "checkpoint_sha256": self._checkpoint_sha256,
            "hparams_sha256": self._hparams_sha256,
            "config_sha256": self._config_sha256,
            "upstream_commit": self.config.upstream_commit,
            "variant": self.config.variant,
            "image_width": request.image_width,
            "image_height": request.image_height,
            "model_input_width": GAZEXPLAIN_MODEL_INPUT_WIDTH,
            "model_input_height": GAZEXPLAIN_MODEL_INPUT_HEIGHT,
            "feature_input_width": GAZEXPLAIN_FEATURE_INPUT_WIDTH,
            "feature_input_height": GAZEXPLAIN_FEATURE_INPUT_HEIGHT,
            "map_width": GAZEXPLAIN_MAP_WIDTH,
            "map_height": GAZEXPLAIN_MAP_HEIGHT,
            "resolved_task_text": request.task_text,
            "task_input_truncated": False,
            "task_tokenizer": {
                "model_id": "FacebookAI/roberta-base",
                "revision": GAZEXPLAIN_ROBERTA_REVISION,
            },
            "explanation_tokenizer": {
                "model_id": "Salesforce/blip-image-captioning-base",
                "revision": GAZEXPLAIN_BLIP_REVISION,
            },
            "feature_backbone": {
                "model_id": "MaskRCNN_ResNet50_FPN_Weights.COCO_V1",
                "sha256": GAZEXPLAIN_MASKRCNN_COCO_SHA256,
            },
            "decoding": {
                "max_generation_length": self.config.max_generation_length,
                "num_explanation_beams": self.config.num_explanation_beams,
                "early_stopping": True,
            },
        }
        for name, value in expected.items():
            if metadata[name] != value:
                raise AdapterOutputError(
                    f"GazeXplain worker metadata mismatch for {name}"
                )
        if metadata["image_cache_scope"] != (
            "decoded-and-preprocessed-image-features-only; no task-conditioned outputs"
        ):
            raise AdapterOutputError(
                "GazeXplain worker reported an unsafe image cache scope"
            )

    def _record(
        self,
        request: InferenceRequest,
        *,
        run_id: str,
        sample: dict[str, Any],
        metadata: dict[str, Any],
        native_directory: Path,
        effective_max: int,
    ) -> PredictionRecord:
        index = sample["sample_index"]
        if sample["sample_id"] != _sample_id(request.request_id, index):
            raise AdapterOutputError("GazeXplain worker returned an unstable sample ID")
        if sample["seed"] != request.base_seed + index:
            raise AdapterOutputError("GazeXplain worker returned the wrong seed")
        assert request.image_width is not None
        assert request.image_height is not None
        fixations: list[FixationEvent] = []
        warnings = list(sample["warnings"])
        alignments = sample["explanation"]["per_fixation"]
        for sequence_index, event in enumerate(sample["events"]):
            if sequence_index >= effective_max:
                raise AdapterOutputError(
                    "GazeXplain exceeded the requested fixation cap"
                )
            duration_ms = _finite(event["duration_ms"], "duration_ms")
            probability = _finite(
                event["action_probability"], "action_probability"
            )
            if duration_ms < 0:
                raise AdapterOutputError("GazeXplain returned a negative duration")
            if not 0 <= probability <= 1:
                raise AdapterOutputError(
                    "GazeXplain action probability is outside [0, 1]"
                )
            x_model = _finite(event["x_model_px"], "x_model_px")
            y_model = _finite(event["y_model_px"], "y_model_px")
            alignment = alignments[sequence_index]
            if (
                event["explanation_text"] != alignment["text"]
                or event["explanation_token_ids"] != alignment["token_ids"]
                or event["explanation_truncated"] != alignment["truncated"]
            ):
                raise AdapterOutputError(
                    "GazeXplain per-fixation explanation alignment is inconsistent"
                )
            x_px = x_model * request.image_width / GAZEXPLAIN_MODEL_INPUT_WIDTH
            y_px = y_model * request.image_height / GAZEXPLAIN_MODEL_INPUT_HEIGHT
            fixation, warning = source_fixation_from_pixel(
                x_px=x_px,
                y_px=y_px,
                image_width=request.image_width,
                image_height=request.image_height,
                sequence_index=sequence_index,
                duration_s=duration_ms * 0.001,
                native_timing={
                    "duration": duration_ms,
                    "unit": "milliseconds",
                    "scale_to_seconds": 0.001,
                    "source": "upstream log-normal duration sample",
                },
                native_metadata={
                    "gazexplain": {
                        "action_index": event["action_index"],
                        "action_probability": probability,
                        "x_model_px": x_model,
                        "y_model_px": y_model,
                        "model_canvas": [
                            GAZEXPLAIN_MODEL_INPUT_WIDTH,
                            GAZEXPLAIN_MODEL_INPUT_HEIGHT,
                        ],
                        "coordinate_transform": (
                            "model_coordinate * original_dimension / model_dimension"
                        ),
                        "explanation_text": event["explanation_text"],
                        "explanation_token_ids": event[
                            "explanation_token_ids"
                        ],
                        "explanation_truncated": event[
                            "explanation_truncated"
                        ],
                    }
                },
                clip_out_of_bounds=False,
            )
            fixations.append(fixation)
            if warning is not None:
                warnings.append(warning)
        explanation = sample["explanation"]
        if explanation["text"] is None:
            warnings.append(
                "GazeXplain produced no explanation text; no text was fabricated"
            )
        if explanation["truncated"] and not any(
            "max_generation_length" in warning for warning in warnings
        ):
            warnings.append(
                "GazeXplain explanation generation reached max_generation_length"
            )
        stopping = {
            "model_stop": StoppingReason.MODEL_STOP,
            "max_fixations": StoppingReason.MAX_FIXATIONS,
        }[sample["stopping_reason"]]
        if stopping is StoppingReason.MAX_FIXATIONS and len(fixations) != effective_max:
            raise AdapterOutputError("GazeXplain reported max_fixations early")
        native_path = Path(sample["native_artifact"])
        try:
            native_path.resolve().relative_to(native_directory.resolve())
        except (OSError, ValueError) as error:
            raise AdapterOutputError(
                "GazeXplain native artifact escaped its directory"
            ) from error
        if not native_path.is_file():
            raise AdapterOutputError("GazeXplain native artifact is missing")
        return PredictionRecord(
            run_id=run_id,
            request_id=request.request_id,
            model=self.identity,
            dataset_id=request.dataset_id,
            dataset_split=request.dataset_split,
            item_id=request.item_id,
            sample_id=sample["sample_id"],
            sample_index=index,
            seed=sample["seed"],
            observer_id=None,
            task_text=request.task_text,
            target_description=request.target_description,
            fixations=tuple(fixations),
            explanation_text=explanation["text"],
            native_artifacts=(
                ArtifactReference(
                    uri=native_path.resolve().as_uri(),
                    kind="model_native_prediction",
                    media_type="application/json",
                    checksum=f"sha256:{sha256_file(native_path)}",
                    metadata={
                        "sample_index": index,
                        "contains_raw_token_ids": True,
                        "contains_token_scores": False,
                        "contains_per_fixation_text_alignment": True,
                    },
                ),
            ),
            warnings=tuple(dict.fromkeys(warnings)),
            native_metadata={
                "gazexplain": {
                    "resolved_task_text": request.task_text,
                    "task_conversion_version": GAZEXPLAIN_TASK_CONVERSION_VERSION,
                    "target_description": request.target_description,
                    "target_description_model_input": False,
                    "task_input_token_count": metadata[
                        "task_input_token_count"
                    ],
                    "task_input_truncated": metadata["task_input_truncated"],
                    "task_tokenizer": metadata["task_tokenizer"],
                    "explanation_tokenizer": metadata[
                        "explanation_tokenizer"
                    ],
                    "feature_backbone": metadata["feature_backbone"],
                    "decoding": metadata["decoding"],
                    "raw_token_ids": explanation["raw_token_ids"],
                    "raw_token_ids_all_steps": explanation[
                        "raw_token_ids_all_steps"
                    ],
                    "token_scores": None,
                    "per_fixation_text_alignment": explanation[
                        "per_fixation"
                    ],
                    "explanation_truncated": explanation["truncated"],
                    "checkpoint_sha256": metadata["checkpoint_sha256"],
                    "hparams_sha256": metadata["hparams_sha256"],
                    "adapter_config_sha256": metadata["config_sha256"],
                    "upstream_commit": metadata["upstream_commit"],
                    "image_cache_scope": metadata["image_cache_scope"],
                    "effective_max_fixations": effective_max,
                    "native_sequence_length": len(
                        sample["native_sequence"]["selected_actions"]
                    ),
                    "nondeterminism": (
                        "per-sample Python, NumPy, Torch, and CUDA seeds are set; "
                        "CUDA kernels can remain hardware/version dependent"
                    ),
                }
            },
            stopping_reason=stopping,
        )

    def _ensure_worker(self, run_id: str) -> None:
        if self._process is not None and self._process.poll() is None:
            if self._worker_run_id == run_id:
                return
            self._stop_worker()
        self._worker_run_id = run_id
        self._native_root = self.config.output_root / run_id / "native" / "gazexplain"
        self._native_root.mkdir(parents=True, exist_ok=True)
        session = uuid.uuid4().hex
        config_path = self._native_root / f"worker-config-{session}.json"
        ready_path = self._native_root / f"worker-ready-{session}.json"
        assert self._checkpoint_sha256 is not None
        assert self._hparams_sha256 is not None
        assert self._config_sha256 is not None
        atomic_write_json(
            config_path,
            {
                "schema_version": PROTOCOL_VERSION,
                "variant": self.config.variant,
                "upstream_root": str(self.config.upstream_root),
                "upstream_commit": self.config.upstream_commit,
                "checkpoint_path": str(self.config.checkpoint_path),
                "checkpoint_sha256": self._checkpoint_sha256,
                "hparams_path": str(self.config.hparams_path),
                "hparams_sha256": self._hparams_sha256,
                "device": self.config.device,
                "model_assets_root": self.config.model_assets_root,
                "model_max_fixations": self.config.model_max_fixations,
                "max_generation_length": self.config.max_generation_length,
                "num_explanation_beams": self.config.num_explanation_beams,
                "model_input_width": GAZEXPLAIN_MODEL_INPUT_WIDTH,
                "model_input_height": GAZEXPLAIN_MODEL_INPUT_HEIGHT,
                "feature_input_width": GAZEXPLAIN_FEATURE_INPUT_WIDTH,
                "feature_input_height": GAZEXPLAIN_FEATURE_INPUT_HEIGHT,
                "map_width": GAZEXPLAIN_MAP_WIDTH,
                "map_height": GAZEXPLAIN_MAP_HEIGHT,
                "roberta_revision": GAZEXPLAIN_ROBERTA_REVISION,
                "blip_revision": GAZEXPLAIN_BLIP_REVISION,
                "feature_backbone_sha256": GAZEXPLAIN_MASKRCNN_COCO_SHA256,
                "native_artifact_policy": GAZEXPLAIN_NATIVE_ARTIFACT_POLICY,
                "config_sha256": self._config_sha256,
                "fake_behavior": self.config.fake_behavior,
            },
        )
        logs = self.config.output_root / run_id / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        self._stdout_log = (logs / "gazexplain-worker.stdout.log").open(
            "a", encoding="utf-8"
        )
        self._stderr_log = (logs / "gazexplain-worker.stderr.log").open(
            "a", encoding="utf-8"
        )
        command = (
            *self.config.launcher,
            "-m",
            self.config.worker_module,
            "--serve",
            "--config",
            str(config_path),
            "--ready",
            str(ready_path),
        )
        environment = os.environ.copy()
        source_root = str(Path(__file__).resolve().parents[3])
        current = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            source_root if not current else f"{source_root}{os.pathsep}{current}"
        )
        environment["PYTHONUNBUFFERED"] = "1"
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["HF_HUB_OFFLINE"] = "1"
        environment["TRANSFORMERS_OFFLINE"] = "1"
        environment["TOKENIZERS_PARALLELISM"] = "false"
        try:
            self._process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=self._stdout_log,
                stderr=self._stderr_log,
                text=True,
                env=environment,
                start_new_session=True,
            )
        except OSError as error:
            self._stop_worker()
            raise GazeXplainWorkerError(
                f"cannot start GazeXplain worker: {error}"
            ) from error
        self._worker_stdin = self._process.stdin
        try:
            _wait_for_path(
                ready_path,
                process=self._process,
                timeout_s=self.config.startup_timeout_s,
                description="GazeXplain worker startup",
            )
            ready = read_json(ready_path, label="GazeXplain worker readiness")
            if not isinstance(ready, dict) or ready.get("status") != "ready":
                raise GazeXplainWorkerError("worker readiness document is invalid")
        except Exception:
            self._stop_worker()
            raise

    def _send_command(self, request_path: Path, response_path: Path) -> None:
        process = self._process
        stream = self._worker_stdin
        if process is None or stream is None or process.poll() is not None:
            raise AdapterLifecycleError("GazeXplain worker is not running")
        try:
            stream.write(
                json.dumps(
                    {
                        "request_path": str(request_path),
                        "response_path": str(response_path),
                    },
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n"
            )
            stream.flush()
            _wait_for_path(
                response_path,
                process=process,
                timeout_s=self.config.worker_timeout_s,
                description="GazeXplain inference",
            )
        except Exception:
            self._stop_worker()
            raise

    def _close(self) -> None:
        self._stop_worker()

    def _stop_worker(self) -> None:
        process = self._process
        self._process = None
        self._worker_stdin = None
        if process is not None:
            try:
                if process.stdin is not None:
                    process.stdin.close()
            except OSError:
                pass
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except OSError:
                    pass
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
        for name in ("_stdout_log", "_stderr_log"):
            handle = getattr(self, name)
            if handle is not None:
                handle.close()
                setattr(self, name, None)


def gazexplain_descriptor(config: GazeXplainAdapterConfig) -> AdapterDescriptor:
    return AdapterDescriptor(
        model_id=GAZEXPLAIN_MODEL_ID,
        identity=_identity(config),
        capabilities=GAZEXPLAIN_CAPABILITIES,
        requirements=RequestRequirements(instruction=True),
        factory=lambda: GazeXplainAdapter(config),
    )


def create_gazexplain_registry(config: GazeXplainAdapterConfig) -> AdapterRegistry:
    registry = AdapterRegistry(allowed_model_ids={GAZEXPLAIN_MODEL_ID})
    registry.register(gazexplain_descriptor(config))
    return registry


def _identity(config: GazeXplainAdapterConfig) -> ModelIdentity:
    checkpoint = (
        config.checkpoint_fingerprint or f"external:{config.checkpoint_path.name}"
    )
    return ModelIdentity(
        model_id=GAZEXPLAIN_MODEL_ID,
        model_variant=config.variant,
        adapter_version=GAZEXPLAIN_ADAPTER_VERSION,
        upstream_version=config.upstream_commit,
        checkpoint_id=checkpoint,
    )


def _sample_id(request_id: str, sample_index: int) -> str:
    return f"{request_id}:sample:{sample_index}"


def _finite(value: object, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise AdapterOutputError(f"GazeXplain {label} must be finite")
    return float(value)


def _require_file(path: Path, name: str) -> None:
    if not path.is_file():
        raise GazeXplainWorkerError(
            f"GazeXplain {name} is missing or not a file: {path}"
        )


def _require_directory(path: Path, name: str) -> None:
    if not path.is_dir():
        raise GazeXplainWorkerError(
            f"GazeXplain {name} is missing or not a directory: {path}"
        )


def _validate_hparams(path: Path) -> None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise GazeXplainWorkerError(f"cannot read GazeXplain hparams: {error}") from error
    if not isinstance(value, dict):
        raise GazeXplainWorkerError("GazeXplain hparams must be an object")
    expected = {
        "width": GAZEXPLAIN_MODEL_INPUT_WIDTH,
        "height": GAZEXPLAIN_MODEL_INPUT_HEIGHT,
        "im_w": GAZEXPLAIN_MAP_WIDTH,
        "im_h": GAZEXPLAIN_MAP_HEIGHT,
        "max_length": 16,
        "min_length": 1,
        "img_hidden_dim": 2048,
        "lm_hidden_dim": 768,
    }
    for name, expected_value in expected.items():
        if value.get(name) != expected_value:
            raise GazeXplainWorkerError(
                f"GazeXplain hparams field {name} is incompatible with the selected adapter"
            )


def _upstream_commit(root: Path) -> str:
    try:
        revision = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        status = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "status",
                "--porcelain",
                "--untracked-files=no",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise GazeXplainWorkerError(
            f"cannot verify GazeXplain checkout at {root}: {error}"
        ) from error
    if status.stdout.strip():
        raise GazeXplainWorkerError("GazeXplain checkout has tracked modifications")
    return revision.stdout.strip()


def _wait_for_path(
    path: Path,
    *,
    process: subprocess.Popen[str],
    timeout_s: float,
    description: str,
) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.is_file():
            return
        code = process.poll()
        if code is not None:
            raise GazeXplainWorkerError(
                f"{description} worker exited with code {code}; inspect logs/gazexplain-worker.stderr.log"
            )
        time.sleep(0.02)
    raise GazeXplainWorkerError(
        f"{description} timed out after {timeout_s:g}s; worker was terminated"
    )
