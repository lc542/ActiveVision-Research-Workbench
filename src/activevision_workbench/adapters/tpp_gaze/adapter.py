
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
from activevision_workbench.adapters.tpp_gaze.config import (
    TPP_GAZE_MODEL_ID,
    TPP_GAZE_TRANSFORMER_CHECKPOINT_SHA256,
    TPP_GAZE_TRANSFORMER_CONFIG_SHA256,
    TPP_GAZE_UPSTREAM_COMMIT,
    TPPGazeAdapterConfig,
    adapter_option_names,
)
from activevision_workbench.adapters.tpp_gaze.protocol import (
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

TPP_GAZE_ADAPTER_VERSION = "0.1.0"
TPP_GAZE_CAPABILITIES = CapabilitySet(
    frozenset(
        {
            Capability.PRODUCES_SCANPATHS,
            Capability.STOCHASTIC_MULTI_SAMPLE,
            Capability.CONTINUOUS_TIME_OUTPUT,
            Capability.FIXATION_DURATION_OUTPUT,
        }
    )
)


class TPPGazeWorkerError(AdapterError):
    pass


class TPPGazeAdapter(ModelAdapter):

    def __init__(self, config: TPPGazeAdapterConfig) -> None:
        super().__init__()
        self.config = config
        self._identity = _identity(config)
        self._process: subprocess.Popen[str] | None = None
        self._worker_run_id: str | None = None
        self._worker_stdin: IO[str] | None = None
        self._stdout_log: IO[str] | None = None
        self._stderr_log: IO[str] | None = None
        self._native_root: Path | None = None
        self._checkpoint_sha256: str | None = None
        self._model_config_sha256: str | None = None
        self._config_sha256: str | None = None

    @property
    def identity(self) -> ModelIdentity:
        return self._identity

    @property
    def capabilities(self) -> CapabilitySet:
        return TPP_GAZE_CAPABILITIES

    @property
    def requirements(self) -> RequestRequirements:
        return RequestRequirements()

    def _validate_request(self, request: InferenceRequest) -> None:
        if request.image_width is None or request.image_height is None:
            raise ContractValidationError(
                "InferenceRequest.image_width",
                "TPP-Gaze coordinate conversion requires image dimensions",
            )
        if request.image_width < 2 or request.image_height < 2:
            raise ContractValidationError(
                "InferenceRequest.image_width",
                "TPP-Gaze canonical coordinates require dimensions >= 2",
            )
        if request.time_horizon_s is None:
            raise ContractValidationError(
                "InferenceRequest.time_horizon_s",
                "is required because TPP-Gaze samples to a continuous-time horizon",
            )
        if request.max_fixations is not None and (
            request.max_fixations > self.config.safety_max_fixations
        ):
            raise ContractValidationError(
                "InferenceRequest.max_fixations",
                "must not exceed the configured TPP-Gaze safety_max_fixations "
                f"({self.config.safety_max_fixations})",
            )
        maximum_seed = request.base_seed + request.num_samples - 1
        if maximum_seed > (2**32 - 1):
            raise ContractValidationError(
                "InferenceRequest.base_seed",
                "base_seed + num_samples - 1 must fit the upstream NumPy uint32 seed",
            )
        options = request.model_options.get(TPP_GAZE_MODEL_ID, {})
        if isinstance(options, Mapping):
            unknown = set(options) - adapter_option_names()
            if unknown:
                name = sorted(unknown)[0]
                raise ContractValidationError(
                    f"InferenceRequest.model_options.tpp_gaze.{name}",
                    "unknown TPP-Gaze adapter option",
                )

    def _prepare(self) -> None:
        _require_directory(self.config.upstream_root, "upstream_root")
        _require_file(self.config.checkpoint_path, "checkpoint_path")
        _require_file(self.config.model_config_path, "model_config_path")
        launcher = self.config.launcher[0]
        if os.sep in launcher:
            if not Path(launcher).is_file():
                raise TPPGazeWorkerError(
                    f"TPP-Gaze launcher does not exist: {launcher}"
                )
        elif shutil.which(launcher) is None:
            raise TPPGazeWorkerError(
                f"TPP-Gaze launcher is not on PATH: {launcher}"
            )
        if not self.config.fake_worker:
            actual_commit = _upstream_commit(self.config.upstream_root)
            if actual_commit != self.config.upstream_commit:
                raise TPPGazeWorkerError(
                    "TPP-Gaze checkout commit mismatch: expected "
                    f"{self.config.upstream_commit}, found {actual_commit}"
                )
        self._checkpoint_sha256 = sha256_file(self.config.checkpoint_path)
        self._model_config_sha256 = sha256_file(self.config.model_config_path)
        if not self.config.fake_worker:
            if self._checkpoint_sha256 != TPP_GAZE_TRANSFORMER_CHECKPOINT_SHA256:
                raise TPPGazeWorkerError(
                    "TPP-Gaze transformer checkpoint SHA-256 does not match the "
                    "pinned model_transformer.pth"
                )
            if self._model_config_sha256 != TPP_GAZE_TRANSFORMER_CONFIG_SHA256:
                raise TPPGazeWorkerError(
                    "TPP-Gaze model config SHA-256 does not match the pinned "
                    "data/config.yaml"
                )
        configured = self.config.checkpoint_fingerprint
        if configured is not None:
            expected = configured.removeprefix("sha256:")
            if len(expected) == 64 and all(
                character in "0123456789abcdefABCDEF" for character in expected
            ):
                if self._checkpoint_sha256.lower() != expected.lower():
                    raise TPPGazeWorkerError(
                        "checkpoint SHA-256 does not match the configured fingerprint"
                    )
        fingerprint_payload = {
            "protocol_version": PROTOCOL_VERSION,
            "adapter_version": TPP_GAZE_ADAPTER_VERSION,
            "variant": self.config.variant,
            "upstream_commit": self.config.upstream_commit,
            "checkpoint_sha256": self._checkpoint_sha256,
            "model_config_sha256": self._model_config_sha256,
            "device": self.config.device,
            "temperature": self.config.temperature,
            "native_time_unit": self.config.native_time_unit,
            "temporal_scale_to_seconds": self.config.temporal_scale_to_seconds,
            "safety_max_fixations": self.config.safety_max_fixations,
            "native_artifact_policy": self.config.native_artifact_policy,
        }
        self._config_sha256 = hashlib.sha256(
            json.dumps(
                fingerprint_payload, separators=(",", ":"), sort_keys=True
            ).encode("utf-8")
        ).hexdigest()

    def _predict(
        self,
        request: InferenceRequest,
        *,
        run_id: str,
    ) -> Iterable[PredictionRecord]:
        self._ensure_worker(run_id)
        assert self._native_root is not None
        request_key = hashlib.sha256(
            request.request_id.encode("utf-8")
        ).hexdigest()[:20]
        request_directory = self._native_root / "requests" / request_key
        request_directory.mkdir(parents=True, exist_ok=True)
        command_id = uuid.uuid4().hex
        native_directory = request_directory / f"attempt-{command_id}"
        request_path = request_directory / f"request-{command_id}.json"
        response_path = request_directory / f"response-{command_id}.json"
        sample_specs = [
            {
                "sample_id": _sample_id(request.request_id, index),
                "sample_index": index,
                "seed": request.base_seed + index,
            }
            for index in range(request.num_samples)
        ]
        assert request.image_width is not None
        assert request.image_height is not None
        assert request.time_horizon_s is not None
        effective_max = request.max_fixations or self.config.safety_max_fixations
        atomic_write_json(
            request_path,
            {
                "schema_version": PROTOCOL_VERSION,
                "request_id": request.request_id,
                "image_ref": request.image_ref,
                "image_width": request.image_width,
                "image_height": request.image_height,
                "sample_specs": sample_specs,
                "time_horizon_s": request.time_horizon_s,
                "max_fixations": effective_max,
                "native_output_dir": str(native_directory.resolve()),
            },
        )
        self._send_command(request_path, response_path)
        response = read_json(response_path, label="TPP-Gaze worker response")
        reported_error = worker_error(response)
        if reported_error is not None:
            raise TPPGazeWorkerError(
                f"TPP-Gaze worker failed: {reported_error}"
            )
        samples, metadata = validate_response(
            response,
            request_id=request.request_id,
            sample_count=request.num_samples,
        )
        self._validate_metadata(request, metadata)
        return tuple(
            self._prediction_record(
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
        if metadata["image_width"] != request.image_width or metadata[
            "image_height"
        ] != request.image_height:
            raise AdapterOutputError(
                "TPP-Gaze worker image dimensions do not match the request"
            )
        if metadata["upstream_commit"] != self.config.upstream_commit:
            raise AdapterOutputError(
                "TPP-Gaze worker upstream commit does not match configuration"
            )
        if metadata["checkpoint_sha256"] != self._checkpoint_sha256:
            raise AdapterOutputError("TPP-Gaze checkpoint fingerprint changed")
        if metadata["model_config_sha256"] != self._model_config_sha256:
            raise AdapterOutputError("TPP-Gaze model-config fingerprint changed")
        if metadata["config_sha256"] != self._config_sha256:
            raise AdapterOutputError("TPP-Gaze adapter-config fingerprint changed")
        if metadata["context_type"] != self.config.variant:
            raise AdapterOutputError(
                "TPP-Gaze context type does not match the configured variant"
            )
        expected_source = (
            "model_config"
            if self.config.temperature is None
            else "adapter_override"
        )
        if metadata["temperature_source"] != expected_source:
            raise AdapterOutputError(
                "TPP-Gaze worker temperature source does not match configuration"
            )
        if self.config.temperature is not None and not math.isclose(
            metadata["temperature"], self.config.temperature, rel_tol=0.0, abs_tol=0.0
        ):
            raise AdapterOutputError(
                "TPP-Gaze worker temperature does not match the configured override"
            )

    def _prediction_record(
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
            raise AdapterOutputError(
                f"TPP-Gaze sample {index} has an unstable sample_id"
            )
        if sample["seed"] != request.base_seed + index:
            raise AdapterOutputError(
                f"TPP-Gaze sample {index} reported the wrong seed"
            )
        assert request.image_width is not None
        assert request.image_height is not None
        assert request.time_horizon_s is not None
        warnings = list(sample["warnings"])
        fixations: list[FixationEvent] = []
        previous_arrival_s: float | None = None
        saw_excluded = False
        for native_index, event in enumerate(sample["events"]):
            included = event["included"]
            if saw_excluded and included:
                raise AdapterOutputError(
                    "TPP-Gaze worker included an event after a terminal event"
                )
            if not included:
                saw_excluded = True
            inter_s = _converted_time(
                event["inter_event_time_native"],
                scale=self.config.temporal_scale_to_seconds,
                label=f"sample {index} event {native_index} inter-event time",
            )
            arrival_s = _converted_time(
                event["arrival_time_native"],
                scale=self.config.temporal_scale_to_seconds,
                label=f"sample {index} event {native_index} arrival time",
            )
            if inter_s is not None and inter_s < 0:
                raise AdapterOutputError(
                    f"TPP-Gaze sample {index} event {native_index} has a "
                    "negative inter-event time"
                )
            if arrival_s is not None and arrival_s < 0:
                raise AdapterOutputError(
                    f"TPP-Gaze sample {index} event {native_index} has a "
                    "negative arrival time"
                )
            if arrival_s is not None:
                if previous_arrival_s is not None and arrival_s <= previous_arrival_s:
                    raise AdapterOutputError(
                        f"TPP-Gaze sample {index} arrival times are not strictly "
                        "increasing"
                    )
                previous_arrival_s = arrival_s
            _validate_model_time(
                inter_s,
                event["model_inter_event_time_s"],
                f"sample {index} event {native_index} inter-event time",
            )
            _validate_model_time(
                arrival_s,
                event["model_arrival_time_s"],
                f"sample {index} event {native_index} arrival time",
            )
            if included and arrival_s is not None and (
                arrival_s > request.time_horizon_s + 1e-9
            ):
                raise AdapterOutputError(
                    f"TPP-Gaze sample {index} silently exceeded the requested horizon"
                )
            if not included:
                continue
            if len(fixations) >= effective_max:
                raise AdapterOutputError(
                    f"TPP-Gaze sample {index} exceeded the maximum fixation cap"
                )
            fixation, coordinate_warning = source_fixation_from_pixel(
                x_px=float(event["x_px"]),
                y_px=float(event["y_px"]),
                image_width=request.image_width,
                image_height=request.image_height,
                sequence_index=len(fixations),
                timestamp_s=arrival_s,
                duration_s=inter_s,
                native_timing={
                    "inter_event_time": {
                        "value": event["inter_event_time_native"],
                        "unit": self.config.native_time_unit,
                        "scale_to_seconds": self.config.temporal_scale_to_seconds,
                        "meaning": "fixation_duration",
                        "canonical_field": (
                            "duration_s" if inter_s is not None else None
                        ),
                    },
                    "arrival_time": {
                        "value": event["arrival_time_native"],
                        "unit": self.config.native_time_unit,
                        "scale_to_seconds": self.config.temporal_scale_to_seconds,
                        "meaning": "absolute_event_arrival_time",
                        "canonical_field": (
                            "timestamp_s" if arrival_s is not None else None
                        ),
                    },
                },
                native_metadata={
                    "tpp_gaze": {
                        "native_event_index": native_index,
                        "mark_x": event["mark_x_native"],
                        "mark_y": event["mark_y_native"],
                        "model_inter_event_time_s": event[
                            "model_inter_event_time_s"
                        ],
                        "model_arrival_time_s": event["model_arrival_time_s"],
                        "pixel_transform": (
                            "clip(((mark+1)/2)*(dimension-1),0,dimension-1)"
                        ),
                    }
                },
                clip_out_of_bounds=False,
            )
            fixations.append(fixation)
            if coordinate_warning is not None:
                warnings.append(
                    f"fixation {len(fixations) - 1}: {coordinate_warning}"
                )

        stopping = {
            "time_horizon": StoppingReason.TIME_HORIZON,
            "max_fixations": StoppingReason.MAX_FIXATIONS,
            "model_stop": StoppingReason.MODEL_STOP,
        }[sample["stopping_reason"]]
        if stopping is StoppingReason.MAX_FIXATIONS and len(fixations) != effective_max:
            raise AdapterOutputError(
                f"TPP-Gaze sample {index} reported max_fixations before reaching it"
            )
        if stopping is StoppingReason.TIME_HORIZON:
            if not sample["events"] or previous_arrival_s is None or (
                previous_arrival_s + 1e-9 < request.time_horizon_s
            ):
                raise AdapterOutputError(
                    f"TPP-Gaze sample {index} reported the horizon without reaching it"
                )
            if saw_excluded:
                warnings.append(
                    "the terminal event crossed the requested horizon; it remains "
                    "in the native artifact and is excluded from canonical fixations"
                )
        if not fixations:
            warnings.append(
                "TPP-Gaze produced an empty in-horizon sequence; no fixation was fabricated"
            )

        native_path = Path(sample["native_artifact"])
        try:
            native_path.resolve().relative_to(native_directory.resolve())
        except (OSError, ValueError) as error:
            raise AdapterOutputError(
                "TPP-Gaze native artifact escaped its request directory"
            ) from error
        if not native_path.is_file():
            raise AdapterOutputError(
                f"TPP-Gaze native artifact does not exist: {native_path}"
            )
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
            observer_id=request.observer_id,
            task_text=request.task_text,
            target_description=request.target_description,
            fixations=tuple(fixations),
            native_artifacts=(
                ArtifactReference(
                    uri=native_path.resolve().as_uri(),
                    kind="model_native_prediction",
                    media_type="application/json",
                    checksum=f"sha256:{sha256_file(native_path)}",
                    metadata={
                        "protocol_version": PROTOCOL_VERSION,
                        "sample_index": index,
                        "native_time_unit": self.config.native_time_unit,
                    },
                ),
            ),
            warnings=tuple(dict.fromkeys(warnings)),
            native_metadata={
                "tpp_gaze": {
                    "upstream_commit": self.config.upstream_commit,
                    "checkpoint_sha256": metadata["checkpoint_sha256"],
                    "model_config_sha256": metadata["model_config_sha256"],
                    "adapter_config_sha256": metadata["config_sha256"],
                    "context_type": metadata["context_type"],
                    "temperature": metadata["temperature"],
                    "temperature_source": metadata["temperature_source"],
                    "time_horizon_s": request.time_horizon_s,
                    "effective_max_fixations": effective_max,
                    "native_time_unit": self.config.native_time_unit,
                    "temporal_scale_to_seconds": (
                        self.config.temporal_scale_to_seconds
                    ),
                    "native_event_count": len(sample["events"]),
                    "canonical_fixation_count": len(fixations),
                    "native_artifact_policy": self.config.native_artifact_policy,
                    "nondeterminism": (
                        "per-simulation Python, NumPy, and Torch seeds are set; "
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
        self._native_root = self.config.output_root / run_id / "native" / "tpp_gaze"
        self._native_root.mkdir(parents=True, exist_ok=True)
        session_id = uuid.uuid4().hex
        config_path = self._native_root / f"worker-config-{session_id}.json"
        ready_path = self._native_root / f"worker-ready-{session_id}.json"
        assert self._checkpoint_sha256 is not None
        assert self._model_config_sha256 is not None
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
                "model_config_path": str(self.config.model_config_path),
                "model_config_sha256": self._model_config_sha256,
                "device": self.config.device,
                "temperature": self.config.temperature,
                "native_time_unit": self.config.native_time_unit,
                "temporal_scale_to_seconds": (
                    self.config.temporal_scale_to_seconds
                ),
                "safety_max_fixations": self.config.safety_max_fixations,
                "native_artifact_policy": self.config.native_artifact_policy,
                "config_sha256": self._config_sha256,
                "fake_behavior": self.config.fake_behavior,
            },
        )
        log_root = self.config.output_root / run_id / "logs"
        log_root.mkdir(parents=True, exist_ok=True)
        self._stdout_log = (log_root / "tpp-gaze-worker.stdout.log").open(
            "a", encoding="utf-8"
        )
        self._stderr_log = (log_root / "tpp-gaze-worker.stderr.log").open(
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
        current_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            source_root
            if not current_pythonpath
            else f"{source_root}{os.pathsep}{current_pythonpath}"
        )
        environment["PYTHONUNBUFFERED"] = "1"
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["TORCH_HOME"] = environment.get(
            "AVRW_TPP_GAZE_TORCH_HOME", str(self._native_root / "torch-cache")
        )
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
            raise TPPGazeWorkerError(
                f"cannot start TPP-Gaze worker with {command[0]!r}: {error}"
            ) from error
        self._worker_stdin = self._process.stdin
        try:
            _wait_for_path(
                ready_path,
                process=self._process,
                timeout_s=self.config.startup_timeout_s,
                description="TPP-Gaze worker startup",
            )
            ready = read_json(ready_path, label="TPP-Gaze worker readiness")
            if not isinstance(ready, dict) or ready.get("status") != "ready":
                raise TPPGazeWorkerError(
                    "TPP-Gaze worker wrote an invalid readiness document"
                )
        except Exception:
            self._stop_worker()
            raise

    def _send_command(self, request_path: Path, response_path: Path) -> None:
        process = self._process
        stream = self._worker_stdin
        if process is None or stream is None or process.poll() is not None:
            raise AdapterLifecycleError("TPP-Gaze worker is not running")
        command = json.dumps(
            {"request_path": str(request_path), "response_path": str(response_path)},
            separators=(",", ":"),
            sort_keys=True,
        )
        try:
            stream.write(f"{command}\n")
            stream.flush()
            _wait_for_path(
                response_path,
                process=process,
                timeout_s=self.config.worker_timeout_s,
                description="TPP-Gaze inference",
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
        for handle_name in ("_stdout_log", "_stderr_log"):
            handle = getattr(self, handle_name)
            if handle is not None:
                handle.close()
                setattr(self, handle_name, None)


def tpp_gaze_descriptor(config: TPPGazeAdapterConfig) -> AdapterDescriptor:

    return AdapterDescriptor(
        model_id=TPP_GAZE_MODEL_ID,
        identity=_identity(config),
        capabilities=TPP_GAZE_CAPABILITIES,
        requirements=RequestRequirements(),
        factory=lambda: TPPGazeAdapter(config),
    )


def create_tpp_gaze_registry(config: TPPGazeAdapterConfig) -> AdapterRegistry:

    registry = AdapterRegistry(allowed_model_ids={TPP_GAZE_MODEL_ID})
    registry.register(tpp_gaze_descriptor(config))
    return registry


def _identity(config: TPPGazeAdapterConfig) -> ModelIdentity:
    checkpoint_id = config.checkpoint_fingerprint
    if checkpoint_id is None:
        checkpoint_id = f"external:{config.checkpoint_path.name}"
    return ModelIdentity(
        model_id=TPP_GAZE_MODEL_ID,
        model_variant=config.variant,
        adapter_version=TPP_GAZE_ADAPTER_VERSION,
        upstream_version=config.upstream_commit,
        checkpoint_id=checkpoint_id,
    )


def _sample_id(request_id: str, sample_index: int) -> str:
    return f"{request_id}:sample:{sample_index}"


def _converted_time(value: object, *, scale: float, label: str) -> float | None:
    if value is None:
        return None
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise AdapterOutputError(f"TPP-Gaze {label} must be finite or null")
    return float(value) * scale


def _validate_model_time(
    converted: float | None, model_seconds: object, label: str
) -> None:
    if model_seconds is None:
        if converted is not None:
            raise AdapterOutputError(
                f"TPP-Gaze {label} lacks its model-seconds provenance"
            )
        return
    if type(model_seconds) not in (int, float) or not math.isfinite(
        float(model_seconds)
    ):
        raise AdapterOutputError(f"TPP-Gaze {label} model value must be finite")
    if converted is None or not math.isclose(
        converted, float(model_seconds), rel_tol=1e-7, abs_tol=1e-9
    ):
        raise AdapterOutputError(
            f"TPP-Gaze {label} disagrees with the explicit unit conversion"
        )


def _require_file(path: Path, name: str) -> None:
    if not path.is_file():
        raise TPPGazeWorkerError(
            f"TPP-Gaze {name} is missing or not a file: {path}"
        )


def _require_directory(path: Path, name: str) -> None:
    if not path.is_dir():
        raise TPPGazeWorkerError(
            f"TPP-Gaze {name} is missing or not a directory: {path}"
        )


def _upstream_commit(root: Path) -> str:
    try:
        result = subprocess.run(
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
        raise TPPGazeWorkerError(
            f"cannot verify TPP-Gaze upstream checkout at {root}: {error}"
        ) from error
    if status.stdout.strip():
        raise TPPGazeWorkerError(
            "TPP-Gaze upstream checkout has tracked modifications; commit or "
            "restore them before inference"
        )
    return result.stdout.strip()


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
        return_code = process.poll()
        if return_code is not None:
            raise TPPGazeWorkerError(
                f"{description} worker exited with code {return_code}; inspect "
                "logs/tpp-gaze-worker.stderr.log"
            )
        time.sleep(0.02)
    raise TPPGazeWorkerError(
        f"{description} timed out after {timeout_s:g}s; the worker was terminated"
    )
