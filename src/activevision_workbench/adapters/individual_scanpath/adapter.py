
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
from activevision_workbench.adapters.individual_scanpath.config import (
    INDIVIDUAL_SCANPATH_CHENLSTM_OSIE_CHECKPOINT_SHA256,
    INDIVIDUAL_SCANPATH_DATASET_ID,
    INDIVIDUAL_SCANPATH_EMBEDDING_DIM,
    INDIVIDUAL_SCANPATH_INPUT_HEIGHT,
    INDIVIDUAL_SCANPATH_INPUT_WIDTH,
    INDIVIDUAL_SCANPATH_MAP_HEIGHT,
    INDIVIDUAL_SCANPATH_MAP_WIDTH,
    INDIVIDUAL_SCANPATH_MODEL_ID,
    INDIVIDUAL_SCANPATH_NATIVE_ARTIFACT_POLICY,
    INDIVIDUAL_SCANPATH_SUBJECT_COUNT,
    IndividualScanpathAdapterConfig,
    adapter_option_names,
)
from activevision_workbench.adapters.individual_scanpath.observer_mapping import (
    ObserverMapping,
    load_observer_mapping,
)
from activevision_workbench.adapters.individual_scanpath.protocol import (
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

INDIVIDUAL_SCANPATH_ADAPTER_VERSION = "0.1.0"
INDIVIDUAL_SCANPATH_CAPABILITIES = CapabilitySet(
    frozenset(
        {
            Capability.PRODUCES_SCANPATHS,
            Capability.STOCHASTIC_MULTI_SAMPLE,
            Capability.OBSERVER_CONDITIONED,
            Capability.FIXATION_DURATION_OUTPUT,
        }
    )
)


class IndividualScanpathWorkerError(AdapterError):
    pass


class IndividualScanpathAdapter(ModelAdapter):

    def __init__(self, config: IndividualScanpathAdapterConfig) -> None:
        super().__init__()
        self.config = config
        self._identity = _identity(config)
        self._mapping: ObserverMapping | None = None
        self._checkpoint_sha256: str | None = None
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
        return INDIVIDUAL_SCANPATH_CAPABILITIES

    @property
    def requirements(self) -> RequestRequirements:
        return RequestRequirements(observer=True)

    def _validate_request(self, request: InferenceRequest) -> None:
        if request.observer_id is None:
            raise ContractValidationError(
                "InferenceRequest.observer_id",
                "is required; observer metadata alone cannot select an embedding",
            )
        if request.dataset_id != INDIVIDUAL_SCANPATH_DATASET_ID:
            raise ContractValidationError(
                "InferenceRequest.dataset_id",
                f"must equal '{INDIVIDUAL_SCANPATH_DATASET_ID}' for this checkpoint",
            )
        if request.image_width is None or request.image_height is None:
            raise ContractValidationError(
                "InferenceRequest.image_width", "image dimensions are required"
            )
        if request.time_horizon_s is not None:
            raise ContractValidationError(
                "InferenceRequest.time_horizon_s",
                "the selected model stops by EOS or fixation count, not time horizon",
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
        options = request.model_options.get(INDIVIDUAL_SCANPATH_MODEL_ID, {})
        if isinstance(options, Mapping):
            unknown = set(options) - adapter_option_names()
            if unknown:
                raise ContractValidationError(
                    f"InferenceRequest.model_options.individualscanpath.{sorted(unknown)[0]}",
                    "unknown IndividualScanpath option",
                )
        self._observer_mapping().index_for(request.observer_id)

    def _prepare(self) -> None:
        _require_directory(self.config.upstream_root, "upstream_root")
        _require_file(self.config.checkpoint_path, "checkpoint_path")
        mapping = self._observer_mapping()
        launcher = self.config.launcher[0]
        if os.sep in launcher:
            if not Path(launcher).is_file():
                raise IndividualScanpathWorkerError(
                    f"IndividualScanpath launcher does not exist: {launcher}"
                )
        elif shutil.which(launcher) is None:
            raise IndividualScanpathWorkerError(
                f"IndividualScanpath launcher is not on PATH: {launcher}"
            )
        if not self.config.fake_worker:
            actual_commit = _upstream_commit(self.config.upstream_root)
            if actual_commit != self.config.upstream_commit:
                raise IndividualScanpathWorkerError(
                    "IndividualScanpath checkout commit mismatch: expected "
                    f"{self.config.upstream_commit}, found {actual_commit}"
                )
        self._checkpoint_sha256 = sha256_file(self.config.checkpoint_path)
        if not self.config.fake_worker and self._checkpoint_sha256 != (
            INDIVIDUAL_SCANPATH_CHENLSTM_OSIE_CHECKPOINT_SHA256
        ):
            raise IndividualScanpathWorkerError(
                "IndividualScanpath checkpoint SHA-256 does not match the pinned OSIE ChenLSTM model"
            )
        configured = self.config.checkpoint_fingerprint
        if configured is not None and configured.removeprefix("sha256:").lower() != (
            self._checkpoint_sha256
        ):
            raise IndividualScanpathWorkerError(
                "checkpoint SHA-256 does not match configured fingerprint"
            )
        payload = {
            "protocol_version": PROTOCOL_VERSION,
            "adapter_version": INDIVIDUAL_SCANPATH_ADAPTER_VERSION,
            "variant": self.config.variant,
            "upstream_commit": self.config.upstream_commit,
            "checkpoint_sha256": self._checkpoint_sha256,
            "observer_mapping_sha256": mapping.sha256,
            "unknown_observer_policy": self.config.unknown_observer_policy,
            "device": self.config.device,
            "min_fixations": self.config.min_fixations,
            "model_max_fixations": self.config.model_max_fixations,
            "native_artifact_policy": self.config.native_artifact_policy,
        }
        self._config_sha256 = hashlib.sha256(
            json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        ).hexdigest()

    def _predict(
        self, request: InferenceRequest, *, run_id: str
    ) -> Iterable[PredictionRecord]:
        self._ensure_worker(run_id)
        assert self._native_root is not None
        assert request.observer_id is not None
        assert request.image_width is not None
        assert request.image_height is not None
        mapping = self._observer_mapping()
        observer_index = mapping.index_for(request.observer_id)
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
                "observer_id": request.observer_id,
                "model_observer_index": observer_index,
                "sample_specs": sample_specs,
                "max_fixations": effective_max,
                "native_output_dir": str(native_directory.resolve()),
            },
        )
        self._send_command(request_path, response_path)
        response = read_json(response_path, label="IndividualScanpath worker response")
        error = worker_error(response)
        if error is not None:
            raise IndividualScanpathWorkerError(
                f"IndividualScanpath worker failed: {error}"
            )
        samples, metadata = validate_response(
            response, request_id=request.request_id, sample_count=request.num_samples
        )
        self._validate_metadata(request, metadata)
        return tuple(
            self._record(
                request,
                run_id=run_id,
                sample=sample,
                metadata=metadata,
                native_directory=native_directory,
                observer_index=observer_index,
                effective_max=effective_max,
            )
            for sample in samples
        )

    def _validate_metadata(
        self, request: InferenceRequest, metadata: dict[str, Any]
    ) -> None:
        mapping = self._observer_mapping()
        expected = {
            "checkpoint_sha256": self._checkpoint_sha256,
            "observer_mapping_sha256": mapping.sha256,
            "config_sha256": self._config_sha256,
            "upstream_commit": self.config.upstream_commit,
            "variant": self.config.variant,
            "image_width": request.image_width,
            "image_height": request.image_height,
            "model_input_width": INDIVIDUAL_SCANPATH_INPUT_WIDTH,
            "model_input_height": INDIVIDUAL_SCANPATH_INPUT_HEIGHT,
            "map_width": INDIVIDUAL_SCANPATH_MAP_WIDTH,
            "map_height": INDIVIDUAL_SCANPATH_MAP_HEIGHT,
        }
        for name, value in expected.items():
            if metadata[name] != value:
                raise AdapterOutputError(
                    f"IndividualScanpath worker metadata mismatch for {name}"
                )
        if metadata["image_cache_scope"] != (
            "decoded-and-preprocessed-image-only; no observer-conditioned outputs"
        ):
            raise AdapterOutputError("IndividualScanpath worker reported an unsafe cache scope")

    def _record(
        self,
        request: InferenceRequest,
        *,
        run_id: str,
        sample: dict[str, Any],
        metadata: dict[str, Any],
        native_directory: Path,
        observer_index: int,
        effective_max: int,
    ) -> PredictionRecord:
        index = sample["sample_index"]
        if sample["sample_id"] != _sample_id(request.request_id, index):
            raise AdapterOutputError("IndividualScanpath worker returned an unstable sample ID")
        if sample["seed"] != request.base_seed + index:
            raise AdapterOutputError("IndividualScanpath worker returned the wrong seed")
        if sample["observer_id"] != request.observer_id or sample[
            "model_observer_index"
        ] != observer_index:
            raise AdapterOutputError("IndividualScanpath worker changed observer identity")
        assert request.image_width is not None
        assert request.image_height is not None
        fixations: list[FixationEvent] = []
        warnings = list(sample["warnings"])
        for sequence_index, event in enumerate(sample["events"]):
            if sequence_index >= effective_max:
                raise AdapterOutputError("IndividualScanpath exceeded the requested fixation cap")
            duration = _finite(event["duration_s"], "duration_s")
            probability = _finite(event["action_probability"], "action_probability")
            if duration < 0:
                raise AdapterOutputError("IndividualScanpath returned a negative duration")
            if not 0 <= probability <= 1:
                raise AdapterOutputError("IndividualScanpath action probability is outside [0, 1]")
            x_model = _finite(event["x_model_px"], "x_model_px")
            y_model = _finite(event["y_model_px"], "y_model_px")
            # Upstream trains/evaluates by dividing MATLAB one-based OSIE
            # coordinates by the resize scale. Invert that exact transform,
            # then subtract one for AVRW's zero-based pixel-center lattice.
            x_px = x_model * request.image_width / INDIVIDUAL_SCANPATH_INPUT_WIDTH - 1.0
            y_px = y_model * request.image_height / INDIVIDUAL_SCANPATH_INPUT_HEIGHT - 1.0
            fixation, warning = source_fixation_from_pixel(
                x_px=x_px,
                y_px=y_px,
                image_width=request.image_width,
                image_height=request.image_height,
                sequence_index=sequence_index,
                duration_s=duration,
                native_timing={
                    "duration": duration,
                    "unit": "seconds",
                    "source": "upstream log-normal duration sample",
                },
                native_metadata={
                    "individualscanpath": {
                        "action_index": event["action_index"],
                        "action_probability": probability,
                        "x_model_px": x_model,
                        "y_model_px": y_model,
                        "model_canvas": [
                            INDIVIDUAL_SCANPATH_INPUT_WIDTH,
                            INDIVIDUAL_SCANPATH_INPUT_HEIGHT,
                        ],
                        "coordinate_transform": (
                            "model_coordinate * original_dimension / model_dimension - 1"
                        ),
                    }
                },
                clip_out_of_bounds=False,
            )
            fixations.append(fixation)
            if warning is not None:
                warnings.append(warning)
        stopping = {
            "model_stop": StoppingReason.MODEL_STOP,
            "max_fixations": StoppingReason.MAX_FIXATIONS,
        }[sample["stopping_reason"]]
        if stopping is StoppingReason.MAX_FIXATIONS and len(fixations) != effective_max:
            raise AdapterOutputError("IndividualScanpath reported max_fixations early")
        native_path = Path(sample["native_artifact"])
        try:
            native_path.resolve().relative_to(native_directory.resolve())
        except (OSError, ValueError) as error:
            raise AdapterOutputError("IndividualScanpath native artifact escaped its directory") from error
        if not native_path.is_file():
            raise AdapterOutputError("IndividualScanpath native artifact is missing")
        mapping = self._observer_mapping()
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
                        "observer_id": request.observer_id,
                        "model_observer_index": observer_index,
                        "sample_index": index,
                    },
                ),
            ),
            warnings=tuple(dict.fromkeys(warnings)),
            native_metadata={
                "individualscanpath": {
                    "observer_id": request.observer_id,
                    "model_observer_index": observer_index,
                    "known_checkpoint_observer": True,
                    "observer_attributes_supported": False,
                    "request_observer_metadata": dict(request.observer_metadata),
                    "observer_mapping_sha256": mapping.sha256,
                    "observer_mapping_identity_source": mapping.identity_source,
                    "unknown_observer_policy": self.config.unknown_observer_policy,
                    "checkpoint_sha256": metadata["checkpoint_sha256"],
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

    def _observer_mapping(self) -> ObserverMapping:
        if self._mapping is None:
            self._mapping = load_observer_mapping(
                self.config.observer_mapping_path,
                expected_fingerprint=self.config.observer_mapping_fingerprint,
                expected_dataset_id=INDIVIDUAL_SCANPATH_DATASET_ID,
                expected_variant=self.config.variant,
                expected_checkpoint_sha256=INDIVIDUAL_SCANPATH_CHENLSTM_OSIE_CHECKPOINT_SHA256,
                expected_subject_count=INDIVIDUAL_SCANPATH_SUBJECT_COUNT,
            )
        return self._mapping

    def _ensure_worker(self, run_id: str) -> None:
        if self._process is not None and self._process.poll() is None:
            if self._worker_run_id == run_id:
                return
            self._stop_worker()
        self._worker_run_id = run_id
        self._native_root = (
            self.config.output_root / run_id / "native" / "individualscanpath"
        )
        self._native_root.mkdir(parents=True, exist_ok=True)
        session = uuid.uuid4().hex
        config_path = self._native_root / f"worker-config-{session}.json"
        ready_path = self._native_root / f"worker-ready-{session}.json"
        assert self._checkpoint_sha256 is not None
        assert self._config_sha256 is not None
        mapping = self._observer_mapping()
        atomic_write_json(
            config_path,
            {
                "schema_version": PROTOCOL_VERSION,
                "variant": self.config.variant,
                "upstream_root": str(self.config.upstream_root),
                "upstream_commit": self.config.upstream_commit,
                "checkpoint_path": str(self.config.checkpoint_path),
                "checkpoint_sha256": self._checkpoint_sha256,
                "observer_mapping_sha256": mapping.sha256,
                "device": self.config.device,
                "min_fixations": self.config.min_fixations,
                "model_max_fixations": self.config.model_max_fixations,
                "input_width": INDIVIDUAL_SCANPATH_INPUT_WIDTH,
                "input_height": INDIVIDUAL_SCANPATH_INPUT_HEIGHT,
                "map_width": INDIVIDUAL_SCANPATH_MAP_WIDTH,
                "map_height": INDIVIDUAL_SCANPATH_MAP_HEIGHT,
                "subject_count": INDIVIDUAL_SCANPATH_SUBJECT_COUNT,
                "embedding_dim": INDIVIDUAL_SCANPATH_EMBEDDING_DIM,
                "native_artifact_policy": INDIVIDUAL_SCANPATH_NATIVE_ARTIFACT_POLICY,
                "config_sha256": self._config_sha256,
                "fake_behavior": self.config.fake_behavior,
            },
        )
        logs = self.config.output_root / run_id / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        self._stdout_log = (logs / "individualscanpath-worker.stdout.log").open(
            "a", encoding="utf-8"
        )
        self._stderr_log = (logs / "individualscanpath-worker.stderr.log").open(
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
        environment["PYTHONPATH"] = source_root if not current else f"{source_root}{os.pathsep}{current}"
        environment["PYTHONUNBUFFERED"] = "1"
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
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
            raise IndividualScanpathWorkerError(
                f"cannot start IndividualScanpath worker: {error}"
            ) from error
        self._worker_stdin = self._process.stdin
        try:
            _wait_for_path(
                ready_path,
                process=self._process,
                timeout_s=self.config.startup_timeout_s,
                description="IndividualScanpath worker startup",
            )
            ready = read_json(ready_path, label="IndividualScanpath worker readiness")
            if not isinstance(ready, dict) or ready.get("status") != "ready":
                raise IndividualScanpathWorkerError("worker readiness document is invalid")
        except Exception:
            self._stop_worker()
            raise

    def _send_command(self, request_path: Path, response_path: Path) -> None:
        process = self._process
        stream = self._worker_stdin
        if process is None or stream is None or process.poll() is not None:
            raise AdapterLifecycleError("IndividualScanpath worker is not running")
        try:
            stream.write(
                json.dumps(
                    {"request_path": str(request_path), "response_path": str(response_path)},
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
                description="IndividualScanpath inference",
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


def individual_scanpath_descriptor(
    config: IndividualScanpathAdapterConfig,
) -> AdapterDescriptor:
    return AdapterDescriptor(
        model_id=INDIVIDUAL_SCANPATH_MODEL_ID,
        identity=_identity(config),
        capabilities=INDIVIDUAL_SCANPATH_CAPABILITIES,
        requirements=RequestRequirements(observer=True),
        factory=lambda: IndividualScanpathAdapter(config),
    )


def create_individual_scanpath_registry(
    config: IndividualScanpathAdapterConfig,
) -> AdapterRegistry:
    registry = AdapterRegistry(allowed_model_ids={INDIVIDUAL_SCANPATH_MODEL_ID})
    registry.register(individual_scanpath_descriptor(config))
    return registry


def _identity(config: IndividualScanpathAdapterConfig) -> ModelIdentity:
    checkpoint = config.checkpoint_fingerprint or f"external:{config.checkpoint_path.name}"
    return ModelIdentity(
        model_id=INDIVIDUAL_SCANPATH_MODEL_ID,
        model_variant=config.variant,
        adapter_version=INDIVIDUAL_SCANPATH_ADAPTER_VERSION,
        upstream_version=config.upstream_commit,
        checkpoint_id=checkpoint,
    )


def _sample_id(request_id: str, sample_index: int) -> str:
    return f"{request_id}:sample:{sample_index}"


def _finite(value: object, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise AdapterOutputError(f"IndividualScanpath {label} must be finite")
    return float(value)


def _require_file(path: Path, name: str) -> None:
    if not path.is_file():
        raise IndividualScanpathWorkerError(
            f"IndividualScanpath {name} is missing or not a file: {path}"
        )


def _require_directory(path: Path, name: str) -> None:
    if not path.is_dir():
        raise IndividualScanpathWorkerError(
            f"IndividualScanpath {name} is missing or not a directory: {path}"
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
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise IndividualScanpathWorkerError(
            f"cannot verify IndividualScanpath checkout at {root}: {error}"
        ) from error
    if status.stdout.strip():
        raise IndividualScanpathWorkerError(
            "IndividualScanpath checkout has tracked modifications"
        )
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
            raise IndividualScanpathWorkerError(
                f"{description} worker exited with code {code}; inspect logs/individualscanpath-worker.stderr.log"
            )
        time.sleep(0.02)
    raise IndividualScanpathWorkerError(
        f"{description} timed out after {timeout_s:g}s; worker was terminated"
    )
