
from __future__ import annotations

import hashlib
import json
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
from activevision_workbench.adapters.scandiff.config import (
    SCANDIFF_FREE_VIEWING_VARIANT,
    SCANDIFF_MAX_FIXATIONS,
    SCANDIFF_MODEL_ID,
    SCANDIFF_UPSTREAM_COMMIT,
    SCANDIFF_VISUAL_SEARCH_VARIANT,
    ScanDiffAdapterConfig,
    adapter_option_names,
)
from activevision_workbench.adapters.scandiff.protocol import (
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

SCANDIFF_ADAPTER_VERSION = "0.1.0"
SCANDIFF_CAPABILITIES = CapabilitySet(
    frozenset(
        {
            Capability.PRODUCES_SCANPATHS,
            Capability.STOCHASTIC_MULTI_SAMPLE,
            Capability.FIXATION_DURATION_OUTPUT,
        }
    )
)
SCANDIFF_VISUAL_SEARCH_CAPABILITIES = CapabilitySet(
    frozenset(
        set(SCANDIFF_CAPABILITIES.values)
        | {Capability.INSTRUCTION_CONDITIONED}
    )
)


class ScanDiffWorkerError(AdapterError):
    pass


class ScanDiffAdapter(ModelAdapter):

    def __init__(self, config: ScanDiffAdapterConfig) -> None:
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
        self._embeddings_sha256: str | None = None
        self._config_sha256: str | None = None

    @property
    def identity(self) -> ModelIdentity:
        return self._identity

    @property
    def capabilities(self) -> CapabilitySet:
        if self.config.variant == SCANDIFF_VISUAL_SEARCH_VARIANT:
            return SCANDIFF_VISUAL_SEARCH_CAPABILITIES
        return SCANDIFF_CAPABILITIES

    @property
    def requirements(self) -> RequestRequirements:
        return RequestRequirements(
            instruction=self.config.variant == SCANDIFF_VISUAL_SEARCH_VARIANT
        )

    def _validate_request(self, request: InferenceRequest) -> None:
        if request.image_width is None or request.image_height is None:
            raise ContractValidationError(
                "InferenceRequest.image_width",
                "ScanDiff conversion requires image_width and image_height",
            )
        if request.image_width < 2 or request.image_height < 2:
            raise ContractValidationError(
                "InferenceRequest.image_width",
                "ScanDiff canonical coordinates require dimensions >= 2",
            )
        if request.max_fixations not in (None, SCANDIFF_MAX_FIXATIONS):
            raise ContractValidationError(
                "InferenceRequest.max_fixations",
                "the pinned ScanDiff checkpoint supports only max_fixations=16",
            )
        maximum_seed = request.base_seed + request.num_samples - 1
        if maximum_seed > (2**32 - 1):
            raise ContractValidationError(
                "InferenceRequest.base_seed",
                "base_seed + num_samples - 1 must fit the NumPy uint32 seed range",
            )
        if self.config.variant == SCANDIFF_FREE_VIEWING_VARIANT:
            if request.task_text is not None:
                raise ContractValidationError(
                    "InferenceRequest.task_text",
                    "free-viewing ScanDiff uses the upstream empty task only",
                )
            if request.target_description is not None:
                raise ContractValidationError(
                    "InferenceRequest.target_description",
                    "free-viewing ScanDiff does not accept a target",
                )
        else:
            supplied = (
                request.task_text is not None,
                request.target_description is not None,
            )
            if all(supplied):
                raise ContractValidationError(
                    "InferenceRequest.task_text",
                    "provide exactly one ScanDiff category in task_text or "
                    "target_description, not both",
                )
            viewing_task = (
                request.task_text
                if request.task_text is not None
                else request.target_description
            )
            if viewing_task == "":
                field = (
                    "InferenceRequest.task_text"
                    if request.task_text is not None
                    else "InferenceRequest.target_description"
                )
                raise ContractValidationError(
                    field,
                    "visual-search ScanDiff requires a non-empty category",
                )

        options = request.model_options.get(SCANDIFF_MODEL_ID, {})
        if not isinstance(options, Mapping):  # guarded by the core contract
            return
        unknown = set(options) - adapter_option_names()
        if unknown:
            name = sorted(unknown)[0]
            raise ContractValidationError(
                f"InferenceRequest.model_options.scandiff.{name}",
                "unknown ScanDiff adapter option",
            )

    def _prepare(self) -> None:
        _require_directory(self.config.upstream_root, "upstream_root")
        _require_file(self.config.checkpoint_path, "checkpoint_path")
        _require_file(self.config.task_embeddings_path, "task_embeddings_path")
        if self.config.feature_root is not None:
            _require_directory(self.config.feature_root, "feature_root")
        launcher = self.config.launcher[0]
        if os.sep in launcher:
            if not Path(launcher).is_file():
                raise ScanDiffWorkerError(
                    f"ScanDiff launcher does not exist: {launcher}"
                )
        elif shutil.which(launcher) is None:
            raise ScanDiffWorkerError(
                f"ScanDiff launcher is not on PATH: {launcher}"
            )
        if not self.config.fake_worker:
            actual_commit = _upstream_commit(self.config.upstream_root)
            if actual_commit != self.config.upstream_commit:
                raise ScanDiffWorkerError(
                    "ScanDiff checkout commit mismatch: expected "
                    f"{self.config.upstream_commit}, found {actual_commit}"
                )
        self._checkpoint_sha256 = sha256_file(self.config.checkpoint_path)
        self._embeddings_sha256 = sha256_file(self.config.task_embeddings_path)
        configured = self.config.checkpoint_fingerprint
        if configured is not None:
            expected = configured.removeprefix("sha256:")
            if len(expected) == 64 and all(
                char in "0123456789abcdefABCDEF" for char in expected
            ):
                if self._checkpoint_sha256.lower() != expected.lower():
                    raise ScanDiffWorkerError(
                        "checkpoint SHA-256 does not match the configured fingerprint"
                    )
        fingerprint_payload = {
            "protocol_version": PROTOCOL_VERSION,
            "adapter_version": SCANDIFF_ADAPTER_VERSION,
            "variant": self.config.variant,
            "upstream_commit": self.config.upstream_commit,
            "checkpoint_sha256": self._checkpoint_sha256,
            "task_embeddings_sha256": self._embeddings_sha256,
            "feature_root": (
                None
                if self.config.feature_root is None
                else str(self.config.feature_root)
            ),
            "device": self.config.device,
            "max_fixations": SCANDIFF_MAX_FIXATIONS,
            "allow_feature_download": self.config.allow_feature_download,
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
        viewing_task = ""
        if self.config.variant == SCANDIFF_VISUAL_SEARCH_VARIANT:
            viewing_task = request.task_text or request.target_description or ""
        sample_specs = [
            {
                "sample_id": _sample_id(request.request_id, index),
                "sample_index": index,
                "seed": request.base_seed + index,
            }
            for index in range(request.num_samples)
        ]
        payload = {
            "schema_version": PROTOCOL_VERSION,
            "request_id": request.request_id,
            "image_ref": request.image_ref,
            "image_width": request.image_width,
            "image_height": request.image_height,
            "viewing_task": viewing_task,
            "sample_specs": sample_specs,
            "max_fixations": SCANDIFF_MAX_FIXATIONS,
            "native_output_dir": str(native_directory.resolve()),
        }
        atomic_write_json(request_path, payload)
        self._send_command(request_path, response_path)
        response = read_json(response_path, label="ScanDiff worker response")
        reported_error = worker_error(response)
        if reported_error is not None:
            raise ScanDiffWorkerError(f"ScanDiff worker failed: {reported_error}")
        samples, metadata = validate_response(
            response,
            request_id=request.request_id,
            sample_count=request.num_samples,
        )
        if metadata["image_width"] != request.image_width or metadata[
            "image_height"
        ] != request.image_height:
            raise AdapterOutputError(
                "ScanDiff worker image dimensions do not match the canonical request"
            )
        if metadata["upstream_commit"] != self.config.upstream_commit:
            raise AdapterOutputError(
                "ScanDiff worker upstream commit does not match adapter configuration"
            )
        if metadata["checkpoint_sha256"] != self._checkpoint_sha256:
            raise AdapterOutputError("ScanDiff worker checkpoint fingerprint changed")
        if metadata["task_embeddings_sha256"] != self._embeddings_sha256:
            raise AdapterOutputError(
                "ScanDiff worker task-embedding fingerprint changed"
            )
        if metadata["config_sha256"] != self._config_sha256:
            raise AdapterOutputError("ScanDiff worker config fingerprint changed")
        return tuple(
            self._prediction_record(
                request,
                run_id=run_id,
                sample=sample,
                metadata=metadata,
                native_directory=native_directory,
            )
            for sample in samples
        )

    def _prediction_record(
        self,
        request: InferenceRequest,
        *,
        run_id: str,
        sample: dict[str, Any],
        metadata: dict[str, Any],
        native_directory: Path,
    ) -> PredictionRecord:
        expected_index = sample["sample_index"]
        expected_id = _sample_id(request.request_id, expected_index)
        expected_seed = request.base_seed + expected_index
        if sample["sample_id"] != expected_id:
            raise AdapterOutputError(
                f"ScanDiff sample {expected_index} has an unstable sample_id"
            )
        if sample["seed"] != expected_seed:
            raise AdapterOutputError(
                f"ScanDiff sample {expected_index} reported the wrong seed"
            )
        warnings = list(sample["warnings"])
        fixations: list[FixationEvent] = []
        assert request.image_width is not None and request.image_height is not None
        for sequence_index, raw in enumerate(sample["fixations"]):
            native_x = float(raw["x_norm"])
            native_y = float(raw["y_norm"])
            native_duration = float(raw["duration_s"])
            duration_s = native_duration if native_duration >= 0.0 else None
            if duration_s is None:
                warnings.append(
                    f"fixation {sequence_index} has a negative native duration; "
                    "it was preserved but not promoted to canonical duration_s"
                )
            fixation, coordinate_warning = source_fixation_from_pixel(
                x_px=native_x * request.image_width,
                y_px=native_y * request.image_height,
                image_width=request.image_width,
                image_height=request.image_height,
                sequence_index=sequence_index,
                duration_s=duration_s,
                native_timing={
                    "value": native_duration,
                    "unit": "s",
                    "meaning": "fixation_duration",
                },
                native_metadata={
                    "scandiff": {
                        "x_normalized": native_x,
                        "y_normalized": native_y,
                        "pixel_transform": "x_norm*width,y_norm*height",
                    }
                },
                clip_out_of_bounds=False,
            )
            fixations.append(fixation)
            if coordinate_warning is not None:
                warnings.append(f"fixation {sequence_index}: {coordinate_warning}")
        native_path = Path(sample["native_artifact"])
        try:
            native_path.resolve().relative_to(native_directory.resolve())
        except (OSError, ValueError) as error:
            raise AdapterOutputError(
                "ScanDiff native artifact escaped its request directory"
            ) from error
        if not native_path.is_file():
            raise AdapterOutputError(
                f"ScanDiff native artifact does not exist: {native_path}"
            )
        stopping = {
            "model_stop": StoppingReason.MODEL_STOP,
            "max_fixations": StoppingReason.MAX_FIXATIONS,
        }.get(sample["stopping_reason"])
        if stopping is None:
            raise AdapterOutputError(
                "ScanDiff worker returned an unknown stopping_reason"
            )
        return PredictionRecord(
            run_id=run_id,
            request_id=request.request_id,
            model=self.identity,
            dataset_id=request.dataset_id,
            dataset_split=request.dataset_split,
            item_id=request.item_id,
            sample_id=sample["sample_id"],
            sample_index=sample["sample_index"],
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
                        "sample_index": sample["sample_index"],
                    },
                ),
            ),
            warnings=tuple(dict.fromkeys(warnings)),
            native_metadata={
                "scandiff": {
                    "upstream_commit": self.config.upstream_commit,
                    "checkpoint_sha256": metadata["checkpoint_sha256"],
                    "task_embeddings_sha256": metadata[
                        "task_embeddings_sha256"
                    ],
                    "feature_sha256": metadata["feature_sha256"],
                    "adapter_config_sha256": metadata["config_sha256"],
                    "viewing_task": request.task_text
                    or request.target_description
                    or "",
                    "max_len": SCANDIFF_MAX_FIXATIONS,
                    "native_artifact_policy": self.config.native_artifact_policy,
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
        self._native_root = self.config.output_root / run_id / "native" / "scandiff"
        self._native_root.mkdir(parents=True, exist_ok=True)
        session_id = uuid.uuid4().hex
        config_path = self._native_root / f"worker-config-{session_id}.json"
        ready_path = self._native_root / f"worker-ready-{session_id}.json"
        assert self._checkpoint_sha256 is not None
        assert self._embeddings_sha256 is not None
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
                "task_embeddings_path": str(self.config.task_embeddings_path),
                "task_embeddings_sha256": self._embeddings_sha256,
                "feature_root": (
                    None
                    if self.config.feature_root is None
                    else str(self.config.feature_root)
                ),
                "device": self.config.device,
                "max_fixations": SCANDIFF_MAX_FIXATIONS,
                "allow_feature_download": self.config.allow_feature_download,
                "native_artifact_policy": self.config.native_artifact_policy,
                "config_sha256": self._config_sha256,
                "fake_behavior": self.config.fake_behavior,
            },
        )
        log_root = self.config.output_root / run_id / "logs"
        log_root.mkdir(parents=True, exist_ok=True)
        stdout_path = log_root / "scandiff-worker.stdout.log"
        stderr_path = log_root / "scandiff-worker.stderr.log"
        self._stdout_log = stdout_path.open("a", encoding="utf-8")
        self._stderr_log = stderr_path.open("a", encoding="utf-8")
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
        if not self.config.allow_feature_download:
            environment["HF_HUB_OFFLINE"] = "1"
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
            raise ScanDiffWorkerError(
                f"cannot start ScanDiff worker with {command[0]!r}: {error}"
            ) from error
        self._worker_stdin = self._process.stdin
        try:
            _wait_for_path(
                ready_path,
                process=self._process,
                timeout_s=self.config.startup_timeout_s,
                description="ScanDiff worker startup",
            )
            ready = read_json(ready_path, label="ScanDiff worker readiness")
            if not isinstance(ready, dict) or ready.get("status") != "ready":
                raise ScanDiffWorkerError(
                    "ScanDiff worker wrote an invalid readiness document"
                )
        except Exception:
            self._stop_worker()
            raise

    def _send_command(self, request_path: Path, response_path: Path) -> None:
        process = self._process
        stream = self._worker_stdin
        if process is None or stream is None or process.poll() is not None:
            raise AdapterLifecycleError("ScanDiff worker is not running")
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
                description="ScanDiff inference",
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


def scandiff_descriptor(config: ScanDiffAdapterConfig) -> AdapterDescriptor:

    capabilities = (
        SCANDIFF_VISUAL_SEARCH_CAPABILITIES
        if config.variant == SCANDIFF_VISUAL_SEARCH_VARIANT
        else SCANDIFF_CAPABILITIES
    )
    requirements = RequestRequirements(
        instruction=config.variant == SCANDIFF_VISUAL_SEARCH_VARIANT
    )
    return AdapterDescriptor(
        model_id=SCANDIFF_MODEL_ID,
        identity=_identity(config),
        capabilities=capabilities,
        requirements=requirements,
        factory=lambda: ScanDiffAdapter(config),
    )


def create_scandiff_registry(config: ScanDiffAdapterConfig) -> AdapterRegistry:

    registry = AdapterRegistry(allowed_model_ids={SCANDIFF_MODEL_ID})
    registry.register(scandiff_descriptor(config))
    return registry


def _identity(config: ScanDiffAdapterConfig) -> ModelIdentity:
    checkpoint_id = config.checkpoint_fingerprint
    if checkpoint_id is None:
        checkpoint_id = f"external:{config.checkpoint_path.name}"
    return ModelIdentity(
        model_id=SCANDIFF_MODEL_ID,
        model_variant=config.variant,
        adapter_version=SCANDIFF_ADAPTER_VERSION,
        upstream_version=config.upstream_commit,
        checkpoint_id=checkpoint_id,
    )


def _sample_id(request_id: str, sample_index: int) -> str:
    return f"{request_id}:sample:{sample_index}"


def _require_file(path: Path, name: str) -> None:
    if not path.is_file():
        raise ScanDiffWorkerError(f"ScanDiff {name} is missing or not a file: {path}")


def _require_directory(path: Path, name: str) -> None:
    if not path.is_dir():
        raise ScanDiffWorkerError(
            f"ScanDiff {name} is missing or not a directory: {path}"
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
        raise ScanDiffWorkerError(
            f"cannot verify ScanDiff upstream checkout at {root}: {error}"
        ) from error
    if status.stdout.strip():
        raise ScanDiffWorkerError(
            "ScanDiff upstream checkout has tracked modifications; commit or "
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
            raise ScanDiffWorkerError(
                f"{description} worker exited with code {return_code}; inspect "
                "logs/scandiff-worker.stderr.log"
            )
        time.sleep(0.02)
    raise ScanDiffWorkerError(
        f"{description} timed out after {timeout_s:g}s; the worker was terminated"
    )
