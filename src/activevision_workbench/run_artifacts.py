
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import socket
import subprocess
import tempfile
import traceback as traceback_module
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from activevision_workbench.contracts import InferenceRequest, PredictionRecord
from activevision_workbench.errors import (
    RunError,
    RunExistsError,
    RunResumeError,
    SerializationError,
)
from activevision_workbench.registry import AdapterDescriptor
from activevision_workbench.run_config import RunConfig
from activevision_workbench.serialization import (
    dumps_prediction,
    loads_prediction,
    read_predictions_jsonl,
)

try:  # Linux/Unix advisory locking; AVRW's supported local/HPC environment.
    import fcntl
except ImportError:  # pragma: no cover - unsupported Windows fallback
    fcntl = None  # type: ignore[assignment]


class RunStatus(str, Enum):

    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    COMPLETED_WITH_ITEM_FAILURES = "completed_with_item_failures"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


@dataclass(frozen=True, slots=True)
class FailureRecord:

    request_id: str
    item_id: str
    exception_type: str
    message: str
    traceback_log_reference: str
    occurred_at: str
    attempt: int
    schema_version: int = 1

    def to_dict(self) -> dict[str, object]:

        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "item_id": self.item_id,
            "exception_type": self.exception_type,
            "message": self.message,
            "traceback_log_reference": self.traceback_log_reference,
            "occurred_at": self.occurred_at,
            "attempt": self.attempt,
        }

    @classmethod
    def from_dict(cls, data: object) -> "FailureRecord":

        if not isinstance(data, Mapping):
            raise SerializationError("failure record must be a JSON object")
        if any(type(key) is not str for key in data):
            raise SerializationError("failure record field names must be strings")
        expected = {
            "schema_version",
            "request_id",
            "item_id",
            "exception_type",
            "message",
            "traceback_log_reference",
            "occurred_at",
            "attempt",
        }
        keys = set(data)
        missing = sorted(expected - keys)
        unknown = sorted(keys - expected)
        if missing:
            raise SerializationError(
                f"failure record field '{missing[0]}' is required"
            )
        if unknown:
            raise SerializationError(
                f"failure record field '{unknown[0]}' is unknown"
            )
        if data["schema_version"] != 1:
            raise SerializationError("failure schema_version must be 1")
        string_fields = (
            "request_id",
            "item_id",
            "exception_type",
            "message",
            "traceback_log_reference",
            "occurred_at",
        )
        for field in string_fields:
            if type(data[field]) is not str or data[field] == "":
                raise SerializationError(
                    f"failure record field '{field}' must be a non-empty string"
                )
        attempt = data["attempt"]
        if type(attempt) is not int or attempt < 1:
            raise SerializationError(
                "failure record field 'attempt' must be an integer >= 1"
            )
        return cls(
            schema_version=1,
            request_id=data["request_id"],  # type: ignore[arg-type]
            item_id=data["item_id"],  # type: ignore[arg-type]
            exception_type=data["exception_type"],  # type: ignore[arg-type]
            message=data["message"],  # type: ignore[arg-type]
            traceback_log_reference=data[
                "traceback_log_reference"
            ],  # type: ignore[arg-type]
            occurred_at=data["occurred_at"],  # type: ignore[arg-type]
            attempt=attempt,
        )


class RunDirectory:

    REQUIRED_DIRECTORIES = ("logs", "native", "figures", "metrics", "slurm")
    CONFIG_NAME = "config.yaml"
    MANIFEST_NAME = "manifest.json"
    PREDICTIONS_NAME = "predictions.jsonl"
    FAILURES_NAME = "failures.jsonl"

    def __init__(self, path: Path) -> None:
        self.path = path

    @classmethod
    def create(
        cls,
        config: RunConfig,
        manifest: Mapping[str, object],
        *,
        overwrite: bool = False,
    ) -> "RunDirectory":

        if config.run_id is None:  # pragma: no cover - engine resolves this
            raise RunError("a concrete run ID is required before directory creation")
        output_root = Path(config.output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        destination = output_root / config.run_id
        if destination.exists() and not overwrite:
            raise RunExistsError(
                f"run directory already exists: {destination}; use resume or "
                "explicit overwrite"
            )

        temporary = Path(
            tempfile.mkdtemp(prefix=f".{config.run_id}.creating-", dir=output_root)
        )
        backup: Path | None = None
        installed = False
        try:
            for name in cls.REQUIRED_DIRECTORIES:
                (temporary / name).mkdir()
            _atomic_write_json(temporary / cls.CONFIG_NAME, config.to_dict())
            _atomic_write_json(temporary / cls.MANIFEST_NAME, manifest)
            _create_empty_file(temporary / cls.PREDICTIONS_NAME)
            _create_empty_file(temporary / cls.FAILURES_NAME)
            _create_empty_file(temporary / "logs" / "events.jsonl")

            if destination.exists():
                backup = output_root / (
                    f".{config.run_id}.replaced-{uuid.uuid4().hex}"
                )
                os.replace(destination, backup)
            try:
                os.replace(temporary, destination)
                installed = True
            except Exception:
                if backup is not None and not destination.exists():
                    os.replace(backup, destination)
                    backup = None
                raise
            if backup is not None:
                shutil.rmtree(backup)
                backup = None
        finally:
            if not installed and temporary.exists():
                shutil.rmtree(temporary)
            if backup is not None and backup.exists() and not destination.exists():
                os.replace(backup, destination)
        return cls(destination)

    @classmethod
    def open(cls, path: str | Path) -> "RunDirectory":

        directory = cls(Path(path))
        required = (
            cls.CONFIG_NAME,
            cls.MANIFEST_NAME,
            cls.PREDICTIONS_NAME,
            cls.FAILURES_NAME,
        )
        missing = [name for name in required if not (directory.path / name).is_file()]
        if missing:
            raise RunResumeError(
                f"run directory '{directory.path}' is incomplete; missing: "
                f"{', '.join(missing)}"
            )
        return directory

    @property
    def manifest_path(self) -> Path:
        return self.path / self.MANIFEST_NAME

    @property
    def predictions_path(self) -> Path:
        return self.path / self.PREDICTIONS_NAME

    @property
    def failures_path(self) -> Path:
        return self.path / self.FAILURES_NAME

    def read_manifest(self) -> dict[str, object]:

        try:
            value = json.loads(
                self.manifest_path.read_text(encoding="utf-8"),
                parse_constant=_reject_json_constant,
            )
        except (OSError, json.JSONDecodeError) as error:
            raise RunResumeError(
                f"cannot read manifest '{self.manifest_path}': {error}"
            ) from error
        if not isinstance(value, dict):
            raise RunResumeError("run manifest must contain a JSON object")
        return value

    def write_manifest(self, manifest: Mapping[str, object]) -> None:

        _atomic_write_json(self.manifest_path, manifest)

    def read_predictions(
        self, *, recover_trailing_record: bool = False
    ) -> tuple[PredictionRecord, ...]:

        try:
            return read_predictions_jsonl(self.predictions_path)
        except SerializationError:
            if not recover_trailing_record:
                raise
            if not _recover_torn_final_line(self.predictions_path, loads_prediction):
                raise
            return read_predictions_jsonl(self.predictions_path)

    def append_predictions(
        self, records: Iterable[PredictionRecord]
    ) -> None:

        batch = tuple(records)
        encoded = tuple(dumps_prediction(record) for record in batch)
        new_keys = [(record.request_id, record.sample_index) for record in batch]
        new_sample_ids = [record.sample_id for record in batch]
        if len(new_keys) != len(set(new_keys)) or len(new_sample_ids) != len(
            set(new_sample_ids)
        ):
            raise RunResumeError("prediction append batch contains duplicate samples")

        with self.predictions_path.open("a+", encoding="utf-8", newline="\n") as stream:
            _lock(stream)
            stream.seek(0)
            existing_keys: set[tuple[str, int]] = set()
            existing_sample_ids: set[str] = set()
            final_line_had_newline = True
            for line_number, line in enumerate(stream, start=1):
                final_line_had_newline = line.endswith("\n")
                if line.strip() == "":
                    raise SerializationError(
                        f"{self.predictions_path}: line {line_number}: blank JSONL "
                        "lines are not allowed"
                    )
                try:
                    record = loads_prediction(line)
                except Exception as error:
                    raise SerializationError(
                        f"{self.predictions_path}: line {line_number}: {error}"
                    ) from error
                existing_keys.add((record.request_id, record.sample_index))
                existing_sample_ids.add(record.sample_id)
            duplicate_keys = existing_keys.intersection(new_keys)
            duplicate_ids = existing_sample_ids.intersection(new_sample_ids)
            if duplicate_keys or duplicate_ids:
                raise RunResumeError(
                    "refusing to append duplicate prediction request/sample identity"
                )
            stream.seek(0, os.SEEK_END)
            if existing_keys and not final_line_had_newline:
                stream.write("\n")
            stream.write("".join(f"{line}\n" for line in encoded))
            stream.flush()
            os.fsync(stream.fileno())

    def read_failures(
        self, *, recover_trailing_record: bool = False
    ) -> tuple[FailureRecord, ...]:

        try:
            return self._read_failures_strict()
        except SerializationError:
            if not recover_trailing_record:
                raise
            if not _recover_torn_final_line(
                self.failures_path, _load_failure_line
            ):
                raise
            return self._read_failures_strict()

    def _read_failures_strict(self) -> tuple[FailureRecord, ...]:
        records: list[FailureRecord] = []
        with self.failures_path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if line.strip() == "":
                    raise SerializationError(
                        f"{self.failures_path}: line {line_number}: blank JSONL "
                        "lines are not allowed"
                    )
                try:
                    value = json.loads(line, parse_constant=_reject_json_constant)
                    records.append(FailureRecord.from_dict(value))
                except (json.JSONDecodeError, SerializationError) as error:
                    raise SerializationError(
                        f"{self.failures_path}: line {line_number}: {error}"
                    ) from error
        return tuple(records)

    def append_failure(
        self,
        request: InferenceRequest,
        error: Exception,
        *,
        attempt: int,
    ) -> FailureRecord:

        trace_dir = self.path / "logs" / "failures"
        trace_dir.mkdir(exist_ok=True)
        safe_id = safe_filename(request.request_id)
        trace_path = trace_dir / f"{safe_id}.attempt-{attempt}.log"
        trace_text = "".join(
            traceback_module.format_exception(type(error), error, error.__traceback__)
        )
        _atomic_write_text(trace_path, trace_text)
        reference = trace_path.relative_to(self.path).as_posix()
        record = FailureRecord(
            request_id=request.request_id,
            item_id=request.item_id,
            exception_type=f"{type(error).__module__}.{type(error).__qualname__}",
            message=str(error) or type(error).__name__,
            traceback_log_reference=reference,
            occurred_at=utc_now(),
            attempt=attempt,
        )
        _append_json_lines(self.failures_path, (record.to_dict(),))
        return record


class StructuredRunLogger:

    def __init__(
        self,
        run_directory: RunDirectory,
        *,
        console: Callable[[str], None] | None = print,
    ) -> None:
        self.path = run_directory.path / "logs" / "events.jsonl"
        self.console = console

    def log(
        self,
        level: str,
        event: str,
        message: str,
        **fields: object,
    ) -> None:

        record: dict[str, object] = {
            "timestamp": utc_now(),
            "level": level,
            "event": event,
            "message": message,
        }
        record.update(fields)
        _append_json_lines(self.path, (record,))
        if self.console is not None:
            self.console(f"[{level}] {message}")


def initial_manifest(
    config: RunConfig,
    descriptor: AdapterDescriptor,
    *,
    repository_root: str | Path | None = None,
    selected_item_ids: tuple[str, ...] | None = None,
) -> dict[str, object]:

    if config.run_id is None:  # pragma: no cover - engine resolves this
        raise RunError("manifest collection requires a concrete run ID")
    warnings = [
        "Seeds are recorded, but full determinism is not guaranteed for upstream "
        "models or accelerator kernels."
    ]
    checkpoint = fingerprint_reference(
        config.checkpoint_path, config.checkpoint_fingerprint
    )
    dataset_root = fingerprint_reference(
        config.dataset_root, config.dataset_root_fingerprint
    )
    if config.checkpoint_path is not None and checkpoint["fingerprint"] is None:
        warnings.append(
            "Checkpoint fingerprint is unavailable for the configured path."
        )
    if config.dataset_root is not None and dataset_root["fingerprint"] is None:
        warnings.append(
            "Dataset root fingerprint is unavailable for the configured path."
        )

    repo_path = Path.cwd() if repository_root is None else Path(repository_root)
    git_state = collect_git_state(repo_path)
    runtime = collect_runtime_environment()
    slurm = {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "node_list": os.environ.get("SLURM_JOB_NODELIST"),
    }
    selected_ids = (
        tuple(item.item_id for item in config.selected_items)
        if selected_item_ids is None
        else selected_item_ids
    )
    return {
        "schema_version": 1,
        "run_id": config.run_id,
        "project": {
            "name": "activevision-research-workbench",
            "version": _project_version(),
        },
        "activevision_git": git_state,
        "model": {
            "id": descriptor.identity.model_id,
            "variant": descriptor.identity.model_variant,
            "adapter_version": descriptor.identity.adapter_version,
            "upstream_version": descriptor.identity.upstream_version,
            "upstream_code_ref": config.upstream_code_ref,
            "upstream_commit": config.upstream_commit,
            "checkpoint": checkpoint,
            "capabilities": descriptor.capabilities.to_dict(),
        },
        "dataset": {
            "id": config.dataset_id,
            "version": config.dataset_version,
            "split": config.dataset_split,
            "root": dataset_root,
            "selected_item_ids": list(selected_ids),
        },
        "resolved_configuration": config.to_dict(),
        "configuration_hash": config.configuration_hash,
        "seeds": {
            "strategy": config.seed_strategy.value,
            "configured_base_seed": config.base_seed,
            "actual_per_request": {},
        },
        "environment": {
            "name": config.environment_name,
            "image_digest": config.environment_image_digest,
            **runtime,
        },
        "started_at": None,
        "ended_at": None,
        "status": RunStatus.CREATED.value,
        "counts": {
            "total": len(selected_ids),
            "succeeded": 0,
            "failed": 0,
            "skipped": 0,
        },
        "warnings": warnings,
        "resume": {
            "attempts": 0,
            "last_decision": None,
        },
        "slurm": slurm,
    }


def collect_git_state(repository_root: Path) -> dict[str, object]:

    commit = _git(repository_root, "rev-parse", "HEAD")
    if commit is None:
        return {"commit": None, "dirty": None}
    status = _git(repository_root, "status", "--porcelain")
    return {
        "commit": commit,
        "dirty": None if status is None else bool(status),
    }


def collect_runtime_environment() -> dict[str, object]:

    try:
        pytorch_version = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        pytorch_version = None
    return {
        "python_version": platform.python_version(),
        "python_executable": os.path.realpath(os.sys.executable),
        "pytorch_version": pytorch_version,
        "cuda_version": os.environ.get("CUDA_VERSION"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
        "host": socket.gethostname(),
        "platform": platform.platform(),
    }


def fingerprint_reference(
    reference: str | None, configured_fingerprint: str | None
) -> dict[str, object]:

    if configured_fingerprint is not None:
        return {
            "path": reference,
            "fingerprint": configured_fingerprint,
            "fingerprint_method": "configured",
        }
    if reference is None or "://" in reference:
        return {
            "path": reference,
            "fingerprint": None,
            "fingerprint_method": None,
        }
    path = Path(reference)
    try:
        stat = path.stat()
    except OSError:
        return {
            "path": reference,
            "fingerprint": None,
            "fingerprint_method": "unavailable",
        }
    metadata = {
        "path": str(path.resolve(strict=False)),
        "kind": "directory" if path.is_dir() else "file",
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    canonical = json.dumps(metadata, separators=(",", ":"), sort_keys=True).encode()
    digest = hashlib.sha256(canonical).hexdigest()
    return {
        "path": reference,
        "fingerprint": f"fast-stat-sha256:{digest}",
        "fingerprint_method": "path-kind-size-mtime_ns; not a content checksum",
    }


def safe_filename(value: str) -> str:

    readable = "".join(
        char if char.isalnum() or char in "._-" else "_" for char in value
    )
    readable = readable.strip(".")[:64] or "record"
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    return f"{readable}-{digest}"


def utc_now() -> str:

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _project_version() -> str:
    try:
        return importlib.metadata.version("activevision-research-workbench")
    except importlib.metadata.PackageNotFoundError:
        return "0.1.0"


def _git(repository_root: Path, *arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ("git", "-C", str(repository_root), *arguments),
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _atomic_write_json(path: Path, value: object) -> None:
    try:
        text = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise SerializationError(f"value is not strict JSON: {error}") from error
    _atomic_write_text(path, f"{text}\n")


def _atomic_write_text(path: Path, text: str) -> None:
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(text)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass


def _create_empty_file(path: Path) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _append_json_lines(path: Path, records: Iterable[object]) -> None:
    try:
        payload = "".join(
            json.dumps(
                record,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
            for record in records
        )
    except (TypeError, ValueError) as error:
        raise SerializationError(f"value is not strict JSON: {error}") from error
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        _lock(stream)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _lock(stream: object) -> None:
    if fcntl is not None:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)  # type: ignore[attr-defined]


def _recover_torn_final_line(
    path: Path, loader: Callable[[str], object]
) -> bool:
    data = path.read_bytes()
    if not data or data.endswith(b"\n"):
        return False
    last_newline = data.rfind(b"\n")
    prefix_end = last_newline + 1
    suffix = data[prefix_end:]
    try:
        loader(suffix.decode("utf-8"))
    except Exception:
        with path.open("r+b") as stream:
            _lock(stream)
            stream.truncate(prefix_end)
            stream.flush()
            os.fsync(stream.fileno())
        return True
    return False


def _reject_json_constant(value: str) -> None:
    raise SerializationError(f"non-standard JSON numeric constant '{value}'")


def _load_failure_line(text: str) -> FailureRecord:
    try:
        value = json.loads(text, parse_constant=_reject_json_constant)
    except json.JSONDecodeError as error:
        raise SerializationError(f"invalid failure JSON: {error}") from error
    return FailureRecord.from_dict(value)
