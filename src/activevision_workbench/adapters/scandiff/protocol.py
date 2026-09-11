
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from activevision_workbench.errors import AdapterOutputError

PROTOCOL_VERSION = 1


def atomic_write_json(path: Path, value: object) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    try:
        temporary.write_text(f"{payload}\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_json(path: Path, *, label: str) -> object:

    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda value: _reject_constant(value, label),
        )
    except (OSError, ValueError) as error:
        raise AdapterOutputError(f"cannot read {label} '{path}': {error}") from error


def validate_response(
    value: object,
    *,
    request_id: str,
    sample_count: int,
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:

    root = _object(value, "ScanDiffWorkerResponse")
    _exact_keys(
        root,
        required={"schema_version", "status", "request_id", "samples", "metadata"},
        label="ScanDiffWorkerResponse",
    )
    if root["schema_version"] != PROTOCOL_VERSION:
        raise AdapterOutputError(
            "ScanDiffWorkerResponse.schema_version: expected 1"
        )
    if root["request_id"] != request_id:
        raise AdapterOutputError(
            "ScanDiffWorkerResponse.request_id does not match the request"
        )
    if root["status"] != "ok":
        raise AdapterOutputError("ScanDiffWorkerResponse.status: expected 'ok'")
    metadata = _object(root["metadata"], "ScanDiffWorkerResponse.metadata")
    _exact_keys(
        metadata,
        required={
            "checkpoint_sha256",
            "task_embeddings_sha256",
            "feature_sha256",
            "config_sha256",
            "image_width",
            "image_height",
            "upstream_commit",
        },
        label="ScanDiffWorkerResponse.metadata",
    )
    for name in (
        "checkpoint_sha256",
        "task_embeddings_sha256",
        "config_sha256",
        "upstream_commit",
    ):
        _string(metadata[name], f"ScanDiffWorkerResponse.metadata.{name}")
    if metadata["feature_sha256"] is not None:
        _string(
            metadata["feature_sha256"],
            "ScanDiffWorkerResponse.metadata.feature_sha256",
        )
    _integer(
        metadata["image_width"],
        "ScanDiffWorkerResponse.metadata.image_width",
        minimum=1,
    )
    _integer(
        metadata["image_height"],
        "ScanDiffWorkerResponse.metadata.image_height",
        minimum=1,
    )
    samples = root["samples"]
    if not isinstance(samples, list):
        raise AdapterOutputError("ScanDiffWorkerResponse.samples must be an array")
    if len(samples) != sample_count:
        raise AdapterOutputError(
            f"ScanDiff worker returned {len(samples)} samples; expected {sample_count}"
        )
    validated: list[dict[str, Any]] = []
    indices: set[int] = set()
    sample_ids: set[str] = set()
    for position, raw in enumerate(samples):
        label = f"ScanDiffWorkerResponse.samples[{position}]"
        sample = _object(raw, label)
        _exact_keys(
            sample,
            required={
                "sample_id",
                "sample_index",
                "seed",
                "fixations",
                "stopping_reason",
                "native_artifact",
                "warnings",
            },
            label=label,
        )
        sample_id = _string(sample["sample_id"], f"{label}.sample_id")
        index = _integer(sample["sample_index"], f"{label}.sample_index", minimum=0)
        _integer(sample["seed"], f"{label}.seed", minimum=0)
        if index in indices:
            raise AdapterOutputError(f"{label}.sample_index is duplicated")
        if sample_id in sample_ids:
            raise AdapterOutputError(f"{label}.sample_id is duplicated")
        indices.add(index)
        sample_ids.add(sample_id)
        fixations = sample["fixations"]
        if not isinstance(fixations, list):
            raise AdapterOutputError(f"{label}.fixations must be an array")
        for fix_index, fixation_raw in enumerate(fixations):
            fixation_label = f"{label}.fixations[{fix_index}]"
            fixation = _object(fixation_raw, fixation_label)
            _exact_keys(
                fixation,
                required={"x_norm", "y_norm", "duration_s"},
                label=fixation_label,
            )
            for name in ("x_norm", "y_norm", "duration_s"):
                _number(fixation[name], f"{fixation_label}.{name}")
        _string(sample["stopping_reason"], f"{label}.stopping_reason")
        _string(sample["native_artifact"], f"{label}.native_artifact")
        warnings = sample["warnings"]
        if not isinstance(warnings, list):
            raise AdapterOutputError(f"{label}.warnings must be an array")
        for warning_index, warning in enumerate(warnings):
            _string(warning, f"{label}.warnings[{warning_index}]")
        validated.append(sample)
    if indices != set(range(sample_count)):
        raise AdapterOutputError(
            "ScanDiffWorkerResponse sample indices must be exactly "
            f"0..{sample_count - 1}"
        )
    validated.sort(key=lambda item: item["sample_index"])
    return tuple(validated), metadata


def worker_error(value: object) -> str | None:

    if not isinstance(value, dict) or value.get("status") != "error":
        return None
    error = value.get("error")
    if not isinstance(error, dict):
        return "worker reported an error without a valid error object"
    error_type = error.get("type", "WorkerError")
    message = error.get("message", "no message")
    return f"{error_type}: {message}"


def sha256_file(path: Path) -> str:

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(type(key) is not str for key in value):
        raise AdapterOutputError(f"{label} must be an object with string keys")
    return value


def _exact_keys(value: dict[str, Any], *, required: set[str], label: str) -> None:
    missing = required - set(value)
    unknown = set(value) - required
    if missing:
        raise AdapterOutputError(f"{label}: missing field '{sorted(missing)[0]}'")
    if unknown:
        raise AdapterOutputError(f"{label}: unknown field '{sorted(unknown)[0]}'")


def _string(value: object, label: str) -> str:
    if type(value) is not str or not value:
        raise AdapterOutputError(f"{label} must be a non-empty string")
    return value


def _integer(value: object, label: str, *, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise AdapterOutputError(f"{label} must be an integer >= {minimum}")
    return value


def _number(value: object, label: str) -> float:
    if type(value) not in (int, float):
        raise AdapterOutputError(f"{label} must be a finite number")
    number = float(value)
    if not (-float("inf") < number < float("inf")):
        raise AdapterOutputError(f"{label} must be a finite number")
    return number


def _reject_constant(value: str, label: str) -> None:
    raise ValueError(f"{label} contains forbidden JSON constant {value}")
