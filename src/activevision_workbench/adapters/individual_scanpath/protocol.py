
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from activevision_workbench.errors import AdapterOutputError

PROTOCOL_VERSION = 1


def atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = json.dumps(
        value, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def worker_error(value: object) -> str | None:
    if not isinstance(value, dict) or value.get("status") != "error":
        return None
    error = value.get("error")
    if not isinstance(error, dict):
        return "worker reported an error without a valid error object"
    return f"{error.get('type', 'WorkerError')}: {error.get('message', 'no message')}"


def validate_response(
    value: object, *, request_id: str, sample_count: int
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    root = _object(value, "IndividualScanpathWorkerResponse")
    _keys(
        root,
        {"schema_version", "status", "request_id", "samples", "metadata"},
        "IndividualScanpathWorkerResponse",
    )
    if root["schema_version"] != PROTOCOL_VERSION or root["status"] != "ok":
        raise AdapterOutputError("IndividualScanpath worker response is not successful")
    if root["request_id"] != request_id:
        raise AdapterOutputError("IndividualScanpath response request_id mismatch")
    metadata = _object(root["metadata"], "IndividualScanpathWorkerResponse.metadata")
    _keys(
        metadata,
        {
            "checkpoint_sha256",
            "observer_mapping_sha256",
            "config_sha256",
            "upstream_commit",
            "variant",
            "image_width",
            "image_height",
            "model_input_width",
            "model_input_height",
            "map_width",
            "map_height",
            "image_cache_scope",
        },
        "IndividualScanpathWorkerResponse.metadata",
    )
    for name in (
        "checkpoint_sha256",
        "observer_mapping_sha256",
        "config_sha256",
        "upstream_commit",
        "variant",
        "image_cache_scope",
    ):
        _string(metadata[name], f"metadata.{name}")
    for name in (
        "image_width",
        "image_height",
        "model_input_width",
        "model_input_height",
        "map_width",
        "map_height",
    ):
        _integer(metadata[name], f"metadata.{name}", minimum=1)
    raw_samples = root["samples"]
    if not isinstance(raw_samples, list) or len(raw_samples) != sample_count:
        raise AdapterOutputError(
            f"IndividualScanpath worker returned the wrong sample count; expected {sample_count}"
        )
    samples: list[dict[str, Any]] = []
    indices: set[int] = set()
    ids: set[str] = set()
    for position, raw in enumerate(raw_samples):
        label = f"samples[{position}]"
        sample = _object(raw, label)
        _keys(
            sample,
            {
                "sample_id",
                "sample_index",
                "seed",
                "observer_id",
                "model_observer_index",
                "events",
                "native_sequence",
                "stopping_reason",
                "warnings",
                "native_artifact",
            },
            label,
        )
        sample_id = _string(sample["sample_id"], f"{label}.sample_id")
        index = _integer(sample["sample_index"], f"{label}.sample_index", minimum=0)
        _integer(sample["seed"], f"{label}.seed", minimum=0)
        _string(sample["observer_id"], f"{label}.observer_id")
        _integer(
            sample["model_observer_index"],
            f"{label}.model_observer_index",
            minimum=0,
        )
        if index in indices or sample_id in ids:
            raise AdapterOutputError("IndividualScanpath sample IDs/indices are duplicated")
        indices.add(index)
        ids.add(sample_id)
        events = sample["events"]
        if not isinstance(events, list):
            raise AdapterOutputError(f"{label}.events must be an array")
        for event_index, event in enumerate(events):
            _event(event, f"{label}.events[{event_index}]")
        sequence = _object(sample["native_sequence"], f"{label}.native_sequence")
        _keys(
            sequence,
            {
                "selected_actions",
                "duration_samples_s",
                "selected_action_probabilities",
                "termination_step",
            },
            f"{label}.native_sequence",
        )
        lengths = []
        for name in (
            "selected_actions",
            "duration_samples_s",
            "selected_action_probabilities",
        ):
            values = sequence[name]
            if not isinstance(values, list):
                raise AdapterOutputError(f"{label}.native_sequence.{name} must be an array")
            for item in values:
                _number(item, f"{label}.native_sequence.{name}")
            lengths.append(len(values))
        if len(set(lengths)) != 1:
            raise AdapterOutputError(f"{label}.native_sequence arrays must align")
        termination = sequence["termination_step"]
        if termination is not None:
            _integer(termination, f"{label}.termination_step", minimum=0)
        if sample["stopping_reason"] not in {"model_stop", "max_fixations"}:
            raise AdapterOutputError(f"{label}.stopping_reason is invalid")
        if not isinstance(sample["warnings"], list):
            raise AdapterOutputError(f"{label}.warnings must be an array")
        for warning in sample["warnings"]:
            _string(warning, f"{label}.warnings")
        _string(sample["native_artifact"], f"{label}.native_artifact")
        samples.append(sample)
    if indices != set(range(sample_count)):
        raise AdapterOutputError("IndividualScanpath sample indices are not contiguous")
    samples.sort(key=lambda sample: sample["sample_index"])
    return tuple(samples), metadata


def _event(value: object, label: str) -> None:
    event = _object(value, label)
    _keys(
        event,
        {
            "x_model_px",
            "y_model_px",
            "duration_s",
            "action_index",
            "action_probability",
        },
        label,
    )
    for name in ("x_model_px", "y_model_px", "duration_s", "action_probability"):
        _number(event[name], f"{label}.{name}")
    _integer(event["action_index"], f"{label}.action_index", minimum=1)


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(type(key) is not str for key in value):
        raise AdapterOutputError(f"{label} must be an object")
    return value


def _keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    missing = expected - set(value)
    unknown = set(value) - expected
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
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise AdapterOutputError(f"{label} must be a finite number")
    return float(value)


def _reject_constant(value: str, label: str) -> None:
    raise ValueError(f"{label} contains non-standard numeric constant {value}")
