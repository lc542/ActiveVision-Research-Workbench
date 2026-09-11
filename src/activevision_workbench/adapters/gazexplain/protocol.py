
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
    root = _object(value, "GazeXplainWorkerResponse")
    _keys(
        root,
        {"schema_version", "status", "request_id", "samples", "metadata"},
        "GazeXplainWorkerResponse",
    )
    if root["schema_version"] != PROTOCOL_VERSION or root["status"] != "ok":
        raise AdapterOutputError("GazeXplain worker response is not successful")
    if root["request_id"] != request_id:
        raise AdapterOutputError("GazeXplain response request_id mismatch")
    metadata = _metadata(root["metadata"])
    raw_samples = root["samples"]
    if not isinstance(raw_samples, list) or len(raw_samples) != sample_count:
        raise AdapterOutputError(
            "GazeXplain worker returned the wrong sample count; "
            f"expected {sample_count}"
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
                "events",
                "explanation",
                "native_sequence",
                "stopping_reason",
                "warnings",
                "native_artifact",
            },
            label,
        )
        sample_id = _text(sample["sample_id"], f"{label}.sample_id")
        index = _integer(sample["sample_index"], f"{label}.sample_index", minimum=0)
        _integer(sample["seed"], f"{label}.seed", minimum=0)
        if index in indices or sample_id in ids:
            raise AdapterOutputError("GazeXplain sample IDs/indices are duplicated")
        indices.add(index)
        ids.add(sample_id)
        events = sample["events"]
        if not isinstance(events, list):
            raise AdapterOutputError(f"{label}.events must be an array")
        for event_index, event in enumerate(events):
            _event(event, f"{label}.events[{event_index}]")
        _explanation(sample["explanation"], f"{label}.explanation", len(events))
        _native_sequence(sample["native_sequence"], f"{label}.native_sequence")
        if len(sample["explanation"]["raw_token_ids_all_steps"]) != len(
            sample["native_sequence"]["selected_actions"]
        ):
            raise AdapterOutputError(
                f"{label} complete token output must align with the native sequence"
            )
        if sample["stopping_reason"] not in {"model_stop", "max_fixations"}:
            raise AdapterOutputError(f"{label}.stopping_reason is invalid")
        warnings = sample["warnings"]
        if not isinstance(warnings, list):
            raise AdapterOutputError(f"{label}.warnings must be an array")
        for warning_index, warning in enumerate(warnings):
            _text(warning, f"{label}.warnings[{warning_index}]")
        _text(sample["native_artifact"], f"{label}.native_artifact")
        samples.append(sample)
    if indices != set(range(sample_count)):
        raise AdapterOutputError("GazeXplain sample indices are not contiguous")
    samples.sort(key=lambda sample: sample["sample_index"])
    return tuple(samples), metadata


def _metadata(value: object) -> dict[str, Any]:
    label = "GazeXplainWorkerResponse.metadata"
    metadata = _object(value, label)
    _keys(
        metadata,
        {
            "checkpoint_sha256",
            "hparams_sha256",
            "config_sha256",
            "upstream_commit",
            "variant",
            "image_width",
            "image_height",
            "model_input_width",
            "model_input_height",
            "feature_input_width",
            "feature_input_height",
            "map_width",
            "map_height",
            "resolved_task_text",
            "task_input_token_count",
            "task_input_truncated",
            "task_tokenizer",
            "explanation_tokenizer",
            "feature_backbone",
            "decoding",
            "image_cache_scope",
        },
        label,
    )
    for name in (
        "checkpoint_sha256",
        "hparams_sha256",
        "config_sha256",
        "upstream_commit",
        "variant",
        "resolved_task_text",
        "image_cache_scope",
    ):
        _text(metadata[name], f"metadata.{name}")
    for name in (
        "image_width",
        "image_height",
        "model_input_width",
        "model_input_height",
        "feature_input_width",
        "feature_input_height",
        "map_width",
        "map_height",
        "task_input_token_count",
    ):
        _integer(metadata[name], f"metadata.{name}", minimum=1)
    if type(metadata["task_input_truncated"]) is not bool:
        raise AdapterOutputError("metadata.task_input_truncated must be a boolean")
    _version(metadata["task_tokenizer"], "metadata.task_tokenizer")
    _version(metadata["explanation_tokenizer"], "metadata.explanation_tokenizer")
    backbone = _object(metadata["feature_backbone"], "metadata.feature_backbone")
    _keys(
        backbone,
        {"model_id", "sha256"},
        "metadata.feature_backbone",
    )
    _text(backbone["model_id"], "metadata.feature_backbone.model_id")
    _text(backbone["sha256"], "metadata.feature_backbone.sha256")
    decoding = _object(metadata["decoding"], "metadata.decoding")
    _keys(
        decoding,
        {"max_generation_length", "num_explanation_beams", "early_stopping"},
        "metadata.decoding",
    )
    _integer(
        decoding["max_generation_length"],
        "metadata.decoding.max_generation_length",
        minimum=3,
    )
    _integer(
        decoding["num_explanation_beams"],
        "metadata.decoding.num_explanation_beams",
        minimum=1,
    )
    if decoding["early_stopping"] is not True:
        raise AdapterOutputError("metadata.decoding.early_stopping must be true")
    return metadata


def _event(value: object, label: str) -> None:
    event = _object(value, label)
    _keys(
        event,
        {
            "x_model_px",
            "y_model_px",
            "duration_ms",
            "action_index",
            "action_probability",
            "explanation_text",
            "explanation_token_ids",
            "explanation_truncated",
        },
        label,
    )
    for name in (
        "x_model_px",
        "y_model_px",
        "duration_ms",
        "action_probability",
    ):
        _number(event[name], f"{label}.{name}")
    _integer(event["action_index"], f"{label}.action_index", minimum=1)
    if event["explanation_text"] is not None:
        _unicode_text(event["explanation_text"], f"{label}.explanation_text")
    _token_ids(event["explanation_token_ids"], f"{label}.explanation_token_ids")
    if type(event["explanation_truncated"]) is not bool:
        raise AdapterOutputError(f"{label}.explanation_truncated must be a boolean")


def _explanation(value: object, label: str, event_count: int) -> None:
    explanation = _object(value, label)
    _keys(
        explanation,
        {
            "text",
            "per_fixation",
            "raw_token_ids",
            "raw_token_ids_all_steps",
            "truncated",
            "token_scores",
        },
        label,
    )
    if explanation["text"] is not None:
        _unicode_text(explanation["text"], f"{label}.text")
    per_fixation = explanation["per_fixation"]
    raw_ids = explanation["raw_token_ids"]
    if not isinstance(per_fixation, list) or len(per_fixation) != event_count:
        raise AdapterOutputError(f"{label}.per_fixation must align with events")
    if not isinstance(raw_ids, list) or len(raw_ids) != event_count:
        raise AdapterOutputError(f"{label}.raw_token_ids must align with events")
    all_raw_ids = explanation["raw_token_ids_all_steps"]
    if not isinstance(all_raw_ids, list):
        raise AdapterOutputError(f"{label}.raw_token_ids_all_steps must be an array")
    for index, token_ids in enumerate(all_raw_ids):
        _token_ids(token_ids, f"{label}.raw_token_ids_all_steps[{index}]")
    for index, alignment in enumerate(per_fixation):
        item = _object(alignment, f"{label}.per_fixation[{index}]")
        _keys(
            item,
            {"sequence_index", "text", "token_ids", "truncated"},
            f"{label}.per_fixation[{index}]",
        )
        if item["sequence_index"] != index:
            raise AdapterOutputError(
                f"{label}.per_fixation[{index}].sequence_index must equal {index}"
            )
        if item["text"] is not None:
            _unicode_text(item["text"], f"{label}.per_fixation[{index}].text")
        _token_ids(item["token_ids"], f"{label}.per_fixation[{index}].token_ids")
        if item["token_ids"] != raw_ids[index]:
            raise AdapterOutputError(f"{label} raw token arrays do not align")
        if type(item["truncated"]) is not bool:
            raise AdapterOutputError(
                f"{label}.per_fixation[{index}].truncated must be a boolean"
            )
    if type(explanation["truncated"]) is not bool:
        raise AdapterOutputError(f"{label}.truncated must be a boolean")
    if explanation["token_scores"] is not None:
        raise AdapterOutputError(
            f"{label}.token_scores must be null for the verified detail=False inference path"
        )


def _native_sequence(value: object, label: str) -> None:
    sequence = _object(value, label)
    _keys(
        sequence,
        {
            "selected_actions",
            "duration_samples_s",
            "selected_action_probabilities",
            "duration_mask",
            "termination_step",
        },
        label,
    )
    lengths: list[int] = []
    for name in (
        "selected_actions",
        "duration_samples_s",
        "selected_action_probabilities",
        "duration_mask",
    ):
        values = sequence[name]
        if not isinstance(values, list):
            raise AdapterOutputError(f"{label}.{name} must be an array")
        for item in values:
            _number(item, f"{label}.{name}")
        lengths.append(len(values))
    if len(set(lengths)) != 1:
        raise AdapterOutputError(f"{label} arrays must align")
    termination = sequence["termination_step"]
    if termination is not None:
        _integer(termination, f"{label}.termination_step", minimum=0)


def _version(value: object, label: str) -> None:
    version = _object(value, label)
    _keys(version, {"model_id", "revision"}, label)
    _text(version["model_id"], f"{label}.model_id")
    _text(version["revision"], f"{label}.revision")


def _token_ids(value: object, label: str) -> None:
    if not isinstance(value, list):
        raise AdapterOutputError(f"{label} must be an array")
    for index, token in enumerate(value):
        _integer(token, f"{label}[{index}]", minimum=0)


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


def _text(value: object, label: str) -> str:
    result = _unicode_text(value, label)
    if not result:
        raise AdapterOutputError(f"{label} must be a non-empty string")
    return result


def _unicode_text(value: object, label: str) -> str:
    if type(value) is not str:
        raise AdapterOutputError(f"{label} must be a string or null")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise AdapterOutputError(f"{label} contains invalid Unicode") from error
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
