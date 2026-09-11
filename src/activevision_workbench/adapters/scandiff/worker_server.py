
from __future__ import annotations

import argparse
import json
import sys
import traceback
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol

from activevision_workbench.adapters.scandiff.config import (
    SCANDIFF_NATIVE_ARTIFACT_POLICY,
)
from activevision_workbench.adapters.scandiff.protocol import (
    PROTOCOL_VERSION,
    atomic_write_json,
)


class WorkerBackend(Protocol):

    def predict(
        self, request: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        pass


class MalformedFakeResponse(Exception):
    pass


def serve(factory: Callable[[dict[str, Any]], WorkerBackend]) -> int:

    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--ready", required=True)
    arguments = parser.parse_args()
    config_path = Path(arguments.config)
    ready_path = Path(arguments.ready)
    config = _read_object(config_path, "worker config")
    _validate_config(config)
    backend = factory(config)
    atomic_write_json(
        ready_path,
        {"schema_version": PROTOCOL_VERSION, "status": "ready"},
    )
    for line in sys.stdin:
        if not line.strip():
            continue
        response_path: Path | None = None
        request_id = "unknown"
        try:
            command = json.loads(line)
            if not isinstance(command, dict) or set(command) != {
                "request_path",
                "response_path",
            }:
                raise ValueError("command must contain request_path and response_path")
            response_path = Path(_text(command["response_path"], "response_path"))
            request_path = Path(_text(command["request_path"], "request_path"))
            request = _read_object(request_path, "worker request")
            _validate_request(request)
            request_id = request["request_id"]
            samples, metadata = backend.predict(request)
            native_directory = Path(request["native_output_dir"]).resolve()
            native_directory.mkdir(parents=True, exist_ok=True)
            response_samples: list[dict[str, Any]] = []
            for raw_sample in samples:
                index = raw_sample.get("sample_index")
                if type(index) is not int or index < 0:
                    raise ValueError(
                        "backend sample_index must be a non-negative integer"
                    )
                native_path = native_directory / f"sample-{index:04d}.json"
                native_document = {
                    "schema_version": PROTOCOL_VERSION,
                    "request_id": request_id,
                    "model": "scandiff",
                    "upstream_commit": config["upstream_commit"],
                    "native_artifact_policy": config[
                        "native_artifact_policy"
                    ],
                    "sample": raw_sample,
                }
                atomic_write_json(native_path, native_document)
                sample = dict(raw_sample)
                sample["native_artifact"] = str(native_path)
                response_samples.append(sample)
            atomic_write_json(
                response_path,
                {
                    "schema_version": PROTOCOL_VERSION,
                    "status": "ok",
                    "request_id": request_id,
                    "samples": response_samples,
                    "metadata": metadata,
                },
            )
        except MalformedFakeResponse:
            assert response_path is not None
            atomic_write_json(response_path, {"status": "deliberately-malformed"})
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            if response_path is not None:
                atomic_write_json(
                    response_path,
                    {
                        "schema_version": PROTOCOL_VERSION,
                        "status": "error",
                        "request_id": request_id,
                        "error": {
                            "type": (
                                f"{type(error).__module__}."
                                f"{type(error).__qualname__}"
                            ),
                            "message": str(error),
                        },
                    },
                )
    return 0


def _read_object(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or any(type(key) is not str for key in value):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _validate_config(config: dict[str, Any]) -> None:
    expected = {
        "schema_version",
        "variant",
        "upstream_root",
        "upstream_commit",
        "checkpoint_path",
        "checkpoint_sha256",
        "task_embeddings_path",
        "task_embeddings_sha256",
        "feature_root",
        "device",
        "max_fixations",
        "allow_feature_download",
        "native_artifact_policy",
        "config_sha256",
        "fake_behavior",
    }
    _keys(config, expected, "worker config")
    if config["schema_version"] != PROTOCOL_VERSION:
        raise ValueError("worker config schema_version must be 1")
    for name in (
        "variant",
        "upstream_root",
        "upstream_commit",
        "checkpoint_path",
        "checkpoint_sha256",
        "task_embeddings_path",
        "task_embeddings_sha256",
        "device",
        "native_artifact_policy",
        "config_sha256",
        "fake_behavior",
    ):
        _text(config[name], f"worker config.{name}")
    if config["feature_root"] is not None:
        _text(config["feature_root"], "worker config.feature_root")
    if type(config["max_fixations"]) is not int or config["max_fixations"] < 1:
        raise ValueError("worker config.max_fixations must be a positive integer")
    if type(config["allow_feature_download"]) is not bool:
        raise ValueError("worker config.allow_feature_download must be a boolean")
    if config["native_artifact_policy"] != SCANDIFF_NATIVE_ARTIFACT_POLICY:
        raise ValueError(
            "worker config.native_artifact_policy must be "
            f"{SCANDIFF_NATIVE_ARTIFACT_POLICY!r}"
        )


def _validate_request(request: dict[str, Any]) -> None:
    expected = {
        "schema_version",
        "request_id",
        "image_ref",
        "image_width",
        "image_height",
        "viewing_task",
        "sample_specs",
        "max_fixations",
        "native_output_dir",
    }
    _keys(request, expected, "worker request")
    if request["schema_version"] != PROTOCOL_VERSION:
        raise ValueError("worker request schema_version must be 1")
    for name in ("request_id", "image_ref", "native_output_dir"):
        _text(request[name], f"worker request.{name}")
    if type(request["viewing_task"]) is not str:
        raise ValueError("worker request.viewing_task must be a string")
    for name in ("image_width", "image_height", "max_fixations"):
        if type(request[name]) is not int or request[name] < 1:
            raise ValueError(f"worker request.{name} must be a positive integer")
    specs = request["sample_specs"]
    if not isinstance(specs, list) or not specs:
        raise ValueError("worker request.sample_specs must be a non-empty array")
    for index, raw_spec in enumerate(specs):
        if not isinstance(raw_spec, dict):
            raise ValueError(f"sample_specs[{index}] must be an object")
        _keys(raw_spec, {"sample_id", "sample_index", "seed"}, f"sample_specs[{index}]")
        _text(raw_spec["sample_id"], f"sample_specs[{index}].sample_id")
        for name in ("sample_index", "seed"):
            if type(raw_spec[name]) is not int or raw_spec[name] < 0:
                raise ValueError(
                    f"sample_specs[{index}].{name} must be a non-negative integer"
                )


def _keys(value: Mapping[str, object], expected: set[str], label: str) -> None:
    if set(value) != expected:
        missing = expected - set(value)
        unknown = set(value) - expected
        if missing:
            raise ValueError(f"{label} missing field {sorted(missing)[0]!r}")
        raise ValueError(f"{label} has unknown field {sorted(unknown)[0]!r}")


def _text(value: object, label: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value
