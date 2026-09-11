
from __future__ import annotations

import argparse
import json
import sys
import traceback
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol

from activevision_workbench.adapters.gazexplain.config import (
    GAZEXPLAIN_JOINT_VARIANT,
    GAZEXPLAIN_NATIVE_ARTIFACT_POLICY,
)
from activevision_workbench.adapters.gazexplain.protocol import (
    PROTOCOL_VERSION,
    atomic_write_json,
)


class WorkerBackend(Protocol):
    def predict(
        self, request: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]: ...


class MalformedFakeResponse(Exception):
    pass


def serve(factory: Callable[[dict[str, Any]], WorkerBackend]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--ready", required=True)
    args = parser.parse_args()
    config = _read(Path(args.config), "worker config")
    _validate_config(config)
    backend = factory(config)
    atomic_write_json(
        Path(args.ready), {"schema_version": PROTOCOL_VERSION, "status": "ready"}
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
                raise ValueError(
                    "command must contain request_path and response_path"
                )
            response_path = Path(_text(command["response_path"], "response_path"))
            request = _read(
                Path(_text(command["request_path"], "request_path")),
                "worker request",
            )
            _validate_request(request, config)
            request_id = request["request_id"]
            samples, metadata = backend.predict(request)
            native_root = Path(request["native_output_dir"]).resolve()
            native_root.mkdir(parents=True, exist_ok=True)
            response_samples = []
            for raw in samples:
                index = raw.get("sample_index")
                if type(index) is not int or index < 0:
                    raise ValueError("backend returned invalid sample_index")
                native_path = native_root / f"sample-{index:04d}.json"
                atomic_write_json(
                    native_path,
                    {
                        "schema_version": PROTOCOL_VERSION,
                        "request_id": request_id,
                        "model": "gazexplain",
                        "variant": config["variant"],
                        "upstream_commit": config["upstream_commit"],
                        "checkpoint_sha256": config["checkpoint_sha256"],
                        "hparams_sha256": config["hparams_sha256"],
                        "resolved_task_text": request["resolved_task_text"],
                        "target_description": request["target_description"],
                        "decoding": {
                            "max_generation_length": config[
                                "max_generation_length"
                            ],
                            "num_explanation_beams": config[
                                "num_explanation_beams"
                            ],
                            "early_stopping": True,
                        },
                        "native_artifact_policy": config[
                            "native_artifact_policy"
                        ],
                        "sample": raw,
                    },
                )
                sample = dict(raw)
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
            atomic_write_json(response_path, {"status": "malformed"})
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


def _validate_config(config: dict[str, Any]) -> None:
    expected = {
        "schema_version",
        "variant",
        "upstream_root",
        "upstream_commit",
        "checkpoint_path",
        "checkpoint_sha256",
        "hparams_path",
        "hparams_sha256",
        "device",
        "model_assets_root",
        "model_max_fixations",
        "max_generation_length",
        "num_explanation_beams",
        "model_input_width",
        "model_input_height",
        "feature_input_width",
        "feature_input_height",
        "map_width",
        "map_height",
        "roberta_revision",
        "blip_revision",
        "feature_backbone_sha256",
        "native_artifact_policy",
        "config_sha256",
        "fake_behavior",
    }
    _keys(config, expected, "worker config")
    if config["schema_version"] != PROTOCOL_VERSION:
        raise ValueError("worker config schema_version must equal 1")
    if config["variant"] != GAZEXPLAIN_JOINT_VARIANT:
        raise ValueError("worker config variant is unsupported")
    for name in (
        "upstream_root",
        "upstream_commit",
        "checkpoint_path",
        "checkpoint_sha256",
        "hparams_path",
        "hparams_sha256",
        "device",
        "model_assets_root",
        "roberta_revision",
        "blip_revision",
        "feature_backbone_sha256",
        "native_artifact_policy",
        "config_sha256",
        "fake_behavior",
    ):
        _text(config[name], f"worker config.{name}")
    if config["native_artifact_policy"] != GAZEXPLAIN_NATIVE_ARTIFACT_POLICY:
        raise ValueError("worker native artifact policy is invalid")
    for name in (
        "model_max_fixations",
        "max_generation_length",
        "num_explanation_beams",
        "model_input_width",
        "model_input_height",
        "feature_input_width",
        "feature_input_height",
        "map_width",
        "map_height",
    ):
        if type(config[name]) is not int or config[name] < 1:
            raise ValueError(
                f"worker config.{name} must be a positive integer"
            )


def _validate_request(request: dict[str, Any], config: dict[str, Any]) -> None:
    expected = {
        "schema_version",
        "request_id",
        "image_ref",
        "image_width",
        "image_height",
        "resolved_task_text",
        "target_description",
        "sample_specs",
        "max_fixations",
        "native_output_dir",
    }
    _keys(request, expected, "worker request")
    if request["schema_version"] != PROTOCOL_VERSION:
        raise ValueError("worker request schema_version must equal 1")
    for name in (
        "request_id",
        "image_ref",
        "resolved_task_text",
        "native_output_dir",
    ):
        _text(request[name], f"worker request.{name}")
    target = request["target_description"]
    if target is not None and type(target) is not str:
        raise ValueError("worker request.target_description must be a string or null")
    for name in ("image_width", "image_height", "max_fixations"):
        if type(request[name]) is not int or request[name] < 1:
            raise ValueError(f"worker request.{name} is invalid")
    if request["max_fixations"] > config["model_max_fixations"]:
        raise ValueError("worker request max_fixations is out of range")
    specs = request["sample_specs"]
    if not isinstance(specs, list) or not specs:
        raise ValueError("worker request sample_specs must be a non-empty array")
    for index, spec in enumerate(specs):
        if not isinstance(spec, dict):
            raise ValueError(f"sample_specs[{index}] must be an object")
        _keys(
            spec,
            {"sample_id", "sample_index", "seed"},
            f"sample_specs[{index}]",
        )
        _text(spec["sample_id"], f"sample_specs[{index}].sample_id")
        if type(spec["sample_index"]) is not int or type(spec["seed"]) is not int:
            raise ValueError(
                f"sample_specs[{index}] index and seed must be integers"
            )


def _read(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _keys(value: Mapping[str, object], expected: set[str], label: str) -> None:
    if set(value) != expected:
        missing = expected - set(value)
        unknown = set(value) - expected
        detail = (
            f"missing field '{sorted(missing)[0]}'"
            if missing
            else f"unknown field '{sorted(unknown)[0]}'"
        )
        raise ValueError(f"{label} {detail}")


def _text(value: object, label: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def main() -> int:
    raise RuntimeError("worker_server requires a backend factory")
