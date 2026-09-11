
from __future__ import annotations

import random
import time
from typing import Any

from activevision_workbench.adapters.scandiff.worker_server import (
    MalformedFakeResponse,
    serve,
)


class FakeScanDiffBackend:

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def predict(
        self, request: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        behavior = self.config["fake_behavior"]
        if behavior == "timeout":
            time.sleep(60)
        if behavior == "malformed":
            raise MalformedFakeResponse()
        if behavior == "error":
            raise RuntimeError("deliberate fake ScanDiff worker failure")
        samples: list[dict[str, Any]] = []
        for spec in request["sample_specs"]:
            random_source = random.Random(spec["seed"])
            length = (spec["seed"] % 4) + 1
            fixations = [
                {
                    "x_norm": random_source.random(),
                    "y_norm": random_source.random(),
                    "duration_s": 0.08 + random_source.random() * 0.2,
                }
                for _ in range(length)
            ]
            samples.append(
                {
                    "sample_id": spec["sample_id"],
                    "sample_index": spec["sample_index"],
                    "seed": spec["seed"],
                    "fixations": fixations,
                    "stopping_reason": "model_stop",
                    "warnings": [],
                }
            )
        metadata = {
            "checkpoint_sha256": self.config["checkpoint_sha256"],
            "task_embeddings_sha256": self.config["task_embeddings_sha256"],
            "feature_sha256": None,
            "config_sha256": self.config["config_sha256"],
            "image_width": request["image_width"],
            "image_height": request["image_height"],
            "upstream_commit": self.config["upstream_commit"],
        }
        return samples, metadata


def main() -> int:

    return serve(FakeScanDiffBackend)


if __name__ == "__main__":
    raise SystemExit(main())
