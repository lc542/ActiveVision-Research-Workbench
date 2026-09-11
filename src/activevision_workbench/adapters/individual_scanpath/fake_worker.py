
from __future__ import annotations

import random
import time
from typing import Any

from activevision_workbench.adapters.individual_scanpath.worker_server import (
    MalformedFakeResponse,
    serve,
)


class FakeIndividualScanpathBackend:
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
            raise RuntimeError("deliberate fake IndividualScanpath error")
        samples = [self._sample(request, spec) for spec in request["sample_specs"]]
        return samples, {
            "checkpoint_sha256": self.config["checkpoint_sha256"],
            "observer_mapping_sha256": self.config["observer_mapping_sha256"],
            "config_sha256": self.config["config_sha256"],
            "upstream_commit": self.config["upstream_commit"],
            "variant": self.config["variant"],
            "image_width": request["image_width"],
            "image_height": request["image_height"],
            "model_input_width": self.config["input_width"],
            "model_input_height": self.config["input_height"],
            "map_width": self.config["map_width"],
            "map_height": self.config["map_height"],
            "image_cache_scope": "decoded-and-preprocessed-image-only; no observer-conditioned outputs",
        }

    def _sample(self, request: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
        observer_index = request["model_observer_index"]
        rng = random.Random(spec["seed"] + observer_index * 100003)
        length = min(request["max_fixations"], 1 + (spec["seed"] + observer_index) % 4)
        events = []
        actions = []
        durations = []
        probabilities = []
        for step in range(length):
            map_x = (observer_index * 3 + step * 5 + spec["seed"]) % self.config["map_width"]
            map_y = (observer_index * 7 + step * 2 + spec["seed"]) % self.config["map_height"]
            action = map_y * self.config["map_width"] + map_x + 1
            duration = 0.1 + rng.random() * 0.2
            probability = 0.2 + rng.random() * 0.7
            actions.append(action)
            durations.append(duration)
            probabilities.append(probability)
            events.append(
                {
                    "x_model_px": (map_x + 0.5) * self.config["input_width"] / self.config["map_width"],
                    "y_model_px": (map_y + 0.5) * self.config["input_height"] / self.config["map_height"],
                    "duration_s": duration,
                    "action_index": action,
                    "action_probability": probability,
                }
            )
        stopped = length < request["max_fixations"]
        if stopped:
            actions.append(0)
            durations.append(0.05)
            probabilities.append(0.5)
        return {
            "sample_id": spec["sample_id"],
            "sample_index": spec["sample_index"],
            "seed": spec["seed"],
            "observer_id": request["observer_id"],
            "model_observer_index": observer_index,
            "events": events,
            "native_sequence": {
                "selected_actions": actions,
                "duration_samples_s": durations,
                "selected_action_probabilities": probabilities,
                "termination_step": length if stopped else None,
            },
            "stopping_reason": "model_stop" if stopped else "max_fixations",
            "warnings": [],
        }


def main() -> int:
    return serve(FakeIndividualScanpathBackend)


if __name__ == "__main__":
    raise SystemExit(main())
