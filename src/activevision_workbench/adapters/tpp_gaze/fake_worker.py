
from __future__ import annotations

import random
import time
from typing import Any

from activevision_workbench.adapters.tpp_gaze.worker_server import (
    MalformedFakeResponse,
    serve,
)


class FakeTPPGazeBackend:

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
            raise RuntimeError("deliberate fake TPP-Gaze worker failure")
        samples = [self._sample(spec, request, behavior) for spec in request["sample_specs"]]
        configured_temperature = self.config["temperature"]
        metadata = {
            "checkpoint_sha256": self.config["checkpoint_sha256"],
            "model_config_sha256": self.config["model_config_sha256"],
            "config_sha256": self.config["config_sha256"],
            "image_width": request["image_width"],
            "image_height": request["image_height"],
            "upstream_commit": self.config["upstream_commit"],
            "context_type": "transformer",
            "temperature": (
                2.0 if configured_temperature is None else configured_temperature
            ),
            "temperature_source": (
                "model_config"
                if configured_temperature is None
                else "adapter_override"
            ),
        }
        return samples, metadata

    def _sample(
        self,
        spec: dict[str, Any],
        request: dict[str, Any],
        behavior: str,
    ) -> dict[str, Any]:
        scale = self.config["temporal_scale_to_seconds"]
        native_per_second = 1.0 / scale
        rng = random.Random(spec["seed"])
        desired_length = (spec["seed"] % 4) + 1
        events: list[dict[str, Any]] = []
        arrival_s = 0.0
        stopping_reason = "model_stop"
        if behavior == "absolute_only":
            events.append(
                self._event(
                    rng,
                    request,
                    inter_s=None,
                    arrival_s=0.1,
                    included=True,
                )
            )
        elif behavior in {"decreasing_time", "negative_time"}:
            first = self._event(
                rng, request, inter_s=0.1, arrival_s=0.1, included=True
            )
            second_inter = -0.05 if behavior == "negative_time" else 0.1
            second_arrival = 0.05 if behavior == "decreasing_time" else 0.05
            events.extend(
                [
                    first,
                    self._event(
                        rng,
                        request,
                        inter_s=second_inter,
                        arrival_s=second_arrival,
                        included=True,
                    ),
                ]
            )
        else:
            for event_index in range(desired_length):
                inter_s = (
                    request["time_horizon_s"] + 0.25
                    if behavior == "empty" and event_index == 0
                    else 0.1 + 0.05 * ((spec["seed"] + event_index) % 3)
                )
                arrival_s += inter_s
                included = arrival_s <= request["time_horizon_s"]
                events.append(
                    self._event(
                        rng,
                        request,
                        inter_s=inter_s,
                        arrival_s=arrival_s,
                        included=included,
                    )
                )
                if arrival_s >= request["time_horizon_s"]:
                    stopping_reason = "time_horizon"
                    break
                if sum(event["included"] for event in events) >= request[
                    "max_fixations"
                ]:
                    stopping_reason = "max_fixations"
                    break
            if behavior == "empty":
                stopping_reason = "time_horizon"
        for event in events:
            if event["inter_event_time_native"] is not None:
                event["inter_event_time_native"] = (
                    event["model_inter_event_time_s"] * native_per_second
                )
            if event["arrival_time_native"] is not None:
                event["arrival_time_native"] = (
                    event["model_arrival_time_s"] * native_per_second
                )
        return {
            "sample_id": spec["sample_id"],
            "sample_index": spec["sample_index"],
            "seed": spec["seed"],
            "events": events,
            "stopping_reason": stopping_reason,
            "warnings": [],
        }

    @staticmethod
    def _event(
        rng: random.Random,
        request: dict[str, Any],
        *,
        inter_s: float | None,
        arrival_s: float | None,
        included: bool,
    ) -> dict[str, Any]:
        mark_x = rng.uniform(-1.0, 1.0)
        mark_y = rng.uniform(-1.0, 1.0)
        return {
            "x_px": ((mark_x + 1.0) / 2.0) * (request["image_width"] - 1),
            "y_px": ((mark_y + 1.0) / 2.0) * (request["image_height"] - 1),
            "mark_x_native": mark_x,
            "mark_y_native": mark_y,
            "inter_event_time_native": inter_s,
            "arrival_time_native": arrival_s,
            "model_inter_event_time_s": inter_s,
            "model_arrival_time_s": arrival_s,
            "included": included,
        }


def main() -> int:

    return serve(FakeTPPGazeBackend)


if __name__ == "__main__":
    raise SystemExit(main())
