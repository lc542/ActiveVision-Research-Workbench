
from __future__ import annotations

import math
import random
import sys
from pathlib import Path
from typing import Any

from activevision_workbench.adapters.tpp_gaze.worker_server import serve


class TPPGazeBackend:

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        upstream_root = Path(config["upstream_root"])
        sys.path.insert(0, str(upstream_root))

        import torch
        import torchvision.models

        if config["device"] == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "TPP-Gaze requested CUDA but torch.cuda.is_available() is false"
            )

        # The pinned upstream constructor asks torch.hub for torchvision v0.10
        # while passing the newer `weights=` API. The checkpoint contains every
        # DenseNet parameter, so construct the identical installed architecture
        # without network access and let strict checkpoint loading populate it.
        original_hub_load = torch.hub.load

        def local_densenet(repo_or_dir: str, model: str, *args: Any, **kwargs: Any):
            if repo_or_dir == "pytorch/vision:v0.10.0" and model == "densenet201":
                if args:
                    raise RuntimeError("unexpected positional DenseNet arguments")
                unknown = set(kwargs) - {"weights"}
                if unknown:
                    raise RuntimeError(
                        "unsupported DenseNet options: " + ", ".join(sorted(unknown))
                    )
                return torchvision.models.densenet201(weights=None)
            return original_hub_load(repo_or_dir, model, *args, **kwargs)

        torch.hub.load = local_densenet
        try:
            from tppgaze.tppgaze import TPPGaze
            from tppgaze.utils import preprocess_image

            self.model = TPPGaze(
                config["model_config_path"],
                config["checkpoint_path"],
                config["device"],
            )
            self.model.load_model()
            # Upstream stores this deterministic encoding tensor as a plain
            # attribute instead of a parameter/buffer, so nn.Module.to() does
            # not move it and official CUDA sampling fails after event one.
            encoder = self.model.context.encoder.encoder
            encoder.position_vec = encoder.position_vec.to(config["device"])
        finally:
            torch.hub.load = original_hub_load
        self.preprocess_image = preprocess_image
        self.torch = torch
        self.context_type = str(self.model.cfg.context.type)
        configured_temperature = config["temperature"]
        self.temperature_source = (
            "model_config"
            if configured_temperature is None
            else "adapter_override"
        )
        self.temperature = float(
            self.model.cfg.temperature
            if configured_temperature is None
            else configured_temperature
        )
        if self.context_type != config["variant"]:
            raise RuntimeError(
                "TPP-Gaze model config context type does not match adapter variant: "
                f"{self.context_type!r} != {config['variant']!r}"
            )

    def predict(
        self, request: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        image, height, width = self.preprocess_image(request["image_ref"])
        if width != request["image_width"] or height != request["image_height"]:
            raise ValueError(
                "decoded image dimensions do not match the canonical request: "
                f"decoded {width}x{height}, requested "
                f"{request['image_width']}x{request['image_height']}"
            )
        image = image.to(self.config["device"])
        samples = [
            self._sample_one(image, width, height, request, spec)
            for spec in request["sample_specs"]
        ]
        metadata = {
            "checkpoint_sha256": self.config["checkpoint_sha256"],
            "model_config_sha256": self.config["model_config_sha256"],
            "config_sha256": self.config["config_sha256"],
            "image_width": width,
            "image_height": height,
            "upstream_commit": self.config["upstream_commit"],
            "context_type": self.context_type,
            "temperature": self.temperature,
            "temperature_source": self.temperature_source,
        }
        return samples, metadata

    def _sample_one(
        self,
        image: Any,
        width: int,
        height: int,
        request: dict[str, Any],
        spec: dict[str, Any],
    ) -> dict[str, Any]:
        torch = self.torch
        seed = spec["seed"]
        random.seed(seed)
        self.model.set_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

        horizon_s = float(request["time_horizon_s"])
        max_fixations = int(request["max_fixations"])
        events: list[dict[str, Any]] = []
        with torch.inference_mode():
            context_init = self.model.context.get_context_init()
            history = context_init[None, None, :]
            images = image.unsqueeze(0) if image.ndim == 3 else image
            metadata = self.model.get_metadata(images)
            next_context = self.model.merge.forward([history, metadata])
            inter_times = torch.empty(1, 0, device=self.model.device)
            marks = torch.empty(
                1, 0, 2, device=self.model.device, dtype=torch.float
            )

            stopping_reason = "max_fixations"
            for _ in range(max_fixations):
                next_inter = self.model.get_inter_time_dist(next_context).sample()
                current_mark = self.model.get_marks_dist(
                    next_context, self.temperature
                ).sample()
                inter_times = torch.cat([inter_times, next_inter], dim=1)
                marks = torch.cat([marks, current_mark], dim=1)
                inter_s = float(next_inter.reshape(-1)[0].item())
                arrival_s = float(inter_times.sum(-1)[0].item())
                mark_x = float(current_mark[0, 0, 0].item())
                mark_y = float(current_mark[0, 0, 1].item())
                for name, value in {
                    "inter-event time": inter_s,
                    "arrival time": arrival_s,
                    "x mark": mark_x,
                    "y mark": mark_y,
                }.items():
                    if not math.isfinite(value):
                        raise ValueError(f"TPP-Gaze sampled non-finite {name}")
                if inter_s < 0 or arrival_s < 0:
                    raise ValueError("TPP-Gaze sampled a negative temporal value")
                included = arrival_s <= horizon_s
                events.append(
                    self._event(
                        mark_x=mark_x,
                        mark_y=mark_y,
                        inter_s=inter_s,
                        arrival_s=arrival_s,
                        included=included,
                        width=width,
                        height=height,
                    )
                )
                if arrival_s >= horizon_s:
                    stopping_reason = "time_horizon"
                    break

                features = self.model.get_features(inter_times, marks)
                history = self.model.get_context(features, remove_last=False)
                context = self.model.merge.forward([history, metadata])
                next_context = context[:, [-1], :]

        return {
            "sample_id": spec["sample_id"],
            "sample_index": spec["sample_index"],
            "seed": seed,
            "events": events,
            "stopping_reason": stopping_reason,
            "warnings": [],
        }

    def _event(
        self,
        *,
        mark_x: float,
        mark_y: float,
        inter_s: float,
        arrival_s: float,
        included: bool,
        width: int,
        height: int,
    ) -> dict[str, Any]:
        native_multiplier = 1.0 / self.config["temporal_scale_to_seconds"]
        x_px = min(max(((mark_x + 1.0) / 2.0) * (width - 1), 0.0), width - 1)
        y_px = min(max(((mark_y + 1.0) / 2.0) * (height - 1), 0.0), height - 1)
        return {
            "x_px": x_px,
            "y_px": y_px,
            "mark_x_native": mark_x,
            "mark_y_native": mark_y,
            "inter_event_time_native": inter_s * native_multiplier,
            "arrival_time_native": arrival_s * native_multiplier,
            "model_inter_event_time_s": inter_s,
            "model_arrival_time_s": arrival_s,
            "included": included,
        }


def main() -> int:

    return serve(TPPGazeBackend)


if __name__ == "__main__":
    raise SystemExit(main())
