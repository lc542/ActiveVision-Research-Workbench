
from __future__ import annotations

import math
import random
import sys
from pathlib import Path
from typing import Any

from activevision_workbench.adapters.individual_scanpath.worker_server import serve


class IndividualScanpathBackend:

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        source_root = (
            Path(config["upstream_root"]) / "OSIE" / "ChenLSTMISP" / "src"
        )
        sys.path.insert(0, str(source_root))
        import numpy as np
        import torch
        from PIL import Image
        from torchvision import transforms

        if config["device"] == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("IndividualScanpath requested CUDA but it is unavailable")
        checkpoint = torch.load(config["checkpoint_path"], map_location="cpu")
        if not isinstance(checkpoint, dict) or set(checkpoint) != {"model"}:
            raise RuntimeError("IndividualScanpath checkpoint must contain only 'model'")
        state = checkpoint["model"]
        if tuple(state["subject_embedding.weight"].shape) != (
            config["subject_count"],
            config["embedding_dim"],
        ):
            raise RuntimeError("checkpoint observer embedding shape is incompatible")

        import models.baseline_attention as baseline_module
        import models.resnet as resnet_module

        # Upstream asks model_zoo for ImageNet weights during construction even
        # though the selected checkpoint contains every retained ResNet tensor.
        # Construct the same upstream backbone without that network side effect;
        # the complete checkpoint is then loaded strictly.
        original_resnet50 = baseline_module.resnet50
        baseline_module.resnet50 = lambda pretrained=True: resnet_module.resnet50(
            pretrained=False
        )
        try:
            model = baseline_module.baseline(
                embed_size=512,
                convLSTM_length=config["model_max_fixations"],
                min_length=config["min_fixations"],
                dropout=0.2,
                subject_num=config["subject_count"],
                embedding_dim=config["embedding_dim"],
                action_map_num=4,
            )
        finally:
            baseline_module.resnet50 = original_resnet50
        model.load_state_dict(state, strict=True)
        self.device = torch.device(config["device"])
        self.model = model.to(self.device).eval()
        self.torch = torch
        self.np = np
        self.Image = Image
        self.transform = transforms.Compose(
            [
                transforms.Resize((config["input_height"], config["input_width"])),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                ),
            ]
        )
        self._image_cache: dict[tuple[str, int, int], tuple[Any, int, int]] = {}

    def predict(
        self, request: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        image, width, height = self._image(request["image_ref"])
        if (width, height) != (request["image_width"], request["image_height"]):
            raise ValueError(
                "decoded image dimensions do not match request: "
                f"{width}x{height} != {request['image_width']}x{request['image_height']}"
            )
        torch = self.torch
        subject = torch.tensor(
            [request["model_observer_index"]], dtype=torch.long, device=self.device
        )
        with torch.inference_mode():
            prediction = self.model(image.to(self.device), subject)
        samples = [
            self._sample(prediction, request, spec) for spec in request["sample_specs"]
        ]
        return samples, {
            "checkpoint_sha256": self.config["checkpoint_sha256"],
            "observer_mapping_sha256": self.config["observer_mapping_sha256"],
            "config_sha256": self.config["config_sha256"],
            "upstream_commit": self.config["upstream_commit"],
            "variant": self.config["variant"],
            "image_width": width,
            "image_height": height,
            "model_input_width": self.config["input_width"],
            "model_input_height": self.config["input_height"],
            "map_width": self.config["map_width"],
            "map_height": self.config["map_height"],
            "image_cache_scope": "decoded-and-preprocessed-image-only; no observer-conditioned outputs",
        }

    def _image(self, reference: str) -> tuple[Any, int, int]:
        path = Path(reference)
        stat = path.stat()
        key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
        cached = self._image_cache.get(key)
        if cached is not None:
            return cached
        with self.Image.open(path) as raw:
            image = raw.convert("RGB")
            width, height = image.size
            tensor = self.transform(image).unsqueeze(0)
        value = (tensor, width, height)
        self._image_cache = {key: value}
        return value

    def _sample(
        self,
        prediction: dict[str, Any],
        request: dict[str, Any],
        spec: dict[str, Any],
    ) -> dict[str, Any]:
        torch = self.torch
        seed = spec["seed"]
        random.seed(seed)
        self.np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        probs = prediction["all_actions_prob"].detach().clone()
        probs[:, : self.config["min_fixations"], 0] = 0
        selected = torch.distributions.Categorical(probs=probs).sample()[0]
        selected_probs = torch.gather(
            prediction["all_actions_prob"],
            2,
            selected.unsqueeze(0).unsqueeze(-1),
        )[0, :, 0]
        mu = prediction["log_normal_mu"][0]
        sigma = prediction["log_normal_sigma2"][0]
        durations = torch.exp(torch.randn_like(mu) * sigma + mu)
        action_values = [int(value) for value in selected.detach().cpu().tolist()]
        duration_values = [
            float(value) for value in durations.detach().cpu().tolist()
        ]
        probability_values = [
            float(value) for value in selected_probs.detach().cpu().tolist()
        ]
        termination_step = next(
            (index for index, action in enumerate(action_values) if action == 0),
            None,
        )
        effective_max = request["max_fixations"]
        stopping_reason = (
            "model_stop"
            if termination_step is not None and termination_step < effective_max
            else "max_fixations"
        )
        events = []
        for step, action in enumerate(action_values[:effective_max]):
            if action == 0:
                break
            map_index = action - 1
            map_x = map_index % self.config["map_width"]
            map_y = map_index // self.config["map_width"]
            event = {
                "x_model_px": (map_x + 0.5)
                * self.config["input_width"]
                / self.config["map_width"],
                "y_model_px": (map_y + 0.5)
                * self.config["input_height"]
                / self.config["map_height"],
                "duration_s": duration_values[step],
                "action_index": action,
                "action_probability": probability_values[step],
            }
            for name, value in event.items():
                if name != "action_index" and not math.isfinite(float(value)):
                    raise ValueError(f"IndividualScanpath sampled non-finite {name}")
            if event["duration_s"] < 0:
                raise ValueError("IndividualScanpath sampled a negative duration")
            events.append(event)
        return {
            "sample_id": spec["sample_id"],
            "sample_index": spec["sample_index"],
            "seed": seed,
            "observer_id": request["observer_id"],
            "model_observer_index": request["model_observer_index"],
            "events": events,
            "native_sequence": {
                "selected_actions": action_values,
                "duration_samples_s": duration_values,
                "selected_action_probabilities": probability_values,
                "termination_step": termination_step,
            },
            "stopping_reason": stopping_reason,
            "warnings": [],
        }


def main() -> int:
    return serve(IndividualScanpathBackend)


if __name__ == "__main__":
    raise SystemExit(main())
