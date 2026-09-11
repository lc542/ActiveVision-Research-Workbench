
from __future__ import annotations

import hashlib
import json
import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from activevision_workbench.adapters.gazexplain.worker_server import serve


class GazeXplainBackend:

    _ALLOWED_MISSING_CHECKPOINT_KEYS = {
        "blip_model.cls.predictions.decoder.weight",
        "blip_model.cls.predictions.decoder.bias",
    }

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        source_root = Path(config["upstream_root"]) / "src"
        sys.path.insert(0, str(source_root))

        import numpy as np
        import torch
        from PIL import Image
        from safetensors.torch import load_file
        from torchvision import transforms
        from torchvision.models.detection import maskrcnn_resnet50_fpn
        from transformers import (
            BertTokenizerFast,
            BlipConfig,
            RobertaConfig,
            RobertaModel,
            RobertaTokenizerFast,
        )
        from transformers.models.blip.modeling_blip_text import (
            BlipTextLMHeadModel,
        )

        if config["device"] == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("GazeXplain requested CUDA but it is unavailable")
        self.device = torch.device(config["device"])
        assets = Path(config["model_assets_root"])
        roberta_assets = assets / "roberta-base"
        blip_assets = assets / "blip-image-captioning-base"
        backbone_checkpoint = assets / "maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth"
        for path, label in (
            (roberta_assets, "RoBERTa assets"),
            (blip_assets, "BLIP assets"),
        ):
            if not path.is_dir():
                raise RuntimeError(f"GazeXplain {label} directory is missing: {path}")
        if not backbone_checkpoint.is_file():
            raise RuntimeError(
                "GazeXplain Mask R-CNN feature-backbone checkpoint is missing: "
                f"{backbone_checkpoint}"
            )
        actual_backbone_sha256 = _sha256_file(backbone_checkpoint)
        if actual_backbone_sha256 != config["feature_backbone_sha256"]:
            raise RuntimeError(
                "GazeXplain Mask R-CNN feature-backbone SHA-256 mismatch"
            )
        self.feature_backbone_sha256 = actual_backbone_sha256

        hparams = json.loads(Path(config["hparams_path"]).read_text(encoding="utf-8"))
        if not isinstance(hparams, dict):
            raise RuntimeError("GazeXplain hparams must be an object")
        hparams["max_generation_length"] = config["max_generation_length"]
        hparams["num_explanation_beams"] = config["num_explanation_beams"]
        args = SimpleNamespace(**hparams)

        task_tokenizer = RobertaTokenizerFast.from_pretrained(
            roberta_assets, local_files_only=True
        )
        explanation_tokenizer = BertTokenizerFast.from_pretrained(
            blip_assets, local_files_only=True
        )
        roberta_config = RobertaConfig.from_pretrained(
            roberta_assets, local_files_only=True
        )
        blip_config = BlipConfig.from_pretrained(
            blip_assets, local_files_only=True
        )

        import lib.models.gazeformer_explanation_alignment as gazeformer_module
        from lib.models.models import Transformer

        original_roberta = RobertaModel.from_pretrained
        original_blip = BlipTextLMHeadModel.from_pretrained
        original_tokenizer = BertTokenizerFast.from_pretrained
        RobertaModel.from_pretrained = classmethod(  # type: ignore[method-assign]
            lambda cls, *unused_args, **unused_kwargs: cls(roberta_config)
        )
        BlipTextLMHeadModel.from_pretrained = classmethod(  # type: ignore[method-assign]
            lambda cls, *unused_args, **unused_kwargs: cls(blip_config.text_config)
        )
        BertTokenizerFast.from_pretrained = classmethod(  # type: ignore[method-assign]
            lambda cls, *unused_args, **unused_kwargs: explanation_tokenizer
        )
        try:
            model = gazeformer_module.gazeformer(
                transformer=Transformer(args=args), args=args
            )
        finally:
            RobertaModel.from_pretrained = original_roberta  # type: ignore[method-assign]
            BlipTextLMHeadModel.from_pretrained = original_blip  # type: ignore[method-assign]
            BertTokenizerFast.from_pretrained = original_tokenizer  # type: ignore[method-assign]

        state = load_file(config["checkpoint_path"], device="cpu")
        incompatible = model.load_state_dict(state, strict=False)
        missing = set(incompatible.missing_keys)
        unexpected = set(incompatible.unexpected_keys)
        if missing != self._ALLOWED_MISSING_CHECKPOINT_KEYS or unexpected:
            raise RuntimeError(
                "GazeXplain checkpoint tensors are incompatible: "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
        model.blip_model.tie_weights()
        self.model = model.to(self.device).eval()

        detector = maskrcnn_resnet50_fpn(
            weights=None, weights_backbone=None
        )
        detector_state = torch.load(backbone_checkpoint, map_location="cpu")
        detector.load_state_dict(detector_state, strict=True)
        self.backbone = detector.backbone.body.to(self.device).eval()
        del detector

        self.task_tokenizer = task_tokenizer
        self.explanation_tokenizer = explanation_tokenizer
        self.torch = torch
        self.np = np
        self.Image = Image
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (
                        config["feature_input_height"],
                        config["feature_input_width"],
                    )
                ),
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
        image_feature, width, height = self._image_feature(request["image_ref"])
        if (width, height) != (request["image_width"], request["image_height"]):
            raise ValueError(
                "decoded image dimensions do not match request: "
                f"{width}x{height} != "
                f"{request['image_width']}x{request['image_height']}"
            )
        task_input = self.task_tokenizer(
            [request["resolved_task_text"]],
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        task_token_count = int(task_input.input_ids.shape[1])
        model_max = int(self.task_tokenizer.model_max_length)
        if task_token_count > model_max:
            raise ValueError(
                "GazeXplain task input exceeds the pinned RoBERTa tokenizer limit "
                f"({task_token_count} > {model_max}); input was not truncated"
            )
        torch = self.torch
        target_scanpath = torch.zeros(
            (
                1,
                self.config["model_max_fixations"],
                self.config["map_width"] * self.config["map_height"] + 1,
            ),
            dtype=torch.float32,
            device=self.device,
        )
        target_scanpath[:, :, 0] = 1.0
        blank_explanation = self.explanation_tokenizer(
            [""] * self.config["model_max_fixations"],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config["max_generation_length"],
        )
        batch = {
            "image_feature": image_feature,
            "task_input": task_input.to(self.device),
            # These two tensors satisfy an unused alignment computation that
            # remains in the verified upstream eval path. They do not select
            # or condition the sampled scanpath or generated explanations.
            "target_scanpath": target_scanpath,
            "explanation": blank_explanation.to(self.device),
        }
        samples = [self._sample(batch, request, spec) for spec in request["sample_specs"]]
        return samples, {
            "checkpoint_sha256": self.config["checkpoint_sha256"],
            "hparams_sha256": self.config["hparams_sha256"],
            "config_sha256": self.config["config_sha256"],
            "upstream_commit": self.config["upstream_commit"],
            "variant": self.config["variant"],
            "image_width": width,
            "image_height": height,
            "model_input_width": self.config["model_input_width"],
            "model_input_height": self.config["model_input_height"],
            "feature_input_width": self.config["feature_input_width"],
            "feature_input_height": self.config["feature_input_height"],
            "map_width": self.config["map_width"],
            "map_height": self.config["map_height"],
            "resolved_task_text": request["resolved_task_text"],
            "task_input_token_count": task_token_count,
            "task_input_truncated": False,
            "task_tokenizer": {
                "model_id": "FacebookAI/roberta-base",
                "revision": self.config["roberta_revision"],
            },
            "explanation_tokenizer": {
                "model_id": "Salesforce/blip-image-captioning-base",
                "revision": self.config["blip_revision"],
            },
            "feature_backbone": {
                "model_id": "MaskRCNN_ResNet50_FPN_Weights.COCO_V1",
                "sha256": self.feature_backbone_sha256,
            },
            "decoding": {
                "max_generation_length": self.config["max_generation_length"],
                "num_explanation_beams": self.config["num_explanation_beams"],
                "early_stopping": True,
            },
            "image_cache_scope": (
                "decoded-and-preprocessed-image-features-only; "
                "no task-conditioned outputs"
            ),
        }

    def _image_feature(self, reference: str) -> tuple[Any, int, int]:
        path = Path(reference)
        stat = path.stat()
        key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
        cached = self._image_cache.get(key)
        if cached is not None:
            return cached
        with self.Image.open(path) as raw:
            image = raw.convert("RGB")
            width, height = image.size
            tensor = self.transform(image).unsqueeze(0).to(self.device)
        with self.torch.inference_mode():
            feature_map = self.backbone(tensor)["3"]
            feature = feature_map.flatten(2).permute(0, 2, 1).contiguous()
        expected = (
            1,
            self.config["map_height"] * self.config["map_width"],
            2048,
        )
        if tuple(feature.shape) != expected:
            raise RuntimeError(
                f"GazeXplain extracted feature shape {tuple(feature.shape)}; "
                f"expected {expected}"
            )
        value = (feature, width, height)
        self._image_cache = {key: value}
        return value

    def _sample(
        self,
        batch: dict[str, Any],
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
        with torch.inference_mode():
            (
                _prediction,
                scanpath_prediction,
                generated_ids,
                sampling_prediction,
                _action_masks,
                duration_masks,
            ) = self.model(batch, 1, False, False)

        scanpath = scanpath_prediction[0, 0].detach().cpu()
        raw_tokens = generated_ids[0, :, 0].detach().cpu().tolist()
        all_raw_tokens = [
            [int(token) for token in row] for row in raw_tokens
        ]
        selected_actions = (
            sampling_prediction["selected_actions"][0, 0].detach().cpu().tolist()
        )
        duration_samples = (
            sampling_prediction["durations"][0, 0].detach().cpu().tolist()
        )
        action_probabilities = (
            sampling_prediction["selected_actions_probs"][0, 0]
            .detach()
            .cpu()
            .tolist()
        )
        duration_mask = duration_masks[0, 0].detach().cpu().tolist()
        termination_step = next(
            (
                index
                for index, action in enumerate(selected_actions)
                if int(action) == 0
            ),
            None,
        )
        effective_max = request["max_fixations"]
        events = []
        alignments = []
        preserved_tokens = []
        warnings = []
        for step in range(effective_max):
            if step >= len(duration_mask) or float(duration_mask[step]) == 0.0:
                break
            token_ids = [int(token) for token in raw_tokens[step]]
            decoded = self.explanation_tokenizer.decode(
                token_ids, skip_special_tokens=True
            ).strip()
            text = decoded if decoded else None
            truncated = self._is_truncated(token_ids)
            if truncated:
                warnings.append(
                    "GazeXplain explanation generation reached max_generation_length; raw tokens and decoded truncated text were preserved"
                )
            event = {
                "x_model_px": float(scanpath[step, 0]),
                "y_model_px": float(scanpath[step, 1]),
                "duration_ms": float(scanpath[step, 2]),
                "action_index": int(selected_actions[step]),
                "action_probability": float(action_probabilities[step]),
                "explanation_text": text,
                "explanation_token_ids": token_ids,
                "explanation_truncated": truncated,
            }
            for name in (
                "x_model_px",
                "y_model_px",
                "duration_ms",
                "action_probability",
            ):
                if not math.isfinite(event[name]):
                    raise ValueError(f"GazeXplain sampled non-finite {name}")
            if event["duration_ms"] < 0:
                raise ValueError("GazeXplain sampled a negative duration")
            events.append(event)
            alignments.append(
                {
                    "sequence_index": step,
                    "text": text,
                    "token_ids": token_ids,
                    "truncated": truncated,
                }
            )
            preserved_tokens.append(token_ids)
        texts = [alignment["text"] for alignment in alignments if alignment["text"]]
        explanation_text = " ".join(texts).strip() if texts else None
        stopping_reason = (
            "model_stop"
            if termination_step is not None and termination_step < effective_max
            else "max_fixations"
        )
        return {
            "sample_id": spec["sample_id"],
            "sample_index": spec["sample_index"],
            "seed": seed,
            "events": events,
            "explanation": {
                "text": explanation_text,
                "per_fixation": alignments,
                "raw_token_ids": preserved_tokens,
                "raw_token_ids_all_steps": all_raw_tokens,
                "truncated": any(
                    alignment["truncated"] for alignment in alignments
                ),
                "token_scores": None,
            },
            "native_sequence": {
                "selected_actions": [int(value) for value in selected_actions],
                "duration_samples_s": [
                    float(value) for value in duration_samples
                ],
                "selected_action_probabilities": [
                    float(value) for value in action_probabilities
                ],
                "duration_mask": [float(value) for value in duration_mask],
                "termination_step": termination_step,
            },
            "stopping_reason": stopping_reason,
            "warnings": list(dict.fromkeys(warnings)),
        }

    def _is_truncated(self, token_ids: list[int]) -> bool:
        if len(token_ids) < self.config["max_generation_length"]:
            return False
        pad = self.explanation_tokenizer.pad_token_id
        last_non_pad = next(
            (token for token in reversed(token_ids) if token != pad), None
        )
        return last_non_pad not in {
            self.explanation_tokenizer.sep_token_id,
            self.explanation_tokenizer.eos_token_id,
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    return serve(GazeXplainBackend)


if __name__ == "__main__":
    raise SystemExit(main())
