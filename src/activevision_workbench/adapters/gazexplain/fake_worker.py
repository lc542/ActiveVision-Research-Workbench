
from __future__ import annotations

import hashlib
import random
import time
from typing import Any

from activevision_workbench.adapters.gazexplain.worker_server import (
    MalformedFakeResponse,
    serve,
)


class FakeGazeXplainBackend:
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
            raise RuntimeError("deliberate fake GazeXplain error")
        samples = [self._sample(request, spec) for spec in request["sample_specs"]]
        if behavior == "malformed_text":
            samples[0]["explanation"]["text"] = 17
        return samples, {
            "checkpoint_sha256": self.config["checkpoint_sha256"],
            "hparams_sha256": self.config["hparams_sha256"],
            "config_sha256": self.config["config_sha256"],
            "upstream_commit": self.config["upstream_commit"],
            "variant": self.config["variant"],
            "image_width": request["image_width"],
            "image_height": request["image_height"],
            "model_input_width": self.config["model_input_width"],
            "model_input_height": self.config["model_input_height"],
            "feature_input_width": self.config["feature_input_width"],
            "feature_input_height": self.config["feature_input_height"],
            "map_width": self.config["map_width"],
            "map_height": self.config["map_height"],
            "resolved_task_text": request["resolved_task_text"],
            "task_input_token_count": len(request["resolved_task_text"].split()) + 2,
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
                "sha256": self.config["feature_backbone_sha256"],
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

    def _sample(
        self, request: dict[str, Any], spec: dict[str, Any]
    ) -> dict[str, Any]:
        task_digest = int.from_bytes(
            hashlib.sha256(request["resolved_task_text"].encode("utf-8")).digest()[:4],
            "big",
        )
        rng = random.Random(spec["seed"] + task_digest)
        length = min(request["max_fixations"], 1 + spec["seed"] % 4)
        events = []
        actions = []
        durations = []
        probabilities = []
        masks = []
        alignments = []
        raw_token_ids = []
        behavior = self.config["fake_behavior"]
        for step in range(length):
            map_x = (task_digest + spec["seed"] + step * 5) % self.config[
                "map_width"
            ]
            map_y = (
                task_digest // 17 + spec["seed"] + step * 3
            ) % self.config["map_height"]
            action = map_y * self.config["map_width"] + map_x + 1
            duration_s = 0.1 + rng.random() * 0.2
            probability = 0.2 + rng.random() * 0.7
            text: str | None
            token_ids: list[int]
            truncated = behavior == "truncated" and step == length - 1
            if behavior == "no_explanation":
                text = None
                token_ids = []
            else:
                text = f"{request['resolved_task_text']} — 注视点 {step + 1}"
                token_ids = [101] + [1000 + ord(character) % 20000 for character in text] + [102]
                if truncated:
                    token_ids = token_ids[: self.config["max_generation_length"]]
            events.append(
                {
                    "x_model_px": (map_x + 0.5)
                    * self.config["model_input_width"]
                    / self.config["map_width"],
                    "y_model_px": (map_y + 0.5)
                    * self.config["model_input_height"]
                    / self.config["map_height"],
                    "duration_ms": duration_s * 1000.0,
                    "action_index": action,
                    "action_probability": probability,
                    "explanation_text": text,
                    "explanation_token_ids": token_ids,
                    "explanation_truncated": truncated,
                }
            )
            alignments.append(
                {
                    "sequence_index": step,
                    "text": text,
                    "token_ids": token_ids,
                    "truncated": truncated,
                }
            )
            raw_token_ids.append(token_ids)
            actions.append(action)
            durations.append(duration_s)
            probabilities.append(probability)
            masks.append(1.0)
        stopped = length < request["max_fixations"]
        if stopped:
            actions.append(0)
            durations.append(0.05)
            probabilities.append(0.5)
            masks.append(0.0)
        all_raw_token_ids = raw_token_ids + [
            [] for _ in range(len(actions) - len(raw_token_ids))
        ]
        texts = [item["text"] for item in alignments if item["text"]]
        explanation_text = " ".join(texts).strip() if texts else None
        is_truncated = any(item["truncated"] for item in alignments)
        warnings = (
            [
                "GazeXplain explanation generation reached max_generation_length; raw tokens and decoded truncated text were preserved"
            ]
            if is_truncated
            else []
        )
        return {
            "sample_id": spec["sample_id"],
            "sample_index": spec["sample_index"],
            "seed": spec["seed"],
            "events": events,
            "explanation": {
                "text": explanation_text,
                "per_fixation": alignments,
                "raw_token_ids": raw_token_ids,
                "raw_token_ids_all_steps": all_raw_token_ids,
                "truncated": is_truncated,
                "token_scores": None,
            },
            "native_sequence": {
                "selected_actions": actions,
                "duration_samples_s": durations,
                "selected_action_probabilities": probabilities,
                "duration_mask": masks,
                "termination_step": length if stopped else None,
            },
            "stopping_reason": "model_stop" if stopped else "max_fixations",
            "warnings": warnings,
        }


def main() -> int:
    return serve(FakeGazeXplainBackend)


if __name__ == "__main__":
    raise SystemExit(main())
