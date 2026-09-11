
from __future__ import annotations

import hashlib
import os
import random
import sys
from pathlib import Path
from typing import Any

from activevision_workbench.adapters.scandiff.worker_server import serve


class RealScanDiffBackend:

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        upstream_root = Path(config["upstream_root"])
        sys.path.insert(0, str(upstream_root))

        import hydra
        import numpy as np
        import torch
        from hydra import compose, initialize_config_dir
        from torch import nn

        self.hydra = hydra
        self.np = np
        self.torch = torch
        self.nn = nn
        if not torch.cuda.is_available():
            raise RuntimeError(
                "the pinned ScanDiff demo requires CUDA, but CUDA is unavailable"
            )
        from src.utils.create_diffusion import create_diffusion

        with initialize_config_dir(
            version_base="1.3",
            config_dir=str((upstream_root / "configs").resolve()),
        ):
            upstream_config = compose(config_name="demo.yaml")
        if upstream_config.data.max_len != config["max_fixations"]:
            raise RuntimeError(
                "pinned upstream max_len differs from the adapter's verified value"
            )
        self.upstream_config = upstream_config
        self.model = hydra.utils.instantiate(upstream_config.model)
        checkpoint = torch.load(config["checkpoint_path"], map_location="cpu")
        if not isinstance(checkpoint, dict) or "model" not in checkpoint:
            raise RuntimeError("ScanDiff checkpoint does not contain the 'model' state")
        self.model.load_state_dict(checkpoint["model"])
        self.model.cuda().eval()
        self.diffusion = create_diffusion(
            upstream_config,
            timestep_respacing="",
            diffusion_steps=upstream_config.diffusion.num_timesteps,
            noise_schedule=upstream_config.diffusion.noise_schedule,
            predict_xstart=upstream_config.diffusion.predict_xstart,
        )
        self.task_embeddings = np.load(
            config["task_embeddings_path"], allow_pickle=True
        ).item()
        if not isinstance(self.task_embeddings, dict):
            raise RuntimeError("task_embeddings.npy does not contain a mapping")
        self._feature_model: Any | None = None
        self._feature_transform: Any | None = None

    def predict(
        self, request: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        image_path = _local_image(request["image_ref"])
        image_features, image_size, feature_sha256 = self._image_features(
            image_path, request["viewing_task"]
        )
        if image_size != (request["image_width"], request["image_height"]):
            raise RuntimeError(
                "canonical image dimensions do not match the image decoded by ScanDiff"
            )
        viewing_task = request["viewing_task"]
        if viewing_task not in self.task_embeddings:
            choices = ", ".join(sorted(str(key) for key in self.task_embeddings))
            raise RuntimeError(
                f"unsupported ScanDiff viewing_task {viewing_task!r}; "
                f"available task-embedding keys: {choices}"
            )
        task_embedding = self.torch.from_numpy(
            self.task_embeddings[viewing_task]
        ).cuda()
        samples: list[dict[str, Any]] = []
        for spec in request["sample_specs"]:
            self._seed_everything(spec["seed"])
            image_condition = image_features.unsqueeze(0).cuda()
            conditioned_task = task_embedding.unsqueeze(0)
            initial_noise = self.torch.randn(
                1,
                self.config["max_fixations"],
                self.model.scanpath_emb_size,
                device="cuda",
            )
            with self.torch.no_grad():
                generated = self.diffusion.p_sample_loop(
                    self.model,
                    initial_noise.shape,
                    initial_noise,
                    clip_denoised=False,
                    model_kwargs={
                        "y": image_condition,
                        "task_embedding": conditioned_task,
                    },
                    progress=False,
                    device="cuda",
                )
                decoded = self.model.get_coords_and_time(generated)
                validity = self.nn.Softmax(dim=-1)(
                    self.model.token_validity_predictor(generated)
                ).argmax(dim=-1)
                length = int(self.torch.cumprod(validity, dim=-1).sum(-1).item())
            native = decoded[0, :length].detach().cpu().tolist()
            samples.append(
                {
                    "sample_id": spec["sample_id"],
                    "sample_index": spec["sample_index"],
                    "seed": spec["seed"],
                    "fixations": [
                        {
                            "x_norm": row[0],
                            "y_norm": row[1],
                            "duration_s": row[2],
                        }
                        for row in native
                    ],
                    "stopping_reason": (
                        "model_stop"
                        if length < self.config["max_fixations"]
                        else "max_fixations"
                    ),
                    "warnings": [],
                }
            )
        metadata = {
            "checkpoint_sha256": self.config["checkpoint_sha256"],
            "task_embeddings_sha256": self.config["task_embeddings_sha256"],
            "feature_sha256": feature_sha256,
            "config_sha256": self.config["config_sha256"],
            "image_width": image_size[0],
            "image_height": image_size[1],
            "upstream_commit": self.config["upstream_commit"],
        }
        return samples, metadata

    def _seed_everything(self, seed: int) -> None:
        random.seed(seed)
        self.np.random.seed(seed)
        self.torch.manual_seed(seed)
        self.torch.cuda.manual_seed_all(seed)

    def _image_features(
        self, image_path: Path, viewing_task: str
    ) -> tuple[Any, tuple[int, int], str]:
        from PIL import Image

        with Image.open(image_path) as image:
            image_size = image.size
        feature_root = self.config["feature_root"]
        if feature_root is not None:
            root = Path(feature_root)
            task_directory = viewing_task.replace(" ", "_")
            candidates = [root / f"{image_path.stem}.pth"]
            if task_directory:
                candidates.append(root / task_directory / f"{image_path.stem}.pth")
            existing = [path for path in candidates if path.is_file()]
            if len(existing) != 1:
                checked = ", ".join(str(path) for path in candidates)
                raise RuntimeError(
                    "expected exactly one precomputed DINOv2 feature file; "
                    f"checked: {checked}"
                )
            feature_path = existing[0]
            features = self.torch.load(feature_path, map_location="cpu")
            if getattr(features, "ndim", None) == 3 and features.shape[0] == 1:
                features = features.squeeze(0)
            return features, image_size, _sha256_file(feature_path)

        # This exactly follows demo.py. Offline mode is set by the parent unless
        # allow_feature_download=true was explicitly configured.
        import timm

        if self._feature_model is None:
            self._feature_model = timm.create_model(
                "vit_base_patch14_reg4_dinov2.lvd142m",
                pretrained=True,
                num_classes=0,
            ).eval()
            data_config = timm.data.resolve_model_data_config(self._feature_model)
            self._feature_transform = timm.data.create_transform(
                **data_config, is_training=False
            )
        with Image.open(image_path) as source:
            resized = source.resize((518, 518))
            inputs = self._feature_transform(resized).unsqueeze(0)
        features = self._feature_model.forward_features(inputs)
        features = features[:, 5:, :].squeeze().detach().cpu()
        digest = hashlib.sha256(features.numpy().tobytes()).hexdigest()
        return features, image_size, digest


def _local_image(reference: str) -> Path:
    if "://" in reference:
        raise RuntimeError("ScanDiff currently accepts only local image paths")
    path = Path(reference)
    if not path.is_file():
        raise RuntimeError(f"ScanDiff image does not exist: {path}")
    return path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:

    os.environ.setdefault("HYDRA_FULL_ERROR", "1")
    return serve(RealScanDiffBackend)


if __name__ == "__main__":
    raise SystemExit(main())
