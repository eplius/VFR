"""Standalone, inference-only VFR runtime. No private project imports."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preprocess(image):
    image = image.convert("RGB").resize((640, 640), Image.Resampling.BILINEAR)
    array = np.asarray(image).copy()
    return torch.from_numpy(array).permute(2, 0, 1).float().div(255).unsqueeze(0)


class VFR:
    def __init__(self, model, weights_dir=None):
        directory = Path(weights_dir) if weights_dir else ROOT / "weights"
        name = f"vfr_{model}_coco"
        self.metadata = json.loads((directory / f"{name}.json").read_text())
        self.path = directory / f"{name}.pt"
        if sha256(self.path) != self.metadata["sha256"]:
            raise ValueError(f"Weight checksum mismatch: {self.path}")
        if not torch.cuda.is_available():
            raise RuntimeError("This release requires CUDA. Select a physical GPU with CUDA_VISIBLE_DEVICES.")
        # The exported graph uses logical cuda:0; physical selection is external.
        self.device = torch.device("cuda:0")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.model = torch.jit.load(str(self.path), map_location=self.device).eval()

    @torch.inference_mode()
    def raw(self, tensor):
        if tensor.ndim != 4 or tensor.shape[0] < 1 or tuple(tensor.shape[1:]) != (3, 640, 640):
            raise ValueError("Expected a nonempty batch of RGB images at 640 x 640.")
        if self.metadata["input_shape"][0] == 1 and tensor.shape[0] != 1:
            raise ValueError("This model's validated deployment profile requires batch size 1.")
        # Precision and position-embedding behavior are baked into each graph.
        with torch.jit.optimized_execution(False):
            logits, points = self.model(tensor.to(self.device))
        if not torch.isfinite(logits).all() or not torch.isfinite(points).all():
            raise FloatingPointError("Non-finite model prediction")
        return logits, points

    @torch.inference_mode()
    def predict(self, image):
        return self.predict_batch([image])[0]

    @torch.inference_mode()
    def predict_batch(self, images):
        tensors = [preprocess(image) for image in images]
        logits, points = self.raw(tensors[0] if len(tensors) == 1 else torch.cat(tensors))
        scores, indices = logits.sigmoid().flatten(1).topk(60, dim=1)
        labels = indices % logits.shape[-1]
        queries = torch.div(indices, logits.shape[-1], rounding_mode="floor")
        points = points.reshape(len(images), -1, 17, 2)
        points = points.gather(1, queries[:, :, None, None].expand(-1, -1, 17, 2))
        points = points * points.new_tensor([image.size for image in images])[:, None, None, :]
        points = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)
        return [
            [{"category_id": int(label), "score": float(score), "keypoints": pose}
             for label, score, pose in zip(image_labels, image_scores, image_points)]
            for image_labels, image_scores, image_points in zip(
                labels.tolist(), scores.tolist(), points.flatten(2).tolist())
        ]
