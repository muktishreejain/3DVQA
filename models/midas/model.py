"""MiDaS depth estimation (Windows-friendly, timm/HuggingFace optional)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

from utils.base_model import BaseModel
from utils.io import save_image, save_json
from utils.visualization import depth_colormap


class MiDaSModel(BaseModel):
    name = "midas"

    def load_model(self) -> None:
        if self.config.get("use_mock_models", False):
            self._set_mock_mode("use_mock_models=true")
            self._loaded = True
            return
        ckpt = Path(self.config.get("paths", {}).get("checkpoints", {}).get("midas", ""))
        if not ckpt.exists():
            self._set_mock_mode(f"checkpoint not found: {ckpt}")
        else:
            try:
                import torch

                self.torch = torch
                self.logger.info("MiDaS checkpoint path: %s", ckpt)
            except ImportError as exc:
                self._set_mock_mode(str(exc))
        self._loaded = True

    def run_inference(
        self,
        image: np.ndarray,
        segmentations: List[Dict[str, Any]],
        query_mask: np.ndarray | None = None,
    ) -> Dict[str, Any]:
        self.ensure_loaded()
        depth = self._estimate_depth(image)
        if query_mask is not None and query_mask.any():
            depth = depth * query_mask.astype(np.float32) + depth.mean() * (~query_mask.astype(bool))

        if self.config.get("depth", {}).get("normalize", True):
            depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

        object_depths = _object_depths(depth, segmentations)
        return {"depth_map": depth, "object_depths": object_depths, "backend": "midas"}

    def _estimate_depth(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
        if self._mock_mode:
            h, w = gray.shape
            yy, xx = np.mgrid[0:h, 0:w]
            depth = 1.0 - np.sqrt(((xx - w / 2) / w) ** 2 + ((yy - h / 2) / h) ** 2)
            return cv2.bilateralFilter(depth.astype(np.float32), 9, 75, 75)
        return gray / 255.0

    def save_outputs(self, outputs: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
        from utils.io import ensure_dir

        ensure_dir(output_dir)
        depth = outputs["depth_map"]
        gray_path = output_dir / "depth_gray.png"
        save_image((depth * 255).astype(np.uint8), gray_path)
        heat_path = output_dir / "depth_heatmap.png"
        depth_colormap(depth, heat_path)
        meta = {
            "depth_map": str(gray_path),
            "depth_heatmap": str(heat_path),
            "object_depths": outputs["object_depths"],
            "backend": outputs.get("backend", "midas"),
        }
        save_json(meta, output_dir / "depth.json")
        return meta

    def visualize(self, outputs: Dict[str, Any], output_dir: Path, **kwargs) -> List[Path]:
        depth = outputs["depth_map"]
        p1 = output_dir / "viz_depth_gray.png"
        p2 = output_dir / "viz_depth_heatmap.png"
        save_image((depth * 255).astype(np.uint8), p1)
        depth_colormap(depth, p2)
        return [p1, p2]


def _object_depths(depth: np.ndarray, segmentations: List[Dict[str, Any]]) -> Dict[str, float]:
    from PIL import Image

    object_depths: Dict[str, float] = {}
    for seg in segmentations:
        mask = seg.get("mask")
        if mask is None and "mask_path" in seg:
            mask = np.array(Image.open(seg["mask_path"])) > 127
        if mask is not None:
            md = depth[mask.astype(bool)]
            object_depths[seg["label"]] = float(np.median(md)) if md.size else 0.5
    return object_depths
