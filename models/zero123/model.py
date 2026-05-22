"""Stable Zero123 multi-view synthesis (Windows-compatible, no Wonder3D)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

from utils.base_model import BaseModel
from utils.io import save_image, save_json
from utils.visualization import view_grid


class Zero123Model(BaseModel):
    name = "zero123"

    def load_model(self) -> None:
        if self.config.get("use_mock_models", False):
            self._set_mock_mode("use_mock_models=true")
        else:
            ckpt = Path(self.config.get("paths", {}).get("checkpoints", {}).get("zero123", ""))
            if not ckpt.exists():
                self._set_mock_mode(f"checkpoint not found: {ckpt}")
        self._loaded = True

    def run_inference(
        self,
        image: np.ndarray,
        depth_map: np.ndarray,
        masks: List[np.ndarray],
        viewpoints: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Lightweight viewpoint synthesis for relational reasoning (not photorealistic).
        """
        self.ensure_loaded()
        h, w = image.shape[:2]
        views = []
        visibility = _mask_visibility(masks, h, w)
        for i, angle in enumerate(viewpoints):
            rot = {"+60": 8, "-60": -8, "+120": 14, "-120": -14, "rear": 0, "top": 5}.get(angle, 0)
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, rot, 1.0)
            out = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            if angle == "rear":
                out = cv2.flip(out, 1)
            if angle == "top":
                out = cv2.flip(out, 0)
            if depth_map is not None:
                z_med = float(np.median(depth_map))
                shade = 0.9 + 0.1 * (1.0 - z_med)
                out = np.clip(out.astype(np.float32) * shade, 0, 255).astype(np.uint8)
            views.append({
                "view_id": f"view_{i + 1:02d}",
                "angle": angle,
                "image": out,
                "visibility_score": visibility,
                "source": "zero123",
            })
        return views

    def save_outputs(self, outputs: List[Dict[str, Any]], output_dir: Path) -> Dict[str, Any]:
        from utils.io import ensure_dir

        ensure_dir(output_dir)
        saved = []
        for v in outputs:
            p = output_dir / f"{v['view_id']}_{v['angle']}.png"
            save_image(v["image"], p)
            saved.append({
                "view_id": v["view_id"],
                "angle": v["angle"],
                "path": str(p),
                "visibility_score": v.get("visibility_score"),
            })
        save_json(saved, output_dir / "views_zero123.json")
        return {"views": saved}

    def visualize(self, outputs: List[Dict[str, Any]], output_dir: Path, **kwargs) -> List[Path]:
        from utils.io import ensure_dir

        ensure_dir(output_dir)
        return [
            view_grid(
                [v["image"] for v in outputs],
                [v["angle"] for v in outputs],
                output_dir / "zero123_view_grid.png",
            )
        ]


def _mask_visibility(masks: List[np.ndarray], h: int, w: int) -> float:
    if not masks:
        return 0.5
    union = np.zeros((h, w), dtype=bool)
    for m in masks:
        union |= m.astype(bool)
    return float(union.sum()) / (h * w + 1e-8)
