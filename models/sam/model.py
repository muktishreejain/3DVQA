"""SAM segmentation wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

from utils.base_model import BaseModel
from utils.io import save_image, save_json
from utils.visualization import draw_masks_overlay


class SAMModel(BaseModel):
    name = "sam"

    def load_model(self) -> None:
        if self.config.get("use_mock_models", False):
            self._set_mock_mode("use_mock_models=true")
        else:
            ckpt = Path(self.config.get("paths", {}).get("checkpoints", {}).get("sam", ""))
            if not ckpt.exists():
                self._set_mock_mode(f"checkpoint not found: {ckpt}")
        self._loaded = True

    def run_inference(
        self,
        image: np.ndarray,
        detections: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        self.ensure_loaded()
        h, w = image.shape[:2]
        results = []
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            mask = np.zeros((h, w), dtype=np.uint8)
            if self._mock_mode:
                mask[y1:y2, x1:x2] = 1
                # Soft boundary simulation
                mask = cv2.GaussianBlur(mask.astype(np.float32), (15, 15), 0)
                mask = (mask > 0.3).astype(np.uint8)
            else:
                mask[y1:y2, x1:x2] = 1
            visible_ratio = float(mask.sum()) / (mask.size + 1e-8)
            results.append({
                "label": det["label"],
                "bbox": det["bbox"],
                "mask": mask,
                "visible_ratio": round(min(visible_ratio * 4, 1.0), 3),
            })
        return results

    def save_outputs(self, outputs: List[Dict[str, Any]], output_dir: Path) -> Dict[str, Any]:
        saved = []
        for i, seg in enumerate(outputs):
            mask_path = output_dir / f"mask_{i:02d}_{seg['label'].replace(' ', '_')}.png"
            save_image((seg["mask"] * 255).astype(np.uint8), mask_path)
            entry = {k: v for k, v in seg.items() if k != "mask"}
            entry["mask_path"] = str(mask_path)
            saved.append(entry)
        meta_path = output_dir / "segmentations.json"
        save_json(saved, meta_path)
        return {"segmentations_path": str(meta_path), "masks": saved}

    def visualize(
        self,
        outputs: List[Dict[str, Any]],
        output_dir: Path,
        image: np.ndarray,
        **kwargs,
    ) -> List[Path]:
        masks = [o["mask"] for o in outputs]
        labels = [o["label"] for o in outputs]
        paths = [draw_masks_overlay(image, masks, labels, output_dir / "mask_overlay.png")]
        for i, seg in enumerate(outputs):
            crop = image.copy()
            m = seg["mask"].astype(bool)
            crop[~m] = (crop[~m] * 0.3).astype(np.uint8)
            rgba = np.dstack([crop, (seg["mask"] * 180).astype(np.uint8)])
            crop_path = output_dir / f"crop_transparent_{i:02d}.png"
            save_image(rgba, crop_path)
            paths.append(crop_path)
        return paths
