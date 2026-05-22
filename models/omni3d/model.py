"""Omni3D lightweight semantic pseudo-3D lifting (no mesh/NeRF/splatting)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from utils.base_model import BaseModel
from utils.io import ensure_dir, save_json


class Omni3DModel(BaseModel):
    name = "omni3d"

    def load_model(self) -> None:
        if self.config.get("use_mock_models", False):
            self._set_mock_mode("use_mock_models=true")
        else:
            ckpt = Path(self.config.get("paths", {}).get("checkpoints", {}).get("omni3d", ""))
            if not ckpt.exists():
                self._set_mock_mode(f"checkpoint not found: {ckpt}")
        self._loaded = True

    def run_inference(
        self,
        segmentations: List[Dict[str, Any]],
        depth_map: np.ndarray,
        object_depths: Dict[str, float],
        views: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        self.ensure_loaded()
        h, w = depth_map.shape[:2]
        view_bias = _view_depth_bias(views) if views else 0.0
        lifted = []
        for seg in segmentations:
            if "bbox" not in seg:
                continue
            x1, y1, x2, y2 = seg["bbox"]
            cx = (x1 + x2) / 2 / w - 0.5
            cy = (y1 + y2) / 2 / h - 0.5
            z = object_depths.get(seg["label"], 0.5) + view_bias
            orientation = [
                round((x2 - x1) / w, 4),
                round((y2 - y1) / h, 4),
                round(seg.get("visible_ratio", 1.0) * 3.14159, 4),
            ]
            lifted.append({
                "object": seg["label"],
                "position": [round(cx, 4), round(cy, 4), round(float(z), 4)],
                "orientation": orientation,
                "bbox_3d": [
                    [cx - 0.05, cy - 0.05, z - 0.05],
                    [cx + 0.05, cy + 0.05, z + 0.05],
                ],
                "pseudo_3d": True,
            })
        return lifted

    def save_outputs(self, outputs: List[Dict[str, Any]], output_dir: Path) -> Dict[str, Any]:
        ensure_dir(output_dir)
        for i, obj in enumerate(outputs):
            pts = _pseudo_point_cloud(obj["position"], n=150)
            safe = "".join(c if c.isalnum() else "_" for c in obj["object"])[:32]
            pc_path = output_dir / f"pc_{i:02d}_{safe}.npy"
            np.save(pc_path, pts)
            obj["point_cloud_path"] = str(pc_path)
        path = output_dir / "lifting.json"
        save_json(outputs, path)
        return {"lifting_path": str(path), "objects": outputs}

    def visualize(self, outputs: List[Dict[str, Any]], output_dir: Path, **kwargs) -> List[Path]:
        paths = _render_3d_preview(outputs, output_dir / "bbox_3d_preview.png")
        paths.extend(_render_pointclouds(outputs, output_dir / "pointcloud_preview.png"))
        return paths


def _view_depth_bias(views: List[Dict[str, Any]]) -> float:
    if not views:
        return 0.0
    scores = [v.get("visibility_score", 0.5) for v in views]
    return 0.02 * (float(np.mean(scores)) - 0.5)


def _pseudo_point_cloud(center: List[float], n: int = 150) -> np.ndarray:
    rng = np.random.default_rng(42)
    return np.array(center) + rng.normal(0, 0.02, size=(n, 3))


def _render_3d_preview(objects: List[Dict[str, Any]], path: Path) -> List[Path]:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    for obj in objects:
        pos = obj["position"]
        ax.scatter(pos[0], pos[1], pos[2], s=80, label=obj["object"])
    ax.legend(fontsize=8)
    ensure_dir(path.parent)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return [path]


def _render_pointclouds(objects: List[Dict[str, Any]], path: Path) -> List[Path]:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    for obj in objects:
        if obj.get("point_cloud_path"):
            pts = np.load(obj["point_cloud_path"])
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=2, alpha=0.5)
    ensure_dir(path.parent)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return [path]
