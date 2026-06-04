"""Analytic virtual-camera reprojection (Image-2, Block 6).

Novel Contribution 2: replaces generative novel-view synthesis (Zero123 /
Wonder3D) with cheap, deterministic geometric reprojection. Instead of
hallucinating RGB views, we place V virtual cameras around the scene centroid
and analytically estimate each object's visibility from every camera using a
centroid z-buffer + angular-overlap occlusion test.

No RGB images are produced here - only per-camera visibility descriptors that
the scoring stage turns into S(v) = IG(v) - lambda * U(v).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from utils.base_stage import BaseStage
from utils.io import save_json


class MultiviewStage(BaseStage):
    stage_name = "views"

    def run(self, context: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
        objects = context.get("lifted_objects", [])
        if not objects:
            raise ValueError("lifted_objects required for analytic reprojection")

        mv_cfg = self.config.get("multiview", {})
        n_cams = int(mv_cfg.get("n_virtual_cams", mv_cfg.get("num_views", 12)))

        centroids = np.array([o["position"] for o in objects], dtype=np.float32)
        scene_center = centroids.mean(axis=0)
        dists = np.linalg.norm(centroids - scene_center, axis=1)
        radius = float(2.0 * dists.mean()) if dists.mean() > 1e-3 else 1.0

        cams = _generate_virtual_cameras(scene_center, radius, n_cams)

        views: List[Dict[str, Any]] = []
        for i, cam in enumerate(cams):
            per_obj_vis = {
                f"n{j}": _reproject_visibility(obj, objects, cam)
                for j, obj in enumerate(objects)
            }
            visible = [v for v in per_obj_vis.values() if v > 0.1]
            views.append({
                "view_id": f"view_{i + 1:02d}",
                "angle": f"az_{int(cam['azimuth_deg']):03d}",
                "azimuth_deg": round(cam["azimuth_deg"], 2),
                "cam": _serializable_cam(cam),
                "per_obj_vis": {k: round(float(v), 4) for k, v in per_obj_vis.items()},
                "visibility_score": round(float(np.mean(visible)) if visible else 0.0, 4),
            })

        save_json(views, output_dir / "views.json")
        _render_reprojection_schematic(objects, views, output_dir / "view_comparison_panel.png")
        return {"views": views}


# ── virtual camera geometry ───────────────────────────────────────────


def _generate_virtual_cameras(center: np.ndarray, radius: float, n: int) -> List[Dict[str, Any]]:
    cams = []
    for i in range(n):
        az = 2 * np.pi * i / max(n, 1)
        pos = center + np.array(
            [radius * np.cos(az), 0.0, radius * np.sin(az)], dtype=np.float32
        )
        R, t = _lookat(pos, center)
        cams.append({"id": f"cam_{i:02d}", "pos": pos, "R": R, "t": t,
                     "azimuth_deg": float(np.degrees(az))})
    return cams


def _lookat(eye: np.ndarray, target: np.ndarray, up=None):
    if up is None:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    forward = target - eye
    forward = forward / (np.linalg.norm(forward) + 1e-8)
    right = np.cross(forward, up)
    rn = np.linalg.norm(right)
    right = np.array([1.0, 0.0, 0.0]) if rn < 1e-6 else right / rn
    up_real = np.cross(right, forward)
    R = np.stack([right, up_real, forward], axis=0).astype(np.float32)
    t = (-R @ eye).astype(np.float32)
    return R, t


def _reproject_visibility(obj: Dict[str, Any], all_objects: List[Dict[str, Any]], cam: Dict[str, Any]) -> float:
    """Centroid z-buffer visibility of `obj` from `cam` in [0, 1]."""
    R, t = cam["R"], cam["t"]
    c = np.asarray(obj["position"], dtype=np.float32)
    obj_cam = R @ c + t
    obj_z = float(obj_cam[2])
    if obj_z <= 0:
        return 0.0  # behind the camera

    blocked = 0.0
    for other in all_objects:
        if other is obj:
            continue
        oc = R @ np.asarray(other["position"], dtype=np.float32) + t
        if oc[2] <= 0 or oc[2] >= obj_z:
            continue  # other is behind obj or behind camera
        sep = float(np.linalg.norm(obj_cam[:2] - oc[:2]))
        r_obj = _angular_radius(obj_cam, obj.get("size_3d", [0.1, 0.1, 0.1]))
        r_other = _angular_radius(oc, other.get("size_3d", [0.1, 0.1, 0.1]))
        overlap = max(0.0, (r_obj + r_other - sep) / (2 * r_obj + 1e-6))
        blocked = min(1.0, blocked + overlap)
    return max(0.0, 1.0 - blocked)


def _angular_radius(cam_pt: np.ndarray, size_3d) -> float:
    max_dim = max(size_3d) / 2.0 if len(size_3d) else 0.05
    return float(max_dim / (cam_pt[2] + 1e-6))


def _serializable_cam(cam: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": cam["id"],
        "pos": [round(float(x), 4) for x in cam["pos"]],
        "R": [[round(float(v), 4) for v in row] for row in cam["R"]],
        "t": [round(float(x), 4) for x in cam["t"]],
        "azimuth_deg": round(cam["azimuth_deg"], 2),
    }


def _render_reprojection_schematic(objects, views, path: Path) -> None:
    import matplotlib.pyplot as plt

    centroids = np.array([o["position"] for o in objects], dtype=np.float32)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(centroids[:, 0], centroids[:, 2], s=120, c="#4C72B0", zorder=3)
    for o in objects:
        ax.annotate(o["object"][:10], (o["position"][0], o["position"][2]), fontsize=7)
    for v in views:
        pos = v["cam"]["pos"]
        score = v["visibility_score"]
        ax.scatter(pos[0], pos[2], marker="^", s=40 + 120 * score, c="#C44E52", alpha=0.7)
    ax.set_xlabel("x")
    ax.set_ylabel("z (depth)")
    ax.set_title("Virtual cameras (top-down) - size = visibility")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
