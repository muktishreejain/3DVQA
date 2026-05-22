"""Stable Zero123 multi-view synthesis (Wonder3D not used)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from models.zero123.model import Zero123Model
from utils.base_stage import BaseStage
from utils.io import save_image, save_json
from utils.visualization import view_grid


class MultiviewStage(BaseStage):
    stage_name = "views"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = Zero123Model(config, device=config.get("device", "cuda"))

    def run(self, context: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
        image = context["image"]
        depth_map = context.get("depth_map")
        if depth_map is None:
            raise ValueError("depth_map missing; run depth stage first")
        masks = _load_masks(context.get("segmentations", []))
        viewpoints = self.config.get("multiview", {}).get(
            "viewpoints", ["+60", "-60", "+120", "-120", "rear", "top"]
        )
        views = self.model.run_inference(image, depth_map, masks, viewpoints)
        for i, v in enumerate(views):
            v["view_id"] = f"view_{i + 1:02d}"
        saved = []
        for v in views:
            p = output_dir / f"{v['view_id']}_{v['angle']}.png"
            save_image(v["image"], p)
            v["path"] = str(p)
            saved.append({"view_id": v["view_id"], "angle": v["angle"], "path": v["path"]})
        save_json(saved, output_dir / "views.json")
        meta = self.model.save_outputs(views, output_dir / "zero123")
        self.model.visualize(views, output_dir)
        view_grid(
            [v["image"] for v in views],
            [f"{v['view_id']} ({v['angle']})" for v in views],
            output_dir / "view_comparison_panel.png",
        )
        return {"views": views, "views_meta": meta}


def _load_masks(segmentations: list) -> list:
    import numpy as np
    from PIL import Image

    masks = []
    for seg in segmentations:
        if "mask" in seg:
            masks.append(seg["mask"])
        elif "mask_path" in seg:
            masks.append(np.array(Image.open(seg["mask_path"])) > 127)
    return masks
