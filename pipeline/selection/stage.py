from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from utils.base_stage import BaseStage
from utils.io import save_json
from utils.visualization import view_grid


class SelectionStage(BaseStage):
    stage_name = "selected_views"

    def run(self, context: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
        ranked = context.get("ranked_scores", [])
        views = context.get("views", [])
        if not ranked:
            raise ValueError("ranked_scores required for selection")
        top_k = self.config.get("selection", {}).get("top_k", 3)
        selected_ids = [r["view"] for r in ranked[:top_k]]
        view_map = {v["view_id"]: v for v in views}
        selected = [view_map[vid] for vid in selected_ids if vid in view_map]
        rejected = [v for v in views if v["view_id"] not in selected_ids]
        result = {
            "selected_views": [
                {"view_id": v["view_id"], "angle": v.get("angle"), "path": v.get("path")}
                for v in selected
            ],
            "rejected_views": [v["view_id"] for v in rejected],
        }
        save_json(result, output_dir / "selected_views.json")
        if selected and rejected:
            imgs = [v["image"] for v in selected[:3] if "image" in v]
            lbls = [f"SEL {v['view_id']}" for v in selected[:3] if "image" in v]
            rej_imgs = [v["image"] for v in rejected[:3] if "image" in v]
            rej_lbls = [f"REJ {v['view_id']}" for v in rejected[:3] if "image" in v]
            view_grid(imgs + rej_imgs, lbls + rej_lbls, output_dir / "selected_vs_rejected.png")
        return {"selection": result}
