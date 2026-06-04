from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from utils.base_stage import BaseStage
from utils.io import save_json


class SelectionStage(BaseStage):
    stage_name = "selected_views"

    def run(self, context: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
        ranked = context.get("ranked_scores", [])
        views = context.get("views", [])
        if not ranked:
            raise ValueError("ranked_scores required for selection")
        score_map = {r["view"]: r for r in ranked}
        top_k = self.config.get("selection", {}).get("top_k", 3)
        selected_ids = [r["view"] for r in ranked[:top_k]]
        view_map = {v["view_id"]: v for v in views}
        selected = [view_map[vid] for vid in selected_ids if vid in view_map]
        result = {
            "selected_views": [
                {
                    "view_id": v["view_id"],
                    "angle": v.get("angle"),
                    "azimuth_deg": v.get("azimuth_deg"),
                    "score": score_map.get(v["view_id"], {}).get("score"),
                    "per_obj_vis": v.get("per_obj_vis", {}),
                    "cam": v.get("cam"),
                }
                for v in selected
            ],
            "rejected_views": [r["view"] for r in ranked[top_k:]],
        }
        save_json(result, output_dir / "selected_views.json")
        return {"selection": result, "selected_views": result["selected_views"]}
