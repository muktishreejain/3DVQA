"""Viewpoint relevance scoring: S_v = R_v - U_v."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from utils.base_stage import BaseStage
from utils.io import save_json
from utils.visualization import score_bar_chart


class ScoringStage(BaseStage):
    stage_name = "view_scores"

    def run(self, context: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
        views = context.get("views", [])
        graph = context.get("scene_graph", {})
        uncertainty = np.array(context.get("uncertainty_scores", [[0.2]]))
        if not views:
            raise ValueError("views required for scoring")
        cfg = self.config.get("scoring", {})
        ranked = []
        for v in views:
            r_v = _relational_visibility(v, graph, context.get("segmentations", []))
            u_v = _viewpoint_ambiguity(v, uncertainty)
            score = r_v - u_v
            ranked.append({
                "view": v["view_id"],
                "angle": v.get("angle"),
                "R_v": round(r_v, 4),
                "U_v": round(u_v, 4),
                "score": round(score, 4),
            })
        ranked.sort(key=lambda x: x["score"], reverse=True)
        save_json(ranked, output_dir / "view_scores.json")
        score_bar_chart(ranked, output_dir / "ranked_views.png")
        _save_comparison_panel(views, ranked, output_dir / "score_overlay_panel.png")
        return {"ranked_scores": ranked}


def _relational_visibility(view: Dict, graph: Dict, segs: List) -> float:
    angle = view.get("angle", "")
    edge_boost = min(len(graph.get("edges", [])) * 0.05, 0.3)
    angle_boost = {"+60": 0.15, "-60": 0.15, "rear": 0.2, "top": 0.1}.get(angle, 0.05)
    vis = np.mean([s.get("visible_ratio", 0.5) for s in segs]) if segs else 0.5
    w = 0.6
    return float(np.clip(w * vis + edge_boost + angle_boost, 0, 1))


def _viewpoint_ambiguity(view: Dict, uncertainty: np.ndarray) -> float:
    if uncertainty.size == 0:
        return 0.2
    return float(np.mean(uncertainty))


def _save_comparison_panel(views: List, ranked: List, path: Path) -> None:
    from utils.visualization import view_grid

    order = {r["view"]: r["score"] for r in ranked}
    imgs, labels = [], []
    for v in views:
        vid = v["view_id"]
        if "image" in v:
            imgs.append(v["image"])
            labels.append(f"{vid} S={order.get(vid, 0):.2f}")
    if imgs:
        view_grid(imgs, labels, path)
