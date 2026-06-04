"""Uncertainty-aware viewpoint scoring: S(v) = IG(v) - lambda * U(v).

Consumes the analytic-reprojection views (per-object visibility) and the BGNN
per-node uncertainty. Information gain rewards making query-relevant, likely-to-
exist objects visible; the penalty discourages views dominated by uncertain
objects.
"""

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
        if not views:
            raise ValueError("views required for scoring")

        graph = context.get("scene_graph", {})
        query = context.get("query", {})
        lam = float(self.config.get("scoring", {}).get("lambda_uncertainty", 0.5))

        unc_by_id, pexist_by_id = _node_uncertainty(context, graph)
        anchor_ids, target_ids = _relevant_node_ids(query, graph)

        ranked = []
        for v in views:
            per_obj_vis = v.get("per_obj_vis", {})
            ig = _information_gain(per_obj_vis, pexist_by_id, anchor_ids, target_ids)
            u_v = _viewpoint_uncertainty(per_obj_vis, unc_by_id)
            score = ig - lam * u_v
            ranked.append({
                "view": v["view_id"],
                "angle": v.get("angle"),
                "IG": round(float(ig), 4),
                "U_v": round(float(u_v), 4),
                "R_v": round(float(ig), 4),  # back-compat alias
                "score": round(float(score), 4),
            })

        ranked.sort(key=lambda x: x["score"], reverse=True)
        save_json(ranked, output_dir / "view_scores.json")
        score_bar_chart(ranked, output_dir / "ranked_views.png")
        return {"ranked_scores": ranked}


def _node_uncertainty(context: Dict, graph: Dict):
    """Return {node_id: epistemic+aleatoric} and {node_id: p_exists}."""
    unc, pexist = {}, {}
    for u in context.get("node_uncertainty", []):
        unc[u["id"]] = float(u.get("epistemic", 0.0)) + float(u.get("aleatoric", 0.0))
        pexist[u["id"]] = float(u.get("p_exists", 1.0))
    for node in graph.get("nodes", []):
        nid = node["id"]
        if nid not in unc and "epistemic" in node:
            unc[nid] = float(node.get("epistemic", 0.0)) + float(node.get("aleatoric", 0.0))
        pexist.setdefault(nid, float(node.get("p_exists", 1.0)))
        unc.setdefault(nid, 0.2)
    return unc, pexist


def _relevant_node_ids(query: Dict, graph: Dict):
    """Identify anchor/target nodes by fuzzy label match against the query."""
    nodes = graph.get("nodes", [])
    anchor_text = (query.get("target") or "").lower()
    anchor_ids = {n["id"] for n in nodes if anchor_text and anchor_text in n.get("label", "").lower()}
    # targets = nodes reachable from the anchor via the query relation
    relation = (query.get("relation") or "").replace(" ", "_")
    target_ids = set()
    if anchor_ids and relation:
        for e in graph.get("edges", []):
            if e.get("source") in anchor_ids and e.get("relation") == relation:
                target_ids.add(e.get("target"))
    return anchor_ids, target_ids


def _information_gain(per_obj_vis, pexist, anchor_ids, target_ids) -> float:
    if not per_obj_vis:
        return 0.0
    total = 0.0
    for nid, vis in per_obj_vis.items():
        relevance = 1.0
        if nid in target_ids:
            relevance = 2.0
        elif nid in anchor_ids:
            relevance = 1.5
        total += float(vis) * relevance * pexist.get(nid, 1.0)
    return total / len(per_obj_vis)


def _viewpoint_uncertainty(per_obj_vis, unc_by_id) -> float:
    visible = [unc_by_id.get(nid, 0.2) for nid, vis in per_obj_vis.items() if vis > 0.1]
    return float(np.mean(visible)) if visible else 1.0
