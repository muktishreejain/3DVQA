"""Graph-CoT reasoning (Image-2, Block 7).

Novel Contribution 3: a query-guided walk over the 3D scene graph that yields a
human-readable reasoning path - the explainability artifact the paper highlights
(e.g. "red cylinder --behind--> green cube --color--> green"). It also selects
the evidence view (the chosen virtual camera where the target is most visible)
and discounts answer confidence by the target's BGNN uncertainty.

This runs purely on the graph + uncertainty + reprojection outputs; it does not
call the VLM. The VLM/fusion stages consume `reasoning_path` downstream.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from utils.base_stage import BaseStage
from utils.io import save_json


class GraphCoTStage(BaseStage):
    stage_name = "graph_cot"

    def run(self, context: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
        graph = context.get("scene_graph") or {}
        nodes = {n["id"]: n for n in graph.get("nodes", [])}
        query = context.get("query", {})
        selected = context.get("selection", {}).get("selected_views", [])

        path: List[Dict[str, Any]] = []
        anchor = _find_node(nodes, query.get("target"))

        if anchor is None:
            path.append({"step": "ERROR", "subject": "anchor_not_found",
                         "detail": query.get("target", "")})
            result = _assemble(path, None, "", 0.0)
            save_json(result, output_dir / "graph_cot.json")
            return {**result, "reasoning_path": path}

        path.append({"step": "FIND", "subject": anchor.get("label", anchor["id"]),
                     "detail": f"P(exists)={anchor.get('p_exists', 1.0):.2f}"})

        relation = (query.get("relation") or "").replace(" ", "_")
        neighbors = _neighbors_by_relation(graph, anchor["id"], relation)
        if not neighbors:
            neighbors = _geometric_behind(graph, anchor)
            if neighbors:
                path.append({"step": "INFER", "subject": f"no '{relation}' edge",
                             "detail": "fell back to geometric depth estimate"})

        if not neighbors:
            path.append({"step": "ERROR", "subject": "target_not_found", "detail": ""})
            result = _assemble(path, None, "", 0.0)
            save_json(result, output_dir / "graph_cot.json")
            return {**result, "reasoning_path": path}

        target = max(neighbors, key=lambda n: n.get("p_exists", 1.0))
        path.append({"step": "TRAVERSE", "subject": relation or "near",
                     "detail": target.get("label", target["id"])})
        path.append({
            "step": "CONFIDENCE",
            "subject": f"P(exists)={target.get('p_exists', 1.0):.2f}",
            "detail": f"epistemic={target.get('epistemic', 0.0):.2f} "
                      f"aleatoric={target.get('aleatoric', 0.0):.2f}",
        })

        attribute = query.get("attribute") or "object"
        attr_value = _read_attribute(target, attribute)
        path.append({"step": "READ", "subject": attribute, "detail": attr_value})

        evidence_view_id = _best_view_for(target["id"], selected)
        confidence = _discounted_confidence(target)

        result = _assemble(path, target, evidence_view_id, confidence)
        save_json(result, output_dir / "graph_cot.json")
        self.logger.info(
            "Graph-CoT: %s --[%s]--> %s | %s=%s | conf=%.2f",
            anchor.get("label"), relation or "near", target.get("label"),
            attribute, attr_value, confidence,
        )
        return {**result, "reasoning_path": path}


# ── graph helpers ─────────────────────────────────────────────────────


def _find_node(nodes: Dict[str, Dict], text: Optional[str]) -> Optional[Dict]:
    if not text:
        return None
    text = text.lower().strip()
    # exact-ish containment first, then any token overlap
    for node in nodes.values():
        if text in node.get("label", "").lower():
            return node
    tokens = set(text.split())
    best, best_overlap = None, 0
    for node in nodes.values():
        overlap = len(tokens & set(node.get("label", "").lower().split()))
        if overlap > best_overlap:
            best, best_overlap = node, overlap
    return best


def _neighbors_by_relation(graph: Dict, src_id: str, relation: str) -> List[Dict]:
    if not relation:
        return []
    nodes = {n["id"]: n for n in graph.get("nodes", [])}
    out = []
    for e in graph.get("edges", []):
        if e.get("source") == src_id and e.get("relation") == relation:
            tgt = nodes.get(e.get("target"))
            if tgt is not None:
                out.append(tgt)
    return out


def _geometric_behind(graph: Dict, anchor: Dict) -> List[Dict]:
    """Fallback: objects with greater depth (z) than the anchor."""
    az = anchor.get("position", [0, 0, 0])[2]
    out = []
    for n in graph.get("nodes", []):
        if n["id"] == anchor["id"]:
            continue
        if n.get("position", [0, 0, 0])[2] - az > 0.02:
            out.append(n)
    return out


def _read_attribute(node: Dict, attribute: str) -> str:
    label = node.get("label", "")
    colors = ["red", "blue", "green", "gray", "grey", "purple", "cyan", "yellow", "brown"]
    shapes = ["cube", "sphere", "ball", "cylinder", "cone", "block"]
    attr = attribute.lower()
    if attr == "color":
        return next((c for c in colors if c in label.lower()), "unknown")
    if attr == "shape":
        return next((s for s in shapes if s in label.lower()), label.split()[-1] if label else "unknown")
    if attr == "size":
        size = node.get("size_3d", [0.1, 0.1, 0.1])
        return "large" if float(np.prod(size)) > 0.01 else "small"
    return label or "unknown"


def _best_view_for(node_id: str, selected_views: List[Dict]) -> str:
    best_id, best_vis = "", -1.0
    for v in selected_views:
        vis = float(v.get("per_obj_vis", {}).get(node_id, 0.0))
        if vis > best_vis:
            best_vis, best_id = vis, v.get("view_id", "")
    return best_id


def _discounted_confidence(target: Dict, base: float = 0.9) -> float:
    u = float(target.get("epistemic", 0.0)) + float(target.get("aleatoric", 0.0))
    return float(np.clip(base * np.exp(-u), 0.0, 1.0))


def _assemble(path, target, evidence_view_id, confidence) -> Dict[str, Any]:
    return {
        "reasoning_path": path,
        "target_object": target,
        "evidence_view_id": evidence_view_id,
        "graph_cot_confidence": round(float(confidence), 4),
    }
