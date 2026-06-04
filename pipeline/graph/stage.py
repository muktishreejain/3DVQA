"""Scene graph construction from 3D lifted objects."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from utils.base_stage import BaseStage
from utils.io import save_json
from utils.visualization import draw_scene_graph


class GraphStage(BaseStage):
    stage_name = "graphs"

    def run(self, context: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
        objects = context.get("lifted_objects", [])
        if not objects:
            raise ValueError("lifted_objects required for graph construction")
        relations = self.config.get("graph", {}).get("relations", [])
        graph = build_scene_graph(objects, relations, self.config.get("graph", {}))
        save_json(graph, output_dir / "scene_graph.json")
        draw_scene_graph(graph, output_dir / "scene_graph.png")
        embeddings = _node_embeddings(graph)
        np.save(output_dir / "node_embeddings.npy", embeddings)
        return {
            "scene_graph": graph,
            "node_embeddings": embeddings,
        }


def build_scene_graph(
    objects: List[Dict[str, Any]],
    allowed_relations: List[str],
    graph_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    nodes = []
    for i, obj in enumerate(objects):
        nodes.append({
            "id": f"n{i}",
            "label": obj["object"],
            "position": obj["position"],
            "size_3d": obj.get("size_3d", [0.1, 0.1, 0.1]),
            "visibility": round(float(obj.get("visibility", 1.0)), 4),
        })
    edges = []
    near_th = graph_cfg.get("depth_threshold_near", 0.15)
    behind_th = graph_cfg.get("depth_threshold_behind", 0.08)
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if i == j:
                continue
            pi, pj = np.array(nodes[i]["position"]), np.array(nodes[j]["position"])
            dist = float(np.linalg.norm(pi - pj))
            rels = []
            if pi[2] > pj[2] + behind_th and "behind" in allowed_relations:
                rels.append(("behind", i, j))
            if pi[2] < pj[2] - behind_th and "in_front_of" in allowed_relations:
                rels.append(("in_front_of", i, j))
            if pi[0] < pj[0] - 0.05 and "left_of" in allowed_relations:
                rels.append(("left_of", i, j))
            if pi[0] > pj[0] + 0.05 and "right_of" in allowed_relations:
                rels.append(("right_of", i, j))
            if dist < near_th and "near" in allowed_relations:
                rels.append(("near", i, j))
            if dist >= near_th and "far" in allowed_relations:
                rels.append(("far", i, j))
            if dist < near_th * 0.5 and "overlap" in allowed_relations:
                rels.append(("overlap", i, j))
            # occluded_by: j is in front of i (smaller depth) and laterally aligned,
            # while i is itself poorly visible.
            if (
                pi[2] < pj[2] - behind_th
                and abs(pi[0] - pj[0]) < 0.1
                and nodes[i].get("visibility", 1.0) < 0.7
                and "occluded_by" in allowed_relations
            ):
                rels.append(("occluded_by", i, j))
            for rel, src, tgt in rels:
                edges.append({
                    "source": nodes[src]["id"],
                    "target": nodes[tgt]["id"],
                    "relation": rel,
                    "confidence": round(max(0.5, 1.0 - dist), 3),
                })
    return {"nodes": nodes, "edges": edges}


def _node_embeddings(graph: Dict[str, Any], dim: int = 128) -> np.ndarray:
    n = len(graph["nodes"])
    emb = np.zeros((n, dim), dtype=np.float32)
    for i, node in enumerate(graph["nodes"]):
        pos = node["position"]
        emb[i, :3] = pos
        emb[i, 3] = hash(node["label"]) % 1000 / 1000.0
        emb[i, 4] = float(node.get("visibility", 1.0))
    return emb
