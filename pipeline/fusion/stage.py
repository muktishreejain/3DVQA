"""Structured multimodal feature export (feeds APC-VLM stage)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np

from utils.base_stage import BaseStage
from utils.io import save_json


class FusionStage(BaseStage):
    stage_name = "fusion"

    def run(self, context: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
        fusion_cfg = self.config.get("fusion", {})
        vdim = fusion_cfg.get("visual_dim", 512)
        gdim = fusion_cfg.get("graph_dim", 128)
        tdim = fusion_cfg.get("text_dim", 768)

        visual = _pool_view_features(context, vdim)
        graph_f = _pool_graph_features(context, gdim)
        text_f = _encode_question(context.get("question", ""), tdim)

        export = {
            "visual_features": str(output_dir / "visual_features.npy"),
            "graph_features": str(output_dir / "graph_features.npy"),
            "text_features": str(output_dir / "text_features.npy"),
            "metadata": {
                "question": context.get("question"),
                "query": context.get("query"),
                "selected_views": context.get("selection", {}).get("selected_views", []),
                "scene_graph_nodes": len(context.get("scene_graph", {}).get("nodes", [])),
            },
        }
        np.save(export["visual_features"], visual)
        np.save(export["graph_features"], graph_f)
        np.save(export["text_features"], text_f)
        save_json(export, output_dir / "fusion_export.json")
        return {"fusion_export": export}


def _pool_view_features(context: Dict, dim: int) -> np.ndarray:
    selection = context.get("selection", {}).get("selected_views", [])
    views = {v["view_id"]: v for v in context.get("views", [])}
    feats = []
    for sel in selection:
        v = views.get(sel["view_id"])
        if v is not None and "image" in v:
            img = v["image"].astype(np.float32).flatten()
            rng = np.random.default_rng(hash(sel["view_id"]) % 2**32)
            proj = rng.standard_normal((img.shape[0], dim))
            feats.append((img @ proj) / (np.linalg.norm(img @ proj) + 1e-8))
    if not feats:
        return np.zeros(dim, dtype=np.float32)
    return np.mean(feats, axis=0).astype(np.float32)


def _pool_graph_features(context: Dict, dim: int) -> np.ndarray:
    emb = context.get("graph_embeddings")
    if emb is None:
        return np.zeros(dim, dtype=np.float32)
    emb = np.asarray(emb, dtype=np.float32)
    out = emb if emb.ndim == 1 else emb.mean(axis=0)
    if out.shape[0] < dim:
        out = np.pad(out, (0, dim - out.shape[0]))
    return out[:dim].astype(np.float32)


def _encode_question(question: str, dim: int) -> np.ndarray:
    rng = np.random.default_rng(abs(hash(question)) % 2**32)
    vec = rng.standard_normal(dim).astype(np.float32)
    return vec / (np.linalg.norm(vec) + 1e-8)
