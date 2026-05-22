from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from models.bgnn.model import BGNNModel
from utils.base_stage import BaseStage


class ReasoningStage(BaseStage):
    stage_name = "bgnn"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = BGNNModel(config, device=config.get("device", "cuda"))

    def run(self, context: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
        graph = context.get("scene_graph")
        embeddings = context.get("node_embeddings")
        if graph is None or embeddings is None:
            raise ValueError("scene_graph and node_embeddings required for BGNN")
        out = self.model.run_inference(graph, embeddings)
        meta = self.model.save_outputs(out, output_dir)
        self.model.visualize(out, output_dir, scene_graph=graph)
        return {
            "graph_embeddings": out["updated_embeddings"],
            "uncertainty_scores": out["uncertainty_scores"],
            "bgnn_meta": meta,
        }
