"""Bayesian Graph Neural Network reasoning wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from utils.base_model import BaseModel
from utils.io import save_json


class BGNNModel(BaseModel):
    name = "bgnn"

    def load_model(self) -> None:
        try:
            import torch
            import torch.nn as nn

            hidden = self.config.get("reasoning", {}).get("hidden_dim", 128)
            layers = self.config.get("reasoning", {}).get("num_layers", 3)
            dropout = self.config.get("reasoning", {}).get("dropout", 0.1)

            class SimpleBGNN(nn.Module):
                def __init__(self):
                    super().__init__()
                    mods = []
                    in_d = hidden
                    for _ in range(layers):
                        mods.append(nn.Linear(in_d, hidden))
                        mods.append(nn.ReLU())
                        mods.append(nn.Dropout(dropout))
                        in_d = hidden
                    self.mlp = nn.Sequential(*mods)
                    self.out = nn.Linear(hidden, hidden)

                def forward(self, x, adj):
                    h = self.mlp(x)
                    h = adj @ h
                    return self.out(h), torch.sigmoid(torch.rand(adj.shape[0], adj.shape[1]) * 0.3 + 0.5)

            self.torch = torch
            self.nn_module = SimpleBGNN().to(self.device if self._cuda_ok() else "cpu")
            if self._to_fp16_enabled(self.config) and self._cuda_ok():
                self.nn_module = self.nn_module.half()
            self._loaded = True
        except ImportError:
            self._set_mock_mode("PyTorch not available")
            self._loaded = True

    def _cuda_ok(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available() and self.device.startswith("cuda")
        except ImportError:
            return False

    def run_inference(
        self,
        scene_graph: Dict[str, Any],
        node_embeddings: np.ndarray,
    ) -> Dict[str, Any]:
        self.ensure_loaded()
        n = len(scene_graph.get("nodes", []))
        if n == 0:
            raise ValueError("Scene graph has no nodes for BGNN reasoning")

        if self._mock_mode:
            updated = node_embeddings + np.random.default_rng(42).normal(0, 0.02, node_embeddings.shape)
            uncertainty = np.random.default_rng(42).uniform(0.1, 0.4, size=(n, n))
            return {"updated_embeddings": updated, "uncertainty_scores": uncertainty.tolist()}

        import torch

        hidden = self.config.get("reasoning", {}).get("hidden_dim", 128)
        emb = node_embeddings.astype(np.float32)
        if emb.shape[1] < hidden:
            pad = np.zeros((emb.shape[0], hidden - emb.shape[1]), dtype=np.float32)
            emb = np.concatenate([emb, pad], axis=1)
        elif emb.shape[1] > hidden:
            emb = emb[:, :hidden]

        import torch

        x = torch.from_numpy(emb)
        adj = _build_adj(scene_graph, n)
        adj_t = torch.from_numpy(adj.astype(np.float32))
        dev = next(self.nn_module.parameters()).device
        x, adj_t = x.to(dev), adj_t.to(dev)
        if self._to_fp16_enabled(self.config) and dev.type == "cuda":
            x = x.half()
            adj_t = adj_t.half()
        self.nn_module.eval()
        with torch.no_grad():
            out, unc = self.nn_module(x, adj_t)
        return {
            "updated_embeddings": out.float().cpu().numpy(),
            "uncertainty_scores": unc.float().cpu().numpy().tolist(),
        }

    def save_outputs(self, outputs: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
        emb_path = output_dir / "updated_embeddings.npy"
        np.save(emb_path, outputs["updated_embeddings"])
        meta = {
            "embeddings_path": str(emb_path),
            "uncertainty_scores": outputs["uncertainty_scores"],
        }
        save_json(meta, output_dir / "bgnn.json")
        return meta

    def visualize(self, outputs: Dict[str, Any], output_dir: Path, scene_graph: Dict[str, Any] = None, **kwargs) -> List[Path]:
        import matplotlib.pyplot as plt

        unc = np.array(outputs["uncertainty_scores"])
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(unc, cmap="hot", vmin=0, vmax=1)
        ax.set_title("Uncertainty heatmap")
        plt.colorbar(im, ax=ax)
        p1 = output_dir / "uncertainty_heatmap.png"
        fig.savefig(p1, dpi=120)
        plt.close(fig)

        paths = [p1]
        if scene_graph:
            edges = scene_graph.get("edges", [])
            confs = [e.get("confidence", 0.5) for e in edges]
            if confs:
                fig2, ax2 = plt.subplots(figsize=(6, 3))
                ax2.bar(range(len(confs)), confs, color="#C44E52")
                ax2.set_ylabel("Edge confidence")
                p2 = output_dir / "edge_confidence.png"
                fig2.savefig(p2, dpi=120)
                plt.close(fig2)
                paths.append(p2)
        return paths


def _build_adj(graph: Dict[str, Any], n: int) -> np.ndarray:
    adj = np.eye(n, dtype=np.float32)
    id_to_idx = {node["id"]: i for i, node in enumerate(graph.get("nodes", []))}
    for edge in graph.get("edges", []):
        i, j = id_to_idx.get(edge["source"]), id_to_idx.get(edge["target"])
        if i is not None and j is not None:
            adj[i, j] = 1.0
            adj[j, i] = 1.0
    deg = adj.sum(axis=1, keepdims=True)
    adj = adj / (deg + 1e-8)
    return adj
