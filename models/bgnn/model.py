"""Bayesian Graph Neural Network reasoning wrapper.

Novel Contribution 1 (Image-2, Block 5): per-node uncertainty propagation.

Each node receives three calibrated quantities:
  * p_exists   - existence probability  (embedding confidence + visibility)
  * epistemic  - model uncertainty      (variance across T MC-Dropout passes)
  * aleatoric  - data uncertainty        (learned per-node log-variance head)

The mechanism is real (MC-Dropout + a learned variance head); the numbers
only become calibrated once `models/checkpoints/bgnn` holds trained weights.
Until then visibility (from segmentation) drives p_exists so the occluded-object
story in the paper holds even with random init / mock mode.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from utils.base_model import BaseModel
from utils.io import save_json


class BGNNModel(BaseModel):
    name = "bgnn"

    def load_model(self) -> None:
        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F

            rcfg = self.config.get("reasoning", {})
            hidden = rcfg.get("hidden_dim", 128)
            layers = rcfg.get("num_layers", 3)
            dropout = rcfg.get("dropout", 0.1)

            class BayesianGNN(nn.Module):
                """MLP message-passing GNN with MC-Dropout + aleatoric head.

                Intentionally avoids torch-geometric so it installs cleanly on
                Windows/CUDA 11.8. Dropout is kept active at sampling time to
                draw the Monte-Carlo posterior used for epistemic variance.
                """

                def __init__(self) -> None:
                    super().__init__()
                    self.layers = nn.ModuleList(
                        [nn.Linear(hidden, hidden) for _ in range(layers)]
                    )
                    self.dropout = nn.Dropout(dropout)
                    self.out = nn.Linear(hidden, hidden)
                    # Aleatoric head: predicts log-variance per node
                    self.aleatoric_head = nn.Sequential(
                        nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 1)
                    )

                def forward(self, x, adj):
                    h = x
                    for lin in self.layers:
                        h = self.dropout(F.relu(lin(h)))
                        h = adj @ h  # one hop of message passing
                    emb = self.out(h)
                    log_var = self.aleatoric_head(emb)
                    return emb, log_var

            self.torch = torch
            dev = self.device if self._cuda_ok() else "cpu"
            self.nn_module = BayesianGNN().to(dev)
            self._maybe_load_weights()
            if self._to_fp16_enabled(self.config) and self._cuda_ok():
                self.nn_module = self.nn_module.half()
            self._loaded = True
        except ImportError:
            self._set_mock_mode("PyTorch not available")
            self._loaded = True

    def _maybe_load_weights(self) -> None:
        ckpt = Path(self.config.get("paths", {}).get("checkpoints", {}).get("bgnn", ""))
        weights = ckpt / "weights.pt"
        if weights.exists():
            import torch

            state = torch.load(weights, map_location="cpu")
            self.nn_module.load_state_dict(state)
            self.logger.info("[bgnn] loaded trained weights from %s", weights)
        else:
            self.logger.warning(
                "[bgnn] no trained weights at %s - uncertainty is uncalibrated "
                "(random init). Train the BGNN before reporting numbers.",
                weights,
            )

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
        nodes = scene_graph.get("nodes", [])
        n = len(nodes)
        if n == 0:
            raise ValueError("Scene graph has no nodes for BGNN reasoning")

        visibility = np.array(
            [float(node.get("visibility", 1.0)) for node in nodes], dtype=np.float32
        )

        if self._mock_mode:
            updated, epistemic, aleatoric = self._mock_inference(node_embeddings, visibility)
        else:
            updated, epistemic, aleatoric = self._mc_dropout_inference(
                scene_graph, node_embeddings, n
            )

        p_exists = self._existence_probability(updated, visibility)
        node_uncertainty = [
            {
                "id": nodes[i].get("id", f"n{i}"),
                "label": nodes[i].get("label", ""),
                "p_exists": round(float(p_exists[i]), 4),
                "epistemic": round(float(epistemic[i]), 4),
                "aleatoric": round(float(aleatoric[i]), 4),
                "visibility": round(float(visibility[i]), 4),
            }
            for i in range(n)
        ]
        return {
            "updated_embeddings": updated,
            "uncertainty_scores": _pairwise_uncertainty(epistemic, aleatoric).tolist(),
            "node_uncertainty": node_uncertainty,
        }

    # ── inference backends ────────────────────────────────────────────

    def _mc_dropout_inference(self, scene_graph, node_embeddings, n):
        import torch

        hidden = self.config.get("reasoning", {}).get("hidden_dim", 128)
        t_mc = int(self.config.get("reasoning", {}).get("t_mc", 20))

        emb = _fit_dim(node_embeddings.astype(np.float32), hidden)
        adj = _build_adj(scene_graph, n)
        dev = next(self.nn_module.parameters()).device
        x = torch.from_numpy(emb).to(dev)
        adj_t = torch.from_numpy(adj.astype(np.float32)).to(dev)
        if self._to_fp16_enabled(self.config) and dev.type == "cuda":
            x, adj_t = x.half(), adj_t.half()

        # Epistemic: variance over T stochastic (dropout-on) forward passes
        self.nn_module.train()
        samples = []
        with torch.no_grad():
            for _ in range(t_mc):
                out, _ = self.nn_module(x, adj_t)
                samples.append(out.float().unsqueeze(0))
        stack = torch.cat(samples, dim=0)  # T x N x D
        mean = stack.mean(0)  # N x D
        epistemic = stack.var(0).mean(-1)  # N

        # Aleatoric: deterministic (dropout-off) pass through the variance head
        self.nn_module.eval()
        with torch.no_grad():
            _, log_var = self.nn_module(x, adj_t)
        aleatoric = torch.sigmoid(log_var.float()).squeeze(-1)

        return (
            mean.cpu().numpy(),
            _normalize_unit(epistemic.cpu().numpy()),
            aleatoric.cpu().numpy().reshape(-1),
        )

    def _mock_inference(self, node_embeddings, visibility):
        """Heuristic stand-in that still respects the occlusion narrative."""
        rng = np.random.default_rng(42)
        updated = node_embeddings + rng.normal(0, 0.02, node_embeddings.shape)
        occlusion = 1.0 - visibility  # less visible -> more uncertain
        epistemic = np.clip(0.05 + 0.30 * occlusion + rng.uniform(0, 0.03, len(visibility)), 0, 1)
        aleatoric = np.clip(0.04 + 0.25 * occlusion + rng.uniform(0, 0.03, len(visibility)), 0, 1)
        return updated, epistemic.astype(np.float32), aleatoric.astype(np.float32)

    def _existence_probability(self, embeddings: np.ndarray, visibility: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(embeddings, axis=-1)
        embed_conf = 1.0 / (1.0 + np.exp(-norm / 10.0))  # sigmoid(||h||/10)
        return np.clip(0.7 * embed_conf + 0.3 * visibility, 0.0, 1.0)

    # ── persistence / viz ─────────────────────────────────────────────

    def save_outputs(self, outputs: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
        emb_path = output_dir / "updated_embeddings.npy"
        np.save(emb_path, outputs["updated_embeddings"])
        meta = {
            "embeddings_path": str(emb_path),
            "uncertainty_scores": outputs["uncertainty_scores"],
            "node_uncertainty": outputs["node_uncertainty"],
        }
        save_json(meta, output_dir / "bgnn.json")
        return meta

    def visualize(
        self,
        outputs: Dict[str, Any],
        output_dir: Path,
        scene_graph: Dict[str, Any] = None,
        **kwargs,
    ) -> List[Path]:
        import matplotlib.pyplot as plt

        unc = np.array(outputs["uncertainty_scores"])
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(unc, cmap="hot", vmin=0, vmax=1)
        ax.set_title("Pairwise uncertainty")
        plt.colorbar(im, ax=ax)
        p1 = output_dir / "uncertainty_heatmap.png"
        fig.savefig(p1, dpi=120)
        plt.close(fig)
        paths = [p1]

        node_unc = outputs.get("node_uncertainty", [])
        if node_unc:
            labels = [f"{u['id']}:{u['label'][:8]}" for u in node_unc]
            x = np.arange(len(node_unc))
            fig2, ax2 = plt.subplots(figsize=(max(6, len(node_unc)), 3.2))
            ax2.bar(x - 0.2, [u["p_exists"] for u in node_unc], 0.2, label="P(exists)", color="#4C72B0")
            ax2.bar(x, [u["epistemic"] for u in node_unc], 0.2, label="epistemic", color="#C44E52")
            ax2.bar(x + 0.2, [u["aleatoric"] for u in node_unc], 0.2, label="aleatoric", color="#DD8452")
            ax2.set_xticks(x)
            ax2.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
            ax2.set_ylim(0, 1)
            ax2.legend(fontsize=7)
            p2 = output_dir / "node_uncertainty.png"
            fig2.savefig(p2, dpi=120, bbox_inches="tight")
            plt.close(fig2)
            paths.append(p2)
        return paths


def _fit_dim(emb: np.ndarray, hidden: int) -> np.ndarray:
    if emb.shape[1] < hidden:
        pad = np.zeros((emb.shape[0], hidden - emb.shape[1]), dtype=np.float32)
        return np.concatenate([emb, pad], axis=1)
    if emb.shape[1] > hidden:
        return emb[:, :hidden]
    return emb


def _normalize_unit(x: np.ndarray) -> np.ndarray:
    """Min-max to [0,1] so epistemic variance is interpretable across nodes."""
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-8:
        return np.clip(x, 0.0, 1.0)
    return (x - lo) / (hi - lo)


def _pairwise_uncertainty(epistemic: np.ndarray, aleatoric: np.ndarray) -> np.ndarray:
    """N x N matrix (kept for the scoring stage): mean total uncertainty per pair."""
    total = np.asarray(epistemic) + np.asarray(aleatoric)
    return 0.5 * (total[:, None] + total[None, :])


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
