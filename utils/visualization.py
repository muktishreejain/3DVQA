"""Shared visualization helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from utils.io import ensure_dir, save_image


def draw_detections(
    image: np.ndarray,
    detections: List[Dict[str, Any]],
    output_path: Path,
) -> Path:
    vis = image.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        conf = det.get("confidence", 0.0)
        label = det.get("label", "obj")
        color = (0, 255, 0) if conf > 0.5 else (255, 165, 0)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        text = f"{label}: {conf:.2f}"
        cv2.putText(vis, text, (x1, max(y1 - 5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    save_image(vis, output_path)
    return output_path


def draw_masks_overlay(
    image: np.ndarray,
    masks: List[np.ndarray],
    labels: List[str],
    output_path: Path,
    alpha: float = 0.45,
) -> Path:
    vis = image.copy().astype(np.float32)
    rng = np.random.default_rng(42)
    for mask, label in zip(masks, labels):
        color = rng.integers(50, 255, size=3)
        m = mask.astype(bool)
        vis[m] = vis[m] * (1 - alpha) + color * alpha
    save_image(vis.astype(np.uint8), output_path)
    return output_path


def depth_colormap(depth: np.ndarray, output_path: Path) -> Path:
    d = depth.astype(np.float32)
    d = (d - d.min()) / (d.max() - d.min() + 1e-8)
    colored = (plt.cm.plasma(d)[:, :, :3] * 255).astype(np.uint8)
    save_image(colored, output_path)
    return output_path


def view_grid(images: List[np.ndarray], labels: List[str], output_path: Path) -> Path:
    n = len(images)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.atleast_1d(axes).flatten()
    for i, (img, lbl) in enumerate(zip(images, labels)):
        axes[i].imshow(img)
        axes[i].set_title(lbl)
        axes[i].axis("off")
    for j in range(len(images), len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def draw_scene_graph(
    graph: Dict[str, Any],
    output_path: Path,
) -> Path:
    G = nx.DiGraph()
    for node in graph.get("nodes", []):
        G.add_node(node["id"], label=node.get("label", node["id"]))
    for edge in graph.get("edges", []):
        G.add_edge(
            edge["source"],
            edge["target"],
            relation=edge.get("relation", ""),
            confidence=edge.get("confidence", 1.0),
        )
    pos = nx.spring_layout(G, seed=42)
    fig, ax = plt.subplots(figsize=(8, 6))
    labels = {n: G.nodes[n].get("label", n) for n in G.nodes}
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color="#6CA6CD", node_size=800)
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=9)
    edge_labels = {
        (u, v): f"{d.get('relation','')}\n{d.get('confidence',1):.2f}"
        for u, v, d in G.edges(data=True)
    }
    nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, arrowsize=15)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, ax=ax)
    ax.axis("off")
    plt.tight_layout()
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def score_bar_chart(scores: List[Dict[str, Any]], output_path: Path) -> Path:
    views = [s["view"] for s in scores]
    vals = [s["score"] for s in scores]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(views, vals, color="#4C72B0")
    ax.set_xlabel("Relevance score S_v = R_v - U_v")
    ax.invert_yaxis()
    plt.tight_layout()
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
