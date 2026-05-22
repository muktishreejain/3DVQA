"""
APC-VLM compatibility layer (downstream reasoning only).

Fork APC-VLM into apc_vlm/APC-VLM/ on Windows. Do not embed geometry modules here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def pack_apc_features(
    visual: np.ndarray,
    graph: np.ndarray,
    text: np.ndarray,
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "modality": "3dvqa_structured",
        "platform": "windows",
        "visual_features_shape": list(visual.shape),
        "graph_features_shape": list(graph.shape),
        "text_features_shape": list(text.shape),
        "visual_features_sample": visual[: min(8, len(visual))].tolist(),
        "graph_features_sample": graph[: min(8, len(graph))].tolist(),
        "metadata": metadata,
        "cot_ready": True,
    }


def run_apc_reasoning(bundle: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Optional hook into APC-VLM inference.
    Uses apc_vlm/APC-VLM when installed; otherwise returns structured placeholder.
    """
    apc_root = Path(__file__).parent / "APC-VLM"
    if not apc_root.exists():
        return {
            "answer": None,
            "note": "APC-VLM fork not found under apc_vlm/APC-VLM",
            "bundle_keys": list(bundle.keys()),
        }
    return {
        "answer": None,
        "note": "Connect apc.apc_pipeline here for full CoT reasoning",
        "checkpoint": config.get("paths", {}).get("checkpoints", {}).get("apc_vlm"),
    }
