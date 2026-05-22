from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np

from utils.base_stage import BaseStage
from utils.depth_factory import create_depth_model
from utils.validation import validate_masks


class DepthStage(BaseStage):
    stage_name = "depth"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = create_depth_model(config, device=config.get("device", "cuda"))

    def run(self, context: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
        image = context["image"]
        segs = validate_masks(context.get("segmentations", []))
        query_mask = _build_query_mask(segs, context.get("query"), image.shape[:2], self.config)
        depth_out = self.model.run_inference(image, segs, query_mask=query_mask)
        meta = self.model.save_outputs(depth_out, output_dir)
        self.model.visualize(depth_out, output_dir)
        return {
            "depth_map": depth_out["depth_map"],
            "object_depths": depth_out["object_depths"],
            "depth_meta": meta,
        }


def _build_query_mask(
    segmentations: list,
    query: Dict[str, Any] | None,
    shape: tuple,
    config: Dict[str, Any],
) -> np.ndarray | None:
    if not config.get("depth", {}).get("use_query_mask", True):
        return None
    h, w = shape
    if not segmentations:
        return None
    target = (query or {}).get("target")
    combined = np.zeros((h, w), dtype=np.uint8)
    for seg in segmentations:
        mask = seg.get("mask")
        if mask is None:
            continue
        label = seg.get("label", "").lower()
        if target is None or target.lower() in label or label in target.lower():
            combined = np.maximum(combined, mask.astype(np.uint8))
    if combined.sum() == 0:
        for seg in segmentations:
            if seg.get("mask") is not None:
                combined = np.maximum(combined, seg["mask"].astype(np.uint8))
    return combined if combined.any() else None
