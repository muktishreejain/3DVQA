from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from models.omni3d.model import Omni3DModel
from utils.base_stage import BaseStage


class LiftingStage(BaseStage):
    stage_name = "lifting"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = Omni3DModel(config, device=config.get("device", "cuda"))

    def run(self, context: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
        segs = context.get("segmentations", [])
        depth_map = context.get("depth_map")
        if depth_map is None:
            raise ValueError("depth_map required for lifting")
        lifted = self.model.run_inference(
            segs,
            depth_map,
            context.get("object_depths", {}),
            views=context.get("views"),
        )
        meta = self.model.save_outputs(lifted, output_dir)
        self.model.visualize(lifted, output_dir)
        return {"lifted_objects": lifted, "lifting_meta": meta}
