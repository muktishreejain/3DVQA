from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from models.sam.model import SAMModel
from utils.base_stage import BaseStage
from utils.validation import validate_detections


class SegmentationStage(BaseStage):
    stage_name = "segmentation"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = SAMModel(config, device=config.get("device", "cuda"))

    def run(self, context: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
        image = context["image"]
        detections = validate_detections(context.get("detections", []))
        segmentations = self.model.run_inference(image, detections)
        meta = self.model.save_outputs(segmentations, output_dir)
        self.model.visualize(segmentations, output_dir, image=image)
        return {"segmentations": segmentations, "segmentation_meta": meta}
