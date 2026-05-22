from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from models.grounding_dino.model import GroundingDINOModel
from utils.base_stage import BaseStage
from utils.validation import validate_question


class DetectionStage(BaseStage):
    stage_name = "detection"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = GroundingDINOModel(config, device=config.get("device", "cuda"))

    def run(self, context: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
        image = context["image"]
        question = validate_question(context["question"])
        result = self.model.run_inference(image, question)
        meta = self.model.save_outputs(result, output_dir)
        self.model.visualize(result, output_dir, image=image)
        return {
            "detections": result["detections"],
            "query": result["query"],
            "detection_meta": meta,
        }
