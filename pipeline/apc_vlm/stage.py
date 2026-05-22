"""APC-VLM downstream integration (isolated from geometry pipeline)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np

from apc_vlm.integration import pack_apc_features, run_apc_reasoning
from utils.base_stage import BaseStage
from utils.io import save_json


class APCVLMStage(BaseStage):
    stage_name = "apc_vlm"

    def run(self, context: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
        fusion = context.get("fusion_export")
        if not fusion:
            raise ValueError("fusion_export required before APC-VLM integration")

        visual = np.load(fusion["visual_features"])
        graph_f = np.load(fusion["graph_features"])
        text_f = np.load(fusion["text_features"])
        metadata = {
            "question": context.get("question"),
            "query": context.get("query"),
            "selected_views": context.get("selection", {}).get("selected_views", []),
            "scene_graph": context.get("scene_graph"),
        }

        bundle = pack_apc_features(visual, graph_f, text_f, metadata)
        save_json(bundle, output_dir / "apc_input_bundle.json")
        result: Dict[str, Any] = {
            "status": "packed",
            "apc_input_bundle": str(output_dir / "apc_input_bundle.json"),
            "reasoning_output": None,
        }

        if self.config.get("apc_vlm", {}).get("run_reasoning", False):
            out = run_apc_reasoning(bundle, self.config)
            save_json(out, output_dir / "apc_reasoning.json")
            result["reasoning_output"] = out
            result["apc_reasoning_path"] = str(output_dir / "apc_reasoning.json")

        save_json(result, output_dir / "apc_vlm_integration.json")
        return {"apc_vlm_result": result}
