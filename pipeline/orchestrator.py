"""End-to-end pipeline orchestration (Windows-native)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from pipeline.apc_vlm.stage import APCVLMStage
from pipeline.depth.stage import DepthStage
from pipeline.detection.stage import DetectionStage
from pipeline.fusion.stage import FusionStage
from pipeline.graph.stage import GraphStage
from pipeline.graph_cot.stage import GraphCoTStage
from pipeline.lifting.stage import LiftingStage
from pipeline.multiview.stage import MultiviewStage
from pipeline.reasoning.stage import ReasoningStage
from pipeline.scoring.stage import ScoringStage
from pipeline.segmentation.stage import SegmentationStage
from pipeline.selection.stage import SelectionStage
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.io import ensure_dir, load_image, save_json
from utils.logging_setup import get_logger, log_gpu_memory
from utils.validation import validate_image, validate_question
from utils.windows import ensure_windows_paths

STAGE_REGISTRY = {
    "detection": DetectionStage,
    "segmentation": SegmentationStage,
    "depth": DepthStage,
    "views": MultiviewStage,
    "multiview": MultiviewStage,
    "lifting": LiftingStage,
    "graphs": GraphStage,
    "graph": GraphStage,
    "bgnn": ReasoningStage,
    "reasoning": ReasoningStage,
    "view_scores": ScoringStage,
    "scoring": ScoringStage,
    "selected_views": SelectionStage,
    "selection": SelectionStage,
    "graph_cot": GraphCoTStage,
    "fusion": FusionStage,
    "apc_vlm": APCVLMStage,
}


class PipelineOrchestrator:
    def __init__(self, config: Dict[str, Any]):
        ensure_windows_paths()
        self.config = config
        self.logger = get_logger()
        stages_cfg = config.get("pipeline", {}).get("stages", list(STAGE_REGISTRY.keys()))
        seen = set()
        self.stages = []
        for name in stages_cfg:
            if name in seen:
                continue
            if name == "apc_vlm" and not config.get("apc_vlm", {}).get("enabled", True):
                continue
            cls = STAGE_REGISTRY.get(name)
            if cls is None:
                self.logger.warning("Unknown stage '%s' — skipped", name)
                continue
            seen.add(name)
            self.stages.append(cls(config))

    def run_single(
        self,
        image_path: str,
        question: str,
        output_dir: Path,
        skip_stages: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        image = validate_image(load_image(image_path))
        question = validate_question(question)
        context: Dict[str, Any] = {
            "image": image,
            "image_path": str(image_path),
            "question": question,
        }
        return self._run_context(context, output_dir, skip_stages)

    def run_batch(
        self,
        samples: List[Dict[str, str]],
        output_root: Path,
        skip_stages: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        results = []
        for i, sample in enumerate(samples):
            out = output_root / f"sample_{i:04d}"
            ensure_dir(out)
            results.append(
                self.run_single(sample["image_path"], sample["question"], out, skip_stages)
            )
        return results

    def _run_context(
        self,
        context: Dict[str, Any],
        output_dir: Path,
        skip_stages: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        ensure_dir(output_dir)
        completed: List[str] = []
        ckpt = load_checkpoint(output_dir) if self.config.get("pipeline", {}).get("resume") else None
        if ckpt:
            completed = ckpt.get("completed_stages", [])
            context.update(ckpt.get("state", {}))

        skip = _normalize_skip(skip_stages)

        for stage in self.stages:
            context = stage.execute(context, output_dir, skip_stages=skip, completed=completed)
            if stage.stage_name not in completed:
                completed.append(stage.stage_name)
            if self.config.get("pipeline", {}).get("save_intermediates", True):
                save_checkpoint(output_dir, completed, _serializable_state(context))

        target = context.get("target_object") or {}
        summary = {
            "question": context.get("question"),
            "query": context.get("query"),
            "output_dir": str(output_dir),
            "answer": target.get("label") if target else None,
            "confidence": context.get("graph_cot_confidence"),
            "reasoning_path": context.get("reasoning_path", []),
            "evidence_view_id": context.get("evidence_view_id"),
            "node_uncertainty": context.get("node_uncertainty", []),
            "fusion_export": context.get("fusion_export"),
            "selection": context.get("selection"),
            "apc_vlm_result": context.get("apc_vlm_result"),
            "num_detections": len(context.get("detections", [])),
        }
        save_json(summary, output_dir / "pipeline_summary.json")
        _print_reasoning(context)
        log_gpu_memory("pipeline_complete", self.config.get("logging", {}).get("log_gpu_memory", True))
        return context


def _print_reasoning(context: Dict[str, Any]) -> None:
    path = context.get("reasoning_path")
    if not path:
        return
    target = context.get("target_object") or {}
    conf = context.get("graph_cot_confidence")
    print("\n" + "=" * 50)
    print(f"  Answer:     {(target.get('label') or 'unknown').upper()}")
    if conf is not None:
        print(f"  Confidence: {conf * 100:.0f}%")
    print("  Reasoning Path:")
    for step in path:
        print(f"    [{step.get('step')}] {step.get('subject')} -> {step.get('detail')}")
    print("=" * 50 + "\n")


def _normalize_skip(skip_stages: Optional[List[str]]) -> Optional[List[str]]:
    if not skip_stages:
        return None
    alias = {
        "multiview": "views",
        "graph": "graphs",
        "reasoning": "bgnn",
        "scoring": "view_scores",
        "selection": "selected_views",
    }
    out = []
    for s in skip_stages:
        out.append(alias.get(s, s))
    return out


def _serializable_state(context: Dict[str, Any]) -> Dict[str, Any]:
    skip_keys = {
        "image",
        "depth_map",
        "segmentations",
        "views",
        "graph_embeddings",
        "node_embeddings",
    }
    state = {}
    for k, v in context.items():
        if k in skip_keys:
            continue
        try:
            import json
            json.dumps(v, default=str)
            state[k] = v
        except (TypeError, ValueError):
            state[k] = str(v)
    return state
