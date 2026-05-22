#!/usr/bin/env python3
"""3DVQA — Windows-compatible query-guided multi-view 3D-VQA pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pipeline.orchestrator import PipelineOrchestrator
from utils.config import load_config
from utils.logging_setup import setup_logging
from utils.seed import set_seed
from utils.windows import ensure_windows_paths


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="3DVQA (Windows-native research pipeline)")
    p.add_argument("--image_path", type=str, help="Input RGB image (.png/.jpg/.jpeg)")
    p.add_argument("--question", type=str, help="VQA question text")
    p.add_argument("--output_dir", type=str, default="outputs", help="Output root directory")
    p.add_argument("--config", type=str, default=None, help="YAML config override")
    p.add_argument("--batch_file", type=str, default=None, help="JSON list of {image_path, question}")
    p.add_argument("--skip_stages", type=str, default="", help="Comma-separated stages to skip")
    p.add_argument("--debug", action="store_true", help="Enable debug logging")
    p.add_argument("--seed", type=int, default=None, help="Random seed override")
    p.add_argument("--use_mock_models", action="store_true", help="Heuristic backends (no checkpoints)")
    p.add_argument("--no_resume", action="store_true", help="Disable checkpoint resume")
    p.add_argument(
        "--depth_backend",
        type=str,
        choices=["depthpro", "midas"],
        default=None,
        help="Depth model backend override",
    )
    return p.parse_args()


def main() -> int:
    ensure_windows_paths()
    args = parse_args()
    cfg = load_config(args.config)
    if args.debug or cfg.get("debug"):
        cfg["debug"] = True
        log_level = "DEBUG"
    else:
        log_level = cfg.get("logging", {}).get("level", "INFO")
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.use_mock_models:
        cfg["use_mock_models"] = True
    if args.no_resume:
        cfg.setdefault("pipeline", {})["resume"] = False
    if args.depth_backend:
        cfg.setdefault("depth", {})["backend"] = args.depth_backend

    output_dir = Path(args.output_dir)
    if args.image_path or args.batch_file:
        output_dir.mkdir(parents=True, exist_ok=True)
    log_file = str(output_dir / "pipeline.log") if (args.image_path or args.batch_file) else None
    setup_logging(log_level, log_file=log_file)
    set_seed(cfg.get("seed", 42))

    skip = [s.strip() for s in args.skip_stages.split(",") if s.strip()]
    orchestrator = PipelineOrchestrator(cfg)

    if args.batch_file:
        with open(args.batch_file, "r", encoding="utf-8") as f:
            samples = json.load(f)
        orchestrator.run_batch(samples, output_dir, skip_stages=skip or None)
        print(f"Batch complete. Outputs: {output_dir.resolve()}")
        return 0

    if not args.image_path or not args.question:
        print("Error: provide --image_path and --question, or --batch_file", file=sys.stderr)
        return 1

    orchestrator.run_single(args.image_path, args.question, output_dir, skip_stages=skip or None)
    print(f"Pipeline complete. Outputs: {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
