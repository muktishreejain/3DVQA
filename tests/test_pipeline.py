"""Smoke tests for 3DVQA pipeline (Windows-compatible stack)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.orchestrator import PipelineOrchestrator
from utils.config import load_config
from utils.validation import validate_image, validate_question


@pytest.fixture
def config():
    cfg = load_config()
    cfg["use_mock_models"] = True
    cfg.setdefault("pipeline", {})["resume"] = False
    cfg.setdefault("apc_vlm", {})["enabled"] = True
    return cfg


@pytest.fixture
def sample_image(tmp_path):
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    img[40:90, 40:90] = [200, 50, 50]
    p = tmp_path / "test.png"
    from PIL import Image
    Image.fromarray(img).save(p)
    return p


def test_validation():
    validate_question("What color?")
    with pytest.raises(ValueError):
        validate_question("  ")
    validate_image(np.zeros((64, 64, 3), dtype=np.uint8))


def test_full_pipeline(config, sample_image, tmp_path):
    out = tmp_path / "out"
    orch = PipelineOrchestrator(config)
    ctx = orch.run_single(
        str(sample_image),
        "What is the color of the object behind the red cylinder?",
        out,
    )
    assert (out / "pipeline_summary.json").exists()
    assert (out / "detection" / "query.json").exists()
    assert (out / "views" / "views.json").exists()
    assert (out / "graphs" / "scene_graph.json").exists()
    assert (out / "bgnn" / "bgnn.json").exists()
    assert (out / "fusion" / "fusion_export.json").exists()
    assert (out / "apc_vlm" / "apc_vlm_integration.json").exists()
    assert ctx["query"]["attribute"] == "color"


def test_stage_skip(config, sample_image, tmp_path):
    out = tmp_path / "out_skip"
    orch = PipelineOrchestrator(config)
    orch.run_single(
        str(sample_image),
        "test question",
        out,
        skip_stages=["views", "lifting", "graphs", "bgnn", "view_scores", "selected_views", "fusion", "apc_vlm"],
    )
    assert (out / "depth" / "metadata.json").exists()


def test_graph_relations():
    from pipeline.graph.stage import build_scene_graph

    objects = [
        {"object": "a", "position": [0.0, 0.0, 0.6]},
        {"object": "b", "position": [0.2, 0.0, 0.4]},
    ]
    g = build_scene_graph(
        objects,
        ["behind", "near", "left_of"],
        {"depth_threshold_near": 0.3, "depth_threshold_behind": 0.05},
    )
    assert len(g["nodes"]) == 2
    assert any(e["relation"] == "behind" for e in g["edges"])


def test_midas_backend(sample_image, tmp_path):
    cfg = load_config()
    cfg["use_mock_models"] = True
    cfg.setdefault("pipeline", {})["resume"] = False
    cfg.setdefault("depth", {})["backend"] = "midas"
    cfg["pipeline"]["stages"] = ["detection", "segmentation", "depth"]
    out = tmp_path / "midas_out"
    PipelineOrchestrator(cfg).run_single(str(sample_image), "object near table", out)
    assert (out / "depth" / "depth.json").exists()
