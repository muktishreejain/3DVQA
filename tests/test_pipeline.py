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
    assert (out / "graph_cot" / "graph_cot.json").exists()
    assert ctx.get("reasoning_path"), "Graph-CoT must produce a reasoning path"
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


# ── Novel contribution tests ──────────────────────────────────────────


def test_bgnn_per_node_uncertainty():
    """Contribution 1: BGNN emits per-node p_exists/epistemic/aleatoric, and a
    less-visible (occluded) node gets lower existence probability."""
    from models.bgnn.model import BGNNModel

    cfg = load_config()
    cfg["use_mock_models"] = True
    graph = {
        "nodes": [
            {"id": "n0", "label": "red cylinder", "position": [0.0, 0.0, 0.5], "visibility": 1.0},
            {"id": "n1", "label": "green cube", "position": [0.05, 0.0, 0.7], "visibility": 0.3},
        ],
        "edges": [{"source": "n0", "target": "n1", "relation": "behind"}],
    }
    emb = np.zeros((2, 128), dtype=np.float32)
    out = BGNNModel(cfg, device="cpu").run_inference(graph, emb)

    nu = {u["id"]: u for u in out["node_uncertainty"]}
    assert set(nu) == {"n0", "n1"}
    for u in nu.values():
        for k in ("p_exists", "epistemic", "aleatoric"):
            assert 0.0 <= u[k] <= 1.0
    # occluded node (lower visibility) has lower existence probability
    assert nu["n1"]["p_exists"] < nu["n0"]["p_exists"]


def test_analytic_reprojection(tmp_path):
    """Contribution 2: reprojection produces virtual cameras + per-object
    visibility in [0,1] and never synthesizes RGB images."""
    from pipeline.multiview.stage import MultiviewStage

    cfg = load_config()
    cfg.setdefault("multiview", {})["n_virtual_cams"] = 8
    ctx = {
        "image": np.zeros((64, 64, 3), dtype=np.uint8),
        "lifted_objects": [
            {"object": "red cylinder", "position": [0.0, 0.0, 0.5], "size_3d": [0.1, 0.2, 0.1]},
            {"object": "green cube", "position": [0.05, 0.0, 0.7], "size_3d": [0.1, 0.1, 0.1]},
        ],
    }
    out = MultiviewStage(cfg).run(ctx, tmp_path)
    views = out["views"]
    assert len(views) == 8
    for v in views:
        assert "image" not in v  # analytic only — no synthesized RGB
        assert "cam" in v and "per_obj_vis" in v
        for vis in v["per_obj_vis"].values():
            assert 0.0 <= vis <= 1.0


def test_graph_cot_reasoning_path(tmp_path):
    """Contribution 3: query-guided traversal yields a FIND/TRAVERSE/READ path
    and an uncertainty-discounted confidence in [0,1]."""
    from pipeline.graph_cot.stage import GraphCoTStage

    cfg = load_config()
    ctx = {
        "image": np.zeros((64, 64, 3), dtype=np.uint8),
        "query": {"target": "red cylinder", "relation": "behind", "attribute": "color"},
        "scene_graph": {
            "nodes": [
                {"id": "n0", "label": "red cylinder", "position": [0.0, 0.0, 0.5],
                 "p_exists": 0.98, "epistemic": 0.03, "aleatoric": 0.04},
                {"id": "n1", "label": "green cube", "position": [0.05, 0.0, 0.7],
                 "p_exists": 0.82, "epistemic": 0.15, "aleatoric": 0.11},
            ],
            "edges": [{"source": "n0", "target": "n1", "relation": "behind"}],
        },
        "selection": {"selected_views": [
            {"view_id": "view_01", "per_obj_vis": {"n0": 0.4, "n1": 0.9}},
        ]},
    }
    out = GraphCoTStage(cfg).run(ctx, tmp_path)
    steps = [s["step"] for s in out["reasoning_path"]]
    assert "FIND" in steps and "TRAVERSE" in steps and "READ" in steps
    assert out["target_object"]["label"] == "green cube"
    assert out["evidence_view_id"] == "view_01"
    assert 0.0 <= out["graph_cot_confidence"] <= 1.0
    # read the color attribute off the target
    read = next(s for s in out["reasoning_path"] if s["step"] == "READ")
    assert read["detail"] == "green"


def test_midas_backend(sample_image, tmp_path):
    cfg = load_config()
    cfg["use_mock_models"] = True
    cfg.setdefault("pipeline", {})["resume"] = False
    cfg.setdefault("depth", {})["backend"] = "midas"
    cfg["pipeline"]["stages"] = ["detection", "segmentation", "depth"]
    out = tmp_path / "midas_out"
    PipelineOrchestrator(cfg).run_single(str(sample_image), "object near table", out)
    assert (out / "depth" / "depth.json").exists()
