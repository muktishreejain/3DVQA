# 3DVQA

**Windows-native**, query-guided, occlusion-aware multi-view **3D Visual Question Answering** framework for research (lightweight pseudo-3D lifting — not full reconstruction).

## Target platform

| Component | Version |
|-----------|---------|
| OS | Windows 11 |
| Python | 3.10 |
| CUDA | 11.8 |
| GPU | RTX 4090 (recommended) |
| PyTorch | 2.1.2 |

No Linux-only modules, WSL, or Docker required.

## Allowed models

| Stage | Model |
|-------|--------|
| Detection | Grounding DINO |
| Segmentation | SAM |
| Depth | DepthPro **or** MiDaS |
| Multi-view | Stable Zero123 only |
| Lifting | Omni3D (pseudo-3D) |
| Reasoning | BGNN |
| Downstream | APC-VLM |

**Not used:** Wonder3D, SAM-3D, NeRF, Gaussian splatting, mesh reconstruction.

## Pipeline

```text
Input Image + Question
        ↓
Grounding DINO (+ query parse: target / relation / attribute)
        ↓
SAM Segmentation
        ↓
Depth (DepthPro or MiDaS, query-masked regions)
        ↓
Stable Zero123 Multi-View → outputs/views/
        ↓
Omni3D Semantic Lifting → outputs/lifting/
        ↓
Scene Graph → outputs/graphs/
        ↓
BGNN → outputs/bgnn/
        ↓
View Scoring (S_v = R_v - U_v) → outputs/view_scores/
        ↓
Top-k Selection → outputs/selected_views/
        ↓
Feature Export → outputs/fusion/
        ↓
APC-VLM Integration → outputs/apc_vlm/
```

## Quick start (Windows)

```powershell
cd D:\MUKTI\3DVQA
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Demo (no checkpoints)
python scripts\run_demo.py

# Single run
python main.py `
  --image_path datasets\demo\test_scene.png `
  --question "What is the color of the object behind the red cylinder?" `
  --output_dir outputs\run_001 `
  --use_mock_models
```

## CLI

```text
python main.py --image_path ... --question "..." --output_dir outputs/
```

| Flag | Description |
|------|-------------|
| `--config` | YAML override |
| `--batch_file` | JSON batch |
| `--skip_stages` | Comma-separated (aliases: multiview→views, graph→graphs) |
| `--depth_backend` | `depthpro` or `midas` |
| `--use_mock_models` | Heuristic backends |
| `--no_resume` | Disable checkpoint resume |
| `--debug` / `--seed` | Logging and reproducibility |

## Configuration

`configs/default.yaml` — checkpoints, viewpoints, scoring weights, depth backend, `top_k`.

Checkpoints (optional):

```text
models/checkpoints/grounding_dino/
models/checkpoints/sam/
models/checkpoints/depthpro/   # or midas/
models/checkpoints/zero123/
models/checkpoints/omni3d/
models/checkpoints/bgnn/
```

## APC-VLM

Fork lives under `apc_vlm/APC-VLM/`. Geometry stays in `pipeline/`; APC-VLM only consumes `outputs/fusion/` and writes `outputs/apc_vlm/`.

## Tests

```powershell
pytest tests\ -v
```

## Research focus

- Viewpoint-aware relational reasoning  
- Occlusion handling  
- Interpretable scene graphs  
- Structured export for future Chain-of-Thought via APC-VLM  
