# APC-VLM (Windows)

Downstream reasoning only — **do not** place detection, depth, multiview, or graph code here.

1. Keep fork at `apc_vlm/APC-VLM/`
2. Pipeline writes `outputs/fusion/` then `outputs/apc_vlm/`
3. Enable full inference: set `apc_vlm.run_reasoning: true` in config and wire `run_apc_reasoning()` in `integration.py`
