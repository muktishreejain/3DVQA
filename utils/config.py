"""YAML configuration loading and merging."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


def load_config(
    path: Optional[Union[str, Path]] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Load default config and optional user YAML with deep merge."""
    root = Path(__file__).resolve().parents[1]
    default_path = root / "configs" / "default.yaml"
    if not default_path.exists():
        raise FileNotFoundError(f"Default config not found: {default_path}")

    with open(default_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if path is not None:
        user_path = Path(path)
        if not user_path.exists():
            raise FileNotFoundError(f"Config file not found: {user_path}")
        with open(user_path, "r", encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, user_cfg)

    if overrides:
        cfg = _deep_merge(cfg, overrides)
    return cfg


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result
