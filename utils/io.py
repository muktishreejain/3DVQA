"""I/O helpers for JSON, images, and tensor exports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
from PIL import Image


def ensure_dir(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data: Any, path: Union[str, Path], indent: int = 2) -> Path:
    p = Path(path)
    ensure_dir(p.parent)

    def _default(obj: Any) -> Any:
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"Not JSON serializable: {type(obj)}")

    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=_default)
    return p


def load_json(path: Union[str, Path]) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_image(array: np.ndarray, path: Union[str, Path]) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    if array.dtype != np.uint8:
        arr = np.clip(array, 0, 255).astype(np.uint8)
    else:
        arr = array
    if arr.ndim == 2:
        Image.fromarray(arr, mode="L").save(p)
    else:
        Image.fromarray(arr).save(p)
    return p


def load_image(path: Union[str, Path]) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")
    suffix = p.suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg"}:
        raise ValueError(f"Unsupported image format: {suffix}. Use .png, .jpg, or .jpeg")
    return np.array(Image.open(p).convert("RGB"))


def stage_output_dir(base: Union[str, Path], stage: str) -> Path:
    return ensure_dir(Path(base) / stage)
