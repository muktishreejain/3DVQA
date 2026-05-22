"""Input validation for pipeline stages."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def validate_image(image: Any) -> np.ndarray:
    if image is None:
        raise ValueError("Image input is None")
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected numpy image array, got {type(image)}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB image, got shape {image.shape}")
    if image.size == 0:
        raise ValueError("Image array is empty")
    return image


def validate_question(question: str) -> str:
    if not question or not str(question).strip():
        raise ValueError("Question must be a non-empty string")
    return str(question).strip()


def validate_detections(detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not isinstance(detections, list):
        raise TypeError("Detections must be a list")
    for i, det in enumerate(detections):
        if "bbox" not in det:
            raise ValueError(f"Detection {i} missing 'bbox'")
        bbox = det["bbox"]
        if len(bbox) != 4:
            raise ValueError(f"Detection {i} bbox must have 4 values")
    return detections


def validate_masks(segmentations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not segmentations:
        raise ValueError("Segmentation list is empty")
    for seg in segmentations:
        if "mask_path" not in seg and "mask" not in seg:
            raise ValueError("Segmentation entry missing mask_path or mask")
    return segmentations


def validate_config_keys(cfg: Dict[str, Any], required: List[str]) -> None:
    for key in required:
        parts = key.split(".")
        cur = cfg
        for part in parts:
            if part not in cur:
                raise KeyError(f"Missing required config key: {key}")
            cur = cur[part]
