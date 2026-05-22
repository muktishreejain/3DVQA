"""Depth backend factory: DepthPro or MiDaS (Windows-compatible)."""

from __future__ import annotations

from typing import Any, Dict

from models.depthpro.model import DepthProModel
from models.midas.model import MiDaSModel
from utils.base_model import BaseModel


def create_depth_model(config: Dict[str, Any], device: str = "cuda") -> BaseModel:
    backend = config.get("depth", {}).get("backend", "depthpro").lower()
    if backend == "midas":
        return MiDaSModel(config, device=device)
    return DepthProModel(config, device=device)
