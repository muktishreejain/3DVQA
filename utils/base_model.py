"""Base model abstraction: load_model, run_inference, save_outputs, visualize."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from utils.logging_setup import get_logger


class BaseModel(ABC):
    """Research-grade model wrapper interface."""

    name: str = "base"

    def __init__(self, config: Dict[str, Any], device: str = "cuda"):
        self.config = config
        self.device = device
        self.model = None
        self._loaded = False
        self._mock_mode = False
        self.logger = get_logger()

    @abstractmethod
    def load_model(self) -> None:
        """Load weights and initialize inference backend."""

    @abstractmethod
    def run_inference(self, *args: Any, **kwargs: Any) -> Any:
        """Run model inference."""

    @abstractmethod
    def save_outputs(self, outputs: Any, output_dir: Path) -> Dict[str, Any]:
        """Persist artifacts and return metadata paths."""

    @abstractmethod
    def visualize(self, outputs: Any, output_dir: Path, **kwargs: Any) -> List[Path]:
        """Generate debug / paper visualizations."""

    def _set_mock_mode(self, reason: str) -> None:
        self._mock_mode = True
        self.logger.warning(
            "[%s] Running in MOCK/heuristic mode: %s. Set use_mock_models=false and "
            "install checkpoints for full inference.",
            self.name,
            reason,
        )

    def ensure_loaded(self) -> None:
        if not self._loaded:
            self.load_model()

    @staticmethod
    def _to_fp16_enabled(config: Dict[str, Any]) -> bool:
        return str(config.get("dtype", "float32")).lower() in ("float16", "fp16", "half")
