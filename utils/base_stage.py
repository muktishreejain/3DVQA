"""Base pipeline stage abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

from utils.checkpoint import is_stage_complete
from utils.io import save_json
from utils.logging_setup import get_logger, log_gpu_memory, stage_timer
from utils.validation import validate_image

_SKIP_ALIASES = {
    "multiview": "views",
    "graph": "graphs",
    "reasoning": "bgnn",
    "scoring": "view_scores",
    "selection": "selected_views",
}


class BaseStage(ABC):
    stage_name: str = "base"

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger()

    @abstractmethod
    def run(self, context: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
        """Execute stage and return updated context."""

    def should_skip(self, output_dir: Path, skip_stages: Optional[list], completed: list) -> bool:
        if skip_stages and self.stage_name in skip_stages:
            self.logger.info("Skipping stage (user flag): %s", self.stage_name)
            return True
        if self.config.get("pipeline", {}).get("resume", True):
            if is_stage_complete(output_dir, self.stage_name, completed):
                self.logger.info("Skipping stage (checkpoint): %s", self.stage_name)
                return True
        return False

    def execute(
        self,
        context: Dict[str, Any],
        output_dir: Path,
        skip_stages=None,
        completed=None,
    ) -> Dict[str, Any]:
        completed = completed or []
        skip = skip_stages
        if skip_stages:
            skip = list(skip_stages)
            for alias, canonical in _SKIP_ALIASES.items():
                if alias in skip and canonical not in skip:
                    skip.append(canonical)
        if self.should_skip(output_dir, skip, completed):
            return context
        validate_image(context.get("image"))
        stage_dir = output_dir / self.stage_name
        stage_dir.mkdir(parents=True, exist_ok=True)
        with stage_timer(self.stage_name):
            log_gpu_memory(
                self.stage_name,
                self.config.get("logging", {}).get("log_gpu_memory", True),
            )
            result = self.run(context, stage_dir)
            save_json(
                {"stage": self.stage_name, "status": "ok"},
                stage_dir / "metadata.json",
            )
        context.update(result)
        return context
