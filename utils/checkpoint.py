"""Pipeline checkpoint resume utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.io import load_json, save_json


CHECKPOINT_FILE = "pipeline_checkpoint.json"


def save_checkpoint(output_dir: Path, completed_stages: List[str], state: Dict[str, Any]) -> Path:
    data = {"completed_stages": completed_stages, "state": state}
    return save_json(data, output_dir / CHECKPOINT_FILE)


def load_checkpoint(output_dir: Path) -> Optional[Dict[str, Any]]:
    path = output_dir / CHECKPOINT_FILE
    if not path.exists():
        return None
    return load_json(path)


def is_stage_complete(output_dir: Path, stage: str, completed: List[str]) -> bool:
    if stage in completed:
        return True
    marker = output_dir / stage / "metadata.json"
    return marker.exists()
