"""Windows compatibility helpers."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def ensure_windows_paths() -> None:
    """Normalize path handling and optional CUDA DLL paths on Windows."""
    if sys.platform != "win32":
        return
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    # Optional: user can set TORCH_LIB in env for CUDA 11.8 on Windows
    torch_lib = os.environ.get("TORCH_LIB")
    if torch_lib:
        os.add_dll_directory(torch_lib)


def resolve_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()
