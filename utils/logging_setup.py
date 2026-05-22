"""Structured logging with stage timing and GPU memory tracking."""

from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional

_LOGGER: Optional[logging.Logger] = None


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    global _LOGGER
    numeric = getattr(logging, level.upper(), logging.INFO)
    logger = logging.getLogger("3dvqa")
    logger.handlers.clear()
    logger.setLevel(numeric)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file:
        from pathlib import Path
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    _LOGGER = logger
    return logger


def get_logger() -> logging.Logger:
    if _LOGGER is None:
        return setup_logging()
    return _LOGGER


def log_gpu_memory(stage: str, enabled: bool = True) -> None:
    if not enabled:
        return
    try:
        import torch

        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / (1024**2)
            reserved = torch.cuda.memory_reserved() / (1024**2)
            get_logger().info(
                "GPU memory [%s]: allocated=%.1f MB reserved=%.1f MB",
                stage,
                alloc,
                reserved,
            )
    except Exception as exc:
        get_logger().warning("GPU memory log failed at %s: %s", stage, exc)


@contextmanager
def stage_timer(stage_name: str) -> Generator[Dict[str, Any], None, None]:
    logger = get_logger()
    start = time.perf_counter()
    info: Dict[str, Any] = {"stage": stage_name, "start": start}
    logger.info("Stage START: %s", stage_name)
    try:
        yield info
    except Exception:
        logger.exception("Stage FAILED: %s", stage_name)
        raise
    finally:
        elapsed = time.perf_counter() - start
        info["elapsed_sec"] = elapsed
        logger.info("Stage END: %s (%.3fs)", stage_name, elapsed)
