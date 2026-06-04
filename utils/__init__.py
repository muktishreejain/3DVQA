# Force a headless matplotlib backend before any pyplot import so the pipeline
# runs on servers / CI with no display (avoids Tk backend errors).
try:  # pragma: no cover - environment dependent
    import matplotlib

    matplotlib.use("Agg")
except Exception:
    pass

from utils.config import load_config
from utils.logging_setup import setup_logging, get_logger
from utils.seed import set_seed

__all__ = ["load_config", "setup_logging", "get_logger", "set_seed"]
