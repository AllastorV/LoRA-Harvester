"""
Central model path registry.

All downloaded models go under  <project_root>/models/
so they are self-contained and portable across machines.
"""
from pathlib import Path

# Project root = 3 levels up from this file (src/core/model_paths.py)
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent

MODELS_DIR: Path   = PROJECT_ROOT / "models"

# Per-model sub-directories
WD14_DIR:        Path = MODELS_DIR / "wd14"
FLORENCE2_DIR:   Path = MODELS_DIR / "florence2"
YOLO_DIR:        Path = MODELS_DIR / "yolo"
INSIGHTFACE_DIR: Path = MODELS_DIR / "insightface"


def ensure_dirs() -> None:
    """Create all model sub-directories if they don't exist yet."""
    for d in (WD14_DIR, FLORENCE2_DIR, YOLO_DIR, INSIGHTFACE_DIR):
        d.mkdir(parents=True, exist_ok=True)


def yolo_model_path(filename: str) -> Path:
    """Return absolute path for a YOLO model file, e.g. 'yolov8n.pt'."""
    ensure_dirs()
    return YOLO_DIR / filename
