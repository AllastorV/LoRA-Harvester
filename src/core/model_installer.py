"""
Model installer — downloads all default models into models/.

Each step is a small function; ModelInstallThread runs them sequentially
and emits progress signals so the UI can show live status.
"""
from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path
from typing import Callable, List, Optional

from PyQt5.QtCore import QThread, pyqtSignal

from src.core.model_paths import (
    WD14_DIR, FLORENCE2_DIR, YOLO_DIR, INSIGHTFACE_DIR, ensure_dirs,
)

logger = logging.getLogger(__name__)

# ── Default model identifiers ────────────────────────────────────────────────

WD14_REPO    = "SmilingWolf/wd-swinv2-tagger-v3"
FLORENCE2_REPO = "microsoft/Florence-2-base"
YOLO_MODEL   = "yolov8n.pt"
INSIGHTFACE_MODEL = "buffalo_l"


# ── Individual install steps ─────────────────────────────────────────────────

def _install_wd14(log: Callable[[str], None], repo_id: str = WD14_REPO) -> None:
    from huggingface_hub import hf_hub_download
    ensure_dirs()
    log(f"Downloading WD14 model ({repo_id})…")
    hf_hub_download(repo_id=repo_id, filename="model.onnx",
                    cache_dir=str(WD14_DIR))
    log("  ✓ model.onnx")
    hf_hub_download(repo_id=repo_id, filename="selected_tags.csv",
                    cache_dir=str(WD14_DIR))
    log("  ✓ selected_tags.csv")
    log(f"✅ WD14 ({repo_id.split('/')[-1]}) ready")


def _install_yolo(log: Callable[[str], None]) -> None:
    ensure_dirs()
    dest = YOLO_DIR / YOLO_MODEL
    if dest.exists():
        log(f"✅ YOLO {YOLO_MODEL} already present")
        return

    log(f"Downloading YOLO {YOLO_MODEL}…")
    # Use ultralytics built-in download if available
    try:
        from ultralytics import YOLO
        import os, tempfile
        # Download to a temp dir, then move to models/yolo/
        with tempfile.TemporaryDirectory() as tmp:
            old_cwd = os.getcwd()
            try:
                os.chdir(tmp)
                mdl = YOLO(YOLO_MODEL)   # triggers auto-download
                # ultralytics downloads to cwd or ~/.ultralytics/assets
                candidates = [
                    Path(tmp) / YOLO_MODEL,
                    Path.home() / ".ultralytics" / "assets" / YOLO_MODEL,
                    Path(YOLO_MODEL),
                ]
                for c in candidates:
                    if c.exists():
                        shutil.copy2(str(c), str(dest))
                        log(f"  ✓ copied to {dest}")
                        break
            finally:
                os.chdir(old_cwd)
    except ImportError:
        # Fallback: direct download from GitHub releases
        import urllib.request
        url = (f"https://github.com/ultralytics/assets/releases/"
               f"download/v8.3.0/{YOLO_MODEL}")
        log(f"  downloading from {url} …")
        urllib.request.urlretrieve(url, str(dest))

    log(f"✅ YOLO {YOLO_MODEL} ready")


def _install_insightface(log: Callable[[str], None]) -> None:
    ensure_dirs()
    log(f"Downloading InsightFace {INSIGHTFACE_MODEL}…")
    try:
        from insightface.app import FaceAnalysis
        import onnxruntime as ort
        providers = (
            ['CUDAExecutionProvider', 'CPUExecutionProvider']
            if 'CUDAExecutionProvider' in ort.get_available_providers()
            else ['CPUExecutionProvider']
        )
        ctx_id = 0 if providers[0] == 'CUDAExecutionProvider' else -1
        app = FaceAnalysis(
            name=INSIGHTFACE_MODEL,
            root=str(INSIGHTFACE_DIR),
            allowed_modules=['detection', 'recognition'],
            providers=providers,
        )
        app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        log(f"✅ InsightFace {INSIGHTFACE_MODEL} ready")
    except ImportError:
        log("⚠️  insightface not installed — skipping (pip install insightface)")


def _install_upscale_models(
    log: Callable[[str], None],
    model_names: Optional[List[str]] = None,
) -> None:
    """
    Download Real-ESRGAN upscale model weights into models/upscale/.

    Args:
        log:          Progress callback.
        model_names:  List of model names from upscale_models.BUILTIN_MODELS.
                      Defaults to the recommended anime model.
    """
    ensure_dirs()
    try:
        import torch
    except ImportError:
        log("  skipping upscale models — PyTorch not installed")
        return

    from src.core.upscale_models import BUILTIN_MODELS
    from src.core.model_paths import upscale_model_path

    if model_names is None:
        model_names = ["RealESRGAN_x4plus_anime_6B"]  # recommended default

    for name in model_names:
        if name not in BUILTIN_MODELS:
            log(f"  unknown model '{name}' — skipping")
            continue

        cfg = BUILTIN_MODELS[name]
        dest = upscale_model_path(cfg['file'])
        if dest.exists():
            log(f"  already present: {cfg['file']}")
            continue

        url = cfg.get('url')
        if not url:
            log(f"  no download URL for '{name}' — place {cfg['file']} in models/upscale/ manually")
            continue

        log(f"  downloading {name} from GitHub releases …")
        try:
            torch.hub.download_url_to_file(url, str(dest), progress=False)
            log(f"  OK {cfg['file']} ({dest.stat().st_size // 1024} KB)")
        except Exception as e:
            log(f"  FAILED {name}: {e}")
            log(f"    Manual: {url}")
            raise

    log("Real-ESRGAN weights ready")


def _install_anime_models(log: Callable[[str], None]) -> None:
    """
    Download anime detection models into models/anime/:
      1. yolov8x6_animeface.pt (~186 MB) — PRIMARY backend (ultralytics YOLOv8,
         GPU, works on any Python/numpy ultralytics supports incl. 3.13).
      2. lbpcascade_animeface.xml (~1.5 MB, always — no extra deps, fallback)
      3. If imgutils is installed, warm-up the deepghs anime models (optional).
    """
    import urllib.request
    from src.core.model_paths import ANIME_DETECTOR_DIR
    ensure_dirs()

    # ── 1. YOLOv8 anime-face weights (best backend) ───────────────────────────
    yolo_path = ANIME_DETECTOR_DIR / "yolov8x6_animeface.pt"
    if yolo_path.exists() and yolo_path.stat().st_size > 1_000_000:
        log(f"  YOLO anime model already present ({yolo_path.stat().st_size // 1024 // 1024} MB)")
    else:
        log("Downloading yolov8x6_animeface.pt (~186 MB, Fuyucchi/yolov8_animeface) ...")
        try:
            from huggingface_hub import hf_hub_download
            import shutil
            src = hf_hub_download("Fuyucchi/yolov8_animeface", "yolov8x6_animeface.pt")
            shutil.copy2(src, yolo_path)
            log(f"  YOLO anime model downloaded ({yolo_path.stat().st_size // 1024 // 1024} MB)")
        except Exception as e:
            # Non-fatal — cascade still works. Log loudly so it's visible.
            log(f"  ⚠️ YOLO anime model download failed: {e}")
            log("  Anime detection will fall back to the Haar cascade (lower quality).")

    cascade_url = (
        "https://raw.githubusercontent.com/nagadomi/"
        "lbpcascade_animeface/master/lbpcascade_animeface.xml"
    )
    cascade_path = ANIME_DETECTOR_DIR / "lbpcascade_animeface.xml"

    if cascade_path.exists() and cascade_path.stat().st_size > 0:
        log("  cascade already present (lbpcascade_animeface.xml)")
    else:
        log("Downloading lbpcascade_animeface.xml ...")
        tmp = cascade_path.with_suffix(".tmp")
        # Only the IO is guarded — a logging/encoding error must NEVER make a
        # successful download look like a failure.
        try:
            urllib.request.urlretrieve(cascade_url, str(tmp))
            tmp.replace(cascade_path)
        except Exception as e:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            raise RuntimeError(f"cascade download failed: {e}")
        log(f"  cascade downloaded ({cascade_path.stat().st_size // 1024} KB)")

    # Warm-up deepghs YOLO models via imgutils (optional, best quality).
    # Note: the correct PyPI package is 'dghs-imgutils' (imported as 'imgutils').
    try:
        import imgutils  # noqa: F401
        have_imgutils = True
    except ImportError:
        have_imgutils = False

    if not have_imgutils:
        log("  imgutils not installed — Haar cascade fallback will be used.")
        log("  For better anime detection: pip install dghs-imgutils")
    else:
        try:
            import numpy as np
            from PIL import Image
            dummy = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
            warmed = []
            try:
                from imgutils.detect import detect_faces
                detect_faces(dummy, conf_threshold=0.5)
                warmed.append("faces")
            except Exception:
                pass
            try:
                from imgutils.detect import detect_person
                detect_person(dummy, conf_threshold=0.5)
                warmed.append("person")
            except Exception:
                pass
            if warmed:
                log(f"  imgutils deepghs models cached ({', '.join(warmed)})")
            else:
                log("  imgutils installed but model warm-up failed "
                    "(may need onnxruntime) — cascade fallback active.")
        except Exception as e:
            log(f"  imgutils warm-up skipped: {e}")

    log("Anime models ready")


def _install_florence2(log: Callable[[str], None]) -> None:
    ensure_dirs()
    log(f"Downloading Florence-2 ({FLORENCE2_REPO}) — may take a few minutes…")
    try:
        from transformers import AutoProcessor, AutoModelForCausalLM
        AutoProcessor.from_pretrained(
            FLORENCE2_REPO, cache_dir=str(FLORENCE2_DIR), trust_remote_code=True)
        log("  ✓ processor")
        AutoModelForCausalLM.from_pretrained(
            FLORENCE2_REPO, cache_dir=str(FLORENCE2_DIR), trust_remote_code=True)
        log("  ✓ model weights")
        log("✅ Florence-2 ready")
    except ImportError:
        log("⚠️  transformers not installed — skipping (pip install transformers)")


# ── Worker thread ────────────────────────────────────────────────────────────

class ModelInstallThread(QThread):
    """Downloads all default models sequentially in a background thread."""

    log_message = pyqtSignal(str)
    progress    = pyqtSignal(int)          # 0–100
    finished_ok = pyqtSignal(bool, str)    # (all_ok, summary)

    # Steps: (label, function, include_by_default)
    STEPS = [
        ("WD14 Tagger",         _install_wd14,           True),
        ("YOLO Detector",       _install_yolo,            True),
        ("InsightFace",         _install_insightface,     True),
        ("Anime Detector",      _install_anime_models,    True),   # cascade ~1.5MB + imgutils warmup
        ("Florence-2",          _install_florence2,       False),  # large; opt-in
        ("Real-ESRGAN (anime)", _install_upscale_models,  False),  # opt-in
    ]

    def __init__(self, include_florence2: bool = False,
                 include_upscale: bool = False,
                 upscale_models: Optional[List[str]] = None,
                 wd14_repo: str = WD14_REPO, parent=None):
        super().__init__(parent)
        self._include_florence2 = include_florence2
        self._include_upscale = include_upscale
        self._upscale_models = upscale_models
        self._wd14_repo = wd14_repo
        self._errors: list[str] = []

    def run(self):
        steps = [
            (label, fn)
            for label, fn, default in self.STEPS
            if default
            or (label == "Florence-2" and self._include_florence2)
            or (label == "Real-ESRGAN (anime)" and self._include_upscale)
        ]
        total = len(steps)
        for i, (label, fn) in enumerate(steps):
            self.log_message.emit(f"\n── {label} ──")
            self.progress.emit(int(i / total * 100))
            try:
                if label == "WD14 Tagger":
                    fn(self.log_message.emit, self._wd14_repo)
                elif label == "Real-ESRGAN (anime)":
                    fn(self.log_message.emit, self._upscale_models)
                else:
                    fn(self.log_message.emit)
            except Exception as exc:
                msg = f"❌ {label} failed: {exc}"
                self.log_message.emit(msg)
                self._errors.append(msg)
                logger.exception("Install step %s failed", label)

        self.progress.emit(100)
        if self._errors:
            self.finished_ok.emit(False, "\n".join(self._errors))
        else:
            self.finished_ok.emit(True, "All models installed successfully ✅")


# ── GPU package installer ─────────────────────────────────────────────────────

class GpuInstallThread(QThread):
    """Runs pip to install GPU packages in the background."""

    log_message = pyqtSignal(str)
    finished_ok = pyqtSignal(bool, str)

    GPU_PACKAGES = [
        "onnxruntime-gpu",
        "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
    ]

    def run(self):
        import subprocess
        all_ok = True
        for pkg in self.GPU_PACKAGES:
            self.log_message.emit(f"pip install {pkg} …")
            cmd = [sys.executable, "-m", "pip", "install"] + pkg.split()
            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=300
                )
                if result.returncode == 0:
                    self.log_message.emit(f"  ✓ {pkg.split()[0]}")
                else:
                    self.log_message.emit(f"  ❌ {pkg.split()[0]}: {result.stderr[-200:]}")
                    all_ok = False
            except Exception as e:
                self.log_message.emit(f"  ❌ {e}")
                all_ok = False

        if all_ok:
            self.finished_ok.emit(True, "GPU packages installed. Restart the app.")
        else:
            self.finished_ok.emit(False, "Some packages failed — check the log.")
