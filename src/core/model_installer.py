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
from typing import Callable

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
        ("WD14 Tagger",    _install_wd14,        True),
        ("YOLO Detector",  _install_yolo,         True),
        ("InsightFace",    _install_insightface,  True),
        ("Florence-2",     _install_florence2,    False),  # large; opt-in
    ]

    def __init__(self, include_florence2: bool = False,
                 wd14_repo: str = WD14_REPO, parent=None):
        super().__init__(parent)
        self._include_florence2 = include_florence2
        self._wd14_repo = wd14_repo
        self._errors: list[str] = []

    def run(self):
        steps = [
            (label, fn)
            for label, fn, default in self.STEPS
            if default or (label == "Florence-2" and self._include_florence2)
        ]
        total = len(steps)
        for i, (label, fn) in enumerate(steps):
            self.log_message.emit(f"\n── {label} ──")
            self.progress.emit(int(i / total * 100))
            try:
                if label == "WD14 Tagger":
                    fn(self.log_message.emit, self._wd14_repo)
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
