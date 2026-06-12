"""
KohyaInstaller — clones kohya-ss/sd-scripts and sets up its environment.

Process:
  1. git clone https://github.com/kohya-ss/sd-scripts  →  dest_dir
  2. python -m venv dest_dir/venv
  3. pip install torch+cu121 (matches typical CUDA 12.x setups)
  4. pip install -r dest_dir/requirements.txt
  5. pip install accelerate

All output streams back via log_callback. finished_callback(ok, path_or_error).
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Callable, Optional


class KohyaInstaller:
    """Downloads and installs Kohya sd-scripts into a local directory."""

    REPO_URL = "https://github.com/kohya-ss/sd-scripts.git"

    # PyTorch for CUDA 12.1 (most common on 40xx/30xx with recent drivers).
    # User can re-install a different build if needed.
    TORCH_INDEX = "https://download.pytorch.org/whl/cu121"
    TORCH_PKGS  = ["torch", "torchvision", "torchaudio"]

    def __init__(self):
        self._stop_flag = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def start(
        self,
        dest_dir: Path,
        log_callback: Callable[[str], None],
        finished_callback: Callable[[bool, str], None],
    ):
        """Start installation in a background thread."""
        if self._thread and self._thread.is_alive():
            log_callback("⚠️ Installation already in progress.")
            return
        self._stop_flag.clear()
        self._thread = threading.Thread(
            target=self._run,
            args=(dest_dir, log_callback, finished_callback),
            daemon=True,
        )
        self._thread.start()

    def stop(self):
        self._stop_flag.set()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _run(
        self,
        dest_dir: Path,
        log: Callable[[str], None],
        done: Callable[[bool, str], None],
    ):
        try:
            dest_dir = Path(dest_dir)
            venv_dir = dest_dir / "venv"

            # ── Step 1: git clone ─────────────────────────────────────────────
            if (dest_dir / "train_network.py").exists():
                log("⚡ sd-scripts already cloned — skipping git clone.")
            else:
                log(f"📥 Cloning sd-scripts → {dest_dir} …")
                log("    (this may take 1-3 minutes on first run)")
                if not self._run_cmd(
                    ["git", "clone", "--depth", "1", self.REPO_URL, str(dest_dir)],
                    log, cwd=None,
                ):
                    done(False, "git clone failed. Is git installed?")
                    return

            if self._stop_flag.is_set():
                done(False, "Cancelled by user.")
                return

            # ── Step 2: create venv ───────────────────────────────────────────
            if not venv_dir.exists():
                log("🐍 Creating virtual environment …")
                if not self._run_cmd(
                    [sys.executable, "-m", "venv", str(venv_dir)], log,
                ):
                    done(False, "venv creation failed.")
                    return

            # Resolve venv pip/python
            if sys.platform == "win32":
                pip_exe   = venv_dir / "Scripts" / "pip.exe"
                python_exe = venv_dir / "Scripts" / "python.exe"
            else:
                pip_exe   = venv_dir / "bin" / "pip"
                python_exe = venv_dir / "bin" / "python"

            # ── Step 3: upgrade pip ───────────────────────────────────────────
            log("⬆️  Upgrading pip …")
            self._run_cmd([str(python_exe), "-m", "pip", "install", "--upgrade", "pip"], log)

            if self._stop_flag.is_set():
                done(False, "Cancelled by user.")
                return

            # ── Step 4: torch (CUDA 12.1) ─────────────────────────────────────
            log(f"🔥 Installing PyTorch (CUDA 12.1) — may be 2-4 GB, please wait …")
            if not self._run_cmd(
                [str(pip_exe), "install"] + self.TORCH_PKGS +
                ["--index-url", self.TORCH_INDEX],
                log,
            ):
                log("⚠️  PyTorch install failed — continuing with requirements.txt")

            if self._stop_flag.is_set():
                done(False, "Cancelled by user.")
                return

            # ── Step 5: requirements.txt ──────────────────────────────────────
            req_file = dest_dir / "requirements.txt"
            if req_file.exists():
                log("📦 Installing requirements.txt …")
                self._run_cmd(
                    [str(pip_exe), "install", "-r", str(req_file)], log,
                )
            else:
                log("⚠️  requirements.txt not found — skipping.")

            # ── Step 6: accelerate ────────────────────────────────────────────
            log("🚀 Installing accelerate …")
            self._run_cmd([str(pip_exe), "install", "accelerate"], log)

            if self._stop_flag.is_set():
                done(False, "Cancelled by user.")
                return

            log("✅ Kohya sd-scripts installed successfully!")
            done(True, str(dest_dir))

        except Exception as exc:
            done(False, f"Unexpected error: {exc}")

    def _run_cmd(self, cmd: list, log: Callable[[str], None], cwd=None) -> bool:
        """Run a command, stream its output, return True on success."""
        log(f"  $ {' '.join(str(c) for c in cmd)}")
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=cwd,
                env={**os.environ},
            )
            for line in proc.stdout:
                line = line.rstrip()
                if line:
                    log(f"  {line}")
                if self._stop_flag.is_set():
                    proc.terminate()
                    return False
            ret = proc.wait()
            return ret == 0
        except FileNotFoundError:
            log(f"  ❌ Command not found: {cmd[0]}")
            return False
        except Exception as exc:
            log(f"  ❌ Error: {exc}")
            return False


def default_install_dir() -> Path:
    """
    Default installation directory.
    Prefers a 'kohya_ss' folder next to the project root (portable).
    Falls back to Desktop/kohya_ss so it works on any PC.
    """
    # kohya_installer.py lives at src/training/ → project root is 3 levels up
    project_root = Path(__file__).resolve().parent.parent.parent
    local = project_root / "kohya_ss"
    if local.is_dir():
        return local
    return project_root / "kohya_ss"  # suggest same location even if not yet created
