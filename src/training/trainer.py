"""
KohyaTrainer — launches kohya_ss / sd-scripts training as a subprocess.

Detects the kohya installation, validates paths, then streams stdout/stderr
back via a callback so the UI can display a live log.

Usage:
    trainer = KohyaTrainer(kohya_dir="/path/to/kohya_ss")
    if trainer.is_available():
        trainer.start(
            train_toml="output_lora/train_config.toml",
            log_callback=lambda msg: print(msg),
            finished_callback=lambda ok, summary: print(ok, summary),
        )
    trainer.stop()  # request graceful stop
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class KohyaTrainer:
    """
    Wraps kohya_ss / sd-scripts training scripts as managed subprocesses.
    """

    # Possible relative paths inside a kohya_ss installation
    _TRAIN_SCRIPTS = [
        "train_network.py",
        "sd-scripts/train_network.py",
    ]
    _SDXL_TRAIN_SCRIPTS = [
        "sdxl_train_network.py",
        "sd-scripts/sdxl_train_network.py",
    ]
    _ACCELERATE_CANDIDATES = [
        "accelerate",
        "accelerate.exe",            # root-level install (some setups put exe here)
        "venv/Scripts/accelerate",
        "venv/Scripts/accelerate.exe",
        "venv/bin/accelerate",
        ".venv/Scripts/accelerate",
        ".venv/Scripts/accelerate.exe",
        ".venv/bin/accelerate",
    ]

    def __init__(self, kohya_dir: Optional[str | Path] = None) -> None:
        self.kohya_dir: Optional[Path] = Path(kohya_dir) if kohya_dir else None
        self._proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()
        self._current_epoch = 0
        self._current_loss: Optional[float] = None
        self._current_step = 0
        self._total_steps = 0
        self._eta_str = ""

    # ── Detection ────────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """Return True if a usable kohya train_network.py can be found."""
        return self._find_train_script() is not None

    def check_missing_deps(self) -> bool:
        """
        Return True if critical Kohya training deps are missing.
        Checks the kohya venv first; falls back to sys.executable.
        """
        return bool(self.missing_deps())

    def missing_deps(self) -> list[str]:
        """Return missing Python modules for the interpreter used for training."""
        script = self._find_train_script()
        if not script:
            return []

        py = self._find_venv_python() or Path(sys.executable)
        required = ["accelerate", "diffusers", "transformers", "safetensors", "toml", "bitsandbytes"]
        code = (
            "import importlib.util as u; "
            f"mods={required!r}; "
            "print('\\n'.join(m for m in mods if u.find_spec(m) is None))"
        )
        try:
            r = subprocess.run(
                [str(py), "-c", code],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=20,
            )
            if r.returncode == 0:
                return [m.strip() for m in r.stdout.splitlines() if m.strip()]
        except Exception:
            pass

        return ["accelerate", "diffusers"]

    def detection_info(self) -> dict:
        """Return installation status info for the UI."""
        script = self._find_train_script()
        accel  = self._find_accelerate()
        if accel == self._ACCEL_MODULE_SENTINEL:
            accel_path = f"{sys.executable} -m accelerate"
        elif accel and accel.startswith("__venv_module__"):
            py = accel[len("__venv_module__"):]
            accel_path = f"{py} -m accelerate"
        else:
            accel_path = str(accel) if accel else None
        return {
            "script_found":      script is not None,
            "script_path":       str(script) if script else None,
            "accelerate_found":  accel is not None,
            "accelerate_path":   accel_path,
            "kohya_dir":         str(self.kohya_dir) if self.kohya_dir else None,
        }

    def _find_train_script(self, sdxl: bool = False) -> Optional[Path]:
        """Locate the right Kohya training script for SD1/SDXL."""
        search_roots = []
        if self.kohya_dir:
            search_roots.append(self.kohya_dir)
        # Project-local kohya_ss (next to main.py) — highest priority after explicit dir
        _project_root = Path(__file__).resolve().parent.parent.parent
        for name in ("kohya_ss", "sd-scripts", "kohya-ss"):
            candidate = _project_root / name
            if candidate.is_dir():
                search_roots.append(candidate)
        # Common system-level install locations (fallback)
        for name in ("kohya_ss", "sd-scripts", "kohya-ss"):
            for parent in (Path.home(), Path("C:/"), Path("/opt")):
                candidate = parent / name
                if candidate.is_dir():
                    search_roots.append(candidate)
        for root in search_roots:
            for rel in (self._SDXL_TRAIN_SCRIPTS if sdxl else self._TRAIN_SCRIPTS):
                p = root / rel
                if p.exists():
                    return p
        # Last resort: check PATH / current venv
        found = shutil.which("sdxl_train_network" if sdxl else "train_network")
        if found:
            return Path(found)
        return None

    def _is_sdxl_config(self, train_toml: Path) -> bool:
        """Return True when train_config.toml targets SDXL training."""
        try:
            text = train_toml.read_text(encoding="utf-8", errors="replace").lower()
        except Exception:
            return False
        return (
            'model_type = "sdxl"' in text
            or "text_encoder_lr1" in text
            or "networks.lora_sdxl" in text
        )

    def _sanitize_sdxl_config(self, train_toml: Path) -> bool:
        """
        Disable SDXL text encoder output caching when dynamic captions are used.

        Kohya rejects cache_text_encoder_outputs with shuffle_caption,
        caption dropout, tag dropout, or token warmup. LoRA-Harvester defaults
        to shuffle_caption, so keeping cache disabled is the compatible path.
        """
        try:
            text = train_toml.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return False

        changed = False
        if re.search(r"(?mi)^\s*cache_text_encoder_outputs\s*=\s*true\s*$", text):
            text = re.sub(
                r"(?mi)^\s*cache_text_encoder_outputs\s*=\s*true\s*$",
                "cache_text_encoder_outputs = false",
                text,
            )
            changed = True
        if re.search(r"(?mi)^\s*cache_text_encoder_outputs_to_disk\s*=\s*true\s*$", text):
            text = re.sub(
                r"(?mi)^\s*cache_text_encoder_outputs_to_disk\s*=\s*true\s*$",
                "cache_text_encoder_outputs_to_disk = false",
                text,
            )
            changed = True

        if changed:
            train_toml.write_text(text, encoding="utf-8")
        return changed

    def _has_module(self, module_name: str) -> bool:
        """Return True if the training Python can import module_name."""
        py = self._find_venv_python() or Path(sys.executable)
        try:
            r = subprocess.run(
                [str(py), "-c", f"import {module_name}"],
                capture_output=True,
                timeout=10,
            )
            return r.returncode == 0
        except Exception:
            return False

    def _sanitize_attention_config(self, train_toml: Path) -> bool:
        """Use SDPA instead of xformers when xformers is not installed."""
        if self._has_module("xformers.ops"):
            return False
        try:
            text = train_toml.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return False

        changed = False
        if re.search(r"(?mi)^\s*xformers\s*=\s*true\s*$", text):
            text = re.sub(r"(?mi)^\s*xformers\s*=\s*true\s*$", "xformers = false", text)
            changed = True
        if re.search(r"(?mi)^\s*sdpa\s*=", text):
            if not re.search(r"(?mi)^\s*sdpa\s*=\s*true\s*$", text):
                text = re.sub(r"(?mi)^\s*sdpa\s*=.*$", "sdpa = true", text)
                changed = True
        elif re.search(r"(?mi)^\s*xformers\s*=\s*false\s*$", text):
            text = re.sub(
                r"(?mi)^(\s*xformers\s*=\s*false\s*)$",
                "\\1\nsdpa = true",
                text,
                count=1,
            )
            changed = True

        if changed:
            train_toml.write_text(text, encoding="utf-8")
        return changed

    def _bitsandbytes_8bit_works(self) -> bool:
        """Return True if bitsandbytes AdamW8bit can run on CUDA."""
        py = self._find_venv_python() or Path(sys.executable)
        code = (
            "import torch, bitsandbytes as bnb; "
            "assert torch.cuda.is_available(); "
            "p=torch.nn.Parameter(torch.randn(8,device='cuda')); "
            "o=bnb.optim.AdamW8bit([p],lr=1e-4); "
            "loss=(p*p).sum(); loss.backward(); o.step()"
        )
        try:
            r = subprocess.run([str(py), "-c", code], capture_output=True, timeout=30)
            return r.returncode == 0
        except Exception:
            return False

    def _sanitize_optimizer_config(self, train_toml: Path) -> bool:
        """Fallback to torch AdamW if bitsandbytes 8-bit optimizer is unavailable."""
        try:
            text = train_toml.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return False
        if not re.search(r'(?mi)^\s*optimizer_type\s*=\s*"AdamW8bit"\s*$', text):
            return False
        if self._bitsandbytes_8bit_works():
            return False
        text = re.sub(
            r'(?mi)^\s*optimizer_type\s*=\s*"AdamW8bit"\s*$',
            'optimizer_type = "AdamW"',
            text,
        )
        train_toml.write_text(text, encoding="utf-8")
        return True

    # Sentinel: accelerate available as python module (not standalone exe).
    # Value is the venv python path that has it, so _build_accel_cmd can use it.
    _ACCEL_MODULE_SENTINEL = "__module__"

    def _find_venv_python(self) -> Optional[Path]:
        """Return the Python executable inside the kohya venv, if found."""
        script = self._find_train_script()
        if not script:
            return None
        kohya_root = script.parent
        for venv_name in ("venv", ".venv"):
            for py_name in ("Scripts/python.exe", "bin/python", "Scripts/python"):
                p = kohya_root / venv_name / py_name
                if p.exists():
                    return p
        return None

    def _find_accelerate(self) -> Optional[str]:
        """
        Locate accelerate. Search order (kohya venv is highest priority):
          1. Kohya venv exe candidates (explicit dir + project-local kohya_ss)
          2. System PATH
          3. Same directory as sys.executable (app venv)
          4. Kohya venv python -m accelerate  → returns venv_python path tagged
          5. sys.executable -m accelerate     → returns _ACCEL_MODULE_SENTINEL
        """
        _project_root = Path(__file__).resolve().parent.parent.parent
        _kohya_search: list = []
        if self.kohya_dir:
            _kohya_search.append(Path(self.kohya_dir))
        for name in ("kohya_ss", "sd-scripts", "kohya-ss"):
            _c = _project_root / name
            if _c.is_dir() and _c not in _kohya_search:
                _kohya_search.append(_c)

        # 1. Kohya venv — check exe candidates first
        for kdir in _kohya_search:
            for rel in self._ACCELERATE_CANDIDATES:
                p = kdir / rel
                if p.exists():
                    return str(p)

        # 2. System PATH
        found = shutil.which("accelerate")
        if found:
            return found

        # 3. App's own venv Scripts directory
        exe_dir = Path(sys.executable).parent
        for name in ("accelerate", "accelerate.exe"):
            p = exe_dir / name
            if p.exists():
                return str(p)

        # 4. Kohya venv python -m accelerate (module mode, preferred over sys.executable)
        venv_py = self._find_venv_python()
        if venv_py:
            try:
                r = subprocess.run(
                    [str(venv_py), "-c", "import accelerate"],
                    capture_output=True, timeout=10,
                )
                if r.returncode == 0:
                    # Return the venv python path; _build_accel_cmd detects this prefix
                    return f"__venv_module__{venv_py}"
            except Exception:
                pass

        # 5. sys.executable -m accelerate (last resort — likely wrong env)
        try:
            r = subprocess.run(
                [sys.executable, "-c", "import accelerate"],
                capture_output=True, timeout=8,
            )
            if r.returncode == 0:
                return self._ACCEL_MODULE_SENTINEL
        except Exception:
            pass

        return None

    def _ensure_accelerate_config(self, train_toml: Path,
                                   explicit_config: Optional[str]) -> str:
        """
        Return a path to a valid accelerate config.
        If an explicit config is given, use it.
        Otherwise generate a minimal single-GPU config next to train_toml so
        accelerate launch does not fail asking to run `accelerate config` first.
        """
        if explicit_config and Path(explicit_config).exists():
            return explicit_config

        # Detect mixed_precision from train_config.toml if possible
        mixed = "fp16"
        try:
            text = train_toml.read_text(encoding="utf-8")
            for line in text.splitlines():
                if "mixed_precision" in line and "=" in line:
                    val = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if val in ("fp16", "bf16", "no"):
                        mixed = val
                    break
        except Exception:
            pass

        config_path = train_toml.parent / "accelerate_config.yaml"
        config_path.write_text(
            f"compute_environment: LOCAL_MACHINE\n"
            f"debug: false\n"
            f"distributed_type: NO\n"
            f"downcast_bf16: 'no'\n"
            f"enable_cpu_affinity: false\n"
            f"gpu_ids: all\n"
            f"machine_rank: 0\n"
            f"main_training_function: main\n"
            f"mixed_precision: {mixed}\n"
            f"num_machines: 1\n"
            f"num_processes: 1\n"
            f"rdzv_backend: static\n"
            f"same_network: true\n"
            f"tpu_env: []\n"
            f"tpu_use_cluster: false\n"
            f"tpu_use_sudo: false\n"
            f"use_cpu: false\n",
            encoding="utf-8",
        )
        return str(config_path)

    def _build_accel_cmd(self, accel: str, script: Path, train_toml: Path,
                         accelerate_config: Optional[str]) -> list:
        """Build the full command list for launching training via accelerate."""
        cfg = self._ensure_accelerate_config(train_toml, accelerate_config)

        if accel == self._ACCEL_MODULE_SENTINEL:
            # sys.executable has accelerate as module (last resort)
            py = sys.executable
            base = [py, "-m", "accelerate", "launch",
                    "--num_cpu_threads_per_process", "2"]
        elif accel.startswith("__venv_module__"):
            # Kohya venv python has accelerate as module
            py = accel[len("__venv_module__"):]
            base = [py, "-m", "accelerate", "launch",
                    "--num_cpu_threads_per_process", "2"]
        else:
            base = [accel, "launch", "--num_cpu_threads_per_process", "2"]

        base += ["--config_file", cfg]
        base += [str(script), "--config_file", str(train_toml)]
        return base

    # ── Training control ─────────────────────────────────────────────────────

    def start(
        self,
        train_toml: str | Path,
        log_callback: Callable[[str], None],
        finished_callback: Callable[[bool, str], None],
        accelerate_config: Optional[str] = None,
    ) -> bool:
        """
        Launch training in a background thread.

        Parameters
        ----------
        train_toml         : Path to the generated train_config.toml.
        log_callback       : Called with each line of training output.
        finished_callback  : Called with (success: bool, summary: str) when done.
        accelerate_config  : Path to accelerate config file (optional).

        Returns True if successfully started.
        """
        if self._proc and self._proc.poll() is None:
            log_callback("⚠️ Training already running.")
            return False

        train_toml = Path(train_toml)
        if not train_toml.exists():
            log_callback(f"❌ Config not found: {train_toml}")
            return False

        sdxl = self._is_sdxl_config(train_toml)
        if sdxl and self._sanitize_sdxl_config(train_toml):
            log_callback("⚠️ SDXL text encoder cache disabled because caption shuffle/dropout is active.")
        if self._sanitize_attention_config(train_toml):
            log_callback("⚠️ xformers not installed; using PyTorch SDPA attention instead.")
        if self._sanitize_optimizer_config(train_toml):
            log_callback("⚠️ bitsandbytes AdamW8bit unavailable; using AdamW optimizer instead.")

        script = self._find_train_script(sdxl=sdxl)
        if not script:
            script_name = "sdxl_train_network.py" if sdxl else "train_network.py"
            log_callback(f"❌ {script_name} not found. Set Kohya path in Settings.")
            return False

        accel = self._find_accelerate()
        if accel:
            cmd = self._build_accel_cmd(accel, script, train_toml, accelerate_config)
        else:
            missing = self.missing_deps()
            log_callback("❌ Kohya training dependencies are incomplete.")
            if missing:
                log_callback(f"   Missing: {', '.join(missing)}")
            log_callback("   Click 'Fix deps' on the Training page, then start training again.")
            return False

        self._stop_flag.clear()
        self._current_epoch = 0
        self._current_loss  = None
        self._current_step  = 0
        self._total_steps   = 0
        self._eta_str       = ""
        log_callback(f"🚀 Starting: {' '.join(cmd[:4])} …")

        try:
            env = {
                **os.environ,
                "PYTHONUTF8": "1",
                "PYTHONIOENCODING": "utf-8",
            }
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=str(script.parent),
                env=env,
            )
        except Exception as exc:
            log_callback(f"❌ Failed to start: {exc}")
            return False

        self._thread = threading.Thread(
            target=self._stream_output,
            args=(log_callback, finished_callback),
            daemon=True,
        )
        self._thread.start()
        return True

    def stop(self):
        """Request graceful stop (sends SIGTERM to the subprocess)."""
        self._stop_flag.set()
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.terminate()
                logger.info("KohyaTrainer: sent terminate")
            except Exception as exc:
                logger.warning("terminate failed: %s", exc)

    @property
    def is_running(self) -> bool:
        return bool(self._proc and self._proc.poll() is None)

    @property
    def current_loss(self) -> Optional[float]:
        return self._current_loss

    @property
    def current_epoch(self) -> int:
        return self._current_epoch

    @property
    def current_step(self) -> int:
        return self._current_step

    @property
    def total_steps(self) -> int:
        return self._total_steps

    @property
    def eta_str(self) -> str:
        return self._eta_str

    # ── Internal streaming ───────────────────────────────────────────────────

    def _stream_output(
        self,
        log_cb: Callable[[str], None],
        finished_cb: Callable[[bool, str], None],
    ):
        """Read subprocess output line-by-line and call log_cb."""
        import re
        loss_pat  = re.compile(r"loss[=:\s]+([0-9.]+)", re.I)
        epoch_pat = re.compile(r"epoch\s+(\d+)", re.I)
        # tqdm step line: "150/1500 [01:23<12:37,  1.80it/s"  (with unicode bar chars)
        tqdm_pat  = re.compile(r"(\d+)/(\d+)\s+\[[\d:]+<([\d:]+)")
        fatal_msg: Optional[str] = None

        lines_seen = 0
        try:
            for line in self._proc.stdout:
                line = line.rstrip()
                log_cb(line)
                lines_seen += 1

                if "No data found" in line or "no images found" in line:
                    fatal_msg = line.strip() or "No training data found."

                m = loss_pat.search(line)
                if m:
                    try:
                        self._current_loss = float(m.group(1))
                    except ValueError:
                        pass

                m = epoch_pat.search(line)
                if m:
                    try:
                        self._current_epoch = int(m.group(1))
                    except ValueError:
                        pass

                # tqdm progress: only real training ticks include loss / lr fields.
                # Dataset scan and latent-cache bars also look like "213/213 [...]"
                # and must not overwrite the training step estimate in the UI.
                m = tqdm_pat.search(line)
                is_training_tick = (
                    "loss" in line.lower()
                    or "lr:" in line.lower()
                    or "lr=" in line.lower()
                    or "loss=" in line.lower()
                )
                if m and is_training_tick:
                    try:
                        cur, tot = int(m.group(1)), int(m.group(2))
                        if tot > 0:
                            self._current_step = cur
                            self._total_steps  = tot
                            # remaining shown as "00:00" when done — blank it then
                            eta = m.group(3)
                            self._eta_str = "" if eta in ("00:00", "0:00") else eta
                    except (ValueError, IndexError):
                        pass

                if self._stop_flag.is_set():
                    self._proc.terminate()
                    break
        except Exception as exc:
            log_cb(f"[stream error] {exc}")

        ret = self._proc.wait()
        success = (ret == 0) and not self._stop_flag.is_set() and not fatal_msg
        if success and self._total_steps == 0 and self._current_step == 0:
            success = False
            fatal_msg = "Training finished without reporting any training steps."
        if success:
            summary = f"Training complete (exit code {ret}, {lines_seen} log lines)."
        elif fatal_msg:
            summary = f"Training failed: {fatal_msg}"
        else:
            summary = f"Training stopped (exit code {ret})."
        finished_cb(success, summary)
