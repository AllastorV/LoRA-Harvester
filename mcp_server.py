"""
LoRA-Harvester MCP Server
==========================

Exposes LoRA-Harvester to MCP clients (Claude Code / Desktop) as tools.

Two tiers:
  - Light tools: call core modules directly (no GPU, instant) —
    dataset scanning, caption validation, kohya export, model registry.
  - Heavy tools: spawn the existing CLI scripts as background jobs —
    video processing, character sorting, model downloads.

Diagnostics:
  - health_check / get_crash_log / run_self_test / gpu_check let the
    client detect real runtime errors in the program.

Run:
    python mcp_server.py            # stdio transport

IMPORTANT: launch with the Python env that has the project's heavy deps
(cv2, torch, PyQt5) so the background jobs work. Light tools and
diagnostics work even without them.
"""
from __future__ import annotations

import os
import sys
import subprocess
import threading
import importlib
from pathlib import Path
from typing import Optional, List, Dict

# Project root on sys.path
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("lora-harvester")

# ──────────────────────────────────────────────────────────────
# Job manager — tracks background subprocess jobs
# ──────────────────────────────────────────────────────────────

_JOB_DIR = _ROOT / ".mcp_jobs"
_JOB_DIR.mkdir(exist_ok=True)


class _JobManager:
    def __init__(self):
        self._jobs: Dict[str, dict] = {}
        self._counter = 0
        self._lock = threading.Lock()

    def _next_id(self) -> str:
        with self._lock:
            self._counter += 1
            return f"job_{self._counter:04d}"

    def start(self, cmd: List[str], label: str) -> str:
        job_id = self._next_id()
        log_path = _JOB_DIR / f"{job_id}.log"

        # Launch detached, redirect stdout+stderr to the log file.
        log_f = open(log_path, "w", encoding="utf-8", errors="replace")
        log_f.write(f"$ {' '.join(cmd)}\n\n")
        log_f.flush()

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                cwd=str(_ROOT),
                env=os.environ.copy(),
            )
        except Exception as e:
            log_f.write(f"\n[LAUNCH ERROR] {e}\n")
            log_f.close()
            self._jobs[job_id] = {
                "status": "failed", "exit_code": -1, "cmd": cmd,
                "label": label, "log_path": str(log_path),
                "proc": None, "started": _now(), "log_f": None,
            }
            return job_id

        self._jobs[job_id] = {
            "status": "running", "exit_code": None, "cmd": cmd,
            "label": label, "log_path": str(log_path),
            "proc": proc, "started": _now(), "log_f": log_f,
        }
        return job_id

    def _refresh(self, job_id: str):
        job = self._jobs.get(job_id)
        if not job or job["proc"] is None:
            return
        rc = job["proc"].poll()
        if rc is not None and job["status"] == "running":
            job["status"] = "done" if rc == 0 else "failed"
            job["exit_code"] = rc
            if job.get("log_f"):
                try:
                    job["log_f"].close()
                except Exception:
                    pass
                job["log_f"] = None

    def status(self, job_id: str) -> dict:
        self._refresh(job_id)
        job = self._jobs.get(job_id)
        if not job:
            return {"error": f"unknown job_id: {job_id}"}
        return {
            "job_id": job_id,
            "status": job["status"],
            "exit_code": job["exit_code"],
            "elapsed_sec": round(_now() - job["started"], 1),
            "label": job["label"],
            "cmd": " ".join(job["cmd"]),
        }

    def log(self, job_id: str, tail: int = 60) -> str:
        job = self._jobs.get(job_id)
        if not job:
            return f"unknown job_id: {job_id}"
        p = Path(job["log_path"])
        if not p.exists():
            return "(no log yet)"
        try:
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception as e:
            return f"(could not read log: {e})"
        return "\n".join(lines[-tail:])

    def list_all(self) -> List[dict]:
        for jid in list(self._jobs.keys()):
            self._refresh(jid)
        return [self.status(jid) for jid in self._jobs]

    def cancel(self, job_id: str) -> dict:
        job = self._jobs.get(job_id)
        if not job:
            return {"error": f"unknown job_id: {job_id}"}
        if job["proc"] and job["status"] == "running":
            try:
                job["proc"].terminate()
                job["status"] = "cancelled"
                job["exit_code"] = -15
            except Exception as e:
                return {"error": str(e)}
        return self.status(job_id)


def _now() -> float:
    # time.monotonic is fine here (server process, not a workflow script)
    import time
    return time.monotonic()


_jobs = _JobManager()


def _py() -> str:
    """Python interpreter to run heavy scripts (inherits current env)."""
    return sys.executable


# ──────────────────────────────────────────────────────────────
# LIGHT TOOLS (direct, synchronous, no GPU)
# ──────────────────────────────────────────────────────────────

@mcp.tool()
def scan_dataset(folder: str) -> dict:
    """Scan a dataset folder and report image/caption pair counts grouped by concept.

    Args:
        folder: Path to a dataset folder (harvester output or any image folder).
    Returns:
        total image count, per-concept counts, and missing-caption count.
    """
    from src.core.dataset_scanner import scan_dataset as _scan, detect_concepts
    root = Path(folder)
    if not root.exists():
        return {"error": f"folder not found: {folder}"}
    pairs = _scan(root)
    concepts = detect_concepts(root)
    missing = sum(1 for p in pairs if p.caption is None)
    return {
        "folder": str(root),
        "total": len(pairs),
        "with_captions": len(pairs) - missing,
        "missing_captions": missing,
        "concepts": {name: len(ps) for name, ps in sorted(concepts.items())},
    }


@mcp.tool()
def validate_captions(folder: str, trigger_word: str = "", min_tags: int = 3,
                      max_samples: int = 25) -> dict:
    """Audit caption quality: missing/empty captions, missing trigger word,
    low tag count, duplicate tags.

    Args:
        folder: Dataset folder to validate.
        trigger_word: If set, flag captions missing this word.
        min_tags: Minimum comma-separated tags before flagging (0 = disable).
        max_samples: Max number of problem samples to return.
    """
    from src.core.dataset_scanner import scan_dataset as _scan, validate_captions as _val
    root = Path(folder)
    if not root.exists():
        return {"error": f"folder not found: {folder}"}
    pairs = _scan(root)
    issues = _val(pairs, trigger_word=trigger_word, min_tags=min_tags)

    by_type: Dict[str, int] = {}
    for iss in issues:
        for t in iss.issues:
            by_type[t] = by_type.get(t, 0) + 1

    samples = [
        {"image": iss.image.name, "issues": iss.issues,
         "tags": iss.tag_count, "caption": iss.caption_text[:80]}
        for iss in issues[:max_samples]
    ]
    return {
        "folder": str(root),
        "total_images": len(pairs),
        "issue_count": len(issues),
        "by_type": by_type,
        "samples": samples,
    }


@mcp.tool()
def list_concepts(folder: str) -> dict:
    """Return concept→image-count for a dataset folder."""
    from src.core.dataset_scanner import detect_concepts
    root = Path(folder)
    if not root.exists():
        return {"error": f"folder not found: {folder}"}
    concepts = detect_concepts(root)
    return {name: len(ps) for name, ps in sorted(concepts.items())}


@mcp.tool()
def dataset_stats(folder: str) -> dict:
    """Compute dataset statistics: concept distribution, resolution buckets,
    and warnings for under/over-sized concepts. Resolution requires Pillow.
    """
    from collections import Counter
    from src.core.dataset_scanner import scan_dataset as _scan
    root = Path(folder)
    if not root.exists():
        return {"error": f"folder not found: {folder}"}
    pairs = _scan(root)
    concept_counter = Counter(p.concept for p in pairs)

    buckets: Counter = Counter()
    res_available = True
    try:
        from PIL import Image as _PIL
        for p in pairs:
            try:
                with _PIL.open(str(p.image)) as img:
                    short = min(img.size)
                if short < 600:
                    buckets["<600px"] += 1
                elif short < 900:
                    buckets["600-900px"] += 1
                elif short < 1200:
                    buckets["900-1200px"] += 1
                else:
                    buckets[">=1200px"] += 1
            except Exception:
                pass
    except ImportError:
        res_available = False

    warnings = []
    for c, n in concept_counter.items():
        if n < 30:
            warnings.append(f"'{c}': {n} frames — may be too few for LoRA")
        elif n > 500:
            warnings.append(f"'{c}': {n} frames — large concept")

    return {
        "folder": str(root),
        "total": len(pairs),
        "concepts": dict(sorted(concept_counter.items(), key=lambda x: -x[1])),
        "resolution_buckets": dict(buckets) if res_available else "Pillow not installed",
        "warnings": warnings,
    }


@mcp.tool()
def list_upscale_models() -> dict:
    """List available Real-ESRGAN upscale models (built-in + custom) and whether
    their weights are downloaded locally."""
    from src.core.upscale_models import list_models
    models = list_models()
    return {
        name: {
            "scale": cfg.get("scale"),
            "arch": cfg.get("arch"),
            "source": cfg.get("source"),
            "available": cfg.get("available"),
            "description": cfg.get("description", ""),
        }
        for name, cfg in models.items()
    }


@mcp.tool()
def export_kohya(source: str, dest: str, repeats: int = 10,
                 copy: bool = True, gen_toml: bool = True) -> dict:
    """Export a dataset to kohya_ss / sd-scripts <repeats>_<concept> folder
    structure with optional dataset_config.toml.

    Args:
        source: Harvester output folder to export.
        dest: Destination folder for the kohya dataset.
        repeats: Training repeats per image (folder-name prefix).
        copy: Copy (safe) vs move (destructive).
        gen_toml: Generate dataset_config.toml.
    """
    from src.core.kohya_exporter import KohyaExporter
    src = Path(source)
    if not src.exists():
        return {"error": f"source not found: {source}"}
    try:
        result = KohyaExporter().export(
            source_root=src, dest_root=Path(dest),
            repeats=repeats, copy=copy, gen_toml=gen_toml,
        )
        return {"exported": result, "total": sum(result.values()), "dest": str(dest)}
    except Exception as e:
        return {"error": str(e)}


# ──────────────────────────────────────────────────────────────
# HEAVY TOOLS (background jobs)
# ──────────────────────────────────────────────────────────────

@mcp.tool()
def process_video(video_path: str, output: str = "output",
                  aspect_ratio: str = "1:1", interval: int = 30,
                  confidence: float = 0.5, quality: bool = False,
                  caption: bool = False, preset: Optional[str] = None,
                  trigger_word: str = "", upscale: bool = False,
                  upscale_model: str = "RealESRGAN_x4plus_anime_6B",
                  ensemble: bool = False) -> dict:
    """Start video→frames extraction as a background job (GPU). Returns a job_id.
    Track with get_job_status / get_job_log.

    Args:
        video_path: Path to the video file (or glob).
        output: Output directory.
        aspect_ratio: Crop ratio e.g. '1:1', '9:16', '3:4'.
        interval: Process every Nth frame.
        confidence: Detection confidence 0-1.
        quality: Enable quality filtering (blur/lighting/duplicates).
        caption: Enable WD14 auto-captioning.
        preset: Tag preset (anime_character/style_lora/realistic_photo/concept_art).
        trigger_word: LoRA trigger word prepended to captions.
        upscale: Enable Real-ESRGAN upscaling.
        upscale_model: Upscale model name (see list_upscale_models).
        ensemble: Enable ensemble detection.
    """
    if not Path(video_path).exists():
        return {"error": f"video not found: {video_path}"}
    cmd = [_py(), "cli.py", video_path,
           "-o", output, "-f", aspect_ratio,
           "-i", str(interval), "-c", str(confidence)]
    if quality:
        cmd.append("--quality")
    if caption:
        cmd.append("--caption")
    if preset:
        cmd += ["--preset", preset]
    if trigger_word:
        cmd += ["--trigger", trigger_word]
    if upscale:
        cmd += ["--upscale", "--upscale-model", upscale_model]
    if ensemble:
        cmd.append("--ensemble")
    job_id = _jobs.start(cmd, label=f"process_video {Path(video_path).name}")
    return {"job_id": job_id, "message": "Started. Use get_job_log(job_id) to follow."}


@mcp.tool()
def sort_characters(input_folder: str, references: Optional[str] = None,
                    output: Optional[str] = None, copy: bool = False,
                    max_characters: int = 6) -> dict:
    """Start face-based character sorting as a background job (GPU). Returns a job_id.

    Args:
        input_folder: Folder of images to sort.
        references: Optional reference faces folder (references/<name>/*.jpg).
        output: Output folder (default: <input>/_sorted).
        copy: Copy instead of move.
        max_characters: Max distinct character folders (1-6).
    """
    if not Path(input_folder).exists():
        return {"error": f"input not found: {input_folder}"}
    cmd = [_py(), "scripts/character_sort.py", input_folder,
           "--max-characters", str(max_characters)]
    if references:
        cmd += ["--references", references]
    if output:
        cmd += ["--output", output]
    if copy:
        cmd.append("--copy")
    job_id = _jobs.start(cmd, label=f"sort_characters {Path(input_folder).name}")
    return {"job_id": job_id, "message": "Started. Use get_job_log(job_id) to follow."}


@mcp.tool()
def download_models(upscale: bool = False, florence2: bool = False,
                    upscale_models_list: Optional[List[str]] = None) -> dict:
    """Start model download as a background job. Returns a job_id.

    Args:
        upscale: Download Real-ESRGAN upscale model(s).
        florence2: Download Florence-2 captioner (~1 GB).
        upscale_models_list: Specific upscale model names to download.
    """
    cmd = [_py(), "scripts/download_models.py"]
    if upscale_models_list:
        cmd += ["--upscale-models"] + list(upscale_models_list)
    elif upscale:
        cmd.append("--upscale")
    if florence2:
        cmd.append("--florence2")
    job_id = _jobs.start(cmd, label="download_models")
    return {"job_id": job_id, "message": "Started. Use get_job_log(job_id) to follow."}


# ──────────────────────────────────────────────────────────────
# JOB MANAGEMENT
# ──────────────────────────────────────────────────────────────

@mcp.tool()
def get_job_status(job_id: str) -> dict:
    """Get status of a background job (running/done/failed/cancelled)."""
    return _jobs.status(job_id)


@mcp.tool()
def get_job_log(job_id: str, tail: int = 60) -> dict:
    """Get the last N lines of a background job's combined stdout/stderr log.
    Readable while the job is still running (live progress + tracebacks)."""
    return {"job_id": job_id, "log": _jobs.log(job_id, tail=tail)}


@mcp.tool()
def list_jobs() -> list:
    """List all background jobs and their statuses."""
    return _jobs.list_all()


@mcp.tool()
def cancel_job(job_id: str) -> dict:
    """Terminate a running background job."""
    return _jobs.cancel(job_id)


# ──────────────────────────────────────────────────────────────
# DIAGNOSTICS — detect real runtime errors
# ──────────────────────────────────────────────────────────────

@mcp.tool()
def health_check() -> dict:
    """Import every core and UI module to detect broken imports / missing
    dependencies. Reports which modules fail and why — the fastest way to
    surface real runtime breakage.
    """
    core_modules = [
        "detector", "ensemble_detector", "cropper", "text_detector",
        "unified_processor", "enhanced_processor",
        "quality_analyzer", "advanced_captioner", "florence2_captioner",
        "character_recognizer", "anime_detector",
        "anime_character_recognizer", "upscaler", "upscale_models",
        "dataset_scanner",
        "kohya_exporter", "model_installer", "model_paths",
    ]
    ui_modules = [
        "translations", "theme", "main_window", "caption_studio_page",
        "character_sort_page", "tag_frequency_page", "review_grid_page",
        "advanced_settings",
    ]

    results: Dict[str, str] = {}
    for m in core_modules:
        results[f"src.core.{m}"] = _try_import(f"src.core.{m}")
    for m in ui_modules:
        results[f"src.ui.{m}"] = _try_import(f"src.ui.{m}")

    deps = {}
    for d in ("cv2", "torch", "PyQt5", "numpy", "PIL", "onnxruntime",
              "insightface", "ultralytics", "imgutils", "hdbscan"):
        deps[d] = "ok" if _importable(d) else "MISSING"

    # Real-ESRGAN/basicsr need the project's torchvision compatibility shim.
    # Directly importing realesrgan gives a false negative on newer torchvision.
    deps["realesrgan"] = "ok" if _try_import("src.core.upscaler") == "ok" else "MISSING"

    failed = {k: v for k, v in results.items() if v != "ok"}
    missing = [d for d, s in deps.items() if s == "MISSING"]
    return {
        "ok": len(failed) == 0,
        "failed_modules": failed,
        "module_count": len(results),
        "ok_count": sum(1 for v in results.values() if v == "ok"),
        "dependencies": deps,
        "missing_deps": missing,
    }


@mcp.tool()
def get_crash_log(tail: int = 80) -> dict:
    """Read the tail of crash_log.txt — the app appends UI-thread crash
    tracebacks here (on_progress / on_finished / stop_processing handlers)."""
    p = _ROOT / "crash_log.txt"
    if not p.exists():
        return {"exists": False, "message": "no crash_log.txt (no crashes recorded)"}
    try:
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        return {"exists": True, "log": "\n".join(lines[-tail:])}
    except Exception as e:
        return {"exists": True, "error": str(e)}


@mcp.tool()
def run_self_test() -> dict:
    """Run the project's test suite (run_tests.py) and return a summary.
    Fast (~1-2s); tests needing GPU/cv2/PyQt5 skip gracefully."""
    try:
        env = {
            **os.environ,
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
        }
        proc = subprocess.run(
            [_py(), "run_tests.py", "-q"],
            cwd=str(_ROOT), capture_output=True, text=True, timeout=180,
            encoding="utf-8", errors="replace", env=env,
        )
    except Exception as e:
        return {"error": str(e)}
    out = (proc.stdout or "") + (proc.stderr or "")
    tail = "\n".join(out.splitlines()[-30:])
    return {
        "exit_code": proc.returncode,
        "passed": proc.returncode == 0,
        "output_tail": tail,
    }


@mcp.tool()
def gpu_check() -> dict:
    """Run GPU/CUDA diagnostics (scripts/check_gpu.py)."""
    script = _ROOT / "scripts" / "check_gpu.py"
    if not script.exists():
        return {"error": "scripts/check_gpu.py not found"}
    try:
        env = {
            **os.environ,
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
        }
        proc = subprocess.run(
            [_py(), str(script)],
            cwd=str(_ROOT), capture_output=True, text=True, timeout=120,
            encoding="utf-8", errors="replace", env=env,
        )
    except Exception as e:
        return {"error": str(e)}
    return {
        "exit_code": proc.returncode,
        "output": ((proc.stdout or "") + (proc.stderr or ""))[-3000:],
    }


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _importable(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False


def _try_import(name: str) -> str:
    try:
        importlib.import_module(name)
        return "ok"
    except Exception as e:
        return f"{type(e).__name__}: {e}"[:200]


if __name__ == "__main__":
    mcp.run()
