"""
Download all default models into models/ folder.
Run automatically by install.bat and install_gpu.bat.

Usage:
  python download_models.py                          # WD14 + YOLO + InsightFace + Anime
  python download_models.py --no-anime               # skip anime detection models
  python download_models.py --upscale                # + RealESRGAN_x4plus_anime_6B (default)
  python download_models.py --upscale-models RealESRGAN_x4plus RealESRGAN_x2plus
  python download_models.py --florence2              # + Florence-2 (large, ~1 GB)
  python download_models.py SmilingWolf/wd-convnext-tagger-v3  # custom WD14 variant
"""
import sys
import os
import argparse

# Force UTF-8 console output so emoji/✓ in progress logs don't crash on
# non-UTF-8 consoles (e.g. Turkish cp1254 on Windows). Python 3.7+.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Pre-load onnxruntime BEFORE model_installer (which imports PyQt5). On Windows,
# PyQt5 loaded first breaks onnxruntime's native DLL → imgutils anime warm-up fails.
try:
    import onnxruntime as _ort  # noqa: F401
except Exception:
    pass

# Ensure project root is on the path
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from src.core.model_paths import WD14_DIR, YOLO_DIR, INSIGHTFACE_DIR, UPSCALE_DIR, ensure_dirs
from src.core.model_installer import (
    _install_wd14, _install_yolo, _install_insightface,
    _install_florence2, _install_upscale_models, _install_anime_models,
    WD14_REPO, YOLO_MODEL, INSIGHTFACE_MODEL,
)


def _log(msg: str):
    print(msg, flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Download LoRA-Harvester models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("wd14_repo", nargs="?", default=None,
                        help="WD14 HuggingFace repo (default: SmilingWolf/wd-swinv2-tagger-v3)")
    parser.add_argument("--upscale", action="store_true",
                        help="Download Real-ESRGAN upscale model(s)")
    parser.add_argument(
        "--upscale-models", nargs="+",
        metavar="MODEL",
        default=None,
        help="Which Real-ESRGAN models to download (default: RealESRGAN_x4plus_anime_6B). "
             "Available: RealESRGAN_x4plus | RealESRGAN_x4plus_anime_6B | "
             "RealESRGAN_x2plus | realesr-general-x4v3 | realesr-animevideov3",
    )
    parser.add_argument("--florence2", action="store_true",
                        help="Download Florence-2 captioning model (~1 GB)")
    parser.add_argument("--no-anime", action="store_true",
                        help="Skip anime detection models (cascade + imgutils warmup)")
    args = parser.parse_args()

    print("=" * 60)
    print("  LoRA-Harvester — Downloading Models")
    print("=" * 60)
    ensure_dirs()

    wd14_repo = args.wd14_repo or "SmilingWolf/wd-swinv2-tagger-v3"
    print(f"WD14 variant: {wd14_repo}")

    steps = [
        ("WD14 Tagger",   lambda log: _install_wd14(log, wd14_repo)),
        ("YOLO Detector", _install_yolo),
        ("InsightFace",   _install_insightface),
    ]

    # Anime detection models (cascade ~1.5MB + imgutils warmup) — default on
    if not args.no_anime:
        steps.append(("Anime Detector", _install_anime_models))

    if args.upscale or args.upscale_models:
        models = args.upscale_models or ["RealESRGAN_x4plus_anime_6B"]
        steps.append(
            ("Real-ESRGAN", lambda log, m=models: _install_upscale_models(log, m))
        )
        print(f"Upscale models: {', '.join(models)}")

    if args.florence2:
        steps.append(("Florence-2", _install_florence2))

    errors = []
    for i, (label, fn) in enumerate(steps, 1):
        print(f"\n[{i}/{len(steps)}] {label}")
        print("-" * 40)
        try:
            fn(_log)
        except Exception as e:
            msg = f"  ERROR: {e}"
            print(msg)
            errors.append(f"{label}: {e}")

    print("\n" + "=" * 60)
    if errors:
        print("  Some downloads failed:")
        for err in errors:
            print(f"    - {err}")
        print("  Run this script again when internet is available.")
        sys.exit(1)
    else:
        print("  All models downloaded successfully!")
        print(f"  Location: {_ROOT}/models/")
    print("=" * 60)


if __name__ == "__main__":
    main()
