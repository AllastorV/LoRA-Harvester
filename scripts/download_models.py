"""
Download all default models into models/ folder.
Run automatically by install.bat and install_gpu.bat.
"""
import sys
import os

# Ensure project root is on the path
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from src.core.model_paths import WD14_DIR, YOLO_DIR, INSIGHTFACE_DIR, ensure_dirs
from src.core.model_installer import (
    _install_wd14, _install_yolo, _install_insightface,
    WD14_REPO, YOLO_MODEL, INSIGHTFACE_MODEL,
)


def _log(msg: str):
    print(msg, flush=True)


def main():
    print("=" * 60)
    print("  LoRA-Harvester — Downloading Default Models")
    print("=" * 60)
    ensure_dirs()

    wd14_repo = sys.argv[1] if len(sys.argv) > 1 else "SmilingWolf/wd-swinv2-tagger-v3"
    print(f"WD14 variant: {wd14_repo}")

    steps = [
        ("WD14 Tagger",   lambda log: _install_wd14(log, wd14_repo)),
        ("YOLO Detector", _install_yolo),
        ("InsightFace",   _install_insightface),
    ]

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
        print(f"  Location: {_HERE}\\models\\")
    print("=" * 60)


if __name__ == "__main__":
    main()
