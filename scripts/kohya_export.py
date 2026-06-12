"""
kohya_export.py — Kohya / sd-scripts Dataset Export Tool
Part of LoRA-Harvester v3.x

Converts a LoRA-Harvester output folder into the kohya_ss / sd-scripts
<N>_<concept> repeats-folder structure and optionally generates
dataset_config.toml.

Usage:
  # Basic export (copy, 10 repeats)
  python kohya_export.py output/video_1x1_yolo/ -o kohya_dataset/

  # Move files, 15 repeats, no toml
  python kohya_export.py output/video_1x1_yolo/ -o kohya_dataset/ --move --repeats 15 --no-toml

  # Include regularisation images
  python kohya_export.py output/video_1x1_yolo/ -o kohya_dataset/ --reg-dir reg_images/

  # Per-concept repeats override
  python kohya_export.py output/ -o kohya_dataset/ --concept-repeats persons=5 character_01=15

  # List discovered concepts without exporting
  python kohya_export.py output/ --list-concepts

  # Export and open result folder
  python kohya_export.py output/ -o kohya_dataset/ --open
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _project_root() -> Path:
    """scripts/ is one level below project root."""
    return Path(__file__).resolve().parent.parent


def _ensure_project_on_path():
    root = str(_project_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a LoRA-Harvester dataset to kohya_ss / sd-scripts structure.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("source", help="Source dataset folder (LoRA-Harvester output)")
    parser.add_argument("-o", "--output", required=True,
                        help="Destination folder for the kohya dataset")
    parser.add_argument("--repeats", type=int, default=10,
                        help="Training repeat count (default: 10)")
    parser.add_argument("--move", action="store_true",
                        help="Move files instead of copying (faster but destructive)")
    parser.add_argument("--no-toml", dest="gen_toml", action="store_false",
                        help="Skip generating dataset_config.toml")
    parser.add_argument("--resolution", type=int, default=1024,
                        help="Resolution written to dataset_config.toml (default: 1024)")
    parser.add_argument("--reg-dir",
                        help="Optional regularisation / class image source folder")
    parser.add_argument("--reg-repeats", type=int, default=1,
                        help="Repeat count for regularisation images (default: 1)")
    parser.add_argument(
        "--concept-repeats", nargs="*", metavar="CONCEPT=N",
        help="Per-concept repeat overrides, e.g.: persons=5 character_01=15"
    )
    parser.add_argument("--list-concepts", action="store_true",
                        help="Only list discovered concepts and exit — no export performed")
    parser.add_argument("--open", action="store_true",
                        help="Open the output folder in Explorer/Finder after export")
    return parser.parse_args()


def main():
    _ensure_project_on_path()
    args = parse_args()

    from src.core.dataset_scanner import detect_concepts
    from src.core.kohya_exporter import KohyaExporter

    source = Path(args.source)
    if not source.exists():
        print(f"[ERROR] Source folder not found: {source}", file=sys.stderr)
        sys.exit(1)

    # --list-concepts
    if args.list_concepts:
        concepts = detect_concepts(source)
        if not concepts:
            print("No images found.")
            return
        print(f"\nDiscovered concepts in: {source}")
        print("-" * 50)
        for name, pairs in sorted(concepts.items()):
            print(f"  {name:30s}  {len(pairs):5d} images")
        print(f"\nTotal concepts: {len(concepts)}")
        return

    dest = Path(args.output)
    dest.mkdir(parents=True, exist_ok=True)

    # Parse per-concept repeats
    concept_overrides = {}
    if args.concept_repeats:
        for token in args.concept_repeats:
            if '=' not in token:
                print(f"[WARN] Ignoring malformed concept-repeats entry: '{token}'", file=sys.stderr)
                continue
            concept, n = token.split('=', 1)
            try:
                concept_overrides[concept.strip()] = {'repeats': int(n)}
            except ValueError:
                print(f"[WARN] Non-integer repeat count ignored: '{token}'", file=sys.stderr)

    exporter = KohyaExporter()
    print(f"\n{'='*60}")
    print(f"  LoRA-Harvester — Kohya Export")
    print(f"{'='*60}")
    print(f"  Source  : {source}")
    print(f"  Dest    : {dest}")
    print(f"  Repeats : {args.repeats}")
    print(f"  Mode    : {'MOVE' if args.move else 'COPY'}")
    print(f"  TOML    : {'yes' if args.gen_toml else 'no'}")
    if concept_overrides:
        print(f"  Overrides: {concept_overrides}")
    print(f"{'='*60}\n")

    try:
        result = exporter.export(
            source_root=source,
            dest_root=dest,
            repeats=args.repeats,
            copy=not args.move,
            concept_overrides=concept_overrides if concept_overrides else None,
            gen_toml=args.gen_toml,
            resolution=args.resolution,
            reg_dir=Path(args.reg_dir) if args.reg_dir else None,
            reg_repeats=args.reg_repeats,
        )
    except Exception as e:
        print(f"\n[ERROR] Export failed: {e}", file=sys.stderr)
        sys.exit(1)

    total = sum(result.values())
    print("\n✅ Export complete!")
    print(f"   {'Concept':<30s}  {'Images':>6s}")
    print(f"   {'-'*30}  {'------':>6s}")
    for concept, count in sorted(result.items()):
        print(f"   {concept:<30s}  {count:>6d}")
    print(f"   {'TOTAL':<30s}  {total:>6d}")
    print(f"\n   Output: {dest.resolve()}")

    if args.gen_toml:
        toml = dest / "dataset_config.toml"
        if toml.exists():
            print(f"   TOML  : {toml}")

    if args.open:
        try:
            if sys.platform == 'win32':
                os.startfile(str(dest))
            elif sys.platform == 'darwin':
                subprocess.run(['open', str(dest)])
            else:
                subprocess.run(['xdg-open', str(dest)])
        except Exception as e:
            print(f"[WARN] Could not open folder: {e}")


if __name__ == '__main__':
    main()
