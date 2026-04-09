"""
character_sort.py — Standalone Character Recognition & Sorting Tool
Part of LoRA-Harvester v2.0

Sorts images by character using InsightFace face recognition.

Usage:
  # Auto-cluster only (no references)
  python character_sort.py input_folder/

  # With reference images
  python character_sort.py input_folder/ --references refs/

  # Custom output directory
  python character_sort.py input_folder/ --output sorted_output/

  # Copy instead of move
  python character_sort.py input_folder/ --copy

  # Adjust sensitivity (lower = stricter)
  python character_sort.py input_folder/ --references refs/ --threshold 0.40

  # CPU only
  python character_sort.py input_folder/ --no-gpu

  # Scan sub-folders recursively
  python character_sort.py input_folder/ --recursive
"""

import argparse
import sys
import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)


def main():
    parser = argparse.ArgumentParser(
        description="🎭 LoRA-Harvester — Character Sort Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Sort with auto-clustering
  python character_sort.py output/video_9x16_yolo/persons/

  # Sort with reference images
  python character_sort.py persons/ --references my_characters/

  # Copy files, strict threshold
  python character_sort.py persons/ --references refs/ --copy --threshold 0.35

Reference directory structure:
  my_characters/
    naruto/
      ref1.jpg
      ref2.jpg
    sasuke/
      ref1.jpg

Output structure:
  _sorted/
    naruto/          ← matched via reference
    sasuke/          ← matched via reference
    character_01/    ← auto-clustered unknown group
    character_02/    ← auto-clustered unknown group
    unknown/         ← couldn't cluster (too few similar faces)
    no_face/         ← no face detected
        """,
    )

    # ── Required ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "input",
        help="Input directory containing images to sort",
    )

    # ── Output ────────────────────────────────────────────────────────────────
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output directory (default: <input>/_sorted/)",
    )

    # ── Reference images ──────────────────────────────────────────────────────
    parser.add_argument(
        "-r", "--references",
        default=None,
        metavar="DIR",
        help="Directory with character sub-folders containing reference images",
    )

    # ── Matching ──────────────────────────────────────────────────────────────
    # NOTE: defaults are None so config.yaml values take effect when the
    # flag is not supplied. Actual fallback defaults are applied below.
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Cosine distance threshold for reference matching (default: 0.45). "
             "Lower = stricter (fewer false positives). Range: 0.2-0.7",
    )
    parser.add_argument(
        "--match-margin",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Minimum distance gap between best and runner-up reference "
             "(default: 0.05). Lower = more aggressive matching of similar "
             "characters. Set to 0 to disable the ambiguity check.",
    )

    # ── Clustering ────────────────────────────────────────────────────────────
    parser.add_argument(
        "--cluster-eps",
        type=float,
        default=None,
        metavar="FLOAT",
        help="DBSCAN eps for clustering unknowns (default: 0.6). "
             "Lower = smaller/tighter clusters.",
    )
    parser.add_argument(
        "--cluster-min",
        type=int,
        default=None,
        metavar="INT",
        help="Minimum images to form a cluster (default: 2). "
             "Images below this form 'unknown' group.",
    )
    parser.add_argument(
        "--no-cluster",
        action="store_true",
        help="Disable auto-clustering. Unmatched faces go to 'unknown/' folder.",
    )
    parser.add_argument(
        "--max-characters",
        type=int,
        default=6,
        metavar="INT",
        help="Maximum character folders to create (1-6, default: 6). "
             "The largest groups are kept; smaller ones are merged into other/.",
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--model",
        default="buffalo_l",
        choices=["buffalo_l", "buffalo_m", "buffalo_s", "antelopev2"],
        help="InsightFace model pack (default: buffalo_l). "
             "buffalo_l = most accurate. buffalo_s = fastest.",
    )

    # ── Hardware ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Run on CPU (slower but no GPU required)",
    )

    # ── File handling ─────────────────────────────────────────────────────────
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of moving them",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan sub-directories",
    )

    # ── Misc ──────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--config",
        default=os.path.join(_PROJECT_ROOT, "config.yaml"),
        help="Path to config.yaml (default: config.yaml)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose logging",
    )

    args = parser.parse_args()

    # ── Setup logging ──────────────────────────────────────────────────────────
    import logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s  %(message)s",
    )

    # ── Try loading config overrides ──────────────────────────────────────────
    config_char = {}
    if os.path.exists(args.config):
        try:
            import yaml
            with open(args.config, 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f)
            config_char = (cfg or {}).get('character_recognition', {})
        except Exception:
            pass

    # Config overrides — precedence: CLI flag > config.yaml > hard-coded default.
    # Because defaults for the CLI flags are None, we can unambiguously tell
    # whether the user explicitly passed a value.
    threshold = (
        args.threshold
        if args.threshold is not None
        else config_char.get('similarity_threshold', 0.45)
    )
    match_margin = (
        args.match_margin
        if args.match_margin is not None
        else config_char.get('match_margin', 0.05)
    )
    cluster_eps = (
        args.cluster_eps
        if args.cluster_eps is not None
        else config_char.get('cluster_eps', 0.6)
    )
    cluster_min = (
        args.cluster_min
        if args.cluster_min is not None
        else config_char.get('cluster_min_samples', 2)
    )
    use_gpu = not args.no_gpu
    # CLI --model only overrides config when it differs from the argparse default
    model_name = args.model
    if model_name == "buffalo_l" and config_char.get('model'):
        model_name = config_char['model']

    # ── Print header ──────────────────────────────────────────────────────────
    print("=" * 60)
    print("🎭 CHARACTER SORT — LoRA-Harvester")
    print("=" * 60)
    print(f"📂 Input   : {args.input}")
    print(f"📁 Output  : {args.output or '<input>/_sorted/'}")
    print(f"🎯 Model   : {model_name}")
    print(f"📚 References: {args.references or 'None (auto-cluster only)'}")
    print(f"🔍 Threshold : {threshold}  (margin: {match_margin})")
    print(f"🔢 Max chars : {args.max_characters}")
    print(f"🔗 Cluster eps: {cluster_eps}  min: {cluster_min}")
    print(f"💻 Device  : {'GPU' if use_gpu else 'CPU'}")
    print(f"📋 Action  : {'Copy' if args.copy else 'Move'}")
    print("=" * 60)

    # ── Validate input ────────────────────────────────────────────────────────
    from pathlib import Path
    if not Path(args.input).exists():
        print(f"❌ Input directory not found: {args.input}")
        sys.exit(1)

    # ── Import recognizer ─────────────────────────────────────────────────────
    try:
        from src.core.character_recognizer import CharacterRecognizer
    except ImportError as e:
        print(f"❌ Import error: {e}")
        sys.exit(1)

    # ── Check InsightFace ─────────────────────────────────────────────────────
    try:
        import insightface  # noqa: F401
    except ImportError:
        print("❌ InsightFace not installed.")
        print("   Install with: pip install insightface onnxruntime")
        print("   For GPU:      pip install insightface onnxruntime-gpu")
        sys.exit(1)

    try:
        import sklearn  # noqa: F401
    except ImportError:
        print("❌ scikit-learn not installed.")
        print("   Install with: pip install scikit-learn")
        sys.exit(1)

    # ── Build recognizer ──────────────────────────────────────────────────────
    effective_cluster_eps = 99.0 if args.no_cluster else cluster_eps
    effective_cluster_min = 99999 if args.no_cluster else cluster_min

    def progress(current, total, msg):
        pct = int(current / total * 100)
        bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
        print(f"\r  [{bar}] {pct:3d}% — {msg[:50]:<50}", end='', flush=True)
        if current == total:
            print()

    recognizer = CharacterRecognizer(
        reference_dir=args.references,
        similarity_threshold=threshold,
        match_margin=match_margin,
        cluster_eps=effective_cluster_eps,
        cluster_min_samples=effective_cluster_min,
        use_gpu=use_gpu,
        model_name=model_name,
        progress_callback=progress,
    )

    # ── Load references ───────────────────────────────────────────────────────
    if args.references:
        print("\n📚 Loading reference images...")
        try:
            counts = recognizer.load_references(args.references)
            if not counts:
                print("⚠️  No valid references found — running in auto-cluster mode")
        except FileNotFoundError as e:
            print(f"❌ {e}")
            sys.exit(1)
    else:
        print("\nℹ️  No references provided — auto-cluster mode only")

    # ── Run sort ──────────────────────────────────────────────────────────────
    print("\n🔄 Scanning and sorting images...")
    try:
        stats = recognizer.sort_directory(
            input_dir=args.input,
            output_dir=args.output,
            copy=args.copy,
            recursive=args.recursive,
            max_characters=max(1, min(args.max_characters, 6)),
        )
    except KeyboardInterrupt:
        print("\n\n⏹️  Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    if not stats:
        print("⚠️  No images were processed.")
    else:
        total_sorted = sum(stats.values())
        known = sum(v for k, v in stats.items() if not k.startswith(('character_', 'unknown', 'no_face', 'multi_face')))
        clustered = sum(v for k, v in stats.items() if k.startswith('character_'))
        print(f"\n✅ Done! {total_sorted} images sorted.")
        if known:
            print(f"   Matched to known characters : {known}")
        if clustered:
            print(f"   Auto-clustered              : {clustered}")

    print()


if __name__ == "__main__":
    main()
