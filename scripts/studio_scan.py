#!/usr/bin/env python3
"""Read-only Studio diagnostics without Qt, model downloads or caption writes."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.core.smart_suggestions import scan_suggestions, ScanOptions
from src.core.clothing_profiles import ClothingProfileStore


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--duplicates', action='store_true', help='Hash files to find byte-identical copies')
    parser.add_argument('--minimum-side', type=int, default=512, help='Short-edge warning in pixels; 0 disables it')
    parser.add_argument('--no-recursive', action='store_true')
    parser.add_argument('--report', type=Path, help='Write a NEW JSON report; existing files are never overwritten')
    args = parser.parse_args(argv)
    try:
        profiles = ClothingProfileStore().load()
        report = scan_suggestions(args.dataset,
            options=ScanOptions(not args.no_recursive, args.minimum_side, args.duplicates),
            masters=[p.master_tag for p in profiles if p.enabled])
        if args.report:
            # Explicit opt-in to create an artifact. Source images/captions remain untouched.
            with args.report.open('x', encoding='utf-8') as handle:
                json.dump(report.to_dict(), handle, ensure_ascii=False, indent=2)
                handle.write('\n')
        print(f'Images: {report.total} | Non-empty captions: {report.captioned} | Missing/empty: {report.missing}')
        print(f'Elapsed: {report.elapsed_seconds:.3f}s | Exact duplicates checked: {args.duplicates}')
        for suggestion in report.suggestions:
            print(f'[{suggestion.severity}] {suggestion.title_en}')
        for warning in report.warnings:
            print(f'[warning] {warning}', file=sys.stderr)
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f'[ERROR] {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
