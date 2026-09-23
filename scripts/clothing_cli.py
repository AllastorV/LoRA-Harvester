#!/usr/bin/env python3
"""Optional clothing CLI. Analysis is read-only unless --apply is explicit."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.core.clothing_backend import OllamaClothingBackend
from src.core.clothing_captions import undo_caption
from src.core.clothing_io import atomic_json, find_images
from src.core.clothing_profiles import ClothingProfileStore
from src.core.clothing_service import run_clothing_batch, undo_last_job


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library-root', type=Path, default=None)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('check', help='Check the configured local Ollama vision model.')
    commands.add_parser('pull', help='Download the configured model (explicit network operation).')
    commands.add_parser('list', help='List outfit profiles without loading a model.')
    commands.add_parser('undo', help='Undo the last saved clothing application.')
    journal = commands.add_parser('undo-journal', help='Recover a specific applied/prepared backup journal.')
    journal.add_argument('journal', type=Path)
    analyze = commands.add_parser('analyze', help='Analyze final image files; preview-only by default.')
    analyze.add_argument('input', type=Path)
    analyze.add_argument('--recursive', action='store_true')
    analyze.add_argument('--apply', action='store_true', help='Write accepted matches and create undo records.')
    analyze.add_argument('--report', type=Path, help='Write inspectable analysis and before/after captions to JSON.')
    args = parser.parse_args(argv)
    store = ClothingProfileStore(args.library_root)
    try:
        if args.command == 'list':
            result = [{'name': p.name, 'master_tag': p.master_tag, 'enabled': p.enabled,
                       'references': len(p.references)} for p in store.load()]
        elif args.command == 'undo':
            result = undo_last_job(store)
        elif args.command == 'undo-journal':
            result = {'restored': undo_caption(args.journal)}
        elif args.command in ('check', 'pull'):
            backend = OllamaClothingBackend(store.load_settings())
            result = backend.check() if args.command == 'check' else backend.pull(
                lambda v: print(v.get('status', ''), file=sys.stderr))
        else:
            paths = find_images(args.input, args.recursive) if args.input.is_dir() else [args.input]
            records = []
            def collect(result, preview):
                value = result.to_dict()
                if preview is not None:
                    value.update(before=preview.old_text, after=preview.new_text, warnings=preview.warnings)
                records.append(value)
            result = run_clothing_batch(paths, store=store, apply=args.apply,
                                         log=lambda msg: print(msg, file=sys.stderr), on_result=collect)
            if args.report:
                atomic_json(args.report, {'stats': result, 'results': records})
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if isinstance(result, dict) and (result.get('errors') or result.get('cancelled')):
            return 2
        return 0
    except KeyboardInterrupt:
        print('Interrupted. Completed caption writes retain their individual undo journals.', file=sys.stderr)
        return 130
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
