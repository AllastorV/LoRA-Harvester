"""Inspect/plan a caption-balanced dataset; source images/captions are never edited."""
from dataclasses import asdict
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.core.dataset_balance import scan_balance, make_plan, export_plan
from src.core.clothing_profiles import ClothingProfileStore
from src.core.clothing_io import atomic_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset',type=Path)
    parser.add_argument('--library-root',type=Path)
    parser.add_argument('--no-recursive',action='store_true')
    parser.add_argument('--dimension',choices=['outfit','pose','angle','joint'],default='joint')
    parser.add_argument('--per-group',type=int,default=0,help='0 = smallest observed eligible group')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--include-unknown',action='store_true')
    parser.add_argument('--keep-duplicates',action='store_true')
    parser.add_argument('--report',type=Path,help='Write a new plan JSON; refuses existing files')
    parser.add_argument('--export',type=Path,help='Explicit copy/export request to a NEW folder outside source')
    args=parser.parse_args()
    try:
        if args.report and args.report.exists():raise ValueError('Report already exists; choose a new filename.')
        store=ClothingProfileStore(args.library_root) if args.library_root else ClothingProfileStore()
        report=scan_balance(args.dataset,recursive=not args.no_recursive,store=store)
        plan=make_plan(report,dimension=args.dimension,per_group=args.per_group,seed=args.seed,
                       include_unknown=args.include_unknown,deduplicate=not args.keep_duplicates)
        print(json.dumps({'summary':report['summary'],'selected':plan['selected_count'],
                          'buckets':plan['buckets'],'skipped':plan['skipped']},ensure_ascii=False,indent=2))
        if args.report:atomic_json(args.report,plan)
        if args.export:print(json.dumps(export_plan(plan,args.export),ensure_ascii=False,indent=2))
        return 0
    except (ValueError,OSError,RuntimeError) as exc:
        print('ERROR: '+str(exc),file=sys.stderr);return 1

if __name__=='__main__':raise SystemExit(main())
