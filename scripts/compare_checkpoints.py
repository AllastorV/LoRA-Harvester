"""Local WebUI comparisons: --catalog reads; a JSON recipe requires explicit --run."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.core.checkpoint_compare import ComparisonConfig, WebUIClient, run_comparison, DEFAULT_OUTPUT


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--endpoint',default='http://127.0.0.1:7860',help='Used for --catalog only; recipe carries its own endpoint')
    parser.add_argument('--catalog',action='store_true')
    parser.add_argument('--recipe',type=Path)
    parser.add_argument('--run',action='store_true',help='Authorize local image generation and writing a new results folder')
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    args=parser.parse_args()
    try:
        if args.catalog:
            print(json.dumps(WebUIClient(args.endpoint).catalog(),ensure_ascii=False,indent=2));return 0
        if not args.recipe:parser.error('Choose --catalog or --recipe FILE [--run].')
        data=json.loads(args.recipe.read_text(encoding='utf-8-sig'))
        config=ComparisonConfig(**data.get('config',data)).validate()
        print(json.dumps({'images':len(config.jobs()),'mode':config.mode,'candidates':config.candidates},indent=2))
        if not args.run:
            print('Validated only. No generation. Add --run to execute.');return 0
        result=run_comparison(config,args.output)
        print(json.dumps(result,ensure_ascii=False,indent=2))
        return 0 if result['status']=='complete' else 1
    except (ValueError,TypeError,OSError,RuntimeError) as exc:
        print('ERROR: '+str(exc),file=sys.stderr);return 1

if __name__=='__main__':raise SystemExit(main())
