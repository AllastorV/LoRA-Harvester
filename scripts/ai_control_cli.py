#!/usr/bin/env python3
"""Print connector configs, check the live bridge or call a named action."""
from pathlib import Path
import argparse
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.core.ai_client import LiveClient, connection_config
from src.core.ai_control import ControlError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('config'); sub.add_parser('check'); sub.add_parser('capabilities')
    action = sub.add_parser('call'); action.add_argument('action'); action.add_argument('--args', default='{}')
    action.add_argument('--command-id', default=None)
    status = sub.add_parser('status'); status.add_argument('request_id')
    args = parser.parse_args()
    try:
        if args.command == 'config':
            print(connection_config(ROOT, sys.executable)); return 0
        client = LiveClient(ROOT / '.ai-control' / 'session.json')
        if args.command == 'call':
            data = client.call(args.action, json.loads(args.args), args.command_id)
        elif args.command == 'status':
            from src.core.ai_control import ID_RE
            if not ID_RE.fullmatch(args.request_id):
                raise ValueError('Invalid request ID.')
            data = client.request('GET', '/requests/' + args.request_id)
        elif args.command == 'capabilities':
            data = client.request('GET', '/capabilities')
        else:
            data = client.call('app.state', {})
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return 1 if data.get('status') in ('failed', 'rejected') else 0
    except (ControlError, OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr); return 2

if __name__ == '__main__':
    raise SystemExit(main())
