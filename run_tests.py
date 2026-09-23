#!/usr/bin/env python3
"""Run all packaged regression tests; never install packages or download models."""
from pathlib import Path
import os
import subprocess
import sys
import importlib.util

def main():
    root = Path(__file__).resolve().parent
    if importlib.util.find_spec('pytest') is None:
        print('Tests require pytest. Install requirements-dev.txt into the project venv.', file=sys.stderr)
        return 2
    env = dict(os.environ, QT_QPA_PLATFORM='offscreen', PYTHONUTF8='1')
    return subprocess.call([sys.executable, '-m', 'pytest', 'tests', *sys.argv[1:]], cwd=root, env=env)

if __name__ == '__main__':
    raise SystemExit(main())
