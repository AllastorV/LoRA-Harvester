"""
Backward-compatible entry point for the LoRA-Harvester CLI.

The implementation lives in scripts/cli.py; ``python cli.py ...`` and
``python scripts/cli.py ...`` accept exactly the same options.
"""

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / "scripts" / "cli.py"),
                   run_name="__main__")
