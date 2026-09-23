"""LoRA Harvester LIVE MCP entry. Uses stdlib only; GUI must explicitly allow it.

Old independent CLI tools are preserved in mcp_legacy_server.py, but are NOT
loaded by this entry and do not share the new GUI permissions or request ledger.
"""
from pathlib import Path
from src.core.ai_mcp import serve

if __name__ == '__main__':
    serve(Path(__file__).resolve().parent / '.ai-control' / 'session.json')
