"""
Lightweight Danbooru tag loader for autocomplete.

Reuses the `selected_tags.csv` that `AdvancedCaptioner` downloads for its
WD14 tagger so the autocomplete UI and the captioner share one cached file
(`.cache/wd14/`). Only the tag list is loaded — no ONNX model, so this is
cheap to call from the UI thread even on a cold start.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

_CACHE_DIR = ".cache/wd14"
_DEFAULT_REPO = "SmilingWolf/wd-v1-4-vit-tagger-v2"

# Module-level cache so repeated calls are free.
_tag_cache: Optional[List[str]] = None


def _parse_csv(csv_path: str) -> List[str]:
    """Parse selected_tags.csv and return a display-friendly tag list."""
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        if 'name' not in df.columns:
            logger.warning("selected_tags.csv has no 'name' column: %s", csv_path)
            return []
        names = df['name'].astype(str).tolist()
    except ImportError:
        # Fallback without pandas.
        names = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            header = f.readline().strip().split(',')
            try:
                name_idx = header.index('name')
            except ValueError:
                logger.warning("selected_tags.csv has no 'name' column: %s", csv_path)
                return []
            for line in f:
                parts = line.rstrip('\n').split(',')
                if len(parts) > name_idx:
                    names.append(parts[name_idx])

    # Danbooru tags use underscores; human-readable form uses spaces. Keep
    # both so the completer matches either typing style. De-dupe while
    # preserving order.
    seen = set()
    tags: List[str] = []
    for raw in names:
        raw = raw.strip()
        if not raw:
            continue
        pretty = raw.replace('_', ' ')
        for form in (pretty, raw):
            if form not in seen:
                seen.add(form)
                tags.append(form)
    return tags


def load_danbooru_tags(repo_id: str = _DEFAULT_REPO) -> List[str]:
    """
    Return a flat list of Danbooru tags suitable for a QCompleter.

    Tries in order:
      1. module-level cache
      2. already-downloaded `selected_tags.csv` under `.cache/wd14/` (scans
         the snapshots folder so we don't need network if any wd14 repo has
         been fetched before)
      3. `huggingface_hub.hf_hub_download` for `repo_id` (requires network)

    Returns an empty list on any failure. Never raises.
    """
    global _tag_cache
    if _tag_cache is not None:
        return _tag_cache

    # 2. Scan local cache for any existing selected_tags.csv.
    cache_root = Path(_CACHE_DIR)
    if cache_root.exists():
        for csv_path in cache_root.rglob("selected_tags.csv"):
            try:
                tags = _parse_csv(str(csv_path))
                if tags:
                    logger.info("Loaded %d Danbooru tags from cache: %s", len(tags), csv_path)
                    _tag_cache = tags
                    return _tag_cache
            except Exception as e:
                logger.warning("Failed to parse cached %s: %s", csv_path, e)

    # 3. Fall back to Hugging Face download.
    try:
        from huggingface_hub import hf_hub_download
        tags_path = hf_hub_download(
            repo_id=repo_id,
            filename="selected_tags.csv",
            cache_dir=_CACHE_DIR,
        )
        tags = _parse_csv(tags_path)
        if tags:
            logger.info("Downloaded %d Danbooru tags from %s", len(tags), repo_id)
            _tag_cache = tags
            return _tag_cache
    except Exception as e:
        logger.warning("Could not download Danbooru tag list (%s): %s", repo_id, e)

    _tag_cache = []
    return _tag_cache


def get_cached_tags() -> List[str]:
    """Return the currently cached tag list (empty if not yet loaded)."""
    return _tag_cache or []
