"""
Danbooru tag loader for autocomplete.

Sources (tried in order):
  1. Local danbooru_tags.csv in models/wd14/
  2. Any selected_tags.csv already present (WD14 download)
  3. Download DominikDoom's comprehensive tag list (~180k tags) from HF
  4. Download WD14 selected_tags.csv as fallback (~10k tags)

Tags are sorted case-insensitively so QCompleter's binary search works.
Only space-form tags are kept (no underscore duplicates) for clean prefix matching.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

from src.core.model_paths import WD14_DIR, ensure_dirs as _ensure_model_dirs
_ensure_model_dirs()

_CACHE_DIR    = str(WD14_DIR)
_LOCAL_CSV    = WD14_DIR / "danbooru_tags.csv"   # comprehensive cache file
_FALLBACK_REPO = "SmilingWolf/wd-swinv2-tagger-v3"

# Comprehensive Danbooru tag list (DominikDoom/a1111-sd-webui-tagcomplete on HF)
_COMPREHENSIVE_REPO = "DominikDoom/a1111-sd-webui-tagcomplete"
_COMPREHENSIVE_FILE = "tags/danbooru.csv"

_tag_cache: Optional[List[str]] = None


# ── Parsers ────────────────────────────────────────────────────────────────────

def _parse_comprehensive(csv_path: str) -> List[str]:
    """Parse DominikDoom's danbooru.csv — keeps space form only."""
    raw_names: List[str] = []
    try:
        import pandas as pd
        df = pd.read_csv(csv_path, usecols=[0], header=0,
                         names=['name'], skiprows=1,
                         dtype=str, on_bad_lines='skip')
        raw_names = df['name'].dropna().tolist()
    except Exception:
        with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
            f.readline()
            for line in f:
                name = line.split(',')[0].strip().strip('"')
                if name:
                    raw_names.append(name)
    return _to_space_sorted(raw_names)


def _parse_selected(csv_path: str) -> List[str]:
    """Parse WD14's selected_tags.csv — keeps space form only."""
    raw_names: List[str] = []
    try:
        import pandas as pd
        df = pd.read_csv(csv_path, dtype=str)
        if 'name' not in df.columns:
            return []
        raw_names = df['name'].dropna().tolist()
    except Exception:
        with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
            header = f.readline().strip().split(',')
            try:
                idx = header.index('name')
            except ValueError:
                return []
            for line in f:
                parts = line.rstrip('\n').split(',')
                if len(parts) > idx and parts[idx].strip():
                    raw_names.append(parts[idx].strip())
    return _to_space_sorted(raw_names)


def _to_space_sorted(raw_names: List[str]) -> List[str]:
    """Convert all tags to space form, deduplicate, sort case-insensitively."""
    seen: set = set()
    tags: List[str] = []
    for raw in raw_names:
        tag = raw.strip().replace('_', ' ')
        if not tag:
            continue
        lo = tag.lower()
        if lo not in seen:
            seen.add(lo)
            tags.append(tag)
    tags.sort(key=str.lower)
    return tags


def _sorted_unique(tags: List[str]) -> List[str]:
    seen: set = set()
    out: List[str] = []
    for t in tags:
        lo = t.lower()
        if lo not in seen:
            seen.add(lo)
            out.append(t)
    out.sort(key=str.lower)
    return out


# ── Main loader ────────────────────────────────────────────────────────────────

def load_danbooru_tags(repo_id: str = _FALLBACK_REPO) -> List[str]:
    """
    Return a sorted, deduplicated list of Danbooru tags for QCompleter.

    Priority:
      1. Memory cache
      2. Local danbooru_tags.csv (comprehensive, 180k+)
      3. Any WD14 selected_tags.csv already on disk
      4. Download comprehensive list from HuggingFace
      5. Download WD14 selected_tags.csv as last resort
    """
    global _tag_cache
    if _tag_cache is not None:
        return _tag_cache

    raw: List[str] = []

    # 2. Local comprehensive CSV
    if _LOCAL_CSV.exists():
        try:
            raw = _parse_comprehensive(str(_LOCAL_CSV))
            if raw:
                logger.info("Loaded %d tags from local %s", len(raw), _LOCAL_CSV)
        except Exception as e:
            logger.warning("Failed reading %s: %s", _LOCAL_CSV, e)
            raw = []

    # 3. Any selected_tags.csv already on disk
    if not raw:
        cache_root = Path(_CACHE_DIR)
        for csv_path in cache_root.rglob("selected_tags.csv"):
            try:
                raw = _parse_selected(str(csv_path))
                if raw:
                    logger.info("Loaded %d tags from %s", len(raw), csv_path)
                    break
            except Exception as e:
                logger.warning("Failed parsing %s: %s", csv_path, e)

    # 4. Download comprehensive list
    if not raw:
        try:
            from huggingface_hub import hf_hub_download
            logger.info("Downloading comprehensive Danbooru tag list…")
            path = hf_hub_download(
                repo_id=_COMPREHENSIVE_REPO,
                filename=_COMPREHENSIVE_FILE,
                cache_dir=_CACHE_DIR,
            )
            raw = _parse_comprehensive(path)
            # Save a local copy (already sorted+both-forms) for fast next launch
            if raw:
                try:
                    _LOCAL_CSV.write_text(
                        '\n'.join(raw), encoding='utf-8')
                except Exception:
                    pass
                logger.info("Downloaded %d Danbooru tags", len(raw))
        except Exception as e:
            logger.warning("Comprehensive tag download failed: %s", e)

    # 5. Fallback: WD14 selected_tags.csv
    if not raw:
        try:
            from huggingface_hub import hf_hub_download
            logger.info("Falling back to WD14 tag list…")
            path = hf_hub_download(
                repo_id=repo_id,
                filename="selected_tags.csv",
                cache_dir=_CACHE_DIR,
            )
            raw = _parse_selected(path)
            logger.info("Loaded %d tags (WD14 fallback)", len(raw))
        except Exception as e:
            logger.warning("WD14 tag download failed: %s", e)

    _tag_cache = _sorted_unique(raw)
    logger.info("Tag cache ready: %d unique tags", len(_tag_cache))
    return _tag_cache


def get_cached_tags() -> List[str]:
    """Return the currently cached tag list (empty if not yet loaded)."""
    return _tag_cache or []


def invalidate_cache() -> None:
    """Force reload on next load_danbooru_tags() call."""
    global _tag_cache
    _tag_cache = None
