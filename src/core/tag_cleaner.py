"""
TagCleaner — removes semantic redundancies and duplicates from Danbooru captions.

Handles per-file:
  1. Exact duplicates          "1girl, 1girl, solo" → "1girl, solo"
  2. Underscore/space aliases  "long_hair" ↔ "long hair" (keeps first seen)
  3. Semantic subsumptions     e.g. "solo female" is redundant if "1girl" present

Usage:
    from src.core.tag_cleaner import TagCleaner
    cleaner = TagCleaner()
    stats = cleaner.clean_folder("path/to/dataset")
    # stats: {'files': N, 'cleaned': M, 'tags_removed': K}
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


# ── Semantic alias groups ────────────────────────────────────────────────────
# Each tuple: (canonical_tag, [aliases_that_are_redundant_if_canonical_present])
# ALL in space form (underscores replaced with spaces).
_ALIAS_GROUPS: List[Tuple[str, List[str]]] = [
    # Count tags
    ("1girl",  ["solo female", "one girl", "girl solo"]),
    ("1boy",   ["solo male", "one boy", "boy solo"]),
    ("solo",   ["alone", "single", "only one"]),
    # Location
    ("outdoors", ["outside", "outdoor"]),
    ("indoors",  ["inside", "indoor"]),
    # Looking
    ("looking at viewer", ["looking at camera", "staring at viewer"]),
    # Hair length
    ("long hair", ["very long hair"]),   # very long hair is separate; keep both? no — very specific
    # Common redundancies
    ("simple background", ["plain background", "blank background"]),
    ("white background",  ["solid white background"]),
    # Body
    ("large breasts",  ["big breasts", "huge breasts"]),
    ("small breasts",  ["flat chest", "tiny breasts"]),
    # Common pairs where one subsumes the other
    ("smile",   ["smiling"]),
    ("blush",   ["blushing"]),
    ("standing", ["standing up"]),
    ("sitting",  ["sitting down"]),
    ("lying",    ["lying down", "laying down"]),
    ("running",  ["running away"]),
    ("holding",  ["holding object"]),
    # Rating tags (keep only one)
    ("rating:general",    ["rating:safe", "safe"]),
    ("rating:sensitive",  []),
    ("rating:questionable", ["rating:q"]),
    ("rating:explicit",   ["rating:e", "explicit content"]),
]

# Tags that are almost always noise for character LoRAs
_NOISE_TAGS: frozenset = frozenset({
    "absurdres", "highres", "best quality", "masterpiece",
    "ultra detailed", "ultra-detailed", "extremely detailed",
    "comic", "scan", "translation request", "translated",
})

# Build fast lookup: alias → canonical
_ALIAS_LOOKUP: Dict[str, str] = {}
for _canon, _aliases in _ALIAS_GROUPS:
    for _a in _aliases:
        _ALIAS_LOOKUP[_a] = _canon
        _ALIAS_LOOKUP[_a.replace(" ", "_")] = _canon.replace(" ", "_")


def _normalize(tag: str) -> str:
    """Lowercase + collapse whitespace. Keep underscores as-is."""
    return " ".join(tag.strip().lower().split())


def _space_form(tag: str) -> str:
    return _normalize(tag).replace("_", " ")


def clean_tags(raw: str, remove_noise: bool = False) -> Tuple[str, int]:
    """
    Clean a comma-separated tag string.

    Returns
    -------
    (cleaned_string, n_removed)
    """
    if not raw or not raw.strip():
        return raw, 0

    tags = [t.strip() for t in raw.split(",") if t.strip()]
    original_count = len(tags)

    seen_space: OrderedDict[str, str] = OrderedDict()  # space_form → original

    for tag in tags:
        sf = _space_form(tag)

        # Noise removal
        if remove_noise and sf in _NOISE_TAGS:
            continue

        # Semantic alias: if this tag is an alias of something already present,
        # skip it.  If the canonical form is NOT yet present, add the canonical.
        canon = _ALIAS_LOOKUP.get(sf) or _ALIAS_LOOKUP.get(tag.lower())
        if canon:
            canon_sf = _space_form(canon)
            if canon_sf in seen_space:
                continue              # canonical already present — drop alias
            # Otherwise add the canonical form (prefer that over the alias)
            seen_space[canon_sf] = canon
            continue

        # Exact/underscore duplicate check
        if sf in seen_space:
            continue

        seen_space[sf] = tag  # keep original casing/form from first occurrence

    cleaned = ", ".join(seen_space.values())
    n_removed = original_count - len(seen_space)
    return cleaned, n_removed


class TagCleaner:
    """
    Cleans all .txt caption files in a dataset folder.
    """

    def __init__(self, remove_noise: bool = False) -> None:
        self.remove_noise = remove_noise

    def clean_folder(
        self,
        root: str | Path,
        dry_run: bool = False,
        recursive: bool = True,
    ) -> dict:
        """
        Walk *root*, clean every .txt caption file in-place.

        Parameters
        ----------
        root       : Dataset root folder.
        dry_run    : If True, report changes without writing.
        recursive  : Recurse into subdirectories.

        Returns
        -------
        dict with keys: files_scanned, files_changed, tags_removed, errors
        """
        root = Path(root)
        stats = {"files_scanned": 0, "files_changed": 0, "tags_removed": 0, "errors": 0}

        pattern = "**/*.txt" if recursive else "*.txt"
        txt_files = list(root.glob(pattern))

        for txt_path in txt_files:
            try:
                original = txt_path.read_text(encoding="utf-8").strip()
                cleaned, n_removed = clean_tags(original, self.remove_noise)
                stats["files_scanned"] += 1

                if n_removed > 0:
                    stats["files_changed"] += 1
                    stats["tags_removed"] += n_removed
                    if not dry_run:
                        txt_path.write_text(cleaned, encoding="utf-8")
                        logger.debug("Cleaned %s: removed %d tag(s)", txt_path.name, n_removed)
            except Exception as exc:
                logger.warning("Failed to process %s: %s", txt_path, exc)
                stats["errors"] += 1

        return stats

    def preview(self, root: str | Path, recursive: bool = True) -> List[dict]:
        """
        Return a list of per-file diff previews without writing anything.

        Each entry: {path, original_count, cleaned_count, removed_tags}
        """
        root = Path(root)
        results = []
        pattern = "**/*.txt" if recursive else "*.txt"

        for txt_path in root.glob(pattern):
            try:
                original = txt_path.read_text(encoding="utf-8").strip()
                if not original:
                    continue
                orig_tags = [t.strip() for t in original.split(",") if t.strip()]
                cleaned, n_removed = clean_tags(original, self.remove_noise)
                if n_removed > 0:
                    clean_tags_list = [t.strip() for t in cleaned.split(",") if t.strip()]
                    removed = [t for t in orig_tags if t not in clean_tags_list]
                    results.append({
                        "path": txt_path,
                        "original_count": len(orig_tags),
                        "cleaned_count": len(clean_tags_list),
                        "removed_tags": removed[:10],  # cap preview
                    })
            except Exception:
                pass

        return results
