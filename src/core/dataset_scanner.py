"""
Dataset Scanner — shared utility for LoRA-Harvester.
Finds (image, caption) pairs and groups them by concept folder.
Used by: ReviewGridPage, KohyaExporter.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}

# Folders to skip when scanning
_SKIP_DIRS = {'_rejected', '_approved', '__pycache__'}
_SKIP_FILES = {'_manifest.json', 'CACHEDIR.TAG'}


@dataclass
class FramePair:
    image: Path
    caption: Optional[Path]   # .txt co-located with image; None if missing
    concept: str              # parent folder name


def scan_dataset(root: Path | str, recursive: bool = True) -> List[FramePair]:
    """
    Walk *root* and return all image/caption pairs found.
    Images are matched with a co-located .txt via image.with_suffix('.txt').
    Skips _rejected/, _approved/, and _manifest.json entries.
    """
    root = Path(root)
    pairs: List[FramePair] = []

    if recursive:
        candidates = [p for p in root.rglob('*') if p.is_file()]
    else:
        candidates = [p for p in root.iterdir() if p.is_file()]

    for p in candidates:
        # Skip helper/metadata files
        if p.name in _SKIP_FILES:
            continue
        # Skip files inside excluded dirs (any ancestor named _rejected etc.)
        if any(part in _SKIP_DIRS for part in p.parts):
            continue
        if p.suffix.lower() not in IMAGE_EXTS:
            continue

        txt = p.with_suffix('.txt')
        caption = txt if txt.exists() else None
        concept = p.parent.name
        pairs.append(FramePair(image=p, caption=caption, concept=concept))

    return pairs


def detect_concepts(root: Path | str) -> Dict[str, List[FramePair]]:
    """
    Group FramePairs by concept name.
    Recognises both flat harvester layout (persons/animals/objects) and
    character-sort layout (_sorted/<character>/).
    """
    root = Path(root)
    all_pairs = scan_dataset(root)

    concepts: Dict[str, List[FramePair]] = {}
    for pair in all_pairs:
        concepts.setdefault(pair.concept, []).append(pair)

    return concepts


# ──────────────────────────────────────────────────────────
# Caption quality validation
# ──────────────────────────────────────────────────────────

ISSUE_MISSING_CAPTION = 'missing_caption'   # no .txt file at all
ISSUE_EMPTY_CAPTION   = 'empty_caption'     # .txt exists but blank
ISSUE_NO_TRIGGER      = 'no_trigger'        # trigger word absent from caption
ISSUE_LOW_TAG_COUNT   = 'low_tag_count'     # fewer tags than min_tags
ISSUE_DUPLICATE_TAGS  = 'duplicate_tags'    # same tag appears more than once


@dataclass
class CaptionIssue:
    image: Path
    caption: Optional[Path]
    issues: List[str] = field(default_factory=list)
    tag_count: int = 0
    caption_text: str = ""


def validate_captions(
    pairs: List[FramePair],
    trigger_word: str = "",
    min_tags: int = 3,
) -> List[CaptionIssue]:
    """
    Scan *pairs* for caption quality problems.

    Returns a list of CaptionIssue (one per problematic image).
    Images with no issues are NOT included.

    Args:
        pairs:        FramePairs to validate (from scan_dataset / detect_concepts).
        trigger_word: If non-empty, flag captions that don't contain it.
        min_tags:     Minimum comma-separated tag count; 0 disables this check.
    """
    issues: List[CaptionIssue] = []
    trigger = trigger_word.strip().lower()

    for pair in pairs:
        found: List[str] = []
        tag_count = 0
        caption_text = ""

        if pair.caption is None:
            found.append(ISSUE_MISSING_CAPTION)
        else:
            try:
                caption_text = pair.caption.read_text(encoding='utf-8').strip()
            except Exception:
                caption_text = ""

            if not caption_text:
                found.append(ISSUE_EMPTY_CAPTION)
            else:
                tags = [t.strip() for t in caption_text.split(',') if t.strip()]
                tag_count = len(tags)

                if trigger and trigger not in caption_text.lower():
                    found.append(ISSUE_NO_TRIGGER)

                if min_tags > 0 and tag_count < min_tags:
                    found.append(ISSUE_LOW_TAG_COUNT)

                # Duplicate detection: case-insensitive, underscore-normalized
                normalized = [t.lower().replace(' ', '_') for t in tags]
                if len(normalized) != len(set(normalized)):
                    found.append(ISSUE_DUPLICATE_TAGS)

        if found:
            issues.append(CaptionIssue(
                image=pair.image,
                caption=pair.caption,
                issues=found,
                tag_count=tag_count,
                caption_text=caption_text,
            ))

    return issues


def sanitize_name(name: str) -> str:
    """
    Normalise a concept/folder name so it is safe to use in a kohya path.
    Lowercase, replace non-[A-Za-z0-9_-] with underscore, collapse runs.
    """
    name = name.lower()
    name = re.sub(r'[^a-z0-9_\-]+', '_', name)
    name = re.sub(r'-+', '-', name)   # collapse repeated hyphens
    name = re.sub(r'_+', '_', name)   # collapse repeated underscores
    name = name.strip('_-')
    return name or 'concept'
