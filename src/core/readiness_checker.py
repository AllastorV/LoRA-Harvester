"""
ReadinessChecker — analyzes a dataset and returns a training-readiness report.

Checks (all zero-dependency except hashlib/pathlib):
  - Image count and recommended range
  - Caption coverage (% images with .txt)
  - Tags per image (avg / min / max)
  - Tag vocabulary size
  - Exact duplicate images (SHA-256)
  - Resolution distribution (uniform vs mixed)
  - Suggested repeats for Kohya/sd-scripts

Usage:
    from src.core.readiness_checker import ReadinessChecker
    report = ReadinessChecker().check("path/to/dataset")
    print(report.summary())
"""

from __future__ import annotations

import hashlib
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# ── Result dataclass ─────────────────────────────────────────────────────────

@dataclass
class Issue:
    level: str          # 'error' | 'warning' | 'info'
    code: str           # short identifier
    message: str

    @property
    def emoji(self) -> str:
        return {"error": "❌", "warning": "⚠️", "info": "ℹ️"}.get(self.level, "•")


@dataclass
class ReadinessReport:
    # Raw stats
    image_count: int = 0
    captioned_count: int = 0
    missing_captions: int = 0
    duplicate_count: int = 0
    avg_tags: float = 0.0
    min_tags: int = 0
    max_tags: int = 0
    vocabulary_size: int = 0
    resolutions: Dict[str, int] = field(default_factory=dict)
    top_tags: List[Tuple[str, int]] = field(default_factory=list)

    # Derived
    issues: List[Issue] = field(default_factory=list)
    score: int = 100                # 0–100
    grade: str = "A"                # A / B / C / D / F
    suggested_repeats: int = 1

    def summary(self) -> str:
        """Human-readable multi-line summary."""
        lines = [
            f"Dataset: {self.image_count} images  |  "
            f"Captioned: {self.captioned_count} ({self._cap_pct():.0f}%)  |  "
            f"Score: {self.score}/100 ({self.grade})",
            f"Tags: avg {self.avg_tags:.1f}  min {self.min_tags}  max {self.max_tags}  "
            f"vocabulary {self.vocabulary_size}",
            f"Duplicates: {self.duplicate_count}  |  "
            f"Resolutions: {len(self.resolutions)} unique  |  "
            f"Suggested repeats: {self.suggested_repeats}",
        ]
        if self.issues:
            lines.append("")
            for iss in self.issues:
                lines.append(f"  {iss.emoji} {iss.message}")
        return "\n".join(lines)

    def _cap_pct(self) -> float:
        return (self.captioned_count / self.image_count * 100) if self.image_count else 0.0


# ── Checker ──────────────────────────────────────────────────────────────────

class ReadinessChecker:
    """
    Analyse a dataset folder and produce a ReadinessReport.
    """

    # Thresholds (character LoRA community consensus)
    MIN_IMAGES_WARN   = 15
    MIN_IMAGES_ERR    = 5
    MAX_IMAGES_INFO   = 500
    MIN_CAP_PCT_WARN  = 80.0   # % captioned
    MIN_CAP_PCT_ERR   = 50.0
    MIN_AVG_TAGS      = 4
    MAX_AVG_TAGS      = 60
    MIN_VOCAB         = 10

    def check(self, root: str | Path, recursive: bool = True) -> ReadinessReport:
        root = Path(root)
        report = ReadinessReport()
        issues: List[Issue] = []

        # ── 1. Collect images & captions ─────────────────────────────────────
        pattern = "**/*" if recursive else "*"
        image_paths: List[Path] = []
        for p in root.glob(pattern):
            if p.suffix.lower() in IMAGE_EXTS:
                image_paths.append(p)

        report.image_count = len(image_paths)
        if report.image_count == 0:
            issues.append(Issue("error", "no_images", "No images found in the selected folder."))
            report.issues = issues
            report.score = 0
            report.grade = "F"
            return report

        # ── 2. Caption coverage ───────────────────────────────────────────────
        tag_counts: List[int] = []
        tag_counter: Counter = Counter()
        missing_caps: List[Path] = []

        for img in image_paths:
            txt = img.with_suffix(".txt")
            if not txt.exists():
                missing_caps.append(img)
                continue
            try:
                text = txt.read_text(encoding="utf-8").strip()
            except Exception:
                missing_caps.append(img)
                continue
            if not text:
                missing_caps.append(img)
                continue
            tags = [t.strip() for t in text.split(",") if t.strip()]
            tag_counts.append(len(tags))
            tag_counter.update(t.lower().replace("_", " ") for t in tags)

        report.captioned_count = len(tag_counts)
        report.missing_captions = len(missing_caps)

        cap_pct = report.captioned_count / report.image_count * 100

        # ── 3. Tag stats ──────────────────────────────────────────────────────
        if tag_counts:
            report.avg_tags = sum(tag_counts) / len(tag_counts)
            report.min_tags = min(tag_counts)
            report.max_tags = max(tag_counts)
        report.vocabulary_size = len(tag_counter)
        report.top_tags = tag_counter.most_common(20)

        # ── 4. Duplicate detection (SHA-256) ──────────────────────────────────
        hashes: Counter = Counter()
        for img in image_paths:
            try:
                h = hashlib.sha256(img.read_bytes()).hexdigest()
                hashes[h] += 1
            except Exception:
                pass
        report.duplicate_count = sum(v - 1 for v in hashes.values() if v > 1)

        # ── 5. Resolution distribution ────────────────────────────────────────
        res_counter: Counter = Counter()
        for img in image_paths[:200]:  # sample first 200 for speed
            try:
                import struct
                data = img.read_bytes()
                if img.suffix.lower() == ".png" and data[:4] == b"\x89PNG":
                    w = struct.unpack(">I", data[16:20])[0]
                    h = struct.unpack(">I", data[20:24])[0]
                    res_counter[f"{w}x{h}"] += 1
                elif img.suffix.lower() in (".jpg", ".jpeg"):
                    # Quick JPEG dimension read
                    _w, _h = _jpeg_dims(data)
                    if _w and _h:
                        res_counter[f"{_w}x{_h}"] += 1
            except Exception:
                pass
        report.resolutions = dict(res_counter.most_common(10))

        # ── 6. Suggested repeats ──────────────────────────────────────────────
        # Community rule of thumb: aim for ~200 effective steps per concept.
        # repeats = ceil(200 / image_count), clamped to [1, 40].
        import math
        report.suggested_repeats = max(1, min(40, math.ceil(200 / report.image_count)))

        # ── 7. Build issues & score ───────────────────────────────────────────
        score = 100

        # Image count
        if report.image_count < self.MIN_IMAGES_ERR:
            issues.append(Issue("error", "too_few_images",
                f"Only {report.image_count} images — minimum recommended is {self.MIN_IMAGES_WARN}."))
            score -= 30
        elif report.image_count < self.MIN_IMAGES_WARN:
            issues.append(Issue("warning", "few_images",
                f"{report.image_count} images is below the recommended minimum of {self.MIN_IMAGES_WARN}."))
            score -= 15
        elif report.image_count > self.MAX_IMAGES_INFO:
            issues.append(Issue("info", "many_images",
                f"{report.image_count} images is large — training may be slow. "
                "Consider filtering by quality first."))

        # Caption coverage
        if cap_pct < self.MIN_CAP_PCT_ERR:
            issues.append(Issue("error", "low_caption_coverage",
                f"Only {cap_pct:.0f}% of images have captions ({report.missing_captions} missing)."))
            score -= 25
        elif cap_pct < self.MIN_CAP_PCT_WARN:
            issues.append(Issue("warning", "partial_caption_coverage",
                f"{cap_pct:.0f}% of images captioned ({report.missing_captions} missing). "
                "Aim for 100% coverage."))
            score -= 10

        # Tag quality
        if report.avg_tags < self.MIN_AVG_TAGS and report.captioned_count > 0:
            issues.append(Issue("warning", "low_avg_tags",
                f"Average {report.avg_tags:.1f} tags per image is low. "
                "More descriptive captions improve concept isolation."))
            score -= 10
        elif report.avg_tags > self.MAX_AVG_TAGS:
            issues.append(Issue("warning", "high_avg_tags",
                f"Average {report.avg_tags:.0f} tags per image is very high. "
                "Consider running Tag Cleaner to remove redundant tags."))
            score -= 5

        if report.vocabulary_size < self.MIN_VOCAB and report.captioned_count > 0:
            issues.append(Issue("warning", "low_vocabulary",
                f"Only {report.vocabulary_size} unique tags — dataset may lack diversity."))
            score -= 10

        # Duplicates
        if report.duplicate_count > 0:
            issues.append(Issue("warning", "duplicates",
                f"{report.duplicate_count} duplicate image(s) detected. "
                "Remove them for cleaner training."))
            score -= min(20, report.duplicate_count * 2)

        # All good
        if not issues:
            issues.append(Issue("info", "all_good",
                "No significant issues found. Dataset looks training-ready!"))

        report.issues = issues
        report.score = max(0, score)
        report.grade = _score_to_grade(report.score)
        return report


# ── Helpers ──────────────────────────────────────────────────────────────────

def _score_to_grade(score: int) -> str:
    if score >= 90: return "A"
    if score >= 75: return "B"
    if score >= 60: return "C"
    if score >= 40: return "D"
    return "F"


def _jpeg_dims(data: bytes) -> Tuple[Optional[int], Optional[int]]:
    """Extract width/height from JPEG without PIL."""
    try:
        import struct
        i = 0
        while i < len(data) - 9:
            if data[i] != 0xFF:
                break
            marker = data[i + 1]
            if marker in (0xC0, 0xC1, 0xC2):  # SOF0 / SOF1 / SOF2
                h = struct.unpack(">H", data[i + 5:i + 7])[0]
                w = struct.unpack(">H", data[i + 7:i + 9])[0]
                return w, h
            length = struct.unpack(">H", data[i + 2:i + 4])[0]
            i += 2 + length
    except Exception:
        pass
    return None, None
