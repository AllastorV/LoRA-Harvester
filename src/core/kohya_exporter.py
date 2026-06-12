"""
Kohya Exporter for LoRA-Harvester.

Converts a harvester output folder into the kohya_ss / sd-scripts
repeats-folder convention:
    <dest>/<repeats>_<concept_name>/image.jpg
    <dest>/<repeats>_<concept_name>/image.txt
    <dest>/dataset_config.toml  (optional)

Usage:
    from src.core.kohya_exporter import KohyaExporter
    exporter = KohyaExporter()
    counts = exporter.export(
        source_root=Path("output/video_1x1_yolo/"),
        dest_root=Path("kohya_dataset/"),
        repeats=10,
        copy=True,
        gen_toml=True,
    )
    # counts = {"persons": 120, "character_01": 45, ...}
"""

from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional

from src.core.dataset_scanner import detect_concepts, sanitize_name

logger = logging.getLogger(__name__)


class KohyaExporter:
    """
    Exports a LoRA-Harvester dataset folder to kohya_ss / sd-scripts structure.

    Supports both the flat harvester layout (persons/animals/objects) and the
    character-sort layout (_sorted/<character>/).
    """

    def export(
        self,
        source_root: Path | str,
        dest_root: Path | str,
        repeats: int = 10,
        copy: bool = True,
        concept_overrides: Optional[Dict[str, dict]] = None,
        gen_toml: bool = True,
        resolution: int = 1024,
        reg_dir: Optional[Path | str] = None,
        reg_repeats: int = 1,
    ) -> Dict[str, int]:
        """
        Export *source_root* into kohya structure under *dest_root*.

        Args:
            source_root:       Harvester output folder to export.
            dest_root:         Destination root for the kohya dataset.
            repeats:           Default training repeat count (used in folder name).
            copy:              If True, copy files; if False, move them.
            concept_overrides: Per-concept overrides:
                               {"persons": {"repeats": 5, "class_token": "person"}}.
            gen_toml:          Write dataset_config.toml.
            resolution:        Resolution for [general] block in toml.
            reg_dir:           Optional regularisation image root.
            reg_repeats:       Repeat count for reg images.

        Returns:
            Dict mapping concept_name → number of images exported.
        """
        source_root = Path(source_root)
        dest_root = Path(dest_root)
        concept_overrides = concept_overrides or {}
        op = shutil.copy2 if copy else shutil.move

        concepts = detect_concepts(source_root)
        if not concepts:
            raise ValueError(f"No images found under {source_root}")

        exported: Dict[str, int] = {}
        concept_dirs: List[Path] = []

        for concept_name, pairs in concepts.items():
            if not pairs:
                logger.warning("Concept '%s' has 0 images — skipping.", concept_name)
                continue

            override = concept_overrides.get(concept_name, {})
            n_repeats = override.get('repeats', repeats)
            safe_name = sanitize_name(concept_name)

            # Avoid collision when two concepts sanitize to the same name
            folder_name = f"{n_repeats}_{safe_name}"
            concept_dir = dest_root / folder_name
            concept_dir = self._unique_dir(concept_dir)
            concept_dir.mkdir(parents=True, exist_ok=True)
            concept_dirs.append(concept_dir)

            count = 0
            for pair in pairs:
                dst_img = self._unique_dest(concept_dir / pair.image.name)
                try:
                    op(str(pair.image), str(dst_img))
                    count += 1
                except Exception as e:
                    logger.warning("Could not export %s: %s", pair.image, e)
                    continue

                if pair.caption and pair.caption.exists():
                    dst_txt = dst_img.with_suffix('.txt')
                    try:
                        op(str(pair.caption), str(dst_txt))
                    except Exception as e:
                        logger.warning("Could not export caption %s: %s", pair.caption, e)
                # No caption → kohya uses folder class token, that's fine.

            exported[concept_name] = count
            logger.info("Exported concept '%s' → %s (%d images)", concept_name, concept_dir, count)

        # Regularisation images
        if reg_dir:
            reg_dir = Path(reg_dir)
            self._export_reg(reg_dir, dest_root, reg_repeats, copy)

        # dataset_config.toml
        if gen_toml:
            toml_path = dest_root / "dataset_config.toml"
            self._write_toml(
                toml_path=toml_path,
                concept_dirs=concept_dirs,
                resolution=resolution,
                reg_dir=dest_root / "reg" if reg_dir else None,
            )
            logger.info("Wrote %s", toml_path)

        return exported

    # ──────────────────────────
    # Internal helpers
    # ──────────────────────────

    def _export_reg(
        self,
        reg_source: Path,
        dest_root: Path,
        reg_repeats: int,
        copy: bool,
    ) -> None:
        """Export regularisation images into dest_root/reg/<repeats>_<concept>/."""
        op = shutil.copy2 if copy else shutil.move
        reg_concepts = detect_concepts(reg_source)
        for concept_name, pairs in reg_concepts.items():
            safe = sanitize_name(concept_name)
            reg_dir = dest_root / "reg" / f"{reg_repeats}_{safe}"
            reg_dir.mkdir(parents=True, exist_ok=True)
            for pair in pairs:
                try:
                    op(str(pair.image), str(self._unique_dest(reg_dir / pair.image.name)))
                except Exception as e:
                    logger.warning("Reg copy failed %s: %s", pair.image, e)

    @staticmethod
    def _unique_dir(path: Path) -> Path:
        """Return a non-colliding directory path by appending _N."""
        if not path.exists():
            return path
        n = 1
        while True:
            candidate = path.parent / f"{path.name}_{n}"
            if not candidate.exists():
                return candidate
            n += 1

    @staticmethod
    def _unique_dest(path: Path) -> Path:
        """Return a non-colliding file path by appending _N before extension."""
        if not path.exists():
            return path
        stem, suffix = path.stem, path.suffix
        n = 1
        while True:
            candidate = path.parent / f"{stem}_{n}{suffix}"
            if not candidate.exists():
                return candidate
            n += 1

    @staticmethod
    def _write_toml(
        toml_path: Path,
        concept_dirs: List[Path],
        resolution: int,
        reg_dir: Optional[Path],
    ) -> None:
        """Write a minimal sd-scripts dataset_config.toml by hand (no dep needed)."""
        lines: List[str] = []
        lines.append("[general]")
        lines.append(f'resolution = {resolution}')
        lines.append('shuffle_caption = true')
        lines.append('keep_tokens = 1')
        lines.append('')
        lines.append('[[datasets]]')
        # Each concept folder → one subset
        for concept_dir in concept_dirs:
            lines.append('  [[datasets.subsets]]')
            lines.append(f'  image_dir = "{concept_dir.as_posix()}"')
            # Parse repeats from folder name (N_concept)
            m = re.match(r'^(\d+)_', concept_dir.name)
            n_repeats = int(m.group(1)) if m else 1
            lines.append(f'  num_repeats = {n_repeats}')
            lines.append('')

        if reg_dir and reg_dir.exists():
            lines.append('[[datasets]]')
            for sub in sorted(reg_dir.iterdir()):
                if sub.is_dir():
                    lines.append('  [[datasets.subsets]]')
                    lines.append(f'  image_dir = "{sub.as_posix()}"')
                    m = re.match(r'^(\d+)_', sub.name)
                    n_repeats = int(m.group(1)) if m else 1
                    lines.append(f'  num_repeats = {n_repeats}')
                    lines.append('  is_reg = true')
                    lines.append('')

        toml_path.write_text('\n'.join(lines), encoding='utf-8')
