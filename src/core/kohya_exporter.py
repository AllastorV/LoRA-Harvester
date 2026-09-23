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
import json
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional
from zipfile import ZipFile, ZIP_DEFLATED, ZIP_STORED

from src.core.dataset_scanner import detect_concepts, sanitize_name
from src.core.dataset_files import transfer_image_pair

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
        source_root = Path(source_root).resolve()
        dest_root = Path(dest_root).resolve()
        if dest_root == source_root or dest_root.is_relative_to(source_root):
            raise ValueError('Export destination must be outside the source dataset.')
        if type(repeats) is not int or repeats < 1:
            raise ValueError('Repeats must be a positive integer.')
        concept_overrides = concept_overrides or {}
        op = shutil.copy2 if copy else shutil.move

        concepts = detect_concepts(source_root)
        if not concepts:
            raise ValueError(f"No images found under {source_root}")

        exported: Dict[str, int] = {}
        concept_dirs: List[Path] = []
        class_tokens = {}

        for concept_name, pairs in concepts.items():
            if not pairs:
                logger.warning("Concept '%s' has 0 images — skipping.", concept_name)
                continue

            override = concept_overrides.get(concept_name, {})
            n_repeats = override.get('repeats', repeats)
            if type(n_repeats) is not int or n_repeats < 1:
                raise ValueError('Concept repeats must be a positive integer.')
            class_token = override.get('class_tokens', override.get('class_token', concept_name))
            if not isinstance(class_token, str) or not class_token.strip():
                raise ValueError('Provide a non-empty class token for each concept.')
            safe_name = sanitize_name(concept_name)

            # Avoid collision when two concepts sanitize to the same name
            folder_name = f"{n_repeats}_{safe_name}"
            concept_dir = dest_root / folder_name
            concept_dir = self._unique_dir(concept_dir)
            concept_dir.mkdir(parents=True, exist_ok=True)
            concept_dirs.append(concept_dir)
            class_tokens[str(concept_dir)] = class_token

            count = 0
            for pair in pairs:
                try:
                    transfer_image_pair(pair.image, concept_dir, copy=copy)
                    count += 1
                except Exception as e:
                    logger.warning("Could not export complete image/caption pair %s: %s", pair.image, e)

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
                class_tokens=class_tokens,
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
                    transfer_image_pair(pair.image, reg_dir, copy=copy)
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
        class_tokens: Optional[Dict[str, str]] = None,
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
            lines.append('  image_dir = ' + json.dumps(concept_dir.as_posix(), ensure_ascii=False))
            token = (class_tokens or {}).get(str(concept_dir), re.sub(r'^\d+_', '', concept_dir.name))
            lines.append('  class_tokens = ' + json.dumps(token, ensure_ascii=False))
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
                    lines.append('  image_dir = ' + json.dumps(sub.as_posix(), ensure_ascii=False))
                    lines.append('  class_tokens = ' + json.dumps(re.sub(r'^\d+_', '', sub.name), ensure_ascii=False))
                    m = re.match(r'^(\d+)_', sub.name)
                    n_repeats = int(m.group(1)) if m else 1
                    lines.append(f'  num_repeats = {n_repeats}')
                    lines.append('  is_reg = true')
                    lines.append('')

        toml_path.write_text('\n'.join(lines), encoding='utf-8')


def export_training_zip(source_root: Path | str, archive_path: Path | str, *,
                        cancel=None, progress=None) -> int:
    """Export configured image/caption pairs to one folder in a new ZIP."""
    root = Path(source_root).resolve()
    archive = Path(archive_path).resolve()
    if not root.is_dir():
        raise ValueError('Prepared dataset folder does not exist.')
    if archive.suffix.lower() != '.zip' or archive.is_relative_to(root):
        raise ValueError('Choose a .zip file outside the source dataset.')
    if not archive.parent.is_dir():
        raise ValueError('ZIP destination folder does not exist.')
    config_path = root / 'dataset_config.toml'
    if not config_path.is_file() or config_path.is_symlink():
        raise ValueError('Prepare the Kohya dataset before exporting a ZIP.')

    # Read configured image folders so unrelated files stay out of the archive.
    subsets = []
    for line in config_path.read_text(encoding='utf-8').splitlines():
        match = re.match(r'^(\s*image_dir\s*=\s*)("(?:\\.|[^"\\])*")\s*$', line)
        if match:
            original = Path(json.loads(match.group(2)))
            primary = (original if original.is_absolute() else root / original).resolve()
            if not original.is_absolute() and not primary.is_relative_to(root):
                raise ValueError(f'Image folder leaves the dataset: {original}')
            if primary.is_dir() and primary.is_relative_to(root):
                valid = [primary]
            else:
                valid = []
                for candidate in (root / original.name, root / 'reg' / original.name):
                    resolved = candidate.resolve()
                    if resolved.is_dir() and resolved.is_relative_to(root) and resolved not in valid:
                        valid.append(resolved)
            if len(valid) != 1:
                raise ValueError(f'Image folder is missing or ambiguous: {original}')
            subsets.append(valid[0])
    if not subsets:
        raise ValueError('dataset_config.toml has no image_dir entries.')
    pairs = sorted(
        (pair for group in detect_concepts(root).values() for pair in group
         if any(pair.image.is_relative_to(folder) for folder in subsets)),
        key=lambda pair: pair.image.relative_to(root).as_posix().casefold())
    if not pairs:
        raise ValueError('Prepared dataset contains no training images.')
    if len(pairs) != len({pair.image for pair in pairs}):
        raise ValueError('Dataset has duplicate training images.')
    members = []
    used_stems = set()
    for pair in pairs:
        stem = pair.image.stem
        if stem.casefold() in used_stems:
            base = f'{sanitize_name(pair.image.parent.name)}_{stem}'
            stem = base
            number = 2
            while stem.casefold() in used_stems:
                stem = f'{base}_{number}'
                number += 1
        used_stems.add(stem.casefold())
        members.append((pair.image, f'{archive.stem}/{stem}{pair.image.suffix}'))
        if pair.caption is not None:
            if pair.caption.is_symlink():
                raise ValueError(f'Caption is a symbolic link: {pair.caption}')
            members.append((pair.caption, f'{archive.stem}/{stem}.txt'))
    created = False
    try:
        with archive.open('xb') as target:
            created = True
            with ZipFile(target, 'w', compression=ZIP_DEFLATED, compresslevel=1) as zipped:
                for index, (path, name) in enumerate(members, 1):
                    if cancel and cancel():
                        raise InterruptedError('ZIP export cancelled.')
                    compressed = path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'}
                    zipped.write(path, name,
                                 compress_type=ZIP_STORED if compressed else ZIP_DEFLATED)
                    if progress:
                        progress(index, len(members))
    except BaseException:
        if created:
            archive.unlink(missing_ok=True)
        raise
    return len(pairs)
