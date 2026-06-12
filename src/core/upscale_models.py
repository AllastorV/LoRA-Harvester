"""
Upscale Model Registry for LoRA-Harvester.

Built-in models are defined here (Real-ESRGAN family).
Custom models are stored in models/upscale/models.json and auto-discovered
from any .pth files placed directly in models/upscale/.

Usage:
    from src.core.upscale_models import list_models, resolve, add_custom_model

    # All available models (built-in + custom + auto-discovered)
    models = list_models()

    # Resolve a model by name → get config dict + ensure weights downloaded
    cfg = resolve("RealESRGAN_x4plus")

    # Register a custom .pth file the user placed manually
    add_custom_model(Path("C:/my_model.pth"), name="my_4x", scale=4)
"""

from __future__ import annotations
import json
import logging
import shutil
from pathlib import Path
from typing import Dict

from src.core.model_paths import UPSCALE_DIR, upscale_model_path, ensure_dirs

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────
# Built-in model catalogue
# ──────────────────────────────────────────────────────────
# Keys: model name shown in UI/CLI
# Values:
#   scale      - output / input resolution multiplier
#   arch       - "RRDBNet" or "SRVGGNetCompact"
#   num_feat   - feature channels (RRDBNet)
#   num_block  - RRDB blocks (RRDBNet) / num_conv (SRVGGNet)
#   num_grow_ch- grow channels for RRDBNet (optional, default 32)
#   url        - direct download URL for the .pth weights
#   file       - filename inside UPSCALE_DIR

BUILTIN_MODELS: Dict[str, dict] = {
    "RealESRGAN_x4plus": {
        "scale": 4,
        "arch": "RRDBNet",
        "num_feat": 64,
        "num_block": 23,
        "num_grow_ch": 32,
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "file": "RealESRGAN_x4plus.pth",
        "description": "General purpose 4× (photo realistic, slower)",
    },
    "RealESRGAN_x4plus_anime_6B": {
        "scale": 4,
        "arch": "RRDBNet",
        "num_feat": 64,
        "num_block": 6,
        "num_grow_ch": 32,
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        "file": "RealESRGAN_x4plus_anime_6B.pth",
        "description": "Anime / illustration 4× (lightweight, recommended for LoRA)",
    },
    "RealESRGAN_x2plus": {
        "scale": 2,
        "arch": "RRDBNet",
        "num_feat": 64,
        "num_block": 23,
        "num_grow_ch": 32,
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        "file": "RealESRGAN_x2plus.pth",
        "description": "General purpose 2× (faster, less aggressive upscale)",
    },
    "realesr-general-x4v3": {
        "scale": 4,
        "arch": "SRVGGNetCompact",
        "num_feat": 64,
        "num_block": 32,  # num_conv for SRVGGNetCompact
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
        "file": "realesr-general-x4v3.pth",
        "description": "General 4× with tunable denoise (fast, SRVGGNet arch)",
    },
    "realesr-animevideov3": {
        "scale": 4,
        "arch": "SRVGGNetCompact",
        "num_feat": 64,
        "num_block": 16,
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth",
        "file": "realesr-animevideov3.pth",
        "description": "Anime video 4× (SRVGGNet, ideal for video frames)",
    },
}

# ──────────────────────────────────────────────────────────
# Custom manifest (persisted to disk)
# ──────────────────────────────────────────────────────────
_MANIFEST_PATH: Path = UPSCALE_DIR / "models.json"


def _load_custom_manifest() -> Dict[str, dict]:
    """Read models/upscale/models.json; return {} if missing or corrupt."""
    ensure_dirs()
    if not _MANIFEST_PATH.exists():
        return {}
    try:
        with open(_MANIFEST_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        logger.warning("upscale_models: failed to read %s: %s", _MANIFEST_PATH, e)
        return {}


def _save_custom_manifest(data: Dict[str, dict]) -> None:
    ensure_dirs()
    with open(_MANIFEST_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ──────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────

def list_models() -> Dict[str, dict]:
    """
    Return all available models:
      1. Built-in models (BUILTIN_MODELS)
      2. Custom models from models/upscale/models.json
      3. Auto-discovered .pth files in models/upscale/ not already listed

    Each entry has at minimum: scale, arch, file, description.
    Auto-discovered files get default arch=RRDBNet, scale=4.
    An 'available' key indicates whether the .pth file exists locally.
    """
    ensure_dirs()
    result: Dict[str, dict] = {}

    # 1. Built-ins
    for name, cfg in BUILTIN_MODELS.items():
        entry = dict(cfg)
        entry['source'] = 'builtin'
        entry['available'] = upscale_model_path(cfg['file']).exists()
        result[name] = entry

    # 2. Custom manifest
    custom = _load_custom_manifest()
    for name, cfg in custom.items():
        if name not in result:
            entry = dict(cfg)
            entry.setdefault('description', 'Custom model')
            entry.setdefault('arch', 'RRDBNet')
            entry.setdefault('scale', 4)
            entry.setdefault('num_feat', 64)
            entry.setdefault('num_block', 23)
            entry['source'] = 'custom'
            entry['available'] = upscale_model_path(cfg['file']).exists()
            result[name] = entry

    # 3. Auto-discover: .pth files in upscale dir not already catalogued
    known_files = {v['file'] for v in result.values()}
    for pth in sorted(UPSCALE_DIR.glob('*.pth')):
        if pth.name not in known_files:
            name = pth.stem
            # avoid clobbering known name collision
            while name in result:
                name += '_'
            result[name] = {
                'scale': 4,
                'arch': 'RRDBNet',
                'num_feat': 64,
                'num_block': 23,
                'num_grow_ch': 32,
                'file': pth.name,
                'description': f'Auto-discovered: {pth.name}',
                'source': 'discovered',
                'available': True,
            }

    return result


def resolve(name: str) -> dict:
    """
    Return the config dict for *name*.
    If the model has a 'url' and the local .pth is missing, attempt download.
    Raises KeyError if name unknown, RuntimeError on download failure.
    """
    models = list_models()
    if name not in models:
        available = ', '.join(sorted(models.keys()))
        raise KeyError(f"Unknown upscale model '{name}'. Available: {available}")

    cfg = dict(models[name])
    pth_path = upscale_model_path(cfg['file'])

    if not pth_path.exists():
        url = cfg.get('url')
        if url:
            logger.info("Downloading upscale model '%s' from %s …", name, url)
            try:
                import torch
                torch.hub.download_url_to_file(url, str(pth_path), progress=True)
                logger.info("Downloaded → %s", pth_path)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to download upscale model '{name}': {e}\n"
                    f"Manual download URL: {url}\n"
                    f"Save to: {pth_path}"
                ) from e
        else:
            raise RuntimeError(
                f"Upscale model '{name}' not found at {pth_path} "
                f"and no download URL is available. "
                f"Place the .pth file manually in {UPSCALE_DIR}/"
            )

    cfg['path'] = str(pth_path)
    return cfg


def add_custom_model(
    pth_path: Path | str,
    name: str,
    scale: int = 4,
    arch: str = "RRDBNet",
    num_block: int = 23,
    num_feat: int = 64,
    description: str = "",
    overwrite: bool = False,
) -> None:
    """
    Register a custom .pth model file.
    Copies the file into models/upscale/ and saves to models.json.

    Args:
        pth_path:    Source .pth file path.
        name:        Name shown in UI / used in CLI.
        scale:       Output scale factor (2 or 4).
        arch:        Architecture: 'RRDBNet' or 'SRVGGNetCompact'.
        num_block:   Number of RRDB blocks (RRDBNet) or conv layers (SRVGGNet).
        num_feat:    Feature channels.
        description: Human-readable description.
        overwrite:   Replace existing entry if name already exists.
    """
    ensure_dirs()
    pth_path = Path(pth_path)
    if not pth_path.exists():
        raise FileNotFoundError(f"Source model not found: {pth_path}")

    existing = list_models()
    if name in existing and not overwrite:
        raise ValueError(
            f"Model '{name}' already exists. "
            f"Set overwrite=True or choose a different name."
        )

    dest = upscale_model_path(pth_path.name)
    if dest != pth_path:
        shutil.copy2(pth_path, dest)
        logger.info("Copied %s → %s", pth_path, dest)

    manifest = _load_custom_manifest()
    manifest[name] = {
        'file': dest.name,
        'scale': scale,
        'arch': arch,
        'num_block': num_block,
        'num_feat': num_feat,
        'num_grow_ch': 32,
        'description': description or f"Custom {scale}× {arch}",
    }
    _save_custom_manifest(manifest)
    logger.info("Registered custom upscale model '%s'", name)


def remove_custom_model(name: str, delete_file: bool = False) -> None:
    """Remove a custom model from the manifest. Optionally delete the .pth file."""
    manifest = _load_custom_manifest()
    if name not in manifest:
        raise KeyError(f"'{name}' not found in custom manifest.")
    entry = manifest.pop(name)
    if delete_file:
        p = upscale_model_path(entry['file'])
        if p.exists():
            p.unlink()
            logger.info("Deleted model file %s", p)
    _save_custom_manifest(manifest)
    logger.info("Removed custom model '%s'", name)
