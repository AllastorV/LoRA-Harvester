"""
TrainingConfigBuilder — generates Kohya ss / sd-scripts TOML training configs.

Supports:
  - SD 1.5 / SDXL LoRA training
  - Auto-calculates steps from dataset size
  - Writes dataset_config.toml + train_config.toml
  - clip_skip, noise_offset, min_snr_gamma, network_dropout
  - lr_scheduler, optimizer_type, mixed_precision overrides

Usage:
    from src.training.config_builder import TrainingConfigBuilder
    builder = TrainingConfigBuilder()
    paths = builder.build(
        dataset_dir="kohya_dataset/",
        output_dir="output_lora/",
        base_model="v1-5-pruned-emaonly.safetensors",
        lora_name="my_character",
    )
    # paths: {"dataset_toml": Path, "train_toml": Path}
"""

from __future__ import annotations

import math
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional


# TOML-writing helpers (no external dep needed)
def _toml_val(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return f'"{v}"'
    if isinstance(v, float):
        return f"{v}"
    return str(v)


def _toml_section(name: str, data: dict) -> str:
    lines = [f"[{name}]"]
    for k, v in data.items():
        if v is None:
            continue
        lines.append(f"{k} = {_toml_val(v)}")
    return "\n".join(lines)


class TrainingConfigBuilder:
    """
    Builds Kohya-compatible training configs from harvester datasets.
    """

    # Recommended defaults per LoRA community
    DEFAULTS = {
        # Resolution / data
        "resolution":                   1024,
        "batch_size":                   1,
        "gradient_accumulation_steps":  4,
        # Network
        "network_dim":                  32,
        "network_alpha":                32,
        "network_dropout":              0.0,
        # Learning rates
        "learning_rate":                1e-4,
        "unet_lr":                      1e-4,
        "text_encoder_lr":              1e-5,
        "lr_scheduler":                 "cosine_with_restarts",
        "lr_warmup_steps":              0,
        "lr_scheduler_num_cycles":      1,
        # Optimizer
        "optimizer":                    "AdamW8bit",
        # Precision
        "mixed_precision":              "fp16",
        # Training
        "save_every_n_epochs":          1,
        "max_train_epochs":             10,
        "clip_skip":                    1,
        # Noise / stability
        "noise_offset":                 0.0,
        "min_snr_gamma":                0.0,
        # Caption
        "caption_extension":            ".txt",
        "shuffle_caption":              True,
        "keep_tokens":                  1,
        # Augmentation
        "flip_aug":                     False,
        # Bucket
        "enable_bucket":                True,
        "min_bucket_reso":              256,
        "max_bucket_reso":              2048,
        "bucket_reso_steps":            64,
        # Perf (8 GB VRAM tuned)
        "cache_latents_to_disk":        True,
        "persistent_data_loader_workers": False,
        "max_data_loader_n_workers":    0,
    }

    @staticmethod
    def _image_count(path: Path) -> int:
        image_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        return sum(1 for p in path.rglob("*")
                   if p.is_file() and p.suffix.lower() in image_exts)

    @staticmethod
    def _caption_count(path: Path, extension: str = ".txt") -> int:
        return sum(1 for p in path.rglob(f"*{extension}")
                   if p.is_file() and p.read_text(encoding="utf-8", errors="ignore").strip())

    @classmethod
    def _sync_missing_captions(cls, subsets: List[dict], extension: str = ".txt") -> int:
        """
        Copy captions from captioned sibling subsets to matching image names.

        This mainly fixes upscaled folders: upscalers often write image files only,
        but the original sibling folder already has captions with the same stems.
        """
        caption_by_stem: dict[str, Path] = {}
        image_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

        for subset in subsets:
            image_dir = Path(subset["image_dir"])
            for txt in image_dir.rglob(f"*{extension}"):
                if not txt.is_file():
                    continue
                try:
                    if not txt.read_text(encoding="utf-8", errors="ignore").strip():
                        continue
                except Exception:
                    continue
                caption_by_stem.setdefault(txt.stem.lower(), txt)

        copied = 0
        if not caption_by_stem:
            return copied

        for subset in subsets:
            image_dir = Path(subset["image_dir"])
            for image in image_dir.rglob("*"):
                if not image.is_file() or image.suffix.lower() not in image_exts:
                    continue
                dst_txt = image.with_suffix(extension)
                if dst_txt.exists() and dst_txt.read_text(encoding="utf-8", errors="ignore").strip():
                    continue
                src_txt = caption_by_stem.get(image.stem.lower())
                if not src_txt or src_txt == dst_txt:
                    continue
                try:
                    shutil.copy2(src_txt, dst_txt)
                    copied += 1
                except Exception:
                    continue
        return copied

    @classmethod
    def _detect_subsets(cls, dataset_dir: Path, repeats: int) -> List[dict]:
        """
        Return Kohya dataset subsets.

        If dataset_dir is already a repeats-folder root, e.g.
        `dataset/10_character/*.png`, each immediate child folder becomes one
        subset. Otherwise dataset_dir itself is treated as a flat image folder.
        The UI repeats value intentionally overrides folder-name repeats.
        """
        repeat_pat = re.compile(r"^\d+_.+")
        child_dirs = [p for p in sorted(dataset_dir.iterdir())
                      if p.is_dir() and repeat_pat.match(p.name)]
        concept_dirs = [p for p in child_dirs if cls._image_count(p) > 0]
        if concept_dirs:
            return [{"image_dir": p, "num_repeats": repeats}
                    for p in concept_dirs]
        return [{"image_dir": dataset_dir, "num_repeats": repeats}]

    def build(
        self,
        dataset_dir: str | Path,
        output_dir: str | Path,
        base_model: str | Path,
        lora_name: str = "my_lora",
        repeats: int = 10,
        config_overrides: Optional[dict] = None,
        sdxl: bool = False,
    ) -> Dict[str, Path]:
        """
        Generate dataset_config.toml and train_config.toml.

        Returns a dict with keys 'dataset_toml' and 'train_toml'.
        """
        dataset_dir = Path(dataset_dir)
        output_dir  = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        cfg = {**self.DEFAULTS, **(config_overrides or {})}

        subsets = self._detect_subsets(dataset_dir, repeats)
        self._sync_missing_captions(subsets, cfg["caption_extension"])

        # ── Count images to auto-calculate steps ──────────────────────────────
        img_count = sum(self._image_count(s["image_dir"]) for s in subsets)
        # target ≈ 1500 steps (community heuristic)
        effective_batch = cfg["batch_size"] * cfg["gradient_accumulation_steps"]
        if img_count > 0 and "max_train_epochs" not in (config_overrides or {}):
            steps_per_epoch = math.ceil(img_count * repeats / effective_batch)
            auto_epochs = max(1, math.ceil(1500 / steps_per_epoch))
            cfg["max_train_epochs"] = auto_epochs

        # ── Dataset TOML ──────────────────────────────────────────────────────
        ds_toml_lines = [
            "[general]",
            f'caption_extension = "{cfg["caption_extension"]}"',
            f'shuffle_caption = {_toml_val(cfg["shuffle_caption"])}',
            f'keep_tokens = {cfg["keep_tokens"]}',
            "",
            "[[datasets]]",
            f'resolution = {cfg["resolution"]}',
            f'batch_size = {cfg["batch_size"]}',
            f'enable_bucket = {_toml_val(cfg["enable_bucket"])}',
            f'min_bucket_reso = {cfg["min_bucket_reso"]}',
            f'max_bucket_reso = {cfg["max_bucket_reso"]}',
            f'bucket_reso_steps = {cfg["bucket_reso_steps"]}',
            "",
        ]
        for subset in subsets:
            ds_toml_lines += [
                "  [[datasets.subsets]]",
                f'  image_dir = "{subset["image_dir"].as_posix()}"',
                f'  num_repeats = {subset["num_repeats"]}',
                f'  flip_aug = {_toml_val(cfg["flip_aug"])}',
                "",
            ]
        ds_toml_path = output_dir / "dataset_config.toml"
        ds_toml_path.write_text("\n".join(ds_toml_lines), encoding="utf-8")

        # ── Training TOML ─────────────────────────────────────────────────────
        network_module = "networks.lora"
        model_lines = [
            "# Generated by LoRA-Harvester v5",
            f'# model_type = "{"sdxl" if sdxl else "sd15"}"',
            "",
            "[model_arguments]",
            f'pretrained_model_name_or_path = "{Path(base_model).as_posix()}"',
        ]
        if not sdxl:
            model_lines += [
                f'v2 = false',
                f'clip_skip = {cfg["clip_skip"]}',
            ]

        train_lines = model_lines + [
            "",
            "[dataset_arguments]",
            f'dataset_config = "{ds_toml_path.as_posix()}"',
            "",
            "[training_arguments]",
            f'output_dir = "{output_dir.as_posix()}"',
            f'output_name = "{lora_name}"',
            f'save_model_as = "safetensors"',
            f'save_every_n_epochs = {cfg["save_every_n_epochs"]}',
            f'max_train_epochs = {cfg["max_train_epochs"]}',
            f'gradient_accumulation_steps = {cfg["gradient_accumulation_steps"]}',
            f'mixed_precision = "{cfg["mixed_precision"]}"',
            f'gradient_checkpointing = true',
            f'xformers = true',
            f'sdpa = false',
            f'cache_latents = true',
            f'cache_latents_to_disk = {_toml_val(cfg["cache_latents_to_disk"])}',
            f'persistent_data_loader_workers = {_toml_val(cfg["persistent_data_loader_workers"])}',
            f'max_data_loader_n_workers = {cfg["max_data_loader_n_workers"]}',
        ]
        if sdxl:
            train_lines.append("cache_text_encoder_outputs = false")

        # Optional: noise_offset (skip if 0)
        if float(cfg.get("noise_offset", 0.0)) > 0.0:
            train_lines.append(f'noise_offset = {cfg["noise_offset"]}')

        # Optional: min_snr_gamma (skip if 0)
        if float(cfg.get("min_snr_gamma", 0.0)) > 0.0:
            train_lines.append(f'min_snr_gamma = {cfg["min_snr_gamma"]}')

        train_lines += [
            "",
            "[optimizer_arguments]",
            f'optimizer_type = "{cfg["optimizer"]}"',
            f'learning_rate = {cfg["learning_rate"]}',
            f'unet_lr = {cfg["unet_lr"]}',
        ]
        if sdxl:
            train_lines += [
                f'text_encoder_lr1 = {cfg["text_encoder_lr"]}',
                f'text_encoder_lr2 = {cfg["text_encoder_lr"]}',
            ]
        else:
            train_lines.append(f'text_encoder_lr = {cfg["text_encoder_lr"]}')
        train_lines += [
            f'lr_scheduler = "{cfg["lr_scheduler"]}"',
            f'lr_warmup_steps = {cfg["lr_warmup_steps"]}',
        ]

        if cfg["lr_scheduler"] == "cosine_with_restarts":
            train_lines.append(
                f'lr_scheduler_num_cycles = {cfg["lr_scheduler_num_cycles"]}'
            )

        train_lines += [
            "",
            "[network_arguments]",
            f'network_module = "{network_module}"',
            f'network_dim = {cfg["network_dim"]}',
            f'network_alpha = {cfg["network_alpha"]}',
        ]

        if float(cfg.get("network_dropout", 0.0)) > 0.0:
            train_lines.append(f'network_dropout = {cfg["network_dropout"]}')

        train_lines += [
            "",
            "[logging_arguments]",
            f'log_with = "tensorboard"',
            f'logging_dir = "{(output_dir / "logs").as_posix()}"',
            "",
        ]

        train_toml_path = output_dir / "train_config.toml"
        train_toml_path.write_text("\n".join(train_lines), encoding="utf-8")

        return {"dataset_toml": ds_toml_path, "train_toml": train_toml_path}

    @staticmethod
    def estimate_steps(
        image_count: int,
        repeats: int = 10,
        batch_size: int = 1,
        grad_accum: int = 4,
        target_steps: int = 1500,
    ) -> dict:
        """Return estimated training parameters given a dataset size."""
        effective_batch = batch_size * grad_accum
        steps_per_epoch = max(1, math.ceil(image_count * repeats / effective_batch))
        epochs = max(1, math.ceil(target_steps / steps_per_epoch))
        total_steps = steps_per_epoch * epochs
        return {
            "steps_per_epoch": steps_per_epoch,
            "recommended_epochs": epochs,
            "total_steps": total_steps,
        }
