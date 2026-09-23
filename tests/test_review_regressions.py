"""Regression expectations for the uploaded LoRA-Harvester snapshot.

Run from the project root:
  python -m pytest /path/to/test_review_regressions.py -q
Or set LORA_HARVESTER_ROOT to the extracted project directory.

This suite does not install packages, download weights, or run inference/training.
All image/caption writes are confined to pytest temporary directories.
The tests assert DESIRED behavior; failures demonstrate snapshot defects.
Requires Python 3.11+ (tomllib), pytest, numpy, opencv-python, torch.
"""
from __future__ import annotations

import ast
import os
from pathlib import Path
import sys
try:
    import tomllib
except ImportError:
    import tomli as tomllib
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(os.environ.get("LORA_HARVESTER_ROOT", os.getcwd())).resolve()
if not (ROOT / "src/core/unified_processor.py").exists():
    raise RuntimeError("Run from LoRA-Harvester root or set LORA_HARVESTER_ROOT.")
sys.path.insert(0, str(ROOT))

from src.core.unified_processor import UnifiedVideoProcessor
from src.core.advanced_captioner import AdvancedCaptioner, TagSettings
from src.core.anime_character_recognizer import AnimeCharacterRecognizer
from src.core.quality_analyzer import QualityAnalyzer
from src.core.kohya_exporter import KohyaExporter
from src.training.config_builder import TrainingConfigBuilder


def processor(tmp_path: Path, detector=None, quality=None):
    return UnifiedVideoProcessor(
        video_paths=[], output_dir=str(tmp_path),
        detector=detector or SimpleNamespace(), text_detector=None,
        cropper=SimpleNamespace(target_format="1:1"),
        batch_size=2, quality_analyzer=quality,
    )


def fake_pair(folder: Path, stem: str, caption: str | None):
    # File contents are not decoded in the file-routing/config tests.
    folder.mkdir(parents=True, exist_ok=True)
    image = folder / f"{stem}.png"
    image.write_bytes(b"synthetic-file-routing-fixture")
    if caption is not None:
        image.with_suffix(".txt").write_text(caption, encoding="utf-8")
    return image


# Positive controls: these should already pass.
def test_control_all_python_sources_parse():
    paths = sorted(ROOT.rglob("*.py"))
    assert len(paths) >= 50
    for path in paths:
        if "venv" not in path.parts and ".venv" not in path.parts:
            ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))


def test_control_export_preserves_existing_image_caption_pair(tmp_path):
    src = tmp_path / "source"
    dst = tmp_path / "destination"
    image = fake_pair(src / "person", "frame", "mychar, person")
    counts = KohyaExporter().export(src, dst)
    assert counts == {"person": 1}
    assert (dst / "10_person/frame.png").read_bytes() == image.read_bytes()
    assert (dst / "10_person/frame.txt").read_text() == "mychar, person"


def test_control_exact_negative_tag_is_removed():
    captioner = AdvancedCaptioner(
        enable_wd14=False, tag_settings=TagSettings(negative_tags=["watermark"])
    )
    assert captioner.process_tags([("watermark", .99, 0), ("solo", .95, 0)]) == ["solo"]


# B01: batch completion calls an undefined method.
def test_batch_completion_returns_stats_without_attribute_error(tmp_path):
    result = processor(tmp_path).process_all_videos()
    assert result["total_frames_saved"] == 0


# B02: distinct source videos must not overwrite each other's frames.
def test_distinct_same_named_videos_have_distinct_saved_paths(tmp_path):
    p = processor(tmp_path)
    p.current_video = "episode_a/clip.mp4"
    p.create_output_structure("clip")
    first = p.save_cropped_frame(np.zeros((8, 8, 3), np.uint8), "person", 30, .9)
    first_bytes = first.read_bytes()
    p.current_video = "episode_b/clip.mp4"
    p.create_output_structure("clip")
    second = p.save_cropped_frame(np.full((8, 8, 3), 255, np.uint8), "person", 30, .9)
    assert first != second and first.read_bytes() == first_bytes, (
        "Second source used the same output path and replaced the first image."
    )


# B03: report success only after a successful disk write.
def test_failed_image_save_returns_none_and_emits_no_success(tmp_path):
    p = processor(tmp_path)
    p.person_dir = tmp_path / "nonexistent-parent"
    callbacks = []
    p._frame_saved_callback = callbacks.append
    saved = p.save_cropped_frame(np.zeros((8, 8, 3), np.uint8), "person", 1, .9)
    assert saved is None and callbacks == [], (
        f"returned={saved}, exists={saved.exists() if saved else None}, callbacks={callbacks}"
    )


# B04: no filename-only caption transfer across unrelated concepts.
def test_caption_sync_does_not_copy_between_unrelated_concepts(tmp_path):
    alice = tmp_path / "10_alice"
    bob = tmp_path / "10_bob"
    fake_pair(alice, "frame_000030", "alice_character, red hair")
    bob_image = fake_pair(bob, "frame_000030", None)
    copied = TrainingConfigBuilder._sync_missing_captions(
        [{"image_dir": alice}, {"image_dir": bob}]
    )
    assert copied == 0 and not bob_image.with_suffix(".txt").exists(), (
        "Alice's caption was written onto Bob's unrelated same-named image."
    )


# B05: SmilingWolf's WD14 category 4 denotes character.
def test_wd14_character_category_can_be_disabled():
    captioner = AdvancedCaptioner(
        enable_wd14=False, tag_settings=TagSettings(keep_character_tags=False)
    )
    result = captioner.process_tags([("example_character", .99, 4)])
    assert result == [], f"Character tag leaked despite disabled option: {result}"


# B06: sorting must preserve the image + caption pair.
def test_anime_sort_move_preserves_caption_pair(tmp_path):
    source = fake_pair(tmp_path / "source", "frame", "mychar, person")
    dest = tmp_path / "sorted/person"
    AnimeCharacterRecognizer._safe_copy_move(source, dest, copy=False)
    assert (dest / "frame.png").exists()
    assert (dest / "frame.txt").exists() and not source.with_suffix(".txt").exists(), (
        "Image moved, caption remained at its old location."
    )


# B07: an OOM retry must not label its own pending frames as duplicates.
def test_oom_retry_preserves_frames_despite_duplicate_history(tmp_path):
    class OomOnceDetector:
        def __init__(self):
            self.calls = 0

        def detect_batch(self, frames):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("CUDA out of memory (synthetic test)")
            return [{} for _ in frames]

    detector = OomOnceDetector()
    quality = QualityAnalyzer(
        blur_threshold=0, brightness_range=(0, 255),
        noise_threshold=1e9, duplicate_threshold=.92,
    )
    p = processor(tmp_path, detector, quality)
    handled = []
    p._process_frame_with_detection = lambda f, n, d: handled.append(n)
    rng = np.random.default_rng(42)
    frames = [rng.integers(0, 256, (64, 64, 3), dtype=np.uint8) for _ in range(2)]
    p._safe_process_batch(frames, [30, 60], skip_text=False, use_quick_text=False)
    assert handled == [30, 60], (
        f"Handled={handled}; detector calls={detector.calls}; stats={p.stats}"
    )


# B08: dataset image_dir must point directly at images, not just ancestors.
def test_nested_harvester_layout_generates_leaf_subsets(tmp_path):
    source = tmp_path / "dataset"
    fake_pair(source / "clip_1x1_yolo/persons", "frame", "mychar, person")
    paths = TrainingConfigBuilder().build(
        source, tmp_path / "training", "model.safetensors"
    )
    config = tomllib.loads(paths["dataset_toml"].read_text())
    subsets = config["datasets"][0]["subsets"]
    for subset in subsets:
        folder = Path(subset["image_dir"])
        assert any(p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
                   for p in folder.iterdir() if p.is_file()), (
            f"image_dir has no directly contained images: {folder}"
        )


# B09: TOML mode needs explicit class_tokens when a caption is missing.
def test_exporter_serializes_requested_class_token(tmp_path):
    source = tmp_path / "source"
    dest = tmp_path / "export"
    fake_pair(source / "persons", "frame", None)
    KohyaExporter().export(
        source, dest,
        concept_overrides={"persons": {"repeats": 5, "class_token": "mychar person"}}
    )
    config = tomllib.loads((dest / "dataset_config.toml").read_text())
    subset = config["datasets"][0]["subsets"][0]
    assert subset.get("class_tokens") == "mychar person", subset


# B10: the README-documented CLI must honor --no-resume.
def test_documented_cli_forwards_resume_flag():
    tree = ast.parse((ROOT / "scripts/cli.py").read_text(encoding="utf-8"))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == "process_all_videos"]
    assert len(calls) == 1
    assert any(k.arg == "resume" and isinstance(k.value, ast.Name)
               and k.value.id == "use_resume" for k in calls[0].keywords), (
        "scripts/cli.py computes use_resume but does not forward it."
    )


# B11: advertised MCP self-test must refer to a file in the snapshot.
def test_mcp_self_test_runner_exists_in_archive():
    assert (ROOT / "run_tests.py").is_file(), (
        "mcp_server.py run_self_test invokes missing run_tests.py."
    )


# B12: a documented *substring* negative pattern must match mid-tag.
def test_contains_wildcard_matches_middle_of_tag():
    captioner = AdvancedCaptioner(
        enable_wd14=False, tag_settings=TagSettings(negative_tags=["*hair*"])
    )
    assert captioner.process_tags([("long_hair_ornament", .99, 0)]) == []
