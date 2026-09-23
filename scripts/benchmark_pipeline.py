"""Read-only CPU bottleneck comparison; no model download or GPU inference.

Old baselines reproduce the sfw-batch1 algorithms. Equal selected video pixels,
noise scores, discovered image/caption pairs and reserved names are required
before a speed ratio is reported. Synthetic timings are NOT whole-app gains.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import tempfile
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from PIL import Image
from src.core.video_sampling import SelectiveVideoReader
from src.core.quality_analyzer import QualityAnalyzer
from src.core.dataset_scanner import scan_dataset, IMAGE_EXTS, _SKIP_DIRS, _SKIP_FILES, FramePair
from src.core.dataset_files import unique_image_path, IMAGE_EXTENSIONS


def _legacy_noise(gray):
    h, w = gray.shape[:2]
    stds = []
    for y in range(0, h - 16, 16):
        for x in range(0, w - 16, 16):
            stds.append(float(np.std(gray[y:y+16, x:x+16])))
    if not stds:
        return 0.0
    stds.sort()
    return float(np.mean(stds[:max(1, len(stds)//5)]))


def _legacy_scan(root):
    root = Path(root).resolve()
    candidates = [p for p in root.rglob('*') if p.is_file()]
    pairs = []
    for p in candidates:
        if p.name in _SKIP_FILES:
            continue
        if any(part in _SKIP_DIRS for part in p.relative_to(root).parts):
            continue
        if p.is_symlink() or p.suffix.lower() not in IMAGE_EXTS:
            continue
        txt = p.with_suffix('.txt')
        pairs.append(FramePair(p, txt if txt.exists() else None, p.parent.name))
    return pairs


def _legacy_name(path):
    taken = set()
    if path.parent.is_dir():
        taken = {p.stem.casefold() for p in path.parent.iterdir()
                 if p.suffix.lower() in IMAGE_EXTENSIONS | {'.txt', '.json'}}
    candidate, number = path, 1
    while candidate.stem.casefold() in taken or candidate.exists():
        candidate = path.with_name(f'{path.stem}_{number}{path.suffix}')
        number += 1
    return candidate


def _pair_signature(pairs):
    return sorted((str(p.image), str(p.caption), p.concept) for p in pairs)


def _video_pass(path, *, selective, interval, limit):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise ValueError(f'Video could not be opened: {path}')
    digest = hashlib.sha256()
    reader = SelectiveVideoReader(cap) if selective else None
    advanced, selected = 0, 0
    start = perf_counter()
    try:
        for index in range(limit):
            pick = (index + 1) % interval == 0
            ok, frame = reader.read(pick) if selective else cap.read()
            if not ok:
                break
            advanced += 1
            if pick:
                selected += 1
                digest.update((index + 1).to_bytes(8, 'little'))
                digest.update(frame.tobytes())
        seconds = perf_counter() - start
        signature = {'frames_advanced': advanced, 'frames_selected': selected,
                     'selected_pixels_sha256': digest.hexdigest()}
        return signature, seconds
    finally:
        cap.release()


def _timed(function):
    start = perf_counter()
    result = function()
    return result, perf_counter() - start


def compare(old, new, repeats):
    # Both paths are warmed first. Alternate order to reduce order bias.
    initial_old, _ = old()
    initial_new, _ = new()
    equal = initial_old == initial_new
    before, after = [], []
    for iteration in range(repeats):
        order = (('old', old), ('new', new)) if iteration % 2 == 0 else (('new', new), ('old', old))
        for label, function in order:
            signature, seconds = function()
            equal = equal and signature == initial_old
            (before if label == 'old' else after).append(seconds)
    old_median, new_median = statistics.median(before), statistics.median(after)
    return {'equivalent_results': bool(equal), 'old_seconds': before, 'new_seconds': after,
            'old_median_seconds': old_median, 'new_median_seconds': new_median,
            'speed_ratio': old_median/new_median if equal and new_median > 0 else None,
            'seconds_reduction_percent': 100*(1-new_median/old_median) if equal and old_median > 0 else None}


def _fixture_video(path, count):
    size = (960, 540)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 30, size)
    if not writer.isOpened():
        raise RuntimeError('MJPG encoding unavailable; supply --video with an existing local video.')
    rng = np.random.default_rng(42)
    texture = rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    try:
        for n in range(count):
            writer.write(np.roll(texture, n * 3, axis=1))
    finally:
        writer.release()


def _fixture_dataset(root):
    root.mkdir()
    buffer = io.BytesIO()
    Image.new('RGB', (16, 16), (32, 80, 140)).save(buffer, format='PNG')
    payload = buffer.getvalue()
    for n in range(300):
        image = root / f'{n:04d}.png'
        image.write_bytes(payload)
        image.with_suffix('.txt').write_text('master, blue skirt', encoding='utf-8')
    for n in range(1500):
        (root / f'other_{n:04d}.json').write_text('{}', encoding='utf-8')
    excluded = root / '.lh-clothing' / 'history'
    excluded.mkdir(parents=True)
    for n in range(6000):
        (excluded / f'record_{n:05d}.json').write_text('{}', encoding='utf-8')
    return root


def run_benchmark(*, video=None, repeats=5, interval=30, limit=120):
    for name, value in [('repeats', repeats), ('interval', interval), ('limit', limit)]:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f'{name} must be a positive integer.')
    if repeats > 20 or limit > 100000:
        raise ValueError('Use repeats <= 20 and limit <= 100000.')
    if interval > limit:
        raise ValueError('interval must be <= limit so at least one frame can be compared.')
    video = Path(video).resolve(strict=True) if video is not None else None
    if video is not None and not video.is_file():
        raise ValueError('Video must be a local file.')
    result = {'scope': 'CPU kernels, video read/materialisation and temporary filesystem scans only',
              'not_measured': ['Windows UI', 'NVIDIA/CUDA', 'models', 'whole pipeline throughput', 'disk durability write cost'],
              'baseline': '2026.09.22-sfw-batch1',
              'environment': {'python': sys.version.split()[0], 'platform': platform.platform(),
                              'numpy': np.__version__, 'opencv': cv2.__version__, 'cpu': platform.processor()},
              'repeats': repeats, 'cache': 'warm-up of both paths; alternating order; OS cache not cleared'}
    with tempfile.TemporaryDirectory(prefix='lh-benchmark-') as temporary:
        work = Path(temporary)
        if video is None:
            video = work / 'synthetic.avi'
            _fixture_video(video, min(limit, 120))
            result['video_source'] = 'synthetic MJPG 960x540 @ 30 FPS'
        else:
            result['video_source'] = str(video)
        before = video.stat()
        result['video_read'] = compare(
            lambda: _video_pass(video, selective=False, interval=interval, limit=limit),
            lambda: _video_pass(video, selective=True, interval=interval, limit=limit), repeats)
        signature, _ = _video_pass(video, selective=True, interval=interval, limit=limit)
        result['video_read'].update(signature)
        result['video_read']['sample_interval'] = interval
        if not signature['frames_selected']:
            raise ValueError('No sampled frames decoded; use a smaller --interval or another video.')
        after = video.stat()
        result['source_video_unchanged_stat'] = (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
        # Independent SHA-256 compares selected decoded pixels, not model decisions.
        if not result['source_video_unchanged_stat']:
            result['video_read']['equivalent_results'] = False
            result['video_read']['speed_ratio'] = None
            result['video_read']['seconds_reduction_percent'] = None
        rng = np.random.default_rng(77)
        frames = [rng.integers(0, 256, (512, 512), dtype=np.uint8) for _ in range(8)]
        analyzer = QualityAnalyzer()
        result['noise_kernel'] = compare(
            lambda: _timed(lambda: [_legacy_noise(gray) for gray in frames]),
            lambda: _timed(lambda: [analyzer._calculate_noise_score(gray) for gray in frames]), repeats)
        result['noise_kernel']['images_per_pass'] = 8
        result['noise_kernel']['size'] = [512, 512]
        dataset = _fixture_dataset(work / 'dataset')
        result['dataset_scan'] = compare(
            lambda: _timed(lambda: _pair_signature(_legacy_scan(dataset))),
            lambda: _timed(lambda: _pair_signature(scan_dataset(dataset))), repeats)
        result['dataset_scan']['fixture'] = {'images': 300, 'captions': 300, 'other_files': 1500, 'excluded_history_files': 6000}
        names = [dataset / f'new_{i}.png' for i in range(8)]
        result['output_naming'] = compare(
            lambda: _timed(lambda: [str(_legacy_name(name)) for name in names]),
            lambda: _timed(lambda: [str(unique_image_path(name)) for name in names]), repeats)
        result['output_naming']['note'] = 'Live collision checks retained; still O(number of folder entries) per proposed name.'
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, help='Optional read-only real video; otherwise use a temporary synthetic MJPG.')
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--interval', type=int, default=30, help='Same 1-based source-frame interval on both paths.')
    parser.add_argument('--limit', type=int, default=120, help='Maximum source frames per pass, not images saved.')
    parser.add_argument('--output', type=Path, help='Create a NEW JSON report; never replace an existing file.')
    args = parser.parse_args(argv)
    try:
        if args.output is not None and (args.output.exists() or args.output.is_symlink()):
            raise FileExistsError(f'Report already exists: {args.output}')
        result = run_benchmark(video=args.video, repeats=args.repeats, interval=args.interval, limit=args.limit)
        text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
        if args.output is not None:
            # Parent directory must already exist; exclusive mode protects races.
            with args.output.open('x', encoding='utf-8') as handle:
                handle.write(text + '\n')
        print(text)
        return 0 if all(result[k]['equivalent_results'] for k in ('video_read', 'noise_kernel', 'dataset_scan', 'output_naming')) else 2
    except (OSError, ValueError, RuntimeError) as exc:
        print(f'Benchmark error: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
