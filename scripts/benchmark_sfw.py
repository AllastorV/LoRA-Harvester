#!/usr/bin/env python3
"""Read-only local SFW/NSFW throughput comparison; no dataset sorting/writes.

Uses the same Falconsai pipeline for sequential and batched modes. By default
weights must already be cached; downloading requires --allow-model-download.
Model load and image decoding are reported separately from warm inference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def benchmark(detector, frames, repeats: int = 3) -> dict:
    """Alternate execution order, compare labels, report rather than assume speed."""
    if len(frames) < 2 or not 1 <= repeats <= 10:
        raise ValueError('Use at least 2 images and 1..10 repeats.')
    if not detector.is_available():
        raise RuntimeError('SFW/NSFW model unavailable: ' + detector.last_error)
    initial_device = detector.actual_device
    # Warm both code paths; callers get a separate cold-load timing metric.
    before = detector.metrics()
    detector.classify(frames[0])
    detector.classify_batch(frames[:detector.effective_batch_size])
    after = detector.metrics()
    if after['failed_images'] > before['failed_images']:
        raise RuntimeError('Warmup failed. Benchmark speed would not be meaningful.')
    runs, agreement = [], []
    for repeat in range(repeats):
        results = {}
        order = ('single', 'batch') if repeat % 2 == 0 else ('batch', 'single')
        for mode in order:
            before = detector.metrics()
            tic = time.perf_counter()
            if mode == 'single':
                outputs = [detector.classify(frame) for frame in frames]
            else:
                outputs = detector.classify_batch(frames)
            elapsed = time.perf_counter() - tic
            after = detector.metrics()
            runs.append({
                'repeat': repeat + 1, 'mode': mode, 'seconds': elapsed,
                'images_per_second': len(frames)/elapsed if elapsed else None,
                'device': detector.actual_device,
                'effective_batch_size': detector.effective_batch_size,
                'pipeline_calls': after['pipeline_calls']-before['pipeline_calls'],
                'oom_retries': after['oom_retries']-before['oom_retries'],
                'failed_images': after['failed_images']-before['failed_images'],
            })
            results[mode] = outputs
        mismatches = [index for index, (a,b) in enumerate(zip(results['single'],results['batch'])) if a[0] != b[0]]
        agreement.append({
            'repeat': repeat+1, 'label_mismatch_indices': mismatches,
            'max_returned_confidence_difference': max(abs(a[1]-b[1]) for a,b in zip(results['single'],results['batch'])),
        })
    medians = {mode: statistics.median(run['seconds'] for run in runs if run['mode']==mode)
               for mode in ('single', 'batch')}
    valid = (all(run['failed_images']==0 for run in runs)
             and all(run['device']==initial_device for run in runs)
             and not any(item['label_mismatch_indices'] for item in agreement))
    return {
        'image_count': len(frames), 'repeats': repeats, 'runs': runs,
        'agreement': agreement, 'median_seconds': medians,
        'comparison_valid': valid,
        'measured_speed_ratio': medians['single']/medians['batch'] if valid and medians['batch'] else None,
        'runtime': detector.metrics(),
        'scope': 'Warm classification including BGR/PIL conversion. Excludes decode, video detection, captioning, saving, and model loading. Not an accuracy evaluation.',
    }


def load_images(folder: Path, limit: int, recursive: bool = True, memory_mb: int = 256):
    """Bound decoded memory and retain file fingerprints without any writes."""
    import cv2
    import numpy as np
    from PIL import Image
    from src.core.dataset_files import IMAGE_EXTENSIONS
    if not folder.is_dir():
        raise ValueError('The dataset folder does not exist.')
    if not 2 <= limit <= 256:
        raise ValueError('limit must be in 2..256.')
    budget = memory_mb * 1024 * 1024
    used = 0
    frames, sources, warnings = [], [], []
    def walk_error(error):
        raise error
    for directory, dirs, filenames in os.walk(folder, onerror=walk_error, followlinks=False):
        dirs[:] = sorted(name for name in dirs if recursive and not name.startswith('.')
                         and not (Path(directory)/name).is_symlink())
        for name in sorted(filenames):
            path = Path(directory)/name
            if path.suffix.lower() not in IMAGE_EXTENSIONS or path.is_symlink():
                continue
            try:
                if path.stat().st_size > budget:
                    raise ValueError('Encoded image exceeds benchmark memory budget.')
                with Image.open(path) as header:
                    if header.width*header.height*3 + used > budget:
                        raise ValueError('Decoded image would exceed benchmark memory budget.')
                data = path.read_bytes()
                frame = cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
                if frame is None:
                    raise ValueError('Image decode failed.')
                frames.append(frame)
                used += frame.nbytes
                sources.append({'path':str(path.relative_to(folder)),
                                'sha256':hashlib.sha256(data).hexdigest(),
                                'shape':list(frame.shape)})
                if len(frames) >= limit:
                    return frames,sources,warnings
            except (OSError,ValueError,Image.DecompressionBombError) as exc:
                warnings.append(f'{path.name}: {exc}')
    return frames,sources,warnings


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path)
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--limit',type=int,default=64)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--device',choices=['auto','cuda','cpu'],default='auto')
    parser.add_argument('--no-recursive',action='store_true')
    parser.add_argument('--allow-model-download',action='store_true')
    parser.add_argument('--output',type=Path,help='Optional NEW JSON file; never overwrites an existing file.')
    args = parser.parse_args(argv)
    detector = None
    try:
        if not 1 <= args.batch_size <= 32 or not 1 <= args.repeats <= 10 or not 2 <= args.limit <= 256:
            raise ValueError('batch-size: 1..32, repeats: 1..10, limit: 2..256.')
        if args.output and (args.output.exists() or args.output.is_symlink()):
            raise ValueError('Output already exists; choose a new report path.')
        if args.output and not args.output.parent.is_dir():
            raise ValueError('Output parent folder does not exist.')
        tic=time.perf_counter()
        frames,sources,warnings=load_images(args.folder,args.limit,not args.no_recursive)
        decode_seconds=time.perf_counter()-tic
        if len(frames)<2:
            raise ValueError('At least two readable images are required.')
        # Must be set before importing Transformers or Hugging Face Hub.
        if not args.allow_model_download:
            os.environ['HF_HUB_OFFLINE']='1'
            os.environ['TRANSFORMERS_OFFLINE']='1'
        from src.core.nsfw_detector import NsfwDetector
        detector=NsfwDetector(device=args.device,batch_size=args.batch_size)
        report=benchmark(detector,frames,args.repeats)
        import torch
        report.update(sources=sources,warnings=warnings,decode_seconds=decode_seconds,
                      environment={'platform':platform.platform(),'python':platform.python_version(),
                                   'torch':torch.__version__,'cuda_available':torch.cuda.is_available()})
        print(f"Device: {detector.actual_device}; effective batch: {detector.effective_batch_size}")
        print(f"Median single: {report['median_seconds']['single']:.4f}s; batch: {report['median_seconds']['batch']:.4f}s")
        print('Classification-only measured ratio:',report['measured_speed_ratio'])
        print('Label mismatches per repeat:',[len(item['label_mismatch_indices']) for item in report['agreement']])
        if args.output:
            with args.output.open('x',encoding='utf-8') as handle:
                json.dump(report,handle,ensure_ascii=False,indent=2)
                handle.write('\n')
        return 0 if report['comparison_valid'] else 2
    except (OSError,ValueError,RuntimeError,ImportError) as exc:
        print(f'[ERROR] {exc}',file=sys.stderr)
        return 1
    finally:
        if detector is not None:
            detector.cleanup()


if __name__=='__main__':
    raise SystemExit(main())
