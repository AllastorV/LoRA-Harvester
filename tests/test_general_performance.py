"""Performance regressions: exact results, bounded notifications, safe UI contracts.

These do not require a model, CUDA or a running Qt event loop. The separate Qt
smoke tests are skipped explicitly when Qt is unavailable.
"""
from __future__ import annotations

import ast
import importlib
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np
import pytest

from src.core.dataset_files import unique_image_path, transfer_image_pair
from src.core.dataset_scanner import scan_dataset, IMAGE_EXTS, _SKIP_DIRS
from src.core.progress_throttle import ProgressThrottle
from src.core.quality_analyzer import QualityAnalyzer
from src.core.video_sampling import SelectiveVideoReader
from src.core import text_detector

ROOT = Path(__file__).resolve().parents[1]


def legacy_noise(gray):
    values = []
    h, w = gray.shape
    for y in range(0, h - 16, 16):
        for x in range(0, w - 16, 16):
            values.append(float(np.std(gray[y:y+16, x:x+16])))
    if not values:
        return 0.0
    values.sort()
    return float(np.mean(values[:max(1, len(values)//5)]))


@pytest.mark.parametrize('shape', [(1,1), (16,16), (17,17), (32,32), (33,49), (128,193), (512,512)])
@pytest.mark.parametrize('kind', ['constant', 'random', 'strided'])
def test_noise_exactly_matches_original_patch_math(shape, kind):
    rng = np.random.default_rng(42)
    gray = (np.full(shape, 127, np.uint8) if kind == 'constant' else
            rng.integers(0, 256, (shape[0], shape[1] * (2 if kind == 'strided' else 1)), dtype=np.uint8))
    if kind == 'strided':
        gray = gray[:, ::2]
    before = gray.copy()
    assert QualityAnalyzer()._calculate_noise_score(gray) == legacy_noise(gray)
    np.testing.assert_array_equal(gray, before)


@pytest.mark.parametrize('threshold', [0, 12., 75., 255.])
def test_quality_acceptance_and_all_scores_remain_identical(threshold):
    old, new = QualityAnalyzer(noise_threshold=threshold), QualityAnalyzer(noise_threshold=threshold)
    old._calculate_noise_score = legacy_noise
    rng = np.random.default_rng(23)
    frames = [rng.integers(0, 256, (258, 410, 3), dtype=np.uint8),
              np.full((200, 210, 3), 128, np.uint8)]
    frames += [frames[0].copy()]
    for frame in frames:
        assert old.analyze_frame(frame) == new.analyze_frame(frame)
    assert old.stats == new.stats


class Capture:
    def __init__(self, frames=6):
        self.count, self.pos, self.calls = frames, 0, []
    def grab(self):
        self.calls.append('grab')
        if self.pos == self.count:
            return False
        self.pos += 1
        return True
    def retrieve(self):
        self.calls.append('retrieve')
        return True, np.full((3, 3, 3), self.pos, np.uint8)
    def read(self):
        self.calls.append('read')
        if self.pos == self.count:
            return False, None
        self.pos += 1
        return True, np.full((3, 3, 3), self.pos, np.uint8)


def test_selective_reader_does_not_retrieve_discarded_pixels():
    cap = Capture()
    reader = SelectiveVideoReader(cap)
    selected = []
    for i in range(6):
        ok, frame = reader.read((i+1) % 3 == 0)
        assert ok
        if (i+1) % 3:
            assert frame is None
        else:
            selected.append(int(frame[0, 0, 0]))
    assert selected == [3, 6]
    assert cap.calls.count('grab') == 6
    assert cap.calls.count('retrieve') == 2
    assert 'read' not in cap.calls
    assert reader.snapshot()['discarded'] == 4
    assert reader.snapshot()['retrieved'] == 2
    assert reader.read() == (False, None)


def test_read_only_plugin_adapter_remains_supported():
    cap = Capture()
    adapter = SimpleNamespace(read=cap.read)
    reader = SelectiveVideoReader(adapter)
    assert reader.read(False) == (True, None)
    ok, frame = reader.read()
    assert ok and frame[0, 0, 0] == 2
    assert reader.snapshot()['mode'] == 'read'
    assert cap.calls == ['read', 'read']


def test_reader_compatibility_switch_and_failed_retrieve():
    cap = Capture()
    assert SelectiveVideoReader(cap, enabled=False).read(False) == (True, None)
    assert cap.calls == ['read']
    cap.retrieve = lambda: (False, None)
    reader = SelectiveVideoReader(cap)
    assert reader.read(True) == (False, None)
    assert reader.snapshot()['retrieved'] == 0


@pytest.fixture
def clip(tmp_path):
    path = tmp_path / 'actual.avi'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 10, (96,64))
    assert writer.isOpened()
    rng = np.random.default_rng(72)
    for i in range(23):
        frame = rng.integers(0, 256, (64,96,3), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


@pytest.mark.parametrize('interval', [1, 2, 5, 50])
def test_real_video_pixels_and_sample_numbers_identical(clip, interval):
    first, second = cv2.VideoCapture(str(clip)), cv2.VideoCapture(str(clip))
    reader = SelectiveVideoReader(second)
    index = 0
    try:
        while True:
            ret, frame = first.read()
            selected = (index + 1) % interval == 0
            ret2, frame2 = reader.read(selected)
            assert ret == ret2
            if not ret:
                break
            if selected:
                np.testing.assert_array_equal(frame, frame2)
            else:
                assert frame2 is None
            index += 1
        assert index == 23
        assert reader.retrieved == 23 // interval
    finally:
        first.release()
        second.release()


@pytest.mark.parametrize('turbo', [False, True])
@pytest.mark.parametrize('interval', [1, 3, 7, 50])
def test_processing_preserves_trim_and_sampling_in_both_modes(clip, tmp_path, turbo, interval):
    from src.core.unified_processor import UnifiedVideoProcessor
    class Detector:
        def detect(self, frame):
            return []
        def detect_batch(self, frames):
            return [[] for f in frames]
    output = tmp_path / ('out-' + str(turbo))
    p = UnifiedVideoProcessor(str(clip), str(output), Detector(), None, SimpleNamespace(target_format="1:1"),
                              use_turbo=turbo, batch_size=4)
    seen = []
    p._process_single_frame = lambda frame, n, *args: seen.append((n, frame.copy()))
    p._safe_process_batch = lambda frames, numbers, *args: seen.extend(
        (n, f.copy()) for n, f in zip(numbers, frames))
    stats = p.process_single_video(str(clip), frame_interval=interval,
                                   skip_text=False, start_skip_seconds=.3,
                                   end_skip_seconds=.2)
    expected = [i for i in range(4,22) if i % interval == 0]
    assert [n for n, _ in seen] == expected
    assert stats['video_read']['retrieved'] == len(expected)
    assert stats['video_read']['advanced'] == 21
    assert stats['stage_seconds']['video_read'] >= 0
    assert stats['other_and_wait_seconds'] >= 0
    original = cv2.VideoCapture(str(clip))
    originals = {}
    for n in range(1, 24):
        ok, frame = original.read()
        assert ok
        originals[n] = frame
    original.release()
    for n, frame in seen:
        np.testing.assert_array_equal(frame, originals[n])


def test_stage_timing_does_not_swallow_exceptions():
    from src.core.unified_processor import UnifiedVideoProcessor
    p = object.__new__(UnifiedVideoProcessor)
    p._stage_seconds = {}
    def fail():
        raise RuntimeError('real failure')
    with pytest.raises(RuntimeError, match='real failure'):
        p._timed('quality', fail)
    assert p._stage_seconds['quality'] >= 0
    assert p._timed('quality', lambda: 3) == 3


def test_quick_text_mode_never_loads_ocr(monkeypatch):
    load = Mock(side_effect=AssertionError('unused OCR must not load'))
    monkeypatch.setattr(text_detector, '_easyocr', SimpleNamespace(Reader=load))
    detector = text_detector.SubtitleDetector()
    detector.quick_text_check(np.zeros((128, 192, 3), np.uint8))
    assert detector.reader is None
    load.assert_not_called()


def test_full_ocr_initializes_once_and_keeps_decisions(monkeypatch):
    reader = SimpleNamespace(readtext=Mock(return_value=[]))
    load = Mock(return_value=reader)
    monkeypatch.setattr(text_detector, '_easyocr', SimpleNamespace(Reader=load))
    detector = text_detector.SubtitleDetector()
    frame = np.zeros((128, 192, 3), np.uint8)
    assert detector.has_text(frame, check_subtitle_region=False) == (False, 0.0)
    assert detector.has_text(frame, check_subtitle_region=False) == (False, 0.0)
    assert load.call_count == 1
    assert reader.readtext.call_count == 2


def test_failed_ocr_load_only_attempted_once(monkeypatch):
    load = Mock(side_effect=RuntimeError('offline'))
    monkeypatch.setattr(text_detector, '_easyocr', SimpleNamespace(Reader=load))
    detector = text_detector.SubtitleDetector()
    detector.quick_text_check = Mock(return_value=True)
    frame = np.zeros((64, 64, 3), np.uint8)
    assert detector.has_text(frame) == (True, 0.0)
    assert detector.has_text(frame) == (True, 0.0)
    load.assert_called_once()


def test_simultaneous_lazy_ocr_requests_share_one_reader(monkeypatch):
    reader = object()
    load = Mock(return_value=reader)
    monkeypatch.setattr(text_detector, '_easyocr', SimpleNamespace(Reader=load))
    detector = text_detector.SubtitleDetector()
    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(lambda _: detector.initialize_ocr(), range(12)))
    assert all(value is reader for value in results)
    load.assert_called_once()


def test_dataset_scan_prunes_before_entering_excluded_trees(tmp_path, monkeypatch):
    import os
    from src.core import dataset_scanner
    (tmp_path / 'regular').mkdir()
    (tmp_path / 'regular' / 'a.PNG').write_bytes(b'x')
    (tmp_path / 'regular' / 'a.txt').write_text('master, blue skirt')
    (tmp_path / 'b.jpg').write_bytes(b'y')
    for name in _SKIP_DIRS:
        hidden = tmp_path / name / 'large-tree'
        hidden.mkdir(parents=True)
        (hidden / 'hidden.png').write_bytes(b'z')
    seen = []
    real = os.scandir
    def spy(path):
        seen.append(str(path))
        return real(path)
    monkeypatch.setattr(dataset_scanner.os, 'scandir', spy)
    pairs = scan_dataset(tmp_path)
    assert {p.image.name for p in pairs} == {'a.PNG', 'b.jpg'}
    assert next(p for p in pairs if p.image.name == 'a.PNG').caption.name == 'a.txt'
    assert not any(set(Path(path).relative_to(tmp_path).parts) & _SKIP_DIRS for path in seen)
    assert [p.image.name for p in scan_dataset(tmp_path, recursive=False)] == ['b.jpg']


def test_dataset_scan_does_not_follow_symlink_images_or_folders(tmp_path):
    real = tmp_path / 'real'
    real.mkdir()
    (real / 'frame.png').write_bytes(b'x')
    try:
        (tmp_path / 'alias').symlink_to(real, target_is_directory=True)
        (tmp_path / 'alias.png').symlink_to(real / 'frame.png')
    except OSError:
        pytest.skip('Symlink creation not permitted on this filesystem')
    assert [p.image for p in scan_dataset(tmp_path)] == [real / 'frame.png']


def test_unique_names_preserve_sidecars_casefold_and_external_changes(tmp_path):
    (tmp_path / 'IMAGE.TXT').write_bytes(b'x')
    (tmp_path / 'image_1.jpg').write_bytes(b'x')
    target = tmp_path / 'image.png'
    assert unique_image_path(target).name == 'image_2.png'
    (tmp_path / 'image_2.json').write_bytes(b'x')
    assert unique_image_path(target).name == 'image_3.png'
    assert not target.exists()


def test_transfer_without_sidecar_does_not_scan_source_siblings(tmp_path, monkeypatch):
    from src.core import dataset_files
    import os
    source = tmp_path / 'src'
    source.mkdir()
    image = source / 'a.png'
    image.write_bytes(b'original')
    scanned = []
    real = os.scandir
    def spy(path):
        scanned.append(Path(path))
        return real(path)
    monkeypatch.setattr(dataset_files.os, 'scandir', spy)
    result = transfer_image_pair(image, tmp_path / 'dst')
    assert result.read_bytes() == image.read_bytes() == b'original'
    assert source not in scanned


def test_transfer_with_conflicting_sidecar_still_refuses(tmp_path):
    source = tmp_path / 'src'
    source.mkdir()
    image = source / 'a.png'
    image.write_bytes(b'first')
    (source / 'a.jpg').write_bytes(b'second')
    (source / 'a.txt').write_bytes(b'master')
    with pytest.raises(ValueError, match='Ambiguous'):
        transfer_image_pair(image, tmp_path / 'dst')
    assert image.read_bytes() == b'first'
    assert (source / 'a.txt').read_bytes() == b'master'


def test_progress_coalesces_burst_but_never_completion():
    out = []
    throttle = ProgressThrottle(lambda *x: out.append(x), clock=lambda: 1.)
    for n in range(1001):
        throttle(n, 1000, str(n))
    assert out == [(0, 1000, '0'), (1000, 1000, '1000')]
    throttle.flush()
    assert len(out) == 2


def test_progress_flush_preserves_latest_partial_result_and_phase_change():
    out, now = [], [0.]
    throttle = ProgressThrottle(lambda *x: out.append(x), clock=lambda: now[0])
    throttle(1, 100, 'first')
    throttle(2, 100, 'hidden')
    throttle(3, 100, 'latest')
    throttle.flush()
    assert out[-1] == (3, 100, 'latest')
    now[0] = .2
    throttle(4, 100, 'timed')
    throttle(1, 50, 'new phase')
    assert [x[2] for x in out] == ['first', 'latest', 'timed', 'new phase']
    with pytest.raises(ValueError):
        ProgressThrottle(print, interval=-1)


def method(file, cls, name, environment=None):
    tree = ast.parse((ROOT / file).read_text(encoding='utf-8'))
    node = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == cls)
    function = next(x for x in node.body if isinstance(x, ast.FunctionDef) and x.name == name)
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), function], type_ignores=[])
    ast.fix_missing_locations(module)
    env = {} if environment is None else dict(environment)
    exec(compile(module, str(file), 'exec'), env)
    return env[name]


def test_review_thumbnail_decoder_never_constructs_qpixmap_in_worker():
    src = ast.unparse(ast.parse((ROOT / 'src/ui/review_grid_page.py').read_text()))
    worker = next(n for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name == '_ThumbnailLoader')
    assert not any(isinstance(n, ast.Name) and n.id == 'QPixmap' for n in ast.walk(worker))
    assert 'setScaledSize' in ast.unparse(worker)


def test_review_reloads_have_no_gui_wait_or_duplicate_concept_scan():
    tree = ast.parse((ROOT / 'src/ui/review_grid_page.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ReviewGridPage')
    for name in ['_load_folder', '_compute_quick_stats', '_on_loading_done']:
        code = ast.unparse(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name))
        assert '.wait(' not in code
        assert 'detect_concepts(' not in code
        assert '.read_text(' not in code
    # Check actual __init__, not comments/class constants.
    tagcls = next(n for n in ast.parse((ROOT / 'src/ui/caption_studio_page.py').read_text()).body
                  if isinstance(n, ast.ClassDef) and n.name == 'TagCompleterTextEdit')
    init = next(n for n in tagcls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    assert '_ensure_tags_loaded' not in ast.unparse(init)


def test_review_removed_row_never_receives_a_different_files_thumbnail():
    delivered = []
    first, second = Path('first.png'), Path('second.png')
    worker = SimpleNamespace(_pairs=[SimpleNamespace(image=first), SimpleNamespace(image=second)])
    item = SimpleNamespace(setIcon=lambda value: delivered.append(value))
    page = SimpleNamespace(_loader=worker, sender=lambda: worker,
        _items_by_path={str(second): item},
        _progress=SimpleNamespace(setValue=lambda v: None))
    fn = method('src/ui/review_grid_page.py', 'ReviewGridPage', '_on_thumb_ready',
                {'QIcon': lambda x: x, 'QPixmap': SimpleNamespace(fromImage=lambda x: x)})
    fn(page, 0, 'wrong')
    fn(page, 1, 'correct')
    assert delivered == ['correct']
    page.sender = lambda: object()
    fn(page, 1, 'stale folder')
    assert delivered == ['correct']


def test_worker_flushes_pending_progress_before_failure():
    out = []
    def work(cancel, progress):
        progress(1, 10, 'first')
        progress(2, 10, 'partial')
        raise ValueError('expected')
    worker = SimpleNamespace(work=work, cancel=threading.Event(),
        progress=SimpleNamespace(emit=lambda *x: out.append(('progress', x))),
        completed=SimpleNamespace(emit=lambda x: out.append(('done', x))),
        failed=SimpleNamespace(emit=lambda x: out.append(('error', x))))
    method('src/ui/studio_tasks.py', 'StudioWorker', 'run')(worker)
    assert out[-2:] == [('progress', (2, 10, 'partial')), ('error', 'expected')]


def test_concurrent_ocr_caller_waits_for_initialization_in_progress(monkeypatch):
    started, release = threading.Event(), threading.Event()
    reader = object()
    def create(*args, **kwargs):
        started.set()
        assert release.wait(2)
        return reader
    monkeypatch.setattr(text_detector, '_easyocr', SimpleNamespace(Reader=create))
    detector = text_detector.SubtitleDetector()
    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(detector.initialize_ocr)
        assert started.wait(2)
        second = pool.submit(detector.initialize_ocr)
        try:
            # An in-progress initialisation must not look like a load failure.
            assert not second.done() or second.result() is reader
        finally:
            release.set()
        assert first.result() is reader
        assert second.result() is reader


def benchmark_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location('lh_pipeline_benchmark', ROOT/'scripts/benchmark_pipeline.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_benchmark_never_claims_speed_for_different_results():
    module = benchmark_module()
    result = module.compare(lambda: ('original', 1.0), lambda: ('different', .1), 2)
    assert not result['equivalent_results']
    assert result['speed_ratio'] is None
    assert result['seconds_reduction_percent'] is None


def test_benchmark_refuses_overwrite_before_running(tmp_path, monkeypatch):
    module = benchmark_module()
    report = tmp_path/'report.json'
    report.write_bytes(b'keep this')
    run = Mock(side_effect=AssertionError('must check output first'))
    monkeypatch.setattr(module, 'run_benchmark', run)
    assert module.main(['--output', str(report)]) == 2
    assert report.read_bytes() == b'keep this'
    run.assert_not_called()


@pytest.mark.parametrize('args', [{'repeats': 0}, {'interval': -1}, {'limit': 0}, {'interval': 10, 'limit': 5}])
def test_benchmark_rejects_invalid_bounds_without_creating_datasets(args):
    with pytest.raises(ValueError):
        benchmark_module().run_benchmark(**args)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_noise_float_compatibility_retains_original_mean_precision(dtype):
    gray = np.random.default_rng(82).uniform(0, 256, (129, 193)).astype(dtype)
    assert QualityAnalyzer()._calculate_noise_score(gray) == legacy_noise(gray)
