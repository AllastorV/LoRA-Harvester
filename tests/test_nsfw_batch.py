"""Ordered classifier/buffering tests. No downloaded weights or CUDA required.

The simulated pipeline models the real HF single-image vs list return shape.
File/video checks use real PNG writes and an actual MJPG decoder.
"""
from pathlib import Path
from types import SimpleNamespace, ModuleType
import ast
import sys
import threading
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from src.core.nsfw_detector import NsfwDetector
from src.core.unified_processor import UnifiedVideoProcessor


class RecordingPipeline:
    def __init__(self, max_size=99, broken=False, gpu=True):
        self.device = 'cuda:0' if gpu else 'cpu'
        self.calls = []
        self.forward_sizes = []
        self.max_size = max_size
        self.broken = broken
        self.model = SimpleNamespace(to=lambda device: None)

    def __call__(self, inputs, **kwargs):
        is_batch = isinstance(inputs, list)
        images = inputs if is_batch else [inputs]
        self.calls.append((len(images), self.device, kwargs, is_batch))
        if self.device.startswith('cuda') and len(images) > self.max_size:
            raise RuntimeError('CUDA out of memory (test fixture)')
        if self.broken:
            raise RuntimeError('Corrupt model output fixture')
        self.forward_sizes.append(len(images))
        rows = []
        for image in images:
            # Derive class from the red channel of the supplied RGB PIL image.
            score = image.getpixel((0, 0))[0] / 255
            rows.append(sorted([{'label': 'nsfw', 'score': score},
                                {'label': 'normal', 'score': 1-score}],
                               key=lambda item: -item['score']))
        return rows if is_batch else rows[0]


def loaded(max_size=99, gpu=True, batch=8):
    detector = NsfwDetector(device='cuda' if gpu else 'cpu', batch_size=batch)
    detector._checked = True
    detector._active_backend = 'falconsai'
    detector.actual_device = 'cuda:0' if gpu else 'cpu'
    detector._effective_batch_size = batch if gpu else 1
    detector._pipe = RecordingPipeline(max_size=max_size, gpu=gpu)
    return detector


def image(value, shape=(16,16)):
    frame = np.zeros((*shape,3),np.uint8)
    frame[:,:,2] = value
    return frame


def test_empty_input_does_not_load_model():
    detector = NsfwDetector()
    assert detector.classify_batch([]) == []
    assert detector._checked is False


@pytest.mark.parametrize('batch', [1, 2, 4, 8, 16])
def test_batch_matches_single_crop_decisions_and_bgr_order(batch):
    frames=[image(value) for value in [0,255,128,60,240,90,10,215,80,230,190]]
    detector=loaded(batch=batch)
    results=detector.classify_batch(frames)
    sequential=loaded(batch=1).classify_batch(frames)
    assert results == sequential
    assert results[0][0]=='sfw' and results[1][0]=='nsfw' and results[2][0]=='uncertain'
    assert sum(detector._pipe.forward_sizes)==len(frames)
    assert max(detector._pipe.forward_sizes)<=batch
    for size, _, kwargs, is_batch in detector._pipe.calls:
        if is_batch:
            assert kwargs['num_workers'] == 0 and kwargs['batch_size']==size
        assert kwargs['top_k']==2


def test_eight_images_really_use_one_pipeline_batch_call():
    detector=loaded()
    detector.classify_batch([image(20)]*8)
    assert detector._pipe.forward_sizes == [8]
    assert detector.metrics()['pipeline_calls']==1
    assert detector.metrics()['model_images']==8


def test_cpu_avoids_dataloader_and_forced_batching():
    detector=loaded(gpu=False)
    detector.classify_batch([image(0), image(255), image(127)])
    assert detector._pipe.forward_sizes == [1,1,1]
    assert all(not call[3] for call in detector._pipe.calls)


def test_invalid_inputs_do_not_shift_other_predictions():
    detector=loaded()
    frames=[image(0),None,image(255),np.empty((0,5,3),np.uint8),
            np.zeros((8,8),np.uint8),image(128).astype(float),image(255)]
    result=detector.classify_batch(frames)
    assert [label for label,_ in result]==['sfw','uncertain','nsfw','uncertain','uncertain','uncertain','nsfw']
    assert detector.metrics()['model_images']==3
    assert detector.metrics()['failed_images']==4


def test_negative_stride_view_is_supported_and_not_modified():
    frame=image(230,(12,24))[:,::-1]
    before=frame.copy()
    assert loaded().classify(frame)[0]=='nsfw'
    np.testing.assert_array_equal(frame,before)


def test_oom_halves_batch_and_remembers_limit():
    detector=loaded(max_size=2)
    frames=[image(i) for i in [0,255,128,20,240,80,230,100,200,30]]
    result=detector.classify_batch(frames)
    assert result == loaded(batch=1).classify_batch(frames)
    assert detector.effective_batch_size==2
    assert detector.metrics()['oom_retries']==2
    assert detector._pipe.forward_sizes==[2]*5
    detector.classify_batch(frames[:4])
    assert detector._pipe.forward_sizes[-2:]==[2,2]


def test_single_oom_moves_same_model_to_cpu_and_retries():
    detector=loaded(max_size=0,batch=1)
    pipe=detector._pipe
    # torch.device assigned by production fallback is stringified by fixture.
    original=pipe.__class__.__call__
    def call(self,*a,**kw):
        self.device=str(self.device)
        return original(self,*a,**kw)
    with patch.object(RecordingPipeline,'__call__',call):
        assert detector.classify(image(255))[0]=='nsfw'
        assert detector.classify(image(0))[0]=='sfw'
    assert detector._pipe is pipe and detector.actual_device=='cpu'
    assert detector.metrics()['cpu_fallbacks']==1
    assert detector.effective_batch_size==1


@pytest.mark.parametrize('row', [None, [], [{}], [{'label':'normal','score':float('nan')}],
                                    [{'label':'normal','score':float('inf')}],
                                    [{'label':'normal','score':2}],
                                    [{'label':'strange','score':.99}]])
def test_malformed_predictions_are_never_sfw(row):
    assert loaded()._parse_prediction(row)==('uncertain',.5)


def test_model_error_does_not_switch_backend_or_drop_results():
    detector=loaded()
    detector._pipe.broken=True
    result=detector.classify_batch([image(0)]*5)
    assert result==[('uncertain',.5)]*5
    assert detector._active_backend=='falconsai'
    assert detector.metrics()['failed_images']==5


def test_model_response_count_mismatch_is_not_silently_zipped():
    detector=loaded()
    detector._pipe=lambda *args,**kw: [[{'label':'normal','score':1}]]
    assert detector.classify_batch([image(255),image(0)])==[('uncertain',.5)]*2


def test_auto_no_model_does_not_announce_wd14_fallback():
    detector=NsfwDetector()
    with patch.object(detector,'_try_init_falconsai',return_value=False):
        assert detector.is_available() is False
    assert detector._active_backend is None
    assert detector.classify(image(0))==('uncertain',.5)


def test_explicit_legacy_backends_still_exist():
    detector=NsfwDetector(backend='wd14_tags')
    assert detector.is_available()
    assert detector.classify_from_wd14_tags({'rating:explicit':.9})[0]=='nsfw'
    assert detector.classify(image(0))==('uncertain',.5)
    assert NsfwDetector(backend='heuristic').classify(image(0))[0]=='sfw'


@pytest.mark.parametrize('batch', [0,-1,33,True,1.2])
def test_invalid_batch_is_rejected(batch):
    with pytest.raises(ValueError):
        NsfwDetector(batch_size=batch)


def test_load_uses_cuda_and_does_not_force_half_precision():
    calls=[]
    module=ModuleType('transformers')
    def factory(*args,**kwargs):
        calls.append(kwargs)
        return RecordingPipeline()
    module.pipeline=factory
    with patch.dict(sys.modules,{'transformers':module}), patch('torch.cuda.is_available',return_value=True):
        detector=NsfwDetector(device='cuda')
        assert detector.is_available()
        assert detector.is_available()
    assert len(calls)==1 and calls[0]['device']==0
    assert 'torch_dtype' not in calls[0] and 'dtype' not in calls[0]
    assert calls[0]['model']=='Falconsai/nsfw_image_detection'
    assert detector.effective_batch_size==8


def test_cleanup_releases_even_when_caller_keeps_detector_reference():
    detector=loaded()
    detector.cleanup()
    assert detector._pipe is None and detector._active_backend is None
    assert not detector._checked


class FixedDetector:
    def detect(self, frame):
        return {'person':[{'bbox':[4,4,28,28]}]}
    def detect_batch(self, frames):
        return [self.detect(frame) for frame in frames]
    def get_primary_subject(self, detections):
        return 'person',detections['person'][0]
    def calculate_head_space(self,*args):
        return 0

class FixedCropper:
    target_format='1:1'
    def calculate_crop_box(self,*args,**kwargs):
        return (4,4,28,28)
    def apply_crop(self,frame,bbox):
        return frame[4:28,4:28]
    def calculate_quality_score(self,*args):
        return .95


def processor(tmp_path,detector=None,turbo=True):
    p=UnifiedVideoProcessor([], str(tmp_path/'output'),FixedDetector(),None,FixedCropper(),
                            use_turbo=turbo,batch_size=4,nsfw_detector=detector or loaded())
    p.create_output_structure('test')
    return p


def assert_caption_pairs(p):
    # A stand-in caption writes the exact final crop's first pixel value.
    def caption(frame,path,number):
        path.with_suffix('.txt').write_text(str(frame[0,0,2]))
        p.stats['captioned_frames']+=1
    p._caption_frame=caption


def test_real_writes_keep_classification_caption_and_callback_order(tmp_path):
    detector=loaded()
    p=processor(tmp_path,detector)
    p._defer_nsfw_saves=True
    saved=[]
    p._frame_saved_callback=saved.append
    assert_caption_pairs(p)
    values=[0,255,128,30,200,40,220,60,240,90]
    for n,value in enumerate(values,1):
        p._submit_crop(image(value),'person',n,.9,False)
    p._flush_pending_crops()
    assert detector._pipe.forward_sizes==[8,2]
    assert len(saved)==len(values)==p.stats['saved_frames']==p.stats['captioned_frames']
    expected=loaded(batch=1).classify_batch([image(value) for value in values])
    for path,value,(label,_) in zip(map(Path,saved),values,expected):
        assert path.parent.name==label
        assert path.with_suffix('.txt').read_text()==str(value)
        np.testing.assert_array_equal(cv2.imread(str(path)),image(value))
    assert p.stats['sfw_frames']+p.stats['nsfw_frames']+p.stats['nsfw_uncertain']==len(values)
    assert not p._pending_crops


def test_mixed_categories_only_classify_persons_and_keep_save_order(tmp_path):
    detector=loaded()
    p=processor(tmp_path,detector)
    p._defer_nsfw_saves=True
    saved=[];p._frame_saved_callback=saved.append
    for i,category in enumerate(['person','animal','object','person']):
        p._submit_crop(image(255),category,i+1,.9,False)
    p._flush_pending_crops()
    assert detector._pipe.forward_sizes==[2]
    assert [Path(path).parent.name for path in saved]==['nsfw','animals','objects','nsfw']
    assert p.stats['nsfw_frames']==2 and p.stats['saved_frames']==4


def test_buffer_compacts_view_and_honors_memory_ceiling(tmp_path):
    p=processor(tmp_path)
    p._defer_nsfw_saves=True
    p._pending_crop_max_bytes=2*16*16*3
    frame=image(255,(64,64))
    view=frame[:16,:16]
    p._submit_crop(view,'person',1,.9,False)
    assert p._pending_crops[0][0].base is None
    frame[:]=0
    assert p._pending_crops[0][0][0,0,2]==255
    for n in [2,3,4,5]:
        p._submit_crop(image(0),'person',n,.9,False)
        assert p._pending_crop_bytes<=p._pending_crop_max_bytes
    p._flush_pending_crops()
    assert p.stats['saved_frames']==5 and p.stats['nsfw_frames']==1


def test_oversized_crop_is_not_buffered_or_dropped(tmp_path):
    p=processor(tmp_path)
    p._defer_nsfw_saves=True
    p._pending_crop_max_bytes=1
    p._submit_crop(image(255),'person',1,.9,False)
    assert p.stats['saved_frames']==1 and p._pending_crop_bytes==0


def test_failed_disk_write_does_not_increment_rating_or_callback(tmp_path):
    p=processor(tmp_path)
    saved=[];p._frame_saved_callback=saved.append
    with patch('src.core.unified_processor._imwrite_unicode',return_value=False):
        p._submit_crop(image(255),'person',1,.9,False)
    assert p.stats['saved_frames']==p.stats['nsfw_frames']==0
    assert p.stats['save_errors']==1 and saved==[]


def test_classifier_oom_cannot_replay_or_drop_sibling_writes(tmp_path):
    detector=loaded(max_size=2)
    p=processor(tmp_path,detector)
    p._defer_nsfw_saves=True
    saved=[];p._frame_saved_callback=saved.append
    for n in range(1,9):
        p._submit_crop(image(255 if n%2 else 0),'person',n,.9,False)
    assert len(saved)==len(set(saved))==8
    assert detector.metrics()['oom_retries']==2
    assert p.stats['nsfw_frames']==p.stats['sfw_frames']==4


def test_external_bad_result_labels_cannot_escape_output_folder(tmp_path):
    p=processor(tmp_path)
    path=p.save_cropped_frame(image(0),'person',1,.9,nsfw_result=('../../outside',.99))
    assert path.parent==p.person_dir/'uncertain'
    assert p.stats['nsfw_uncertain']==1


def test_explicit_legacy_uncertain_folder_choice_keeps_uncertain_count(tmp_path):
    p=processor(tmp_path);p.nsfw_uncertain_folder=False
    path=p.save_cropped_frame(image(128),'person',1,.9)
    assert path.parent.name=='sfw' and p.stats['nsfw_uncertain']==1 and p.stats['sfw_frames']==0


@pytest.fixture
def clip(tmp_path):
    path=tmp_path/'synthetic.avi'
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),10,(32,32))
    assert writer.isOpened()
    for i in range(10):
        writer.write(image(255 if i%2 else 0,(32,32)))
    writer.release()
    return path


@pytest.mark.parametrize('turbo',[False,True])
def test_video_tail_and_second_video_own_directory(clip,tmp_path,turbo):
    detector=loaded()
    p=processor(tmp_path,detector,turbo)
    p.video_paths=[str(clip),str(clip)]
    callbacks=[]
    stats=p.process_all_videos(frame_interval=1,skip_text=False,frame_saved_callback=callbacks.append)
    assert stats['total_frames_saved']==20
    assert len(callbacks)==20 and len(set(callbacks))==20
    assert len({Path(path).parents[2] for path in callbacks})==2
    assert not p._pending_crops and not p._defer_nsfw_saves
    for video in stats['videos_stats']:
        assert video['stats']['nsfw_crops_analyzed']==10
        assert video['stats']['nsfw_batches']<10
    assert detector._pipe.forward_sizes==([4,4,2]*2 if turbo else [8,2]*2)


def test_stop_drains_only_prepared_standard_crops(clip,tmp_path):
    p=processor(tmp_path,turbo=False)
    stats=p.process_single_video(str(clip),frame_interval=1,skip_text=False,
                                stop_callback=lambda:p.stats['processed_frames']>=3)
    assert stats['saved_frames']==3 and stats['nsfw_crops_analyzed']==3
    assert not p._pending_crops


def test_pause_flushes_before_wait_then_stop_returns(clip,tmp_path):
    p=processor(tmp_path,turbo=False)
    pause=threading.Event();pause.set()
    stop=threading.Event()
    original=p._process_single_frame
    def process(*args):
        original(*args)
        if p.stats['processed_frames']==2:
            pause.clear()
    p._process_single_frame=process
    errors=[]
    def work():
        try:
            p.process_single_video(str(clip),frame_interval=1,skip_text=False,
                                   pause_event=pause,stop_callback=stop.is_set)
        except Exception as exc:
            errors.append(exc)
    worker=threading.Thread(target=work)
    worker.start()
    import time
    deadline=time.monotonic()+3
    while p.stats['saved_frames']<2 and time.monotonic()<deadline:
        time.sleep(.01)
    stop.set();worker.join(2)
    if worker.is_alive():
        pause.set();worker.join(2)
    assert not worker.is_alive() and not errors
    assert p.stats['saved_frames']==2 and not p._pending_crops


def test_worker_cleanup_precedes_clothing_and_missing_model_is_error():
    # Static contract: no Qt dependency. Actual visual UI is not exercised.
    path=Path(__file__).resolve().parents[1]/'src/ui/main_window.py'
    source=path.read_text()
    tree=ast.parse(source)
    worker=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ProcessingThread')
    run=next(n for n in worker.body if isinstance(n,ast.FunctionDef) and n.name=='run')
    text=ast.get_source_segment(source,run)
    assert text.index('nsfw_detector.cleanup()')<text.index("optional_pipeline_pass(")
    assert "res.get('gpu_enabled', True)" in text
    assert 'separation was not silently disabled' in text
    assert 'continuing with heuristic' not in text


def test_animals_alone_do_not_wait_for_nsfw_batch(tmp_path):
    p=processor(tmp_path)
    p._defer_nsfw_saves=True
    p._submit_crop(image(0),'animal',1,.9,False)
    assert p.stats['saved_frames']==1 and not p._pending_crops
    assert p.nsfw_detector.metrics()['pipeline_calls']==0


def test_benchmark_alternates_order_and_measures_actual_calls():
    from scripts.benchmark_sfw import benchmark
    result=benchmark(loaded(),[image(0),image(255)]*4,repeats=2)
    assert [run['mode'] for run in result['runs']]==['single','batch','batch','single']
    assert [run['pipeline_calls'] for run in result['runs']]==[8,1,1,8]
    assert result['comparison_valid']
    assert all(not item['label_mismatch_indices'] for item in result['agreement'])


def test_benchmark_model_error_is_not_reported_as_high_speed():
    from scripts.benchmark_sfw import benchmark
    detector=loaded();detector._pipe.broken=True
    with pytest.raises(RuntimeError,match='Warmup failed'):
        benchmark(detector,[image(0)]*2)


def test_benchmark_reads_images_without_touching_caption_or_metadata(tmp_path):
    from scripts.benchmark_sfw import load_images
    from hashlib import sha256
    paths=[]
    for name,value in [('a',0),('b',255)]:
        path=tmp_path/f'{name}.png';cv2.imwrite(str(path),image(value))
        path.with_suffix('.txt').write_text(f'caption,{name}')
        paths.extend([path,path.with_suffix('.txt')])
    hidden=tmp_path/'.lh-clothing';hidden.mkdir()
    cv2.imwrite(str(hidden/'ignore.png'),image(128))
    before={path:sha256(path.read_bytes()).hexdigest() for path in paths}
    frames,sources,warnings=load_images(tmp_path,limit=8)
    assert len(frames)==len(sources)==2 and not warnings
    assert {path:sha256(path.read_bytes()).hexdigest() for path in paths}==before


def test_benchmark_never_overwrites_existing_report(tmp_path):
    from scripts.benchmark_sfw import main
    report=tmp_path/'existing.txt';report.write_text('keep')
    assert main([str(tmp_path),'--output',str(report)])==1
    assert report.read_text()=='keep'


def test_explicit_cpu_is_respected_even_when_cuda_exists():
    calls=[];module=ModuleType('transformers')
    def factory(*args,**kwargs):
        calls.append(kwargs['device'])
        return RecordingPipeline(gpu=False)
    module.pipeline=factory
    with patch.dict(sys.modules,{'transformers':module}),patch('torch.cuda.is_available',return_value=True):
        detector=NsfwDetector(device='cpu')
        assert detector.is_available()
    assert calls==[-1] and detector.actual_device=='cpu' and detector.effective_batch_size==1


def test_gpu_load_oom_keeps_same_model_on_cpu_not_heuristic():
    calls=[];module=ModuleType('transformers')
    def factory(*args,**kwargs):
        calls.append(kwargs)
        if kwargs['device']==0:
            raise RuntimeError('CUDA out of memory: load test')
        return RecordingPipeline(gpu=False)
    module.pipeline=factory
    with patch.dict(sys.modules,{'transformers':module}),patch('torch.cuda.is_available',return_value=True),patch.object(NsfwDetector,'_clear_cuda_cache'):
        detector=NsfwDetector(device='cuda')
        assert detector.is_available()
    assert [c['device'] for c in calls]==[0,-1]
    assert calls[0]['model']==calls[1]['model']
    assert detector.actual_device=='cpu' and detector._active_backend=='falconsai'
    assert detector.metrics()['cpu_fallbacks']==1


def test_pipeline_load_exception_is_reported_not_hidden():
    module=ModuleType('transformers')
    def factory(*a,**kw):
        raise OSError('missing cached model')
    module.pipeline=factory
    with patch.dict(sys.modules,{'transformers':module}):
        detector=NsfwDetector(device='cpu')
        assert not detector.is_available()
    assert 'missing cached model' in detector.last_error
    assert detector._pipe is None
