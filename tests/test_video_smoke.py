"""Decode real synthetic videos on CPU; detector/caption outputs are fixtures."""
from pathlib import Path
from unittest.mock import patch
import threading
import cv2
import numpy as np
import pytest
from src.core.unified_processor import UnifiedVideoProcessor
from src.core.enhanced_processor import EnhancedVideoProcessor


class FixedDetector:
    def detect(self, frame):
        return [{'bbox': [8, 8, 48, 48], 'confidence': .99}]
    def detect_batch(self, frames):
        return [self.detect(frame) for frame in frames]
    def get_primary_subject(self, detections):
        return 'person', detections[0]
    def calculate_head_space(self, *args):
        return 0


class FixedCropper:
    target_format='1:1'
    def calculate_crop_box(self, shape, bbox, category, head, **kwargs):
        return (8, 8, 48, 48)
    def apply_crop(self, frame, bbox):
        x1,y1,x2,y2 = bbox
        return frame[y1:y2, x1:x2].copy()
    def calculate_quality_score(self, *args):
        return .99


@pytest.fixture
def clip(tmp_path):
    path = tmp_path/'sample.avi'
    writer=cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 10, (64,64))
    assert writer.isOpened(), 'MJPG writer required by this CPU smoke test'
    for i in range(12):
        frame=np.zeros((64,64,3), np.uint8)
        frame[:,:,i%3] = 60+i*10
        writer.write(frame)
    writer.release()
    return path


def processor(path, out, turbo):
    return UnifiedVideoProcessor(str(path), str(out), FixedDetector(), None, FixedCropper(),
                                 use_turbo=turbo, batch_size=4)


@pytest.mark.parametrize('turbo', [False, True])
def test_unified_real_decode_keeps_last_frame_and_saves_exactly_once(clip,tmp_path,turbo):
    p=processor(clip,tmp_path/'output',turbo)
    callbacks=[]
    stats=p.process_all_videos(frame_interval=1, skip_text=False, frame_saved_callback=callbacks.append)
    images=list((tmp_path/'output').rglob('*.png'))
    assert stats['total_frames_saved'] == len(images) == len(callbacks) == 12
    assert any('frame_000012' in image.name for image in images)
    assert all(cv2.imread(str(image)).shape[:2] == (40,40) for image in images)
    original={image:image.read_bytes() for image in images}
    # Reusing an instance must report this run's totals, not cumulative totals.
    stats=p.process_all_videos(frame_interval=1, skip_text=False)
    assert stats['total_frames_saved'] == 12
    assert len(list((tmp_path/'output').rglob('*.png'))) == 24
    assert all(path.read_bytes() == data for path,data in original.items())


@pytest.mark.parametrize('turbo',[False,True])
def test_trim_is_zero_based_half_open_interval(clip,tmp_path,turbo):
    p=processor(clip,tmp_path/'output',turbo)
    stats=p.process_single_video(str(clip), frame_interval=1, skip_text=False,
                                 start_skip_seconds=.2, end_skip_seconds=.2)
    names={image.stem.split('_q')[0] for image in (tmp_path/'output').rglob('*.png')}
    assert stats['saved_frames'] == 8
    assert names == {f'frame_{i:06d}' for i in range(3,11)}


def test_legacy_stop_near_eof_keeps_resume_and_success_removes_it(clip,tmp_path):
    p=EnhancedVideoProcessor(str(clip),str(tmp_path/'output'),FixedDetector(),None,FixedCropper(),
                             enable_quality_check=False)
    first=p.process_single_video(str(clip),frame_interval=1,skip_text=False,resume=False,
                                  stop_callback=lambda:p.stats['processed_frames']>=5)
    assert first['saved_frames'] == 5
    checkpoint=p.load_checkpoint(str(clip))
    assert checkpoint is not None and checkpoint.last_frame == 5
    with patch('builtins.input', return_value='y'):
        final=p.process_single_video(str(clip),frame_interval=1,skip_text=False,resume=True)
    assert final['saved_frames'] == 12
    assert len(list((tmp_path/'output').rglob('*.png'))) == 12
    assert p.load_checkpoint(str(clip)) is None


def test_bad_legacy_video_does_not_return_previous_videos_stats(clip,tmp_path):
    p=EnhancedVideoProcessor(str(clip),str(tmp_path/'output'),FixedDetector(),None,FixedCropper(),
                             enable_quality_check=False)
    assert p.process_single_video(str(clip),frame_interval=1,skip_text=False,resume=False)['saved_frames'] == 12
    assert p.process_single_video(str(tmp_path/'missing.avi'),resume=False)['saved_frames'] == 0


@pytest.mark.parametrize('interval',[0,-1,True])
def test_unified_rejects_invalid_interval_before_open(clip,tmp_path,interval):
    p=processor(clip,tmp_path/'output',False)
    with pytest.raises(ValueError):
        p.process_single_video(str(clip),frame_interval=interval)

@pytest.mark.parametrize('turbo',[False,True])
def test_paused_processor_honors_stop_without_manual_resume(clip,tmp_path,turbo):
    p=processor(clip,tmp_path/'output',turbo)
    pause=threading.Event()
    stop=threading.Event()
    errors=[]
    def run():
        try:
            p.process_single_video(str(clip),frame_interval=1,pause_event=pause,
                                   stop_callback=stop.is_set)
        except Exception as exc:
            errors.append(exc)
    worker=threading.Thread(target=run,daemon=True)
    worker.start()
    stop.set()
    worker.join(2)
    if worker.is_alive():
        pause.set()  # release the worker even when the regression fails
        worker.join(2)
        pytest.fail('Paused processor ignored stop until manually resumed')
    assert errors == []


@pytest.mark.parametrize('legacy',[False,True])
def test_videos_stats_report_each_videos_own_output_dir(clip,tmp_path,legacy):
    videos=[str(clip), str(tmp_path/'missing.avi')]
    if legacy:
        p=EnhancedVideoProcessor(videos,str(tmp_path/'output'),FixedDetector(),None,FixedCropper(),
                                 enable_quality_check=False)
        stats=p.process_all_videos(frame_interval=1,skip_text=False,resume=False)
    else:
        p=UnifiedVideoProcessor(videos,str(tmp_path/'output'),FixedDetector(),None,FixedCropper())
        stats=p.process_all_videos(frame_interval=1,skip_text=False)
    first,missing=stats['videos_stats']
    assert len(list(Path(first['output_dir']).rglob('*.png'))) == 12
    # A video that cannot be opened must not inherit the previous video's folder.
    assert missing['output_dir'] is None


def test_cli_captions_only_folders_written_by_this_run(clip,tmp_path,monkeypatch):
    import importlib.util
    import sys
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('lh_scripts_cli', root/'scripts'/'cli.py')
    cli=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    captioned=[]
    class RecordingCaptioner:
        def caption_directory(self, directory, **kwargs):
            captioned.append(Path(directory))
    monkeypatch.setattr(cli,'ObjectDetector',lambda **kw: FixedDetector())
    monkeypatch.setattr(cli,'SmartCropper',lambda **kw: FixedCropper())
    monkeypatch.setattr(cli,'AdvancedCaptioner',lambda **kw: RecordingCaptioner())
    output=tmp_path/'output'
    unrelated=output/'older_dataset'/'persons'
    unrelated.mkdir(parents=True)
    monkeypatch.setattr(sys,'argv',['cli.py',str(clip),'-o',str(output),'-i','1',
                                    '--no-skip-text','--no-resume','--caption'])
    cli.main()
    assert captioned, 'captioning was not run on the new frames'
    assert all(path.parent.parent == output and path.parent.name.startswith('sample_')
               for path in captioned)
    assert unrelated not in captioned


@pytest.mark.parametrize('script',['scripts/cli.py','cli.py'])
def test_cli_preset_keeps_its_values_and_applies_documented_suffix(clip,tmp_path,monkeypatch,script):
    import importlib.util
    import sys
    from src.core.advanced_captioner import CAPTIONER_PRESETS
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('lh_cli_under_test', root/script)
    cli=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    captured={}
    class RecordingCaptioner:
        def __init__(self, tag_settings=None, **kwargs):
            captured['settings']=tag_settings
        def caption_directory(self, directory, **kwargs):
            pass
    monkeypatch.setattr(cli,'ObjectDetector',lambda **kw: FixedDetector())
    monkeypatch.setattr(cli,'SmartCropper',lambda **kw: FixedCropper())
    monkeypatch.setattr(cli,'AdvancedCaptioner',RecordingCaptioner)
    preset=CAPTIONER_PRESETS['anime_character']
    before=(preset.trigger_word, preset.max_tags, preset.caption_suffix)
    # The README's "With captions" example.
    monkeypatch.setattr(sys,'argv',['cli.py',str(clip),'-o',str(tmp_path/'out'),'-i','6',
                                    '--no-skip-text','--no-resume','--caption',
                                    '--preset','anime_character','--trigger','mychar',
                                    '--suffix','masterpiece, best quality'])
    cli.main()
    settings=captured['settings']
    assert settings.trigger_word == 'mychar'
    assert settings.caption_suffix == 'masterpiece, best quality'
    assert settings.max_tags == preset.max_tags == 25
    assert settings.min_confidence == preset.min_confidence
    assert (preset.trigger_word, preset.max_tags, preset.caption_suffix) == before
