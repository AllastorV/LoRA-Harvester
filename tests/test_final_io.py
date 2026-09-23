"""Final-release storage and legacy-pipeline regressions (no models or UI)."""
from __future__ import annotations
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
from PIL import Image
from src.core.dataset_files import transfer_image_pair
from src.core.clothing_io import digest_bytes, file_digest
from src.core.advanced_captioner import AdvancedCaptioner, TagSettings
from src.core.enhanced_processor import AsyncFrameSaver, EnhancedVideoProcessor


def image_pair(folder, caption='tag, blue skirt'):
    folder.mkdir(parents=True, exist_ok=True)
    image = folder / 'same.png'
    Image.new('RGB', (20, 20), 'blue').save(image)
    image.with_suffix('.txt').write_text(caption)
    return image


def payload():
    stream = io.BytesIO()
    Image.new('RGB', (40, 40), 'blue').save(stream, format='PNG')
    return stream.getvalue()


def legacy(tmp_path):
    return EnhancedVideoProcessor([], str(tmp_path), SimpleNamespace(), None,
        SimpleNamespace(target_format='1:1'), enable_quality_check=False)


def test_upscale_preserves_each_sources_own_caption_with_colliding_names(tmp_path):
    a = image_pair(tmp_path/'a', 'first')
    b = image_pair(tmp_path/'b', 'second')
    first = transfer_image_pair(a, tmp_path/'out', image_bytes=payload(), target_name='same.png')
    second = transfer_image_pair(b, tmp_path/'out', image_bytes=payload(), target_name='same.png')
    assert first != second
    assert first.with_suffix('.txt').read_text() == 'first'
    assert second.with_suffix('.txt').read_text() == 'second'
    assert Image.open(first).size == (40, 40)
    assert Image.open(a).size == (20, 20)


def test_upscale_changes_ownership_digest_without_owning_unrelated_tags(tmp_path):
    source = image_pair(tmp_path/'a')
    name = digest_bytes(source.name.encode()) + '.json'
    state = source.parent/'.lh-clothing/state'/name
    state.parent.mkdir(parents=True)
    state.write_text(json.dumps({'image_digest':file_digest(source), 'owned_tags':['blue skirt']}))
    dest = transfer_image_pair(source, tmp_path/'out', image_bytes=payload())
    saved = json.loads((dest.parent/'.lh-clothing/state'/name).read_text())
    assert saved['image_digest'] == file_digest(dest)
    assert saved['derived_from_image_digest'] == file_digest(source)
    assert saved['owned_tags'] == ['blue skirt']
    assert json.loads(state.read_text())['image_digest'] == file_digest(source)


def test_upscale_into_source_folder_preserves_source(tmp_path):
    source = image_pair(tmp_path)
    original = source.read_bytes()
    dest = transfer_image_pair(source, tmp_path, image_bytes=payload(), target_name='same.png')
    assert dest != source
    assert source.read_bytes() == original
    assert dest.with_suffix('.txt').read_text() == source.with_suffix('.txt').read_text()


def test_upscale_refuses_source_changed_during_inference(tmp_path):
    source = image_pair(tmp_path/'a')
    before = file_digest(source)
    source.write_bytes(payload())
    with pytest.raises(ValueError, match='changed'):
        transfer_image_pair(source, tmp_path/'out', image_bytes=payload(), expected_image_digest=before)
    assert list((tmp_path/'out').glob('*.png')) == []


def test_failed_derived_sidecar_publication_cleans_image(tmp_path):
    source = image_pair(tmp_path/'a')
    import src.core.dataset_files as mod
    publish = mod._publish_exclusive
    def fail_caption(temp, final):
        if final.suffix == '.txt':
            raise OSError('disk full')
        publish(temp, final)
    with patch.object(mod, '_publish_exclusive', side_effect=fail_caption):
        with pytest.raises(OSError):
            transfer_image_pair(source, tmp_path/'out', image_bytes=payload())
    assert list((tmp_path/'out').glob('*.png')) == []
    assert source.with_suffix('.txt').read_text() == 'tag, blue skirt'


def test_explicit_rating_switch_includes_category9(tmp_path):
    captioner = AdvancedCaptioner(enable_wd14=False,
        tag_settings=TagSettings(include_rating_tags=True, include_categories=['general']))
    assert 'general' in captioner.process_tags([('general', .99, 9), ('solo', .99, 0)])


def test_async_false_encode_is_error_not_success(tmp_path):
    saver = AsyncFrameSaver(num_workers=1)
    with patch('src.core.enhanced_processor.cv2.imencode', return_value=(False, None)), \
         patch('src.core.enhanced_processor.cv2.imwrite', return_value=False):
        saver.start()
        saver.save(np.zeros((16,16,3), dtype=np.uint8), str(tmp_path/'frame.png'))
        saver.stop()
    assert saver.get_stats()['saved'] == 0
    assert saver.get_stats()['errors'] == 1


def test_async_saver_does_not_overwrite_colliding_caption_or_frame(tmp_path):
    (tmp_path/'frame.txt').write_text('orphan caption')
    saver = AsyncFrameSaver(num_workers=2)
    saver.start()
    for _ in range(2):
        saver.save(np.zeros((16,16,3), dtype=np.uint8), str(tmp_path/'frame.png'))
    saver.stop()
    assert saver.get_stats()['saved'] == 2
    assert len(list(tmp_path.glob('*.png'))) == 2
    assert not (tmp_path/'frame.png').exists()
    assert (tmp_path/'frame.txt').read_text() == 'orphan caption'


def test_async_zero_workers_is_rejected():
    with pytest.raises(ValueError):
        AsyncFrameSaver(num_workers=0)


def test_legacy_output_and_checkpoints_distinguish_same_named_sources(tmp_path):
    proc = legacy(tmp_path/'out')
    proc.total_frames = 200
    outputs=[]
    for folder in ['a', 'b']:
        video = tmp_path/folder/'clip.mp4'
        video.parent.mkdir()
        video.write_bytes(folder.encode())
        proc.current_video = str(video)
        outputs.append(proc.create_output_structure('clip'))
        proc.save_checkpoint(str(video), 10, {})
    assert outputs[0] != outputs[1]
    assert len(list(proc.checkpoint_dir.glob('*.json'))) == 2
    assert proc.load_checkpoint(str(tmp_path/'a/clip.mp4')).last_frame == 10


def test_legacy_checkpoint_is_not_advanced_after_failed_write(tmp_path):
    proc=legacy(tmp_path/'out')
    video=tmp_path/'clip.mp4'
    video.write_bytes(b'video')
    proc.async_saver.error_count=1
    with pytest.raises(OSError):
        proc.save_checkpoint(str(video), 100, {})
    assert not list(proc.checkpoint_dir.glob('*.json'))


def test_changed_video_invalidates_checkpoint(tmp_path):
    proc=legacy(tmp_path/'out')
    video=tmp_path/'clip.mp4'
    video.write_bytes(b'old video')
    proc.total_frames = 200
    proc.save_checkpoint(str(video), 100, {})
    video.write_bytes(b'replacement with different size')
    assert proc.load_checkpoint(str(video)) is None
