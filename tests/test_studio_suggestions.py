import json
import os
from pathlib import Path
import shutil
import threading
import time
from unittest.mock import patch

import pytest
from PIL import Image

from src.core.smart_suggestions import (scan_suggestions, ScanOptions, ScanCancelled,
    MetadataCache, MAX_CAPTION_BYTES, build_suggestions)


def image(root, name='test.png', color=(30, 90, 150), size=(600, 800), caption='1girl, standing'):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new('RGB', size, color).save(path)
    if caption is not None:
        path.with_suffix('.txt').write_text(caption, encoding='utf-8')
    return path


def suggestions(report):
    return {s.key: s for s in report.suggestions}


def test_empty_dataset_is_not_training_readiness(tmp_path):
    report = scan_suggestions(tmp_path)
    assert report.total == report.captioned == report.missing == report.flagged == 0
    data = report.to_dict()
    assert 'health_score' not in data and 'training_ready' not in data


def test_missing_and_empty_and_invalid_caption_are_distinct(tmp_path):
    a = image(tmp_path, 'missing.png', caption=None)
    b = image(tmp_path, 'empty.png', caption=' \n\t')
    c = image(tmp_path, 'unreadable.png')
    c.with_suffix('.txt').write_bytes(b'\xff\xfe')
    d = image(tmp_path, 'valid.png')
    report = scan_suggestions(tmp_path)
    assert report.total == 4 and report.captioned == 1 and report.missing == 2
    by_path = {r.path: r for r in report.records}
    assert by_path[str(c)].caption_state == 'unreadable'
    assert set(suggestions(report)) == {'missing_caption', 'empty_caption', 'unreadable_caption'}


def test_bom_caption_not_empty_or_missing(tmp_path):
    a = image(tmp_path, caption='')
    a.with_suffix('.txt').write_bytes(b'\xef\xbb\xbfstanding\r\n')
    report = scan_suggestions(tmp_path)
    assert report.records[0].caption == 'standing'
    assert report.captioned == 1 and report.missing == 0


def test_sizes_are_facts_not_quality_scores(tmp_path):
    image(tmp_path, size=(511, 800))
    report = scan_suggestions(tmp_path)
    assert report.records[0].width == 511
    assert 'small_image' in report.records[0].issues
    assert 'estetik' in suggestions(report)['small_image'].detail_tr
    assert not scan_suggestions(tmp_path, options=ScanOptions(minimum_side=0)).suggestions


def test_corrupt_image_is_not_valid(tmp_path):
    (tmp_path / 'broken.png').write_bytes(b'not an image')
    report = scan_suggestions(tmp_path)
    assert 'unreadable_image' in report.records[0].issues
    assert report.records[0].width == 0


def test_exact_duplicates_are_opt_in(tmp_path):
    a = image(tmp_path, 'a.png')
    b = tmp_path / 'b.png'; shutil.copyfile(a, b)
    b.with_suffix('.txt').write_text('different caption')
    default = scan_suggestions(tmp_path)
    assert default.duplicate_extras == 0
    assert 'duplicate_image' not in suggestions(default)
    report = scan_suggestions(tmp_path, options=ScanOptions(exact_duplicates=True))
    assert report.duplicate_extras == 1
    assert suggestions(report)['duplicate_image'].count == 2
    assert {r.caption for r in report.records} == {'1girl, standing', 'different caption'}


def test_same_pixels_different_file_bytes_not_claimed_exact(tmp_path):
    a = image(tmp_path, 'a.png')
    image(tmp_path, 'b.bmp')
    report = scan_suggestions(tmp_path, options=ScanOptions(exact_duplicates=True))
    assert report.duplicate_extras == 0


def test_shared_caption_stem_detected_with_case_extension(tmp_path):
    image(tmp_path, 'same.png')
    image(tmp_path, 'same.jpg')
    image(tmp_path / 'separate', 'same.png')
    report = scan_suggestions(tmp_path)
    assert suggestions(report)['caption_collision'].count == 2


def test_duplicate_tags_normalize_case_underscore_but_preserve_literal(tmp_path):
    cap = 'MasterCUSTOM, Blue_Skirt, blue skirt, standing'
    image(tmp_path, caption=cap)
    report = scan_suggestions(tmp_path)
    assert report.records[0].caption == cap and report.records[0].tag_count == 4
    assert 'duplicate_tags' in report.records[0].issues


def test_profile_masters_color_variants_remain_separate(tmp_path):
    image(tmp_path, 'blue.png', caption='blueUniform, standing')
    image(tmp_path, 'red.png', caption='redUniform, from side')
    report = scan_suggestions(tmp_path, masters=['blueUniform', 'redUniform'])
    assert {r.outfit for r in report.records} == {'blueUniform', 'redUniform'}
    assert not any('multiple_outfits' in r.issues for r in report.records)


def test_multiple_masters_warn_without_deleting_caption(tmp_path):
    path = image(tmp_path, caption='blueUniform, redUniform, 2girls')
    before = path.with_suffix('.txt').read_bytes()
    report = scan_suggestions(tmp_path, masters=['blueUniform', 'redUniform'])
    assert suggestions(report)['multiple_outfits'].count == 1
    assert path.with_suffix('.txt').read_bytes() == before


def test_no_guessing_missing_pose_or_color(tmp_path):
    image(tmp_path, caption='blue skirt, looking at viewer')
    report = scan_suggestions(tmp_path, masters=['blueUniform'])
    record = report.records[0]
    assert record.outfit == record.pose == record.angle == 'unknown'


def test_underrepresented_outfits_use_only_observed_labels(tmp_path):
    for i in range(9):
        image(tmp_path, f'{i}.png', caption='blueUniform' if i < 8 else 'redUniform')
    report = scan_suggestions(tmp_path, masters=['blueUniform', 'redUniform', 'not_in_dataset'])
    suggestion = suggestions(report)['outfit_imbalance']
    assert suggestion.count == 1 and suggestion.route == 'balance'
    assert 'not_in_dataset' not in suggestion.title_en


def test_balanced_data_not_flagged(tmp_path):
    image(tmp_path, 'a.png', caption='blueUniform')
    image(tmp_path, 'b.png', caption='redUniform')
    assert 'outfit_imbalance' not in suggestions(scan_suggestions(tmp_path, masters=['blueUniform', 'redUniform']))


def test_prunes_metadata_and_respects_recursive_flag(tmp_path):
    image(tmp_path, 'a.png')
    image(tmp_path / 'nested', 'b.png')
    for d in ('.lh-clothing', '.lh-studio', '_rejected', '_approved', '__pycache__', 'venv'):
        image(tmp_path / d, 'not_input.png')
    assert scan_suggestions(tmp_path).total == 2
    assert scan_suggestions(tmp_path, options=ScanOptions(recursive=False)).total == 1


def test_symlink_images_and_directories_not_followed(tmp_path):
    source = tmp_path / 'source'; source.mkdir()
    dataset = tmp_path / 'dataset'; dataset.mkdir()
    a = image(source)
    try:
        (dataset / 'image.png').symlink_to(a)
        (dataset / 'sub').symlink_to(source, target_is_directory=True)
    except OSError:
        pytest.skip('Symlink creation is not permitted on this system')
    assert scan_suggestions(dataset).total == 0


def test_symlink_caption_not_read(tmp_path):
    a = image(tmp_path, caption=None)
    external = tmp_path / 'other.dat'; external.write_text('secret')
    try:
        a.with_suffix('.txt').symlink_to(external)
    except OSError:
        pytest.skip('Symlink creation is not permitted on this system')
    report = scan_suggestions(tmp_path)
    assert report.records[0].caption_state == 'unreadable'
    assert report.records[0].caption == ''


def test_large_caption_is_bounded(tmp_path):
    a = image(tmp_path)
    a.with_suffix('.txt').write_bytes(b'x' * (MAX_CAPTION_BYTES + 1))
    report = scan_suggestions(tmp_path)
    assert report.records[0].caption_state == 'unreadable'


def test_source_dataset_is_byte_identical_after_scan(tmp_path):
    image(tmp_path, 'a.png')
    image(tmp_path, 'b.png', caption=None)
    before = {str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    scan_suggestions(tmp_path, options=ScanOptions(exact_duplicates=True))
    after = {str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    assert before == after


def test_cancel_does_not_return_partial_complete_report(tmp_path):
    image(tmp_path)
    cancel = threading.Event(); cancel.set()
    with pytest.raises(ScanCancelled):
        scan_suggestions(tmp_path, cancel=cancel)


def test_cancel_during_duplicate_hash(tmp_path):
    image(tmp_path)
    cancel = threading.Event()
    from src.core import smart_suggestions as module
    real = module._digest
    def stop(*args):
        cancel.set()
        return real(*args)
    with patch.object(module, '_digest', stop):
        with pytest.raises(ScanCancelled):
            scan_suggestions(tmp_path, options=ScanOptions(exact_duplicates=True), cancel=cancel)


def test_scan_cache_reuses_metadata_but_rereads_captions(tmp_path):
    folder = tmp_path / 'dataset'; folder.mkdir()
    a = image(folder)
    cache = tmp_path / 'cache.sqlite3'
    first = scan_suggestions(folder, cache_path=cache, options=ScanOptions(exact_duplicates=True))
    a.with_suffix('.txt').write_text('new caption, master')
    second = scan_suggestions(folder, cache_path=cache, options=ScanOptions(exact_duplicates=True))
    assert first.cache_hits == 0 and second.cache_hits == 1
    assert second.records[0].caption == 'new caption, master'


def test_cache_invalidated_on_changed_image(tmp_path):
    folder = tmp_path / 'dataset'; folder.mkdir()
    a = image(folder)
    cache = tmp_path / 'cache.sqlite3'
    scan_suggestions(folder, cache_path=cache)
    image(folder, size=(333, 444))
    second = scan_suggestions(folder, cache_path=cache)
    assert second.cache_hits == 0 and second.records[0].width == 333


def test_unavailable_cache_is_not_scan_failure(tmp_path):
    folder = tmp_path / 'dataset'; folder.mkdir(); image(folder)
    bad = tmp_path / 'bad.sqlite3'; bad.write_bytes(b'not a database')
    report = scan_suggestions(folder, cache_path=bad)
    assert report.total == 1 and report.warnings


def test_image_modified_mid_scan_is_not_cached_or_duplicate(tmp_path):
    a = image(tmp_path)
    from src.core import smart_suggestions as module
    def mutating_digest(path, cancel):
        path.write_bytes(path.read_bytes() + b'extra')
        return 'arbitrary hash'
    with patch.object(module, '_digest', mutating_digest):
        report = scan_suggestions(tmp_path, options=ScanOptions(exact_duplicates=True))
    assert 'changed_image' in report.records[0].issues
    assert report.records[0].width == 0 and report.duplicate_extras == 0


@pytest.mark.parametrize('value', [-1, 9000, 1.1, True, '512'])
def test_invalid_dimension_option_rejected(tmp_path, value):
    with pytest.raises(ValueError):
        scan_suggestions(tmp_path, options=ScanOptions(minimum_side=value))


def test_missing_folder_is_error(tmp_path):
    with pytest.raises(ValueError):
        scan_suggestions(tmp_path / 'missing')


def test_progress_is_throttled_and_has_final_total(tmp_path):
    for i in range(12): image(tmp_path, f'{i}.png')
    calls = []
    scan_suggestions(tmp_path, progress=lambda *a: calls.append(a))
    assert calls[-1][:2] == (12, 12)
    assert len(calls) <= 12


def test_report_serializable_and_no_html_interpretation(tmp_path):
    image(tmp_path, 'less<than.png', caption='<b>literal</b>')
    report = scan_suggestions(tmp_path)
    payload = json.loads(json.dumps(report.to_dict()))
    assert payload['records'][0]['caption'] == '<b>literal</b>'
