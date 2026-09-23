import json
from pathlib import Path
import threading
import pytest
from PIL import Image
from src.core.clothing_profiles import ClothingProfileStore, ClothingProfile, ClothingPart
from src.core.dataset_balance import (caption_labels, scan_balance, make_plan, export_plan,
    save_override, validate_plan, BalanceCancelled, UNKNOWN, AMBIGUOUS)


def image(root, name, text='uniform, standing, from above', color=(35, 65, 95)):
    root.mkdir(parents=True, exist_ok=True)
    path = root / name
    Image.new('RGB', (32, 40), color).save(path)
    if text is not None:
        path.with_suffix('.txt').write_text(text, encoding='utf-8')
    return path


@pytest.fixture
def library(tmp_path):
    store = ClothingProfileStore(tmp_path / 'library')
    store.upsert(ClothingProfile('Uniform', 'uniform', [ClothingPart('blue skirt')], id='blue'))
    store.upsert(ClothingProfile('Red', 'uniform_red', [ClothingPart('red skirt')], id='red'))
    return store


def test_tags_normalize_without_guessing_pose_or_front():
    assert caption_labels('1girl, blue_skirt', ['uniform']) == dict(outfit=UNKNOWN, pose=UNKNOWN, angle=UNKNOWN)
    assert caption_labels('uniform, standing, from_above, from_behind', ['uniform']) == dict(outfit='uniform', pose='standing', angle='back+high')


def test_conflicting_masters_do_not_choose_first():
    assert caption_labels('uniform, uniform_red, sitting, standing', ['uniform', 'uniform_red'])['outfit'] == AMBIGUOUS
    assert caption_labels('from above, from below', [])['angle'] == AMBIGUOUS


def test_specific_motion_and_lying_aliases():
    assert caption_labels('walking, standing', [])['pose'] == 'walking'
    assert caption_labels('lying, on_stomach', [])['pose'] == 'lying'


def test_scan_skips_helper_directories_and_counts_unknown(tmp_path, library):
    root = tmp_path / 'dataset'
    image(root, 'a.png')
    image(root, 'b.png', '1girl', (50, 20, 55))
    image(root / '.lh-dataset', 'private.png')
    report = scan_balance(root, store=library)
    assert len(report['items']) == 2
    assert report['summary']['outfit'] == {'uniform': 1, 'unknown': 1}


def test_no_fake_balance_from_unlabeled_data(tmp_path, library):
    root=tmp_path/'data'; image(root,'a.png','1girl')
    report=scan_balance(root,store=library)
    assert make_plan(report)['selected_count'] == 0
    assert make_plan(report, include_unknown=True)['selected_count'] == 1


def test_exact_duplicate_with_same_caption_selects_one(tmp_path, library):
    root=tmp_path/'data'; image(root,'a.png'); image(root,'b.png')
    report=scan_balance(root,store=library)
    assert report['summary']['exact_duplicates'] == 1
    assert make_plan(report)['selected_count'] == 1
    assert make_plan(report,deduplicate=False)['selected_count'] == 2


def test_duplicate_with_conflicting_caption_excludes_both(tmp_path, library):
    root=tmp_path/'data'; image(root,'a.png'); image(root,'b.png','uniform, sitting, from above')
    report=scan_balance(root,store=library)
    assert all('duplicate_label_conflict' in i.issues for i in report['items'])
    assert make_plan(report)['selected_count'] == 0


def test_same_stem_formats_are_not_exported(tmp_path, library):
    root=tmp_path/'data'; image(root,'a.png'); image(root,'a.jpg',color=(20,20,20))
    report=scan_balance(root,store=library)
    assert make_plan(report)['selected_count'] == 0
    assert all('same_stem_caption_conflict' in i.issues for i in report['items'])


def test_stratified_sampling_is_deterministic_and_never_oversamples(tmp_path, library):
    root=tmp_path/'data'
    for i in range(10):
        image(root,f'{i:02}.png',('uniform' if i < 7 else 'uniform_red')+', standing, from above',(i*20,8,9))
    report=scan_balance(root,store=library)
    one=make_plan(report,dimension='outfit',seed=7)
    two=make_plan(report,dimension='outfit',seed=7)
    assert one == two and one['selected_count']==6
    assert all(b['selected']==3 for b in one['buckets'])
    plan=make_plan(report,dimension='outfit',per_group=9)
    assert plan['selected_count']==10 and sum(b['shortfall'] for b in plan['buckets'])==8


def test_manual_override_does_not_edit_caption_and_goes_stale(tmp_path, library):
    root=tmp_path/'data';path=image(root,'a.png','1girl')
    report=scan_balance(root,store=library);item=report['items'][0]
    original=path.with_suffix('.txt').read_bytes()
    labels=dict(outfit='uniform',pose='sitting',angle='front')
    save_override(root,item,labels)
    after=scan_balance(root,store=library)
    assert after['items'][0].overridden and after['items'][0].pose=='sitting'
    assert path.with_suffix('.txt').read_bytes()==original
    path.with_suffix('.txt').write_text('1girl, walking',encoding='utf-8')
    stale=scan_balance(root,store=library)
    assert 'stale_override' in stale['items'][0].issues
    assert not stale['items'][0].overridden


def test_manual_change_invalidates_existing_plan(tmp_path, library):
    root=tmp_path/'data'; image(root,'a.png');report=scan_balance(root,store=library)
    plan=make_plan(report)
    save_override(root,report['items'][0],dict(outfit='uniform',pose='sitting',angle='front'))
    with pytest.raises(ValueError,match='Manual labels changed'):
        validate_plan(plan)


def test_override_rejects_stale_image_or_invalid_label(tmp_path, library):
    root=tmp_path/'data';path=image(root,'a.png'); item=scan_balance(root,store=library)['items'][0]
    with pytest.raises(ValueError):
        save_override(root,item,dict(outfit='uniform',pose='invented pose',angle='front'))
    image(root,'a.png',color=(90,90,90))
    with pytest.raises(ValueError,match='changed'):
        save_override(root,item,dict(outfit='uniform',pose='sitting',angle='front'))


def test_export_preserves_image_and_sidecars_without_overwrite(tmp_path, library):
    root=tmp_path/'data';path=image(root,'a.png')
    path.with_suffix('.json').write_bytes(b'{"source":"a"}')
    originals={str(p):p.read_bytes() for p in root.iterdir() if p.is_file()}
    plan=make_plan(scan_balance(root,store=library))
    result=export_plan(plan,tmp_path/'export')
    out=next((tmp_path/'export'/'images').glob('*.png'))
    assert result['images']==1
    assert out.read_bytes()==path.read_bytes()
    assert out.with_suffix('.txt').read_bytes()==path.with_suffix('.txt').read_bytes()
    assert out.with_suffix('.json').read_bytes()==path.with_suffix('.json').read_bytes()
    for p,data in originals.items(): assert Path(p).read_bytes()==data
    with pytest.raises(ValueError,match='NEW output'):
        export_plan(plan,tmp_path/'export')


def test_export_refuses_source_nesting_and_changed_caption(tmp_path, library):
    root=tmp_path/'data';path=image(root,'a.png');plan=make_plan(scan_balance(root,store=library))
    with pytest.raises(ValueError,match='separate'):
        export_plan(plan,root/'export')
    path.with_suffix('.txt').write_text('changed',encoding='utf-8')
    with pytest.raises(ValueError,match='Dataset changed'):
        export_plan(plan,tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_export_cancellation_removes_only_staging(tmp_path, library):
    root=tmp_path/'data';image(root,'a.png');image(root,'b.png',color=(90,70,90))
    plan=make_plan(scan_balance(root,store=library));cancel=threading.Event()
    with pytest.raises(BalanceCancelled):
        export_plan(plan,tmp_path/'out',cancel=cancel,progress=lambda *a:cancel.set())
    assert not (tmp_path/'out').exists()
    assert not list(tmp_path.glob('.lh-balance-*'))
    assert len(list(root.glob('*.png')))==2


def test_tampered_plan_is_refused(tmp_path, library):
    root=tmp_path/'data';image(root,'a.png');plan=make_plan(scan_balance(root,store=library))
    plan['selected'][0]['outfit']='tampered'
    with pytest.raises(ValueError,match='invalid'):
        validate_plan(plan)


def test_scan_cancellation_and_missing_caption(tmp_path, library):
    root=tmp_path/'data';image(root,'a.png',None)
    report=scan_balance(root,store=library)
    assert report['summary']['issues']['missing_caption']==1
    assert make_plan(report,include_unknown=True)['selected_count']==0
    flag=threading.Event();flag.set()
    with pytest.raises(BalanceCancelled):scan_balance(root,store=library,cancel=flag)


def test_symlink_override_storage_is_rejected(tmp_path, library):
    root=tmp_path/'data';image(root,'a.png');target=tmp_path/'external';target.mkdir()
    (root/'.lh-dataset').symlink_to(target,target_is_directory=True)
    with pytest.raises(ValueError,match='metadata'):
        scan_balance(root,store=library)
