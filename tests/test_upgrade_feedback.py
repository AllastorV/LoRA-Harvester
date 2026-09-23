import json
import threading
from pathlib import Path
from unittest.mock import Mock
import pytest
from PIL import Image
from src.core.clothing_profiles import ClothingProfileStore, ClothingProfile, ClothingPart
from src.core.clothing_feedback import ClothingFeedbackStore, selection_key
from src.core.clothing_io import file_digest
from src.core.clothing_captions import prepare_caption, apply_caption, undo_caption, CaptionConflict
from src.core.clothing_backend import image_base64
from src.core.clothing_service import run_clothing_batch, save_selection


@pytest.fixture
def setup(tmp_path):
    store=ClothingProfileStore(tmp_path/'library')
    ref=tmp_path/'ref.png';Image.new('RGB',(40,50),(40,70,100)).save(ref)
    p=ClothingProfile('Uniform','shhooldress',[ClothingPart('serafaku'),ClothingPart('blue skirt'),ClothingPart('black thighighhs')],
                      references=[store.import_reference(ref)],id='blue')
    store.upsert(p)
    target=tmp_path/'target.png';Image.new('RGB',(50,70),(60,80,120)).save(target)
    return store,p,target,ClothingFeedbackStore(store)


def remember(setup, tags=None, **kw):
    store,p,target,memory=setup
    return memory.remember(target,[0.,0.,1.,1.],profile_id=p.id,visible_tags=tags or ['serafaku','blue skirt'],
                            expected_image_digest=file_digest(target),**kw)


def test_master_and_visible_tags_are_literal_and_ordered(setup):
    result=remember(setup,['blue skirt','serafaku'])
    assert result.tags==['shhooldress','serafaku','blue skirt']
    assert result.decision_source=='human' and result.score==0
    assert result.parts[-1]['visibility']=='not_visible'


def test_remember_only_previews_and_apply_is_still_explicit(setup):
    store,p,target,memory=setup
    target.with_suffix('.txt').write_bytes(b'1girl, sitting\n')
    before=target.with_suffix('.txt').read_bytes()
    result=remember(setup);preview=prepare_caption(result)
    assert target.with_suffix('.txt').read_bytes()==before
    journal=apply_caption(preview,store)
    assert target.with_suffix('.txt').read_text().startswith('shhooldress, serafaku, blue skirt, 1girl')
    undo_caption(journal)
    assert target.with_suffix('.txt').read_bytes()==before


def test_exact_replay_requires_same_selection_image_and_library(setup):
    store,p,target,memory=setup;remember(setup)
    assert memory.exact(target,[0,0,1,1],store.load()).tags[:2]==['shhooldress','serafaku']
    assert memory.exact(target,[0,0,.7,1],store.load()) is None
    assert memory.exact(target,None,store.load()) is None
    rival=ClothingProfile('Red','redmaster',[ClothingPart('red skirt')],references=p.references,id='red')
    store.upsert(rival)
    assert memory.exact(target,[0,0,1,1],store.load()) is None


def test_changed_profile_invalidates_exact_and_examples(setup):
    store,p,target,memory=setup;remember(setup)
    p.notes='Changed distinctiveness';store.upsert(p)
    assert memory.exact(target,[0,0,1,1],store.load()) is None
    assert memory.examples(p,image_base64(Image.open(target)))==[]


def test_forget_invalidates_previews_without_removing_caption(setup):
    store,p,target,memory=setup;result=remember(setup);preview=prepare_caption(result)
    assert memory.forget(target,[0,0,1,1])==1
    assert memory.exact(target,[0,0,1,1],store.load()) is None
    assert memory.load()[0]['active'] is False
    with pytest.raises(CaptionConflict,match='Correction memory'):
        apply_caption(preview,store)


def test_bad_manual_tags_and_empty_selection_are_rejected(setup):
    store,p,target,memory=setup
    kwargs=dict(profile_id=p.id,expected_image_digest=file_digest(target))
    for tags in (['not_a_part'],[],[123],['serafaku','serafaku']):
        with pytest.raises(ValueError):memory.remember(target,[0,0,1,1],visible_tags=tags,**kwargs)
    with pytest.raises(ValueError):memory.remember(target,None,visible_tags=['serafaku'],**kwargs)


def test_no_match_can_teach_rejection_but_never_adds_master(setup):
    store,p,target,memory=setup
    result=memory.remember(target,[0,0,1,1],profile_id='',visible_tags=[],rejected_profile_id=p.id,
                           expected_image_digest=file_digest(target),note='Different collar')
    assert result.status=='no_match' and not result.tags
    examples=memory.examples(p,image_base64(Image.open(target)))
    assert len(examples)==1 and not examples[0]['label']['matches_this_profile']
    with pytest.raises(ValueError):prepare_caption(result)


def test_example_opt_out_retains_exact_annotation_only(setup):
    store,p,target,memory=setup;remember(setup,use_as_example=False)
    assert memory.exact(target,[0,0,1,1],store.load()) is not None
    assert memory.examples(p,image_base64(Image.open(target)))==[]


def test_remember_replaces_old_record_without_destroying_history(setup):
    store,p,target,memory=setup;remember(setup);remember(setup,['serafaku'])
    records=memory.load();assert len(records)==2 and not records[0]['active'] and records[1]['active']
    assert memory.exact(target,[0,0,1,1],store.load()).tags==['shhooldress','serafaku']


def test_exact_replay_in_batch_needs_no_ollama(setup):
    store,p,target,memory=setup;remember(setup)
    save_selection(store,target,[0,0,1,1])
    backend=Mock();backend.check.side_effect=AssertionError('Must not call model for explicit identical correction')
    seen=[]
    stats=run_clothing_batch([target],store=store,backend=backend,on_result=lambda *a:seen.append(a))
    assert stats['matched']==1 and stats['errors']==0
    assert seen[0][0].decision_source=='human'
    assert not backend.check.called
    assert not target.with_suffix('.txt').exists()


def test_stale_image_cannot_be_recorded(setup):
    store,p,target,memory=setup;digest=file_digest(target)
    Image.new('RGB',(50,70),(2,3,4)).save(target)
    with pytest.raises(ValueError,match='changed'):
        memory.remember(target,[0,0,1,1],profile_id=p.id,visible_tags=['serafaku'],expected_image_digest=digest)


def test_example_limit_and_corrupted_crop_are_safe(setup):
    store,p,target,memory=setup
    for i in range(4):
        other=target.with_name(f'img{i}.png');Image.new('RGB',(40,50),(i*30,50,60)).save(other)
        memory.remember(other,[0,0,1,1],profile_id=p.id,visible_tags=['serafaku'],expected_image_digest=file_digest(other))
    encoded=image_base64(Image.open(target));assert len(memory.examples(p,encoded))==2
    for record in memory.load():
        (memory.root/record['crop']).write_bytes(b'corrupt')
    assert memory.examples(p,encoded)==[]


def test_inference_includes_only_approved_examples(setup):
    from src.core.clothing_matcher import ClothingMatcher
    from src.core.clothing_profiles import ClothingSettings
    store,p,target,memory=setup;remember(setup)
    matcher=ClothingMatcher(store,ClothingSettings(),Mock())
    target_encoded=image_base64(Image.open(target))
    matcher._feedback_context={p.id:memory.examples(p,target_encoded)}
    response={'reference_single_outfit':True,'target_single_outfit':True,'outfit_visible':True,
              'verdict':'match','identity_score':.95,'visible_design_match':True,'visible_color_match':'yes','evidence':'visible collar',
              'parts':[{'id':f'p{i}','visibility':'visible' if i<2 else 'not_visible','score':.9,'evidence':'observation'} for i in range(3)]}
    matcher.vision.infer.return_value=response
    matcher._compare(target_encoded,p,target_encoded)
    prompt,images,schema=matcher.vision.infer.call_args.args
    assert len(images)==3 and 'HUMAN-CORRECTED' in prompt and 'blue skirt' in prompt
    assert 'black thighighhs' in prompt  # requested parts, not necessarily visible


def test_persisted_corrupt_memory_fails_closed(setup):
    store,p,target,memory=setup;memory.root.mkdir(parents=True)
    memory.path.write_text('{"schema_version":9,"records":[]}',encoding='utf-8')
    with pytest.raises(ValueError,match='Unsupported'):
        memory.exact(target,[0,0,1,1],store.load())
