"""Live adapter with actual editor source methods + controlled Qt-like objects.

This is not a Qt signal/painting or real model test. It executes real file writes,
settings bindings and job/error handling without requiring GPU or PyQt.
"""
from __future__ import annotations
import ast
import base64
import io
import json
import sys
import threading
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace, MethodType
from unittest.mock import Mock

import pytest
from PIL import Image
from src.core.ai_control import Broker, ControlError, CATALOG, digest
from src.core.ai_mcp import MCPServer
from src.core.ai_client import LiveClient
from src.core.ai_control import ControlServer
from src.core.caption_sync import path_key, read_caption_snapshot
from src.core.clothing_profiles import ClothingProfileStore, ClothingSettings, ClothingProfile, ClothingPart
from src.core.clothing_io import file_digest
from src.core.clothing_feedback import ClothingFeedbackStore
from src.core.clothing_captions import prepare_caption, apply_caption
from src.ui.ai_live_actions import LiveActions, DEFERRED, jsonable

ROOT=Path(__file__).resolve().parents[1]

class Signal:
    def __init__(self): self.slots=[]; self.emissions=[]
    def connect(self,fn,*args): self.slots.append(fn)
    def emit(self,*args):
        self.emissions.append(args)
        for fn in list(self.slots): fn(*args)

class Control:
    def __init__(self, value='',lo=0,hi=100): self.v=value; self.lo=lo; self.hi=hi; self.enabled=True
    def text(self):return str(self.v)
    def toPlainText(self):return str(self.v)
    def setText(self,v):self.v=v
    def setPlainText(self,v):self.v=v
    def blockSignals(self,v):pass
    def value(self):return self.v
    def setValue(self,v):self.v=max(self.lo,min(self.hi,v))
    def minimum(self):return self.lo
    def maximum(self):return self.hi
    def isChecked(self):return bool(self.v)
    def setChecked(self,v):self.v=bool(v)
    def isEnabled(self):return self.enabled
    def setEnabled(self,v):self.enabled=v
    def currentData(self):return self.v
    def currentText(self):return str(self.v)
    def findData(self,v):return v
    def setCurrentIndex(self,v):self.v=v
    def setCurrentRow(self,v):self.v=v
    def update(self):pass


def bind_source(instance, file, cls, methods, extras=None):
    tree=ast.parse((ROOT/file).read_text())
    node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls)
    env={'Path':Path,'path_key':path_key,'read_caption_snapshot':read_caption_snapshot}
    if extras:env.update(extras)
    for name in methods:
        method=next(n for n in node.body if isinstance(n,ast.FunctionDef) and n.name==name)
        module=ast.Module(body=[method],type_ignores=[])
        exec(compile(ast.fix_missing_locations(module),file,'exec'),env)
        setattr(instance,name,MethodType(env[name],instance))

@pytest.fixture
def setup(tmp_path):
    root=tmp_path/'dataset';root.mkdir()
    image=root/'örnek.png';other=root/'other.png'
    for p in (image,other):
        Image.new('RGB',(32,48),(90,40,15)).save(p);p.with_suffix('.txt').write_bytes(b'1girl, standing\r\n')
    items=[(str(p),str(p.with_suffix('.txt'))) for p in (image,other)]
    snapshots={cp:read_caption_snapshot(ip) for ip,cp in items}
    edit=SimpleNamespace(_folder=str(root),_items=items,_current_idx=0,_captions={cp:snap.text for cp,snap in snapshots.items()},
        _snapshots=snapshots,_dirty_paths=set(),_dirty=False,_read_errors={},caption_edit=Control(snapshots[items[0][1]].text),
        image_list=Control(),captions_saved=Signal(),status_lbl=Control(),lang='tr',_build_chips=Mock())
    def refresh():edit.caption_edit.setPlainText(edit._captions[edit._items[edit._current_idx][1]])
    edit._refresh_editor=refresh
    bind_source(edit,'src/ui/caption_studio_page.py','_EditTab',['_mark_dirty','_commit_current','has_unsaved_edits','_set_caption_text','refresh_saved_captions'])
    store=ClothingProfileStore(tmp_path/'library')
    clothing=SimpleNamespace(store=store,_paths=[str(image),str(other)],_results={},_previews={},_active_image=str(image),
        is_busy=lambda:False,has_unsaved_profile_edits=lambda:False,_read_settings=lambda:ClothingSettings(),
        _invalidate_previews=Mock(),ensure_image=Mock(),_reload_profiles=Mock(),canvas=Control(),stop=Mock(),worker=None)
    gen=SimpleNamespace(selected_folder=str(root),captioning_thread=None)
    gen.trigger_edit=Control('');gen.suffix_edit=Control('');gen._last_words_edit=Control('')
    gen.max_tags_spin=Control(25,5,100);gen._max_tags_vis=Control(25,5,150)
    gen.conf_spin=Control(.35,0,1);gen._wd14_sl=Control(35,0,100)
    gen.neg_edit=Control('');gen._neg_prompt_edit=Control('');gen.wd14_cb=Control(True);gen._wd14_vis_cb=Control(True)
    gen._f2_vis_cb=Control(False);gen.keep_char_cb=Control(True);gen.json_cb=Control(False)
    gen.overwrite_cb=Control(False);gen._overwrite_rb=Control(False);gen.recursive_cb=Control(False)
    gen.wd14_combo=Control('SmilingWolf/wd-swinv2-tagger-v3');gen.f2_combo=Control('florence-2-base');gen.f2_task_combo=Control('<DETAILED_CAPTION>')
    bind_source(gen,'src/ui/caption_studio_page.py','_GenerateTab',['get_settings'],{'Dict':dict})
    balance=SimpleNamespace(folder=Control(str(root)),recursive=Control(True),dimension=Control('outfit'),target=Control(0,0,100000),
        seed=Control(42,0,2147483647),unknown=Control(False),dedupe=Control(True),report=None,plan=None,_thread=None,stop=Mock())
    studio=SimpleNamespace(edit_tab=edit,clothing_tab=clothing,generate_tab=gen,balance_tab=balance,any_tool_busy=lambda:False)
    w=SimpleNamespace(caption_studio_page=studio,_route='edit',_close_pending=False,_writers_active=lambda:False,_thread_running=lambda:False,
        _scan_worker=None,_VIDEO_EXTENSIONS={'.mp4','.mkv','.avi'},video_paths=[],_output_path_lbl=Control(str(tmp_path)),interval_slider=Control(30,1,300),conf_spinbox=Control(35,1,100),
        padding_spinbox=Control(10,0,150),turbo_cb=Control(True),nsfw_cb=Control(True),ratio_combo=Control('1:1'),detection_mode_combo=Control('yolo'),
        ensemble_cb=Control(False),voting_spinbox=Control(1,1,3),skip_subtitle_cb=Control(False),subtitle_removal_cb=Control(False),
        trim_start_spin=Control(0),trim_end_spin=Control(0),_resource_cfg={'gpu':True},page_stack=Control(),processing_thread=None)
    for name in ('quality_panel','caption_panel','tags_panel','upscale_panel'):
        setattr(w,name,SimpleNamespace(get_settings=lambda:{'enabled':False}))
    w.navigate=lambda route:setattr(w,'_route',route)
    w.on_files_dropped=lambda paths:setattr(w,'video_paths',paths)
    b=Broker();b.configure(enabled=True,mode='delegated',roots=[tmp_path])
    live=LiveActions(w,b)
    def command(action,args=None,id=None):
        r=b.submit(action,args or {},id)
        if r['status']=='pending_approval': b.approve(r['id'],True)
        request=b.take()
        try:
            result=live.dispatch(request)
            if result is not DEFERRED:b.finish(r['id'],result)
        except Exception as exc:
            b.finish(r['id'],error={'code':getattr(exc,'code','failed'),'message':str(exc)})
            raise
        return result
    return SimpleNamespace(root=root,image=image,other=other,edit=edit,clothing=clothing,gen=gen,balance=balance,w=w,b=b,live=live,command=command)


def test_all_catalog_actions_have_real_adapter_handlers(setup):
    assert set(CATALOG)==set(setup.live.handlers)


def test_live_state_and_shared_image_list(setup):
    state=setup.command('app.state')
    assert state['dataset']==str(setup.root) and state['selected_image']==str(setup.image)
    rows=setup.command('dataset.list',{'limit':1})
    assert rows['total']==2 and rows['next_offset']==1


def test_non_gui_dispatch_is_refused(setup):
    errors=[]
    def call():
        try:setup.live.dispatch({'action':'app.state','arguments':{},'id':'x'})
        except ControlError as e:errors.append(e.code)
    t=threading.Thread(target=call);t.start();t.join()
    assert errors==['thread']


def test_caption_stage_preserves_disk_and_save_emits_existing_sync(setup):
    s=setup;before=s.image.with_suffix('.txt').read_bytes()
    read=s.command('caption.read',{'image':str(s.image)})
    r=s.command('caption.set',{'image':str(s.image),'text':'master, blue skirt','expected_revision':read['revision']})
    assert r['dirty'] and not r['saved_to_disk'] and not s.edit.captions_saved.emissions
    assert s.image.with_suffix('.txt').read_bytes()==before
    saved=s.command('caption.set',{'image':str(s.image),'text':r['draft'],'expected_revision':r['revision'],'save':True})
    assert saved['journal'] and not saved['dirty']
    assert s.edit.caption_edit.toPlainText()=='master, blue skirt'
    assert s.edit.captions_saved.emissions==[([str(s.image.with_suffix('.txt'))],)]
    r=s.command('caption.undo',{'image':str(s.image),'journal':saved['journal']})
    assert s.image.with_suffix('.txt').read_bytes()==before and not r['dirty']

@pytest.mark.parametrize('changed', ['disk','draft','source'])
def test_live_revision_conflicts_prevent_overwrite(setup,changed):
    s=setup;old=s.command('caption.read',{'image':str(s.image)})
    if changed=='disk':s.image.with_suffix('.txt').write_text('human on disk')
    elif changed=='draft':s.edit.caption_edit.setPlainText('human draft')
    else:Image.new('RGB',(64,80)).save(s.image)
    with pytest.raises(ControlError) as exc:s.command('caption.set',{'image':str(s.image),'text':'bad','expected_revision':old['revision'],'save':True})
    assert exc.value.code=='stale_revision'
    assert 'bad' not in s.image.with_suffix('.txt').read_text()


def test_fresh_token_cannot_override_editor_stale_disk_baseline(setup):
    s=setup;s.image.with_suffix('.txt').write_text('external')
    fresh=s.command('caption.read',{'image':str(s.image)})
    with pytest.raises(ControlError) as exc:s.command('caption.set',{'image':str(s.image),'text':'bad','expected_revision':fresh['revision'],'save':True})
    assert exc.value.code=='stale_editor'


def test_caption_noncurrent_save_preserves_selected_human_draft(setup):
    s=setup;s.edit.caption_edit.setPlainText('human current draft')
    r=s.command('caption.read',{'image':str(s.other)})
    s.command('caption.set',{'image':str(s.other),'text':'other changed','expected_revision':r['revision'],'save':True})
    assert s.edit.caption_edit.toPlainText()=='human current draft'
    assert s.other.with_suffix('.txt').read_text()=='other changed'


def test_dataset_open_refuses_unsaved_edits_before_loader(setup):
    s=setup;s.edit.caption_edit.setPlainText('draft');s.edit.reload_folder=Mock()
    with pytest.raises(ControlError,match='drafts'):s.command('dataset.open',{'folder':str(s.root)})
    s.edit.reload_folder.assert_not_called()


def test_dataset_tree_preflight_rejects_redirected_caption(setup,tmp_path):
    s=setup;external=tmp_path/'secret.txt';external.write_text('secret')
    s.image.with_suffix('.txt').unlink();s.image.with_suffix('.txt').symlink_to(external)
    s.edit.reload_folder=Mock()
    with pytest.raises(ControlError,match='redirected'):s.command('dataset.open',{'folder':str(s.root)})
    s.edit.reload_folder.assert_not_called()


def test_ungiven_dataset_grant_hides_paths(setup,tmp_path):
    s=setup;grant=tmp_path/'else';grant.mkdir();s.b.configure(enabled=True,mode='review',roots=[grant])
    state=s.command('app.state')
    assert state['dataset'] is None and state['selected_image'] is None and state['dataset_requires_grant']
    with pytest.raises(ControlError):s.command('caption.read',{'image':str(s.image)})


def test_settings_fingerprint_prevents_unseen_config_change(setup):
    s=setup;rev=s.command('workflow.settings')['revision'];s.w.interval_slider.setValue(10)
    with pytest.raises(ControlError,match='Settings changed'):
        s.command('video.configure',{'expected_revision':rev,'nsfw':False})
    assert s.w.nsfw_cb.isChecked()


def test_video_config_validates_all_before_mutating(setup):
    s=setup;rev=s.command('workflow.settings')['revision']
    with pytest.raises(ControlError,match='supported range'):
        s.command('video.configure',{'expected_revision':rev,'frame_interval':9999,'nsfw':False})
    assert s.w.interval_slider.value()==30 and s.w.nsfw_cb.isChecked()


def test_video_config_updates_real_widgets_without_starting(setup,tmp_path):
    s=setup;v=tmp_path/'video.mp4';v.write_bytes(b'test fixture, not a codec')
    rev=s.command('workflow.settings')['revision']
    r=s.command('video.configure',{'expected_revision':rev,'videos':[str(v)],'frame_interval':20,'nsfw':False,'turbo':True,'output_dir':str(tmp_path)})
    assert r['settings']['video']['videos']==[str(v)] and s.w.interval_slider.value()==20
    assert not s.w.nsfw_cb.isChecked() and s.w.processing_thread is None


def test_caption_config_visible_controls_are_source_of_truth(setup):
    s=setup;rev=s.command('workflow.settings')['revision']
    result=s.command('caption.configure',{'expected_revision':rev,'max_tags':120,'confidence_percent':45,'trigger_word':'shhooldress','negative_tags':['watermark','text'],'overwrite':True})
    config=result['settings']['caption']['settings']
    assert config['max_tags']==120 and config['min_confidence']==.45
    assert config['trigger_word']=='shhooldress' and config['negative_tags']==['watermark','text'] and config['overwrite']
    assert s.gen.captioning_thread is None


def test_balancing_controls_follow_settings_revision(setup):
    s=setup;rev=s.command('workflow.settings')['revision']
    r=s.command('balance.configure',{'expected_revision':rev,'dimension':'pose','per_group':30,'seed':8})
    assert r['settings']['balance']['dimension']=='pose' and s.balance.target.value()==30 and s.balance.seed.value()==8


def test_clothing_area_selection_invalidates_only_that_preview(setup):
    s=setup;r=s.command('clothing.selection',{'image':str(s.image),'bbox':[.1,.2,.8,.9]})
    assert tuple(r['bbox'])==(.1,.2,.8,.9)
    s.clothing._invalidate_previews.assert_called_once_with(str(s.image))
    assert tuple(s.clothing.canvas.bbox)==(.1,.2,.8,.9)


def test_invalid_clothing_box_does_not_invalidate_existing_preview(setup):
    s=setup
    with pytest.raises(ValueError):s.command('clothing.selection',{'image':str(s.image),'bbox':[.9,.2,.1,.9]})
    s.clothing._invalidate_previews.assert_not_called()


def test_outfit_preview_revision_and_apply_retain_original_master_and_visible_tags(setup,monkeypatch):
    s=setup;store=s.clothing.store
    p=ClothingProfile('Blue','shhooldress',[ClothingPart('serafaku'),ClothingPart('blue skirt'),ClothingPart('black thighighhs')],references=[store.import_reference(s.other)],id='blue')
    store.upsert(p)
    memory=ClothingFeedbackStore(store)
    decision=memory.remember(s.image,[0,0,1,1],profile_id='blue',visible_tags=['serafaku','blue skirt'],expected_image_digest=file_digest(s.image))
    preview=prepare_caption(decision)
    s.clothing._results[str(s.image)]=decision;s.clothing._previews[str(s.image)]=preview
    result=s.command('clothing.results',{'images':[str(s.image)]})['results'][0]
    assert result['preview']['tags']==['shhooldress','serafaku','blue skirt']
    def start(operation,**kwargs):
        assert operation=='apply'
        for item in kwargs['previews']:apply_caption(item,store)
        s.edit.refresh_saved_captions([str(s.image.with_suffix('.txt'))])
    s.clothing._start=start
    monkeypatch.setattr(s.live,'_start_observed',lambda rid,owner,start,*a,**kw: (start(), {'applied':True})[1])
    s.command('clothing.apply',{'items':[{'image':str(s.image),'expected_revision':result['preview']['revision']}]})
    assert s.edit.caption_edit.toPlainText().startswith('shhooldress, serafaku, blue skirt')
    assert 'black thighighhs' not in s.edit.caption_edit.toPlainText()


def test_missing_or_changed_outfit_preview_refuses_apply(setup):
    with pytest.raises(ControlError,match='preview changed'):
        setup.command('clothing.apply',{'items':[{'image':str(setup.image),'expected_revision':'0'*64}]})


def test_image_sharing_is_separately_opted_in(setup):
    with pytest.raises(ControlError) as exc:setup.command('image.preview',{'image':str(setup.image)})
    assert exc.value.code=='image_sharing_disabled'


def test_preview_real_pil_decode_and_bounded_dimensions(setup,monkeypatch):
    s=setup;s.b.share_images=True
    Image.new('RGB',(1600,900),(70,120,50)).save(s.image)
    def background(rid,work,*args,**kwargs):return work(threading.Event(),lambda *a:None)
    monkeypatch.setattr(s.live,'_background',background)
    result=s.command('image.preview',{'image':str(s.image),'max_edge':256})
    decoded=Image.open(io.BytesIO(base64.b64decode(result['image']['data'])))
    assert decoded.width==256 and decoded.height<=256
    assert Image.open(s.image).size==(1600,900)


def test_preview_queue_refuses_parallel_model_or_decode_work(setup):
    s=setup;s.b.share_images=True;s.live.active['existing']={}
    with pytest.raises(ControlError) as exc:s.command('image.preview',{'image':str(s.image)})
    assert exc.value.code=='busy'


def test_no_workflow_overlap_with_existing_ui_job(setup):
    s=setup;rev=s.command('workflow.settings')['revision'];s.w._writers_active=lambda:True
    with pytest.raises(ControlError,match='job is running'):s.command('workflow.start',{'tool':'video','expected_revision':rev})


def test_cancel_only_owned_worker_and_preserves_completed_writes(setup):
    s=setup;r=s.b.submit('dataset.scan',{},'own')
    if r['status']=='pending_approval':s.b.approve('own',True)
    s.b.take();stop=Mock();s.live.active['own']={'stop':stop,'cancel_requested':False}
    result=s.live.cancel({'request_id':'own'},'cancel')
    assert result['status']=='cancelling' and s.live.active['own']['cancel_requested']
    stop.assert_called_once()
    with pytest.raises(ControlError):s.live.cancel({'request_id':'someone_else'},'cancel')


def test_worker_tracks_success_only_after_native_finished(setup,monkeypatch):
    s=setup;qt=SimpleNamespace(Qt=SimpleNamespace(QueuedConnection=2))
    monkeypatch.setitem(sys.modules,'PyQt5.QtCore',qt)
    r=s.b.submit('app.state',{},'job');s.b.take()
    worker=SimpleNamespace(completed=Signal(),failed=Signal(),progress=Signal(),finished=Signal())
    s.live._track('job',worker,'completed','failed','progress',Mock())
    worker.progress.emit(1,2,'working');worker.completed.emit({'written':1})
    assert s.b.status('job')['status']=='running'
    worker.finished.emit()
    assert s.b.status('job')['status']=='succeeded' and not s.live.active

@pytest.mark.parametrize('data,error,expected', [
    (None,None,'failed'), ({'written':1,'errors':['disk full']},None,'failed'),
    ({'clothing_error':'model not loaded'},None,'failed'), ({'written':0},None,'succeeded'),
    ({'written':1},{'code':'test','message':'failed'},'failed'),
])
def test_worker_terminal_status_is_not_fabricated(setup,data,error,expected):
    s=setup;s.b.submit('app.state',{},'job');s.b.take()
    s.live.active['job']={'data':data,'error':error,'cancel_requested':False}
    s.live._finished('job')
    assert s.b.status('job')['status']==expected
    if isinstance(data,dict):assert s.b.status('job')['result']==data


def test_full_loopback_mcp_caption_flow_with_real_files_and_original_edit_methods(setup,tmp_path):
    s=setup;server=ControlServer(s.b,tmp_path/'.ai-control'/'session.json');server.start()
    try:
        mcp=MCPServer(server.session_file);mcp.initialized=True
        original_call=mcp.client.call
        mcp.client.call=lambda action,args,request_id=None:original_call(action,args,request_id,wait=0)
        def step(name,args):
            reply=mcp.handle({'jsonrpc':'2.0','id':1,'method':'tools/call','params':{'name':name,'arguments':args}})
            rid=reply['result']['structuredContent']['id']
            request=s.b.take(); result=s.live.dispatch(request);s.b.finish(rid,result)
            return mcp.handle({'jsonrpc':'2.0','id':2,'method':'tools/call','params':{'name':'harvester_request_status','arguments':{'request_id':rid}}})['result']['structuredContent']
        state=step('harvester_caption_read',{'image':str(s.image),'command_id':'read1'})
        saved=step('harvester_caption_set',{'image':str(s.image),'expected_revision':state['result']['revision'],'text':'shhooldress, blue skirt','save':True,'command_id':'write1'})
        assert saved['status']=='succeeded' and s.edit.caption_edit.toPlainText()=='shhooldress, blue skirt'
        assert s.image.with_suffix('.txt').read_text()=='shhooldress, blue skirt'
        assert s.edit.captions_saved.emissions
        retry=mcp.client.call('caption.set',{'image':str(s.image),'expected_revision':state['result']['revision'],'text':'shhooldress, blue skirt','save':True},'write1')
        assert retry['status']=='succeeded' and s.b.take() is None
        undone=step('harvester_caption_undo',{'image':str(s.image),'journal':saved['result']['journal']})
        assert undone['status']=='succeeded' and s.image.with_suffix('.txt').read_bytes()==b'1girl, standing\r\n'
    finally:server.stop()


def test_completed_command_is_not_reported_as_cancelled(setup):
    s=setup;s.command('app.state',id='done')
    r=s.command('workflow.cancel',{'request_id':'done'})
    assert r['status']=='succeeded' and r['already_finished'] and 'cancelled_before_start' not in r


def test_clothing_config_does_not_persist_unrelated_user_auto_write_change(setup):
    s=setup
    changed=ClothingSettings(auto_after_video=True)
    s.clothing._read_settings=lambda:changed
    r=s.command('workflow.settings')
    with pytest.raises(ControlError) as exc:
        s.command('clothing.configure',{'expected_revision':r['revision'],'person_mode':'manual'})
    assert exc.value.code=='unsaved_settings'
    assert not s.clothing.store.load_settings().auto_after_video


def test_clothing_config_saved_fields_and_identity_preserved(setup):
    s=setup;p=s.clothing
    p.identity_spin=Control(.85,.05,1);p.part_spin=Control(.8,.05,1);p.person_combo=Control('largest')
    p.cache_check=Control(True);p.unload_check=Control(True)
    def read():return ClothingSettings(identity_threshold=p.identity_spin.value(),part_threshold=p.part_spin.value(),
        person_mode=p.person_combo.currentData(),use_cache=p.cache_check.isChecked(),unload_after_job=p.unload_check.isChecked())
    p._read_settings=read;p.store.save_settings(read());p._persist_settings=lambda:p.store.save_settings(read())
    revision=s.command('workflow.settings')['revision']
    result=s.command('clothing.configure',{'expected_revision':revision,'person_mode':'manual','identity_threshold':.9})
    assert p.store.load_settings().person_mode=='manual' and p.store.load_settings().identity_threshold==.9
    assert not p.store.load_settings().auto_after_video and result['settings']['clothing']['model']=='qwen3-vl:4b'


def test_video_failed_to_create_worker_is_controlled_error(setup,tmp_path):
    s=setup;v=tmp_path/'video.mp4';v.write_bytes(b'fixture');s.w.video_paths=[str(v)]
    s.w.start_processing=Mock()
    rev=s.command('workflow.settings')['revision']
    with pytest.raises(ControlError) as exc:s.command('workflow.start',{'tool':'video','expected_revision':rev})
    assert exc.value.code=='not_started'


def test_panel_timer_prevents_reentrant_modal_dispatch(setup):
    s=setup;panel=SimpleNamespace(window_ref=s.w,_dispatching=True,broker=s.b,live=s.live)
    bind_source(panel,'src/ui/ai_control_panel.py','AIControlPanel',['_tick'],{'DEFERRED':DEFERRED})
    s.b.submit('app.state',{},'pending')
    panel._tick()
    assert s.b.status('pending')['status']=='queued'


def test_both_ui_entrypoints_attach_the_same_optional_panel():
    for file in ('src/ui/main_window.py','src/ui/main_window_v2.py'):
        tree=ast.parse((ROOT/file).read_text())
        function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='create_app')
        calls=[n for n in ast.walk(function) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='attach_ai_control']
        assert len(calls)==1


def test_fast_worker_completion_is_connected_before_start(setup,monkeypatch):
    s=setup;monkeypatch.setitem(sys.modules,'PyQt5.QtCore',SimpleNamespace(Qt=SimpleNamespace(QueuedConnection=2)))
    s.b.submit('app.state',{},'fast');s.b.take()
    worker=SimpleNamespace(completed=Signal(),failed=Signal(),progress=Signal(),finished=Signal(),stop=Mock(),
                           isRunning=lambda:False,isFinished=lambda:True)
    owner=SimpleNamespace()
    def start():
        owner._ai_before_worker_start(worker)
        worker.completed.emit({'count':0});worker.finished.emit()
    assert s.live._start_observed('fast',owner,start,'completed','failed','progress') is DEFERRED
    assert s.b.status('fast')['status']=='succeeded'
    assert not hasattr(owner,'_ai_before_worker_start')


def test_real_controller_source_hooks_are_before_worker_start():
    cases=[('src/ui/main_window.py','VideoSmartCropperUI','start_processing'),
           ('src/ui/caption_studio_page.py','_GenerateTab','start_captioning'),
           ('src/ui/clothing_tagger_widget.py','ClothingTaggerWidget','_start'),
           ('src/ui/studio_tasks.py','StudioTaskWidget','start_task'),
           ('src/ui/main_window_v2.py','StudioMainWindow','start_scan')]
    for file,cls,method in cases:
        root=ast.parse((ROOT/file).read_text())
        c=next(n for n in root.body if isinstance(n,ast.ClassDef) and n.name==cls)
        fn=next(n for n in c.body if isinstance(n,ast.FunctionDef) and n.name==method)
        text=ast.get_source_segment((ROOT/file).read_text(),fn)
        assert "'_ai_before_worker_start'" in text
        assert text.index('observer(')<text.rfind('.start(')
