"""Execute selected actual UI methods against small fakes; not a Qt render test."""
import ast
import logging
from pathlib import Path
from types import SimpleNamespace, MethodType

ROOT=Path(__file__).resolve().parents[1]


def method(file, klass, name, env=None):
    module=ast.parse((ROOT/file).read_text(encoding='utf-8'))
    cls=next(node for node in module.body if isinstance(node,ast.ClassDef) and node.name==klass)
    fn=next(node for node in cls.body if isinstance(node,ast.FunctionDef) and node.name==name)
    namespace=dict(env or {})
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(ROOT/file),'exec'),namespace)
    return namespace[name]


def event():
    calls=[]
    return SimpleNamespace(ignore=lambda:calls.append('ignore'),accept=lambda:calls.append('accept'),calls=calls)


def test_close_waits_for_workers_and_requests_stop_only_once():
    stops=[]; alive=[True]; enabled=[]; scheduled=[];saved=[]
    worker=SimpleNamespace(isRunning=lambda:alive[0],stop=lambda:stops.append('stop'))
    clothing=SimpleNamespace(confirm_discard_profile_edits=lambda:True,worker=worker)
    studio=SimpleNamespace(clothing_tab=clothing,has_unsaved_caption_edits=lambda:False)
    window=SimpleNamespace(caption_studio_page=studio,findChildren=lambda cls:[worker],current_lang='en',
        centralWidget=lambda:SimpleNamespace(setEnabled=enabled.append),close=lambda:None,
        _save_pending_theme_on_close=lambda:saved.append(True))
    close=method('src/ui/main_window.py','VideoSmartCropperUI','closeEvent',
                  {'QThread':object,'QTimer':SimpleNamespace(singleShot=lambda ms,fn:scheduled.append(ms)),
                   'logging':logging})
    one=event();close(window,one)
    two=event();close(window,two)
    assert one.calls == two.calls == ['ignore']
    assert stops == ['stop'] and enabled == [False,False]
    assert saved == []
    alive[0]=False
    three=event();close(window,three)
    assert three.calls == ['accept'] and saved == [True]


def test_rejected_discard_blocks_close_before_worker_stop():
    window=SimpleNamespace(caption_studio_page=SimpleNamespace(
        clothing_tab=SimpleNamespace(confirm_discard_profile_edits=lambda:False)))
    close=method('src/ui/main_window.py','VideoSmartCropperUI','closeEvent')
    attempt=event();close(window,attempt)
    assert attempt.calls == ['ignore']
    assert not hasattr(window,'_close_pending')


def test_small_crop_drag_preserves_previous_selection():
    original=[.1,.2,.8,.9]
    emitted=[]
    canvas=SimpleNamespace(_origin=(.5,.5),_previous_box=original[:],bbox=original[:],
        _normalized_point=lambda pos:(.501,.501),update=lambda:None,
        box_changed=SimpleNamespace(emit=emitted.append))
    move=method('src/ui/clothing_tagger_widget.py','ClothingCropCanvas','mouseMoveEvent')
    canvas.mouseMoveEvent=MethodType(move,canvas)
    release=method('src/ui/clothing_tagger_widget.py','ClothingCropCanvas','mouseReleaseEvent',
                   {'Qt':SimpleNamespace(LeftButton=1)})
    release(canvas,SimpleNamespace(button=lambda:1,pos=lambda:None))
    assert canvas.bbox == original and emitted == []
    assert canvas._origin is None


def test_caption_restart_is_blocked_while_old_worker_cleanup_pending():
    messages=[]
    generate=SimpleNamespace(captioning_thread=object(),window=lambda:SimpleNamespace(),log=messages.append)
    start=method('src/ui/caption_studio_page.py','_GenerateTab','start_captioning')
    start(generate)
    assert len(messages) == 1 and 'finishing' in messages[0]


def test_model_download_completion_cannot_start_caption_when_closing():
    generate=SimpleNamespace(window=lambda:SimpleNamespace(_close_pending=True))
    start=method('src/ui/caption_studio_page.py','_GenerateTab','start_captioning')
    start(generate)  # no access to caption/model fields on a closing window
