"""Actual Qt smoke tests when available, plus source-method lifecycle checks."""
import ast
import copy
import importlib.util
import os
from pathlib import Path
import threading
from types import SimpleNamespace
import time
import pytest

ROOT=Path(__file__).resolve().parents[1]
HAVE_QT=importlib.util.find_spec('PyQt5') is not None
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')


def method(file,cls,name,scope=None):
    tree=ast.parse((ROOT/file).read_text('utf-8'))
    node=copy.deepcopy(next(n for c in tree.body if isinstance(c,ast.ClassDef) and c.name==cls
                           for n in c.body if isinstance(n,ast.FunctionDef) and n.name==name))
    node.decorator_list=[]
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),node],type_ignores=[])
    ast.fix_missing_locations(module);ns=dict(scope or {})
    exec(compile(module,str(file),'exec'),ns);return ns[name]


def test_kohya_qthread_waits_for_real_process_monitor():
    done=[]; joined=[]
    def start(**kw):
        def join():
            joined.append(True);kw['finished_callback'](True,'saved')
        trainer._thread=SimpleNamespace(join=join)
        return True
    trainer=SimpleNamespace(start=start)
    fake=SimpleNamespace(_trainer=trainer,_train_toml='config.toml',_cancel=threading.Event(),
        log_msg=SimpleNamespace(emit=lambda x:None),finished_sig=SimpleNamespace(emit=lambda *args:done.append(args)))
    method('src/ui/training_page.py','_TrainThread','run')(fake)
    assert joined==[True] and done==[(True,'saved')]


def test_kohya_start_failure_not_left_busy():
    done=[]
    fake=SimpleNamespace(_trainer=SimpleNamespace(start=lambda **kw:False),_train_toml='config',_cancel=threading.Event(),
        log_msg=SimpleNamespace(emit=lambda x:None),finished_sig=SimpleNamespace(emit=lambda *args:done.append(args)))
    method('src/ui/training_page.py','_TrainThread','run')(fake)
    assert len(done)==1 and done[0][0] is False


def test_kohya_cancel_before_launch():
    cancel=threading.Event();cancel.set();done=[]
    fake=SimpleNamespace(_cancel=cancel,finished_sig=SimpleNamespace(emit=lambda *a:done.append(a)))
    method('src/ui/training_page.py','_TrainThread','run')(fake)
    assert done and done[0][0] is False


def test_training_hub_exposes_active_toolkit_worker():
    idle=SimpleNamespace(isRunning=lambda:False);busy=SimpleNamespace(isRunning=lambda:True)
    fake=SimpleNamespace(kohya=SimpleNamespace(_thread=idle),toolkit=SimpleNamespace(_thread=busy))
    assert method('src/ui/training_hub_page.py','TrainingHubPage','_worker')(fake,'_thread') is busy


def test_worker_stops_runner_not_qthread_terminate():
    stopped=[];interrupt=[];cancel=threading.Event()
    fake=SimpleNamespace(cancel=cancel,runner=SimpleNamespace(stop=lambda:stopped.append(1)),
        requestInterruption=lambda:interrupt.append(1))
    method('src/ui/ai_toolkit_training_page.py','ToolkitWorker','stop')(fake)
    assert cancel.is_set() and stopped and interrupt


def test_source_main_uses_hub_and_classic_ai_respects_training():
    assert 'TrainingHubPage as TrainingPage' in (ROOT/'src/ui/main_window.py').read_text()
    tree=ast.parse((ROOT/'src/ui/ai_live_actions.py').read_text())
    body=next(ast.get_source_segment((ROOT/'src/ui/ai_live_actions.py').read_text(),n)
              for c in tree.body if isinstance(c,ast.ClassDef) for n in c.body
              if isinstance(n,ast.FunctionDef) and n.name=='_idle')
    assert 'training_page' in body and "'busy'" in body


def test_form_has_all_native_option_fields_and_no_unwired_te_lr():
    from dataclasses import fields
    from src.training.ai_toolkit_config import ToolkitOptions
    tree=ast.parse((ROOT/'src/ui/ai_toolkit_training_page.py').read_text())
    controls=set()
    for n in ast.walk(tree):
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in (
            '_line','_path_row','_integer','_decimal','_flag','_combo'):
            if len(n.args)>1 and isinstance(n.args[1],ast.Constant):controls.add(n.args[1].value)
    expected={f.name for f in fields(ToolkitOptions)}-{'sample_prompts'}
    assert expected<=controls
    assert 'text_encoder_lr' not in controls


@pytest.fixture
def qapp():
    if not HAVE_QT: pytest.skip('PyQt5 unavailable: AI Toolkit widget not executed')
    from PyQt5.QtWidgets import QApplication
    app=QApplication.instance() or QApplication([])
    yield app
    app.processEvents()


@pytest.fixture
def widget(qapp,monkeypatch):
    from src.ui.ai_toolkit_training_page import AIToolkitTrainingPage
    monkeypatch.setattr(AIToolkitTrainingPage,'_load_settings',lambda self:None)
    w=AIToolkitTrainingPage('tr')
    yield w
    if w._thread:w._thread.stop();w._thread.wait(5000)
    w.deleteLater();qapp.processEvents()


def test_real_widget_defaults(widget):
    assert widget._value('steps')==3000
    assert not widget._value('train_text_encoder')
    assert not widget._value('allow_downloads')
    assert widget._controls['learning_rate'].decimals()==10


def test_real_widget_family_rules(widget):
    c=widget._controls['architecture'];c.setCurrentIndex(c.findData('flux'))
    assert widget._value('quantize') and widget._value('min_snr_gamma')==0
    assert not widget._controls['train_text_encoder'].isEnabled()
    c.setCurrentIndex(c.findData('sdxl'))
    assert not widget._value('quantize') and widget._controls['train_text_encoder'].isEnabled()


def test_real_widget_theme_preserves_values(widget):
    from src.ui import theme
    previous=theme.state_key()
    try:
        widget._controls['trigger_word'].setText('literal_master')
        theme.set_theme('light',1.1,'#a855f7',persist=False);widget.refresh_styles()
        widget.update_language('en')
        assert widget._value('trigger_word')=='literal_master'
    finally:theme.set_theme(*previous,persist=False)


def test_real_worker_build_no_gui_block(qapp,widget,tmp_path):
    from PIL import Image
    ds=tmp_path/'dataset';ds.mkdir();Image.new('RGB',(32,48)).save(ds/'p.png');(ds/'p.txt').write_text('master, shirt')
    widget._settings_file=tmp_path/'settings.json'
    widget._controls['dataset_dir'].setText(str(ds));widget._controls['output_dir'].setText(str(tmp_path/'output'))
    widget._controls['base_model'].setText('owner/base');widget._build()
    deadline=time.monotonic()+5
    while widget._thread is not None and time.monotonic()<deadline:
        qapp.processEvents();time.sleep(.01)
    assert widget._bundle and widget._thread is None
    assert 'sd_trainer' in widget.preview.toPlainText()
    widget._controls['steps'].setValue(4000)
    assert widget._bundle is None


def test_real_hub_blocks_switch_while_busy(qapp,monkeypatch):
    from src.ui.training_page import TrainingPage
    from src.ui.ai_toolkit_training_page import AIToolkitTrainingPage
    from src.ui.training_hub_page import TrainingHubPage
    monkeypatch.setattr(TrainingPage,'_detect_kohya',lambda self:None)
    monkeypatch.setattr(AIToolkitTrainingPage,'_load_settings',lambda self:None)
    hub=TrainingHubPage('tr')
    try:
        hub.engine.setCurrentIndex(1);assert hub.stack.currentWidget() is hub.toolkit
        hub._busy(hub.toolkit,True);hub.engine.setCurrentIndex(0)
        assert hub.stack.currentWidget() is hub.toolkit
        hub._busy(hub.toolkit,False);hub.engine.setCurrentIndex(0)
        assert hub.stack.currentWidget() is hub.kohya
    finally:hub.deleteLater();qapp.processEvents()
