"""Source contracts + optional actual Qt smoke tests (reported separately)."""
import ast
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import time
import pytest
ROOT=Path(__file__).resolve().parents[1]
HAVE_QT=importlib.util.find_spec('PyQt5') is not None
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')


def method(file,cls,name):
    tree=ast.parse((ROOT/file).read_text(encoding='utf-8'))
    node=next(n for c in tree.body if isinstance(c,ast.ClassDef) and c.name==cls
              for n in c.body if isinstance(n,ast.FunctionDef) and n.name==name)
    module=ast.Module(body=[node],type_ignores=[]);ast.fix_missing_locations(module)
    ns={};exec(compile(module,str(file),'exec'),ns);return ns[name]


def test_parallel_dataset_writers_disabled_and_restored():
    class Page:
        def __init__(self,value=True):self.value=value
        def isEnabled(self):return self.value
        def setEnabled(self,value):self.value=value
    pages={n:Page(n!='quality_tab') for n in ['generate_tab','edit_tab','quality_tab','clothing_tab','balance_tab','compare_tab']}
    window=SimpleNamespace(start_btn=Page(),training_page=Page(False))
    widget=SimpleNamespace(**pages,window=lambda:window)
    fn=method('src/ui/caption_studio_page.py','CaptionStudioPage','_on_tools_busy')
    owner=pages['balance_tab'];fn(widget,owner,True)
    assert owner.isEnabled() and not window.start_btn.isEnabled()
    assert all(not p.isEnabled() for n,p in pages.items() if n!='balance_tab')
    fn(widget,owner,False)
    assert window.start_btn.isEnabled() and not window.training_page.isEnabled()
    assert not pages['quality_tab'].isEnabled() and pages['compare_tab'].isEnabled()


def test_worker_propagates_errors_instead_of_crashing_event_loop():
    run=method('src/ui/studio_tasks.py','StudioWorker','run');events=[]
    def fail(*a):raise ValueError('sentinel')
    fake=SimpleNamespace(work=fail,cancel=None,progress=SimpleNamespace(emit=lambda *a:None),
        completed=SimpleNamespace(emit=lambda v:events.append(('ok',v))),
        failed=SimpleNamespace(emit=lambda v:events.append(('error',v))))
    run(fake);assert events==[('error','sentinel')]
    fake.work=lambda *a:{'count':2};run(fake);assert events[-1]==('ok',{'count':2})


def test_new_tabs_and_theme_bindings_present():
    source=(ROOT/'src/ui/caption_studio_page.py').read_text()
    assert 'self.tabs.addTab(self.balance_tab' in source and 'self.tabs.addTab(self.compare_tab' in source
    for file in ['dataset_balance_widget.py','checkpoint_compare_widget.py','maintenance_widget.py']:
        assert 'theme.bind_style' in (ROOT/'src/ui'/file).read_text()
    source=(ROOT/'main.py').read_text()
    assert source.index('_runtime_guard.__enter__')<source.index('import onnxruntime')
    assert 'pip install' not in source


@pytest.fixture
def qapp():
    if not HAVE_QT:pytest.skip('PyQt5 unavailable: actual upgraded widget not executed')
    from PyQt5.QtWidgets import QApplication
    app=QApplication.instance() or QApplication([])
    yield app
    app.processEvents()


def test_balance_real_widget_theme_preserves_inputs(qapp,tmp_path):
    from src.ui import theme
    from src.ui.dataset_balance_widget import DatasetBalanceWidget
    from src.core.clothing_profiles import ClothingProfileStore
    previous=theme.state_key();widget=DatasetBalanceWidget('tr',store=ClothingProfileStore(tmp_path/'library'))
    try:
        widget.seed.setValue(991);widget.target.setValue(17)
        theme.set_theme('light',1.0,'#a855f7',persist=False);widget.refresh_styles()
        assert widget.seed.value()==991 and widget.target.value()==17
        assert not widget.is_busy()
    finally:
        widget.deleteLater();qapp.processEvents();theme.set_theme(*previous,persist=False)


def test_comparison_real_widget_defaults_and_saved_config(qapp):
    from src.ui.checkpoint_compare_widget import CheckpointCompareWidget
    widget=CheckpointCompareWidget('tr')
    try:
        assert widget.seeds.text()=='42, 1234' and not widget.is_busy()
        assert widget.width.value()==1024 and widget.catalog_data is None
        widget.prompts.setPlainText('unsaved prompt');widget.refresh_styles()
        assert widget.prompts.toPlainText()=='unsaved prompt'
    finally:widget.deleteLater();qapp.processEvents()


def test_maintenance_real_widget_is_lazy(qapp,monkeypatch):
    import subprocess
    monkeypatch.setattr(subprocess,'Popen',lambda *a,**k:pytest.fail('automatic launch'))
    from src.ui.maintenance_widget import MaintenanceWidget
    widget=MaintenanceWidget('tr')
    try:assert 'sihirbaz' in widget.button.text()
    finally:widget.deleteLater();qapp.processEvents()


def test_studio_real_worker_runs_outside_gui_thread(qapp,tmp_path):
    from PyQt5.QtCore import QThread
    from src.ui.dataset_balance_widget import DatasetBalanceWidget
    from src.core.clothing_profiles import ClothingProfileStore
    widget=DatasetBalanceWidget(store=ClothingProfileStore(tmp_path/'library'));results=[]
    gui_thread=QThread.currentThread()
    try:
        assert widget.start_task(lambda cancel,progress:QThread.currentThread()!=gui_thread,results.append)
        assert not widget.start_task(lambda *a:None,results.append)
        deadline=time.monotonic()+5
        while widget.is_busy() and time.monotonic()<deadline:
            qapp.processEvents();time.sleep(.01)
        assert results==[True] and not widget.is_busy()
    finally:
        if widget._thread:widget.stop();widget._thread.wait(2000)
        widget.deleteLater();qapp.processEvents()
