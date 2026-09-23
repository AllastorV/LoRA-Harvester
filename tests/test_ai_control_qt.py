"""Real Qt opt-in/queued-action smoke tests; skipped explicitly without PyQt5."""
import os
import importlib.util
from types import SimpleNamespace
import pytest

os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
HAS_QT=importlib.util.find_spec('PyQt5') is not None
pytestmark=pytest.mark.skipif(not HAS_QT, reason='PyQt5 unavailable: actual live AI control dock/queued GUI tests not run')

@pytest.fixture
def panel(tmp_path,monkeypatch):
    from PyQt5.QtWidgets import QApplication,QMainWindow
    from src.ui import ai_control_panel, theme
    monkeypatch.setattr(theme, "_current_lang", "en")
    app=QApplication.instance() or QApplication([])
    window=QMainWindow();window._close_pending=False
    monkeypatch.setattr(ai_control_panel,'ROOT',tmp_path)
    p=ai_control_panel.attach_ai_control(window,app)
    try:yield p,app,window
    finally:p.shutdown();window.close();app.processEvents()


def test_qt_panel_is_opt_in_without_open_socket(panel):
    p,app,window=panel
    window.show();app.processEvents()
    assert p.server is None and not p.enabled.isChecked() and not p.isVisible()
    assert p.indicator.text()=='AI off' and p.warning.isVisible()
    assert not window.menuBar().actions()
    assert p.toggle_action.shortcut().toString()=='Ctrl+Shift+A'
    p.indicator.click();app.processEvents()
    assert p.isVisible()


def test_qt_panel_follows_language_switch(panel, monkeypatch):
    from src.ui import theme
    p, app, window = panel
    assert p.windowTitle() == "AI connection · Codex / Claude"
    assert p.enabled.text() == "Enable AI connection"
    monkeypatch.setattr(theme, "_current_lang", "tr")
    p.update_language()
    assert p.indicator.text() == "AI pasif"
    assert p.enabled.text() == "AI bağlantısını aç"


def test_qt_panel_enable_disable_local_session(panel,tmp_path):
    p,app,window=panel
    window.show();app.processEvents()
    p._grant(tmp_path);p.enabled.setChecked(True);app.processEvents()
    assert p.server is not None and p.server.session_file.exists()
    assert p.indicator.text()=='AI on' and not p.warning.isVisible()
    p.enabled.setChecked(False);app.processEvents()
    assert p.server is None and not (tmp_path/'.ai-control'/'session.json').exists()
    assert p.indicator.text()=='AI off' and p.warning.isVisible()


def test_qt_queued_command_runs_on_gui_timer(panel,tmp_path,monkeypatch):
    p,app,window=panel
    p._grant(tmp_path);p.enabled.setChecked(True)
    monkeypatch.setattr(p.live,'dispatch',lambda request:{'real_qt_timer':True})
    rid=p.broker.submit('app.state',{})['id']
    p._tick()
    assert p.broker.status(rid)['result']=={'real_qt_timer':True}


def test_qt_human_approval_cannot_be_replaced_by_receipt(panel,tmp_path):
    p,app,window=panel
    p._grant(tmp_path);p.enabled.setChecked(True)
    r=p.broker.submit('dataset.scan',{})
    assert p.broker.take() is None
    p._refresh();p.requests.setCurrentRow(0);p._approve(False)
    assert p.broker.status(r['id'])['status']=='rejected'
