"""Real offscreen Qt tests. Skipped rather than simulated when Qt is unavailable."""
import importlib.util
import os
from pathlib import Path
import time
import pytest
from PIL import Image

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
HAVE_QT = importlib.util.find_spec('PyQt5') is not None
pytestmark = pytest.mark.skipif(not HAVE_QT, reason='PyQt5 unavailable: new Studio Qt/render tests not run')


@pytest.fixture
def desktop(tmp_path, monkeypatch):
    from PyQt5.QtWidgets import QApplication, QMessageBox
    from src.ui import theme, caption_studio_page as captions
    from src.ui.design_system.media import MediaLoader
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(captions, '_shared_tag_load_started', True)
    # Each Qt fixture destroys its editor; rebuild the process-wide model for the next one.
    monkeypatch.setattr(captions, '_shared_tag_model', None)
    monkeypatch.setattr(QMessageBox, 'question', lambda *args: QMessageBox.Yes)
    monkeypatch.setattr(QMessageBox, 'warning', lambda *args: QMessageBox.Ok)
    media = MediaLoader()
    created = []
    yield app, media, created
    for widget in created:
        widget.close()
        widget.deleteLater()
    media.shutdown()
    media.deleteLater()
    app.processEvents()


def until(app, predicate, seconds=5):
    start = time.monotonic()
    while not predicate():
        app.processEvents()
        if time.monotonic() - start > seconds:
            raise AssertionError('Qt condition not reached before timeout')
        time.sleep(.01)


def make_editor(desktop):
    from src.ui.caption_studio_page import _EditTab
    from src.ui.design_system.editor_layout import EditorStudioLayout
    app, media, widgets = desktop
    editor = _EditTab('tr')
    editor._studio_layout = EditorStudioLayout(editor, media)
    editor.resize(1050, 720)
    widgets.append(editor)
    editor.show()
    app.processEvents()
    return editor


def test_real_button_to_parameterless_signal(desktop):
    from PyQt5.QtCore import QObject, pyqtSignal
    from src.ui.design_system.widgets import button
    class Receiver(QObject):
        action = pyqtSignal()
    app, _, widgets = desktop
    receiver = Receiver(); called = []
    receiver.action.connect(lambda: called.append(True))
    b = button('Test', receiver.action.emit); widgets.append(b)
    b.click(); app.processEvents()
    assert called == [True]


def test_real_editor_uses_existing_text_store_and_saves(desktop, tmp_path):
    app, _, _ = desktop
    source = tmp_path / 'image.png'; Image.new('RGB', (120, 180)).save(source)
    source.with_suffix('.txt').write_text('master, blue skirt')
    editor = make_editor(desktop)
    editor.reload_folder(str(tmp_path))
    assert editor.caption_edit.isVisible()
    editor.caption_edit.setPlainText('master, blue skirt, sitting')
    assert editor._save_current()
    assert source.with_suffix('.txt').read_text() == 'master, blue skirt, sitting'
    assert not editor.has_unsaved_edits()
    until(app, lambda: editor.preview_lbl.pixmap() is not None)


def test_real_editor_drafts_survive_selection(desktop, tmp_path):
    for i in range(2):
        p = tmp_path / f'{i}.png'; Image.new('RGB', (100, 150)).save(p)
        p.with_suffix('.txt').write_text(f'source{i}')
    editor = make_editor(desktop)
    editor.reload_folder(str(tmp_path))
    editor.caption_edit.setPlainText('unsaved master, blue skirt')
    editor.image_list.setCurrentRow(1)
    editor.image_list.setCurrentRow(0)
    assert editor.caption_edit.toPlainText() == 'unsaved master, blue skirt'
    assert (tmp_path / '0.txt').read_text() == 'source0'


def test_real_empty_folder_clears_delayed_preview(desktop, tmp_path):
    source = tmp_path / 'data'; source.mkdir()
    empty = tmp_path / 'empty'; empty.mkdir()
    Image.new('RGB', (100, 150)).save(source / 'a.png')
    editor = make_editor(desktop)
    editor.reload_folder(str(source)); editor.reload_folder(str(empty))
    assert editor._studio_layout._selected is None
    desktop[0].processEvents()
    assert not editor._items


def test_real_library_filters_and_opens_exact_path(desktop, tmp_path):
    from src.core.smart_suggestions import scan_suggestions
    from src.ui.pages.studio_pages import LibraryPage
    app, media, widgets = desktop
    for name in ('a.png', 'b.png'):
        Image.new('RGB', (100, 150)).save(tmp_path / name)
    report = scan_suggestions(tmp_path)
    page = LibraryPage(media); widgets.append(page)
    page.set_report(report)
    page.filter_paths([str(tmp_path / 'b.png')], 'test filter')
    assert page.proxy.rowCount() == 1
    calls = []; page.open_requested.connect(lambda p, r: calls.append((p,r)))
    page.open_current('edit')
    assert calls == [(str(tmp_path / 'b.png'), 'edit')]


def test_real_disclosure_keeps_settings_values(desktop):
    from PyQt5.QtWidgets import QSpinBox
    from src.ui.design_system.widgets import Disclosure
    spin = QSpinBox(); spin.setValue(37)
    widget = Disclosure('Advanced', spin); desktop[2].append(widget)
    widget.header.setChecked(True); widget.header.setChecked(False)
    assert spin.value() == 37 and spin.isEnabled()


def test_real_media_loader_decodes_offscreen_and_reports_missing(desktop, tmp_path):
    app, media, _ = desktop
    source = tmp_path / 'a.png'; Image.new('RGB', (100, 180)).save(source)
    media.request(source)
    until(app, lambda: media.request(source) is not None)
    cached = media.request(source)
    assert not cached[0].isNull() and cached[0].width() <= 96 and cached[0].height() <= 96
    missing = media.request(tmp_path / 'missing.png')
    assert missing[0].isNull() and missing[2]


def test_real_overview_uses_no_fabricated_metrics(desktop, tmp_path):
    from src.ui.pages.studio_pages import MetricStrip
    from src.core.smart_suggestions import scan_suggestions
    metrics = MetricStrip(); desktop[2].append(metrics)
    assert [v.text() for v in metrics.values] == ['—'] * 4
    metrics.set_report(scan_suggestions(tmp_path))
    assert [v.text() for v in metrics.values] == ['0','0','0','Taranmadı']
