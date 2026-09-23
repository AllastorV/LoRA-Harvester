"""Optional real offscreen Qt signal, selection, draft and worker-flow tests.

These are skipped, never simulated or reported passed, if PyQt5 is unavailable.
No model weights or external server are needed, even for the threaded write test.
"""
import importlib.util
import os
import time
from pathlib import Path
import pytest
from PIL import Image

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
HAVE_QT = importlib.util.find_spec('PyQt5') is not None


@pytest.fixture
def pair(tmp_path, monkeypatch):
    if not HAVE_QT:
        pytest.skip('PyQt5 unavailable: real caption sync widget test not executed')
    from PyQt5.QtWidgets import QApplication, QMessageBox
    from PyQt5.QtCore import QCoreApplication, QEvent
    from src.ui import caption_studio_page as studio
    from src.ui.clothing_tagger_widget import ClothingTaggerWidget
    from src.core.clothing_profiles import ClothingProfileStore, ClothingProfile, ClothingPart
    from src.core.clothing_feedback import ClothingFeedbackStore
    from src.core.clothing_io import file_digest
    from src.core.clothing_captions import prepare_caption
    app = QApplication.instance() or QApplication([])
    # No tag download on incidental focus when running headlessly.
    monkeypatch.setattr(studio, '_shared_tag_load_started', True)
    monkeypatch.setattr(QMessageBox, 'warning', lambda *args: QMessageBox.Ok)
    monkeypatch.setattr(QMessageBox, 'question', lambda *args: QMessageBox.Yes)
    store = ClothingProfileStore(tmp_path / 'library')
    source = tmp_path / 'reference.png'; Image.new('RGB', (40, 60), (40, 70, 110)).save(source)
    profile = ClothingProfile('Uniform', 'schoolmaster', [ClothingPart('blue skirt')],
                              references=[store.import_reference(source)], id='uniform')
    store.upsert(profile)
    folder = tmp_path / 'dataset'; folder.mkdir()
    images = [folder / '01.png', folder / '02.png']
    for i, path in enumerate(images):
        Image.new('RGB', (40, 60), (40+i, 70, 110)).save(path)
        path.with_suffix('.txt').write_text(f'original{i}')
    edit = studio._EditTab('en')
    clothing = ClothingTaggerWidget('en', store=store)
    edit.folder_changed.connect(clothing.reload_folder)
    edit.captions_saved.connect(clothing.refresh_captions)
    clothing.captions_changed.connect(edit.refresh_saved_captions)
    edit.reload_folder(str(folder))
    result = ClothingFeedbackStore(store).remember(images[0], [0, 0, 1, 1], profile_id=profile.id,
                            visible_tags=['blue skirt'], expected_image_digest=file_digest(images[0]))
    clothing._on_result(result, prepare_caption(result))
    app.processEvents()
    yield app, edit, clothing, images
    if clothing.is_busy():
        clothing.shutdown(5000)
        app.processEvents()
    edit.deleteLater(); clothing.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    app.processEvents()


def test_real_edit_signal_rebases_without_changing_selection_or_crop(pair):
    app, edit, clothing, images = pair
    selection = clothing.canvas.bbox[:]
    edit.caption_edit.setPlainText('manual tag, 1girl')
    assert edit._save_current()
    app.processEvents()
    assert clothing.before_text.toPlainText() == 'manual tag, 1girl'
    assert clothing.after_text.toPlainText().startswith('schoolmaster, blue skirt, manual tag')
    assert clothing.canvas.bbox == selection and clothing._active_image == str(images[0])
    assert edit.image_list.currentRow() == 0
    assert not edit.has_unsaved_edits()


def test_real_drafts_survive_switching_rows_and_saving_another_image(pair):
    app, edit, clothing, images = pair
    edit.caption_edit.setPlainText('draft one')
    edit.image_list.setCurrentRow(1)
    edit.caption_edit.setPlainText('saved two')
    assert edit._save_current()
    edit.image_list.setCurrentRow(0)
    assert edit.caption_edit.toPlainText() == 'draft one'
    assert edit.has_unsaved_edits()
    assert images[0].with_suffix('.txt').read_text() == 'original0'


def test_real_refresh_updates_text_without_spurious_dirty_flag_or_cursor_reset(pair):
    app, edit, clothing, images = pair
    cursor = edit.caption_edit.textCursor(); cursor.setPosition(3); edit.caption_edit.setTextCursor(cursor)
    images[0].with_suffix('.txt').write_text('new external caption')
    edit.refresh_saved_captions([str(images[0].with_suffix('.txt'))])
    assert edit.caption_edit.toPlainText() == 'new external caption'
    assert edit.caption_edit.textCursor().position() == 3
    assert not edit.has_unsaved_edits()


def test_real_save_and_next_signal_preserves_file_contents(pair):
    app, edit, clothing, images = pair
    edit.caption_edit.setPlainText('first saved')
    edit._save_and_next(); app.processEvents()
    assert edit.image_list.currentRow() == 1
    assert edit.caption_edit.toPlainText() == 'original1'
    assert not edit.has_unsaved_edits()
    assert clothing.before_text.toPlainText() == 'first saved'


def test_real_folder_reload_does_not_commit_previous_text_into_new_folder(pair, tmp_path):
    app, edit, clothing, images = pair
    other = tmp_path / 'other'; other.mkdir()
    image = other / 'new.png'; Image.new('RGB', (30, 30)).save(image)
    image.with_suffix('.txt').write_text('new folder caption')
    edit.reload_folder(str(other))
    assert edit.caption_edit.toPlainText() == 'new folder caption'
    assert list(edit._captions.values()) == ['new folder caption']
    assert not edit.has_unsaved_edits()


def test_real_qthread_apply_and_undo_notify_after_worker_finishes(pair):
    app, edit, clothing, images = pair
    def wait_finished():
        until = time.monotonic() + 8
        while clothing.worker is not None and time.monotonic() < until:
            app.processEvents(); time.sleep(.005)
        assert clothing.worker is None
    notifications = []
    clothing.captions_changed.connect(lambda paths: notifications.append(clothing.worker))
    clothing._start('apply', previews=[clothing._previews[str(images[0])]])
    wait_finished()
    assert edit.caption_edit.toPlainText().startswith('schoolmaster, blue skirt')
    clothing._start('undo')
    wait_finished()
    assert edit.caption_edit.toPlainText() == 'original0'
    assert notifications == [None, None]
    assert not edit.has_unsaved_edits()
