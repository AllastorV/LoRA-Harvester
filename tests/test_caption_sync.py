"""Real local files + executed UI source contracts, without a vision model.

UI fakes exercise the exact shipped methods, NOT Qt event delivery/painting.
Actual Qt tests are separately marked in test_caption_sync_qt.py.
"""
from __future__ import annotations
import ast
from dataclasses import replace
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest
from PIL import Image
from src.core import caption_sync
from src.core.caption_sync import (CaptionPreviewRebaser, CaptionSnapshot, editor_text,
                                   path_key, read_caption_snapshot, save_edited_caption)
from src.core.clothing_captions import CaptionConflict, prepare_caption, apply_caption, undo_caption
from src.core.clothing_io import FileLock, file_digest
from src.core.clothing_profiles import ClothingProfileStore, ClothingProfile, ClothingPart
from src.core.clothing_feedback import ClothingFeedbackStore
from src.core.clothing_service import save_selection

ROOT = Path(__file__).resolve().parents[1]
EDIT = ('src/ui/caption_studio_page.py', '_EditTab')
CLOTHING = ('src/ui/clothing_tagger_widget.py', 'ClothingTaggerWidget')
STUDIO = ('src/ui/caption_studio_page.py', 'CaptionStudioPage')


@pytest.fixture
def sample(tmp_path):
    store = ClothingProfileStore(tmp_path / 'library')
    ref = tmp_path / 'ref.png'
    Image.new('RGB', (32, 48), (10, 70, 120)).save(ref)
    profile = ClothingProfile('Blue uniform', 'shhooldress',
        [ClothingPart('serafaku'), ClothingPart('blue skirt'), ClothingPart('black thighighhs')],
        references=[store.import_reference(ref)], id='blue')
    store.upsert(profile)
    image = tmp_path / 'dataset' / 'örnek.png'
    image.parent.mkdir()
    Image.new('RGB', (48, 64), (15, 75, 130)).save(image)
    image.with_suffix('.txt').write_bytes(b'1girl, standing\n')
    memory = ClothingFeedbackStore(store)
    result = memory.remember(image, [0., 0., 1., 1.], profile_id='blue',
                             visible_tags=['serafaku', 'blue skirt'],
                             expected_image_digest=file_digest(image))
    return SimpleNamespace(store=store, image=image, profile=profile, result=result, memory=memory)


def test_editor_snapshot_normalizes_bom_crlf_without_losing_bytes(sample, monkeypatch):
    raw = b'\xef\xbb\xbf1girl, standing\r\n'
    sample.image.with_suffix('.txt').write_bytes(raw)
    snap = read_caption_snapshot(sample.image)
    assert snap.raw == raw and snap.text == '1girl, standing\n'
    monkeypatch.setattr(caption_sync, 'atomic_bytes', Mock(side_effect=AssertionError('unneeded rewrite')))
    assert save_edited_caption(snap, snap.text) == snap


def test_edit_preserves_encoding_and_newline_style(sample):
    path = sample.image.with_suffix('.txt')
    path.write_bytes(b'\xef\xbb\xbf1girl\r\n')
    saved = save_edited_caption(read_caption_snapshot(sample.image), '1girl, gözlük\n')
    assert path.read_bytes() == b'\xef\xbb\xbf' + '1girl, gözlük\r\n'.encode()
    assert saved.text == '1girl, gözlük\n'


def test_missing_caption_can_be_saved_and_baseline_distinguishes_empty(sample):
    path = sample.image.with_suffix('.txt'); path.unlink()
    snap = read_caption_snapshot(sample.image)
    assert snap.raw is None and snap.text == ''
    result = save_edited_caption(snap, '')
    assert path.exists() and result.raw == b''


def test_concurrent_external_change_is_not_overwritten(sample):
    snap = read_caption_snapshot(sample.image)
    path = sample.image.with_suffix('.txt'); path.write_text('external edit')
    with pytest.raises(CaptionConflict, match='changed on disk'):
        save_edited_caption(snap, 'my draft')
    assert path.read_text() == 'external edit'


def test_atomic_write_failure_preserves_original_file(sample, monkeypatch):
    snap = read_caption_snapshot(sample.image)
    monkeypatch.setattr(caption_sync, 'atomic_bytes', Mock(side_effect=OSError('disk full')))
    with pytest.raises(OSError, match='disk full'):
        save_edited_caption(snap, 'new')
    assert snap.path.read_bytes() == snap.raw


def test_shared_clothing_lock_serializes_manual_writes(sample):
    snap = read_caption_snapshot(sample.image)
    with FileLock(sample.image.parent / '.lh-clothing' / 'write.lock'):
        with pytest.raises((OSError, RuntimeError)):
            save_edited_caption(snap, 'new')
    assert snap.path.read_bytes() == snap.raw


def test_shared_caption_basename_is_rejected(sample):
    snap = read_caption_snapshot(sample.image)
    Image.new('RGB', (20, 20)).save(sample.image.with_suffix('.jpg'))
    with pytest.raises(CaptionConflict, match='same .txt'):
        save_edited_caption(snap, 'new')
    assert snap.path.read_bytes() == snap.raw


def test_symbolic_caption_is_not_followed(sample, tmp_path):
    other = tmp_path / 'other.txt'; other.write_text('protected')
    path = sample.image.with_suffix('.txt'); path.unlink()
    try:
        path.symlink_to(other)
    except OSError:
        pytest.skip('Platform does not permit creating symlinks')
    with pytest.raises(CaptionConflict, match='symbolic'):
        read_caption_snapshot(sample.image)
    assert other.read_text() == 'protected'


def test_unreadable_caption_does_not_become_empty_baseline(sample):
    sample.image.with_suffix('.txt').write_bytes(b'\xff\xfe\xfa')
    with pytest.raises(UnicodeError):
        read_caption_snapshot(sample.image)


def test_missing_source_blocks_saving(sample):
    snap = read_caption_snapshot(sample.image); sample.image.unlink()
    with pytest.raises(CaptionConflict, match='missing'):
        save_edited_caption(snap, 'new')
    assert snap.path.read_bytes() == snap.raw


def test_rebase_updates_text_preserving_master_visible_parts_without_inference(sample, monkeypatch):
    from src.core.clothing_backend import OllamaClothingBackend
    monkeypatch.setattr(OllamaClothingBackend, '__init__', Mock(side_effect=AssertionError('model used')))
    snap = read_caption_snapshot(sample.image)
    saved = save_edited_caption(snap, '1girl, sitting, from side')
    preview = CaptionPreviewRebaser(sample.store).prepare(sample.result)
    assert preview.before == saved.raw
    assert preview.new_text == 'shhooldress, serafaku, blue skirt, 1girl, sitting, from side\n'
    assert 'black thighighhs' not in preview.new_text
    assert saved.path.read_bytes() == saved.raw  # Preview refresh never applies itself.


def test_rebased_apply_and_undo_restore_new_manual_edit(sample):
    snap = save_edited_caption(read_caption_snapshot(sample.image), 'manual tag, 1girl')
    preview = CaptionPreviewRebaser(sample.store).prepare(sample.result)
    journal = apply_caption(preview, sample.store)
    assert snap.path.read_text().startswith('shhooldress, serafaku, blue skirt, manual tag')
    undo_caption(journal)
    assert snap.path.read_bytes() == snap.raw


def test_rebased_preview_retains_last_moment_conflict_guard(sample):
    preview = CaptionPreviewRebaser(sample.store).prepare(sample.result)
    sample.image.with_suffix('.txt').write_text('another writer')
    with pytest.raises(CaptionConflict, match='changed after preview'):
        apply_caption(preview, sample.store)


@pytest.mark.parametrize('change', ['image', 'profile', 'rival', 'reference', 'memory', 'selection'])
def test_visual_or_library_changes_cannot_become_valid_through_text_refresh(sample, change):
    if change == 'image':
        Image.new('RGB', (48, 64), (250, 0, 0)).save(sample.image)
    elif change == 'profile':
        sample.profile.notes = 'New meaning'; sample.store.upsert(sample.profile)
    elif change == 'rival':
        rival = ClothingProfile('Red uniform', 'redmaster', [ClothingPart('red skirt')],
                                references=sample.profile.references, id='red')
        sample.store.upsert(rival)
    elif change == 'reference':
        Image.new('RGB', (30, 40), (250, 0, 0)).save(sample.store.reference_path(sample.profile.references[0]))
    elif change == 'memory':
        sample.memory.forget(sample.image, [0, 0, 1, 1])
    else:
        save_selection(sample.store, sample.image, [.1, .1, .8, .8])
    with pytest.raises(CaptionConflict):
        CaptionPreviewRebaser(sample.store).prepare(sample.result)


def test_library_fingerprints_are_not_recomputed_for_every_caption(sample, monkeypatch):
    real = sample.store.fingerprint
    counter = Mock(wraps=real); monkeypatch.setattr(sample.store, 'fingerprint', counter)
    refresh = CaptionPreviewRebaser(sample.store)
    for _ in range(10):
        assert refresh.prepare(sample.result).profile_id == 'blue'
    assert counter.call_count == 1


def test_review_and_no_match_do_not_gain_tags_on_refresh(sample):
    for status in ('review', 'no_match', 'error'):
        assert CaptionPreviewRebaser(sample.store).prepare(replace(sample.result, status=status)) is None


class Signal:
    def __init__(self):
        self.slots, self.calls = [], []
    def connect(self, slot):
        self.slots.append(slot)
    def emit(self, *args):
        self.calls.append(args)
        for slot in list(self.slots):
            slot(*args)


class Text:
    def __init__(self, value=''):
        self.value, self.blocked = value, False
        self.position = self.anchor = self.scroll = 0
    def toPlainText(self): return self.value
    def setPlainText(self, text): self.value = text
    def setText(self, text): self.value = text
    def text(self): return self.value
    def clear(self): self.value = ''
    def blockSignals(self, state):
        old, self.blocked = self.blocked, state
        return old
    def textCursor(self): return Cursor(self)
    def setTextCursor(self, cursor): self.position, self.anchor = cursor.pos, cursor.anc
    def verticalScrollBar(self):
        return SimpleNamespace(value=lambda: self.scroll, setValue=lambda value: setattr(self, 'scroll', value))
    def document(self): return SimpleNamespace(characterCount=lambda: len(self.value) + 1)


class Cursor:
    def __init__(self, text): self.pos, self.anc = text.position, text.anchor
    def position(self): return self.pos
    def anchor(self): return self.anc
    def setPosition(self, value, mode=None):
        self.pos = value
        if mode is None: self.anc = value


class Cell(Text):
    def __init__(self, text=''):
        super().__init__(text); self.checked = 0
    def setCheckState(self, state): self.checked = state
    def checkState(self): return self.checked


class Table:
    def __init__(self, paths):
        self.cells = {(r, c): Cell() for r, _ in enumerate(paths) for c in range(5)}
        self.hidden = {}; self.selected_row = 0
    def item(self, r, c): return self.cells[r, c]
    def setRowHidden(self, row, hidden): self.hidden[row] = hidden


def bind_source(obj, file, cls, names, env=None):
    tree = ast.parse((ROOT / file).read_text(encoding='utf-8'))
    klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    warnings = []
    ns = dict(Path=Path, path_key=path_key, read_caption_snapshot=read_caption_snapshot,
              save_edited_caption=save_edited_caption, CaptionPreviewRebaser=CaptionPreviewRebaser,
              Qt=SimpleNamespace(Checked=2, Unchecked=0),
              QTextCursor=SimpleNamespace(KeepAnchor=1),
              get_text=lambda key, lang: key, json=__import__('json'),
              QMessageBox=SimpleNamespace(warning=lambda *args: warnings.append(args)))
    ns.update(env or {})
    for name in names:
        node = next(n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == name)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), file, 'exec'), ns)
        setattr(obj, name, MethodType(ns[name], obj))
    return warnings


def edit_fake(images):
    images = [Path(i).resolve() for i in images]
    snapshots = {str(i.with_suffix('.txt')): read_caption_snapshot(i) for i in images}
    edit = SimpleNamespace(lang='tr', _folder=str(images[0].parent), _current_idx=0,
        _items=[(str(i), str(i.with_suffix('.txt'))) for i in images],
        _captions={cp: snap.text for cp, snap in snapshots.items()}, _snapshots=snapshots,
        _dirty_paths=set(), _read_errors={}, _dirty=False,
        caption_edit=Text(next(iter(snapshots.values())).text),
        captions_saved=Signal(), folder_changed=Signal(), status_lbl=Text(),
        _build_chips=lambda text: None)
    edit.image_list = SimpleNamespace(setCurrentRow=lambda row: setattr(edit, '_current_idx', row))
    names = ['_commit_current', '_mark_dirty', 'has_unsaved_edits', '_report_save_errors',
             'refresh_saved_captions', '_on_caption_changed', '_save_current',
             '_save_and_next', '_save_all', '_refresh_editor', '_revert_current']
    edit.warnings = bind_source(edit, *EDIT, names)
    return edit


def clothing_fake(sample, extra_images=()):
    paths = [str(sample.image.resolve())] + [str(Path(p).resolve()) for p in extra_images]
    table = Table(paths); table.item(0, 0).setCheckState(2)
    widget = SimpleNamespace(lang='tr', store=sample.store, _paths=paths,
        _row_by_path={p: i for i, p in enumerate(paths)}, _results={paths[0]: sample.result},
        _previews={paths[0]: prepare_caption(sample.result)}, _active_image=paths[0],
        _sync_errors={}, _pending_caption_paths=set(), _pending_caption_all=False,
        _caption_write_paths=[], _last_operation='', worker=None,
        before_text=Text(), after_text=Text(), detail_text=Text(),
        image_table=table, only_review=SimpleNamespace(isChecked=lambda: False),
        captions_changed=Signal(), status=Text(), _busy_calls=[],
        canvas=SimpleNamespace(bbox=[.1, .2, .8, .9]))
    widget._set_busy = widget._busy_calls.append
    widget.progress = SimpleNamespace(maximum=lambda: 1)
    bind_source(widget, *CLOTHING, ['_text', 'is_busy', '_show_details', 'refresh_captions',
                                  '_filter_results', '_job_finished'])
    return widget


def connect_tabs(edit, clothing):
    studio = SimpleNamespace(edit_tab=edit, clothing_tab=clothing)
    bind_source(studio, *STUDIO, ['_on_clothing_changed', 'has_unsaved_caption_edits'])
    edit.captions_saved.connect(clothing.refresh_captions)
    clothing.captions_changed.connect(studio._on_clothing_changed)
    return studio


def draft(edit, text):
    edit.caption_edit.setPlainText(text); edit._on_caption_changed()


def second_image(sample):
    path = sample.image.parent / 'second.png'
    Image.new('RGB', (20, 30), (30, 40, 50)).save(path)
    path.with_suffix('.txt').write_text('other caption')
    return path


def test_actual_edit_save_method_refreshes_clothing_without_apply_or_folder_reload(sample):
    edit = edit_fake([sample.image]); clothing = clothing_fake(sample)
    connect_tabs(edit, clothing)
    crop = list(clothing.canvas.bbox)
    draft(edit, '1girl, sitting')
    assert edit._save_current()
    assert clothing.before_text.value == '1girl, sitting'
    assert clothing.after_text.value == 'shhooldress, serafaku, blue skirt, 1girl, sitting\n'
    assert sample.image.with_suffix('.txt').read_text() == '1girl, sitting'
    assert edit._current_idx == 0 and clothing._active_image == str(sample.image)
    assert clothing.canvas.bbox == crop
    assert edit.folder_changed.calls == [] and clothing.captions_changed.calls == []
    assert clothing.image_table.item(0, 0).checkState() == 0  # Changed proposal requires renewed review.
    assert not edit._dirty


def test_save_updates_only_exact_paths_not_an_unrelated_same_basename(sample, tmp_path):
    elsewhere = tmp_path / 'other' / sample.image.name; elsewhere.parent.mkdir()
    Image.new('RGB', (20, 20)).save(elsewhere)
    elsewhere.with_suffix('.txt').write_text('unrelated')
    edit = edit_fake([elsewhere]); clothing = clothing_fake(sample); connect_tabs(edit, clothing)
    original = clothing._previews[str(sample.image)]
    draft(edit, 'changed elsewhere'); assert edit._save_current()
    assert clothing._previews[str(sample.image)] is original
    assert sample.image.with_suffix('.txt').read_text() == '1girl, standing\n'


def test_failed_save_never_emits_success_or_advances(sample, monkeypatch):
    other = second_image(sample); edit = edit_fake([sample.image, other])
    draft(edit, 'new draft')
    original = sample.image.with_suffix('.txt').read_bytes()
    monkeypatch.setattr(caption_sync, 'atomic_bytes', Mock(side_effect=OSError('disk full')))
    edit._save_and_next()
    assert edit.captions_saved.calls == [] and edit._dirty and edit._current_idx == 0
    assert edit.caption_edit.value == 'new draft' and edit.warnings
    assert sample.image.with_suffix('.txt').read_bytes() == original


def test_successful_save_next_emits_before_advancing(sample):
    edit = edit_fake([sample.image, second_image(sample)])
    seen = []; edit.captions_saved.connect(lambda paths: seen.append(edit._current_idx))
    draft(edit, 'new'); edit._save_and_next()
    assert seen == [0] and edit._current_idx == 1


def test_save_all_partial_failure_notifies_only_saved_paths_and_keeps_failed_draft(sample, monkeypatch):
    other = second_image(sample); edit = edit_fake([sample.image, other])
    first_cp, second_cp = str(sample.image.with_suffix('.txt')), str(other.with_suffix('.txt'))
    draft(edit, 'first new'); edit._captions[second_cp] = 'second draft'; edit._mark_dirty(second_cp)
    real = caption_sync.atomic_bytes
    def write(path, data):
        if path == other.with_suffix('.txt'): raise OSError('read-only')
        return real(path, data)
    monkeypatch.setattr(caption_sync, 'atomic_bytes', write)
    assert not edit._save_all()
    assert edit.captions_saved.calls == [([first_cp],)]
    assert edit._dirty_paths == {second_cp}
    assert edit._captions[second_cp] == 'second draft'
    assert other.with_suffix('.txt').read_text() == 'other caption'


def test_revert_current_keeps_other_image_drafts(sample):
    other = second_image(sample); edit = edit_fake([sample.image, other])
    cp = str(other.with_suffix('.txt'))
    draft(edit, 'first unsaved'); edit._captions[cp] = 'other unsaved'; edit._mark_dirty(cp)
    assert edit._revert_current()
    assert edit.caption_edit.value == '1girl, standing\n'
    assert edit._dirty and edit._dirty_paths == {cp}


def test_external_refresh_preserves_drafts_and_refreshes_clean_siblings(sample):
    other = second_image(sample); edit = edit_fake([sample.image, other])
    draft(edit, 'my unsaved draft')
    sample.image.with_suffix('.txt').write_text('outside one')
    other.with_suffix('.txt').write_text('outside two')
    changed = edit.refresh_saved_captions()
    assert edit.caption_edit.value == 'my unsaved draft'
    assert edit._captions[str(other.with_suffix('.txt'))] == 'outside two'
    assert changed == [str(other.with_suffix('.txt'))]
    assert edit.has_unsaved_edits() and edit.folder_changed.calls == []
    assert edit._save_current() is False  # Old disk snapshot still protects the conflict.


def test_clean_current_refresh_keeps_cursor_scroll_and_selection(sample):
    edit = edit_fake([sample.image])
    edit.caption_edit.position = 5; edit.caption_edit.anchor = 2; edit.caption_edit.scroll = 7
    sample.image.with_suffix('.txt').write_text('new caption that is long enough')
    edit.refresh_saved_captions()
    assert (edit.caption_edit.position, edit.caption_edit.anchor, edit.caption_edit.scroll) == (5, 2, 7)
    assert edit._current_idx == 0 and not edit._dirty
    assert edit.captions_saved.calls == []


def finish_write(clothing, operation, paths):
    retired = []; clothing.worker = SimpleNamespace(deleteLater=lambda: retired.append(True))
    clothing._last_operation = operation; clothing._caption_write_paths = paths
    clothing._job_finished()
    assert retired == [True] and clothing.worker is None


def test_actual_clothing_apply_finish_refreshes_both_tabs_in_place(sample):
    edit = edit_fake([sample.image]); clothing = clothing_fake(sample); studio = connect_tabs(edit, clothing)
    apply_caption(clothing._previews[str(sample.image)], sample.store)
    finish_write(clothing, 'apply', [str(sample.image.with_suffix('.txt'))])
    assert edit.caption_edit.value.startswith('shhooldress, serafaku, blue skirt')
    assert clothing.before_text.value == edit.caption_edit.value
    assert str(sample.image) in clothing._results
    assert not studio.has_unsaved_caption_edits()
    assert edit.folder_changed.calls == [] and edit.captions_saved.calls == []
    assert len(clothing.captions_changed.calls) == 1


def test_actual_undo_finish_refreshes_edit_and_keeps_model_result(sample):
    edit = edit_fake([sample.image]); clothing = clothing_fake(sample); connect_tabs(edit, clothing)
    journal = apply_caption(clothing._previews[str(sample.image)], sample.store)
    finish_write(clothing, 'apply', [str(sample.image.with_suffix('.txt'))])
    undo_caption(journal)
    finish_write(clothing, 'undo', None)
    assert edit.caption_edit.value == '1girl, standing\n'
    assert clothing.before_text.value == edit.caption_edit.value
    assert clothing.after_text.value.startswith('shhooldress')  # Proposal, NOT re-applied after Undo.
    assert sample.image.with_suffix('.txt').read_text() == '1girl, standing\n'
    assert clothing.image_table.item(0, 0).checkState() == 0


def test_cancel_or_failure_after_partial_apply_still_reconciles_without_completed_signal(sample):
    other = second_image(sample); edit = edit_fake([sample.image, other]); clothing = clothing_fake(sample, [other])
    connect_tabs(edit, clothing)
    apply_caption(clothing._previews[str(sample.image)], sample.store)
    # Native QThread finished occurs on success, error and cancellation. No completed event required.
    finish_write(clothing, 'apply', [str(p.with_suffix('.txt')) for p in (sample.image, other)])
    assert edit._captions[str(sample.image.with_suffix('.txt'))].startswith('shhooldress')
    assert edit._captions[str(other.with_suffix('.txt'))] == 'other caption'


def test_busy_reader_defers_events_until_native_worker_finished(sample):
    edit = edit_fake([sample.image]); clothing = clothing_fake(sample); connect_tabs(edit, clothing)
    clothing.worker = SimpleNamespace(deleteLater=lambda: None)
    clothing._last_operation = 'analyze'
    old = clothing._previews[str(sample.image)]
    draft(edit, 'new during job'); assert edit._save_current()
    assert clothing._previews[str(sample.image)] is old
    assert clothing._pending_caption_paths
    clothing._job_finished()
    assert clothing.before_text.value == 'new during job'
    assert clothing._pending_caption_paths == set() and clothing.captions_changed.calls == []


def test_expired_clothing_preview_is_not_applyable_and_is_visible_in_review_filter(sample):
    clothing = clothing_fake(sample)
    clothing.only_review = SimpleNamespace(isChecked=lambda: True)
    sample.profile.notes = 'changed'; sample.store.upsert(sample.profile)
    clothing.refresh_captions([str(sample.image.with_suffix('.txt'))])
    assert str(sample.image) not in clothing._previews
    assert str(sample.image) in clothing._sync_errors
    assert clothing.image_table.item(0, 0).checkState() == 0
    assert clothing.image_table.hidden[0] is False
    assert 'caption_sync_error' in clothing.detail_text.value


def test_unanalysed_image_current_caption_refreshes_without_inventing_match(sample):
    clothing = clothing_fake(sample); clothing._results.clear(); clothing._previews.clear()
    sample.image.with_suffix('.txt').write_text('new before analysis')
    clothing.refresh_captions()
    assert clothing.before_text.value == 'new before analysis' and clothing.after_text.value == ''
    assert clothing._results == {} and clothing._previews == {}


def test_manual_text_change_never_rewrites_outfit_library_or_feedback(sample):
    before_library = (sample.store.root / 'profiles.json').read_bytes()
    before_memory = sample.memory.fingerprint()
    edit = edit_fake([sample.image]); clothing = clothing_fake(sample); connect_tabs(edit, clothing)
    draft(edit, 'user tag, red skirt'); assert edit._save_current()
    assert (sample.store.root / 'profiles.json').read_bytes() == before_library
    assert sample.memory.fingerprint() == before_memory


def test_source_connections_exist_and_reload_loop_was_removed():
    source = (ROOT / EDIT[0]).read_text()
    assert 'self.edit_tab.captions_saved.connect(self.clothing_tab.refresh_captions)' in source
    tree = ast.parse(source)
    studio = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == STUDIO[1])
    fn = next(n for n in studio.body if isinstance(n, ast.FunctionDef) and n.name == '_on_clothing_changed')
    assert 'reload_folder' not in ast.unparse(fn)
    assert 'refresh_saved_captions(paths)' in ast.unparse(fn)


def test_dependent_balance_report_invalidated_only_for_matching_paths(sample):
    calls = []
    page = SimpleNamespace(report={'items': [SimpleNamespace(path=str(sample.image))]},
                           _invalidate_scan=lambda: calls.append('invalidate'), status=Text())
    studio = SimpleNamespace(balance_tab=page, lang='tr')
    bind_source(studio, *STUDIO, ['_invalidate_caption_dependents'])
    studio._invalidate_caption_dependents([str(sample.image.parent / 'elsewhere.txt')])
    assert calls == []
    studio._invalidate_caption_dependents([str(sample.image.with_suffix('.txt'))])
    assert calls == ['invalidate']


def test_dirty_whitespace_is_not_misclassified_as_saved(sample):
    edit = edit_fake([sample.image])
    draft(edit, '1girl, standing\n\n')
    assert edit.has_unsaved_edits()
    assert edit._save_current()
    assert sample.image.with_suffix('.txt').read_bytes() == b'1girl, standing\n\n'


def test_bom_crlf_saved_caption_remains_clean_on_read_notification(sample):
    sample.image.with_suffix('.txt').write_bytes(b'\xef\xbb\xbf1girl\r\n')
    edit = edit_fake([sample.image])
    edit.refresh_saved_captions()
    assert edit.has_unsaved_edits() is False


def test_zero_image_refresh_notifications_are_safe(sample):
    clothing = clothing_fake(sample); clothing._paths = []; clothing._active_image = None
    clothing.refresh_captions([])
    assert clothing.captions_changed.calls == []


def test_undo_to_absent_caption_is_reflected_in_editor(sample):
    sample.image.with_suffix('.txt').unlink()
    edit = edit_fake([sample.image]); clothing = clothing_fake(sample); connect_tabs(edit, clothing)
    journal = apply_caption(clothing._previews[str(sample.image)], sample.store)
    finish_write(clothing, 'apply', [str(sample.image.with_suffix('.txt'))])
    undo_caption(journal); finish_write(clothing, 'undo', None)
    assert edit.caption_edit.value == ''
    assert edit._snapshots[str(sample.image.with_suffix('.txt'))].raw is None
    assert not sample.image.with_suffix('.txt').exists()
