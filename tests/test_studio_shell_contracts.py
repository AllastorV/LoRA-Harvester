"""Source-contract tests execute real shell methods with small non-Qt test doubles.

These are NOT GUI/rendering tests. Real offscreen widget tests live separately.
"""
import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from src.core.caption_sync import path_key

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / 'src/ui/main_window_v2.py'
TREE = ast.parse(PATH.read_text(encoding='utf-8'))
CLASS = next(n for n in TREE.body if isinstance(n, ast.ClassDef) and n.name == 'StudioMainWindow')
ROUTES = ast.literal_eval(next(n for n in TREE.body if isinstance(n, ast.Assign)
                              and any(isinstance(t, ast.Name) and t.id == 'ROUTES' for t in n.targets)).value)
WORKFLOWS = ast.literal_eval(next(n for n in TREE.body if isinstance(n, ast.Assign)
                                 and any(isinstance(t, ast.Name) and t.id == 'WORKFLOWS' for t in n.targets)).value)
ROUTE_GROUP = {route: group for group, routes in WORKFLOWS.items() for route in routes}


def method(name, extra=None):
    node = next(n for n in CLASS.body if isinstance(n, ast.FunctionDef) and n.name == name)
    ns = {'Path': Path, 'path_key': path_key, 'ROUTES': ROUTES, 'WORKFLOWS': WORKFLOWS, 'ROUTE_GROUP': ROUTE_GROUP, 'Qt': SimpleNamespace()}
    ns.update(extra or {})
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(PATH), 'exec'), ns)
    return ns[name]


def owner(**kwargs):
    state = SimpleNamespace(_studio_epoch=0, _scan_generation=2, _scan_worker=None,
        _scan_pending=False, _report_stale=False, _report=None, _close_pending=False,
        _dataset_root='', metrics=Mock(), suggestions=Mock(), suggestions_page=Mock(),
        library=Mock(), overview=Mock(), _scan_note=Mock(), _rescan_timer=Mock(),
        _update_draft_badge=Mock(), _notify=Mock(), current_lang='tr',
        _workflow_bar=Mock(), _workflow_buttons={key: Mock() for key in
            ('library', 'review', 'caption', 'edit', 'caption_quality', 'balance', 'frequency')})
    state._t = lambda tr, en: tr
    for key, value in kwargs.items():
        setattr(state, key, value)
    return state


@pytest.mark.parametrize('route,index,tab', [
    ('video',0,None), ('edit',1,1), ('caption',1,0), ('clothing',1,3), ('balance',1,4),
    ('compare',1,5), ('caption_quality',1,2), ('character',2,None), ('frequency',3,None),
    ('review',4,None), ('export',4,None), ('settings',5,None), ('upscale',6,None), ('training',7,None)])
def test_all_existing_tools_stay_reachable(route, index, tab):
    assert ROUTES[route] == (index, tab)


def test_new_suggestions_available_as_full_page_for_small_windows():
    assert ROUTES['suggestions'] == (10, None)
    state = owner(width=lambda: 1200, navigate=Mock(), _route='edit')
    method('_toggle_suggestions')(state, False)
    state.navigate.assert_called_once_with('suggestions')


def test_suggestions_open_as_page_while_editing():
    state = owner(width=lambda: 1600, navigate=Mock(), _route='edit', _refresh_responsive=Mock())
    method('_toggle_suggestions')(state, False)
    state.navigate.assert_called_once_with('suggestions')
    state._refresh_responsive.assert_not_called()


def test_side_panel_toggles_on_overview():
    state = owner(width=lambda: 1600, navigate=Mock(), _route='overview', _refresh_responsive=Mock())
    method('_toggle_suggestions')(state, False)
    assert state._suggestions_requested is False
    state.navigate.assert_not_called()
    state._refresh_responsive.assert_called_once()


def test_scan_results_cannot_replace_different_generation(tmp_path):
    state = owner(_dataset_root=str(tmp_path))
    report = SimpleNamespace(root=str(tmp_path))
    method('_scan_completed')(state, 1, report)
    assert state._report is None
    state.metrics.set_report.assert_not_called()


def test_scan_results_cannot_replace_another_folder(tmp_path):
    state = owner(_dataset_root=str(tmp_path / 'new'))
    report = SimpleNamespace(root=str(tmp_path / 'old'))
    method('_scan_completed')(state, 2, report)
    assert state._report is None


def test_new_scan_result_published_to_all_views_together(tmp_path):
    state = owner(_dataset_root=str(tmp_path), _report_stale=True)
    report = SimpleNamespace(root=str(tmp_path), created_at='2026-09-22T12:34:00+00:00')
    method('_scan_completed')(state, 2, report)
    assert state._report is report and not state._report_stale
    for widget in (state.metrics, state.suggestions, state.suggestions_page, state.library, state.overview):
        widget.set_report.assert_called_once_with(report)


def test_late_completion_ignored_during_close(tmp_path):
    state = owner(_dataset_root=str(tmp_path), _close_pending=True)
    method('_scan_completed')(state, 2, SimpleNamespace(root=str(tmp_path)))
    assert state._report is None


def test_save_invalidates_all_views_and_cancels_current_read(tmp_path):
    worker = Mock()
    state = owner(_scan_worker=worker)
    method('_files_changed')(state, [str(tmp_path / 'a.txt')])
    assert state._report_stale and state._scan_pending and state._scan_generation == 3
    state.metrics.set_report.assert_called_once_with(None)
    state.suggestions.mark_stale.assert_called_once()
    state.suggestions_page.mark_stale.assert_called_once()
    state.library.mark_stale.assert_called_once()
    worker.stop.assert_called_once()


def test_pending_scan_waits_for_writers_instead_of_losing_refresh():
    state = owner(_scan_pending=True, _writers_active=lambda: True, start_scan=Mock())
    method('_run_pending_scan')(state)
    state.start_scan.assert_not_called()
    state._rescan_timer.start.assert_called_once_with(1000)


def test_pending_scan_runs_when_idle():
    state = owner(_scan_pending=True, _writers_active=lambda: False, start_scan=Mock())
    method('_run_pending_scan')(state)
    state.start_scan.assert_called_once()


def test_scan_not_restarted_after_confirmed_close():
    state = owner(_scan_pending=True, _close_pending=True, _writers_active=Mock(), start_scan=Mock())
    method('_run_pending_scan')(state)
    state.start_scan.assert_not_called()


def test_stale_suggestion_never_opens_old_path_list():
    state = owner(_report_stale=True, start_scan=Mock())
    method('_activate_suggestion')(state, SimpleNamespace(paths=('old.png',)))
    state.start_scan.assert_called_once()
    state.library.filter_paths.assert_not_called()


def test_suggestion_opens_exact_report_paths_without_mutating_files():
    state = owner(_report=object(), _search=Mock(), _search_timer=Mock(), navigate=Mock())
    suggestion = SimpleNamespace(paths=('a.png','b.png'), title_tr='İncele', title_en='Review')
    method('_activate_suggestion')(state, suggestion)
    state.library.filter_paths.assert_called_once_with(suggestion.paths, 'İncele')
    state.navigate.assert_called_once_with('library')


def test_navigation_reactivates_same_nested_tab_for_saved_caption_sync():
    tabs = Mock(); tabs.currentIndex.return_value = 1
    studio = SimpleNamespace(tabs=tabs, _on_tab_activated=Mock(),
                             edit_tab=SimpleNamespace(_studio_layout=Mock()))
    state = owner(_route='library', caption_studio_page=studio, page_stack=Mock(),
                  _studio_nav={'edit': Mock()}, _refresh_responsive=Mock())
    method('navigate')(state, 'edit')
    studio._on_tab_activated.assert_called_once_with(1)
    state.page_stack.setCurrentIndex.assert_called_once_with(1)


def test_navigation_does_not_double_refresh_on_changed_tab():
    tabs = Mock(); tabs.currentIndex.return_value = 3
    studio = SimpleNamespace(tabs=tabs, _on_tab_activated=Mock(),
                             edit_tab=SimpleNamespace(_studio_layout=Mock()))
    state = owner(_route='clothing', caption_studio_page=studio, page_stack=Mock(),
                  _studio_nav={'edit': Mock()}, _refresh_responsive=Mock())
    method('navigate')(state, 'edit')
    studio._on_tab_activated.assert_not_called()


def test_grouped_navigation_selects_parent_and_local_tab():
    studio = SimpleNamespace(tabs=Mock(), _on_tab_activated=Mock())
    state = owner(_route='overview', caption_studio_page=studio, page_stack=Mock(),
                  _studio_nav={key: Mock() for key in ('overview', 'library', 'caption', 'balance')},
                  _refresh_responsive=Mock())
    method('navigate')(state, 'frequency')
    assert state._route == 'frequency'
    state._studio_nav['balance'].setChecked.assert_called_once_with(True)
    state._workflow_bar.setVisible.assert_called_once_with(True)
    state._workflow_buttons['frequency'].setChecked.assert_called_once_with(True)
    state.page_stack.setCurrentIndex.assert_called_once_with(3)
    method('navigate')(state, 'export')
    assert state._route == 'review'
    state._studio_nav['library'].setChecked.assert_called_with(True)
    state._workflow_buttons['review'].setChecked.assert_called_with(True)
    state.page_stack.setCurrentIndex.assert_called_with(4)


def test_video_navigation_sits_directly_below_overview():
    source = PATH.read_text(encoding='utf-8')
    assert "('overview', 'video', 'library', 'caption', 'clothing', 'balance', 'compare')" in source


def test_open_image_cannot_run_during_writer():
    state = owner(_writers_active=lambda: True, navigate=Mock())
    method('open_image')(state, 'not-used.png')
    state.navigate.assert_not_called()
    state._notify.assert_called_once()


def test_declined_folder_reload_does_not_open_wrong_image(tmp_path):
    image = tmp_path / 'image.png'; image.write_bytes(b'x')
    edit = SimpleNamespace(_items=[], reload_folder=Mock())
    state = owner(_writers_active=lambda: False, _dataset_root=str(tmp_path),
                  caption_studio_page=SimpleNamespace(edit_tab=edit), navigate=Mock())
    method('open_image')(state, str(image))
    edit.reload_folder.assert_called_once()
    state.navigate.assert_not_called()


def test_caption_studio_retains_both_original_sync_connections():
    source = (ROOT / 'src/ui/caption_studio_page.py').read_text()
    assert 'self.edit_tab.captions_saved.connect(self.clothing_tab.refresh_captions)' in source
    assert 'self.clothing_tab.captions_changed.connect(self._on_clothing_changed)' in source
    assert 'self.edit_tab.refresh_saved_captions(paths)' in source


def test_default_entry_point_and_classic_fallback():
    assert 'from src.ui.main_window_v2 import create_app' in (ROOT / 'main.py').read_text()
    assert "'--classic-ui' in sys.argv" in PATH.read_text()
    assert 'main.py %*' in (ROOT / 'run.bat').read_text()
    assert '--classic-ui' in (ROOT / 'run_classic.bat').read_text()


def test_settings_disclosure_never_disables_controls():
    source = (ROOT / 'src/ui/design_system/widgets.py').read_text()
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Disclosure')
    fn = next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_toggle')
    code = ast.get_source_segment(source, fn)
    assert 'setVisible(checked)' in code and 'setEnabled' not in code


def test_thumbnail_decode_is_bounded_and_uses_image_not_pixmap():
    source = (ROOT / 'src/ui/design_system/media.py').read_text()
    assert 'max_workers=2' in source and 'len(self.pending) < 40' in source
    assert 'memory_limit=48 * 1024 * 1024' in source
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MediaLoader')
    fn = next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_decode')
    assert 'QPixmap' not in ast.get_source_segment(source,fn)


def test_appearance_factory_keeps_custom_theme_preferences():
    source = (ROOT / 'src/ui/theme.py').read_text()
    tree = ast.parse(source)
    fn = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='enable_studio_design')
    assert 'if not _PREFS_PATH.exists():' in ast.get_source_segment(source,fn)


def clothing_ensure_method():
    source = ROOT / 'src/ui/clothing_tagger_widget.py'
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ClothingTaggerWidget')
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'ensure_image')
    class Item:
        def __init__(self, text): self.text = text
        def setFlags(self, value): self.flags = value
        def setCheckState(self, value): self.checked = value
        def setToolTip(self, value): self.tooltip = value
    ns = {'Path': Path, 'QTableWidgetItem': Item, 'Qt': SimpleNamespace(
        ItemIsEnabled=1, ItemIsUserCheckable=2, ItemIsSelectable=4, Unchecked=0)}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), ns)
    return ns['ensure_image']


def test_library_appends_clothing_image_without_destroying_other_results(tmp_path):
    p = tmp_path / 'next.png'; p.write_bytes(b'image')
    table = Mock(); table.blockSignals.return_value = False
    result, preview = object(), object()
    state = SimpleNamespace(is_busy=lambda: False, image_table=table,
        _paths=['previous.png'], _row_by_path={'previous.png': 0},
        _results={'previous.png':result}, _previews={'previous.png':preview},
        _sync_errors={'previous.png':'preserved'}, _active_image='previous.png')
    assert clothing_ensure_method()(state, p) == 1
    assert state._paths == ['previous.png',str(p.resolve())]
    assert state._results == {'previous.png':result}
    assert state._previews == {'previous.png':preview}
    assert state._active_image == 'previous.png'
    assert state._sync_errors == {'previous.png':'preserved'}
    table.clear.assert_not_called(); table.setRowCount.assert_not_called()
    table.insertRow.assert_called_once_with(1)
    assert table.blockSignals.call_args_list[-1].args == (False,)


def test_library_clothing_existing_row_revealed_without_duplicate(tmp_path):
    p = tmp_path / 'known.png'; p.write_bytes(b'image')
    table = Mock()
    state = SimpleNamespace(is_busy=lambda: False, image_table=table,
        _paths=[str(p)], _row_by_path={str(p):0})
    assert clothing_ensure_method()(state, p) == 0
    table.setRowHidden.assert_called_once_with(0, False)
    table.insertRow.assert_not_called()
    assert state._paths == [str(p)]


def test_library_clothing_busy_or_missing_does_not_change_table(tmp_path):
    state = SimpleNamespace(is_busy=lambda: True, image_table=Mock())
    assert clothing_ensure_method()(state, tmp_path/'missing.png') is None
    state.is_busy = lambda: False
    assert clothing_ensure_method()(state, tmp_path/'missing.png') is None
    state.image_table.insertRow.assert_not_called()


def test_shell_opens_new_clothing_selection_through_preserving_helper(tmp_path):
    p = tmp_path / 'new.png'; p.write_bytes(b'image')
    edit = SimpleNamespace(_items=[(str(p), '')])
    clothing = SimpleNamespace(ensure_image=Mock(return_value=2), tabs=Mock(),
                               image_table=Mock(), _select_image=Mock())
    clothing.image_table.currentRow.return_value = 0
    state = owner(_writers_active=lambda: False, navigate=Mock(),
                  caption_studio_page=SimpleNamespace(edit_tab=edit, clothing_tab=clothing))
    method('open_image')(state, str(p), 'clothing')
    clothing.ensure_image.assert_called_once_with(p)
    clothing.image_table.setCurrentCell.assert_called_once_with(2, 1)
    clothing._select_image.assert_not_called()  # row signal performs this once
    state.navigate.assert_called_once_with('clothing')


def test_declined_generate_folder_switch_preserves_other_tools(tmp_path):
    previous = str(tmp_path / 'before')
    edit = SimpleNamespace(_folder=previous, reload_folder=Mock())
    studio = SimpleNamespace(edit_tab=edit, quality_tab=Mock(), balance_tab=Mock())
    state = owner(_dataset_root=previous, _writers_active=lambda: False,
        caption_studio_page=studio, _set_generator_folder=Mock())
    method('_accept_generator_folder')(state, str(tmp_path / 'after'))
    state._set_generator_folder.assert_called_once_with(previous)
    studio.quality_tab.reload_folder.assert_not_called()
    studio.balance_tab.reload_folder.assert_not_called()


def test_empty_generator_folder_is_not_scanned_as_working_directory():
    gen = Mock(); gen.selected_folder = 'rejected-folder'
    gen.blockSignals.return_value = False
    state = owner(caption_studio_page=SimpleNamespace(generate_tab=gen))
    method('_set_generator_folder')(state, '')
    gen._set_folder.assert_not_called()
    gen.start_btn.setEnabled.assert_called_once_with(False)
    assert gen.selected_folder == ''


def test_generator_has_one_folder_commit_path_in_new_shell():
    code = ast.get_source_segment(PATH.read_text(),
        next(n for n in CLASS.body if isinstance(n, ast.FunctionDef) and n.name == 'init_ui'))
    assert 'gen.folder_changed.disconnect(dependent.reload_folder)' in code
    assert 'self.caption_studio_page.quality_tab, self.caption_studio_page.balance_tab' in code
    assert 'gen.folder_changed.connect(self._accept_generator_folder)' in code
