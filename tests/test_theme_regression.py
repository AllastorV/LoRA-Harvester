"""Display-independent theme regression tests (stdlib only).

Run from the project root: python -m unittest discover -s tests -v
The widget doubles test state/style logic, NOT Qt rendering or Windows latency.
See test_theme_qt.py for optional real Qt widget tests.
"""
from __future__ import annotations

import ast
import copy
import gc
import sys
import types
import unittest
import weakref
from pathlib import Path
from time import perf_counter
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.ui import theme
from src.ui import theme_bindings as binding


class Signal:
    def __init__(self, *args): self.slots = []
    def connect(self, callback): self.slots.append(callback)
    def emit(self, *args):
        for callback in tuple(self.slots): callback(*args)
    def __getitem__(self, key): return self


class Widget:
    """Only stores style, state and parent ownership; has no rendering backend."""
    def __init__(self, text='', parent=None):
        if isinstance(text, Widget): parent, text = text, ''
        self._text = text if isinstance(text, str) else ''
        self._parent = parent
        self._children = []
        if parent is not None: parent._children.append(self)
        self._sheet = ''
        self.style_writes = 0
        self.destroyed = Signal()
        self.clicked = Signal()
        self._layout = None
        self._props = {}
        self.enabled = True
        self._updates = True
        self.update_history = []
        self.selected = False
        self.scroll_position = 73
    def setStyleSheet(self, value): self._sheet = value; self.style_writes += 1
    def styleSheet(self): return self._sheet
    def setText(self, value): self._text = value
    def text(self): return self._text
    def isChecked(self): return self.selected
    def setChecked(self, value): self.selected = value
    def setEnabled(self, value): self.enabled = value
    def setUpdatesEnabled(self, value): self._updates = value; self.update_history.append(value)
    def updatesEnabled(self): return self._updates
    def isAncestorOf(self, other):
        while other is not None:
            if other._parent is self: return True
            other = other._parent
        return False
    def layout(self): return self._layout
    def setLayout(self, layout): self._layout = layout; layout.parent = self
    def property(self, key): return self._props.get(key)
    def setProperty(self, key, value): self._props[key] = value
    def setObjectName(self, name): self.name = name
    def setFixedSize(self, *args): pass
    def setFixedWidth(self, *args): pass
    def setFixedHeight(self, *args): pass
    def setAlignment(self, *args): pass
    def setFrameShape(self, *args): pass
    def setCursor(self, *args): pass
    def setToolTip(self, *args): pass
    def setWordWrap(self, *args): pass
    def setTextFormat(self, *args): pass
    def deleteLater(self): self.destroyed.emit(self)
    HLine = 1


class Layout:
    def __init__(self, parent=None):
        self.parent = parent
        self.items = []
        if parent is not None: parent._layout = self
    def addWidget(self, widget, *args, **kwargs):
        self.items.append(widget)
        if self.parent is not None:
            widget._parent = self.parent
            self.parent._children.append(widget)
    def addLayout(self, layout, *args, **kwargs): self.items.append(layout)
    def addStretch(self, *args): pass
    def setContentsMargins(self, *args): pass
    def setSpacing(self, *args): pass
    def setAlignment(self, *args): pass
    def count(self): return len(self.items)


class Timer:
    def __init__(self): self.active = False; self.starts = 0
    def start(self, interval): self.active = True; self.starts += 1
    def stop(self): self.active = False


QT = types.SimpleNamespace(AlignCenter=0, AlignHCenter=0, AlignTop=0, AlignRight=0,
                           PointingHandCursor=0, RichText=0)


def get_class(file, name):
    tree = ast.parse((ROOT / 'src/ui' / file).read_text(encoding='utf-8'))
    return next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)


def method(file, cls, name, extra=None):
    """Compile the actual project method, not a test-side reimplementation."""
    node = copy.deepcopy(next(n for n in get_class(file, cls).body
                              if isinstance(n, ast.FunctionDef) and n.name == name))
    node.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), node], type_ignores=[])
    ast.fix_missing_locations(module)
    scope = dict(theme=theme, QFrame=Widget, QLabel=Widget, QPushButton=Widget,
                 QWidget=Widget, QVBoxLayout=Layout, QHBoxLayout=Layout,
                 Qt=QT, perf_counter=perf_counter)
    scope.update(extra or {})
    exec(compile(module, str(ROOT/'src/ui'/file), 'exec'), scope)
    return scope[name]


def owner_methods(owner, file, cls, names, extra=None):
    for name in names:
        setattr(owner, name, types.MethodType(method(file, cls, name, extra), owner))
    return owner


class ThemeRegressionTests(unittest.TestCase):
    def setUp(self):
        self.previous = theme.state_key()
        binding._widgets.clear()
        theme.set_theme('dark', 1.0, '#10b981', persist=False)
    def tearDown(self):
        binding._widgets.clear()
        theme.set_theme(*self.previous, persist=False)

    def test_live_binding_replaces_old_palette_and_preserves_state(self):
        w = Widget('unsaved caption'); w.selected = True; w.enabled = False
        theme.bind_style(w, lambda: f'{theme.BG_CARD};{theme.ORANGE};{theme.TEXT_PRIMARY};{theme.fs(12)}')
        old = w.styleSheet()
        theme.set_theme('light', 1.4, '#a855f7', persist=False)
        result = theme.refresh_styles()
        self.assertNotEqual(old, w.styleSheet())
        self.assertIn(theme.BG_CARD, w.styleSheet())
        self.assertEqual((w.text(), w.selected, w.enabled, w.scroll_position),
                         ('unsaved caption', True, False, 73))
        self.assertEqual(result, dict(visited=1, updated=1, failed=0))

    def test_identical_styles_do_not_repolish(self):
        w = Widget(); theme.bind_style(w, theme.btn_primary)
        for _ in range(20): self.assertEqual(theme.refresh_styles()['updated'], 0)
        self.assertEqual(w.style_writes, 1)

    def test_dynamic_widgets_are_registered_after_startup(self):
        a = Widget(); theme.bind_style(a, theme.label_muted)
        theme.set_theme('dark', 1.0, '#e8832a', persist=False); theme.refresh_styles()
        b = Widget(); theme.bind_style(b, theme.btn_primary)
        theme.set_theme('dark', 1.0, '#a855f7', persist=False); theme.refresh_styles()
        self.assertIn('#a855f7', b.styleSheet())
        self.assertEqual(a.styleSheet(), theme.label_muted())

    def test_rebinding_keeps_latest_state_style(self):
        w=Widget(); theme.bind_style(w, theme.drop_zone_frame_default)
        theme.bind_style(w, theme.drop_zone_frame_success)
        theme.set_theme('light', 1.0, '#a855f7', persist=False)
        self.assertEqual(theme.refresh_styles()['visited'], 1)
        self.assertEqual(w.styleSheet(), theme.drop_zone_frame_success())

    def test_root_refresh_does_not_touch_other_pages(self):
        a,b=Widget(),Widget(); x,y=Widget(parent=a),Widget(parent=b)
        theme.bind_style(x, theme.btn_primary); theme.bind_style(y, theme.btn_primary)
        old=y.styleSheet(); theme.set_theme('dark',1.0,'#a855f7',persist=False)
        self.assertEqual(theme.refresh_styles(a)['visited'],1)
        self.assertEqual(old,y.styleSheet())
        self.assertNotEqual(old,x.styleSheet())

    def test_registry_does_not_keep_unused_widgets_alive(self):
        w=Widget(); theme.bind_style(w, theme.btn_primary)
        ref=weakref.ref(w); del w; gc.collect()
        self.assertIsNone(ref()); self.assertEqual(len(binding._widgets),0)

    def test_cpp_destroyed_signal_removes_live_wrapper(self):
        w=Widget(); theme.bind_style(w, theme.btn_primary)
        w.deleteLater()
        self.assertEqual(theme.refresh_styles()['visited'],0)

    def test_loop_factory_parameters_are_distinct(self):
        ws=[]
        for i in range(4):
            w=Widget(); theme.bind_style(w,lambda i=i:f'{theme.ORANGE};radius:{i}px'); ws.append(w)
        theme.set_theme('dark',1.0,'#a855f7',persist=False); theme.refresh_styles()
        for i,w in enumerate(ws): self.assertIn(f'radius:{i}px',w.styleSheet())

    def test_explicit_semantic_colors_are_not_accent_replacements(self):
        w=Widget(); theme.bind_style(w,lambda:f'background:{theme.BG_CARD};color:#7f1d1d;')
        theme.set_theme('dark',1.0,'#ec4899',persist=False); theme.refresh_styles()
        self.assertIn('color:#7f1d1d',w.styleSheet())

    def test_invalid_renderer_rejected(self):
        with self.assertRaises(TypeError): theme.bind_style(Widget(),'frozen stylesheet')
        with self.assertRaises(TypeError): theme.bind_style(Widget(),lambda:None)

    def test_broken_factory_does_not_block_other_widgets(self):
        bad,good=Widget(),Widget()
        theme.bind_style(bad,theme.btn_primary); theme.bind_style(good,theme.btn_primary)
        def broken(): raise ValueError('deliberate test')
        bad._lh_theme_style_factory=broken
        theme.set_theme('light',1.0,'#a855f7',persist=False)
        with self.assertLogs('src.ui.theme_bindings',level='ERROR'):
            stats=theme.refresh_styles()
        self.assertEqual(stats['failed'],1); self.assertEqual(good.styleSheet(),theme.btn_primary())

    def test_equivalent_color_and_scale_are_noop(self):
        theme.set_theme('dark',1.0,'#aabbcc',persist=False)
        with patch.object(theme,'save_prefs') as save:
            self.assertFalse(theme.set_theme('dark','1.0','#ABC'))
            save.assert_not_called()

    def test_invalid_preferences_are_normalized(self):
        self.assertEqual(theme.normalize_state('bad',float('nan'),'#zzzzzz'),('dark',1.0,'#e8832a'))
        self.assertEqual(theme.normalize_state('light',99,'#ABC'),('light',1.4,'#aabbcc'))
        self.assertEqual(theme.normalize_state('dark',-5,'#123456'),('dark',0.8,'#123456'))

    def test_global_stylesheet_is_applied_once(self):
        app=Widget()
        self.assertTrue(theme.apply_global_style(app))
        self.assertFalse(theme.apply_global_style(app))
        self.assertEqual(app.style_writes,1)

    def test_training_and_upscale_cards_follow_full_theme_cycle(self):
        train=method('training_page.py','TrainingPage','_make_card')(object())
        upscale_cls=next(n.name for n in ast.parse((ROOT/'src/ui/upscale_page.py').read_text()).body
                         if isinstance(n,ast.ClassDef) and any(isinstance(m,ast.FunctionDef) and m.name=='_card' for m in n.body))
        up,_=method('upscale_page.py',upscale_cls,'_card')(object())
        for mode,accent in [('dark','#e8832a'),('dark','#a855f7'),('light','#a855f7'),('dark','#10b981')]:
            with self.subTest(mode=mode,accent=accent):
                theme.set_theme(mode,1.2,accent,persist=False); theme.refresh_styles()
                for w in (train,up):
                    self.assertIn(theme.BG_CARD,w.styleSheet()); self.assertIn(theme.BORDER_LIGHT,w.styleSheet())
                    self.assertIn('10px',w.styleSheet())

    def test_settings_section_preserves_radius_and_content(self):
        content=Widget('user settings')
        make=method('main_window.py','VideoSmartCropperUI','_settings_section')
        card=make(object(),'','Appearance',content)
        theme.set_theme('light',1.4,'#a855f7',persist=False); theme.refresh_styles()
        self.assertIn('12px',card.styleSheet()); self.assertIn(theme.BG_CARD,card.styleSheet())
        self.assertIn(theme.TEXT_PRIMARY,card._children[0].styleSheet())
        self.assertEqual(content.text(),'user settings')

    def test_review_stat_labels_refresh_without_replacing_values(self):
        card=method('review_grid_page.py','ReviewGridPage','_make_stat_card')(object(),'Frames','260')
        ids=(id(card._val_lbl),id(card._txt_lbl))
        theme.set_theme('light',1.4,'#a855f7',persist=False); theme.refresh_styles()
        self.assertIn(theme.BG_CARD,card.styleSheet()); self.assertIn('8px',card.styleSheet())
        self.assertIn(theme.TEXT_PRIMARY,card._val_lbl.styleSheet())
        self.assertEqual((id(card._val_lbl),id(card._txt_lbl)),ids)
        self.assertEqual(card._val_lbl.text(),'260')

    def test_caption_chips_keep_text_and_identity(self):
        file='caption_studio_page.py'
        cls=next(n.name for n in ast.parse((ROOT/'src/ui'/file).read_text()).body
                 if isinstance(n,ast.ClassDef) and any(isinstance(m,ast.FunctionDef) and m.name=='_make_chip' for m in n.body))
        owner=types.SimpleNamespace(_remove_chip_tag=lambda _:None)
        chip=method(file,cls,'_make_chip')(owner,'machine, view')
        ids=tuple(map(id,chip._children))
        theme.set_theme('light',1.2,'#ec4899',persist=False); theme.refresh_styles()
        self.assertIn(theme.BG_SURFACE,chip.styleSheet()); self.assertIn('12px',chip.styleSheet())
        self.assertEqual(tuple(map(id,chip._children)),ids)
        self.assertEqual(chip._children[0].text(),'machine, view')

    def test_theme_mode_and_swatches_are_live(self):
        w=Widget(); w._theme_dark_btn=Widget(parent=w); w._theme_light_btn=Widget(parent=w)
        w._accent_swatches=[Widget(parent=w),Widget(parent=w)]
        for b,c in zip(w._accent_swatches,['#10b981','#a855f7']): b.setProperty('accent_color',c)
        owner_methods(w,'main_window.py','VideoSmartCropperUI',[
            '_update_theme_mode_btns','_theme_mode_btn_style','_refresh_accent_swatches','_accent_swatch_style'])
        w._update_theme_mode_btns(); w._refresh_accent_swatches()
        theme.set_theme('light',1.0,'#a855f7',persist=False); theme.refresh_styles()
        self.assertIn(theme.ORANGE_SUBTLE,w._theme_light_btn.styleSheet())
        self.assertIn('background: transparent',w._theme_dark_btn.styleSheet())
        self.assertIn(f'border: 2px solid {theme.TEXT_PRIMARY}',w._accent_swatches[1].styleSheet())
        self.assertIn('border: 2px solid transparent',w._accent_swatches[0].styleSheet())

    def test_coalescing_defers_palette_and_saves_only_once(self):
        w=types.SimpleNamespace(_pending_theme=None,_style_timer=Timer(),_refresh_all_styles=Mock())
        owner_methods(w,'main_window.py','VideoSmartCropperUI',['_queue_theme_change','_commit_theme_change'])
        before=theme.state_key()
        with patch.object(theme,'save_prefs') as save:
            w._queue_theme_change(accent='#a855f7'); w._queue_theme_change(mode='light')
            w._queue_theme_change(font_scale=1.4); w._queue_theme_change(accent='#ec4899')
            self.assertEqual(theme.state_key(),before); save.assert_not_called()
            w._commit_theme_change()
            self.assertEqual(theme.state_key(),('light',1.4,'#ec4899'))
            save.assert_called_once(); w._refresh_all_styles.assert_called_once()
            self.assertFalse(w._style_timer.active)

    def test_selection_back_to_current_cancels_pending_transaction(self):
        w=types.SimpleNamespace(_pending_theme=None,_style_timer=Timer(),_refresh_all_styles=Mock())
        owner_methods(w,'main_window.py','VideoSmartCropperUI',['_queue_theme_change','_commit_theme_change'])
        w._queue_theme_change(accent='#a855f7'); w._queue_theme_change(accent='#10b981')
        self.assertIsNone(w._pending_theme); self.assertFalse(w._style_timer.active)
        with patch.object(theme,'save_prefs') as save:
            w._commit_theme_change(); save.assert_not_called()
        w._refresh_all_styles.assert_not_called()

    def test_duplicate_pending_request_does_not_restart_timer(self):
        w=types.SimpleNamespace(_pending_theme=None,_style_timer=Timer())
        owner_methods(w,'main_window.py','VideoSmartCropperUI',['_queue_theme_change'])
        w._queue_theme_change(accent='#a855f7'); w._queue_theme_change(accent='#A855F7')
        self.assertEqual(w._style_timer.starts,1)

    def test_progress_steps_preserve_identity_and_active_step(self):
        node=copy.deepcopy(get_class('animations.py','ProgressSteps'))
        scope=dict(QWidget=Widget,QLabel=Widget,QFrame=Widget,QHBoxLayout=Layout,QVBoxLayout=Layout,Qt=QT,theme=theme,pyqtSignal=Signal)
        module=ast.Module(body=[node],type_ignores=[]); ast.fix_missing_locations(module)
        exec(compile(module,'ProgressSteps','exec'),scope)
        steps=scope['ProgressSteps'](['Source','Configure','Process'],current=1)
        ids=tuple(map(id,steps._step_circles+steps._step_labels+steps._step_lines))
        for accent in ['#e8832a','#a855f7','#ec4899']:
            theme.set_theme('dark',1.0,accent,persist=False); theme.refresh_styles()
        steps.set_step(2)
        self.assertEqual(ids,tuple(map(id,steps._step_circles+steps._step_labels+steps._step_lines)))
        self.assertEqual(steps._current,2)
        self.assertEqual([w.text() for w in steps._step_circles],['✓','✓','3'])
        self.assertIn(theme.ORANGE,steps._step_circles[2].styleSheet())

    def test_refresh_restores_updates_after_exception(self):
        import logging
        app=Widget(); w=Widget(); w.page_stack=object(); w._last_theme_key=('dark',1.0,'#old')
        refresh=method('main_window.py','VideoSmartCropperUI','_refresh_all_styles',dict(
            QApplication=types.SimpleNamespace(instance=lambda:app),
            finish_page_switch=lambda _:None,logging=logging))
        with patch.object(theme,'apply_global_style',side_effect=RuntimeError('deliberate')):
            with self.assertRaises(RuntimeError): refresh(w)
        self.assertTrue(w.updatesEnabled()); self.assertEqual(w.update_history,[False,True])

    def test_pending_theme_is_saved_on_close_without_repainting(self):
        w=types.SimpleNamespace(_pending_theme=('light',1.2,'#a855f7'),
                                _style_timer=Timer(),_refresh_all_styles=Mock())
        flush=method('main_window.py','VideoSmartCropperUI','_save_pending_theme_on_close')
        with patch.object(theme,'save_prefs') as save:
            flush(w); flush(w)
            self.assertEqual(theme.state_key(),('light',1.2,'#a855f7'))
            save.assert_called_once()
        w._refresh_all_styles.assert_not_called()
        self.assertIsNone(w._pending_theme)

    def test_drop_zone_rich_text_uses_current_palette_without_changing_files(self):
        from html import escape
        from src.ui.translations import get_text
        render=method('main_window.py','VideoSmartCropperUI','update_drop_zone_text',
                      dict(escape=escape,get_text=get_text))
        files=['C:/video/a&b.mp4']
        w=types.SimpleNamespace(drop_zone=Widget(),current_lang='tr',video_paths=files)
        render(w)
        theme.set_theme('light',1.0,'#a855f7',persist=False); render(w)
        for token in (theme.ORANGE,theme.TEXT_PRIMARY,theme.TEXT_MUTED):
            self.assertIn(token,w.drop_zone.text())
        self.assertIn('a&amp;b.mp4',w.drop_zone.text())
        self.assertIs(w.video_paths,files)
        self.assertEqual(w.video_paths,['C:/video/a&b.mp4'])

    def test_full_refresh_is_single_pass_and_skips_unchanged_state(self):
        import logging
        app=Widget(); w=Widget(); w.page_stack=object(); w._last_theme_key=theme.state_key()
        w._brand_label=types.SimpleNamespace(set_colors=Mock())
        w._nav_indicator=types.SimpleNamespace(set_color=Mock())
        w._progress_glow=types.SimpleNamespace(set_color=Mock())
        w._sidebar_pulse=types.SimpleNamespace(set_color=Mock())
        w.tag_freq_page=types.SimpleNamespace(_refresh_table_blacklist_color=Mock())
        w.update_drop_zone_text=Mock()
        child=Widget('unsaved caption',parent=w); child.selected=True
        theme.bind_style(child,theme.btn_primary)
        refresh=method('main_window.py','VideoSmartCropperUI','_refresh_all_styles',dict(
            QApplication=types.SimpleNamespace(instance=lambda:app),
            finish_page_switch=lambda _:None,logging=logging))
        theme.set_theme('light',1.2,'#a855f7',persist=False)
        with patch.object(theme,'refresh_styles',wraps=theme.refresh_styles) as bound, \
             patch.object(theme,'apply_global_style',wraps=theme.apply_global_style) as global_style:
            refresh(w); refresh(w)
            bound.assert_called_once(); global_style.assert_called_once_with(app)
        self.assertEqual(child.styleSheet(),theme.btn_primary())
        self.assertEqual(child.text(),'unsaved caption'); self.assertTrue(child.selected)
        self.assertEqual(w.update_history,[False,True])
        self.assertEqual(w._last_theme_refresh['failed'],0)
        w._progress_glow.set_color.assert_called_once_with(theme.get_accent())
        w.tag_freq_page._refresh_table_blacklist_color.assert_called_once()

    def test_base_factory_is_not_overwritten_by_animation_overlay(self):
        w=Widget(); theme.bind_style(w,theme.btn_primary)
        w.setStyleSheet('temporary pulse overlay')
        theme.set_theme('light',1.0,'#a855f7',persist=False)
        self.assertEqual(theme.render_style(w),theme.btn_primary())

    def test_no_theme_change_rebuilds_widgets_or_walks_widget_tree(self):
        methods=[('main_window.py','VideoSmartCropperUI','_refresh_all_styles'),
                 ('caption_studio_page.py','CaptionStudioPage','refresh_styles'),
                 ('training_page.py','TrainingPage','refresh_styles'),
                 ('review_grid_page.py','ReviewGridPage','refresh_styles')]
        forbidden={'_build','_build_chips','findChildren','_populate_table','_load_folder'}
        for file,cls,name in methods:
            n=next(n for n in get_class(file,cls).body if isinstance(n,ast.FunctionDef) and n.name==name)
            calls={c.func.attr for c in ast.walk(n) if isinstance(c,ast.Call) and isinstance(c.func,ast.Attribute)}
            self.assertFalse(calls&forbidden,(file,calls&forbidden))

    def test_no_unbound_theme_qss_left_in_ui(self):
        for p in (ROOT/'src/ui').glob('*.py'):
            if p.name=='theme_bindings.py': continue
            for n in ast.walk(ast.parse(p.read_text())):
                if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='setStyleSheet':
                    self.assertFalse(any(isinstance(a,ast.Attribute) and isinstance(a.value,ast.Name)
                                         and a.value.id in ('theme','_t') and a.attr!='render_style'
                                         for a in ast.walk(n)),f'{p.name}:{n.lineno}')

    def test_palette_only_factories_evaluate_in_both_modes(self):
        count=0
        for p in (ROOT/'src/ui').glob('*.py'):
            tree=ast.parse(p.read_text())
            for n in ast.walk(tree):
                if not (isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='bind_style' and len(n.args)==2): continue
                renderer=n.args[1]
                names={x.id for x in ast.walk(renderer) if isinstance(x,ast.Name) and isinstance(x.ctx,ast.Load)}
                if names-{'theme','_t'}: continue
                with self.subTest(file=p.name,line=n.lineno):
                    expression=ast.Expression(copy.deepcopy(renderer)); ast.fix_missing_locations(expression)
                    factory=eval(compile(expression,str(p),'eval'),{'theme':theme,'_t':theme})
                    for mode in ['dark','light']:
                        theme.set_theme(mode,1.3,'#a855f7',persist=False)
                        self.assertIsInstance(factory(),str)
                count+=1
        self.assertGreater(count,450)


if __name__ == '__main__': unittest.main(verbosity=2)
