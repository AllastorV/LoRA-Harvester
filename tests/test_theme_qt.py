"""Optional real PyQt5 widget regression tests.

Run: python -m unittest discover -s tests -v
These run offscreen; they do not measure native Windows painting latency.
A missing PyQt5 installation is reported as skipped, never as passed.
"""
from __future__ import annotations

import gc
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
try:
    from PyQt5.QtCore import QCoreApplication, QEvent
    from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QStackedWidget
    QT_AVAILABLE = True
except ImportError:
    QT_AVAILABLE = False

from src.ui import theme
from src.ui import theme_bindings as binding


@unittest.skipUnless(QT_AVAILABLE, 'PyQt5 unavailable: real widget tests not executed')
class RealQtThemeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.previous = theme.state_key()
        self.owned = []
        theme.set_theme('dark', 1.0, '#10b981', persist=False)

    def own(self, widget):
        self.owned.append(widget)
        return widget

    def tearDown(self):
        for widget in self.owned:
            widget.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        self.app.processEvents()
        self.owned.clear()
        gc.collect()
        theme.set_theme(*self.previous, persist=False)
        theme.apply_global_style(self.app)

    def test_real_button_preserves_user_state(self):
        button = self.own(QPushButton('Unsaved caption'))
        button.setCheckable(True)
        button.setChecked(True)
        button.setEnabled(False)
        theme.bind_style(button, theme.btn_primary)
        theme.set_theme('light', 1.2, '#a855f7', persist=False)
        theme.apply_global_style(self.app)
        stats = theme.refresh_styles(button)
        self.assertEqual(stats['failed'], 0)
        self.assertEqual(button.styleSheet(), theme.btn_primary())
        self.assertEqual(button.text(), 'Unsaved caption')
        self.assertTrue(button.isChecked())
        self.assertFalse(button.isEnabled())

    def test_dynamic_child_and_scoped_refresh(self):
        root = self.own(QWidget())
        other = self.own(QPushButton('Elsewhere'))
        theme.bind_style(other, theme.btn_primary)
        before = other.styleSheet()
        child = QPushButton('Added later', root)
        theme.bind_style(child, theme.btn_primary)
        theme.set_theme('light', 1.0, '#ec4899', persist=False)
        theme.refresh_styles(root)
        self.assertEqual(child.styleSheet(), theme.btn_primary())
        self.assertEqual(other.styleSheet(), before)

    def test_identical_global_and_local_styles_are_skipped(self):
        child = self.own(QPushButton('No-op'))
        theme.bind_style(child, theme.btn_primary)
        theme.apply_global_style(self.app)
        self.assertFalse(theme.apply_global_style(self.app))
        self.assertEqual(theme.refresh_styles(child)['updated'], 0)

    def test_qobject_deletion_removes_registry_entry(self):
        child = QPushButton('Temporary')
        theme.bind_style(child, theme.btn_primary)
        key = id(child)
        self.assertIn(key, binding._widgets)
        child.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        self.assertNotIn(key, binding._widgets)

    def test_progress_steps_keep_real_widget_identity(self):
        from src.ui.animations import ProgressSteps
        steps = self.own(ProgressSteps(['Source', 'Configure', 'Run'], current=1))
        before = tuple(id(w) for w in steps._step_circles + steps._step_labels + steps._step_lines)
        theme.set_theme('light', 1.3, '#a855f7', persist=False)
        theme.apply_global_style(self.app)
        self.assertEqual(theme.refresh_styles(steps)['failed'], 0)
        steps.set_step(2)
        after = tuple(id(w) for w in steps._step_circles + steps._step_labels + steps._step_lines)
        self.assertEqual(before, after)
        self.assertEqual(steps._current, 2)
        self.assertEqual([w.text() for w in steps._step_circles], ['✓', '✓', '3'])

    def test_interrupted_page_transition_releases_opacity_effect(self):
        from src.ui.animations import animate_page_switch, finish_page_switch
        stack = self.own(QStackedWidget())
        stack.addWidget(QWidget())
        stack.addWidget(QWidget())
        stack.resize(640, 480)
        stack.show()
        self.app.processEvents()
        animate_page_switch(stack, 0, 1)
        finish_page_switch(stack)
        self.assertEqual(stack.currentIndex(), 1)
        self.assertIsNone(stack.widget(1).graphicsEffect())
        self.assertIsNone(stack._lh_page_transition)
        # A second navigation must not inherit the previous animation's state.
        animate_page_switch(stack, 1, 0)
        finish_page_switch(stack)
        self.assertEqual(stack.currentIndex(), 0)
        self.assertIsNone(stack.widget(0).graphicsEffect())

    def test_cyclic_garbage_widget_destruction_does_not_abort(self):
        # PyQt5 aborts on an exception raised inside a slot, so the scenario runs
        # in a child process: bound widgets collected as a reference cycle must
        # not call a GC-cleared ``destroyed`` handler.
        import subprocess
        script = '\n'.join([
            'import gc',
            'from PyQt5.QtWidgets import QApplication, QWidget, QPushButton',
            'from src.ui import theme',
            'app = QApplication([])',
            'for _ in range(50):',
            '    w = QWidget(); b = QPushButton("x", w)',
            '    theme.bind_style(w, lambda: ""); theme.bind_style(b, lambda: "")',
            '    b.clicked.connect(lambda *_a, w=w: w.update())',
            '    del w, b',
            'gc.collect()',
            'app.processEvents()',
            'print("survived")',
        ])
        root = Path(__file__).resolve().parents[1]
        env = dict(os.environ, QT_QPA_PLATFORM='offscreen')
        result = subprocess.run([sys.executable, '-c', script], cwd=root, env=env,
                                capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        self.assertIn('survived', result.stdout)


if __name__ == '__main__':
    unittest.main(verbosity=2)
