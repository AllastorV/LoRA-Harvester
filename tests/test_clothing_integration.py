"""Source-level integration checks and optional real offscreen Qt widget tests."""
from __future__ import annotations
import ast
import os
import sys
import tempfile
import unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def cls_file(file, name):
    tree = ast.parse((ROOT / file).read_text(encoding='utf-8'))
    return next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)


def method(file, cls, name):
    node = cls_file(file, cls)
    return next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == name)


class SourceIntegrationTests(unittest.TestCase):
    def test_caption_studio_contains_a_real_clothing_tab(self):
        text = ast.unparse(method('src/ui/caption_studio_page.py', 'CaptionStudioPage', '_init_ui'))
        self.assertIn('ClothingTaggerWidget', text)
        self.assertIn('self.tabs.addTab(self.clothing_tab', text)
        self.assertIn('self.edit_tab.folder_changed.connect(self.clothing_tab.reload_folder)', text)

    def test_new_worker_does_not_shadow_qthread_finished(self):
        node = cls_file('src/workers/clothing_tagger_worker.py', 'ClothingTaggerWorker')
        fields = [target.id for statement in node.body if isinstance(statement, ast.Assign)
                  for target in statement.targets if isinstance(target, ast.Name)]
        self.assertNotIn('finished', fields)
        self.assertIn('completed', fields)

    def test_caption_worker_native_finished_is_used_for_cleanup(self):
        text = ast.unparse(method('src/ui/caption_studio_page.py', '_GenerateTab', 'start_captioning'))
        self.assertIn('captioning_finished.connect(self._on_finished)', text)
        self.assertIn('finished.connect(self._safe_delete_thread)', text)
        node = cls_file('src/ui/caption_studio_page.py', 'CaptioningThread')
        fields = [target.id for statement in node.body if isinstance(statement, ast.Assign)
                  for target in statement.targets if isinstance(target, ast.Name)]
        self.assertNotIn('finished', fields)

    def test_all_final_crops_collected_before_thumbnail_throttle(self):
        text = ast.unparse(method('src/ui/main_window.py', 'ProcessingThread', '_on_frame_saved'))
        self.assertLess(text.index('self._clothing_paths.append(path)'), text.index('if now -'))

    def test_video_model_cleanup_precedes_clothing_and_completion(self):
        text = ast.unparse(method('src/ui/main_window.py', 'ProcessingThread', 'run'))
        self.assertLess(text.index('self._cleanup()'), text.index('optional_pipeline_pass('))
        self.assertLess(text.index('optional_pipeline_pass('), text.index('self.processing_finished.emit'))

    def test_caption_model_cleanup_precedes_clothing(self):
        text = ast.unparse(method('src/ui/caption_studio_page.py', 'CaptioningThread', 'run'))
        self.assertLess(text.index('self._release_caption_models()'), text.index('optional_pipeline_pass('))

    def test_all_changed_python_files_parse(self):
        for path in ROOT.rglob('*.py'):
            with self.subTest(path=str(path.relative_to(ROOT))):
                ast.parse(path.read_text(encoding='utf-8'))

    def test_new_ui_has_no_eager_ml_dependency_import(self):
        tree = ast.parse((ROOT / 'src/ui/clothing_tagger_widget.py').read_text())
        modules = [node.module for node in tree.body if isinstance(node, ast.ImportFrom)]
        self.assertFalse(any(m and m.startswith(('torch', 'transformers', 'ultralytics', 'onnxruntime'))
                             for m in modules))

    def test_apply_requires_user_confirmation(self):
        text = ast.unparse(method('src/ui/clothing_tagger_widget.py', 'ClothingTaggerWidget', '_apply_checked'))
        self.assertIn('QMessageBox.question', text)
        self.assertIn('QMessageBox.No', text)

    def test_clothing_library_change_does_not_rebuild_theme(self):
        text = ast.unparse(method('src/ui/clothing_tagger_widget.py', 'ClothingTaggerWidget', 'refresh_styles'))
        self.assertIn('theme.refresh_styles(self)', text)
        self.assertNotIn('_build', text)


os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
try:
    from PyQt5.QtCore import QCoreApplication, QEvent
    from PyQt5.QtWidgets import QApplication
    from src.ui.clothing_tagger_widget import ClothingTaggerWidget
    from src.core.clothing_profiles import ClothingProfileStore
    from src.ui import theme
    QT_AVAILABLE = True
except ImportError:
    QT_AVAILABLE = False


@unittest.skipUnless(QT_AVAILABLE, 'PyQt5 unavailable: actual clothing widget tests not executed')
class ClothingQtTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = ClothingProfileStore(Path(self.temp.name) / 'library')
        self.widget = ClothingTaggerWidget('tr', store=self.store)

    def tearDown(self):
        self.widget.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        self.app.processEvents()
        self.temp.cleanup()

    def test_widget_starts_without_model_or_server(self):
        self.assertEqual(self.widget.tabs.count(), 3)
        self.assertIsNone(self.widget.worker)
        self.assertFalse(self.widget.auto_video.isChecked())
        self.assertFalse(self.widget.auto_caption.isChecked())

    def test_split_preserves_literal_typos_and_master(self):
        self.widget.full_prompt.setPlainText('shhooldress,serafaku,blue skirt,black thighighhs')
        self.widget._split_prompt()
        self.assertEqual(self.widget.master_edit.text(), 'shhooldress')
        self.assertEqual(self.widget.parts_table.item(2, 0).text(), 'black thighighhs')

    def test_theme_refresh_preserves_editor_state(self):
        original = theme.state_key()
        try:
            self.widget.name_edit.setText('unsaved outfit')
            theme.set_theme('light', 1.0, '#a855f7', persist=False)
            report = self.widget.refresh_styles()
            self.assertEqual(report['failed'], 0)
            self.assertEqual(self.widget.name_edit.text(), 'unsaved outfit')
            self.assertIn(theme.BG_CARD, self.widget.canvas.styleSheet())
        finally:
            theme.set_theme(*original, persist=False)

    def test_language_change_preserves_profile_text(self):
        self.widget.master_edit.setText('literal_tag')
        self.widget.update_language('en')
        self.assertEqual(self.widget.master_edit.text(), 'literal_tag')
        self.assertEqual(self.widget.tabs.tabText(0), 'Outfit library')


if __name__ == '__main__':
    unittest.main()
