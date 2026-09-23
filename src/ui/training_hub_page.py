"""Two explicit training backends. Existing Kohya controls/config remain intact."""
from __future__ import annotations
from pathlib import Path
import json

from PyQt5.QtCore import QThread, QTimer, pyqtSignal
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QLabel, QStackedWidget, QMessageBox
from .training_page import TrainingPage
from .ai_toolkit_training_page import AIToolkitTrainingPage
from . import theme


class TrainingHubPage(QWidget):
    busy_changed = pyqtSignal(bool)

    def __init__(self, lang='en', parent=None):
        super().__init__(parent)
        self.lang=lang
        self._busy_owner=None
        self._busy_flag=False
        root=QVBoxLayout(self); root.setContentsMargins(0,0,0,0)
        row=QHBoxLayout(); row.setContentsMargins(20,10,20,0)
        self.label=QLabel(); self.engine=QComboBox()
        self.engine.addItem('Kohya · sd-scripts','kohya')
        self.engine.addItem('AI Toolkit · Ostris','ai-toolkit')
        row.addWidget(self.label); row.addWidget(self.engine,1); root.addLayout(row)
        self.stack=QStackedWidget(); root.addWidget(self.stack)
        self.kohya=TrainingPage(lang,self)
        self.toolkit=AIToolkitTrainingPage(lang,self)
        self.stack.addWidget(self.kohya); self.stack.addWidget(self.toolkit)
        self.kohya.operation_guard=self._guard
        self.kohya.lifecycle_callback=lambda busy:self._busy(self.kohya,busy)
        self.toolkit.operation_guard=self._guard
        self.toolkit.busy_changed.connect(lambda busy:self._busy(self.toolkit,busy))
        self.engine.currentIndexChanged.connect(self._switch)
        self._update_label()

    def _update_label(self):
        self.label.setText('Eğitim motoru' if self.lang=='tr' else 'Training engine')

    def _worker(self,attr):
        # Used by classic/Studio close and resource guards. Return the real
        # QThread for whichever backend still owns an active task.
        workers=[getattr(page,attr,None) for page in (self.kohya,self.toolkit)]
        for w in workers:
            try:
                if w is not None and w.isRunning(): return w
            except RuntimeError: pass
        return None

    @property
    def _thread(self): return self._worker('_thread')
    @property
    def _install_thread(self): return self._worker('_install_thread')
    @property
    def _repair_thread(self): return self._worker('_repair_thread')

    def _guard(self):
        window=self.window()
        reason=''
        if self._busy_flag or getattr(window,'_close_pending',False):
            reason='Bir iş zaten çalışıyor / A task is already running.'
        elif callable(getattr(window,'_writers_active',None)) and window._writers_active():
            reason='Önce diğer işlemi tamamla / Finish the other task first.'
        else:
            studio=getattr(window,'caption_studio_page',None)
            if studio and (studio.has_unsaved_caption_edits() or studio.any_tool_busy()):
                reason='Önce caption taslaklarını kaydet ve diğer işi tamamla / Save caption drafts and finish other tasks.'
            for name in ('processing_thread',):
                w=getattr(window,name,None)
                if w is not None and w.isRunning(): reason='Video işlemi çalışıyor / Video processing is active.'
            # Classic window does not have _writers_active.
            for page in (getattr(window,'char_sort_page',None),getattr(window,'upscale_page',None),
                         getattr(studio,'generate_tab',None),self.kohya):
                for attr in ('_thread','_install_thread','_repair_thread','captioning_thread'):
                    w=getattr(page,attr,None)
                    if w is not None and w.isRunning(): reason='Önce çalışan işi tamamla / Finish the running task first.'
        if reason:
            if not getattr(window,'_close_pending',False): QMessageBox.warning(self,'Training',reason)
            return False
        return True

    def _busy(self,owner,busy):
        self._busy_flag=busy; self._busy_owner=owner if busy else None
        self.engine.setEnabled(not busy)
        studio=getattr(self.window(),'caption_studio_page',None)
        if studio: studio._on_tools_busy(self,busy)
        self.busy_changed.emit(busy)

    def _switch(self,index):
        if self._busy_flag or self._thread or self._install_thread or self._repair_thread:
            old=self.engine.blockSignals(True); self.engine.setCurrentIndex(self.stack.currentIndex())
            self.engine.blockSignals(old); return
        self.stack.setCurrentIndex(index)
        self._inherit_dataset()
        # Deliberately no parameter conversion between epochs/TOML and steps/YAML.

    def _inherit_dataset(self):
        if self._busy_flag:
            return
        studio = getattr(self.window(), 'caption_studio_page', None)
        path = getattr(getattr(studio, 'edit_tab', None), '_folder', '')
        if path and self.stack.currentWidget() is self.toolkit and not self.toolkit._value('dataset_dir'):
            self.toolkit.set_dataset_path(path)

    def showEvent(self, event):
        super().showEvent(event)
        self._inherit_dataset()

    def set_dataset_path(self,path):
        if not self._busy_flag:
            self.kohya.set_dataset_path(path)
            self.toolkit.set_dataset_path(path)

    def update_language(self,lang):
        self.lang=lang; self._update_label()
        # Kohya's old update_language probes Python; avoid resetting a live trainer.
        if not self._busy_flag: self.kohya.update_language(lang)
        self.toolkit.update_language(lang)

    def refresh_styles(self):
        self.kohya.refresh_styles(); self.toolkit.refresh_styles()
        return theme.refresh_styles(self)
