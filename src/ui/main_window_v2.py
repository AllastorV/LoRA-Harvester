"""Native Studio desktop shell over the existing, synchronized Harvester tools.

The old pages/controllers are reused, not duplicated. No separate caption store,
fictional projects, cloud accounts, model chat, or invented health scores.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
import threading

from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal, QUrl
from PyQt5.QtGui import QIcon, QKeySequence, QDesktopServices
from PyQt5.QtWidgets import (QApplication, QWidget, QFrame, QVBoxLayout, QHBoxLayout,
    QScrollArea, QSplitter, QPushButton, QLineEdit, QToolButton, QMenu, QFileDialog,
    QMessageBox, QShortcut, QSizePolicy)

from src.core.caption_sync import path_key
from src.core.smart_suggestions import scan_suggestions, ScanCancelled
from src.ui import theme
from src.ui.main_window import VideoSmartCropperUI, drop_opencv_qt_overrides
from src.ui.design_system.widgets import label, button, NavButton, Disclosure, line_icon
from src.ui.design_system.media import MediaLoader, attach_image_thumbnails
from src.ui.design_system.editor_layout import EditorStudioLayout
from src.ui.pages.studio_pages import OverviewPage, LibraryPage, MetricStrip, SuggestionPanel, text

ROOT = Path(__file__).resolve().parents[2]
ROUTES = {
    'overview': (8, None), 'library': (9, None), 'edit': (1, 1), 'caption': (1, 0),
    'clothing': (1, 3), 'balance': (1, 4), 'compare': (1, 5), 'caption_quality': (1, 2),
    'video': (0, None), 'character': (2, None), 'frequency': (3, None),
    'review': (4, None), 'settings': (5, None), 'upscale': (6, None), 'training': (7, None),
    'export': (4, None), 'suggestions': (10, None),
}
LABELS = {
    'overview': ('Genel bakış', 'Overview'), 'library': ('Görseller', 'Images'),
    'edit': ('Veri düzenleme', 'Edit dataset'), 'caption': ('Caption', 'Captions'),
    'clothing': ('Kıyafet eşleştirme', 'Outfit matching'), 'balance': ('Veri analizi', 'Data analysis'),
    'compare': ('Karşılaştırma', 'Compare models'), 'export': ('Dışa aktar', 'Export'),
    'settings': ('Ayarlar', 'Settings'), 'video': ('Videodan görsel çıkar', 'Harvest video'),
    'character': ('Karakter sıralama', 'Character sorting'), 'frequency': ('Etiket sıklığı', 'Tag frequency'),
    'review': ('İncele ve dışa aktar', 'Review and export'),
    'caption_quality': ('Caption kalite kontrolü', 'Caption audit'),
    'upscale': ('Çözünürlük artırma', 'Upscale'), 'training': ('LoRA eğitimi · Kohya / AI Toolkit', 'LoRA training · Kohya / AI Toolkit'),
}


WORKFLOWS = {
    'library': ('library', 'review'),
    'caption': ('caption', 'edit', 'caption_quality'),
    'balance': ('balance', 'frequency'),
}
ROUTE_GROUP = {route: group for group, routes in WORKFLOWS.items() for route in routes}
TAB_LABELS = {
    'library': ('Kütüphane', 'Library'), 'review': ('İncele ve dışa aktar', 'Review and export'),
    'caption': ('Otomatik etiketleme', 'Auto caption'), 'edit': ('Düzenle', 'Edit'),
    'caption_quality': ('Kalite kontrol', 'Quality check'),
    'balance': ('Denge', 'Balance'), 'frequency': ('Etiket sıklığı', 'Tag frequency'),
}
NAV_TIPS = {
    'overview': ('Veri setini aç ve tara.', 'Open and scan a dataset.'),
    'video': ('Videodan kare çıkar.', 'Extract frames from video.'),
    'library': ('Görselleri bul, incele veya dışa aktar.', 'Find, review or export images.'),
    'caption': ('Caption üret, düzenle ve denetle.', 'Generate, edit and audit captions.'),
    'clothing': ('Kıyafet profilleriyle görselleri eşleştir.', 'Match images to outfit profiles.'),
    'balance': ('Dengeyi ve etiket sıklığını incele.', 'Inspect balance and tag frequency.'),
    'compare': ('Modelleri aynı ayarlarla karşılaştır.', 'Compare models with identical settings.'),
    'settings': ('Uygulama ayarlarını aç.', 'Open app settings.'),
}


class SuggestionWorker(QThread):
    progress = pyqtSignal(int, int, str)
    completed = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)

    def __init__(self, generation, root, options, masters, parent=None):
        super().__init__(parent)
        self.generation, self.root, self.options, self.masters = generation, root, options, masters
        self.cancel = threading.Event()

    def stop(self):
        self.cancel.set()
        self.requestInterruption()

    def run(self):
        try:
            report = scan_suggestions(self.root, options=self.options, masters=self.masters,
                cache_path=ROOT / 'data' / 'studio' / 'metadata.sqlite3', cancel=self.cancel,
                progress=self.progress.emit)
            if not self.cancel.is_set():
                self.completed.emit(self.generation, report)
        except ScanCancelled:
            self.failed.emit(self.generation, 'cancelled')
        except Exception as exc:
            self.failed.emit(self.generation, str(exc))


class StudioMainWindow(VideoSmartCropperUI):
    def __init__(self):
        theme.enable_studio_design()
        self._studio_ready = False
        self._dataset_root = ''
        self._route = 'overview'
        self._report = None
        self._report_stale = False
        self._scan_worker = None
        self._scan_generation = 0
        self._scan_pending = False
        self._suggestions_requested = True
        self._studio_epoch = 0
        super().__init__()

    def _settings_section(self, icon, title, content):
        # All original controls and handlers remain accessible; only presentation changes.
        widget = Disclosure(title, content, expanded=icon in ('🎨', '🌐'))
        key_by_icon = {'🎨': 'settings_sec_appearance', '🌐': 'settings_sec_language',
            '📁': 'settings_sec_output_paths', '⚡': 'settings_sec_gpu', '🚀': 'settings_sec_performance',
            '🧵': 'settings_sec_cpu', '💾': 'settings_sec_memory', '🔧': 'settings_sec_misc'}
        widget._translation_key = key_by_icon.get(icon, '')
        return widget

    def init_ui(self):
        # Build each proven tool once. Only navigation/chrome is replaced afterwards.
        super().init_ui()
        self._legacy_shell = self.takeCentralWidget()
        self._legacy_shell.setParent(self)
        self._legacy_shell.hide()
        self._legacy_sidebar = self._sidebar
        self._legacy_sidebar.hide()
        self._brand_label.stop_shimmer()
        if hasattr(self, '_nav_indicator'):
            self._nav_indicator.hide()
        self.media = MediaLoader(self)
        edit = self.caption_studio_page.edit_tab
        edit._studio_layout = EditorStudioLayout(edit, self.media)
        self.caption_studio_page.tabs.tabBar().hide()
        # Individual tool routes replace the redundant outer tab strip.
        studio_scroll = self.page_stack.widget(1)
        studio_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.overview = OverviewPage(self.current_lang)
        self.library = LibraryPage(self.media, self.current_lang)
        pages = self.caption_studio_page
        attach_image_thumbnails(self.library.table, self.media,
            lambda i: self.library.model.records[self.library.proxy.mapToSource(i).row()].path, 0, size=60)
        attach_image_thumbnails(pages.balance_tab.table, self.media,
            lambda i: pages.balance_tab.model.items[i.row()].path, 0)
        attach_image_thumbnails(pages.clothing_tab.image_table, self.media,
            lambda i: pages.clothing_tab._paths[i.row()], 1)
        attach_image_thumbnails(pages.quality_tab._table, self.media,
            lambda i: pages.quality_tab._table.item(i.row(), 0).data(Qt.UserRole).image, 0)
        attach_image_thumbnails(pages.clothing_tab.reference_list, self.media,
            lambda i: pages.clothing_tab._refs[i.row()].path)
        attach_image_thumbnails(pages.compare_tab.sample_list, self.media,
            lambda i: pages.compare_tab.samples[i.row()][0])
        overview_scroll = QScrollArea()
        overview_scroll.setWidgetResizable(True)
        overview_scroll.setFrameShape(QFrame.NoFrame)
        overview_scroll.setWidget(self.overview)
        self.page_stack.addWidget(overview_scroll)  # existing indices 0..7 stay intact
        self.page_stack.addWidget(self.library)
        self.suggestions_page = SuggestionPanel(self.current_lang)
        self.suggestions_page.setContentsMargins(20, 18, 20, 18)
        self.page_stack.addWidget(self.suggestions_page)

        central = QWidget()
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        topbar = QFrame()
        topbar.setObjectName('studioTopbar')
        topbar.setMinimumHeight(62)
        top = QHBoxLayout(topbar)
        top.setContentsMargins(18, 10, 18, 10)
        top.setSpacing(14)
        self._studio_top_layout = top
        self._studio_brand = label('LoRA Harvester Studio', role='heading')
        self._studio_brand.setWordWrap(False)
        top.addWidget(self._studio_brand)
        self._folder_button = button('', self.open_dataset)
        self._folder_button.setIcon(line_icon('folder'))
        self._folder_button.setMaximumWidth(260)
        top.addWidget(self._folder_button)
        self._search = QLineEdit()
        self._search.setClearButtonEnabled(True)
        self._search.setMinimumWidth(170)
        top.addWidget(self._search, 1)
        self._draft_badge = label('', role='muted')
        self._draft_badge.setWordWrap(False)
        top.addWidget(self._draft_badge)
        self._theme_button = button('', self.toggle_theme)
        top.addWidget(self._theme_button)
        self._suggestions_button = QPushButton()
        self._suggestions_button.setCheckable(True)
        self._suggestions_button.setChecked(True)
        self._suggestions_button.clicked.connect(self._toggle_suggestions)
        theme.bind_style(self._suggestions_button, theme.btn_secondary)
        top.addWidget(self._suggestions_button)
        theme.bind_style(topbar, lambda: f'QFrame#studioTopbar{{background:{theme.BG_CARD};border:none;border-bottom:1px solid {theme.BORDER};}}')
        theme.bind_style(self._search, lambda: f'QLineEdit{{background:{theme.BG_SURFACE};border:1px solid {theme.BORDER};border-radius:7px;padding:8px 10px;color:{theme.TEXT_PRIMARY};}}QLineEdit:focus{{border-color:{theme.get_accent()};}}')
        outer.addWidget(topbar)

        middle = QHBoxLayout()
        middle.setContentsMargins(0, 0, 0, 0)
        middle.setSpacing(0)
        self._sidebar = QFrame()
        self._sidebar.setObjectName('studioSidebar')
        self._sidebar.setFixedWidth(204)
        navlayout = QVBoxLayout(self._sidebar)
        navlayout.setContentsMargins(10, 16, 10, 12)
        navlayout.setSpacing(3)
        self._studio_nav = {}
        for route in ('overview', 'video', 'library', 'caption', 'clothing', 'balance', 'compare'):
            btn = NavButton(route, '')
            btn.clicked.connect(lambda checked=False, route=route: self.navigate(route))
            self._studio_nav[route] = btn
            navlayout.addWidget(btn)
        self._tool_button = QToolButton()
        self._tool_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._tool_button.setPopupMode(QToolButton.InstantPopup)
        self._tool_button.setIcon(line_icon('tools'))
        self._tool_button.setMinimumHeight(42)
        self._tool_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._tools_menu = QMenu(self._tool_button)
        self._tool_button.setMenu(self._tools_menu)
        theme.bind_style(self._tool_button, lambda: f'QToolButton{{border:none;background:transparent;color:{theme.TEXT_SECONDARY};text-align:left;padding:9px 12px;}}QToolButton:hover{{background:{theme.BG_HOVER};border-radius:6px;}}')
        navlayout.addWidget(self._tool_button)
        navlayout.addStretch()
        btn = NavButton('settings', '')
        btn.clicked.connect(lambda: self.navigate('settings'))
        self._studio_nav['settings'] = btn
        navlayout.addWidget(btn)
        theme.bind_style(self._sidebar, lambda: f'QFrame#studioSidebar{{background:{theme.SIDEBAR_BG};border:none;border-right:1px solid {theme.BORDER};}}')
        middle.addWidget(self._sidebar)

        body = QVBoxLayout()
        body.setContentsMargins(16, 14, 16, 10)
        body.setSpacing(12)
        self._main_content_layout = body  # inherited crash banners use the visible layout
        self.metrics = MetricStrip(self.current_lang)
        body.addWidget(self.metrics)
        self._scan_note = label('', role='muted')
        self._scan_note.hide()
        body.addWidget(self._scan_note)
        # The existing completion banner continues to route to real tools.
        body.addWidget(self._next_step_banner)
        self._workflow_bar = QWidget()
        workflow_layout = QHBoxLayout(self._workflow_bar)
        workflow_layout.setContentsMargins(0, 0, 0, 0)
        workflow_layout.setSpacing(6)
        self._workflow_buttons = {}
        for route in TAB_LABELS:
            tab_button = QPushButton()
            tab_button.setCheckable(True)
            tab_button.setCursor(Qt.PointingHandCursor)
            tab_button.clicked.connect(lambda checked=False, route=route: self.navigate(route))
            theme.bind_style(tab_button, lambda: f'QPushButton{{background:{theme.BG_CARD};border:1px solid {theme.BORDER};border-radius:6px;padding:7px 12px;color:{theme.TEXT_SECONDARY};}}QPushButton:checked{{background:{theme.ORANGE_SUBTLE};color:{theme.get_accent()};border-color:{theme.get_accent()};}}')
            workflow_layout.addWidget(tab_button)
            self._workflow_buttons[route] = tab_button
        workflow_layout.addStretch()
        self._workflow_bar.hide()
        body.addWidget(self._workflow_bar)
        self._content_splitter = QSplitter(Qt.Horizontal)
        self._content_splitter.setChildrenCollapsible(False)
        self.page_stack.setMinimumSize(0, 0)
        self.page_stack.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self._content_splitter.addWidget(self.page_stack)
        self.suggestions = SuggestionPanel(self.current_lang)
        self.suggestions.setMinimumWidth(250)
        self.suggestions.setMaximumWidth(360)
        self._content_splitter.addWidget(self.suggestions)
        self._content_splitter.setStretchFactor(0, 1)
        self._content_splitter.setStretchFactor(1, 0)
        self._content_splitter.setSizes([980, 294])
        body.addWidget(self._content_splitter, 1)
        center = QWidget()
        center.setLayout(body)
        middle.addWidget(center, 1)
        outer.addLayout(middle, 1)

        status = QFrame()
        status.setObjectName('studioStatus')
        bottom = QHBoxLayout(status)
        bottom.setContentsMargins(14, 5, 14, 5)
        bottom.addWidget(self._status_dot)
        bottom.addWidget(self._status_label)
        bottom.addStretch()
        bottom.addWidget(self._topbar_monitor)
        bottom.addWidget(self._gpu_badge)
        theme.bind_style(status, lambda: f'QFrame#studioStatus{{background:{theme.BG_CARD};border:none;border-top:1px solid {theme.BORDER};}}')
        outer.addWidget(status)
        self.setCentralWidget(central)

        self._rescan_timer = QTimer(self)
        self._rescan_timer.setSingleShot(True)
        self._rescan_timer.setInterval(600)
        self._rescan_timer.timeout.connect(self._run_pending_scan)
        self._search_timer = QTimer(self)
        self._search_timer.setSingleShot(True)
        self._search_timer.setInterval(180)
        self._search_timer.timeout.connect(self._apply_search)
        self._search.textChanged.connect(lambda _: self._search_timer.start())
        self._search_shortcut = QShortcut(QKeySequence('Ctrl+K'), self)
        self._search_shortcut.activated.connect(self._search.setFocus)
        self._open_shortcut = QShortcut(QKeySequence('Ctrl+O'), self)
        self._open_shortcut.activated.connect(self.open_dataset)
        self.overview.open_requested.connect(self.open_dataset)
        self.overview.scan_requested.connect(self.start_scan)
        self.overview.stop_requested.connect(self.stop_scan)
        self.overview.navigate_requested.connect(self.navigate)
        self.suggestions.refresh_requested.connect(self.start_scan)
        self.suggestions.activated.connect(self._activate_suggestion)
        self.suggestions_page.refresh_requested.connect(self.start_scan)
        self.suggestions_page.activated.connect(self._activate_suggestion)
        self.library.open_requested.connect(self.open_image)
        # Saved changes keep the existing two-way Edit/Clothing signals intact.
        edit.folder_changed.connect(self._dataset_changed)
        edit.captions_saved.connect(self._files_changed)
        edit.caption_edit.textChanged.connect(self._update_draft_badge)
        clothing = self.caption_studio_page.clothing_tab
        clothing.captions_changed.connect(self._files_changed)
        clothing.busy_changed.connect(self._after_tool_busy)
        self.caption_studio_page.balance_tab.busy_changed.connect(self._after_tool_busy)
        self.caption_studio_page.compare_tab.busy_changed.connect(self._after_tool_busy)
        gen = self.caption_studio_page.generate_tab
        # Commit one folder selection through Edit's draft-protection transaction.
        # Dependent views follow Edit only after it accepted the folder change.
        for dependent in (edit, self.caption_studio_page.quality_tab, self.caption_studio_page.balance_tab):
            gen.folder_changed.disconnect(dependent.reload_folder)
        gen.folder_changed.connect(self._accept_generator_folder)
        self._studio_ready = True
        self.update_ui_texts()
        self.setWindowTitle('LoRA Harvester Studio')
        screen = QApplication.primaryScreen().availableGeometry()
        self.setMinimumSize(min(1040, screen.width()), min(650, max(400, screen.height() - 30)))
        self.resize(min(1600, int(screen.width() * .94)), min(1040, int(screen.height() * .90)))
        self.move(screen.x() + (screen.width() - self.width()) // 2,
                  screen.y() + (screen.height() - self.height()) // 2)
        self.navigate('overview')
        self._refresh_responsive()
        self._update_titlebar()

    def _update_titlebar(self):
        if sys.platform == 'win32':
            try:
                import ctypes
                value = ctypes.c_int(int(theme.get_mode() == 'dark'))
                ctypes.windll.dwmapi.DwmSetWindowAttribute(
                    int(self.winId()), 20, ctypes.byref(value), ctypes.sizeof(value))
            except Exception:
                logging.getLogger(__name__).debug('Native titlebar color unavailable', exc_info=True)

    def _t(self, tr, en):
        return text(self.current_lang, tr, en)

    def _notify(self, tr, en):
        self._scan_note.setText(self._t(tr, en))
        self._scan_note.show()

    def _writers_active(self):
        if getattr(self, '_close_pending', False):
            return True
        studio = self.caption_studio_page
        if self._thread_running() or studio.any_tool_busy():
            return True
        gen = studio.generate_tab
        objects = [gen, self.char_sort_page, self.upscale_page, self.training_page, self.review_grid_page]
        for obj in objects:
            for name in ('captioning_thread', '_thread', '_install_thread', '_repair_thread', '_loader', '_stats_loader'):
                worker = getattr(obj, name, None)
                try:
                    if worker is not None and worker.isRunning():
                        return True
                except RuntimeError:
                    pass
        return False

    def open_dataset(self):
        if self._writers_active() or self._scan_worker is not None:
            self._notify('Önce çalışan işlemi tamamla veya durdur.', 'Finish or stop the current task first.')
            return
        folder = QFileDialog.getExistingDirectory(self, self._t('Veri seti klasörü', 'Dataset folder'), self._dataset_root)
        if folder:
            self.caption_studio_page.edit_tab.reload_folder(folder)

    def _accept_generator_folder(self, folder):
        previous = self._dataset_root
        edit = self.caption_studio_page.edit_tab
        if not self._writers_active() and self._scan_worker is None:
            edit.reload_folder(folder)
        if not edit._folder or path_key(edit._folder) != path_key(folder):
            # Edit can decline a folder switch to protect drafts. Restore all consumers.
            self._set_generator_folder(previous)
            self._notify('Klasör değişimi iptal edildi; mevcut taslaklar korundu.',
                         'Folder switch cancelled; current drafts were preserved.')

    def _set_generator_folder(self, folder):
        gen = self.caption_studio_page.generate_tab
        if getattr(gen, 'selected_folder', '') == folder:
            return
        old = gen.blockSignals(True)
        try:
            if folder:
                gen._set_folder(folder)
            else:
                # Empty means no dataset, not Path('') / the application folder.
                gen.selected_folder = ''
                gen.folder_label.setText(self._t('Veri seti seçilmedi', 'No dataset selected'))
                gen.drop_zone.setToolTip('')
                gen.image_count_label.setText('0')
                gen.start_btn.setEnabled(False)
                if hasattr(gen, '_target_path_lbl'):
                    gen._target_path_lbl.setText('—')
                if hasattr(gen, '_img_count_lbl'):
                    gen._img_count_lbl.setText('0')
        finally:
            gen.blockSignals(old)

    def _dataset_changed(self, folder):
        self._dataset_root = str(Path(folder).resolve())
        self._studio_epoch += 1
        self._scan_generation += 1
        if self._scan_worker:
            self._scan_worker.stop()
        self._set_generator_folder(self._dataset_root)
        self.caption_studio_page.quality_tab.reload_folder(self._dataset_root)
        self.overview.set_root(self._dataset_root)
        self._folder_button.setText(Path(folder).name or folder)
        self._folder_button.setToolTip(folder)
        self._report, self._report_stale = None, False
        self._scan_note.clear()
        self._scan_note.hide()
        self.metrics.set_report(None)
        self.suggestions.set_report(None)
        self.suggestions_page.set_report(None)
        self.library.set_report(None)
        self.overview.set_report(None)
        self._scan_pending = True
        self._rescan_timer.start()
        self._update_draft_badge()

    def _files_changed(self, paths=None):
        self._studio_epoch += 1
        self._report_stale = True
        self.metrics.set_report(None)
        self.suggestions.mark_stale()
        self.suggestions_page.mark_stale()
        self.library.mark_stale()
        self._notify('Kaydedilen dosyalar değişti; öneriler yenilenecek.', 'Saved files changed; suggestions need refreshing.')
        if self._scan_worker:
            self._scan_generation += 1
            self._scan_worker.stop()
        self._scan_pending = True
        self._rescan_timer.start()
        self._update_draft_badge()

    def _after_tool_busy(self, busy):
        if not busy and self._scan_pending:
            self._rescan_timer.start()

    def _run_pending_scan(self):
        if not self._scan_pending or getattr(self, '_close_pending', False):
            return
        if self._writers_active():
            self._rescan_timer.start(1000)
        elif self._scan_worker is None:
            self.start_scan()

    def start_scan(self):
        if not self._dataset_root:
            self.open_dataset()
            return
        if self._writers_active() or self._scan_worker is not None:
            self._scan_pending = True
            self._notify('Tarama için çalışan işlemin tamamlanması gerekiyor.', 'Finish the running task before scanning.')
            return
        try:
            profiles = self.caption_studio_page.clothing_tab.store.load()
            masters = tuple(p.master_tag for p in profiles if p.enabled)
            options = self.overview.options()
        except Exception as exc:
            self._scan_error(self._scan_generation, str(exc))
            return
        self._scan_generation += 1
        self._scan_pending = False
        worker = SuggestionWorker(self._scan_generation, self._dataset_root, options, masters, self)
        self._scan_worker = worker
        worker.progress.connect(self._scan_progress)
        worker.completed.connect(self._scan_completed)
        worker.failed.connect(self._scan_error)
        worker.finished.connect(self._scan_finished)
        self.overview.set_busy(True)
        self.suggestions.refresh_button.setEnabled(False)
        self.suggestions_page.refresh_button.setEnabled(False)
        self._folder_button.setEnabled(False)
        self._notify('Veri seti taranıyor; kaynak dosyalar değiştirilmez.', 'Scanning dataset; source files are not modified.')
        observer = getattr(self, '_ai_before_worker_start', None)
        if observer is not None:
            observer(worker)
        worker.start()

    def stop_scan(self):
        self._scan_pending = False
        if self._scan_worker:
            self._scan_worker.stop()

    def _scan_progress(self, current, total, filename):
        self.overview.progress.setRange(0, max(1, total))
        self.overview.progress.setValue(current)
        self.overview.status.setText(f'{current} / {total} · {filename}')

    def _scan_completed(self, generation, report):
        if generation != self._scan_generation or getattr(self, '_close_pending', False):
            return
        if path_key(report.root) != path_key(self._dataset_root):
            return
        self._report, self._report_stale = report, False
        self.metrics.set_report(report)
        self.suggestions.set_report(report)
        self.suggestions_page.set_report(report)
        self.library.set_report(report)
        self.overview.set_report(report)
        local_stamp = report.created_at[:19].replace('T', ' ') + ' UTC'
        self.metrics.setToolTip(self._t('Tarandı', 'Scanned') + f' · {local_stamp}')
        self._scan_note.clear()
        self._scan_note.hide()

    def _scan_error(self, generation, message):
        if generation != self._scan_generation or getattr(self, '_close_pending', False):
            return
        if message == 'cancelled':
            message = self._t('Tarama durduruldu. Önceki sonuçlar değiştirilmedi.',
                              'Scan cancelled. Previous results were retained.')
        self.overview.status.setText(message)
        self._scan_note.setText(message)
        self._scan_note.show()

    def _scan_finished(self):
        worker, self._scan_worker = self._scan_worker, None
        self.overview.set_busy(False)
        self.suggestions.refresh_button.setEnabled(True)
        self.suggestions_page.refresh_button.setEnabled(True)
        self._folder_button.setEnabled(True)
        if worker:
            worker.deleteLater()
        if self._scan_pending and not getattr(self, '_close_pending', False):
            self._rescan_timer.start()

    def _activate_suggestion(self, suggestion):
        if self._report_stale or not self._report:
            self.start_scan()
            return
        self._search.blockSignals(True)
        self._search.clear()
        self._search.blockSignals(False)
        self._search_timer.stop()
        self.library.filter_paths(suggestion.paths,
            suggestion.title_tr if self.current_lang == 'tr' else suggestion.title_en)
        self.navigate('library')

    def _apply_search(self):
        query = self._search.text()
        if query.strip():
            self.library.proxy.apply(paths=None, issue='', query=query)
            self.navigate('library')
        else:
            self.library.search('')

    def open_image(self, image, route='edit'):
        if self._writers_active() or getattr(self, '_close_pending', False):
            self._notify('İlgili araç çalışıyor; işlem bitince görseli aç.', 'A tool is running; open the image after it finishes.')
            return
        path = Path(image)
        if not path.is_file():
            self._files_changed()
            return
        edit = self.caption_studio_page.edit_tab
        target = path_key(path)
        index = next((i for i, (p, _) in enumerate(edit._items) if path_key(p) == target), -1)
        if index < 0:
            edit.reload_folder(self._dataset_root)
            index = next((i for i, (p, _) in enumerate(edit._items) if path_key(p) == target), -1)
        if index < 0:
            return  # reload may have been declined to preserve drafts
        if route == 'clothing':
            clothing = self.caption_studio_page.clothing_tab
            # Preserve other images' accepted analyses and pending previews.
            row = clothing.ensure_image(path)
            if row is not None:
                clothing.tabs.setCurrentIndex(1)
                previous_row = clothing.image_table.currentRow()
                clothing.image_table.setCurrentCell(row, 1)
                if previous_row == row:
                    clothing._select_image(row)
        else:
            edit.image_list.setCurrentRow(index)
            edit.image_list.scrollToItem(edit.image_list.item(index))
        self.navigate(route)

    def navigate(self, route):
        if route == 'export':
            route = 'review'
        if route not in ROUTES or getattr(self, '_close_pending', False):
            return
        index, tab = ROUTES[route]
        old_route = self._route
        self._route = route
        if route == 'review':
            # Review and export share one existing page.
            if self._dataset_root and not self._writers_active():
                page = self.review_grid_page
                if not page._folder or path_key(page._folder) != path_key(self._dataset_root):
                    page._folder = Path(self._dataset_root)
                    page._load_folder()
        self.page_stack.setCurrentIndex(index)
        if tab is not None:
            tabs = self.caption_studio_page.tabs
            previous_tab = tabs.currentIndex()
            tabs.setCurrentIndex(tab)
            if previous_tab == tab:
                self.caption_studio_page._on_tab_activated(tab)
        group = ROUTE_GROUP.get(route, route)
        for key, btn in self._studio_nav.items():
            btn.setChecked(key == group)
        visible_tabs = WORKFLOWS.get(group, ())
        self._workflow_bar.setVisible(bool(visible_tabs))
        for key, btn in self._workflow_buttons.items():
            btn.setVisible(key in visible_tabs)
            btn.setChecked(key == route)
        if route == 'edit':
            self.caption_studio_page.edit_tab._studio_layout.request_visible_thumbnails()
        if old_route == 'caption_quality' and self._dataset_root:
            # The legacy audit may modify captions; synchronize only saved files.
            self.caption_studio_page.edit_tab.refresh_saved_captions(None)
            self.caption_studio_page.clothing_tab.refresh_captions(None)
            self._files_changed()
        self._refresh_responsive()

    def switch_page(self, index):
        if not self._studio_ready:
            return super().switch_page(index)
        mapping = {0: 'video', 1: 'caption', 2: 'character', 3: 'frequency',
                   4: 'review', 5: 'settings', 6: 'upscale', 7: 'training', 8: 'overview', 9: 'library'}
        self.navigate(mapping.get(index, 'overview'))

    def _update_video_badge(self, count):
        super()._update_video_badge(count)
        if getattr(self, '_studio_ready', False):
            self._studio_nav['video'].setText(
                self._t(*LABELS['video']) + (f' ({count})' if count else ''))

    def _update_theme_button(self):
        dark = theme.get_mode() == 'dark'
        self._theme_button.setText(self._t('Açık mod', 'Light mode') if dark
                                   else self._t('Koyu mod', 'Dark mode'))
        self._theme_button.setToolTip(self._t('Temayı değiştir', 'Switch theme'))

    def toggle_theme(self):
        self._apply_theme_mode('dark' if theme.get_mode() == 'light' else 'light')

    def _toggle_suggestions(self, enabled):
        if self.width() < 1400 or self._route != 'overview':
            self.navigate('suggestions')
            return
        self._suggestions_requested = enabled
        self._refresh_responsive()

    def _refresh_responsive(self):
        if not self._studio_ready:
            return
        # Context panels should not squeeze the actual editing/tool controls.
        self.suggestions.setVisible(self._suggestions_requested and self.width() >= 1400 and
                                    self._route == 'overview')
        panel_available = self.width() >= 1400 and self._route == 'overview'
        self._suggestions_button.setCheckable(panel_available)
        if panel_available:
            self._suggestions_button.setChecked(self._suggestions_requested)
        self._topbar_monitor.setVisible(self.width() >= 1280)
        self._draft_badge.setVisible(self.width() >= 1180 and bool(self._draft_badge.text()))
        self.metrics.setVisible(self._route in ('overview', 'library', 'edit', 'clothing', 'balance'))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_responsive()

    def _update_draft_badge(self):
        if not self._studio_ready:
            return
        dirty = self.caption_studio_page.edit_tab._dirty
        self._draft_badge.setText(self._t('Taslak · kaydedilmedi', 'Draft · unsaved') if dirty else '')
        self._draft_badge.setToolTip(self._t('Kaydetmeden sayfa değiştirebilirsin; taslağın korunur.',
                                           'You can switch pages before saving; the draft remains.') if dirty else '')
        self._draft_badge.setVisible(dirty and self.width() >= 1180)

    def update_ui_texts(self):
        super().update_ui_texts()
        if not self._studio_ready:
            return
        self.setWindowTitle('LoRA Harvester Studio')
        for route, btn in self._studio_nav.items():
            btn.setText(self._t(*LABELS[route]))
            btn.setToolTip(self._t(*NAV_TIPS[route]))
        self._update_video_badge(len(self.video_paths))
        for route, btn in self._workflow_buttons.items():
            btn.setText(self._t(*TAB_LABELS[route]))
            btn.setToolTip(self._t(*TAB_LABELS[route]))
        self._tool_button.setText(self._t('Diğer araçlar', 'More tools'))
        from src.ui.translations import get_text
        for section in self.settings_page.findChildren(Disclosure):
            if section._translation_key:
                section.header.setText(get_text(section._translation_key, self.current_lang))
        self._tools_menu.clear()
        for route in ('character', 'upscale', 'training'):
            action = self._tools_menu.addAction(self._t(*LABELS[route]))
            action.triggered.connect(lambda checked=False, route=route: self.navigate(route))
        if not self._dataset_root:
            self._folder_button.setText(self._t('Veri seti aç', 'Open dataset'))
            self._folder_button.setToolTip(self._t('Yerel görsel klasörünü aç.', 'Open a local image folder.'))
        self._search.setPlaceholderText(self._t('Dosya veya tag ara…  Ctrl+K', 'Search files or tags…  Ctrl+K'))
        self._update_theme_button()
        self._suggestions_button.setText(self._t('Öneriler', 'Suggestions'))
        self._suggestions_button.setToolTip(self._t('Akıllı önerileri aç; geniş pencerede yan panel kullanılır',
                                                 'Open smart suggestions; wide windows use the side panel'))
        for widget in (self.metrics, self.suggestions, self.suggestions_page, self.overview, self.library):
            widget.update_language(self.current_lang)
        self._update_draft_badge()

    def _refresh_all_styles(self):
        super()._refresh_all_styles()
        if self._studio_ready:
            self._update_theme_button()
            for btn in self._studio_nav.values():
                btn.refresh_icon()
            self._tool_button.setIcon(line_icon('tools'))
            self._folder_button.setIcon(line_icon('folder'))
            self._update_titlebar()

    def closeEvent(self, event):
        super().closeEvent(event)
        if getattr(self, '_close_pending', False):
            self._scan_pending = False
            self._rescan_timer.stop()
            self._search_timer.stop()
        if event.isAccepted():
            self.media.shutdown()


def create_app():
    drop_opencv_qt_overrides()
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    for name in ('icon.png', 'icon.ico'):
        path = ROOT / 'assets' / name
        if path.exists():
            app.setWindowIcon(QIcon(str(path)))
            break
    app.setStyle('Fusion')
    window = VideoSmartCropperUI() if '--classic-ui' in sys.argv else StudioMainWindow()
    from src.ui.ai_control_panel import attach_ai_control
    attach_ai_control(window, app)
    window.show()
    return app, window
