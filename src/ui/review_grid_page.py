"""
Review Grid Page for LoRA-Harvester.
Thumbnail grid with multi-select reject/keep for manual dataset curation.
Also hosts the "Export to Kohya" button.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import List, Optional

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QPixmap, QIcon, QKeySequence
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QListWidget, QListWidgetItem, QAbstractItemView,
    QComboBox, QSplitter, QFrame, QTextEdit, QShortcut, QMessageBox,
    QDialog, QSpinBox, QRadioButton, QCheckBox, QDialogButtonBox,
    QProgressBar,
)

from src.core.dataset_scanner import FramePair, scan_dataset, detect_concepts
from src.ui.translations import get_text
from src.ui import theme


# ──────────────────────────────────────────────────────────
# Background thread: gathers dataset statistics
# ──────────────────────────────────────────────────────────

class _StatsLoader(QThread):
    """Computes concept distribution, resolution buckets, warnings."""
    stats_ready = pyqtSignal(dict)

    def __init__(self, pairs: List[FramePair], folder: Path):
        super().__init__()
        self._pairs = pairs
        self._folder = folder

    def run(self):
        from collections import Counter

        # Concept distribution
        concept_counter: Counter = Counter(p.concept for p in self._pairs)

        # Resolution buckets (use PIL — lightweight)
        buckets: Counter = Counter()
        try:
            from PIL import Image as _PIL
            for p in self._pairs:
                try:
                    with _PIL.open(str(p.image)) as img:
                        w, h = img.size
                        short = min(w, h)
                        if short < 600:
                            buckets['< 600px'] += 1
                        elif short < 900:
                            buckets['600–900px'] += 1
                        elif short < 1200:
                            buckets['900–1200px'] += 1
                        else:
                            buckets['≥ 1200px'] += 1
                except Exception:
                    pass
        except ImportError:
            pass

        # Warnings
        warnings = []
        for concept, cnt in concept_counter.items():
            if cnt < 30:
                warnings.append(f"⚠️ '{concept}': {cnt} frames — may be too few for LoRA")
            elif cnt > 500:
                warnings.append(f"ℹ️ '{concept}': {cnt} frames — large concept")

        self.stats_ready.emit({
            'concepts': dict(concept_counter),
            'resolution_buckets': dict(buckets),
            'warnings': warnings,
        })


# ──────────────────────────────────────────────────────────
# Background thread: loads thumbnails without freezing UI
# ──────────────────────────────────────────────────────────

class _ThumbnailLoader(QThread):
    """Emits (index, QPixmap) for each image found."""
    thumbnail_ready = pyqtSignal(int, QPixmap)
    finished_loading = pyqtSignal(int)  # total count

    def __init__(self, pairs: List[FramePair], thumb_size: int = 160):
        super().__init__()
        self._pairs = pairs
        self._thumb_size = thumb_size
        self._abort = False

    def abort(self):
        self._abort = True

    def run(self):
        for i, pair in enumerate(self._pairs):
            if self._abort:
                break
            try:
                px = QPixmap(str(pair.image))
                if px.isNull():
                    continue
                px = px.scaled(
                    self._thumb_size, self._thumb_size,
                    Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                self.thumbnail_ready.emit(i, px)
            except Exception:
                pass
        self.finished_loading.emit(len(self._pairs))


# ──────────────────────────────────────────────────────────
# Kohya Export Dialog
# ──────────────────────────────────────────────────────────

class _KohyaExportDialog(QDialog):
    def __init__(self, source_folder: Optional[Path], parent=None, lang: str = 'en'):
        super().__init__(parent)
        self.lang = lang
        _t = lambda k: get_text(k, lang)
        self.setWindowTitle(_t('kohya_dlg_title'))
        self.setMinimumWidth(480)
        self._source = source_folder
        self._dest: Optional[Path] = None

        lay = QVBoxLayout(self)
        lay.setSpacing(10)

        # Source (read-only display)
        src_row = QHBoxLayout()
        src_row.addWidget(QLabel(_t('kohya_dlg_source')))
        src_display = QLabel(str(source_folder) if source_folder else "(none)")
        src_display.setStyleSheet(f"color:{theme.TEXT_SECONDARY}; font-size:{theme.fs(11)};")
        src_row.addWidget(src_display, 1)
        lay.addLayout(src_row)

        # Destination picker
        dest_row = QHBoxLayout()
        dest_row.addWidget(QLabel(_t('kohya_dlg_dest')))
        self._dest_lbl = QLabel(_t('kohya_dlg_not_selected'))
        self._dest_lbl.setStyleSheet(f"color:{theme.TEXT_SECONDARY}; font-size:{theme.fs(11)};")
        dest_btn = QPushButton(_t('kohya_dlg_browse'))
        dest_btn.setStyleSheet(theme.btn_secondary())
        dest_btn.clicked.connect(self._browse_dest)
        dest_row.addWidget(self._dest_lbl, 1)
        dest_row.addWidget(dest_btn)
        lay.addLayout(dest_row)

        # Repeats
        rep_row = QHBoxLayout()
        rep_row.addWidget(QLabel(_t('kohya_dlg_repeats')))
        self._repeats_spin = QSpinBox()
        self._repeats_spin.setRange(1, 100)
        self._repeats_spin.setValue(10)
        self._repeats_spin.setToolTip(_t('kohya_dlg_repeats_tip'))
        rep_row.addWidget(self._repeats_spin)
        rep_row.addStretch()
        lay.addLayout(rep_row)

        # Copy vs Move
        op_row = QHBoxLayout()
        op_row.addWidget(QLabel(_t('kohya_dlg_op')))
        self._copy_radio = QRadioButton(_t('kohya_dlg_copy'))
        self._move_radio = QRadioButton(_t('kohya_dlg_move'))
        self._copy_radio.setChecked(True)
        op_row.addWidget(self._copy_radio)
        op_row.addWidget(self._move_radio)
        op_row.addStretch()
        lay.addLayout(op_row)

        # Generate toml
        self._toml_cb = QCheckBox(_t('kohya_dlg_gen_toml'))
        self._toml_cb.setChecked(True)
        lay.addWidget(self._toml_cb)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)

    def _browse_dest(self):
        folder = QFileDialog.getExistingDirectory(self, get_text('review_select_export_dest', self.lang))
        if folder:
            self._dest = Path(folder)
            self._dest_lbl.setText(str(self._dest))

    def get_values(self):
        return {
            'source': self._source,
            'dest': self._dest,
            'repeats': self._repeats_spin.value(),
            'copy': self._copy_radio.isChecked(),
            'gen_toml': self._toml_cb.isChecked(),
        }


# ──────────────────────────────────────────────────────────
# Main page
# ──────────────────────────────────────────────────────────

class ReviewGridPage(QWidget):
    """
    Thumbnail grid for manual dataset curation.
    - Multi-select bad frames with Ctrl/Shift or rubber-band drag.
    - Reject: move to _rejected/ or delete (also removes co-located .txt).
    - Export to Kohya: launch KohyaExporter on the loaded folder.
    """

    def __init__(self, lang: str = 'en', parent=None):
        super().__init__(parent)
        self.lang = lang
        self._folder: Optional[Path] = None
        self._pairs: List[FramePair] = []
        self._loader: Optional[_ThumbnailLoader] = None
        self._total_count = 0
        self._rejected_count = 0
        self._init_ui()
        self._setup_shortcuts()

    # ──────────────────────────
    # UI construction
    # ──────────────────────────

    def _init_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(10)
        root.setContentsMargins(20, 14, 20, 14)

        # ── Toolbar ──
        tb = QHBoxLayout()
        tb.setSpacing(8)

        _t = lambda k: get_text(k, self.lang)

        self._path_btn = QPushButton(_t('review_select_folder'))
        self._path_btn.setStyleSheet(theme.btn_secondary())
        self._path_btn.setToolTip(_t('review_select_folder_tip'))
        self._path_btn.clicked.connect(self._browse_folder)
        tb.addWidget(self._path_btn)

        self._reload_btn = QPushButton(_t('review_reload'))
        self._reload_btn.setStyleSheet(theme.btn_secondary())
        self._reload_btn.setToolTip(_t('review_reload_tip'))
        self._reload_btn.clicked.connect(self._reload)
        self._reload_btn.setEnabled(False)
        tb.addWidget(self._reload_btn)

        tb.addStretch()

        self._sel_all_btn = QPushButton(_t('review_select_all'))
        self._sel_all_btn.setStyleSheet(theme.btn_secondary())
        self._sel_all_btn.clicked.connect(lambda: self._list.selectAll())
        self._sel_all_btn.setEnabled(False)
        tb.addWidget(self._sel_all_btn)

        self._keep_btn = QPushButton(_t('review_keep'))
        self._keep_btn.setStyleSheet(theme.btn_secondary())
        self._keep_btn.setToolTip(_t('review_keep_tip'))
        self._keep_btn.clicked.connect(self._keep_selected)
        self._keep_btn.setEnabled(False)
        tb.addWidget(self._keep_btn)

        # Reject mode selector
        self._mode_combo = QComboBox()
        self._mode_combo.setStyleSheet(theme.spinbox_compact())
        self._mode_combo.addItem(_t('review_mode_move'), "move")
        self._mode_combo.addItem(_t('review_mode_delete'), "delete")
        tb.addWidget(self._mode_combo)

        self._reject_btn = QPushButton(_t('review_reject'))
        self._reject_btn.setStyleSheet(theme.btn_danger())
        self._reject_btn.setToolTip(_t('review_reject_tip'))
        self._reject_btn.clicked.connect(self._reject_selected)
        self._reject_btn.setEnabled(False)
        tb.addWidget(self._reject_btn)

        self._export_btn = QPushButton(_t('review_export_kohya'))
        self._export_btn.setStyleSheet(theme.btn_primary())
        self._export_btn.setToolTip(_t('review_export_kohya_tip'))
        self._export_btn.clicked.connect(self._export_kohya)
        self._export_btn.setEnabled(False)
        tb.addWidget(self._export_btn)

        root.addLayout(tb)

        # ── Stat cards ──
        stats_row = QHBoxLayout()
        stats_row.setSpacing(8)
        self._stat_total = self._make_stat_card(_t('review_stat_total'), "0")
        self._stat_selected = self._make_stat_card(_t('review_stat_selected'), "0")
        self._stat_rejected = self._make_stat_card(_t('review_stat_rejected'), "0")
        self._stat_concepts = self._make_stat_card(_t('review_stat_concepts'), "0")
        self._stat_missing = self._make_stat_card(_t('review_stat_missing'), "0")
        self._stat_avg_tags = self._make_stat_card(_t('review_stat_avg_tags'), "—")
        for card in (self._stat_total, self._stat_selected,
                     self._stat_rejected, self._stat_concepts,
                     self._stat_missing, self._stat_avg_tags):
            stats_row.addWidget(card)
        stats_row.addStretch()
        root.addLayout(stats_row)

        # ── Progress bar (shown during thumbnail loading) ──
        self._progress = QProgressBar()
        self._progress.setVisible(False)
        self._progress.setStyleSheet(f"QProgressBar::chunk {{ background: {theme.ORANGE}; }}")
        root.addWidget(self._progress)

        # ── Content splitter: grid | caption preview ──
        splitter = QSplitter(Qt.Horizontal)

        self._list = QListWidget()
        self._list.setViewMode(QListWidget.IconMode)
        self._list.setIconSize(QSize(160, 160))
        self._list.setResizeMode(QListWidget.Adjust)
        self._list.setMovement(QListWidget.Static)
        self._list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._list.setStyleSheet(f"""
            QListWidget {{
                background: {theme.BG_CARD};
                border: 1px solid {theme.BORDER};
                border-radius: 8px;
            }}
            QListWidget::item:selected {{
                background: {theme.ORANGE};
                border-radius: 4px;
            }}
        """)
        self._list.itemSelectionChanged.connect(self._on_selection_changed)
        self._list.currentItemChanged.connect(self._on_current_changed)
        splitter.addWidget(self._list)

        # Right panel: Preview tab + Stats tab
        from PyQt5.QtWidgets import QTabWidget as _QTW
        right_tabs = _QTW()
        right_tabs.setMinimumWidth(220)
        right_tabs.setMaximumWidth(380)
        right_tabs.setStyleSheet(f"""
            QTabWidget::pane{{border:1px solid {theme.BORDER};background:{theme.BG_CARD};border-radius:8px;}}
            QTabBar::tab{{background:transparent;color:{theme.TEXT_MUTED};
                padding:7px 14px;font-size:{theme.fs(11)};font-weight:600;
                border-bottom:2px solid transparent;}}
            QTabBar::tab:selected{{color:{theme.ORANGE};border-bottom:2px solid {theme.ORANGE};}}
        """)

        # ── Tab 0: Preview ──
        preview_widget = QWidget()
        preview_lay = QVBoxLayout(preview_widget)
        preview_lay.setContentsMargins(10, 10, 10, 10)
        self._preview_img = QLabel()
        self._preview_img.setFixedSize(200, 200)
        self._preview_img.setAlignment(Qt.AlignCenter)
        self._preview_img.setStyleSheet(f"background: {theme.BG_WINDOW}; border-radius: 4px;")
        preview_lay.addWidget(self._preview_img, alignment=Qt.AlignCenter)
        cap_hdr = QLabel(_t('review_caption_label'))
        cap_hdr.setStyleSheet(f"color:{theme.TEXT_SECONDARY}; font-size:{theme.fs(11)}; font-weight:600;")
        preview_lay.addWidget(cap_hdr)
        self._cap_hdr = cap_hdr
        self._caption_text = QTextEdit()
        self._caption_text.setReadOnly(True)
        self._caption_text.setStyleSheet(theme.text_edit_input())
        self._caption_text.setPlaceholderText(_t('review_caption_placeholder'))
        preview_lay.addWidget(self._caption_text, 1)
        right_tabs.addTab(preview_widget, _t('review_tab_preview'))
        self._right_tabs = right_tabs

        # ── Tab 1: Stats ──
        stats_widget = QWidget()
        stats_lay = QVBoxLayout(stats_widget)
        stats_lay.setContentsMargins(10, 10, 10, 10)
        stats_lay.setSpacing(8)

        self._stats_text = QTextEdit()
        self._stats_text.setReadOnly(True)
        self._stats_text.setStyleSheet(theme.text_edit_input())
        self._stats_text.setPlaceholderText(_t('review_stats_placeholder'))
        stats_lay.addWidget(self._stats_text, 1)

        self._stats_chart = QLabel()  # custom bar chart rendered via HTML
        self._stats_chart.setWordWrap(True)
        self._stats_chart.setTextFormat(Qt.RichText)
        self._stats_chart.setStyleSheet(f"color:{theme.TEXT_PRIMARY};font-size:{theme.fs(10)};")
        stats_lay.addWidget(self._stats_chart)

        right_tabs.addTab(stats_widget, _t('review_tab_stats'))

        splitter.addWidget(right_tabs)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter, 1)

        # ── Status bar ──
        self._status_lbl = QLabel(_t('review_status_initial'))
        self._status_lbl.setStyleSheet(
            f"color:{theme.TEXT_SECONDARY}; font-size:{theme.fs(11)}; padding: 2px 0;"
        )
        root.addWidget(self._status_lbl)

    def _make_stat_card(self, label: str, value: str) -> QFrame:
        card = QFrame()
        card.setFixedSize(100, 52)
        card.setStyleSheet(
            f"QFrame {{ background: {theme.BG_CARD}; border: 1px solid {theme.BORDER}; border-radius: 8px; }}"
        )
        lay = QVBoxLayout(card)
        lay.setContentsMargins(8, 4, 8, 4)
        val_lbl = QLabel(value)
        val_lbl.setAlignment(Qt.AlignCenter)
        val_lbl.setStyleSheet(f"color:{theme.TEXT_PRIMARY}; font-size:{theme.fs(16)}; font-weight:700; border:none;")
        txt_lbl = QLabel(label)
        txt_lbl.setAlignment(Qt.AlignCenter)
        txt_lbl.setStyleSheet(f"color:{theme.TEXT_SECONDARY}; font-size:{theme.fs(10)}; border:none;")
        lay.addWidget(val_lbl)
        lay.addWidget(txt_lbl)
        card._val_lbl = val_lbl
        card._txt_lbl = txt_lbl
        return card

    def _set_stat(self, card: QFrame, value):
        card._val_lbl.setText(str(value))

    def _setup_shortcuts(self):
        QShortcut(QKeySequence(Qt.Key_Delete), self, self._reject_selected)
        QShortcut(QKeySequence("K"), self, self._keep_selected)
        QShortcut(QKeySequence("Ctrl+A"), self, lambda: self._list.selectAll())

    # ──────────────────────────
    # Folder loading
    # ──────────────────────────

    def _browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, get_text('review_select_dataset_folder', self.lang))
        if folder:
            self._folder = Path(folder)
            self._load_folder()

    def _reload(self):
        if self._folder:
            self._load_folder()

    def _load_folder(self):
        if self._loader and self._loader.isRunning():
            self._loader.abort()
            self._loader.wait(2000)

        self._list.clear()
        self._pairs = []
        self._rejected_count = 0
        self._status_lbl.setText(get_text('review_status_scanning', self.lang).format(self._folder))

        try:
            self._pairs = scan_dataset(self._folder)
        except Exception as e:
            self._status_lbl.setText(get_text('review_status_scan_error', self.lang).format(e))
            return

        if not self._pairs:
            self._status_lbl.setText(get_text('review_no_images', self.lang))
            return

        self._total_count = len(self._pairs)

        # Pre-populate items with placeholder icons
        for pair in self._pairs:
            item = QListWidgetItem(pair.image.name[:28])
            caption_text = ""
            if pair.caption and pair.caption.exists():
                try:
                    caption_text = pair.caption.read_text(encoding='utf-8')
                except Exception:
                    pass
            item.setToolTip(caption_text or get_text('review_caption_placeholder', self.lang))
            item.setData(Qt.UserRole, pair)
            self._list.addItem(item)

        # Concepts
        concepts = detect_concepts(self._folder)
        self._set_stat(self._stat_total, self._total_count)
        self._set_stat(self._stat_selected, 0)
        self._set_stat(self._stat_rejected, self._rejected_count)
        self._set_stat(self._stat_concepts, len(concepts))

        # Enable buttons
        for btn in (self._reload_btn, self._sel_all_btn, self._keep_btn,
                    self._reject_btn, self._export_btn):
            btn.setEnabled(True)

        # Start thumbnail loader
        self._progress.setMaximum(self._total_count)
        self._progress.setValue(0)
        self._progress.setVisible(True)

        self._loader = _ThumbnailLoader(self._pairs)
        self._loader.thumbnail_ready.connect(self._on_thumb_ready)
        self._loader.finished_loading.connect(self._on_loading_done)
        self._loader.start()

        self._status_lbl.setText(
            get_text('review_status_loaded', self.lang).format(self._total_count, self._folder.name))

    # ──────────────────────────
    # Thumbnail loading callbacks
    # ──────────────────────────

    def _on_thumb_ready(self, index: int, pixmap: QPixmap):
        if index < self._list.count():
            self._list.item(index).setIcon(QIcon(pixmap))
            self._progress.setValue(index + 1)

    def _on_loading_done(self, total: int):
        self._progress.setVisible(False)
        concepts = detect_concepts(self._folder)
        self._status_lbl.setText(
            get_text('review_status_ready', self.lang).format(total, len(concepts)))
        # Compute quick stats (no PIL — just caption analysis)
        self._compute_quick_stats()
        # Launch stats loader for resolution data
        self._launch_stats_loader()

    def _compute_quick_stats(self):
        """Compute stats that don't need PIL (fast, runs on main thread)."""
        missing = sum(1 for p in self._pairs if p.caption is None)
        self._set_stat(self._stat_missing, missing)

        tag_counts = []
        for p in self._pairs:
            if p.caption and p.caption.exists():
                try:
                    text = p.caption.read_text(encoding='utf-8').strip()
                    if text:
                        tag_counts.append(len([t for t in text.split(',') if t.strip()]))
                except Exception:
                    pass
        avg = round(sum(tag_counts) / len(tag_counts), 1) if tag_counts else 0
        self._set_stat(self._stat_avg_tags, str(avg))

    def _launch_stats_loader(self):
        """Start background thread to gather resolution + concept distribution."""
        if not self._pairs or not self._folder:
            return
        self._stats_loader = _StatsLoader(self._pairs, self._folder)
        self._stats_loader.stats_ready.connect(self._update_stats_panel)
        self._stats_loader.start()

    def _update_stats_panel(self, stats: dict):
        """Populate the Stats tab with gathered data."""
        lines = []

        # Concept distribution bar chart
        concepts = stats.get('concepts', {})
        if concepts:
            max_count = max(concepts.values()) if concepts else 1
            lines.append("<b>Concept distribution:</b>")
            for name, count in sorted(concepts.items(), key=lambda x: -x[1]):
                bar_len = max(1, int(count / max_count * 30))
                color = theme.ORANGE if count < 30 else (theme.TEXT_SECONDARY if count > 500 else '#4CAF50')
                bar = f'<span style="color:{color};">{"█" * bar_len}</span>'
                warn = " ⚠️" if count < 30 else (" ℹ️" if count > 500 else "")
                lines.append(f"  {name[:22]:<22} {bar}  {count}{warn}")

        # Resolution buckets
        buckets = stats.get('resolution_buckets', {})
        if buckets:
            lines.append("")
            lines.append("<b>Resolution buckets:</b>")
            for bucket, cnt in sorted(buckets.items()):
                lines.append(f"  {bucket}: {cnt}")

        # Warnings
        warnings = stats.get('warnings', [])
        if warnings:
            lines.append("")
            lines.append("<b>Warnings:</b>")
            for w in warnings:
                lines.append(f"  {w}")

        self._stats_text.setPlainText('\n'.join(
            line.replace('<b>', '').replace('</b>', '')
                .replace('<span style="color:#FF6B35;">','').replace('<span style="color:#4CAF50;">','')
                .replace(f'<span style="color:{theme.TEXT_SECONDARY};">', '').replace('</span>', '')
            for line in lines
        ))

    # ──────────────────────────
    # Selection callbacks
    # ──────────────────────────

    def _on_selection_changed(self):
        count = len(self._list.selectedItems())
        self._set_stat(self._stat_selected, count)

    def _on_current_changed(self, current: Optional[QListWidgetItem], _prev):
        if current is None:
            self._preview_img.clear()
            self._caption_text.clear()
            return
        pair: FramePair = current.data(Qt.UserRole)
        # Large preview
        px = QPixmap(str(pair.image))
        if not px.isNull():
            self._preview_img.setPixmap(
                px.scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
        else:
            self._preview_img.clear()
        # Caption
        caption = ""
        if pair.caption and pair.caption.exists():
            try:
                caption = pair.caption.read_text(encoding='utf-8')
            except Exception:
                caption = get_text('review_caption_unreadable', self.lang)
        self._caption_text.setPlainText(caption)

    # ──────────────────────────
    # Reject / Keep actions
    # ──────────────────────────

    def _keep_selected(self):
        """Clear selection (disk no-op)."""
        self._list.clearSelection()

    def _reject_selected(self):
        selected = self._list.selectedItems()
        if not selected:
            return

        mode = self._mode_combo.currentData()  # "move" or "delete"

        # Confirm delete
        if mode == "delete":
            reply = QMessageBox.question(
                self, get_text('review_confirm_delete_title', self.lang),
                get_text('review_confirm_delete_msg', self.lang).format(len(selected)),
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return

        # Disable buttons during operation
        self._reject_btn.setEnabled(False)
        errors: list = []

        for item in selected:
            pair: FramePair = item.data(Qt.UserRole)
            try:
                if mode == "move":
                    # Dest: <root>/_rejected/<concept>/
                    rejected_dir = self._folder / "_rejected" / pair.concept
                    rejected_dir.mkdir(parents=True, exist_ok=True)

                    # Handle filename collision
                    dst_img = self._unique_dest(rejected_dir / pair.image.name)
                    shutil.move(str(pair.image), dst_img)

                    if pair.caption and pair.caption.exists():
                        dst_txt = dst_img.with_suffix('.txt')
                        shutil.move(str(pair.caption), dst_txt)

                else:  # delete
                    pair.image.unlink(missing_ok=True)
                    if pair.caption and pair.caption.exists():
                        pair.caption.unlink(missing_ok=True)

                row = self._list.row(item)
                self._list.takeItem(row)
                self._rejected_count += 1
                self._total_count -= 1

            except Exception as e:
                errors.append(f"{pair.image.name}: {e}")

        self._set_stat(self._stat_total, self._total_count)
        self._set_stat(self._stat_rejected, self._rejected_count)
        self._set_stat(self._stat_selected, 0)
        self._reject_btn.setEnabled(True)

        if errors:
            self._status_lbl.setText(
                get_text('review_status_errors', self.lang).format('; '.join(errors)))
        else:
            action = (get_text('review_action_moved', self.lang) if mode == "move"
                      else get_text('review_action_deleted', self.lang))
            self._status_lbl.setText(
                get_text('review_status_rejected', self.lang).format(
                    len(selected), action, self._total_count))

    @staticmethod
    def _unique_dest(path: Path) -> Path:
        """Return a non-colliding path by appending _N suffix."""
        if not path.exists():
            return path
        stem = path.stem
        suffix = path.suffix
        n = 1
        while True:
            candidate = path.parent / f"{stem}_{n}{suffix}"
            if not candidate.exists():
                return candidate
            n += 1

    # ──────────────────────────
    # Kohya Export
    # ──────────────────────────

    def _export_kohya(self):
        if not self._folder:
            return
        dlg = _KohyaExportDialog(self._folder, self, lang=self.lang)
        if dlg.exec_() != QDialog.Accepted:
            return
        vals = dlg.get_values()
        if not vals['dest']:
            QMessageBox.warning(
                self, get_text('review_export_nodest_title', self.lang),
                get_text('review_export_nodest_msg', self.lang))
            return

        try:
            from src.core.kohya_exporter import KohyaExporter
            exporter = KohyaExporter()
            result = exporter.export(
                source_root=vals['source'],
                dest_root=vals['dest'],
                repeats=vals['repeats'],
                copy=vals['copy'],
                gen_toml=vals['gen_toml'],
            )
            total = sum(result.values())
            msg = "\n".join(
                f"  {concept}: {count} images → {vals['repeats']}_{concept}/"
                for concept, count in result.items()
            )
            QMessageBox.information(
                self, get_text('review_export_done_title', self.lang),
                get_text('review_export_done_msg', self.lang).format(total, vals['dest'], msg)
            )
            self._status_lbl.setText(
                get_text('review_status_export_done', self.lang).format(total, vals['dest'].name))
        except Exception as e:
            QMessageBox.critical(self, get_text('review_export_error_title', self.lang), str(e))

    # ──────────────────────────
    # Theme / language
    # ──────────────────────────

    def refresh_styles(self):
        self.setStyleSheet("")  # trigger repaint; sub-widgets inherit theme

    def update_language(self, lang: str):
        self.lang = lang
        _t = lambda k: get_text(k, self.lang)

        # Toolbar buttons + tooltips
        self._path_btn.setText(_t('review_select_folder'))
        self._path_btn.setToolTip(_t('review_select_folder_tip'))
        self._reload_btn.setText(_t('review_reload'))
        self._reload_btn.setToolTip(_t('review_reload_tip'))
        self._sel_all_btn.setText(_t('review_select_all'))
        self._keep_btn.setText(_t('review_keep'))
        self._keep_btn.setToolTip(_t('review_keep_tip'))
        self._reject_btn.setText(_t('review_reject'))
        self._reject_btn.setToolTip(_t('review_reject_tip'))
        self._export_btn.setText(_t('review_export_kohya'))
        self._export_btn.setToolTip(_t('review_export_kohya_tip'))

        # Reject mode combo items
        self._mode_combo.setItemText(0, _t('review_mode_move'))
        self._mode_combo.setItemText(1, _t('review_mode_delete'))

        # Stat-card titles
        self._stat_total._txt_lbl.setText(_t('review_stat_total'))
        self._stat_selected._txt_lbl.setText(_t('review_stat_selected'))
        self._stat_rejected._txt_lbl.setText(_t('review_stat_rejected'))
        self._stat_concepts._txt_lbl.setText(_t('review_stat_concepts'))
        self._stat_missing._txt_lbl.setText(_t('review_stat_missing'))
        self._stat_avg_tags._txt_lbl.setText(_t('review_stat_avg_tags'))

        # Right tabs + preview/stats labels
        self._right_tabs.setTabText(0, _t('review_tab_preview'))
        self._right_tabs.setTabText(1, _t('review_tab_stats'))
        self._cap_hdr.setText(_t('review_caption_label'))
        self._caption_text.setPlaceholderText(_t('review_caption_placeholder'))
        self._stats_text.setPlaceholderText(_t('review_stats_placeholder'))

        # Status label (only reset to initial when nothing is loaded)
        if not self._pairs:
            self._status_lbl.setText(_t('review_status_initial'))
