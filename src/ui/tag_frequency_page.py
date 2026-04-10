"""
Tag Frequency Analyzer page for LoRA-Harvester.
Scans caption .txt files, counts tag frequencies, supports blacklist.
"""

from pathlib import Path
from collections import Counter
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QTableWidget, QTableWidgetItem, QHeaderView,
    QLineEdit, QTextEdit, QSplitter, QFrame, QAbstractItemView,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QColor
from src.ui.translations import get_text
from src.ui import theme


class TagFrequencyPage(QWidget):
    """
    Scans a folder of .txt caption files, counts comma-separated tag
    frequencies, displays them in a sortable table, and lets the user
    build a blacklist to bulk-remove unwanted tags.
    """

    def __init__(self, lang: str = "en", parent=None):
        super().__init__(parent)
        self.lang = lang
        self._folder = None
        self._caption_files = []          # list of Path
        self._tag_counts = Counter()      # tag → count
        self._blacklist = set()           # tags to remove
        self._init_ui()

    def _init_ui(self):
        root = QVBoxLayout()
        root.setSpacing(12)
        root.setContentsMargins(20, 16, 20, 16)
        self.setLayout(root)

        # Title
        self.title_lbl = QLabel(get_text("tag_freq_title", self.lang))
        self.title_lbl.setFont(QFont("Arial", 20, QFont.Bold))
        self.title_lbl.setStyleSheet(f"color: {theme.ORANGE_LIGHT};")
        self.title_lbl.setAlignment(Qt.AlignCenter)
        root.addWidget(self.title_lbl)

        self.subtitle_lbl = QLabel(get_text("tag_freq_subtitle", self.lang))
        self.subtitle_lbl.setAlignment(Qt.AlignCenter)
        self.subtitle_lbl.setStyleSheet(theme.label_muted())
        root.addWidget(self.subtitle_lbl)

        # Toolbar
        tb = QHBoxLayout()
        self.load_btn = QPushButton(get_text("tag_freq_load", self.lang))
        self.load_btn.setStyleSheet(theme.btn_primary())
        self.load_btn.setToolTip(get_text("tag_freq_load_tooltip", self.lang))
        self.load_btn.clicked.connect(self._browse_folder)
        tb.addWidget(self.load_btn)

        self.apply_bl_btn = QPushButton(get_text("tag_freq_apply_bl", self.lang))
        self.apply_bl_btn.setStyleSheet(theme.btn_primary())
        self.apply_bl_btn.setToolTip(get_text("tag_freq_apply_bl_tooltip", self.lang))
        self.apply_bl_btn.clicked.connect(self._apply_blacklist)
        self.apply_bl_btn.setEnabled(False)
        tb.addWidget(self.apply_bl_btn)

        tb.addSpacing(10)

        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText(get_text("tag_freq_filter", self.lang))
        self.filter_edit.setStyleSheet(theme.line_edit())
        self.filter_edit.setToolTip(get_text("tag_freq_filter_tooltip", self.lang))
        self.filter_edit.textChanged.connect(self._filter_table)
        self.filter_edit.setMaximumWidth(250)
        tb.addWidget(self.filter_edit)

        tb.addStretch()
        root.addLayout(tb)

        # Splitter: left = tag table, right = blacklist editor
        splitter = QSplitter(Qt.Horizontal)

        # ── Left: tag frequency table ───────────────────────────
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels([
            get_text("tag_freq_col_tag", self.lang),
            get_text("tag_freq_col_count", self.lang),
            get_text("tag_freq_col_pct", self.lang),
        ])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSortingEnabled(True)
        self.table.setStyleSheet(
            f"QTableWidget {{ background-color: {theme.BG_DEEP}; color: {theme.TEXT_PRIMARY}; "
            f"border: 1px solid {theme.BORDER}; gridline-color: {theme.BORDER}; }}"
            f"QHeaderView::section {{ background-color: {theme.BG_SURFACE}; color: {theme.ORANGE_LIGHT}; "
            f"border: 1px solid {theme.BORDER}; padding: 4px; font-weight: bold; }}"
        )
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._on_table_context)
        splitter.addWidget(self.table)

        # ── Right: blacklist panel ──────────────────────────────
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(8, 0, 0, 0)

        bl_title = QLabel(get_text("tag_freq_bl_title", self.lang))
        bl_title.setFont(QFont("Arial", 12, QFont.Bold))
        bl_title.setStyleSheet(f"color: {theme.ORANGE};")
        rl.addWidget(bl_title)
        self._bl_title = bl_title

        bl_hint = QLabel(get_text("tag_freq_bl_hint", self.lang))
        bl_hint.setWordWrap(True)
        bl_hint.setStyleSheet(theme.label_muted())
        rl.addWidget(bl_hint)
        self._bl_hint = bl_hint

        self.bl_edit = QTextEdit()
        self.bl_edit.setPlaceholderText("bad_tag\nanother_tag\n...")
        self.bl_edit.setStyleSheet(theme.log_area())
        rl.addWidget(self.bl_edit)

        add_sel_btn = QPushButton(get_text("tag_freq_add_selected", self.lang))
        add_sel_btn.setStyleSheet(theme.btn_secondary())
        add_sel_btn.setToolTip(get_text("tag_freq_add_selected_tooltip", self.lang))
        add_sel_btn.clicked.connect(self._add_selected_to_blacklist)
        rl.addWidget(add_sel_btn)
        self._add_sel_btn = add_sel_btn

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter, stretch=1)

        # Status
        self.status_lbl = QLabel(get_text("tag_freq_no_data", self.lang))
        self.status_lbl.setStyleSheet(theme.label_muted())
        root.addWidget(self.status_lbl)

    # ─── Folder loading ─────────────────────────────────────────────────

    def _browse_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, get_text("tag_freq_select_folder", self.lang))
        if folder:
            self._scan_folder(folder)

    def _scan_folder(self, folder: str):
        self._folder = folder
        self._caption_files.clear()
        self._tag_counts.clear()

        p = Path(folder)
        total_tags = 0
        for txt in sorted(p.rglob("*.txt")):
            if not txt.is_file():
                continue
            try:
                text = txt.read_text(encoding="utf-8").strip()
            except Exception:
                continue
            if not text:
                continue
            self._caption_files.append(txt)
            tags = [t.strip() for t in text.split(",") if t.strip()]
            total_tags += len(tags)
            self._tag_counts.update(tags)

        self._populate_table()
        self.apply_bl_btn.setEnabled(bool(self._tag_counts))
        self.status_lbl.setText(
            get_text("tag_freq_loaded", self.lang).format(
                len(self._caption_files), len(self._tag_counts), total_tags))

    # ─── Table ──────────────────────────────────────────────────────────

    def _populate_table(self):
        self.table.setSortingEnabled(False)
        self.table.setRowCount(0)
        total = sum(self._tag_counts.values()) or 1
        for tag, count in self._tag_counts.most_common():
            row = self.table.rowCount()
            self.table.insertRow(row)

            tag_item = QTableWidgetItem(tag)
            tag_item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
            if tag.lower() in self._blacklist:
                tag_item.setForeground(QColor(theme.RED))
            self.table.setItem(row, 0, tag_item)

            count_item = QTableWidgetItem()
            count_item.setData(Qt.DisplayRole, count)
            count_item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
            self.table.setItem(row, 1, count_item)

            pct_item = QTableWidgetItem()
            pct_item.setData(Qt.DisplayRole, round(count / total * 100, 1))
            pct_item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
            self.table.setItem(row, 2, pct_item)

        self.table.setSortingEnabled(True)

    def _filter_table(self, text: str):
        text = text.lower()
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            match = text in item.text().lower() if item else False
            self.table.setRowHidden(row, bool(text) and not match)

    def _on_table_context(self, pos):
        """Right-click adds selected tags to blacklist."""
        self._add_selected_to_blacklist()

    # ─── Blacklist ──────────────────────────────────────────────────────

    def _add_selected_to_blacklist(self):
        rows = set(idx.row() for idx in self.table.selectedIndexes())
        current = self.bl_edit.toPlainText().strip()
        lines = set(current.splitlines()) if current else set()
        for row in rows:
            item = self.table.item(row, 0)
            if item:
                lines.add(item.text())
        self.bl_edit.setPlainText("\n".join(sorted(lines)))

    def _apply_blacklist(self):
        bl_text = self.bl_edit.toPlainText()
        self._blacklist = {
            t.strip().lower() for t in bl_text.splitlines() if t.strip()
        }
        if not self._blacklist:
            return

        removed_total = 0
        files_changed = 0
        for txt in self._caption_files:
            try:
                text = txt.read_text(encoding="utf-8").strip()
            except Exception:
                continue
            parts = [t.strip() for t in text.split(",")]
            new_parts = [t for t in parts if t.strip().lower() not in self._blacklist]
            if len(new_parts) != len(parts):
                removed_total += len(parts) - len(new_parts)
                files_changed += 1
                txt.write_text(", ".join(new_parts), encoding="utf-8")

        self.status_lbl.setText(
            get_text("tag_freq_bl_applied", self.lang).format(
                removed_total, files_changed))
        # Rescan to refresh counts
        if self._folder:
            self._scan_folder(self._folder)

    # ─── Language ───────────────────────────────────────────────────────

    def update_language(self, lang: str):
        self.lang = lang
        self.title_lbl.setText(get_text("tag_freq_title", lang))
        self.subtitle_lbl.setText(get_text("tag_freq_subtitle", lang))
        self.load_btn.setText(get_text("tag_freq_load", lang))
        self.load_btn.setToolTip(get_text("tag_freq_load_tooltip", lang))
        self.apply_bl_btn.setText(get_text("tag_freq_apply_bl", lang))
        self.apply_bl_btn.setToolTip(get_text("tag_freq_apply_bl_tooltip", lang))
        self.filter_edit.setPlaceholderText(get_text("tag_freq_filter", lang))
        self.filter_edit.setToolTip(get_text("tag_freq_filter_tooltip", lang))
        self._bl_title.setText(get_text("tag_freq_bl_title", lang))
        self._bl_hint.setText(get_text("tag_freq_bl_hint", lang))
        self._add_sel_btn.setText(get_text("tag_freq_add_selected", lang))
        self._add_sel_btn.setToolTip(get_text("tag_freq_add_selected_tooltip", lang))
        self.table.setHorizontalHeaderLabels([
            get_text("tag_freq_col_tag", lang),
            get_text("tag_freq_col_count", lang),
            get_text("tag_freq_col_pct", lang),
        ])
