"""
Tag Frequency Analyzer page for LoRA-Harvester.
Scans caption .txt files, counts tag frequencies, supports blacklist.
"""

from pathlib import Path
from collections import Counter
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QTableWidget, QTableWidgetItem, QHeaderView,
    QLineEdit, QListWidget, QProgressBar, QSplitter, QFrame,
    QAbstractItemView, QMenu, QAction,
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
        self._caption_files = []
        self._tag_counts = Counter()
        self._blacklist = set()
        self._init_ui()

    def _init_ui(self):
        root = QVBoxLayout()
        root.setSpacing(12)
        root.setContentsMargins(24, 16, 24, 16)
        self.setLayout(root)

        # ── Toolbar ──
        tb = QHBoxLayout()
        tb.setSpacing(8)

        self._path_btn = QPushButton("Select Folder")
        self._path_btn.setStyleSheet(theme.btn_secondary())
        self._path_btn.setToolTip("Select the dataset folder containing .txt caption files")
        self._path_btn.clicked.connect(self._browse_folder)
        tb.addWidget(self._path_btn)

        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("Filter tags…")
        self.filter_edit.setStyleSheet(theme.line_edit_compact())
        self.filter_edit.setFixedWidth(200)
        self.filter_edit.setToolTip("Filter the tag list by name")
        self.filter_edit.textChanged.connect(self._filter_table)
        tb.addWidget(self.filter_edit)

        tb.addStretch()

        scan_btn = QPushButton("Scan Dataset")
        scan_btn.setStyleSheet(theme.btn_primary())
        scan_btn.setToolTip("Scan all .txt files in the selected folder and count tag frequencies")
        scan_btn.clicked.connect(self._browse_folder)
        tb.addWidget(scan_btn)
        self.load_btn = scan_btn  # alias for update_language compat

        self.apply_bl_btn = QPushButton("Apply Blacklist")
        self.apply_bl_btn.setStyleSheet(theme.btn_danger())
        self.apply_bl_btn.setToolTip("Remove all blacklisted tags from every caption file in the dataset")
        self.apply_bl_btn.clicked.connect(self._apply_blacklist)
        self.apply_bl_btn.setEnabled(False)
        tb.addWidget(self.apply_bl_btn)

        root.addLayout(tb)

        # ── Stat cards ──
        stats_row = QHBoxLayout()
        stats_row.setSpacing(8)
        self._stat_instances_lbl = self._make_stat_card(stats_row, "Total Instances", "0", "#", theme.ORANGE)
        self._stat_unique_lbl    = self._make_stat_card(stats_row, "Unique Tags",     "0", "#", theme.BLUE)
        self._stat_blacklist_lbl = self._make_stat_card(stats_row, "Blacklisted",     "0", "#", theme.RED)
        self._stat_files_lbl     = self._make_stat_card(stats_row, "Caption Files",   "0", "#", theme.GREEN)
        root.addLayout(stats_row)

        # ── Split: table + blacklist panel ──
        split = QHBoxLayout()
        split.setSpacing(12)

        # Frequency table
        self.table = QTableWidget(0, 4)
        self.table.setObjectName("freq_table")
        self.table.setHorizontalHeaderLabels(["Tag", "Count", "%", "Frequency"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSortingEnabled(True)
        self.table.setStyleSheet(f"""
            QTableWidget {{
                background: {theme.BG_CARD};
                border: 1px solid {theme.BORDER};
                border-radius: 10px;
                gridline-color: {theme.BORDER};
                color: {theme.TEXT_PRIMARY};
                font-family: 'JetBrains Mono', monospace;
                font-size: {theme.fs(12)};
            }}
            QTableWidget::item:selected {{
                background: {theme.ORANGE_SUBTLE};
                color: {theme.TEXT_PRIMARY};
            }}
            QTableWidget::item:hover {{ background: rgba(255,255,255,0.02); }}
            QHeaderView::section {{
                background: {theme.BG_DARK};
                color: {theme.TEXT_MUTED};
                border: none;
                border-bottom: 1px solid {theme.BORDER};
                padding: 6px 10px;
                font-size: {theme.fs(11)};
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.06em;
            }}
        """)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._on_table_context)
        split.addWidget(self.table)

        # Blacklist panel
        bl_panel = QFrame()
        bl_panel.setObjectName("blacklist_panel")
        bl_panel.setFixedWidth(288)
        bl_panel.setStyleSheet(f"""
            QFrame#blacklist_panel {{
                background: {theme.BG_CARD};
                border: 1px solid {theme.BORDER};
                border-radius: 10px;
            }}
        """)
        bl_lay = QVBoxLayout(bl_panel)
        bl_lay.setContentsMargins(0, 0, 0, 0)
        bl_lay.setSpacing(0)

        # Panel header
        bl_hdr = QLabel("  Blacklist")
        bl_hdr.setFixedHeight(44)
        bl_hdr.setStyleSheet(
            f"color: {theme.TEXT_PRIMARY}; font-size: {theme.fs(13)}; font-weight: 600;"
            f" border-bottom: 1px solid {theme.BORDER}; background: transparent;"
        )
        bl_lay.addWidget(bl_hdr)
        self._bl_title = bl_hdr

        # Add tag input
        bl_input_row = QHBoxLayout()
        bl_input_row.setContentsMargins(8, 6, 8, 4)
        self._bl_input = QLineEdit()
        self._bl_input.setPlaceholderText("Add tag to blacklist…")
        self._bl_input.setStyleSheet(theme.line_edit_compact())
        self._bl_input.returnPressed.connect(self._add_tag_to_blacklist)
        bl_input_row.addWidget(self._bl_input)
        add_btn = QPushButton("+")
        add_btn.setFixedSize(28, 28)
        add_btn.setStyleSheet(theme.btn_primary())
        add_btn.clicked.connect(self._add_tag_to_blacklist)
        bl_input_row.addWidget(add_btn)
        bl_lay.addLayout(bl_input_row)

        # Blacklist items list
        self._bl_list = QListWidget()
        self._bl_list.setStyleSheet(f"""
            QListWidget {{
                background: transparent; border: none; padding: 4px;
            }}
            QListWidget::item {{
                background: rgba(239,68,68,0.12); color: {theme.RED};
                border: 1px solid rgba(239,68,68,0.3); border-radius: 6px;
                padding: 4px 8px; margin-bottom: 3px;
                font-family: 'JetBrains Mono', monospace; font-size: {theme.fs(11)};
            }}
            QListWidget::item:selected {{ background: rgba(239,68,68,0.25); }}
        """)
        self._bl_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self._bl_list.customContextMenuRequested.connect(self._on_bl_context)
        bl_lay.addWidget(self._bl_list)

        # Actions footer
        bl_footer = QVBoxLayout()
        bl_footer.setContentsMargins(8, 4, 8, 8)
        bl_footer.setSpacing(4)

        apply_btn2 = QPushButton("Apply to Dataset")
        apply_btn2.setStyleSheet(theme.btn_danger())
        apply_btn2.clicked.connect(self._apply_blacklist)
        save_bl_btn = QPushButton("Save Blacklist as .txt")
        save_bl_btn.setStyleSheet(theme.btn_secondary())
        save_bl_btn.clicked.connect(self._save_blacklist)
        self._add_sel_btn = QPushButton("Add Selected from Table")
        self._add_sel_btn.setStyleSheet(theme.btn_secondary())
        self._add_sel_btn.clicked.connect(self._add_selected_to_blacklist)

        bl_footer.addWidget(self._add_sel_btn)
        bl_footer.addWidget(apply_btn2)
        bl_footer.addWidget(save_bl_btn)
        bl_lay.addLayout(bl_footer)

        split.addWidget(bl_panel)
        root.addLayout(split, stretch=1)

        # Status
        self.status_lbl = QLabel("No dataset loaded.")
        self.status_lbl.setStyleSheet(theme.label_muted())
        root.addWidget(self.status_lbl)

    # ─── Stat card helper ───────────────────────────────────────────────

    def _make_stat_card(self, layout: QHBoxLayout, title: str, value: str, icon: str, color: str) -> QLabel:
        card = QFrame()
        card.setStyleSheet(
            f"QFrame {{ background: {theme.BG_CARD}; border: 1px solid {theme.BORDER};"
            f" border-radius: 10px; padding: 12px; }}"
        )
        cl = QHBoxLayout(card)
        ico = QLabel(icon)
        ico.setFixedSize(32, 32)
        ico.setAlignment(Qt.AlignCenter)
        ico.setStyleSheet("background: transparent; border: none; font-size: 18px;")
        info = QVBoxLayout()
        t = QLabel(title)
        t.setStyleSheet(f"color: {theme.TEXT_MUTED}; font-size: {theme.fs(10)}; font-family: 'JetBrains Mono', monospace; background: transparent; border: none;")
        v = QLabel(value)
        v.setStyleSheet(f"color: {theme.TEXT_PRIMARY}; font-size: {theme.fs(20)}; font-weight: 700; background: transparent; border: none; letter-spacing: -0.02em;")
        info.addWidget(t); info.addWidget(v)
        cl.addWidget(ico); cl.addLayout(info)
        layout.addWidget(card)
        return v  # return the value label for updates

    # ─── Folder loading ─────────────────────────────────────────────────

    def _browse_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, get_text("tag_freq_select_folder", self.lang))
        if folder:
            short = folder if len(folder) <= 40 else "..." + folder[-37:]
            self._path_btn.setText(short)
            self._path_btn.setToolTip(folder)
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

        # Update stat cards
        self._stat_instances_lbl.setText(f"{total_tags:,}")
        self._stat_unique_lbl.setText(f"{len(self._tag_counts):,}")
        self._stat_blacklist_lbl.setText(str(len(self._blacklist)))
        self._stat_files_lbl.setText(f"{len(self._caption_files):,}")

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

            pct = round(count / total * 100, 1)
            pct_item = QTableWidgetItem()
            pct_item.setData(Qt.DisplayRole, pct)
            pct_item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
            self.table.setItem(row, 2, pct_item)

            # Frequency bar using QProgressBar in cell
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(int(pct))
            bar.setTextVisible(False)
            bar.setFixedHeight(6)
            bar.setStyleSheet(f"""
                QProgressBar {{ background: {theme.BORDER}; border: none; border-radius: 3px; }}
                QProgressBar::chunk {{ background: {theme.ORANGE}; border-radius: 3px; }}
            """)
            bar_container = QWidget()
            bar_lay = QHBoxLayout(bar_container)
            bar_lay.setContentsMargins(8, 0, 8, 0)
            bar_lay.addWidget(bar)
            bar_container.setStyleSheet("background: transparent;")
            self.table.setCellWidget(row, 3, bar_container)

        self.table.setSortingEnabled(True)

    def _filter_table(self, text: str):
        text = text.lower()
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            match = text in item.text().lower() if item else False
            self.table.setRowHidden(row, bool(text) and not match)

    def _on_table_context(self, pos):
        row = self.table.rowAt(pos.y())
        if row < 0:
            return
        tag_item = self.table.item(row, 0)
        if not tag_item:
            return
        tag = tag_item.text()
        menu = QMenu(self)
        menu.setStyleSheet(
            f"QMenu {{ background: {theme.BG_CARD}; color: {theme.TEXT_PRIMARY};"
            f" border: 1px solid {theme.BORDER_LIGHT}; border-radius: 6px; padding: 4px; }}"
            f" QMenu::item:selected {{ background: {theme.ORANGE_SUBTLE}; }}"
        )
        act = menu.addAction(f"Add '{tag}' to blacklist")
        action = menu.exec_(self.table.viewport().mapToGlobal(pos))
        if action == act:
            self._add_tag_str_to_blacklist(tag)

    # ─── Blacklist ──────────────────────────────────────────────────────

    def _add_tag_to_blacklist(self):
        tag = self._bl_input.text().strip()
        if tag:
            self._add_tag_str_to_blacklist(tag)
            self._bl_input.clear()

    def _add_tag_str_to_blacklist(self, tag: str):
        if tag.lower() not in self._blacklist:
            self._blacklist.add(tag.lower())
            self._bl_list.addItem(tag)
            self._stat_blacklist_lbl.setText(str(len(self._blacklist)))
            self._refresh_table_blacklist_color()

    def _on_bl_context(self, pos):
        item = self._bl_list.itemAt(pos)
        if not item:
            return
        menu = QMenu(self)
        menu.setStyleSheet(
            f"QMenu {{ background: {theme.BG_CARD}; color: {theme.TEXT_PRIMARY};"
            f" border: 1px solid {theme.BORDER_LIGHT}; border-radius: 6px; padding: 4px; }}"
        )
        rm = menu.addAction("Remove from blacklist")
        action = menu.exec_(self._bl_list.viewport().mapToGlobal(pos))
        if action == rm:
            self._blacklist.discard(item.text().lower())
            self._bl_list.takeItem(self._bl_list.row(item))
            self._stat_blacklist_lbl.setText(str(len(self._blacklist)))
            self._refresh_table_blacklist_color()

    def _add_selected_to_blacklist(self):
        rows = set(idx.row() for idx in self.table.selectedIndexes())
        for row in rows:
            item = self.table.item(row, 0)
            if item:
                self._add_tag_str_to_blacklist(item.text())

    def _refresh_table_blacklist_color(self):
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if not item:
                continue
            if item.text().lower() in self._blacklist:
                item.setForeground(QColor(theme.RED))
            else:
                item.setForeground(QColor(theme.TEXT_PRIMARY))

    def _apply_blacklist(self):
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
        if self._folder:
            self._scan_folder(self._folder)

    def _save_blacklist(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Blacklist", "blacklist.txt", "Text files (*.txt)")
        if path:
            Path(path).write_text("\n".join(sorted(self._blacklist)), encoding="utf-8")

    # ─── Language ───────────────────────────────────────────────────────

    def update_language(self, lang: str):
        self.lang = lang
        self.filter_edit.setPlaceholderText("Filter tags…")
        self.table.setHorizontalHeaderLabels(["Tag", "Count", "%", "Frequency"])

    # ─── Theme ──────────────────────────────────────────────────────────

    def refresh_styles(self):
        self.load_btn.setStyleSheet(theme.btn_primary())
        self.apply_bl_btn.setStyleSheet(theme.btn_danger())
        self.filter_edit.setStyleSheet(theme.line_edit_compact())
        self.table.setStyleSheet(f"""
            QTableWidget {{
                background: {theme.BG_CARD}; border: 1px solid {theme.BORDER};
                border-radius: 10px; gridline-color: {theme.BORDER};
                color: {theme.TEXT_PRIMARY};
                font-family: 'JetBrains Mono', monospace; font-size: {theme.fs(12)};
            }}
            QTableWidget::item:selected {{ background: {theme.ORANGE_SUBTLE}; color: {theme.TEXT_PRIMARY}; }}
            QHeaderView::section {{
                background: {theme.BG_DARK}; color: {theme.TEXT_MUTED};
                border: none; border-bottom: 1px solid {theme.BORDER};
                padding: 6px 10px; font-size: {theme.fs(11)}; font-weight: 600;
            }}
        """)
        self.status_lbl.setStyleSheet(theme.label_muted())
