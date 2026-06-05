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

        self._path_btn = QPushButton(get_text("tag_freq_select_folder_btn", self.lang))
        self._path_btn.setStyleSheet(theme.btn_secondary())
        self._path_btn.setToolTip(get_text("tag_freq_select_folder_btn_tooltip", self.lang))
        self._path_btn.clicked.connect(self._browse_folder)
        tb.addWidget(self._path_btn)

        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText(get_text("tag_freq_filter", self.lang))
        self.filter_edit.setStyleSheet(theme.line_edit_compact())
        self.filter_edit.setFixedWidth(200)
        self.filter_edit.setToolTip(get_text("tag_freq_filter_tooltip", self.lang))
        self.filter_edit.textChanged.connect(self._filter_table)
        tb.addWidget(self.filter_edit)

        tb.addStretch()

        scan_btn = QPushButton(get_text("tag_freq_scan", self.lang))
        scan_btn.setStyleSheet(theme.btn_primary())
        scan_btn.setToolTip(get_text("tag_freq_scan_tooltip", self.lang))
        scan_btn.clicked.connect(self._browse_folder)
        tb.addWidget(scan_btn)
        self.load_btn = scan_btn  # alias for update_language compat

        self.apply_bl_btn = QPushButton(get_text("tag_freq_apply_bl", self.lang))
        self.apply_bl_btn.setStyleSheet(theme.btn_danger())
        self.apply_bl_btn.setToolTip(get_text("tag_freq_apply_bl_tooltip", self.lang))
        self.apply_bl_btn.clicked.connect(self._apply_blacklist)
        self.apply_bl_btn.setEnabled(False)
        tb.addWidget(self.apply_bl_btn)

        # Tag cleaner button
        self.clean_btn = QPushButton(get_text("tag_freq_clean", self.lang))
        self.clean_btn.setStyleSheet(theme.btn_secondary())
        self.clean_btn.setToolTip(get_text("tag_freq_clean_tooltip", self.lang))
        self.clean_btn.clicked.connect(self._run_tag_cleaner)
        self.clean_btn.setEnabled(False)
        tb.addWidget(self.clean_btn)

        # Readiness checker button
        self.readiness_btn = QPushButton(get_text("tag_freq_readiness", self.lang))
        self.readiness_btn.setStyleSheet(theme.btn_secondary())
        self.readiness_btn.setToolTip(get_text("tag_freq_readiness_tooltip", self.lang))
        self.readiness_btn.clicked.connect(self._run_readiness_check)
        self.readiness_btn.setEnabled(False)
        tb.addWidget(self.readiness_btn)

        root.addLayout(tb)

        # ── Stat cards ──
        stats_row = QHBoxLayout()
        stats_row.setSpacing(8)
        self._stat_instances_card = self._make_stat_card(stats_row, get_text("tag_freq_stat_instances", self.lang), "0", "#", theme.ORANGE)
        self._stat_unique_card    = self._make_stat_card(stats_row, get_text("tag_freq_stat_unique", self.lang),    "0", "#", theme.BLUE)
        self._stat_blacklist_card = self._make_stat_card(stats_row, get_text("tag_freq_stat_blacklisted", self.lang), "0", "#", theme.RED)
        self._stat_files_card     = self._make_stat_card(stats_row, get_text("tag_freq_stat_files", self.lang),   "0", "#", theme.GREEN)
        self._stat_instances_lbl = self._stat_instances_card[1]
        self._stat_unique_lbl    = self._stat_unique_card[1]
        self._stat_blacklist_lbl = self._stat_blacklist_card[1]
        self._stat_files_lbl     = self._stat_files_card[1]
        root.addLayout(stats_row)

        # ── Split: table + blacklist panel ──
        split = QHBoxLayout()
        split.setSpacing(12)

        # Frequency table
        self.table = QTableWidget(0, 4)
        self.table.setObjectName("freq_table")
        self.table.setHorizontalHeaderLabels(self._table_headers())
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
        bl_hdr = QLabel("  " + get_text("tag_freq_bl_title", self.lang))
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
        self._bl_input.setPlaceholderText(get_text("tag_freq_bl_input_ph", self.lang))
        self._bl_input.setStyleSheet(theme.line_edit_compact())
        self._bl_input.returnPressed.connect(self._add_tag_to_blacklist)
        bl_input_row.addWidget(self._bl_input)
        self._bl_add_btn = QPushButton("+")
        self._bl_add_btn.setFixedSize(28, 28)
        self._bl_add_btn.setStyleSheet(theme.btn_primary())
        self._bl_add_btn.clicked.connect(self._add_tag_to_blacklist)
        bl_input_row.addWidget(self._bl_add_btn)
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

        self._apply_to_ds_btn = QPushButton(get_text("tag_freq_apply_to_dataset", self.lang))
        apply_btn2 = self._apply_to_ds_btn
        apply_btn2.setStyleSheet(theme.btn_danger())
        apply_btn2.clicked.connect(self._apply_blacklist)
        self._save_bl_btn = QPushButton(get_text("tag_freq_save_bl", self.lang))
        save_bl_btn = self._save_bl_btn
        save_bl_btn.setStyleSheet(theme.btn_secondary())
        save_bl_btn.clicked.connect(self._save_blacklist)
        self._add_sel_btn = QPushButton(get_text("tag_freq_add_selected", self.lang))
        self._add_sel_btn.setStyleSheet(theme.btn_secondary())
        self._add_sel_btn.clicked.connect(self._add_selected_to_blacklist)

        bl_footer.addWidget(self._add_sel_btn)
        bl_footer.addWidget(apply_btn2)
        bl_footer.addWidget(save_bl_btn)
        bl_lay.addLayout(bl_footer)

        split.addWidget(bl_panel)
        root.addLayout(split, stretch=1)

        # Status
        self.status_lbl = QLabel(get_text("tag_freq_no_data", self.lang))
        self.status_lbl.setStyleSheet(theme.label_muted())
        root.addWidget(self.status_lbl)

    # ─── Stat card helper ───────────────────────────────────────────────

    def _make_stat_card(self, layout: QHBoxLayout, title: str, value: str, icon: str, color: str):
        card = QFrame()
        card.setProperty("lhCard", True)
        card.setStyleSheet(
            f"QFrame {{ background: {theme.BG_CARD}; border: 1px solid {theme.BORDER_LIGHT};"
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
        return (t, v)  # (title label, value label) for updates

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
        _has_data = bool(self._tag_counts)
        self.apply_bl_btn.setEnabled(_has_data)
        self.clean_btn.setEnabled(_has_data)
        self.readiness_btn.setEnabled(_has_data)

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
        act = menu.addAction(get_text("tag_freq_ctx_add", self.lang).format(tag))
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
        rm = menu.addAction(get_text("tag_freq_ctx_remove", self.lang))
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
        path, _ = QFileDialog.getSaveFileName(
            self, get_text("tag_freq_save_bl_title", self.lang),
            "blacklist.txt", get_text("tag_freq_txt_filter", self.lang))
        if path:
            Path(path).write_text("\n".join(sorted(self._blacklist)), encoding="utf-8")

    # ─── Language ───────────────────────────────────────────────────────

    def _table_headers(self) -> list:
        return [
            get_text("tag_freq_col_tag", self.lang),
            get_text("tag_freq_col_count", self.lang),
            get_text("tag_freq_col_pct", self.lang),
            get_text("tag_freq_col_freq", self.lang),
        ]

    def update_language(self, lang: str):
        self.lang = lang

        # Toolbar buttons + tooltips
        self._path_btn.setText(get_text("tag_freq_select_folder_btn", self.lang))
        # Only reset the path button tooltip when no folder is loaded (it holds the path otherwise)
        if not self._folder:
            self._path_btn.setToolTip(get_text("tag_freq_select_folder_btn_tooltip", self.lang))
        self.filter_edit.setPlaceholderText(get_text("tag_freq_filter", self.lang))
        self.filter_edit.setToolTip(get_text("tag_freq_filter_tooltip", self.lang))
        self.load_btn.setText(get_text("tag_freq_scan", self.lang))
        self.load_btn.setToolTip(get_text("tag_freq_scan_tooltip", self.lang))
        self.apply_bl_btn.setText(get_text("tag_freq_apply_bl", self.lang))
        self.apply_bl_btn.setToolTip(get_text("tag_freq_apply_bl_tooltip", self.lang))
        self.clean_btn.setText(get_text("tag_freq_clean", self.lang))
        self.clean_btn.setToolTip(get_text("tag_freq_clean_tooltip", self.lang))
        self.readiness_btn.setText(get_text("tag_freq_readiness", self.lang))
        self.readiness_btn.setToolTip(get_text("tag_freq_readiness_tooltip", self.lang))

        # Stat-card titles
        self._stat_instances_card[0].setText(get_text("tag_freq_stat_instances", self.lang))
        self._stat_unique_card[0].setText(get_text("tag_freq_stat_unique", self.lang))
        self._stat_blacklist_card[0].setText(get_text("tag_freq_stat_blacklisted", self.lang))
        self._stat_files_card[0].setText(get_text("tag_freq_stat_files", self.lang))

        # Table headers
        self.table.setHorizontalHeaderLabels(self._table_headers())

        # Blacklist panel
        self._bl_title.setText("  " + get_text("tag_freq_bl_title", self.lang))
        self._bl_input.setPlaceholderText(get_text("tag_freq_bl_input_ph", self.lang))
        self._add_sel_btn.setText(get_text("tag_freq_add_selected", self.lang))
        self._apply_to_ds_btn.setText(get_text("tag_freq_apply_to_dataset", self.lang))
        self._save_bl_btn.setText(get_text("tag_freq_save_bl", self.lang))

        # Status label (only reset when no data is loaded)
        if not self._tag_counts:
            self.status_lbl.setText(get_text("tag_freq_no_data", self.lang))

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
        if hasattr(self, '_bl_add_btn'):
            self._bl_add_btn.setStyleSheet(theme.btn_primary())
        if hasattr(self, '_add_sel_btn'):
            self._add_sel_btn.setStyleSheet(theme.btn_secondary())
        if hasattr(self, '_bl_input'):
            self._bl_input.setStyleSheet(theme.line_edit_compact())

    # ── Tag Cleaner ──────────────────────────────────────────────────────────

    def _run_tag_cleaner(self):
        if not self._folder:
            return
        from PyQt5.QtWidgets import QMessageBox, QDialog, QVBoxLayout, QLabel, QPushButton, QHBoxLayout
        # Preview first
        try:
            from src.core.tag_cleaner import TagCleaner
            cleaner = TagCleaner(remove_noise=False)
            preview = cleaner.preview(self._folder)
        except Exception as exc:
            QMessageBox.critical(self, "Tag Cleaner", f"Hata / Error:\n{exc}")
            return

        if not preview:
            QMessageBox.information(
                self, "Tag Cleaner",
                "Temizlenecek tekrar veya örtüşen tag bulunamadi.\n"
                "No duplicate or redundant tags found."
            )
            return

        # Build preview message
        total_removable = sum(p['original_count'] - p['cleaned_count'] for p in preview)
        files_affected = len(preview)
        examples = []
        for p in preview[:5]:
            fname = p['path'].name
            removed = ", ".join(p['removed_tags'][:5])
            examples.append(f"  {fname}: -{p['original_count'] - p['cleaned_count']} tag ({removed})")

        msg = (
            f"{files_affected} dosyada {total_removable} gereksiz tag bulundu.\n"
            f"{files_affected} files, {total_removable} tags to remove.\n\n"
            + "\n".join(examples)
            + ("\n  ..." if len(preview) > 5 else "")
            + "\n\nDevam etmek istiyor musunuz? / Proceed?"
        )
        reply = QMessageBox.question(
            self, "Tag Cleaner", msg,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        try:
            stats = cleaner.clean_folder(self._folder)
            QMessageBox.information(
                self, "Tag Cleaner",
                f"Tamamlandi / Done!\n\n"
                f"Taranan dosya / Files scanned: {stats['files_scanned']}\n"
                f"Degistirilen / Changed: {stats['files_changed']}\n"
                f"Kaldirilan tag / Tags removed: {stats['tags_removed']}\n"
                f"Hata / Errors: {stats['errors']}"
            )
            # Rescan to refresh tag counts
            self._scan_folder(self._folder)
        except Exception as exc:
            QMessageBox.critical(self, "Tag Cleaner", f"Temizleme hatasi:\n{exc}")

    # ── Readiness Checker ────────────────────────────────────────────────────

    def _run_readiness_check(self):
        if not self._folder:
            return
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton, QScrollArea, QWidget
        try:
            from src.core.readiness_checker import ReadinessChecker
            report = ReadinessChecker().check(self._folder)
        except Exception as exc:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Readiness Check", f"Hata / Error:\n{exc}")
            return

        # Build dialog
        dlg = QDialog(self)
        dlg.setWindowTitle("Dataset Readiness Check")
        dlg.setMinimumWidth(560)
        dlg.setMinimumHeight(420)
        dlg.setStyleSheet(f"background: {theme.BG_DARK}; color: {theme.TEXT_PRIMARY};")

        lay = QVBoxLayout(dlg)
        lay.setContentsMargins(20, 16, 20, 16)
        lay.setSpacing(10)

        # Grade header
        grade_colors = {"A": theme.GREEN, "B": "#7ec87e", "C": theme.YELLOW,
                        "D": "#ffa94d", "F": theme.RED}
        gc = grade_colors.get(report.grade, theme.TEXT_PRIMARY)
        header = QLabel(
            f"<span style='font-size:32px;color:{gc};font-weight:900;'>{report.grade}</span>"
            f"&nbsp;&nbsp;<span style='font-size:18px;font-weight:600;'>"
            f"Score: {report.score}/100</span>"
        )
        header.setTextFormat(Qt.RichText)
        lay.addWidget(header)

        # Stats grid
        stats_html = (
            f"<table style='color:{theme.TEXT_PRIMARY};font-size:13px;'>"
            f"<tr><td>Images</td><td>&nbsp;<b>{report.image_count}</b></td>"
            f"<td>&nbsp;&nbsp;&nbsp;Captioned</td><td>&nbsp;<b>{report.captioned_count} "
            f"({report.captioned_count/max(1,report.image_count)*100:.0f}%)</b></td></tr>"
            f"<tr><td>Avg tags</td><td>&nbsp;<b>{report.avg_tags:.1f}</b></td>"
            f"<td>&nbsp;&nbsp;&nbsp;Vocabulary</td><td>&nbsp;<b>{report.vocabulary_size}</b></td></tr>"
            f"<tr><td>Duplicates</td><td>&nbsp;<b>{report.duplicate_count}</b></td>"
            f"<td>&nbsp;&nbsp;&nbsp;Suggested repeats</td>"
            f"<td>&nbsp;<b style='color:{theme.ORANGE};'>{report.suggested_repeats}</b></td></tr>"
            f"</table>"
        )
        stats_lbl = QLabel(stats_html)
        stats_lbl.setTextFormat(Qt.RichText)
        stats_lbl.setStyleSheet(f"background:{theme.BG_CARD};border:1px solid {theme.BORDER};"
                                f"border-radius:6px;padding:12px;")
        lay.addWidget(stats_lbl)

        # Issues
        if report.issues:
            issues_inner = QWidget()
            issues_lay = QVBoxLayout(issues_inner)
            issues_lay.setContentsMargins(0, 0, 0, 0)
            issues_lay.setSpacing(4)
            for iss in report.issues:
                lbl = QLabel(f"{iss.emoji}  {iss.message}")
                lbl.setWordWrap(True)
                lbl.setStyleSheet(f"color:{theme.TEXT_PRIMARY};font-size:12px;"
                                  f"background:transparent;border:none;padding:2px 0;")
                issues_lay.addWidget(lbl)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(0)
            scroll.setWidget(issues_inner)
            scroll.setMaximumHeight(160)
            scroll.setStyleSheet(f"background:{theme.BG_CARD};border:1px solid {theme.BORDER};"
                                 f"border-radius:6px;")
            lay.addWidget(scroll)

        # Top tags
        if report.top_tags:
            top_str = "  ".join(f"{t} ({c})" for t, c in report.top_tags[:10])
            top_lbl = QLabel(f"<b>Top tags:</b> {top_str}")
            top_lbl.setWordWrap(True)
            top_lbl.setStyleSheet(f"color:{theme.TEXT_MUTED};font-size:11px;"
                                  f"background:transparent;border:none;")
            lay.addWidget(top_lbl)

        close_btn = QPushButton("Kapat / Close")
        close_btn.setStyleSheet(theme.btn_primary())
        close_btn.clicked.connect(dlg.accept)
        lay.addWidget(close_btn)

        dlg.exec_()
