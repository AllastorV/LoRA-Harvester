"""Non-destructive outfit/pose/camera balance inspector."""
from __future__ import annotations
import json
from pathlib import Path
import time
import uuid
from PyQt5.QtCore import Qt, QAbstractTableModel, QModelIndex
from PyQt5.QtWidgets import (QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QPushButton,
    QLineEdit, QFileDialog, QCheckBox, QComboBox, QSpinBox, QProgressBar, QTableView,
    QAbstractItemView, QHeaderView, QMessageBox, QPlainTextEdit, QSplitter, QWidget)
from . import theme
from .studio_tasks import StudioTaskWidget
from src.core.dataset_balance import (scan_balance, make_plan, export_plan, save_override,
                                       POSE_ALIASES, ANGLE_CHOICES, UNKNOWN, AMBIGUOUS)
from src.core.clothing_profiles import ClothingProfileStore


class BalanceTableModel(QAbstractTableModel):
    HEADERS = ('File', 'Outfit', 'Pose', 'Angle', 'Issues')
    def __init__(self, parent=None):
        super().__init__(parent); self.items = []
    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.items)
    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else 5
    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or not 0 <= index.row() < len(self.items):
            return None
        item = self.items[index.row()]
        if role in (Qt.DisplayRole, Qt.ToolTipRole):
            return (item.relative_path, item.outfit, item.pose, item.angle, ', '.join(item.issues))[index.column()]
        return None
    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            return self.HEADERS[section] if orientation == Qt.Horizontal else section + 1
        return None
    def replace(self, items):
        self.beginResetModel(); self.items = list(items); self.endResetModel()


class DatasetBalanceWidget(StudioTaskWidget):
    def __init__(self, lang='en', parent=None, store=None):
        super().__init__(lang, parent)
        self.store = store or ClothingProfileStore()
        self.report = None; self.plan = None
        layout = QVBoxLayout(self)
        row = QHBoxLayout(); self.folder = QLineEdit(); self.folder.setReadOnly(True)
        self.browse = self._button('Dataset klasörü', 'Dataset folder', self._browse)
        self.recursive = QCheckBox(); self.recursive.setChecked(True)
        self.label_text(self.recursive, 'Alt klasörler', 'Recursive')
        self.scan = self._button('Tara', 'Scan', self._scan, True)
        row.addWidget(self.folder, 1); row.addWidget(self.browse); row.addWidget(self.recursive); row.addWidget(self.scan)
        layout.addLayout(row)
        controls = QHBoxLayout()
        self.dimension = QComboBox()
        for label, value in [('Kıyafet / Outfit', 'outfit'), ('Poz / Pose', 'pose'), ('Açı / Angle', 'angle'),
                             ('Kıyafet × poz × açı', 'joint')]:
            self.dimension.addItem(label, value)
        self.dimension.setCurrentIndex(3)
        self.target = QSpinBox(); self.target.setRange(0, 100000); self.target.setSpecialValueText('Auto: min group')
        self.seed = QSpinBox(); self.seed.setRange(0, 2147483647); self.seed.setValue(42)
        controls.addWidget(QLabel('Gruplama / Group')); controls.addWidget(self.dimension)
        controls.addWidget(QLabel('Grup başı / Per group')); controls.addWidget(self.target)
        controls.addWidget(QLabel('Seed')); controls.addWidget(self.seed)
        layout.addLayout(controls)
        flags = QHBoxLayout()
        self.unknown = QCheckBox(); self.label_text(self.unknown, 'Unknown grubunu dahil et', 'Include unknown group')
        self.dedupe = QCheckBox(); self.dedupe.setChecked(True)
        self.label_text(self.dedupe, 'Birebir kopyaları tekilleştir', 'Deduplicate identical files')
        flags.addWidget(self.unknown); flags.addWidget(self.dedupe); flags.addStretch()
        layout.addLayout(flags)
        split = QSplitter(Qt.Horizontal)
        self.table = QTableView(); self.model = BalanceTableModel(self)
        self.table.setModel(self.model); self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.selectionModel().currentRowChanged.connect(self._selected)
        split.addWidget(self.table)
        side = QWidget(); detail = QVBoxLayout(side)
        self.summary = QPlainTextEdit(); self.summary.setReadOnly(True)
        detail.addWidget(self.summary, 1)
        form = QFormLayout()
        self.outfit = QComboBox(); self.pose = QComboBox(); self.angle = QComboBox()
        self.pose.addItems([UNKNOWN, AMBIGUOUS] + list(POSE_ALIASES))
        self.angle.addItems(ANGLE_CHOICES + [AMBIGUOUS])
        form.addRow('Kıyafet / Outfit', self.outfit); form.addRow('Poz / Pose', self.pose); form.addRow('Açı / Angle', self.angle)
        detail.addLayout(form)
        self.override = self._button('Sınıfları kaydet', 'Save labels', self._save_labels)
        self.clear_override = self._button('Düzeltmeyi kaldır', 'Clear override', lambda: self._save_labels(True))
        detail.addWidget(self.override); detail.addWidget(self.clear_override)
        split.addWidget(side); split.setSizes([650, 320]); layout.addWidget(split, 1)
        actions = QHBoxLayout()
        self.plan_btn = self._button('Planı önizle', 'Preview plan', self._make_plan, True)
        self.export_btn = self._button('Ayrı dataset kopyası oluştur', 'Export a separate dataset', self._export)
        actions.addWidget(self.plan_btn); actions.addWidget(self.export_btn)
        self.stop_button = self._button('Durdur', 'Stop', self.stop); self.stop_button.setEnabled(False)
        actions.addWidget(self.stop_button); layout.addLayout(actions)
        self.progress = QProgressBar(); layout.addWidget(self.progress)
        self.status = QLabel(); self.status.setWordWrap(True); layout.addWidget(self.status)
        self._controls = [self.browse, self.scan, self.recursive, self.dimension, self.target, self.seed,
                          self.unknown, self.dedupe, self.override, self.clear_override, self.plan_btn,
                          self.export_btn, self.table]
        for widget, tr, en in (
            (self.scan, 'Caption etiketlerinden dengeyi tara.', 'Scan balance from caption tags.'),
            (self.dimension, 'Kıyafet, poz veya açıya göre grupla.', 'Group by outfit, pose or angle.'),
            (self.target, 'Grup başına seçilecek görsel sayısı.', 'Images to select per group.'),
            (self.override, 'Sınıfı düzeltir; caption değişmez.', 'Corrects labels without changing captions.'),
            (self.export_btn, 'Seçilenlerle ayrı bir dataset kopyası oluşturur.',
                              'Creates a separate dataset copy from the selection.'),
        ):
            self.tip_text(widget, tr, en)
        for widget in (self.dimension, self.target, self.seed):
            signal = widget.currentIndexChanged if isinstance(widget, QComboBox) else widget.valueChanged
            signal.connect(self._invalidate_plan)
        self.unknown.toggled.connect(self._invalidate_plan); self.dedupe.toggled.connect(self._invalidate_plan)
        self.recursive.toggled.connect(self._invalidate_scan)

    def _button(self, tr, en, fn, primary=False):
        button = QPushButton(); self.label_text(button, tr, en)
        button.clicked.connect(fn); theme.bind_style(button, theme.btn_primary if primary else theme.btn_secondary)
        return button

    def reload_folder(self, folder):
        if self.is_busy():
            return
        self.folder.setText(str(folder)); self._invalidate_scan()

    def _invalidate_scan(self, *_):
        self.report = self.plan = None; self.model.replace([]); self.summary.clear()

    def _invalidate_plan(self, *_):
        self.plan = None

    def _browse(self):
        folder = QFileDialog.getExistingDirectory(self, 'Dataset', self.folder.text())
        if folder:
            self.reload_folder(folder)

    def _scan(self):
        folder, recursive = self.folder.text(), self.recursive.isChecked()
        if not folder:
            self.error(self.text('Önce dataset klasörü seç.', 'Select a dataset folder first.')); return
        self.start_task(lambda cancel, progress: scan_balance(folder, recursive=recursive, store=self.store,
                                                              cancel=cancel, progress=progress), self._scanned)

    def _scanned(self, report):
        self.report = report; self.plan = None
        self.model.replace(report['items'])
        self.outfit.clear(); self.outfit.addItems([UNKNOWN, AMBIGUOUS] + report['masters'])
        self.summary.setPlainText(json.dumps(report['summary'], ensure_ascii=False, indent=2))
        self.status.setText(self.text(f'{len(report["items"])} görsel tarandı. Planı önizle.',
                                      f'{len(report["items"])} images scanned. Preview a plan.'))

    def _selected(self, index, _previous=None):
        if not index.isValid() or not self.report:
            return
        item = self.report['items'][index.row()]
        for combo, value in ((self.outfit, item.outfit), (self.pose, item.pose), (self.angle, item.angle)):
            if combo.findText(value) < 0:
                combo.addItem(value)
            combo.setCurrentText(value)

    def _save_labels(self, clear=False):
        index = self.table.currentIndex()
        if not self.report or not index.isValid():
            self.error(self.text('Tablodan bir görsel seç.', 'Select an image in the table.')); return
        item, root = self.report['items'][index.row()], self.report['root']
        labels = {'outfit': self.outfit.currentText(), 'pose': self.pose.currentText(), 'angle': self.angle.currentText()}
        recursive = self.recursive.isChecked()
        def task(cancel, progress):
            save_override(root, item, labels, clear=bool(clear))
            return scan_balance(root, recursive=recursive, store=self.store, cancel=cancel, progress=progress)
        self.start_task(task, self._scanned)

    def _make_plan(self):
        if not self.report:
            self.error(self.text('Önce dataset tara.', 'Scan a dataset first.')); return
        report = self.report
        settings = dict(dimension=self.dimension.currentData(), per_group=self.target.value(),
                        seed=self.seed.value(), include_unknown=self.unknown.isChecked(), deduplicate=self.dedupe.isChecked())
        def done(plan):
            self.plan = plan
            self.summary.setPlainText(json.dumps({k:v for k,v in plan.items() if k not in ('selected', 'fingerprint')},
                                                 ensure_ascii=False, indent=2))
            self.status.setText(self.text(f'{plan["selected_count"]}/{plan["source_count"]} görsel seçildi; hiçbir kaynak değiştirilmedi.',
                                          f'{plan["selected_count"]}/{plan["source_count"]} selected; no source changed.'))
        self.start_task(lambda cancel, progress: make_plan(report, **settings), done)

    def _export(self):
        if not self.plan:
            self.error(self.text('Önce güncel planı önizle.', 'Preview a current plan first.')); return
        parent = QFileDialog.getExistingDirectory(self, self.text('Dataset DIŞINDA hedef üst klasörü seç', 'Choose an output parent OUTSIDE the dataset'))
        if not parent:
            return
        dest = Path(parent) / ('balanced-' + time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6])
        if QMessageBox.question(self, 'Dataset', str(dest) + '\n' + self.text('Ayrı kopya oluşturulsun mu?', 'Create a separate copy?'),
                                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        plan = self.plan
        self.start_task(lambda cancel, progress: export_plan(plan, dest, cancel=cancel, progress=progress),
                        lambda data: self.status.setText(json.dumps(data, ensure_ascii=False)))
