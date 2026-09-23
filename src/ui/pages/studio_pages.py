"""Overview, file table and suggestion panel backed by real scan snapshots."""
from __future__ import annotations
from pathlib import Path
from PyQt5.QtCore import (Qt, QAbstractTableModel, QModelIndex, QSortFilterProxyModel,
                         pyqtSignal, QUrl)
from PyQt5.QtGui import QPixmap, QDesktopServices
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QScrollArea,
    QLabel, QFrame, QPushButton, QTableView, QHeaderView, QAbstractItemView, QSplitter,
    QPlainTextEdit, QComboBox, QCheckBox, QSpinBox, QProgressBar)
from src.core.smart_suggestions import DatasetReport, ScanOptions
from src.ui import theme
from src.ui.design_system.widgets import Card, label, button


def text(lang, tr, en):
    return tr if lang == 'tr' else en


class MetricStrip(QWidget):
    def __init__(self, lang='tr', parent=None):
        super().__init__(parent)
        self.lang = lang
        self.report = None
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        self.values, self.labels = [], []
        for _ in range(4):
            card = Card()
            card.body.setContentsMargins(14, 10, 14, 10)
            card.body.setSpacing(5)
            caption, value = label(role='muted'), label('—', role='title')
            card.body.addWidget(caption)
            card.body.addWidget(value)
            self.labels.append(caption)
            self.values.append(value)
            layout.addWidget(card, 1)
        self.update_language(lang)

    def set_report(self, report):
        self.report = report
        self._refresh()

    def _refresh(self):
        report = self.report
        values = ['—'] * 4 if report is None else [str(report.total), str(report.captioned),
                  str(report.missing), str(report.duplicate_extras) if report.options.exact_duplicates
                  else text(self.lang, 'Taranmadı', 'Not scanned')]
        for value, widget in zip(values, self.values):
            widget.setText(value)

    def update_language(self, lang):
        self.lang = lang
        names = [('Toplam görsel', 'Total images'), ('Dolu caption', 'Non-empty captions'),
                 ('Eksik / boş caption', 'Missing / empty captions'), ('Birebir fazla kopya', 'Exact extra copies')]
        for widget, (tr, en) in zip(self.labels, names):
            widget.setText(text(lang, tr, en))
        self._refresh()


class SuggestionPanel(QWidget):
    activated = pyqtSignal(object)
    refresh_requested = pyqtSignal()

    def __init__(self, lang='tr', parent=None):
        super().__init__(parent)
        self.lang, self.report, self.stale = lang, None, False
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        self.heading = label(role='heading')
        root.addWidget(self.heading)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        root.addWidget(self.scroll, 1)
        self.refresh_button = button('', self.refresh_requested.emit)
        root.addWidget(self.refresh_button)
        self.update_language(lang)

    def set_report(self, report):
        self.report, self.stale = report, False
        self._render()

    def mark_stale(self):
        self.stale = True
        self._render()

    def _render(self):
        old = self.scroll.takeWidget()
        if old:
            old.deleteLater()
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        if not self.report:
            layout.addWidget(label(text(self.lang,
                'Bir veri seti aç. Öneriler kaydedilmiş dosyalardan üretilecek.',
                'Open a dataset. Suggestions will use its saved files.'), role='muted'))
        elif not self.report.records:
            layout.addWidget(label(text(self.lang, 'Seçilen kapsamda desteklenen görsel yok.',
                'No supported images in the selected scope.'), role='muted'))
        elif self.stale:
            layout.addWidget(label(text(self.lang,
                'Dosyalar değişti. Eski sayaçları kullanmamak için yeniden tara.',
                'Files changed. Rescan before using the previous counts.'), role='muted'))
        elif not self.report.suggestions:
            layout.addWidget(label(text(self.lang,
                'Bu taramada seçilen kontroller için sorun bulunmadı. Model doğruluğu ve eğitim kalitesi ölçülmedi.',
                'No issues found by the selected checks. Model accuracy and training quality were not evaluated.'), role='muted'))
        else:
            for suggestion in self.report.suggestions:
                card = Card()
                card.body.setContentsMargins(12, 12, 12, 12)
                title = suggestion.title_tr if self.lang == 'tr' else suggestion.title_en
                detail = suggestion.detail_tr if self.lang == 'tr' else suggestion.detail_en
                card.body.addWidget(label(title, role='heading'))
                card.body.addWidget(label(detail, role='muted'))
                card.body.addWidget(button(text(self.lang, 'İlgili görselleri göster →', 'Show affected images →'),
                                          lambda checked=False, s=suggestion: self.activated.emit(s)))
                layout.addWidget(card)
        layout.addStretch()
        self.scroll.setWidget(content)

    def update_language(self, lang):
        self.lang = lang
        self.heading.setText(text(lang, 'Akıllı öneriler', 'Smart suggestions'))
        self.heading.setToolTip(text(lang, 'Yerel kontrol; dosyalar otomatik değiştirilmez.',
                                      'Local checks; files are not changed automatically.'))
        self.refresh_button.setText(text(lang, 'Taramayı yenile', 'Refresh scan'))
        self._render()


class OverviewPage(QWidget):
    open_requested = pyqtSignal()
    scan_requested = pyqtSignal()
    stop_requested = pyqtSignal()
    navigate_requested = pyqtSignal(str)

    def __init__(self, lang='tr', parent=None):
        super().__init__(parent)
        self.lang, self.report, self.busy = lang, None, False
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(16)
        self.title = label(role='title')
        root.addWidget(self.title)
        self.source_card = Card()
        self.source_label = label(role='heading')
        self.path_label = label(role='muted')
        self.source_card.body.addWidget(self.source_label)
        self.source_card.body.addWidget(self.path_label)
        self.path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        actions = QHBoxLayout()
        self.open_button = button('', self.open_requested.emit, True)
        self.library_button = button('', lambda: self.navigate_requested.emit('library'))
        actions.addWidget(self.open_button)
        actions.addWidget(self.library_button)
        actions.addStretch()
        self.source_card.body.addLayout(actions)
        root.addWidget(self.source_card)
        controls = Card()
        self.scan_heading = label(role='heading')
        controls.body.addWidget(self.scan_heading)
        self.recursive = QCheckBox()
        self.recursive.setChecked(True)
        self.duplicates = QCheckBox()
        self.minimum_label = label(role='muted')
        self.minimum = QSpinBox()
        self.minimum.setRange(0, 8192)
        self.minimum.setValue(512)
        self.minimum.setSuffix(' px')
        row = QHBoxLayout()
        row.addWidget(self.recursive)
        row.addWidget(self.duplicates)
        row.addStretch()
        controls.body.addLayout(row)
        row2 = QHBoxLayout()
        row2.addWidget(self.minimum_label)
        row2.addWidget(self.minimum)
        row2.addStretch()
        self.scan_button = button('', self.scan_requested.emit, True)
        self.stop_button = button('', self.stop_requested.emit)
        self.stop_button.setEnabled(False)
        row2.addWidget(self.scan_button)
        row2.addWidget(self.stop_button)
        controls.body.addLayout(row2)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        controls.body.addWidget(self.progress)
        self.status = label(role='muted')
        controls.body.addWidget(self.status)
        root.addWidget(controls)
        self.summary = label()
        root.addWidget(self.summary)
        root.addStretch()
        self.update_language(lang)

    def options(self):
        return ScanOptions(self.recursive.isChecked(), self.minimum.value(), self.duplicates.isChecked())

    def set_root(self, path):
        self.path_label.setText(path or text(self.lang, 'Henüz veri seti seçilmedi.', 'No dataset selected.'))

    def set_report(self, report):
        self.report = report
        if report:
            self.summary.setText(text(self.lang,
                f'{report.total} görsel tarandı · {report.flagged} görselde en az bir bulgu · {report.elapsed_seconds:.2f} sn',
                f'{report.total} images scanned · {report.flagged} with one or more findings · {report.elapsed_seconds:.2f} s'))
            self.status.setText(text(self.lang, 'Tarama tamamlandı.', 'Scan complete.') +
                ('\n' + '\n'.join(report.warnings[:3]) if report.warnings else ''))
        else:
            self.summary.clear()

    def set_busy(self, busy):
        self.busy = busy
        for widget in (self.recursive, self.duplicates, self.minimum, self.scan_button, self.open_button):
            widget.setEnabled(not busy)
        self.stop_button.setEnabled(busy)
        if busy:
            self.progress.setRange(0, 0)
        elif self.progress.maximum() == 0:
            self.progress.setRange(0, 1)
            self.progress.setValue(0)

    def update_language(self, lang):
        self.lang = lang
        for widget, tr, en in [
            (self.title, 'Veri seti çalışma alanı', 'Dataset workspace'),
            (self.source_label, 'Çalışma klasörü', 'Working folder'),
            (self.open_button, 'Veri seti aç', 'Open dataset'), (self.library_button, 'Kütüphaneyi aç', 'Open library'),
            (self.scan_heading, 'Gerçek veriden öneri üret', 'Generate evidence-based suggestions'),
            (self.recursive, 'Alt klasörleri dahil et', 'Include subfolders'),
            (self.duplicates, 'Birebir kopyaları tara (SHA256)', 'Scan exact duplicates (SHA256)'),
            (self.minimum_label, 'Kısa kenar uyarısı', 'Short-edge warning'),
            (self.scan_button, 'Veri setini tara', 'Scan dataset'), (self.stop_button, 'Durdur', 'Stop'),
]:
            widget.setText(text(lang, tr, en))
        self.title.setToolTip(text(lang, 'Veri setini aç, tara ve sonuçları incele.',
                                   'Open, scan and inspect a dataset.'))
        self.open_button.setToolTip(text(lang, 'Yerel görsel klasörünü seç.', 'Choose a local image folder.'))
        self.scan_button.setToolTip(text(lang, 'Caption, görsel ve tekrar kontrollerini çalıştırır; dosyalara yazmaz.',
                                         'Checks captions, images and duplicates without writing files.'))
        self.recursive.setToolTip(text(lang, 'Alt klasörleri de tara.', 'Scan subfolders too.'))
        self.duplicates.setToolTip(text(lang, 'Birebir aynı görselleri bul.', 'Find exact image duplicates.'))
        self.minimum.setToolTip(text(lang, 'Küçük görseller için uyarı eşiği.', 'Warning threshold for small images.'))
        if not self.path_label.text():
            self.set_root('')
        self.set_report(self.report)


class DatasetTableModel(QAbstractTableModel):
    def __init__(self, lang='tr', parent=None):
        super().__init__(parent)
        self.lang, self.records = lang, []

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.records)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else 5

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return [('Dosya', 'File'), ('Boyut', 'Dimensions'), ('Caption', 'Caption'),
                    ('Tag', 'Tags'), ('Kıyafet', 'Outfit')][section][self.lang != 'tr']
        if role == Qt.ToolTipRole and orientation == Qt.Horizontal and section == 4:
            return text(self.lang, 'Kıyafet master etiketi', 'Outfit master tag')
        return None

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or not 0 <= index.row() < len(self.records):
            return None
        r = self.records[index.row()]
        if role == Qt.ToolTipRole:
            details = [r.path]
            if index.column() == 4 and r.outfit not in ('unknown', 'ambiguous'):
                details.append(r.outfit)
            if r.issues:
                details.append(', '.join(r.issues))
            return '\n'.join(details)
        if role == Qt.DisplayRole:
            states = {'present': ('Dolu', 'Present'), 'empty': ('Boş', 'Empty'),
                      'missing': ('Yok', 'Missing'), 'unreadable': ('Okunamadı', 'Unreadable')}
            return [r.relative_path, f'{r.width} × {r.height}' if r.width else '—',
                    states.get(r.caption_state, (r.caption_state, r.caption_state))[self.lang != 'tr'],
                    str(r.tag_count), r.outfit if r.outfit not in ('unknown', 'ambiguous') else '—'][index.column()]
        return None

    def set_records(self, records):
        self.beginResetModel()
        self.records = list(records)
        self.endResetModel()


class DatasetProxy(QSortFilterProxyModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.query, self.issue, self.paths = '', '', None
        self.setFilterCaseSensitivity(Qt.CaseInsensitive)

    def filterAcceptsRow(self, row, parent):
        record = self.sourceModel().records[row]
        if self.paths is not None and record.path not in self.paths:
            return False
        if self.issue and self.issue not in record.issues:
            return False
        return not self.query or self.query in (record.relative_path + ' ' + record.caption).casefold()

    def apply(self, *, query=None, issue=None, paths='unchanged'):
        if query is not None:
            self.query = query.strip().casefold()
        if issue is not None:
            self.issue = issue
        if paths != 'unchanged':
            self.paths = None if paths is None else set(paths)
        self.invalidateFilter()


class LibraryPage(QWidget):
    open_requested = pyqtSignal(str, str)

    def __init__(self, media, lang='tr', parent=None):
        super().__init__(parent)
        self.lang, self.media, self.report, self._active = lang, media, None, None
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 12, 16, 12)
        toolbar = QHBoxLayout()
        self.title = label(role='heading')
        toolbar.addWidget(self.title)
        toolbar.addStretch()
        self.filter = QComboBox()
        self.filter.currentIndexChanged.connect(self._filter_changed)
        toolbar.addWidget(self.filter)
        self.reset_button = button('', self.clear_filters)
        toolbar.addWidget(self.reset_button)
        root.addLayout(toolbar)
        self.notice = label(role='muted')
        root.addWidget(self.notice)
        split = QSplitter(Qt.Horizontal)
        self.table = QTableView()
        self.model = DatasetTableModel(lang, self)
        self.proxy = DatasetProxy(self)
        self.proxy.setSourceModel(self.model)
        self.table.setModel(self.proxy)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Interactive)
        for col, width in enumerate((270, 104, 76, 52, 108)):
            self.table.setColumnWidth(col, width)
        self.table.selectionModel().currentRowChanged.connect(lambda *_: self._selected())
        self.table.doubleClicked.connect(lambda _: self.open_current('edit'))
        theme.bind_style(self.table, lambda: f'QTableView{{background:{theme.BG_CARD};alternate-background-color:{theme.BG_SURFACE};border:1px solid {theme.BORDER};gridline-color:{theme.BORDER};selection-background-color:{theme.ORANGE_SUBTLE};selection-color:{theme.TEXT_PRIMARY};}} QHeaderView::section{{background:{theme.BG_SURFACE};color:{theme.TEXT_SECONDARY};padding:9px;border:none;}}')
        split.addWidget(self.table)
        detail = Card()
        detail.setMinimumWidth(230)
        self.preview = QLabel()
        self.preview.setMinimumSize(200, 270)
        self.preview.setAlignment(Qt.AlignCenter)
        detail.body.addWidget(self.preview)
        self.info = label(role='muted')
        detail.body.addWidget(self.info)
        self.caption = QPlainTextEdit()
        self.caption.setReadOnly(True)
        self.caption.setMaximumHeight(180)
        detail.body.addWidget(self.caption)
        self.edit_button = button('', lambda: self.open_current('edit'), True)
        self.clothing_button = button('', lambda: self.open_current('clothing'))
        self.folder_button = button('', self.open_folder)
        for btn in (self.edit_button, self.clothing_button, self.folder_button):
            detail.body.addWidget(btn)
        detail.body.addStretch()
        split.addWidget(detail)
        split.setSizes([610, 740])
        root.addWidget(split, 1)
        media.ready.connect(self._preview_ready)
        self.update_language(lang)
        self._selected()

    def current_record(self):
        index = self.proxy.mapToSource(self.table.currentIndex())
        return self.model.records[index.row()] if index.isValid() else None

    def set_report(self, report):
        previous = self.current_record()
        self.report = report
        self.model.set_records(report.records if report else [])
        self.notice.setText('' if report else
                            text(self.lang, 'Veri seti açıp tara.', 'Open and scan a dataset.'))
        self.notice.setVisible(not bool(report))
        if previous:
            for row, r in enumerate(self.model.records):
                if r.path == previous.path:
                    mapped = self.proxy.mapFromSource(self.model.index(row, 0))
                    if mapped.isValid():
                        self.table.setCurrentIndex(mapped)
                        break
        self._selected()

    def mark_stale(self):
        self.notice.setText(text(self.lang, 'Dosyalar değişti; yeniden tara.',
                               'Files changed; rescan.'))
        self.notice.show()

    def filter_paths(self, paths, name=''):
        self.proxy.apply(paths=paths, issue='', query='')
        self.filter.blockSignals(True)
        self.filter.setCurrentIndex(0)
        self.filter.blockSignals(False)
        self.notice.setText(name)
        self.notice.setVisible(bool(name))
        if self.proxy.rowCount():
            self.table.setCurrentIndex(self.proxy.index(0, 0))

    def search(self, query):
        self.proxy.apply(query=query)

    def clear_filters(self):
        self.proxy.apply(query='', issue='', paths=None)
        self.filter.setCurrentIndex(0)
        self.notice.clear()
        self.notice.hide()

    def _filter_changed(self):
        self.proxy.apply(issue=self.filter.currentData() or '', paths=None)

    def _selected(self):
        record = self.current_record()
        self._active = record.path if record else None
        self._awaiting_preview = bool(record)
        for btn in (self.edit_button, self.clothing_button, self.folder_button):
            btn.setEnabled(record is not None)
        self.preview.clear()
        self.caption.setPlainText(record.caption if record else '')
        self.info.setText(record.relative_path + ('\n' + ', '.join(record.issues) if record.issues else '') if record else '')
        if record:
            cached = self.media.request(record.path, (560, 560))
            if cached:
                self._show_preview(cached[0])

    def _show_preview(self, image):
        self._awaiting_preview = False
        if not image.isNull():
            width = min(320, max(200, self.preview.width() - 16))
            self.preview.setPixmap(QPixmap.fromImage(image).scaled(
                width, 280, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _preview_ready(self, payload):
        key, image, _, _ = payload
        if key[0] == self._active and key[-1] == (560, 560):
            self._show_preview(image)
        elif self._active and self._awaiting_preview:
            cached = self.media.request(self._active, (560, 560))
            if cached:
                self._show_preview(cached[0])

    def open_current(self, route):
        record = self.current_record()
        if record:
            self.open_requested.emit(record.path, route)

    def open_folder(self):
        record = self.current_record()
        if record:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(record.path).parent)))

    def update_language(self, lang):
        self.lang = lang
        self.model.lang = lang
        self.model.headerDataChanged.emit(Qt.Horizontal, 0, 4)
        self.title.setText(text(lang, 'Görsel kütüphanesi', 'Image library'))
        self.title.setToolTip(text(lang, 'Kaydedilmiş dosyaların son taraması.',
                                   'Latest scan of saved files.'))
        self.reset_button.setText(text(lang, 'Filtreleri temizle', 'Clear filters'))
        self.edit_button.setText(text(lang, 'Edit’te aç', 'Open in Edit'))
        self.clothing_button.setText(text(lang, 'Kıyafetlerde aç', 'Open in Clothing'))
        self.folder_button.setText(text(lang, 'Klasörde göster', 'Show folder'))
        previous = self.filter.currentData()
        self.filter.blockSignals(True)
        self.filter.clear()
        for key, tr, en in [('', 'Tümü', 'All'), ('missing_caption', 'Caption eksik', 'Missing caption'),
            ('empty_caption', 'Caption boş', 'Empty caption'), ('small_image', 'Düşük çözünürlük', 'Small dimensions'),
            ('duplicate_image', 'Birebir kopyalar', 'Exact duplicates'),
            ('caption_collision', 'Dosya adı çakışması', 'Name conflicts')]:
            self.filter.addItem(text(lang, tr, en), key)
        self.filter.setCurrentIndex(max(0, self.filter.findData(previous)))
        self.filter.blockSignals(False)
