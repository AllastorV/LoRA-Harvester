"""Outfit library, selected-person canvas and preview-first batch caption tagging."""
from __future__ import annotations
import json
from dataclasses import asdict
from pathlib import Path
from PIL import Image
from PyQt5.QtCore import Qt, QRectF, QSize, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QSplitter, QListWidget,
    QListWidgetItem, QLineEdit, QPlainTextEdit, QPushButton, QLabel, QCheckBox,
    QDoubleSpinBox, QSpinBox, QComboBox, QTabWidget, QTableWidget, QTableWidgetItem,
    QHeaderView, QFileDialog, QMessageBox, QDialog, QDialogButtonBox, QGroupBox,
    QProgressBar, QAbstractItemView, QScrollArea,
)
from src.ui import theme
from src.core.clothing_profiles import (ClothingProfileStore, ClothingProfile,
                                        ClothingPart, ClothingReference, ClothingSettings,
                                        parse_tags)
from src.core.clothing_io import find_images, load_rgb, file_digest
from src.core.clothing_service import load_selection, save_selection
from src.core.caption_sync import CaptionPreviewRebaser, path_key
from src.workers.clothing_tagger_worker import ClothingTaggerWorker

IMAGE_FILTER = 'Images (*.png *.jpg *.jpeg *.webp *.bmp *.PNG *.JPG *.JPEG *.WEBP *.BMP)'


class ClothingCropCanvas(QWidget):
    """Draw a normalized analysis box; never modifies the source image itself."""
    box_changed = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(230, 210)
        self._pixmap = None
        self.bbox = None
        self._origin = None
        self._previous_box = None
        self.setMouseTracking(True)
        theme.bind_style(self, lambda: f'background: {theme.BG_CARD}; border: 1px solid {theme.BORDER};')

    def set_image(self, image, bbox=None):
        self._origin = None
        from PIL import Image
        preview = image.copy()
        preview.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
        raw = preview.convert('RGB').tobytes()
        qimage = QImage(raw, preview.width, preview.height,
                        preview.width * 3, QImage.Format_RGB888).copy()
        self._pixmap = QPixmap.fromImage(qimage)
        self.bbox = list(bbox) if bbox is not None else None
        self.update()

    def clear(self):
        self._pixmap, self.bbox, self._origin = None, None, None
        self.update()

    def _image_rect(self):
        if self._pixmap is None:
            return QRectF()
        scale = min(self.width() / self._pixmap.width(), self.height() / self._pixmap.height())
        w, h = self._pixmap.width() * scale, self._pixmap.height() * scale
        return QRectF((self.width() - w) / 2, (self.height() - h) / 2, w, h)

    def _normalized_point(self, point):
        r = self._image_rect()
        if r.isEmpty():
            return None
        return (min(1., max(0., (point.x() - r.x()) / r.width())),
                min(1., max(0., (point.y() - r.y()) / r.height())))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(theme.BG_CARD))
        if self._pixmap is None:
            painter.setPen(QColor(theme.TEXT_MUTED))
            painter.drawText(self.rect(), Qt.AlignCenter, 'Select image / Görsel seç')
            return
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        r = self._image_rect()
        painter.drawPixmap(r, self._pixmap, QRectF(self._pixmap.rect()))
        if self.bbox:
            x1, y1, x2, y2 = self.bbox
            painter.setPen(QPen(QColor(theme.get_accent()), 2))
            painter.drawRect(QRectF(r.x() + x1 * r.width(), r.y() + y1 * r.height(),
                                    (x2 - x1) * r.width(), (y2 - y1) * r.height()))

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self._image_rect().contains(event.pos()):
            self._previous_box = list(self.bbox) if self.bbox else None
            self._origin = self._normalized_point(event.pos())

    def mouseMoveEvent(self, event):
        if self._origin is not None:
            end = self._normalized_point(event.pos())
            x, y = self._origin
            self.bbox = [min(x, end[0]), min(y, end[1]), max(x, end[0]), max(y, end[1])]
            self.update()

    def mouseReleaseEvent(self, event):
        if self._origin is not None and event.button() == Qt.LeftButton:
            self.mouseMoveEvent(event)
            self._origin = None
            if self.bbox and self.bbox[2] - self.bbox[0] > .01 and self.bbox[3] - self.bbox[1] > .01:
                self.box_changed.emit(list(self.bbox))
            else:
                self.bbox = self._previous_box
            self.update()


class ClothingTaggerWidget(QWidget):
    captions_changed = pyqtSignal(object)  # Absolute .txt paths, or None for undo/recovery.
    busy_changed = pyqtSignal(bool)

    def __init__(self, lang='en', parent=None, store=None):
        super().__init__(parent)
        self.setObjectName('clothing_tagger')
        self.lang = lang
        self.store = store or ClothingProfileStore()
        self.worker = None
        self.current_profile_id = None
        self._profiles = []
        self._refs = []
        self._paths = []
        self._results = {}
        self._previews = {}
        self._active_image = None
        self._folder = ''
        self._translations = []
        self._last_operation = ''
        self._clean_profile_snapshot = None
        self._row_by_path = {}
        self._sync_errors = {}
        self._pending_caption_paths = set()
        self._pending_caption_all = False
        self._caption_write_paths = []
        self._build()
        try:
            self._load_settings()
            self._reload_profiles()
        except Exception as exc:
            self.status.setText(str(exc))

    def _text(self, tr, en):
        return tr if self.lang == 'tr' else en

    def _label(self, tr, en):
        label = QLabel(self._text(tr, en))
        label.setWordWrap(True)
        self._translations.append((label.setText, tr, en))
        return label

    def _button(self, tr, en, callback, primary=False):
        button = QPushButton(self._text(tr, en))
        button.clicked.connect(callback)
        button.setCursor(Qt.PointingHandCursor)
        theme.bind_style(button, theme.btn_primary if primary else theme.btn_secondary)
        self._translations.append((button.setText, tr, en))
        return button

    def _check(self, tr, en):
        check = QCheckBox(self._text(tr, en))
        self._translations.append((check.setText, tr, en))
        return check

    def _tip(self, widget, tr, en):
        self._translations.append((widget.setToolTip, tr, en))
        widget.setToolTip(self._text(tr, en))
        return widget

    def _group(self, tr, en):
        group = QGroupBox(self._text(tr, en))
        self._translations.append((group.setTitle, tr, en))
        theme.bind_style(group, lambda: f'QGroupBox {{ color: {theme.TEXT_PRIMARY}; '
                         f'border: 1px solid {theme.BORDER}; border-radius: 8px; '
                         f'margin-top: 14px; padding-top: 12px; }} '
                         f'QGroupBox::title {{ color: {theme.get_accent()}; subcontrol-origin: margin; '
                         'left: 12px; padding: 0 4px; }')
        return group

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        self.tabs = QTabWidget()
        self.library = self._build_library()
        self.batch = self._build_batch()
        self.settings_page = self._build_settings()
        for widget, tr, en in ((self.library, 'Kıyafet kütüphanesi', 'Outfit library'),
                               (self.batch, 'Analiz ve önizleme', 'Analyze and preview'),
                               (self.settings_page, 'Model ve ayarlar', 'Model and settings')):
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(widget)
            self.tabs.addTab(scroll, self._text(tr, en))
        self.tabs.tabBar().setStyleSheet('QTabBar::tab { min-width: 200px; }')
        root.addWidget(self.tabs, 1)
        self.status = self._label('Profil ve referans ekle.', 'Add an outfit and reference.')
        root.addWidget(self.status)
        bottom = QHBoxLayout()
        self.progress = QProgressBar()
        bottom.addWidget(self.progress, 1)
        self.cancel_btn = self._button('Durdur', 'Cancel', self.stop)
        self.cancel_btn.setEnabled(False)
        bottom.addWidget(self.cancel_btn)
        root.addLayout(bottom)

    def _build_library(self):
        page = QWidget()
        layout = QHBoxLayout(page)
        split = QSplitter(Qt.Horizontal)
        layout.addWidget(split)
        left = QWidget()
        ll = QVBoxLayout(left)
        self.profile_list = QListWidget()
        self.profile_list.currentRowChanged.connect(self._select_profile)
        ll.addWidget(self.profile_list, 1)
        row = QHBoxLayout()
        row.addWidget(self._button('Yeni', 'New', self._new_profile))
        row.addWidget(self._button('Sil', 'Delete', self._delete_profile))
        ll.addLayout(row)
        split.addWidget(left)
        editor = QWidget()
        el = QVBoxLayout(editor)
        form = QFormLayout()
        self.name_edit = QLineEdit()
        self.master_edit = QLineEdit()
        self.family_edit = QLineEdit()
        self.enabled = self._check('Etkin', 'Enabled')
        self.enabled.setChecked(True)
        form.addRow(self._label('Kıyafet adı', 'Outfit name'), self.name_edit)
        form.addRow(self._label('Master tag (benzersiz)', 'Master tag (unique)'), self.master_edit)
        form.addRow(self._label('Grup (isteğe bağlı)', 'Group (optional)'), self.family_edit)
        form.addRow(self.enabled)
        el.addLayout(form)
        self.full_prompt = QPlainTextEdit()
        self.full_prompt.setPlaceholderText('shhooldress, serafaku, blue skirt, black thighighhs')
        self.full_prompt.setMaximumHeight(80)
        el.addWidget(self._label('Kıyafet promptu', 'Outfit prompt'))
        self._tip(self.full_prompt, 'İlk tag master etiket olur.', 'The first tag becomes the master tag.')
        el.addWidget(self.full_prompt)
        el.addWidget(self._button('Promptu parçalara ayır', 'Split prompt into parts', self._split_prompt))
        self.parts_table = QTableWidget(0, 2)
        self.parts_table.setHorizontalHeaderLabels(['Tag', self._text('Görsel anlamı (düzenlenebilir)', 'Visual meaning (editable)')])
        self.parts_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.parts_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.parts_table.setMinimumHeight(130)
        el.addWidget(self.parts_table, 1)
        part_row = QHBoxLayout()
        part_row.addWidget(self._button('Parça ekle', 'Add part', self._add_part))
        part_row.addWidget(self._button('Parçayı kaldır', 'Remove part', self._remove_part))
        el.addLayout(part_row)
        self.reference_list = QListWidget()
        self.reference_list.setIconSize(QSize(48, 64))
        self.reference_list.setMinimumHeight(98)
        self.reference_list.setMaximumHeight(115)
        el.addWidget(self._label('Referanslar', 'References'))
        self._tip(self.reference_list, 'Çok kişili görselde hedef kıyafeti kırp.',
                  'Crop the target outfit in a multi-person image.')
        el.addWidget(self.reference_list)
        refs = QHBoxLayout()
        refs.addWidget(self._button('Görsel ekle', 'Add images', self._add_references))
        refs.addWidget(self._button('Alan seç', 'Select area', self._crop_reference))
        refs.addWidget(self._button('Kaldır', 'Remove', self._remove_reference))
        el.addLayout(refs)
        self.notes_edit = QPlainTextEdit()
        self.notes_edit.setMaximumHeight(65)
        el.addWidget(self._label('Tasarım ve renk notları', 'Design and color notes'))
        self._tip(self.notes_edit, 'Benzer kıyafetleri ayıran görünür ayrıntılar.',
                  'Visible details that distinguish similar outfits.')
        el.addWidget(self.notes_edit)
        actions = QHBoxLayout()
        actions.addWidget(self._button('Referansları analiz et', 'Analyze references', self._analyze_references))
        actions.addWidget(self._button('Profili kaydet', 'Save profile', self._save_profile_clicked, True))
        el.addLayout(actions)
        split.addWidget(editor)
        split.setSizes([210, 630])
        return page

    def _build_batch(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        files = QHBoxLayout()
        self.folder_label = QLineEdit()
        self.folder_label.setReadOnly(True)
        files.addWidget(self.folder_label, 1)
        self.browse_folder_btn = self._button('Klasör', 'Folder', self._choose_folder)
        self.browse_files_btn = self._button('Görseller', 'Images', self._choose_images)
        files.addWidget(self.browse_folder_btn)
        files.addWidget(self.browse_files_btn)
        self.recursive = self._check('Alt klasörler', 'Recursive')
        self.recursive.toggled.connect(self._rescan_folder)
        files.addWidget(self.recursive)
        layout.addLayout(files)
        split = QSplitter(Qt.Horizontal)
        self.image_table = QTableWidget(0, 5)
        self.image_table.setHorizontalHeaderLabels(['✓', self._text('Görsel', 'Image'),
                                                    self._text('Durum', 'Status'),
                                                    self._text('Kıyafet', 'Outfit'),
                                                    self._text('Skor', 'Score')])
        self.image_table.horizontalHeaderItem(4).setToolTip(
            self._text('Modelin eşleşme güveni; düşük değerleri incele.',
                       'Model confidence; review low scores.'))
        self.image_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.image_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.image_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.image_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.image_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.image_table.currentCellChanged.connect(self._select_image)
        split.addWidget(self.image_table)
        detail = QWidget()
        dl = QVBoxLayout(detail)
        self.canvas = ClothingCropCanvas()
        self._tip(self.canvas, 'Bu görselde ana kişiyi sürükleyerek seç.',
                  'Drag to select the main person in this image.')
        self.canvas.box_changed.connect(self._set_selection)
        dl.addWidget(self.canvas, 2)
        self.clear_box_btn = self._button('Manuel seçimi temizle', 'Clear manual selection', self._clear_selection)
        dl.addWidget(self.clear_box_btn)
        self.before_text = QPlainTextEdit()
        self.after_text = QPlainTextEdit()
        self.detail_text = QPlainTextEdit()
        for text in (self.before_text, self.after_text, self.detail_text):
            text.setReadOnly(True)
            text.setMaximumHeight(115)
        dl.addWidget(self._label('Mevcut caption', 'Current caption'))
        dl.addWidget(self.before_text)
        dl.addWidget(self._label('Yeni caption — master tag başta', 'New caption — master tag first'))
        dl.addWidget(self.after_text)
        self._tip(self.after_text, 'Önizleme; caption ancak Uygula ile yazılır.',
                  'Preview only; Apply writes the caption.')
        dl.addWidget(self.detail_text)
        split.addWidget(detail)
        split.setSizes([460, 420])
        layout.addWidget(split, 1)
        row = QHBoxLayout()
        self.analyze_one_btn = self._button('Seçileni analiz et', 'Analyze selected', self._analyze_one)
        self.analyze_all_btn = self._button('Sırayla analiz et', 'Analyze in order', self._analyze_all, True)
        self._tip(self.analyze_all_btn, 'Kıyafetleri sırayla dener; eşleşen görseli tekrar taramaz.',
                  'Checks outfits in order; skips images already matched.')
        self.apply_btn = self._button('İşaretlilere uygula', 'Apply checked', self._apply_checked, True)
        self._tip(self.apply_btn, 'İşaretli caption önerilerini dosyalara yazar.',
                  'Writes checked caption proposals to files.')
        self.undo_btn = self._button('Son uygulamayı geri al', 'Undo last application', self._undo)
        for button in (self.analyze_one_btn, self.analyze_all_btn, self.apply_btn, self.undo_btn):
            row.addWidget(button)
        layout.addLayout(row)
        feedback_row = QHBoxLayout()
        self.correct_btn = self._button('Sonucu düzelt / hafızaya kaydet', 'Correct result / remember', self._correct_result)
        self._tip(self.correct_btn, 'Düzeltme sonraki karşılaştırmalara rehberlik eder; caption yazmaz.',
                  'Guides later comparisons; does not write a caption.')
        self.forget_btn = self._button('Bu seçimin düzeltmesini unut', 'Forget this selection correction', self._forget_result)
        feedback_row.addWidget(self.correct_btn)
        feedback_row.addWidget(self.forget_btn)
        layout.addLayout(feedback_row)
        self.only_review = self._check('Yalnız belirsiz/hatalı sonuçları göster', 'Show only review/error results')
        self.only_review.toggled.connect(self._filter_results)
        layout.addWidget(self.only_review)
        return page

    def _build_settings(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        form = QFormLayout()
        self.endpoint_edit = QLineEdit('http://127.0.0.1:11434')
        self.model_edit = QComboBox()
        self.model_edit.setEditable(True)
        self.model_edit.addItems(['qwen3-vl:4b', 'qwen3-vl:2b', 'qwen3-vl:8b'])
        self._tip(self.endpoint_edit, 'Yalnız yerel Ollama servisi.', 'Local Ollama only.')
        self._tip(self.model_edit, 'Görüntü destekli yerel model seç.', 'Choose a local vision model.')
        self.identity_spin = QDoubleSpinBox()
        self.part_spin = QDoubleSpinBox()
        for spin in (self.identity_spin, self.part_spin):
            spin.setRange(0.05, 1.0)
            spin.setSingleStep(0.05)
        self.person_combo = QComboBox()
        self.person_combo.addItem(self._text('En büyük kişi', 'Largest person'), 'largest')
        self.person_combo.addItem(self._text('Merkeze en yakın kişi', 'Closest to center'), 'center')
        self.person_combo.addItem(self._text('Yalnız manuel seçim', 'Manual selection only'), 'manual')
        self.size_spin = QSpinBox()
        self.size_spin.setRange(384, 1536)
        self.size_spin.setSingleStep(128)
        self.context_spin = QSpinBox()
        self.context_spin.setRange(4096, 32768)
        self.context_spin.setSingleStep(4096)
        self.timeout_spin = QSpinBox()
        self.timeout_spin.setRange(15, 1800)
        self.timeout_spin.setSuffix(' s')
        for widget, tr, en in (
            (self.endpoint_edit, 'Yerel servis adresi', 'Local endpoint'),
            (self.model_edit, 'Görüntü modeli', 'Vision model'),
            (self.identity_spin, 'Kıyafet kimlik eşiği', 'Outfit identity threshold'),
            (self.part_spin, 'Görünür parça eşiği', 'Visible-part threshold'),
            (self.person_combo, 'Otomatik ana kişi seçimi', 'Automatic main-person selection'),
            (self.size_spin, 'Analiz görüntüsü uzun kenarı', 'Analysis image longest side'),
            (self.context_spin, 'Model bağlamı', 'Model context window'),
            (self.timeout_spin, 'İstek zaman aşımı', 'Request timeout'),
        ):
            form.addRow(self._label(tr, en), widget)
        layout.addLayout(form)
        self.cache_check = self._check('Analiz önbelleği', 'Cache analysis results')
        self.unload_check = self._check('İşlem sonunda modeli bellekten çıkar', 'Unload model when a job ends')
        self.auto_video = self._check('Video çıkarımı bittikten sonra otomatik uygula',
                                      'Automatically apply after video extraction')
        self.auto_caption = self._check('Caption üretimi bittikten sonra otomatik uygula',
                                        'Automatically apply after caption generation')
        for check in (self.cache_check, self.unload_check, self.auto_video, self.auto_caption):
            layout.addWidget(check)
        self._tip(self.auto_video, 'Kabul edilen eşleşmeleri video sonrası uygula; yedek oluşturur.',
                  'Apply accepted matches after video; creates a backup.')
        self._tip(self.auto_caption, 'Kabul edilen eşleşmeleri caption sonrası uygula; yedek oluşturur.',
                  'Apply accepted matches after captioning; creates a backup.')
        actions = QHBoxLayout()
        self.check_model_btn = self._button('Modeli kontrol et', 'Check model', lambda: self._start('check'))
        self.pull_model_btn = self._button('Modeli indir', 'Download model', self._pull_model)
        actions.addWidget(self.check_model_btn)
        actions.addWidget(self.pull_model_btn)
        actions.addWidget(self._button('Ayarları kaydet', 'Save settings', self._save_settings_clicked, True))
        layout.addLayout(actions)
        layout.addStretch()
        return page

    def _load_settings(self):
        settings = self.store.load_settings()
        self.endpoint_edit.setText(settings.endpoint)
        self.model_edit.setCurrentText(settings.model)
        self.identity_spin.setValue(settings.identity_threshold)
        self.part_spin.setValue(settings.part_threshold)
        self.person_combo.setCurrentIndex(self.person_combo.findData(settings.person_mode))
        self.size_spin.setValue(settings.max_side)
        self.context_spin.setValue(settings.num_ctx)
        self.timeout_spin.setValue(settings.timeout_seconds)
        self.cache_check.setChecked(settings.use_cache)
        self.unload_check.setChecked(settings.unload_after_job)
        self.auto_video.setChecked(settings.auto_after_video)
        self.auto_caption.setChecked(settings.auto_after_caption)

    def _read_settings(self):
        return ClothingSettings(
            endpoint=self.endpoint_edit.text().strip(), model=self.model_edit.currentText().strip(),
            identity_threshold=self.identity_spin.value(), part_threshold=self.part_spin.value(),
            person_mode=self.person_combo.currentData(), max_side=self.size_spin.value(),
            num_ctx=self.context_spin.value(), timeout_seconds=self.timeout_spin.value(),
            use_cache=self.cache_check.isChecked(), unload_after_job=self.unload_check.isChecked(),
            auto_after_video=self.auto_video.isChecked(), auto_after_caption=self.auto_caption.isChecked(),
        ).validate()

    def _persist_settings(self):
        settings = self._read_settings()
        changed = asdict(settings) != asdict(self.store.load_settings())
        self.store.save_settings(settings)
        if changed:
            self._invalidate_previews()
        return settings, changed

    def _save_settings_clicked(self):
        try:
            self._persist_settings()
            self.status.setText(self._text('Ayarlar kaydedildi.', 'Settings saved.'))
        except Exception as exc:
            self._error(exc)

    def _reload_profiles(self, select_id=None):
        self._profiles = self.store.load()
        self.profile_list.blockSignals(True)
        self.profile_list.clear()
        selected = -1
        for i, profile in enumerate(self._profiles):
            label = f'{profile.name}  [{profile.master_tag}]'
            if not profile.enabled:
                label += self._text(' — pasif', ' — disabled')
            self.profile_list.addItem(label)
            if profile.id == select_id:
                selected = i
        self.profile_list.blockSignals(False)
        if selected < 0 and self._profiles:
            selected = 0
        self.profile_list.setCurrentRow(selected)
        if selected < 0:
            self._new_profile()

    def _profile_snapshot(self):
        return {
            'name': self.name_edit.text(), 'master': self.master_edit.text(),
            'family': self.family_edit.text(), 'enabled': self.enabled.isChecked(),
            'notes': self.notes_edit.toPlainText(), 'prompt': self.full_prompt.toPlainText(),
            'parts': [[self.parts_table.item(r, c).text() if self.parts_table.item(r, c) else ''
                       for c in range(2)] for r in range(self.parts_table.rowCount())],
            'references': [asdict(ref) for ref in self._refs],
        }

    def has_unsaved_profile_edits(self):
        return (self._clean_profile_snapshot is not None
                and self._profile_snapshot() != self._clean_profile_snapshot)

    def confirm_discard_profile_edits(self):
        if not self.has_unsaved_profile_edits():
            return True
        text = self._text('Kaydedilmemiş kıyafet değişiklikleri var. Kaydetmeden bırakılsın mı?',
                          'Outfit changes are not saved. Discard them?')
        return QMessageBox.question(self, 'Clothing', text, QMessageBox.Yes | QMessageBox.No,
                                    QMessageBox.No) == QMessageBox.Yes

    def _select_profile(self, row):
        if not 0 <= row < len(self._profiles):
            return
        profile = self._profiles[row]
        if not self.confirm_discard_profile_edits():
            old_row = next((i for i, p in enumerate(self._profiles) if p.id == self.current_profile_id), -1)
            self.profile_list.blockSignals(True)
            self.profile_list.setCurrentRow(old_row)
            self.profile_list.blockSignals(False)
            return
        self.current_profile_id = profile.id
        self.name_edit.setText(profile.name)
        self.master_edit.setText(profile.master_tag)
        self.family_edit.setText(profile.family)
        self.enabled.setChecked(profile.enabled)
        self.notes_edit.setPlainText(profile.notes)
        self.full_prompt.setPlainText(', '.join([profile.master_tag] + [p.tag for p in profile.parts]))
        self.parts_table.setRowCount(len(profile.parts))
        for i, part in enumerate(profile.parts):
            self.parts_table.setItem(i, 0, QTableWidgetItem(part.tag))
            self.parts_table.setItem(i, 1, QTableWidgetItem(part.description))
        self._refs = [ClothingReference(r.path, list(r.bbox) if r.bbox else None) for r in profile.references]
        self._refresh_refs()
        self._clean_profile_snapshot = self._profile_snapshot()

    def _new_profile(self):
        if not self.confirm_discard_profile_edits():
            return
        self.current_profile_id = None
        self.profile_list.blockSignals(True)
        self.profile_list.setCurrentRow(-1)
        self.profile_list.blockSignals(False)
        for edit in (self.name_edit, self.master_edit, self.family_edit):
            edit.clear()
        self.full_prompt.clear()
        self.notes_edit.clear()
        self.enabled.setChecked(True)
        self.parts_table.setRowCount(0)
        self._refs = []
        self._refresh_refs()
        self._clean_profile_snapshot = self._profile_snapshot()

    def _gather_profile(self):
        parts = []
        for row in range(self.parts_table.rowCount()):
            tag_item, description_item = self.parts_table.item(row, 0), self.parts_table.item(row, 1)
            tag = tag_item.text().strip() if tag_item else ''
            if tag:
                parts.append(ClothingPart(tag, description_item.text().strip() if description_item else ''))
        kwargs = {'id': self.current_profile_id} if self.current_profile_id else {}
        profile = ClothingProfile(
            name=self.name_edit.text().strip(), master_tag=self.master_edit.text().strip(),
            parts=parts, references=list(self._refs), enabled=self.enabled.isChecked(),
            family=self.family_edit.text().strip(), notes=self.notes_edit.toPlainText().strip(), **kwargs)
        profile.validate()
        if profile.enabled and not profile.references:
            raise ValueError(self._text('Etkin kıyafete en az bir referans ekle.', 'Add a reference before enabling the outfit.'))
        return profile

    def _save_profile(self):
        profile = self._gather_profile()
        self.store.upsert(profile)
        self.current_profile_id = profile.id
        self._clean_profile_snapshot = self._profile_snapshot()
        self._reload_profiles(profile.id)
        self._invalidate_previews()
        return profile

    def _save_profile_clicked(self):
        try:
            self._save_profile()
            self.status.setText(self._text('Kıyafet kaydedildi.', 'Outfit saved.'))
        except Exception as exc:
            self._error(exc)

    def _delete_profile(self):
        if not self.current_profile_id:
            return
        if QMessageBox.question(self, 'Clothing', self._text('Seçili profil silinsin mi?', 'Delete the selected profile?'),
                                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        try:
            self.store.delete(self.current_profile_id)
            self._clean_profile_snapshot = None
            self._invalidate_previews()
            self._reload_profiles()
        except Exception as exc:
            self._error(exc)

    def _split_prompt(self):
        try:
            tags = parse_tags(self.full_prompt.toPlainText())
            if len(tags) < 2:
                raise ValueError(self._text('Bir master ve en az bir parça tagi yaz.', 'Enter a master and at least one part tag.'))
            previous = {self.parts_table.item(i, 0).text(): self.parts_table.item(i, 1).text()
                        for i in range(self.parts_table.rowCount())
                        if self.parts_table.item(i, 0) and self.parts_table.item(i, 1)}
            self.master_edit.setText(tags[0])
            self.parts_table.setRowCount(len(tags) - 1)
            for i, tag in enumerate(tags[1:]):
                self.parts_table.setItem(i, 0, QTableWidgetItem(tag))
                self.parts_table.setItem(i, 1, QTableWidgetItem(previous.get(tag, '')))
        except Exception as exc:
            self._error(exc)

    def _add_part(self):
        row = self.parts_table.rowCount()
        self.parts_table.insertRow(row)
        self.parts_table.setItem(row, 0, QTableWidgetItem(''))
        self.parts_table.setItem(row, 1, QTableWidgetItem(''))
        self.parts_table.setCurrentCell(row, 0)
        self.parts_table.editItem(self.parts_table.item(row, 0))

    def _remove_part(self):
        row = self.parts_table.currentRow()
        if row >= 0:
            self.parts_table.removeRow(row)

    def _refresh_refs(self):
        self.reference_list.clear()
        for i, ref in enumerate(self._refs):
            suffix = self._text(' • alan seçildi', ' • cropped') if ref.bbox else ''
            image = load_rgb(self.store.reference_path(ref))
            image.thumbnail((48, 64), Image.Resampling.LANCZOS)
            raw = image.tobytes()
            qimage = QImage(raw, image.width, image.height, image.width * 3,
                            QImage.Format_RGB888).copy()
            pixmap = QPixmap.fromImage(qimage)
            item = QListWidgetItem()
            item.setSizeHint(QSize(0, 72))
            self.reference_list.addItem(item)
            row = QWidget(self.reference_list)
            row.setAttribute(Qt.WA_TransparentForMouseEvents)
            contents = QHBoxLayout(row)
            contents.setContentsMargins(6, 4, 6, 4)
            contents.setSpacing(8)
            thumbnail = QLabel(row)
            thumbnail.setFixedSize(48, 64)
            thumbnail.setAlignment(Qt.AlignCenter)
            thumbnail.setPixmap(pixmap)
            contents.addWidget(thumbnail)
            contents.addWidget(QLabel(
                f'{i + 1:02d}  {Path(ref.path).name[:24]}{suffix}', row), 1)
            self.reference_list.setItemWidget(item, row)

    def _add_references(self):
        paths, _ = QFileDialog.getOpenFileNames(self, 'References', '', IMAGE_FILTER)
        try:
            for path in paths:
                reference = self.store.import_reference(Path(path))
                if not any(r.path == reference.path for r in self._refs):
                    self._refs.append(reference)
            self._refresh_refs()
        except Exception as exc:
            self._error(exc)

    def _remove_reference(self):
        row = self.reference_list.currentRow()
        if row >= 0:
            self._refs.pop(row)
            self._refresh_refs()

    def _crop_reference(self):
        row = self.reference_list.currentRow()
        if row < 0:
            return
        try:
            ref = self._refs[row]
            dialog = QDialog(self)
            dialog.setWindowTitle(self._text('Referanstaki ana kıyafet', 'Main outfit in reference'))
            dialog.resize(660, 720)
            lay = QVBoxLayout(dialog)
            canvas = ClothingCropCanvas()
            canvas.setToolTip(self._text('Hedef kişi ve kıyafeti çevrele.', 'Select the person and outfit.'))
            canvas.set_image(load_rgb(self.store.reference_path(ref)), ref.bbox)
            lay.addWidget(canvas, 1)
            clear = QPushButton(self._text('Tam görsel', 'Full image'))
            def reset():
                canvas.bbox = None
                canvas.update()
            clear.clicked.connect(reset)
            lay.addWidget(clear)
            buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            buttons.accepted.connect(dialog.accept)
            buttons.rejected.connect(dialog.reject)
            lay.addWidget(buttons)
            if dialog.exec_() == QDialog.Accepted:
                self._refs[row].bbox = list(canvas.bbox) if canvas.bbox else None
                self._refresh_refs()
        except Exception as exc:
            self._error(exc)

    def _analyze_references(self):
        try:
            profile = self._save_profile()
            self._start('references', profile=profile)
        except Exception as exc:
            self._error(exc)

    def _rescan_folder(self):
        if self._folder and not self.is_busy():
            self.reload_folder(self._folder)

    def _choose_folder(self):
        folder = QFileDialog.getExistingDirectory(self, 'Images', self._folder)
        if folder:
            self.reload_folder(folder)

    def _choose_images(self):
        paths, _ = QFileDialog.getOpenFileNames(self, 'Images', self._folder, IMAGE_FILTER)
        if paths:
            self._folder = ''
            self.folder_label.setText(self._text(f'{len(paths)} görsel seçildi', f'{len(paths)} images selected'))
            self.set_images(paths)

    def reload_folder(self, folder):
        if self.is_busy():
            return
        try:
            self._folder = str(folder)
            self.folder_label.setText(self._folder)
            self.set_images(find_images(Path(folder), self.recursive.isChecked()))
        except Exception as exc:
            self.status.setText(str(exc))

    def set_images(self, paths):
        if self.is_busy():
            return
        self._paths = list(dict.fromkeys(str(Path(p).resolve()) for p in paths))
        self._row_by_path = {path: row for row, path in enumerate(self._paths)}
        self._results.clear()
        self._previews.clear()
        self._sync_errors.clear()
        self._active_image = None
        self.image_table.blockSignals(True)
        self.image_table.setRowCount(len(self._paths))
        for row, path in enumerate(self._paths):
            check = QTableWidgetItem('')
            check.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable)
            check.setCheckState(Qt.Unchecked)
            self.image_table.setItem(row, 0, check)
            for col, text in enumerate((Path(path).name, '—', '', ''), start=1):
                item = QTableWidgetItem(text)
                item.setToolTip(path)
                self.image_table.setItem(row, col, item)
        self.image_table.blockSignals(False)
        self.canvas.clear()
        self.before_text.clear()
        self.after_text.clear()
        self.detail_text.clear()
        if self._paths:
            self.image_table.setCurrentCell(0, 1)
            self._select_image(0, 1, -1, -1)
        self._filter_results()

    def ensure_image(self, image):
        """Append a library-selected image without clearing other analyses or drafts.

        This navigation helper does not select, analyze, or write captions. Existing
        review checkboxes, results, crop selections and undo history are preserved.
        """
        if self.is_busy():
            return None
        path = Path(image).resolve()
        if not path.is_file():
            return None
        key = str(path)
        if key in self._row_by_path:
            row = self._row_by_path[key]
            self.image_table.setRowHidden(row, False)
            return row
        row = len(self._paths)
        previous_signals = self.image_table.blockSignals(True)
        try:
            self.image_table.insertRow(row)
            check = QTableWidgetItem('')
            check.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable)
            check.setCheckState(Qt.Unchecked)
            self.image_table.setItem(row, 0, check)
            for col, value in enumerate((path.name, '—', '', ''), start=1):
                item = QTableWidgetItem(value)
                item.setToolTip(key)
                self.image_table.setItem(row, col, item)
            self._paths.append(key)
            self._row_by_path[key] = row
            self.image_table.setRowHidden(row, False)
        finally:
            self.image_table.blockSignals(previous_signals)
        return row

    def _select_image(self, row, column=0, old_row=-1, old_column=-1):
        if not 0 <= row < len(self._paths):
            return
        self._active_image = self._paths[row]
        try:
            bbox = load_selection(self.store, self._active_image)
            result = self._results.get(self._active_image)
            if bbox is None and result is not None and result.bbox:
                bbox = result.bbox
            self._displayed_digest = file_digest(Path(self._active_image))
            self.canvas.set_image(load_rgb(Path(self._active_image)), bbox)
            self._show_details(self._active_image)
        except Exception as exc:
            self.canvas.clear()
            self.status.setText(str(exc))

    def _show_details(self, path):
        result, preview = self._results.get(path), self._previews.get(path)
        try:
            caption = Path(path).with_suffix('.txt')
            current = caption.read_text(encoding='utf-8-sig') if caption.exists() else ''
        except (OSError, UnicodeError) as exc:
            current = str(exc)
        self.before_text.setPlainText(current)  # Never display an obsolete before-snapshot.
        self.after_text.setPlainText(preview.new_text if preview else '')
        detail = result.to_dict() if result else {}
        if preview and preview.warnings:
            detail['warnings'] = preview.warnings
        if path in self._sync_errors:
            detail['caption_sync_error'] = self._sync_errors[path]
        self.detail_text.setPlainText(json.dumps(detail, ensure_ascii=False, indent=2) if detail else '')

    def refresh_captions(self, paths=None):
        """Rebase saved text only. Preserve images, crops, unrelated results and drafts.

        Read notifications never emit captions_changed, preventing signal loops.
        A busy worker owns its result set; defer refresh until native finished.
        """
        if self.is_busy():
            if paths is None:
                self._pending_caption_all = True
            else:
                self._pending_caption_paths.update(path_key(p) for p in paths)
            return
        wanted = None if paths is None else {path_key(p) for p in paths}
        targets = [p for p in self._paths
                   if wanted is None or path_key(Path(p).with_suffix('.txt')) in wanted]
        rebaser = CaptionPreviewRebaser(self.store)
        for path in targets:
            result = self._results.get(path)
            row = self._row_by_path[path]
            previous = self._previews.get(path)
            had_error = path in self._sync_errors
            self._sync_errors.pop(path, None)
            if result is None or result.status != 'matched':
                self._previews.pop(path, None)
                continue
            try:
                preview = rebaser.prepare(result)
                self._previews[path] = preview
                if (previous is None or preview.before != previous.before
                        or preview.after != previous.after or preview.state_before != previous.state_before):
                    # A changed proposal needs renewed review, never silently re-check it.
                    self.image_table.item(row, 0).setCheckState(Qt.Unchecked)
                if had_error:
                    self.image_table.item(row, 2).setText(self._text('Eşleşti', 'Matched'))
            except Exception as exc:
                self._previews.pop(path, None)
                self._sync_errors[path] = str(exc)
                self.image_table.item(row, 0).setCheckState(Qt.Unchecked)
                self.image_table.item(row, 2).setText(self._text('İnceleme', 'Review'))
        if self._active_image in targets:
            self._show_details(self._active_image)
        self._filter_results()

    def _set_selection(self, bbox):
        if not self._active_image or self.is_busy():
            return
        try:
            save_selection(self.store, self._active_image, bbox)
            self._invalidate_previews(self._active_image)
            self.status.setText(self._text('Ana kıyafet alanı kaydedildi; yeniden analiz et.',
                                          'Main-outfit area saved; analyze again.'))
        except Exception as exc:
            self._error(exc)

    def _clear_selection(self):
        if self._active_image:
            self._set_selection(None)
            self.canvas.bbox = None
            self.canvas.update()

    def _invalidate_previews(self, path=None):
        paths = [path] if isinstance(path, str) else list(self._paths)
        for item in paths:
            self._results.pop(item, None)
            self._previews.pop(item, None)
            self._sync_errors.pop(item, None)
            if item in self._paths:
                row = self._row_by_path[item]
                self.image_table.item(row, 0).setCheckState(Qt.Unchecked)
                for column, text in ((2, '—'), (3, ''), (4, '')):
                    self.image_table.item(row, column).setText(text)
        if self._active_image:
            self._show_details(self._active_image)
        self._filter_results()

    def _correct_result(self):
        if not self._active_image or self.is_busy():
            return
        try:
            path, bbox = self._active_image, self.canvas.bbox
            if not bbox:
                raise ValueError(self._text('Önce ana kişiyi/kıyafeti sürükleyerek seç.',
                                            'Draw a main-person/outfit box first.'))
            expected = getattr(self, '_displayed_digest', '')
            if not expected or expected != file_digest(Path(path)):
                raise ValueError('Image changed since display. Reload it before correcting.')
            profiles = [p for p in self.store.load() if p.enabled]
            old_result = self._results.get(path)
            dialog = QDialog(self)
            dialog.setWindowTitle(self._text('Kıyafet düzeltmesi', 'Outfit correction'))
            dialog.resize(560, 540)
            lay = QVBoxLayout(dialog)
            combo = QComboBox(); combo.addItem(self._text('Kayıtlı kıyafetlerden hiçbiri', 'None of the registered outfits'), '')
            combo.setToolTip(self._text('Doğru kıyafeti seç; master tag korunur.',
                                        'Choose the outfit; the master tag is preserved.'))
            for p in profiles:
                combo.addItem(p.name + ' — ' + p.master_tag, p.id)
            lay.addWidget(combo)
            parts = QListWidget()
            parts.setToolTip(self._text('Yalnız bu kişide görünen parçaları işaretle; caption henüz yazılmaz.',
                                        'Mark only visible parts; caption is not written yet.'))
            lay.addWidget(parts, 1)
            note = QPlainTextEdit(); note.setMaximumHeight(80)
            note.setPlaceholderText(self._text('İsteğe bağlı: model hangi ayrıntıyı karıştırdı?',
                                               'Optional: which detail did the model confuse?'))
            lay.addWidget(note)
            example = QCheckBox(self._text('Benzer görsellerde düzeltme örneği olarak kullan',
                                           'Use as a correction example for similar images'))
            example.setChecked(True); lay.addWidget(example)
            def fill(_index=None):
                parts.clear()
                profile = next((p for p in profiles if p.id == combo.currentData()), None)
                known = set(old_result.tags if old_result is not None and old_result.profile_id == combo.currentData() else [])
                if profile:
                    for part in profile.parts:
                        item = QListWidgetItem(part.tag); item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                        item.setCheckState(Qt.Checked if part.tag in known else Qt.Unchecked); parts.addItem(item)
            combo.currentIndexChanged.connect(fill)
            if old_result and combo.findData(old_result.profile_id) >= 0:
                combo.setCurrentIndex(combo.findData(old_result.profile_id))
            fill()
            buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
            def accept():
                if combo.currentData() and not any(parts.item(i).checkState() == Qt.Checked for i in range(parts.count())):
                    QMessageBox.warning(dialog, 'Clothing', self._text('En az bir görünür parça seç.', 'Select at least one visible part.'))
                    return
                dialog.accept()
            buttons.accepted.connect(accept); buttons.rejected.connect(dialog.reject); lay.addWidget(buttons)
            if dialog.exec_() != QDialog.Accepted:
                return
            feedback = {'image': path, 'bbox': list(bbox), 'profile_id': combo.currentData(),
                        'visible_tags': [parts.item(i).text() for i in range(parts.count()) if parts.item(i).checkState() == Qt.Checked],
                        'expected_image_digest': expected,
                        'rejected_profile_id': old_result.profile_id if old_result else '',
                        'note': note.toPlainText(), 'use_as_example': example.isChecked()}
            self._invalidate_previews()  # memory update invalidates all previous model-context previews
            self._start('feedback', feedback=feedback)
        except Exception as exc:
            self._error(exc)

    def _forget_result(self):
        if not self._active_image or not self.canvas.bbox or self.is_busy():
            return
        if QMessageBox.question(self, 'Clothing', self._text(
                'Bu görsel/alan için düzeltme pasifleştirilsin mi? Caption değişmez.',
                'Deactivate corrections for this image/box? The caption stays unchanged.'),
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) == QMessageBox.Yes:
            feedback = {'image': self._active_image, 'bbox': list(self.canvas.bbox)}
            self._invalidate_previews()
            self._start('forget_feedback', feedback=feedback)

    def _analyze_one(self):
        if self._active_image:
            self._invalidate_previews(self._active_image)
            self._start('analyze', paths=[self._active_image])

    def _analyze_all(self):
        if not self._paths:
            self._error(self._text('Önce görsel veya klasör seç.', 'Select images or a folder first.'))
            return
        self._invalidate_previews()
        self._start('analyze', paths=self._paths)

    def _apply_checked(self):
        previews = [self._previews[path] for i, path in enumerate(self._paths)
                    if self.image_table.item(i, 0).checkState() == Qt.Checked and path in self._previews]
        if not previews:
            self._error(self._text('Önce analiz et ve kabul edilmiş sonuçları işaretle.',
                                  'Analyze first and check accepted results.'))
            return
        question = self._text(f'{len(previews)} caption dosyasına yazılsın mı? Geri alma kaydı oluşturulacak.',
                              f'Write {len(previews)} caption files? Undo records will be created.')
        if QMessageBox.question(self, 'Clothing', question, QMessageBox.Yes | QMessageBox.No,
                                QMessageBox.No) == QMessageBox.Yes:
            self._start('apply', previews=previews)

    def _undo(self):
        question = self._text('Son kıyafet uygulaması geri alınsın mı? Sonradan değişen dosyalar korunur.',
                              'Undo the last clothing application? Files edited afterwards are protected.')
        if QMessageBox.question(self, 'Clothing', question, QMessageBox.Yes | QMessageBox.No,
                                QMessageBox.No) == QMessageBox.Yes:
            self._start('undo')

    def _pull_model(self):
        text = self._text('Seçilen model internetten indirilecek; birkaç GB alan kullanabilir. Devam edilsin mi?',
                          'The selected model will be downloaded; it may require several GB. Continue?')
        if QMessageBox.question(self, 'Ollama', text, QMessageBox.Yes | QMessageBox.No,
                                QMessageBox.No) == QMessageBox.Yes:
            self._start('pull')

    def _other_job_running(self):
        window = self.window()
        video = getattr(window, 'processing_thread', None)
        studio = getattr(window, 'caption_studio_page', None)
        generate = getattr(studio, 'generate_tab', None)
        caption = getattr(generate, 'captioning_thread', None)
        others = [getattr(getattr(window, name, None), '_thread', None)
                  for name in ('char_sort_page', 'upscale_page', 'training_page')]
        return any(thread is not None and thread.isRunning() for thread in [video, caption] + others)

    def _start(self, operation, **kwargs):
        if self.is_busy():
            return
        try:
            if self._other_job_running():
                raise RuntimeError(self._text('Önce video/caption işlemini bitir veya durdur.',
                                              'Finish or stop the video/caption job first.'))
            studio = getattr(self.window(), 'caption_studio_page', None)
            if operation in ('apply', 'undo') and studio is not None and studio.has_unsaved_caption_edits():
                raise RuntimeError(self._text('Edit sekmesinde kaydedilmemiş değişiklikler var. Önce kaydet veya geri al.',
                                              'There are unsaved edits in the Edit tab. Save or revert them first.'))
            if operation in ('undo', 'feedback', 'forget_feedback'):
                settings, changed = ClothingSettings(), False
            else:
                settings, changed = self._persist_settings()
            if operation == 'apply' and changed:
                raise RuntimeError(self._text('Ayarlar değişti; tekrar analiz et.', 'Settings changed; analyze again.'))
            self._last_operation = operation
            self._caption_write_paths = (None if operation == 'undo' else
                [str(Path(p.image).with_suffix('.txt')) for p in kwargs.get('previews', [])]
                if operation == 'apply' else [])
            self.worker = ClothingTaggerWorker(operation, self.store, settings, parent=self, **kwargs)
            self.worker.image_result.connect(self._on_result)
            self.worker.progress.connect(self._on_progress)
            self.worker.log_message.connect(self.status.setText)
            self.worker.completed.connect(self._on_completed)
            self.worker.failed.connect(self._error)
            self.worker.finished.connect(self._job_finished)  # native QThread finished, never shadowed
            self._set_busy(True)
            self.progress.setRange(0, 0)
            observer = getattr(self, '_ai_before_worker_start', None)
            if observer is not None:
                observer(self.worker)
            self.worker.start()
        except Exception as exc:
            if self.worker is not None and not self.worker.isRunning():
                self.worker.deleteLater()
                self.worker = None
            self._set_busy(False)
            self._error(exc)

    def _set_busy(self, busy):
        self.busy_changed.emit(busy)
        self.library.setEnabled(not busy)
        self.settings_page.setEnabled(not busy)
        for control in (self.browse_folder_btn, self.browse_files_btn, self.recursive,
                        self.analyze_one_btn, self.analyze_all_btn, self.apply_btn,
                        self.undo_btn, self.clear_box_btn, self.canvas, self.correct_btn, self.forget_btn):
            control.setEnabled(not busy)
        self.cancel_btn.setEnabled(busy)

    def is_busy(self):
        # Stay busy until the native finished handler has cleared this worker.
        return self.worker is not None

    def stop(self):
        if self.worker is not None:
            self.worker.stop()
            self.status.setText(self._text('Durduruluyor; tamamlanmış yazmalar geri alınabilir.',
                                          'Cancelling; completed writes remain undoable.'))

    def _on_progress(self, current, total, text):
        self.progress.setRange(0, max(total, 1))
        self.progress.setValue(current)
        self.status.setText(text)

    def _on_result(self, result, preview):
        if result.image not in self._paths:
            return
        self._results[result.image] = result
        self._sync_errors.pop(result.image, None)
        if preview is not None and result.status == 'matched':
            self._previews[result.image] = preview
        else:
            self._previews.pop(result.image, None)
        row = self._row_by_path[result.image]
        labels = {'matched': ('Eşleşti', 'Matched'), 'review': ('İnceleme', 'Review'),
                  'no_match': ('Eşleşmedi', 'No match'), 'error': ('Hata', 'Error')}
        label = self._text(*labels.get(result.status, ('?', '?')))
        self.image_table.item(row, 2).setText(label)
        self.image_table.item(row, 3).setText(result.profile_name)
        self.image_table.item(row, 4).setText(f'{result.score:.2f}' if result.score else '')
        self.image_table.item(row, 0).setCheckState(Qt.Checked if result.image in self._previews else Qt.Unchecked)
        if self._active_image == result.image:
            self._select_image(row)
        self.image_table.setRowHidden(row, self.only_review.isChecked() and result.status not in ('review', 'error'))

    def _filter_results(self):
        for row, path in enumerate(self._paths):
            result = self._results.get(path)
            hide = (self.only_review.isChecked() and result is not None
                    and result.status not in ('review', 'error') and path not in self._sync_errors)
            self.image_table.setRowHidden(row, hide)

    def _on_completed(self, data):
        if self._last_operation == 'references' and 'suggestions' in data:
            for row in range(self.parts_table.rowCount()):
                tag = self.parts_table.item(row, 0).text()
                item = self.parts_table.item(row, 1)
                if item is not None and not item.text().strip():
                    item.setText(data['suggestions'].get(tag, ''))
            if not self.notes_edit.toPlainText().strip():
                self.notes_edit.setPlainText(data.get('summary', ''))
            self.status.setText(self._text('Öneriler dolduruldu. Kontrol edip Profili kaydet düğmesine bas.',
                                          'Suggestions filled. Review them and click Save profile.'))
        else:
            self.status.setText(json.dumps(data, ensure_ascii=False))
        # Disk synchronization happens in _job_finished, including failure/cancel paths.
        # Do not clear visual decisions for a caption-only write.

    def _job_finished(self):
        worker, self.worker = self.worker, None
        try:
            paths = self._caption_write_paths if self._last_operation in ('apply', 'undo') else []
            pending = None if self._pending_caption_all or paths is None else list(
                self._pending_caption_paths | {path_key(p) for p in paths})
            self._pending_caption_paths.clear()
            self._pending_caption_all = False
            if pending is None or pending:
                self.refresh_captions(pending)
            if self._last_operation in ('apply', 'undo'):
                # Also reconcile partially written, cancelled or failed jobs. Readers
                # inspect disk; this notification does not claim all writes succeeded.
                self.captions_changed.emit(paths)
        finally:
            self._set_busy(False)
            if self.progress.maximum() == 0:
                self.progress.setRange(0, 1)
                self.progress.setValue(1)
            if worker is not None:
                worker.deleteLater()

    def _error(self, error):
        self.status.setText(str(error))
        QMessageBox.warning(self, 'Clothing', str(error))

    def shutdown(self, timeout_ms=2500):
        if self.worker is not None:
            self.worker.stop()
            return self.worker.wait(timeout_ms)
        return True

    def update_language(self, lang):
        self.lang = lang
        for setter, tr, en in self._translations:
            setter(self._text(tr, en))
        for i, names in enumerate((('Kıyafet kütüphanesi', 'Outfit library'),
                                    ('Analiz ve önizleme', 'Analyze and preview'),
                                    ('Model ve ayarlar', 'Model and settings'))):
            self.tabs.setTabText(i, self._text(*names))
        for i, tips in enumerate((('Kıyafet profili ve referanslar.', 'Outfit profiles and references.'),
                                  ('Görselleri sırayla analiz et.', 'Analyze images in order.'),
                                  ('Yerel model ve otomasyon.', 'Local model and automation.'))):
            self.tabs.setTabToolTip(i, self._text(*tips))
        for i, names in enumerate((('En büyük kişi', 'Largest person'),
                                    ('Merkeze en yakın kişi', 'Closest to center'),
                                    ('Yalnız manuel seçim', 'Manual selection only'))):
            self.person_combo.setItemText(i, self._text(*names))
        self.image_table.setHorizontalHeaderLabels(['✓', self._text('Görsel', 'Image'),
                                                    self._text('Durum', 'Status'),
                                                    self._text('Kıyafet', 'Outfit'),
                                                    self._text('Skor', 'Score')])
        self.image_table.horizontalHeaderItem(4).setToolTip(
            self._text('Modelin eşleşme güveni; düşük değerleri incele.',
                       'Model confidence; review low scores.'))
        self.parts_table.setHorizontalHeaderLabels(['Tag', self._text('Görsel anlamı (düzenlenebilir)', 'Visual meaning (editable)')])

    def refresh_styles(self):
        self.canvas.update()
        return theme.refresh_styles(self)
