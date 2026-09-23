"""Fixed-recipe local checkpoint/LoRA comparison UI."""
from __future__ import annotations
import json
from pathlib import Path
from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtGui import QDesktopServices, QPixmap
from PyQt5.QtWidgets import (QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QPushButton,
    QLineEdit, QFileDialog, QCheckBox, QComboBox, QSpinBox, QDoubleSpinBox, QProgressBar,
    QPlainTextEdit, QListWidget, QListWidgetItem, QSplitter, QWidget, QMessageBox, QScrollArea)
from . import theme
from .studio_tasks import StudioTaskWidget
from src.core.checkpoint_compare import ComparisonConfig, WebUIClient, run_comparison, DEFAULT_OUTPUT


class CheckpointCompareWidget(StudioTaskWidget):
    def __init__(self, lang='en', parent=None):
        super().__init__(lang, parent)
        self.catalog_data = None
        self.last_report = None
        self.samples = []
        self._sample_pixmap = None
        layout = QVBoxLayout(self)
        top = QHBoxLayout(); self.endpoint = QLineEdit('http://127.0.0.1:7860')
        self.connect_btn = self._button('Bağlan / modelleri yenile', 'Connect / refresh models', self._connect)
        self.mode = QComboBox(); self.mode.addItem('LoRA epoch / checkpoint', 'lora'); self.mode.addItem('Tam model / Full checkpoint', 'checkpoint')
        top.addWidget(self.endpoint, 1); top.addWidget(self.connect_btn); top.addWidget(self.mode)
        layout.addLayout(top)
        split = QSplitter(Qt.Horizontal)
        input_panel = QWidget(); fields = QVBoxLayout(input_panel)
        fields.addWidget(QLabel('Adaylar / Candidates (max 16)'))
        self.candidates = QListWidget(); self.candidates.setMinimumHeight(95); fields.addWidget(self.candidates, 1)
        form = QFormLayout(); self.base = QComboBox(); form.addRow('Sabit ana model / Base', self.base)
        self.sampler = QComboBox(); self.sampler.addItem('Euler'); form.addRow('Sampler', self.sampler)
        self.scheduler = QLineEdit('Automatic'); form.addRow('Scheduler', self.scheduler)
        self.vae = QLineEdit('Automatic'); form.addRow('VAE (WebUI name)', self.vae)
        self.steps = QSpinBox(); self.steps.setRange(1, 150); self.steps.setValue(28)
        self.cfg = QDoubleSpinBox(); self.cfg.setRange(0.1, 30); self.cfg.setValue(6)
        self.weight = QDoubleSpinBox(); self.weight.setRange(-2, 2); self.weight.setSingleStep(.1); self.weight.setValue(1)
        self.clip_skip = QSpinBox(); self.clip_skip.setRange(1, 12); self.clip_skip.setValue(1)
        self.width = QSpinBox(); self.height = QSpinBox()
        for dim in (self.width, self.height):
            dim.setRange(256, 2048); dim.setSingleStep(64); dim.setValue(1024)
        compact = QHBoxLayout()
        for label, widget in [('Steps', self.steps), ('CFG', self.cfg), ('LoRA', self.weight), ('Clip skip', self.clip_skip)]:
            compact.addWidget(QLabel(label)); compact.addWidget(widget)
        form.addRow(compact)
        sizes = QHBoxLayout(); sizes.addWidget(self.width); sizes.addWidget(QLabel('×')); sizes.addWidget(self.height)
        form.addRow('Çözünürlük / Size', sizes)
        self.seeds = QLineEdit('42, 1234'); form.addRow('Sabit seedler / Fixed seeds', self.seeds)
        self.baseline = QCheckBox(); self.baseline.setChecked(True)
        self.label_text(self.baseline, 'LoRA olmadan ana modeli de üret', 'Include no-LoRA baseline')
        form.addRow(self.baseline)
        fields.addLayout(form)
        fields.addWidget(QLabel('Promptlar — her satır ayrı bir test / One test per line'))
        self.prompts = QPlainTextEdit(); self.prompts.setPlaceholderText('1girl, your_trigger, full body, standing, simple background')
        self.prompts.setMaximumHeight(100); fields.addWidget(self.prompts)
        fields.addWidget(QLabel('Negative prompt'))
        self.negative = QPlainTextEdit('watermark, signature, text'); self.negative.setMaximumHeight(65); fields.addWidget(self.negative)
        output_row = QHBoxLayout(); self.output = QLineEdit(str(DEFAULT_OUTPUT)); self.output.setReadOnly(True)
        self.output_btn = self._button('Çıktı', 'Output', self._output)
        output_row.addWidget(self.output, 1); output_row.addWidget(self.output_btn); fields.addLayout(output_row)
        scroller = QScrollArea(); scroller.setWidgetResizable(True); scroller.setWidget(input_panel)
        split.addWidget(scroller)
        right = QWidget(); results = QVBoxLayout(right)
        self.sample_list = QListWidget(); self.sample_list.setMaximumHeight(170)
        self.sample_list.currentRowChanged.connect(self._show_sample); results.addWidget(self.sample_list)
        self.preview = QLabel(); self.preview.setMinimumSize(200, 250); self.preview.setAlignment(Qt.AlignCenter)
        results.addWidget(self.preview, 1)
        self.notes = QPlainTextEdit(); self.notes.setReadOnly(True); self.notes.setMaximumHeight(110); results.addWidget(self.notes)
        report_actions = QHBoxLayout()
        self.open_btn = self._button('Yan yana raporu aç', 'Open side-by-side report', self._open_report)
        self.load_btn = self._button('Önceki deneyi aç', 'Open saved comparison', self._load_report)
        report_actions.addWidget(self.open_btn); report_actions.addWidget(self.load_btn); results.addLayout(report_actions)
        split.addWidget(right); split.setSizes([600, 430]); layout.addWidget(split, 1)
        actions = QHBoxLayout(); self.run_btn = self._button('Karşılaştırmayı başlat', 'Start comparison', self._run, True)
        self.stop_button = self._button('Örnek sonunda durdur', 'Stop after sample', self.stop); self.stop_button.setEnabled(False)
        actions.addWidget(self.run_btn); actions.addWidget(self.stop_button); actions.addStretch(); layout.addLayout(actions)
        self.progress = QProgressBar(); layout.addWidget(self.progress)
        self.status = QLabel(); self.status.setWordWrap(True); layout.addWidget(self.status)
        self._controls = [self.endpoint, self.connect_btn, self.mode, input_panel, self.run_btn, self.load_btn]
        for widget, tr, en in (
            (self.endpoint, 'Yerel Forge/A1111 API adresi.', 'Local Forge/A1111 API address.'),
            (self.connect_btn, 'WebUI --api ile açık olmalı.', 'WebUI must run with --api.'),
            (self.run_btn, 'Adaylar aynı prompt ve seedlerle üretilir.',
                           'Candidates use the same prompts and seeds.'),
            (self.stop_button, 'Geçerli örnek bitince durur.', 'Stops after the current sample.'),
        ):
            self.tip_text(widget, tr, en)
        self.mode.currentIndexChanged.connect(self._populate)
        self.endpoint.textChanged.connect(self._invalidate_catalog)

    def _button(self, tr, en, fn, primary=False):
        button = QPushButton(); self.label_text(button, tr, en)
        theme.bind_style(button, theme.btn_primary if primary else theme.btn_secondary)
        button.clicked.connect(fn); return button

    def _invalidate_catalog(self):
        self.catalog_data = None; self.candidates.clear(); self.base.clear()

    def _connect(self):
        endpoint = self.endpoint.text().strip()
        self.start_task(lambda cancel, progress: WebUIClient(endpoint, cancel=cancel).catalog(), self._connected)

    def _connected(self, catalog):
        self.catalog_data = catalog
        self.base.clear(); self.base.addItems([m['title'] for m in catalog['checkpoints'] if m.get('title')])
        self.sampler.clear(); self.sampler.addItems([m['name'] for m in catalog['samplers'] if m.get('name')])
        if self.sampler.findText('Euler') >= 0:
            self.sampler.setCurrentText('Euler')
        self._populate()
        self.status.setText(self.text('Yerel model listesi alındı; karşılaştırılacak adayları işaretle.',
                                      'Local models loaded. Check the candidates to compare.'))

    def _populate(self, *_):
        self.candidates.clear()
        is_lora = self.mode.currentData() == 'lora'
        self.base.setEnabled(is_lora); self.weight.setEnabled(is_lora); self.baseline.setEnabled(is_lora)
        if not self.catalog_data:
            return
        field = 'name' if is_lora else 'title'
        for model in self.catalog_data['loras' if is_lora else 'checkpoints']:
            if model.get(field):
                item = QListWidgetItem(model[field]); item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Unchecked); self.candidates.addItem(item)

    def _config(self):
        candidates = [self.candidates.item(i).text() for i in range(self.candidates.count())
                      if self.candidates.item(i).checkState() == Qt.Checked]
        seeds = [int(s.strip()) for s in self.seeds.text().split(',') if s.strip()]
        return ComparisonConfig(endpoint=self.endpoint.text().strip(), mode=self.mode.currentData(),
            candidates=candidates, base_checkpoint=self.base.currentText(),
            prompts=[p.strip() for p in self.prompts.toPlainText().splitlines() if p.strip()],
            negative_prompt=self.negative.toPlainText().strip(), seeds=seeds,
            sampler=self.sampler.currentText(), scheduler=self.scheduler.text().strip(),
            steps=self.steps.value(), cfg_scale=self.cfg.value(), width=self.width.value(), height=self.height.value(),
            clip_skip=self.clip_skip.value(), vae=self.vae.text().strip(), lora_weight=self.weight.value(),
            include_baseline=self.baseline.isChecked()).validate()

    def _run(self):
        try:
            if not self.catalog_data:
                raise ValueError(self.text('Önce yerel WebUI servisine bağlan.', 'Connect to local WebUI first.'))
            config = self._config(); output = self.output.text()
            question = self.text(f'{len(config.jobs())} görsel yerel serviste üretilecek. Başlatılsın mı?',
                                 f'Generate {len(config.jobs())} images on your local service?')
            if QMessageBox.question(self, 'Comparison', question, QMessageBox.Yes | QMessageBox.No,
                                    QMessageBox.No) != QMessageBox.Yes:
                return
            self.start_task(lambda cancel, progress: run_comparison(config, output, cancel=cancel, progress=progress), self._done)
        except Exception as exc:
            self.error(str(exc))

    def _output(self):
        folder = QFileDialog.getExistingDirectory(self, 'Comparison output (not training dataset)', self.output.text())
        if folder:
            self.output.setText(folder)

    def _done(self, data):
        self.status.setText(json.dumps(data, ensure_ascii=False))
        self._read_report(Path(data['manifest']))
        if data.get('error'):
            self.error(data['error'])

    def _read_report(self, path):
        path = Path(path).resolve()
        if path.stat().st_size > 40_000_000:
            raise ValueError('Comparison report too large.')
        data = json.loads(path.read_text(encoding='utf-8'))
        if data.get('schema_version') != 1 or not isinstance(data.get('results'), list):
            raise ValueError('Invalid comparison report.')
        samples = []
        for sample in data['results']:
            name = sample['file']; target = path.parent / name
            if Path(name).name != name or '\\' in name or target.is_symlink() or not target.resolve().is_relative_to(path.parent):
                raise ValueError('Unsafe sample path in comparison report.')
            samples.append((target, sample))
        self.samples = samples; self.last_report = path.parent / 'index.html'
        self.sample_list.clear()
        for _, sample in samples:
            self.sample_list.addItem(f'{sample["candidate"] or "Base"} | prompt {sample["prompt_index"]+1} | seed {sample["seed"]}')
        self.notes.setPlainText(self.text('Koşullar sabitlenir; farklı donanım/eklenti/sürümde piksel eşitliği garanti edilmez.',
                                         'Settings are controlled, not guaranteed pixel equality across hardware/extensions/versions.'))
        if samples:
            self.sample_list.setCurrentRow(0)
        else:
            self.preview.clear(); self._sample_pixmap = None

    def _load_report(self):
        path, _ = QFileDialog.getOpenFileName(self, 'comparison.json', str(DEFAULT_OUTPUT), 'JSON (*.json)')
        if path:
            try:
                self._read_report(Path(path))
            except Exception as exc:
                self.error(str(exc))

    def _show_sample(self, index):
        if not 0 <= index < len(self.samples):
            return
        path, sample = self.samples[index]
        self._sample_pixmap = QPixmap(str(path))
        self._resize_preview()
        self.notes.setPlainText(json.dumps({'candidate': sample['candidate'], 'seed': sample['seed'],
                                           'actual': sample.get('actual', {}), 'warnings': sample.get('warnings', [])}, ensure_ascii=False, indent=2))

    def _resize_preview(self):
        if self._sample_pixmap is not None and not self._sample_pixmap.isNull():
            self.preview.setPixmap(self._sample_pixmap.scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event); self._resize_preview()

    def _open_report(self):
        if self.last_report and self.last_report.is_file():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.last_report)))
        else:
            self.status.setText(self.text('Önce bir karşılaştırma üret veya aç.', 'Generate or load a comparison first.'))
