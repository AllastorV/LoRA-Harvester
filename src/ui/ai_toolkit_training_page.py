"""Native Qt training frontend for an independently installed Ostris AI Toolkit."""
from __future__ import annotations
from dataclasses import asdict, fields
import json
from pathlib import Path
import threading

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QUrl
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel,
    QLineEdit, QPushButton, QComboBox, QCheckBox, QSpinBox, QDoubleSpinBox,
    QPlainTextEdit, QProgressBar, QFileDialog, QMessageBox, QTabWidget, QScrollArea)

from . import theme
from src.training.ai_toolkit_config import ToolkitOptions, build_bundle, json_bytes
from src.training.ai_toolkit_runner import (ToolkitRunner, check_runtime,
                                           interpreter_candidates, _atomic_json)


class ToolkitWorker(QThread):
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)
    progress = pyqtSignal(int, int, str)
    log = pyqtSignal(str)

    def __init__(self, work, parent=None, runner=None):
        super().__init__(parent)
        self.work = work
        self.cancel = runner.cancel if runner else threading.Event()
        self.runner = runner

    def run(self):
        try:
            self.completed.emit(self.work(self))
        except InterruptedError:
            self.failed.emit('İşlem iptal edildi / Operation cancelled.')
        except Exception as exc:
            self.failed.emit(str(exc))

    def stop(self):
        self.cancel.set()
        if self.runner:
            self.runner.stop()
        self.requestInterruption()


class AIToolkitTrainingPage(QWidget):
    busy_changed = pyqtSignal(bool)

    def __init__(self, lang='en', parent=None):
        super().__init__(parent)
        self.lang = lang
        self._thread = None
        self._bundle = None
        self._bundle_options = None
        self._translations = []
        self._controls = {}
        self._settings_file = Path(__file__).resolve().parents[2] / 'data' / 'ai_toolkit_training.json'
        self.operation_guard = lambda: True
        root = QVBoxLayout(self); root.setContentsMargins(20, 12, 20, 12); root.setSpacing(10)
        title = QLabel('AI Toolkit · LoRA Training')
        theme.bind_style(title, lambda: f'font-size: {theme.fs(20)}; font-weight: 600; color: {theme.TEXT_PRIMARY};')
        root.addWidget(title)
        self.info = QLabel(); self.info.setWordWrap(True)
        self._text(self.info,
            'Kohya’dan bağımsız, step tabanlı eğitim. Aynı caption ve master tagler kullanılır. '
            'Model/ortam ayrı kurulur; bu form başka paketleri yükseltmez.',
            'Step-based training, separate from Kohya. Uses your captions and master tags. '
            'The model/runtime is installed separately; this form does not upgrade packages.')
        root.addWidget(self.info)
        self.tabs = QTabWidget(); root.addWidget(self.tabs)
        basic, basic_form = self._tab('Temel', 'Basic')
        self._path_row(basic_form, 'dataset_dir', 'Dataset klasörü', 'Dataset folder', 'folder')
        self._path_row(basic_form, 'output_dir', 'Çıktı ana klasörü', 'Output parent folder', 'folder')
        self._line(basic_form, 'name', 'Eğitim adı', 'Training name', 'my_lora')
        self._combo(basic_form, 'architecture', 'Model ailesi', 'Model family', [
            ('SDXL / Illustrious', 'sdxl'), ('Stable Diffusion 1.5', 'sd1'), ('FLUX.1-dev', 'flux')])
        self._path_row(basic_form, 'base_model', 'Ana model / Hugging Face ID', 'Base model / Hugging Face ID', 'model')
        self._controls['base_model'].setPlaceholderText('D:\\Models\\model.safetensors  /  owner/repository')
        self._line(basic_form, 'trigger_word', 'Trigger (isteğe bağlı)', 'Trigger (optional)', '')
        self._integer(basic_form, 'steps', 'Toplam optimizer step', 'Total optimizer steps', 1, 1000000, 3000)
        self._integer(basic_form, 'resolution', 'Çözünürlük', 'Resolution', 256, 2048, 1024, 64)
        self._integer(basic_form, 'batch_size', 'Batch boyutu', 'Batch size', 1, 64, 1)
        self._decimal(basic_form, 'learning_rate', 'Öğrenme oranı (LR)', 'Learning rate', 1e-10, 1, .0001, 10)
        self._integer(basic_form, 'rank', 'LoRA rank', 'LoRA rank', 1, 256, 32)
        self._integer(basic_form, 'alpha', 'LoRA alpha', 'LoRA alpha', 1, 256, 16)
        self._integer(basic_form, 'save_every', 'Checkpoint aralığı (step)', 'Checkpoint interval (steps)', 1, 1000000, 250)
        self._integer(basic_form, 'max_saves', 'Saklanacak ara checkpoint', 'Intermediate checkpoints to retain', 1, 1000, 10)
        self._flag(basic_form, 'resume', 'Aynı işe devam et / başarısız işi yeniden dene', 'Resume / retry this tracked run', False)

        advanced, form = self._tab('Gelişmiş', 'Advanced')
        self._combo(form, 'optimizer', 'Optimizer', 'Optimizer', [(s, s) for s in ('adamw8bit','adamw','adafactor')])
        self._combo(form, 'dtype', 'Hesap hassasiyeti', 'Compute precision', [(s,s) for s in ('bf16','fp16','fp32')])
        self._integer(form, 'gradient_accumulation', 'Gradient accumulation', 'Gradient accumulation', 1, 128, 1)
        self._integer(form, 'seed', 'Seed', 'Seed', 0, 2147483647, 42)
        self._integer(form, 'device_index', 'CUDA aygıt numarası', 'CUDA device index', 0, 31, 0)
        self._integer(form, 'num_workers', 'Veri okuma işçisi (Windows: 0)', 'Data loader workers (Windows: 0)', 0, 16, 0)
        self._flag(form, 'gradient_checkpointing', 'Gradient checkpointing', 'Gradient checkpointing', True)
        self._flag(form, 'cache_latents', 'Latent disk önbelleği (görsellerin yanında dosya oluşturabilir)',
                   'Latent disk cache (may create files beside source images)', True)
        self._flag(form, 'recursive', 'Alt klasörleri dahil et', 'Include subfolders', True)
        self._flag(form, 'honor_folder_repeats', 'Kohya N_klasör tekrarlarını koru', 'Honor Kohya N_folder repeats', True)
        self._flag(form, 'require_captions', 'Caption eksik/boşsa engelle', 'Reject missing/empty captions', True)
        self._flag(form, 'shuffle_tokens', 'Caption etiketlerini karıştır', 'Shuffle caption tags', False)
        self._integer(form, 'keep_tokens', 'Baştaki etiketi koru (adet)', 'Preserve leading tags (count)', 0, 64, 1)
        self._decimal(form, 'caption_dropout', 'Caption dropout', 'Caption dropout', 0, .99, 0, 3)
        self._flag(form, 'flip_x', 'Yatay aynalama (asimetrik kıyafet için kapalı tut)',
                   'Horizontal flip (keep off for asymmetric outfits)', False)
        self._flag(form, 'train_text_encoder', 'Text encoder eğit (aynı LR)', 'Train text encoder (shared LR)', False)
        self._decimal(form, 'min_snr_gamma', 'Min-SNR gamma (0: kapalı)', 'Min-SNR gamma (0: off)', 0, 100, 5, 2)
        self._decimal(form, 'noise_offset', 'Noise offset', 'Noise offset', 0, 1, 0, 3)
        self._decimal(form, 'network_dropout', 'Network dropout', 'Network dropout', 0, .9, 0, 3)
        self._flag(form, 'quantize', 'FLUX quantization', 'FLUX quantization', False)

        samples, form = self._tab('Örnek üretimi', 'Samples')
        note = QLabel(); note.setWordWrap(True)
        self._text(note, 'Her satıra bir test promptu. Boşsa örnek üretimi kapalıdır. '
                   'Örnekler ek süre ve VRAM kullanır.',
                   'One sample prompt per line. Empty disables sampling. Samples use extra time and VRAM.')
        form.addRow(note)
        self.prompts = QPlainTextEdit(); self.prompts.setPlaceholderText('master_tag, portrait, ...')
        self.prompts.setMaximumHeight(160); form.addRow(self.prompts)
        self._integer(form, 'sample_every', 'Örnek aralığı (step)', 'Sample interval (steps)', 1, 1000000, 250)
        self._integer(form, 'sample_steps', 'Örnek diffusion step', 'Sample diffusion steps', 1, 100, 20)
        self._decimal(form, 'sample_guidance', 'Örnek CFG / guidance', 'Sample CFG / guidance', 0, 30, 5, 2)

        runtime, form = self._tab('Toolkit bağlantısı', 'Toolkit runtime')
        self._path_row(form, 'toolkit_root', 'AI Toolkit kaynak klasörü (run.py)', 'AI Toolkit checkout (run.py)', 'toolkit')
        self._path_row(form, 'toolkit_python', 'Toolkit venv Python', 'Toolkit venv Python', 'python')
        self._flag(form, 'allow_downloads', 'Eğitim sırasında eksik model dosyalarını indirmeye izin ver',
                   'Allow missing model files to download during training', False)
        notice = QLabel(); notice.setWordWrap(True)
        self._text(notice, 'Kurulum: resmi Ostris AI Toolkit deposunu ayrı klasör/venv ile kur. '
            'Sonra run.py klasörünü ve o kurulumun Python dosyasını seç. '
            'Model lisansı/erişim izni sana aittir. Harvester ortamını seçme.',
            'Install official Ostris AI Toolkit in a separate folder/venv, then select its checkout '
            'and Python executable. Model licenses/access remain your responsibility. Do not select Harvester Python.')
        form.addRow(notice)
        docs = QPushButton(); self._text(docs, 'Resmi kurulum belgesini aç', 'Open official installation docs')
        docs.clicked.connect(lambda: QDesktopServices.openUrl(QUrl('https://github.com/ostris/ai-toolkit#readme')))
        form.addRow(docs)
        self.check_button = QPushButton(); self._text(self.check_button, 'Toolkit ortamını kontrol et', 'Check Toolkit runtime')
        self.check_button.clicked.connect(self._check); form.addRow(self.check_button)
        self.runtime_status = QLabel(); self.runtime_status.setWordWrap(True); form.addRow(self.runtime_status)

        self.preview = QPlainTextEdit(); self.preview.setReadOnly(True); self.preview.setMaximumBlockCount(5000)
        self.preview.setMinimumHeight(160)
        self.tabs.addTab(self.preview, self._t('YAML önizlemesi', 'YAML preview'))
        self._tab_labels.append((self.tabs.count()-1, 'YAML önizlemesi', 'YAML preview'))
        buttons = QHBoxLayout()
        self.build_button = QPushButton(); self._text(self.build_button, 'YAML oluştur', 'Build YAML')
        self.start_button = QPushButton(); self._text(self.start_button, 'Eğitimi başlat', 'Start training')
        self.stop_button = QPushButton(); self._text(self.stop_button, 'Güvenli durdur', 'Stop at step boundary')
        self.force_button = QPushButton(); self._text(self.force_button, 'Zorla durdur…', 'Force stop…')
        self.open_button = QPushButton(); self._text(self.open_button, 'Çıktıyı aç', 'Open output')
        for btn in (self.build_button, self.start_button, self.stop_button, self.force_button, self.open_button):
            buttons.addWidget(btn)
        self.stop_button.setEnabled(False); self.force_button.setEnabled(False)
        self.build_button.clicked.connect(self._build)
        self.start_button.clicked.connect(self._start)
        self.stop_button.clicked.connect(self._stop)
        self.force_button.clicked.connect(self._force)
        self.open_button.clicked.connect(self._open_output)
        root.addLayout(buttons)
        self.progress = QProgressBar(); self.progress.setRange(0,1); self.progress.setValue(0)
        self.status = QLabel(); self.status.setWordWrap(True)
        root.addWidget(self.progress); root.addWidget(self.status)
        self.log = QPlainTextEdit(); self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000); self.log.setMinimumHeight(120); self.log.setMaximumHeight(240)
        root.addWidget(self.log)
        self._load_settings()
        self._controls['architecture'].currentIndexChanged.connect(self._architecture_changed)
        for key, ctrl in self._controls.items():
            signal = (ctrl.textChanged if isinstance(ctrl, QLineEdit) else
                      ctrl.valueChanged if isinstance(ctrl, (QSpinBox, QDoubleSpinBox)) else
                      ctrl.currentIndexChanged if isinstance(ctrl, QComboBox) else ctrl.toggled)
            signal.connect(self._invalidate)
        self.prompts.textChanged.connect(self._invalidate)
        self._apply_family_controls()

    def _t(self, tr, en):
        return tr if self.lang == 'tr' else en

    def _text(self, widget, tr, en):
        self._translations.append((widget, tr, en)); widget.setText(self._t(tr, en))

    def _tab(self, tr, en):
        if not hasattr(self, '_tab_labels'): self._tab_labels = []
        body = QWidget(); form = QFormLayout(body)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form.setRowWrapPolicy(QFormLayout.WrapLongRows); form.setSpacing(9)
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(body)
        index = self.tabs.addTab(scroll, self._t(tr,en)); self._tab_labels.append((index,tr,en))
        return body, form

    def _row(self, form, tr, en, widget):
        label = QLabel(); label.setWordWrap(True); self._text(label, tr, en); form.addRow(label, widget)

    def _line(self, form, key, tr, en, value):
        c = QLineEdit(value); self._controls[key] = c; self._row(form,tr,en,c); return c

    def _path_row(self, form, key, tr, en, kind):
        line = QWidget(); layout = QHBoxLayout(line); layout.setContentsMargins(0,0,0,0)
        c = QLineEdit(); self._controls[key] = c; layout.addWidget(c,1)
        browse = QPushButton('…'); browse.setMaximumWidth(32)
        browse.clicked.connect(lambda checked=False: self._browse(key,kind))
        layout.addWidget(browse)
        if kind == 'model':
            directory = QPushButton('▤'); directory.setMaximumWidth(32)
            directory.setToolTip('Diffusers model folder')
            directory.clicked.connect(lambda checked=False: self._browse(key,'folder'))
            layout.addWidget(directory)
        self._row(form,tr,en,line)

    def _integer(self, form,key,tr,en,low,high,value,step=1):
        c=QSpinBox(); c.setRange(low,high); c.setValue(value); c.setSingleStep(step)
        self._controls[key]=c; self._row(form,tr,en,c)

    def _decimal(self,form,key,tr,en,low,high,value,decimals):
        c=QDoubleSpinBox(); c.setDecimals(decimals); c.setRange(low,high); c.setValue(value)
        c.setSingleStep(.00001 if 'lr' in key or key=='learning_rate' else .01)
        self._controls[key]=c; self._row(form,tr,en,c)

    def _combo(self,form,key,tr,en,items):
        c=QComboBox()
        for label,value in items: c.addItem(label,value)
        self._controls[key]=c; self._row(form,tr,en,c)

    def _flag(self,form,key,tr,en,value):
        c=QCheckBox(); c.setChecked(value); self._text(c,tr,en)
        self._controls[key]=c; form.addRow(c)

    def _value(self,key):
        c=self._controls[key]
        if isinstance(c,QLineEdit): return c.text().strip()
        if isinstance(c,QComboBox): return c.currentData()
        if isinstance(c,QCheckBox): return c.isChecked()
        return c.value()

    def _options(self):
        values={f.name:self._value(f.name) for f in fields(ToolkitOptions) if f.name!='sample_prompts'}
        values['sample_prompts']=tuple(line.strip() for line in self.prompts.toPlainText().splitlines() if line.strip())
        options=ToolkitOptions(**values); options.validate(); return options

    def _browse(self,key,kind):
        if self._thread is not None: return
        current=self._value(key)
        if kind in ('folder','toolkit'):
            path=QFileDialog.getExistingDirectory(self,'AI Toolkit',current)
        else:
            path,_=QFileDialog.getOpenFileName(self,'AI Toolkit',current,
                'Model (*.safetensors)' if kind=='model' else 'Python (python.exe python python3);;All files (*)')
        if not path: return
        self._controls[key].setText(path)
        if key=='dataset_dir' and not self._value('output_dir'):
            self._controls['output_dir'].setText(str(Path(path).parent / 'ai_toolkit_output'))
        if kind=='toolkit':
            candidates=interpreter_candidates(Path(path))
            if candidates: self._controls['toolkit_python'].setText(str(candidates[0]))

    def set_dataset_path(self,path):
        if self._thread is None and path:
            self._controls['dataset_dir'].setText(str(path))
            if not self._value('output_dir'):
                self._controls['output_dir'].setText(str(Path(path).parent / 'ai_toolkit_output'))

    def _architecture_changed(self):
        flux=self._value('architecture')=='flux'
        self._controls['resolution'].setValue(512 if self._value('architecture')=='sd1' else 1024)
        self._controls['min_snr_gamma'].setValue(0 if flux else 5)
        self._controls['noise_offset'].setValue(0)
        self._controls['quantize'].setChecked(flux)
        if flux:
            self._controls['train_text_encoder'].setChecked(False)
            self._controls['dtype'].setCurrentIndex(self._controls['dtype'].findData('bf16'))
        self._apply_family_controls()

    def _apply_family_controls(self):
        flux=self._value('architecture')=='flux'
        for key in ('train_text_encoder','min_snr_gamma','noise_offset'):
            self._controls[key].setEnabled(not flux)
        self._controls['quantize'].setEnabled(flux)
        self._controls['dtype'].setEnabled(not flux)

    def _invalidate(self,*args):
        if self._bundle is not None:
            self._bundle=None; self._bundle_options=None
            self.preview.clear()
            self.status.setText(self._t('Ayar değişti; YAML yeniden oluşturulmalı.', 'Settings changed; rebuild YAML.'))

    def _save_settings(self):
        value={key:self._value(key) for key in self._controls}
        value['sample_prompts']=self.prompts.toPlainText()
        self._settings_file.parent.mkdir(parents=True,exist_ok=True)
        _atomic_json(self._settings_file,value)

    def _load_settings(self):
        if not self._settings_file.exists(): return
        try:
            if self._settings_file.stat().st_size > 128*1024: raise ValueError('Settings too large')
            settings=json.loads(self._settings_file.read_text('utf-8'))
            for key,c in self._controls.items():
                if key not in settings: continue
                value=settings[key]
                if isinstance(c,QLineEdit) and isinstance(value,str): c.setText(value)
                elif isinstance(c,QComboBox):
                    index=c.findData(value)
                    if index>=0: c.setCurrentIndex(index)
                elif isinstance(c,QCheckBox) and type(value) is bool: c.setChecked(value)
                elif isinstance(c,QSpinBox) and type(value) is int: c.setValue(value)
                elif isinstance(c,QDoubleSpinBox) and type(value) in (int,float): c.setValue(value)
            self.prompts.setPlainText(str(settings.get('sample_prompts','')))
        except Exception as exc:
            self.log.appendPlainText('Toolkit settings not loaded: '+str(exc))
        finally:
            # A partially malformed settings file must not restore permissions either.
            self._controls['resume'].setChecked(False)
            self._controls['allow_downloads'].setChecked(False)

    def _error(self,message):
        self.status.setText(message); self.log.appendPlainText(message)
        if not getattr(self.window(),'_close_pending',False): QMessageBox.warning(self,'AI Toolkit',message)

    def _launch(self,work,done,runner=None):
        if self._thread is not None or not self.operation_guard(): return False
        try: self._save_settings()
        except Exception as exc: self._error(str(exc)); return False
        worker=ToolkitWorker(work,self,runner); self._thread=worker
        worker.completed.connect(done,Qt.QueuedConnection)
        worker.failed.connect(self._error,Qt.QueuedConnection)
        worker.progress.connect(self._progress,Qt.QueuedConnection)
        worker.log.connect(self.log.appendPlainText,Qt.QueuedConnection)
        worker.finished.connect(self._finished,Qt.QueuedConnection)
        self.tabs.setEnabled(False); self.build_button.setEnabled(False); self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True); self.force_button.setEnabled(bool(runner))
        self.progress.setRange(0,0); self.busy_changed.emit(True)
        worker.start(); return True

    def _finished(self):
        worker=self._thread; self._thread=None
        if worker: worker.deleteLater()
        self.tabs.setEnabled(True); self.build_button.setEnabled(True); self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False); self.force_button.setEnabled(False)
        if self.progress.maximum()==0: self.progress.setRange(0,1); self.progress.setValue(0)
        self.busy_changed.emit(False)

    def _progress(self,n,total,message):
        self.progress.setRange(0,max(1,total)); self.progress.setValue(min(n,max(1,total)))
        self.status.setText(message)

    def _build(self):
        try: options=self._options()
        except Exception as exc: self._error(str(exc)); return
        def work(w): return build_bundle(options,cancel=w.cancel,progress=w.progress.emit)
        self._launch(work,lambda bundle:self._built(bundle,options))

    def _built(self,bundle,options):
        self._bundle=bundle; self._bundle_options=options
        self.preview.setPlainText(bundle.yaml_text); self.tabs.setCurrentWidget(self.preview)
        self.status.setText(self._t(f'{bundle.image_count} görsel · tekrarlarla {bundle.weighted_count} · YAML hazır',
                                   f'{bundle.image_count} images · {bundle.weighted_count} with repeats · YAML ready'))
        self.log.appendPlainText(str(bundle.config_path))
        for warning in bundle.warnings: self.log.appendPlainText(warning)

    def _check(self):
        root,python=self._value('toolkit_root'),self._value('toolkit_python')
        def work(w): return check_runtime(root,python,cancel=w.cancel)
        def done(report):
            text=f'Python {report["python"]} · '+', '.join(report['devices'])
            self.runtime_status.setText(text); self.status.setText(text)
            self.log.appendPlainText(json.dumps(report,ensure_ascii=False,indent=2))
        self._launch(work,done)

    def _start(self):
        if self._bundle is None:
            self._error(self._t('Önce YAML oluştur ve önizlemeyi kontrol et.', 'Build YAML and inspect it first.')); return
        try:
            options=self._options()
            if options!=self._bundle_options: raise ValueError('YAML ayarları değişti / rebuild YAML.')
        except Exception as exc: self._error(str(exc)); return
        text=self._t(
            f'AI Toolkit ile {options.steps} step eğitim başlayacak.\nÇıktı: {self._bundle.output_path}\n'
            f'Model indirme izni: {options.allow_downloads}\n'
            'Dataset ve caption eğitim sürerken değiştirilmemeli. Latent önbelleği yan dosya oluşturabilir. '
            'Checkpoint saklama sınırı eski ara kayıtları kaldırabilir. Devam edilsin mi?',
            f'Start {options.steps} steps in AI Toolkit?\nOutput: {self._bundle.output_path}\n'
            f'Model downloads allowed: {options.allow_downloads}\n'
            'Do not modify images/captions during training. Latent caching may create side files. '
            'The retention limit may remove older intermediate checkpoints. Continue?')
        if QMessageBox.question(self,'AI Toolkit',text,QMessageBox.Yes|QMessageBox.No,QMessageBox.No)!=QMessageBox.Yes: return
        root,python=self._value('toolkit_root'),self._value('toolkit_python')
        manifest=self._bundle.manifest_path; resume=self._value('resume'); runner=ToolkitRunner()
        def work(w):
            return runner.run(root,python,manifest,resume=resume,log=w.log.emit,
                progress=lambda e:w.progress.emit(e['step'],e['total'],f"Step {e['step']} / {e['total']}"))
        def done(result):
            self.status.setText(result.message); self.log.appendPlainText(result.message+'\n'+result.log_path)
            if result.status=='completed': self.progress.setRange(0,100); self.progress.setValue(100)
        self._launch(work,done,runner)

    def _stop(self):
        if self._thread:
            self._thread.stop()
            self.stop_button.setEnabled(False)
            self.status.setText(self._t('Durdurma istendi; model hazırlığı veya mevcut adım/kayıt tamamlanması bekleniyor.',
                                        'Stop requested; waiting for initialization or the current step/save to finish.'))

    def _force(self):
        if not self._thread or not self._thread.runner: return
        text=self._t('Zorla sonlandırma son checkpointi bozabilir. Yalnız işlem yanıt vermiyorsa kullan. Sonlandır?',
                     'Force termination may corrupt the latest checkpoint. Use only for an unresponsive process. Terminate?')
        if QMessageBox.question(self,'AI Toolkit',text,QMessageBox.Yes|QMessageBox.No,QMessageBox.No)==QMessageBox.Yes:
            self._thread.runner.force_stop()

    def _open_output(self):
        path=self._bundle.output_path if self._bundle else Path(self._value('output_dir'))
        if path.is_dir(): QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def update_language(self,lang):
        self.lang=lang
        for widget,tr,en in self._translations: widget.setText(self._t(tr,en))
        for index,tr,en in self._tab_labels: self.tabs.setTabText(index,self._t(tr,en))

    def refresh_styles(self):
        return theme.refresh_styles(self)
