"""
Training Page — LoRA-Harvester v5.

Full training workflow:
  1. Select dataset folder (Kohya format, from KohyaExporter or raw)
  2. Select base model (.safetensors / .ckpt)
  3. Configure network rank, LR, epochs, resolution
  4. Build config TOMLs → start training
  5. Live log + epoch progress + loss display
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QPushButton, QLineEdit, QSpinBox, QDoubleSpinBox,
    QTextEdit, QProgressBar, QCheckBox, QComboBox,
    QFileDialog, QSizePolicy,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont

from src.ui import theme
from src.ui.translations import get_text


# ─────────────────────────────────────────────────────────────────────────────
# Background thread
# ─────────────────────────────────────────────────────────────────────────────

class _TrainThread(QThread):
    log_msg      = pyqtSignal(str)
    finished_sig = pyqtSignal(bool, str)

    def __init__(self, trainer, train_toml: str, parent=None):
        super().__init__(parent)
        self._trainer = trainer
        self._train_toml = train_toml

    def run(self):
        self._trainer.start(
            train_toml=self._train_toml,
            log_callback=lambda msg: self.log_msg.emit(msg),
            finished_callback=lambda ok, s: self.finished_sig.emit(ok, s),
        )

    def stop(self):
        self._trainer.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Page
# ─────────────────────────────────────────────────────────────────────────────

class TrainingPage(QWidget):
    """LoRA Training page — generates Kohya config and launches training."""

    def __init__(self, lang: str = "en", parent=None):
        super().__init__(parent)
        self.lang = lang
        self._trainer = None
        self._thread: Optional[_TrainThread] = None
        self._loss_history: list = []
        self._build_ui()
        QTimer.singleShot(200, self._detect_kohya)

    # ── UI construction ──────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 16, 24, 20)
        root.setSpacing(12)

        # Title
        self._title = QLabel(get_text('training_title', self.lang))
        self._title.setStyleSheet(theme.label_section())
        root.addWidget(self._title)

        self._subtitle = QLabel(get_text('training_subtitle', self.lang))
        self._subtitle.setStyleSheet(theme.label_muted())
        root.addWidget(self._subtitle)

        # Kohya status badge
        self._kohya_badge = QLabel(get_text('training_detecting', self.lang))
        self._kohya_badge.setStyleSheet(
            f"color: {theme.YELLOW}; font-size: {theme.fs(11)}; "
            f"background: transparent; border: none; padding: 4px 0;"
        )
        root.addWidget(self._kohya_badge)

        # ── Two columns ──
        cols = QHBoxLayout()
        cols.setSpacing(12)

        # LEFT: paths & config
        left = QVBoxLayout()
        left.setSpacing(8)

        # Section labels are stored with their translation key so update_language()
        # can re-apply get_text() to each one.
        self._section_lbls: dict = {}

        self._sec1_lbl = self._section_lbl('training_sec_dataset')
        left.addWidget(self._sec1_lbl)

        self._ds_lbl, self._ds_edit = self._path_row(
            left,
            'training_dataset_lbl',
            self._browse_dataset,
        )

        # Kohya structure status + prepare button
        prep_row = QHBoxLayout()
        prep_row.setContentsMargins(164, 0, 0, 0)   # indent under path label
        self._kohya_struct_lbl = QLabel(get_text('training_no_folder', self.lang))
        self._kohya_struct_lbl.setStyleSheet(
            f"color: {theme.TEXT_MUTED}; font-size: {theme.fs(10)}; "
            f"background: transparent; border: none;"
        )
        prep_row.addWidget(self._kohya_struct_lbl)
        prep_row.addStretch()
        self._prepare_btn = QPushButton(get_text('training_prepare_btn', self.lang))
        self._prepare_btn.setToolTip(get_text('training_prepare_tooltip', self.lang))
        self._prepare_btn.setStyleSheet(
            f"QPushButton {{ background: {theme.BG_ELEVATED}; color: {theme.ORANGE}; "
            f"border: 1px solid {theme.ORANGE_DIM}; border-radius: 5px; "
            f"padding: 4px 12px; font-size: {theme.fs(11)}; font-weight: 600; }}"
            f"QPushButton:hover {{ background: {theme.ORANGE_SUBTLE}; }}"
            f"QPushButton:disabled {{ color: {theme.TEXT_MUTED}; border-color: {theme.BORDER}; }}"
        )
        self._prepare_btn.setEnabled(False)
        self._prepare_btn.clicked.connect(self._prepare_kohya_structure)
        prep_row.addWidget(self._prepare_btn)
        left.addLayout(prep_row)

        self._model_lbl, self._model_edit = self._path_row(
            left,
            'training_model_lbl',
            self._browse_model,
        )
        self._kohya_lbl, self._kohya_edit = self._path_row(
            left,
            'training_kohya_lbl',
            self._browse_kohya,
        )
        self._out_lbl, self._out_edit = self._path_row(
            left,
            'training_output_lbl',
            self._browse_output,
        )

        # LoRA name
        name_row = QHBoxLayout()
        self._name_lbl = QLabel(get_text('training_lora_name', self.lang))
        self._name_lbl.setStyleSheet(theme.label_frame())
        self._name_lbl.setMinimumWidth(160)
        self._name_edit = QLineEdit("my_lora")
        self._name_edit.setStyleSheet(theme.line_edit_compact())
        name_row.addWidget(self._name_lbl)
        name_row.addWidget(self._name_edit)
        left.addLayout(name_row)

        left.addWidget(self._section_lbl('training_sec_network'))
        left.addLayout(self._spin_row(
            "Network dim (rank):", "network_dim", 1, 256, 32))
        left.addLayout(self._spin_row(
            "Network alpha:", "network_alpha", 1, 256, 32))

        left.addWidget(self._section_lbl('training_sec_training'))
        left.addLayout(self._spin_row(
            "Max epochs:", "epochs", 1, 100, 10))
        left.addLayout(self._spin_row(
            "Batch size:", "batch_size", 1, 16, 1))
        left.addLayout(self._spin_row(
            "Grad accum:", "grad_accum", 1, 32, 4))
        left.addLayout(self._spin_row(
            "Resolution:", "resolution", 256, 2048, 1024, step=64))
        left.addLayout(self._dspin_row(
            "UNet LR:", "unet_lr", 1e-5, 1e-2, 1e-4))
        left.addLayout(self._dspin_row(
            "TE LR:", "te_lr", 1e-6, 1e-3, 1e-5))
        left.addLayout(self._spin_row(
            "Repeats:", "repeats", 1, 100, 10))

        self._sdxl_cb = QCheckBox(get_text('training_sdxl_mode', self.lang))
        self._sdxl_cb.setStyleSheet(theme.checkbox_frame())
        left.addWidget(self._sdxl_cb)
        left.addStretch()
        cols.addLayout(left, stretch=1)

        # RIGHT: log + progress
        right = QVBoxLayout()
        right.setSpacing(8)
        right.addWidget(self._section_lbl('training_sec_progress'))

        # Step / epoch counters
        info_row = QHBoxLayout()
        self._epoch_caption_lbl, self._epoch_lbl = self._stat_pill(
            info_row, 'training_stat_epoch', "—")
        self._loss_caption_lbl, self._loss_lbl = self._stat_pill(
            info_row, 'training_stat_loss', "—")
        right.addLayout(info_row)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setStyleSheet(theme.progress_bar())
        self._progress.setFixedHeight(8)
        right.addWidget(self._progress)

        # Log
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setStyleSheet(theme.log_area())
        self._log.setPlaceholderText(get_text('training_log_placeholder', self.lang))
        right.addWidget(self._log, stretch=1)
        cols.addLayout(right, stretch=1)

        root.addLayout(cols, stretch=1)

        # ── Bottom: Build + Start + Stop ─────────────────────────────────────
        btn_row = QHBoxLayout()
        self._build_btn = QPushButton(get_text('training_build_btn', self.lang))
        self._build_btn.setToolTip(get_text('training_build_tooltip', self.lang))
        self._build_btn.setStyleSheet(theme.btn_secondary())
        self._build_btn.clicked.connect(self._build_config)
        btn_row.addWidget(self._build_btn)

        self._start_btn = QPushButton(get_text('training_start_btn', self.lang))
        self._start_btn.setStyleSheet(theme.btn_action_start())
        self._start_btn.clicked.connect(self._start_training)
        btn_row.addWidget(self._start_btn, stretch=1)

        self._stop_btn = QPushButton(get_text('training_stop_btn', self.lang))
        self._stop_btn.setStyleSheet(theme.btn_danger())
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._stop_training)
        btn_row.addWidget(self._stop_btn)

        root.addLayout(btn_row)

        # ── Poll timer for live loss/epoch ────────────────────────────────────
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(1000)
        self._poll_timer.timeout.connect(self._poll_trainer)

        # Store spin refs
        self._spin_refs: dict = {}

    # ── Builder helpers ──────────────────────────────────────────────────────

    def _section_lbl(self, key: str) -> QLabel:
        lbl = QLabel(get_text(key, self.lang))
        lbl.setStyleSheet(
            f"color: {theme.ORANGE}; font-size: {theme.fs(11)}; font-weight: 700; "
            f"letter-spacing: 0.06em; margin-top: 6px; background: transparent; border: none;"
        )
        # Remember the key so update_language() can re-translate it.
        self._section_lbls[key] = lbl
        return lbl

    def _path_row(self, layout, key: str, browse_fn):
        row = QHBoxLayout()
        lbl = QLabel(get_text(key, self.lang))
        lbl.setStyleSheet(theme.label_frame())
        lbl.setMinimumWidth(160)
        edit = QLineEdit()
        edit.setStyleSheet(theme.line_edit_compact())
        btn = QPushButton("…")
        btn.setFixedWidth(28)
        btn.setStyleSheet(theme.btn_browse())
        btn.clicked.connect(browse_fn)
        row.addWidget(lbl)
        row.addWidget(edit)
        row.addWidget(btn)
        layout.addLayout(row)
        return lbl, edit

    def _spin_row(self, label: str, key: str, mn: int, mx: int, default: int, step: int = 1):
        row = QHBoxLayout()
        lbl = QLabel(label)
        lbl.setStyleSheet(theme.label_frame())
        lbl.setMinimumWidth(160)
        sp = QSpinBox()
        sp.setRange(mn, mx)
        sp.setValue(default)
        sp.setSingleStep(step)
        sp.setStyleSheet(
            f"QSpinBox {{ background: {theme.BG_ELEVATED}; color: {theme.TEXT_PRIMARY}; "
            f"border: 1px solid {theme.BORDER}; border-radius: 4px; padding: 2px 6px; }}"
        )
        row.addWidget(lbl)
        row.addStretch()
        row.addWidget(sp)
        self._spin_refs[key] = sp
        return row

    def _dspin_row(self, label: str, key: str, mn: float, mx: float, default: float):
        row = QHBoxLayout()
        lbl = QLabel(label)
        lbl.setStyleSheet(theme.label_frame())
        lbl.setMinimumWidth(160)
        sp = QDoubleSpinBox()
        sp.setRange(mn, mx)
        sp.setValue(default)
        sp.setDecimals(6)
        sp.setSingleStep(default / 10)
        sp.setStyleSheet(
            f"QDoubleSpinBox {{ background: {theme.BG_ELEVATED}; color: {theme.TEXT_PRIMARY}; "
            f"border: 1px solid {theme.BORDER}; border-radius: 4px; padding: 2px 6px; }}"
        )
        row.addWidget(lbl)
        row.addStretch()
        row.addWidget(sp)
        self._spin_refs[key] = sp
        return row

    def _stat_pill(self, layout, key: str, value: str):
        frame = QFrame()
        frame.setStyleSheet(
            f"QFrame {{ background: {theme.BG_CARD}; border: 1px solid {theme.BORDER}; "
            f"border-radius: 6px; }}"
        )
        fl = QHBoxLayout(frame)
        fl.setContentsMargins(10, 4, 10, 4)
        fl.setSpacing(6)
        lbl = QLabel(get_text(key, self.lang))
        lbl.setStyleSheet(f"color: {theme.TEXT_MUTED}; font-size: {theme.fs(11)}; "
                          f"background: transparent; border: none;")
        val = QLabel(value)
        val.setStyleSheet(f"color: {theme.TEXT_PRIMARY}; font-size: {theme.fs(13)}; "
                          f"font-weight: 700; background: transparent; border: none;")
        fl.addWidget(lbl)
        fl.addWidget(val)
        layout.addWidget(frame)
        return lbl, val

    # ── Browse callbacks ─────────────────────────────────────────────────────

    def _browse_dataset(self, path: str = ""):
        if not path:
            path = QFileDialog.getExistingDirectory(
                self, get_text('training_dlg_select_dataset', self.lang))
        if not path:
            return
        self._ds_edit.setText(path)
        if not self._out_edit.text():
            self._out_edit.setText(str(Path(path).parent / "lora_output"))
        self._prepare_btn.setEnabled(True)
        self._check_kohya_structure(path)

    def _check_kohya_structure(self, folder: str):
        """Detect whether *folder* is already in Kohya repeats format."""
        import re
        p = Path(folder)
        kohya_pat = re.compile(r"^\d+_.+$")
        subdirs = [d for d in p.iterdir() if d.is_dir()] if p.exists() else []
        already_kohya = any(kohya_pat.match(d.name) for d in subdirs)

        # Count images
        image_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        img_count = sum(1 for pp in p.rglob("*") if pp.suffix.lower() in image_exts)

        if already_kohya:
            self._kohya_struct_lbl.setText(
                f"{get_text('training_kohya_format', self.lang)}  "
                f"({img_count} {get_text('training_images_word', self.lang)})"
            )
            self._kohya_struct_lbl.setStyleSheet(
                f"color: {theme.GREEN}; font-size: {theme.fs(10)}; "
                f"background: transparent; border: none;"
            )
            self._prepare_btn.setText(get_text('training_reprepare', self.lang))
        else:
            self._kohya_struct_lbl.setText(
                f"{get_text('training_flat_folder', self.lang)}  "
                f"({img_count} {get_text('training_images_word', self.lang)})"
            )
            self._kohya_struct_lbl.setStyleSheet(
                f"color: {theme.YELLOW}; font-size: {theme.fs(10)}; "
                f"background: transparent; border: none;"
            )

    def _browse_model(self):
        f, _ = QFileDialog.getOpenFileName(
            self, get_text('training_dlg_select_model', self.lang),
            filter="Model Files (*.safetensors *.ckpt *.pt)"
        )
        if f:
            self._model_edit.setText(f)

    def _browse_kohya(self):
        d = QFileDialog.getExistingDirectory(
            self, get_text('training_dlg_select_kohya', self.lang))
        if d:
            self._kohya_edit.setText(d)
            self._detect_kohya()

    def _browse_output(self):
        d = QFileDialog.getExistingDirectory(
            self, get_text('training_dlg_select_output', self.lang))
        if d:
            self._out_edit.setText(d)

    def _prepare_kohya_structure(self):
        """
        Run KohyaExporter on the selected source folder.
        Destination: <source>_kohya/  (next to source, auto-named)
        After export, updates the dataset path and shows a summary.
        """
        from PyQt5.QtWidgets import QMessageBox
        src = self._ds_edit.text().strip()
        if not src or not Path(src).is_dir():
            QMessageBox.warning(
                self,
                get_text('training_prep_dlg_title', self.lang),
                get_text('training_prep_need_folder', self.lang),
            )
            return

        src_path = Path(src)
        repeats = self._spin_refs.get("repeats")
        rep_val = repeats.value() if repeats else 10
        concept_name = self._name_edit.text().strip() or src_path.name
        dest_path = src_path.parent / f"{src_path.name}_kohya"

        # Confirm
        msg = get_text('training_prep_confirm', self.lang).format(
            source=src_path.name,
            dest=dest_path.name,
            repeats=rep_val,
            concept=concept_name,
        )
        reply = QMessageBox.question(
            self,
            get_text('training_prepare_btn', self.lang),
            msg, QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
        )
        if reply != QMessageBox.Yes:
            return

        self._log.append(
            get_text('training_log_converting', self.lang).format(
                src=src_path.name, dest=dest_path.name)
        )
        self._prepare_btn.setEnabled(False)

        try:
            from src.core.kohya_exporter import KohyaExporter
            exporter = KohyaExporter()
            counts = exporter.export(
                source_root=src_path,
                dest_root=dest_path,
                repeats=rep_val,
                copy=True,
                gen_toml=True,
                resolution=self._spin_refs.get("resolution", {}).value()
                    if "resolution" in self._spin_refs else 1024,
            )

            total = sum(counts.values())
            img_word = get_text('training_images_word', self.lang)
            details = "\n".join(
                f"  {concept}: {n} {img_word}"
                for concept, n in counts.items()
            )
            self._log.append(
                get_text('training_log_done', self.lang).format(
                    total=total, details=details, folder=dest_path)
            )

            # Auto-update dataset path to the prepared kohya folder
            self._ds_edit.setText(str(dest_path))
            if not self._out_edit.text():
                self._out_edit.setText(str(dest_path.parent / "lora_output"))

            # Refresh status badge
            self._check_kohya_structure(str(dest_path))

        except Exception as exc:
            self._log.append(
                get_text('training_log_convert_error', self.lang).format(error=exc)
            )
        finally:
            self._prepare_btn.setEnabled(True)

    # ── Kohya detection ──────────────────────────────────────────────────────

    def _detect_kohya(self):
        from src.training.trainer import KohyaTrainer
        kohya_dir = self._kohya_edit.text().strip() or None
        self._trainer = KohyaTrainer(kohya_dir=kohya_dir)
        info = self._trainer.detection_info()
        if info["script_found"] and info["accelerate_found"]:
            self._kohya_badge.setText(
                f"{get_text('training_kohya_ready', self.lang)}: "
                f"{Path(info['script_path']).parent.name}"
            )
            self._kohya_badge.setStyleSheet(
                f"color: {theme.GREEN}; font-size: {theme.fs(11)}; "
                f"background: transparent; border: none; padding: 4px 0;"
            )
        elif info["script_found"]:
            self._kohya_badge.setText(get_text('training_kohya_partial', self.lang))
            self._kohya_badge.setStyleSheet(
                f"color: {theme.YELLOW}; font-size: {theme.fs(11)}; "
                f"background: transparent; border: none; padding: 4px 0;"
            )
        else:
            self._kohya_badge.setText(get_text('training_kohya_missing', self.lang))
            self._kohya_badge.setStyleSheet(
                f"color: {theme.RED}; font-size: {theme.fs(11)}; "
                f"background: transparent; border: none; padding: 4px 0;"
            )

    # ── Config building ──────────────────────────────────────────────────────

    def _get_config_overrides(self) -> dict:
        sp = self._spin_refs
        return {
            "network_dim":    sp["network_dim"].value(),
            "network_alpha":  sp["network_alpha"].value(),
            "max_train_epochs": sp["epochs"].value(),
            "batch_size":     sp["batch_size"].value(),
            "gradient_accumulation_steps": sp["grad_accum"].value(),
            "resolution":     sp["resolution"].value(),
            "unet_lr":        sp["unet_lr"].value(),
            "text_encoder_lr": sp["te_lr"].value(),
        }

    def _build_config(self) -> Optional[str]:
        """Build TOMLs. Returns path to train_config.toml or None on error."""
        from src.training.config_builder import TrainingConfigBuilder

        ds = self._ds_edit.text().strip()
        model = self._model_edit.text().strip()
        out = self._out_edit.text().strip()

        if not ds:
            self._log.append(get_text('training_log_need_dataset', self.lang))
            return None
        if not model:
            self._log.append(get_text('training_log_need_model', self.lang))
            return None
        if not out:
            self._log.append(get_text('training_log_need_output', self.lang))
            return None

        try:
            builder = TrainingConfigBuilder()
            paths = builder.build(
                dataset_dir=ds,
                output_dir=out,
                base_model=model,
                lora_name=self._name_edit.text().strip() or "my_lora",
                repeats=self._spin_refs["repeats"].value(),
                config_overrides=self._get_config_overrides(),
                sdxl=self._sdxl_cb.isChecked(),
            )
            self._log.append(
                get_text('training_log_config_done', self.lang).format(
                    path=paths['train_toml'])
            )
            return str(paths["train_toml"])
        except Exception as exc:
            self._log.append(
                get_text('training_log_config_error', self.lang).format(error=exc)
            )
            return None

    # ── Training control ─────────────────────────────────────────────────────

    def _start_training(self):
        if not self._trainer:
            self._detect_kohya()
        train_toml = self._build_config()
        if not train_toml:
            return
        if not self._trainer.is_available():
            self._log.append(get_text('training_log_kohya_unavailable', self.lang))
            return

        self._log.clear()
        self._loss_history.clear()
        self._progress.setValue(0)
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)

        self._thread = _TrainThread(self._trainer, train_toml, parent=self)
        self._thread.log_msg.connect(self._on_log, Qt.QueuedConnection)
        self._thread.finished_sig.connect(self._on_finished, Qt.QueuedConnection)
        self._thread.start()
        self._poll_timer.start()

    def _stop_training(self):
        if self._thread:
            self._thread.stop()
        self._poll_timer.stop()

    def _on_log(self, msg: str):
        self._log.append(msg)
        # Auto-scroll
        sb = self._log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_finished(self, success: bool, summary: str):
        self._poll_timer.stop()
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._progress.setValue(100 if success else 0)
        icon = "✅" if success else "❌"
        self._log.append(f"\n{icon} {summary}")

    def _poll_trainer(self):
        if not self._trainer:
            return
        epoch = self._trainer.current_epoch
        loss  = self._trainer.current_loss
        if epoch:
            self._epoch_lbl.setText(str(epoch))
        if loss is not None:
            self._loss_lbl.setText(f"{loss:.4f}")
            self._loss_history.append(loss)

    # ── External hooks ───────────────────────────────────────────────────────

    def set_dataset_path(self, path: str):
        """Called from other pages to pre-fill dataset path."""
        self._browse_dataset(path)

    def update_language(self, lang: str):
        """Re-translate every static, translatable widget on the page."""
        self.lang = lang

        # Header
        self._title.setText(get_text('training_title', lang))
        self._subtitle.setText(get_text('training_subtitle', lang))

        # Section labels (keyed map populated during _build_ui)
        for key, lbl in self._section_lbls.items():
            lbl.setText(get_text(key, lang))

        # Path-row labels
        self._ds_lbl.setText(get_text('training_dataset_lbl', lang))
        self._model_lbl.setText(get_text('training_model_lbl', lang))
        self._kohya_lbl.setText(get_text('training_kohya_lbl', lang))
        self._out_lbl.setText(get_text('training_output_lbl', lang))
        self._name_lbl.setText(get_text('training_lora_name', lang))

        # Prepare button + tooltip. Preserve the "re-prepare" wording when the
        # current dataset is already in Kohya format.
        src = self._ds_edit.text().strip()
        if src and Path(src).is_dir():
            self._check_kohya_structure(src)
        else:
            self._kohya_struct_lbl.setText(get_text('training_no_folder', lang))
            self._prepare_btn.setText(get_text('training_prepare_btn', lang))
        self._prepare_btn.setToolTip(get_text('training_prepare_tooltip', lang))

        # SDXL checkbox
        self._sdxl_cb.setText(get_text('training_sdxl_mode', lang))

        # Stat pill captions
        self._epoch_caption_lbl.setText(get_text('training_stat_epoch', lang))
        self._loss_caption_lbl.setText(get_text('training_stat_loss', lang))

        # Log placeholder (history is left intact)
        self._log.setPlaceholderText(get_text('training_log_placeholder', lang))

        # Bottom buttons + tooltips
        self._build_btn.setText(get_text('training_build_btn', lang))
        self._build_btn.setToolTip(get_text('training_build_tooltip', lang))
        self._start_btn.setText(get_text('training_start_btn', lang))
        self._stop_btn.setText(get_text('training_stop_btn', lang))

        # Kohya status badge — re-run detection so it renders in the new language
        self._detect_kohya()

    def refresh_styles(self):
        pass
