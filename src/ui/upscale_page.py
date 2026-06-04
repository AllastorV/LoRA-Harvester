"""
Upscale Page for LoRA-Harvester v3.x

Standalone batch image upscaler (Real-ESRGAN). Fully independent from the
video pipeline — point it at a folder (or individual images), pick a model,
hit Run. Useful for upscaling an existing dataset after the fact.

Uses src.core.upscaler.FrameUpscaler (lazy-loaded). If realesrgan/basicsr
are missing, the page reports it clearly instead of silently doing nothing.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QCheckBox, QSpinBox, QComboBox, QTextEdit, QPushButton,
    QProgressBar, QFileDialog,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from src.ui import theme


_IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'}

# Max-resolution cap presets (longest side, px). Mirrors
# advanced_settings.MAX_RES_PRESETS so both screens offer the same choices.
_MAX_RES_PRESETS = [1024, 1080, 2048, 3840, 4096]

# Minimal bilingual strings — kept local so the page is self-contained.
_TX = {
    'en': {
        'title': '🔼 Batch Upscale',
        'subtitle': 'Upscale an existing image folder with Real-ESRGAN.',
        'in_title': '1 · Input',
        'browse_folder': 'Browse Folder…',
        'browse_files': 'Pick Images…',
        'recursive': 'Include subfolders',
        'in_none': 'No input selected',
        'in_count': '{n} image(s) selected',
        'out_title': '2 · Output',
        'browse_out': 'Output Folder…',
        'out_default': 'Default: <input>/upscaled',
        'opt_title': '3 · Model & Options',
        'model': 'Model',
        'face': 'Face enhance (GFPGAN)',
        'gpu': 'Use GPU',
        'tile': 'Tile (0 = off, lower if OOM)',
        'max_res': 'Max resolution (downscale if over)',
        'max_res_off': 'Off',
        'run': '🚀 Run Upscale',
        'stop': '■ Stop',
        'log_ready': 'Ready. Select input and a downloaded model, then Run.',
        'err_no_input': '⚠️ No input images selected.',
        'err_no_model': '⚠️ Selected model has no downloaded weights. Pick one marked available, '
                        'or download it from Settings → Upscale.',
        'err_deps': '❌ Real-ESRGAN deps missing. Install: pip install realesrgan basicsr',
        'done': '✅ Done. {ok}/{total} upscaled → {out}',
        'model_dl': 'Downloaded',
        'model_missing': 'not downloaded',
    },
    'tr': {
        'title': '🔼 Toplu Upscale',
        'subtitle': 'Var olan bir görsel klasörünü Real-ESRGAN ile büyüt.',
        'in_title': '1 · Girdi',
        'browse_folder': 'Klasör Seç…',
        'browse_files': 'Görsel Seç…',
        'recursive': 'Alt klasörleri dahil et',
        'in_none': 'Girdi seçilmedi',
        'in_count': '{n} görsel seçildi',
        'out_title': '2 · Çıktı',
        'browse_out': 'Çıktı Klasörü…',
        'out_default': 'Varsayılan: <girdi>/upscaled',
        'opt_title': '3 · Model ve Ayarlar',
        'model': 'Model',
        'face': 'Yüz iyileştirme (GFPGAN)',
        'gpu': 'GPU kullan',
        'tile': 'Tile (0 = kapalı, OOM olursa düşür)',
        'max_res': 'Maks çözünürlük (aşarsa küçült)',
        'max_res_off': 'Kapalı',
        'run': '🚀 Upscale Başlat',
        'stop': '■ Durdur',
        'log_ready': 'Hazır. Girdi ve inmiş bir model seç, sonra Başlat.',
        'err_no_input': '⚠️ Görsel seçilmedi.',
        'err_no_model': '⚠️ Seçili modelin ağırlıkları inmemiş. Available işaretli birini seç '
                        'veya Ayarlar → Upscale’den indir.',
        'err_deps': '❌ Real-ESRGAN bağımlılıkları eksik. Kur: pip install realesrgan basicsr',
        'done': '✅ Bitti. {ok}/{total} büyütüldü → {out}',
        'model_dl': 'İnmiş',
        'model_missing': 'inmemiş',
    },
}


def _imread_unicode(path: str) -> Optional[np.ndarray]:
    """cv2.imread that survives non-ASCII Windows paths."""
    import cv2
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        # IMREAD_UNCHANGED keeps the alpha channel (transparent PNGs).
        # Dropping it turned transparent areas into a solid green matte.
        return cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    except Exception:
        return None


def _imwrite_unicode(path: str, img: np.ndarray) -> bool:
    """cv2.imwrite that survives non-ASCII Windows paths."""
    import cv2
    ext = os.path.splitext(path)[1] or '.png'
    try:
        ok, buf = cv2.imencode(ext, img)
        if not ok:
            return False
        buf.tofile(path)
        return True
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Worker thread
# ─────────────────────────────────────────────────────────────────────────────

class UpscaleThread(QThread):
    progress = pyqtSignal(int, int, str)   # current, total, filename
    log_msg = pyqtSignal(str)
    upscale_finished = pyqtSignal(dict)    # {ok, total, out_dir, stopped}
    error = pyqtSignal(str)

    def __init__(self, images: List[str], out_dir: str, settings: Dict, parent=None):
        super().__init__(parent)
        self.images = images
        self.out_dir = out_dir
        self.settings = settings
        self._running = True

    def stop(self):
        self._running = False

    def run(self):
        try:
            import gc
            import cv2
            from src.core.upscaler import FrameUpscaler
            from src.ui.resource_settings import load_settings as _load_res
            s = self.settings
            res = _load_res()   # global resource/perf settings from the drawer

            # Honour the global VRAM cap from Resource Settings.
            # torch.cuda.set_per_process_memory_fraction() hard-caps how much
            # VRAM PyTorch can allocate — the OS (and other apps) keep the rest.
            # This is the most reliable way to prevent a full PC freeze on OOM.
            _auto_gc = res.get('auto_gc', True)
            try:
                import torch
                if torch.cuda.is_available():
                    vram_pct = res.get('gpu_mem_limit_pct', 80) / 100.0
                    vram_pct = max(0.1, min(0.95, vram_pct))  # clamp 10–95 %
                    torch.cuda.set_per_process_memory_fraction(vram_pct)
                    self.log_msg.emit(f"  ⚙️ VRAM cap applied: {vram_pct*100:.0f}% of total")
            except Exception as _e:
                self.log_msg.emit(f"  ⚠️ VRAM cap skipped: {_e}")

            up = FrameUpscaler(
                model_name=s.get('model', 'RealESRGAN_x4plus_anime_6B'),
                tile=int(s.get('tile', 0)),
                use_gpu=bool(s.get('use_gpu', True)),
                face_enhance=bool(s.get('face_enhance', False)),
            )
            if not up.is_available():
                self.error.emit(up._load_error or 'deps')
                return

            # Log VRAM before starting (useful for diagnosing OOM crashes)
            try:
                import torch
                if torch.cuda.is_available():
                    free_mb = torch.cuda.mem_get_info()[0] / 1024 / 1024
                    self.log_msg.emit(
                        f"Model ready: {s.get('model')} (scale={up.get_scale()}×) "
                        f"| VRAM free: {free_mb:.0f} MB"
                    )
                else:
                    self.log_msg.emit(
                        f"Model ready: {s.get('model')} (scale={up.get_scale()}×) | CPU mode"
                    )
            except Exception:
                self.log_msg.emit(f"Model ready: {s.get('model')} (scale={up.get_scale()}×)")

            os.makedirs(self.out_dir, exist_ok=True)

            total = len(self.images)
            ok = 0
            # Flush GPU cache every N images to prevent VRAM accumulation
            _FLUSH_EVERY = 10

            for i, src in enumerate(self.images, 1):
                if not self._running:
                    break
                name = os.path.basename(src)
                self.progress.emit(i, total, name)

                img = _imread_unicode(src)
                if img is None:
                    self.log_msg.emit(f"  ⚠️ skip (unreadable): {name}")
                    continue

                out = up.upscale(img)

                # Free input immediately — no longer needed
                del img

                # Cap the longest side if a max resolution is set.
                max_res = int(s.get('max_res', 0) or 0)
                if max_res > 0 and out is not None:
                    h, w = out.shape[:2]
                    longest = max(h, w)
                    if longest > max_res:
                        scale = max_res / float(longest)
                        nw = max(1, int(round(w * scale)))
                        nh = max(1, int(round(h * scale)))
                        out = cv2.resize(out, (nw, nh),
                                         interpolation=cv2.INTER_AREA)

                # Always emit lossless PNG output
                stem = os.path.splitext(name)[0]
                dst = os.path.join(self.out_dir, f"{stem}.png")
                if os.path.abspath(dst) == os.path.abspath(src):
                    dst = os.path.join(self.out_dir, f"{stem}_up.png")
                if _imwrite_unicode(dst, out):
                    ok += 1
                else:
                    self.log_msg.emit(f"  ⚠️ write failed: {name}")

                # Free upscaled frame immediately after writing
                del out

                # Periodic GPU/CPU memory flush (respects auto_gc from Resource Settings)
                if _auto_gc and i % _FLUSH_EVERY == 0:
                    gc.collect()
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            free_mb = torch.cuda.mem_get_info()[0] / 1024 / 1024
                            self.log_msg.emit(
                                f"  🧹 [{i}/{total}] Memory flushed | VRAM free: {free_mb:.0f} MB"
                            )
                    except Exception:
                        pass

            # Final cleanup
            if _auto_gc:
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats()
                except Exception:
                    pass

            self.upscale_finished.emit({
                'ok': ok, 'total': total, 'out_dir': self.out_dir,
                'stopped': not self._running,
            })
        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n{traceback.format_exc()}")


# ─────────────────────────────────────────────────────────────────────────────
# Page
# ─────────────────────────────────────────────────────────────────────────────

class UpscalePage(QWidget):
    def __init__(self, lang: str = 'en', parent=None):
        super().__init__(parent)
        self.lang = lang if lang in _TX else 'en'
        self._images: List[str] = []
        self._out_dir: Optional[str] = None
        self._input_root: Optional[str] = None
        self._thread: Optional[UpscaleThread] = None
        self._build_ui()
        self._refresh_input_label()

    def _t(self, key: str) -> str:
        return _TX.get(self.lang, _TX['en']).get(key, key)

    # ── UI ──────────────────────────────────────────────────────────────────
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(14)

        self._title_lbl = QLabel(self._t('title'))
        self._title_lbl.setStyleSheet(theme.label_section())
        root.addWidget(self._title_lbl)
        self._subtitle_lbl = QLabel(self._t('subtitle'))
        self._subtitle_lbl.setStyleSheet(theme.label_muted())
        root.addWidget(self._subtitle_lbl)

        root.addWidget(self._build_input_card())
        root.addWidget(self._build_output_card())
        root.addWidget(self._build_options_card())

        # Run / Stop
        btn_row = QHBoxLayout()
        self.run_btn = QPushButton(self._t('run'))
        self.run_btn.setCursor(Qt.PointingHandCursor)
        self.run_btn.setStyleSheet(theme.btn_action_start())
        self.run_btn.clicked.connect(self._start)
        self.stop_btn = QPushButton(self._t('stop'))
        self.stop_btn.setCursor(Qt.PointingHandCursor)
        self.stop_btn.setStyleSheet(theme.btn_danger())
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop)
        btn_row.addWidget(self.run_btn, stretch=2)
        btn_row.addWidget(self.stop_btn, stretch=1)
        root.addLayout(btn_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet(theme.progress_bar())
        self.progress_bar.setValue(0)
        root.addWidget(self.progress_bar)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(160)
        self.log_text.setStyleSheet(theme.log_area())
        self.log_text.setText(self._t('log_ready'))
        root.addWidget(self.log_text, stretch=1)

    def _card(self) -> (QFrame, QVBoxLayout):
        f = QFrame()
        f.setStyleSheet(theme.card_frame())
        lay = QVBoxLayout(f)
        lay.setContentsMargins(16, 14, 16, 14)
        lay.setSpacing(10)
        return f, lay

    def _build_input_card(self) -> QFrame:
        f, lay = self._card()
        self._in_title = QLabel(self._t('in_title'))
        self._in_title.setStyleSheet(theme.label_default())
        lay.addWidget(self._in_title)

        row = QHBoxLayout()
        self.browse_folder_btn = QPushButton(self._t('browse_folder'))
        self.browse_folder_btn.setStyleSheet(theme.btn_browse())
        self.browse_folder_btn.clicked.connect(self._pick_folder)
        self.browse_files_btn = QPushButton(self._t('browse_files'))
        self.browse_files_btn.setStyleSheet(theme.btn_browse())
        self.browse_files_btn.clicked.connect(self._pick_files)
        row.addWidget(self.browse_folder_btn)
        row.addWidget(self.browse_files_btn)
        row.addStretch()
        lay.addLayout(row)

        self.recursive_cb = QCheckBox(self._t('recursive'))
        self.recursive_cb.setStyleSheet(theme.checkbox_frame())
        self.recursive_cb.stateChanged.connect(self._rescan_folder)
        lay.addWidget(self.recursive_cb)

        self.input_count_lbl = QLabel(self._t('in_none'))
        self.input_count_lbl.setStyleSheet(theme.label_muted())
        lay.addWidget(self.input_count_lbl)
        return f

    def _build_output_card(self) -> QFrame:
        f, lay = self._card()
        self._out_title = QLabel(self._t('out_title'))
        self._out_title.setStyleSheet(theme.label_default())
        lay.addWidget(self._out_title)

        row = QHBoxLayout()
        self.browse_out_btn = QPushButton(self._t('browse_out'))
        self.browse_out_btn.setStyleSheet(theme.btn_browse())
        self.browse_out_btn.clicked.connect(self._pick_out)
        row.addWidget(self.browse_out_btn)
        row.addStretch()
        lay.addLayout(row)

        self.out_status_lbl = QLabel(self._t('out_default'))
        self.out_status_lbl.setStyleSheet(theme.label_muted())
        lay.addWidget(self.out_status_lbl)
        return f

    def _build_options_card(self) -> QFrame:
        f, lay = self._card()
        self._opt_title = QLabel(self._t('opt_title'))
        self._opt_title.setStyleSheet(theme.label_default())
        lay.addWidget(self._opt_title)

        mrow = QHBoxLayout()
        self.model_lbl = QLabel(self._t('model'))
        self.model_lbl.setStyleSheet(theme.label_frame())
        self.model_combo = QComboBox()
        self.model_combo.setStyleSheet(theme.combo())
        self._populate_models()
        mrow.addWidget(self.model_lbl)
        mrow.addWidget(self.model_combo, stretch=1)
        lay.addLayout(mrow)

        self.gpu_cb = QCheckBox(self._t('gpu'))
        self.gpu_cb.setStyleSheet(theme.checkbox_frame())
        self.gpu_cb.setChecked(True)
        lay.addWidget(self.gpu_cb)

        self.face_cb = QCheckBox(self._t('face'))
        self.face_cb.setStyleSheet(theme.checkbox_frame())
        lay.addWidget(self.face_cb)

        trow = QHBoxLayout()
        self.tile_lbl = QLabel(self._t('tile'))
        self.tile_lbl.setStyleSheet(theme.label_frame())
        self.tile_spin = QSpinBox()
        self.tile_spin.setRange(0, 1024)
        self.tile_spin.setSingleStep(64)
        self.tile_spin.setValue(0)
        self.tile_spin.setStyleSheet(theme.spinbox())
        trow.addWidget(self.tile_lbl)
        trow.addWidget(self.tile_spin)
        trow.addStretch()
        lay.addLayout(trow)

        mrow = QHBoxLayout()
        self.maxres_lbl = QLabel(self._t('max_res'))
        self.maxres_lbl.setStyleSheet(theme.label_frame())
        self.maxres_combo = QComboBox()
        self.maxres_combo.setStyleSheet(theme.combo())
        self.maxres_combo.addItem(self._t('max_res_off'), 0)
        for _p in _MAX_RES_PRESETS:
            self.maxres_combo.addItem(f"{_p} px", _p)
        # Default to 2048 px cap — prevents RAM/VRAM exhaustion on large batches.
        # Users can raise this manually if they need full 4× output.
        _default_idx = _MAX_RES_PRESETS.index(2048) + 1 if 2048 in _MAX_RES_PRESETS else 0
        self.maxres_combo.setCurrentIndex(_default_idx)
        mrow.addWidget(self.maxres_lbl)
        mrow.addWidget(self.maxres_combo)
        mrow.addStretch()
        lay.addLayout(mrow)
        return f

    def _populate_models(self):
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        first_available = -1
        try:
            from src.core.upscale_models import list_models
            for idx, (name, cfg) in enumerate(list_models().items()):
                scale = cfg.get('scale', '?')
                available = cfg.get('available', False)
                tag = self._t('model_dl') if available else self._t('model_missing')
                self.model_combo.addItem(f"{name}  [{scale}×]  · {tag}", name)
                if available and first_available < 0:
                    first_available = idx
        except Exception as e:
            self.model_combo.addItem(f"(error: {e})", "RealESRGAN_x4plus_anime_6B")
        if first_available >= 0:
            self.model_combo.setCurrentIndex(first_available)
        self.model_combo.blockSignals(False)

    # ── input handling ──────────────────────────────────────────────────────
    def _scan_dir(self, root: str, recursive: bool) -> List[str]:
        out = []
        if recursive:
            for dp, _dn, fn in os.walk(root):
                for f in fn:
                    if os.path.splitext(f)[1].lower() in _IMG_EXTS:
                        out.append(os.path.join(dp, f))
        else:
            for f in os.listdir(root):
                p = os.path.join(root, f)
                if os.path.isfile(p) and os.path.splitext(f)[1].lower() in _IMG_EXTS:
                    out.append(p)
        return sorted(out)

    def _pick_folder(self):
        d = QFileDialog.getExistingDirectory(self, self._t('browse_folder'))
        if not d:
            return
        self._input_root = d
        self._images = self._scan_dir(d, self.recursive_cb.isChecked())
        self._refresh_input_label()

    def _rescan_folder(self):
        if self._input_root and os.path.isdir(self._input_root):
            self._images = self._scan_dir(self._input_root, self.recursive_cb.isChecked())
            self._refresh_input_label()

    def _pick_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, self._t('browse_files'), '',
            'Images (*.jpg *.jpeg *.png *.bmp *.webp *.tiff *.tif)')
        if not files:
            return
        self._input_root = os.path.dirname(files[0])
        self._images = list(files)
        self._refresh_input_label()

    def _pick_out(self):
        d = QFileDialog.getExistingDirectory(self, self._t('browse_out'))
        if d:
            self._out_dir = d
            self.out_status_lbl.setText(d)

    def _refresh_input_label(self):
        n = len(self._images)
        if n == 0:
            self.input_count_lbl.setText(self._t('in_none'))
        else:
            self.input_count_lbl.setText(self._t('in_count').format(n=n))

    def _resolve_out_dir(self) -> str:
        if self._out_dir:
            return self._out_dir
        base = self._input_root or os.path.dirname(self._images[0])
        return os.path.join(base, 'upscaled')

    # ── run / stop ──────────────────────────────────────────────────────────
    def _start(self):
        if not self._images:
            self.log_text.append(self._t('err_no_input'))
            return
        if not self.model_combo.currentData():
            self.log_text.append(self._t('err_no_model'))
            return
        out_dir = self._resolve_out_dir()
        settings = {
            'model': self.model_combo.currentData(),
            'use_gpu': self.gpu_cb.isChecked(),
            'face_enhance': self.face_cb.isChecked(),
            'tile': self.tile_spin.value(),
            'max_res': self.maxres_combo.currentData() or 0,
        }
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.log_text.clear()

        self._thread = UpscaleThread(list(self._images), out_dir, settings)
        self._thread.progress.connect(self._on_progress, Qt.QueuedConnection)
        self._thread.log_msg.connect(self._on_log, Qt.QueuedConnection)
        self._thread.upscale_finished.connect(self._on_finished, Qt.QueuedConnection)
        self._thread.error.connect(self._on_error, Qt.QueuedConnection)
        self._thread.start()

    def _stop(self):
        if self._thread is not None:
            try:
                if self._thread.isRunning():
                    self._thread.stop()
            except RuntimeError:
                pass

    def _on_progress(self, cur: int, total: int, name: str):
        pct = int(cur / total * 100) if total else 0
        self.progress_bar.setValue(pct)
        if cur % 5 == 0 or cur == total:
            self.log_text.append(f"  [{cur}/{total}] {name}")

    def _on_log(self, msg: str):
        self.log_text.append(msg)

    def _on_finished(self, stats: dict):
        self.progress_bar.setValue(100)
        self.log_text.append(self._t('done').format(
            ok=stats.get('ok', 0), total=stats.get('total', 0),
            out=stats.get('out_dir', '')))
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._cleanup_thread()

    def _on_error(self, msg: str):
        if 'realesrgan' in msg.lower() or 'basicsr' in msg.lower() or msg == 'deps':
            self.log_text.append(self._t('err_deps'))
        else:
            self.log_text.append(f"❌ {msg}")
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._cleanup_thread()

    def _cleanup_thread(self):
        t = self._thread
        self._thread = None
        if t is not None:
            try:
                if t.isRunning():
                    t.stop()
                    t.wait(3000)
            except RuntimeError:
                pass

    # ── external hooks (match other pages) ──────────────────────────────────
    def update_language(self, lang: str):
        self.lang = lang if lang in _TX else 'en'
        self._title_lbl.setText(self._t('title'))
        self._subtitle_lbl.setText(self._t('subtitle'))
        self._in_title.setText(self._t('in_title'))
        self.browse_folder_btn.setText(self._t('browse_folder'))
        self.browse_files_btn.setText(self._t('browse_files'))
        self.recursive_cb.setText(self._t('recursive'))
        self._out_title.setText(self._t('out_title'))
        self.browse_out_btn.setText(self._t('browse_out'))
        if not self._out_dir:
            self.out_status_lbl.setText(self._t('out_default'))
        self._opt_title.setText(self._t('opt_title'))
        self.model_lbl.setText(self._t('model'))
        self.gpu_cb.setText(self._t('gpu'))
        self.face_cb.setText(self._t('face'))
        self.tile_lbl.setText(self._t('tile'))
        self.maxres_lbl.setText(self._t('max_res'))
        self.maxres_combo.setItemText(0, self._t('max_res_off'))
        self.run_btn.setText(self._t('run'))
        self.stop_btn.setText(self._t('stop'))
        self._refresh_input_label()
        self._populate_models()

    def refresh_styles(self):
        self._title_lbl.setStyleSheet(theme.label_section())
        self._subtitle_lbl.setStyleSheet(theme.label_muted())
        self.run_btn.setStyleSheet(theme.btn_action_start())
        self.stop_btn.setStyleSheet(theme.btn_danger())
        self.progress_bar.setStyleSheet(theme.progress_bar())
        self.log_text.setStyleSheet(theme.log_area())
        for b in (self.browse_folder_btn, self.browse_files_btn, self.browse_out_btn):
            b.setStyleSheet(theme.btn_browse())
        self.model_combo.setStyleSheet(theme.combo())
        self.tile_spin.setStyleSheet(theme.spinbox())
        self.maxres_combo.setStyleSheet(theme.combo())
