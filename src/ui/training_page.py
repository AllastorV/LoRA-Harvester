"""
Training Page — LoRA-Harvester v5.

Full training workflow:
  1. Select dataset folder (Kohya format, from KohyaExporter or raw)
  2. Select base model (.safetensors / .ckpt)
  3. Pick a quality preset (Fast / Balanced / Quality / Custom)
  4. Start training — config is built automatically
  5. Live log + epoch progress + loss display
"""

from __future__ import annotations

import os
import math
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
# Preset data
# ─────────────────────────────────────────────────────────────────────────────

# Subject-type training presets — tuned for SDXL / Anime Style LoRA.
# All presets use resolution 1024 and clip_skip 1 (SDXL defaults).
_PRESETS = {
    # Person / character identity — strong identity capture, flexible across poses.
    'character': {
        'network_dim': 32, 'network_alpha': 16, 'epochs': 12,
        'unet_lr': 1e-4, 'te_lr': 2e-5, 'batch_size': 1,
        'grad_accum': 4, 'resolution': 1024, 'repeats': 10,
        'clip_skip': 1, 'noise_offset': 0.04, 'min_snr_gamma': 5.0,
        'network_dropout': 0.0, 'save_every_n': 2,
        'keep_tokens': 1, 'flip_aug': True,
        'lr_scheduler': 'cosine_with_restarts', 'optimizer': 'AdamW8bit', 'mixed_precision': 'fp16',
    },
    # Art style / aesthetic — higher dim for broad capture, low TE LR (style binds
    # to the UNet, not to words), keep_tokens 1 for trigger consistency.
    'style': {
        'network_dim': 64, 'network_alpha': 32, 'epochs': 16,
        'unet_lr': 1e-4, 'te_lr': 1e-5, 'batch_size': 1,
        'grad_accum': 4, 'resolution': 1024, 'repeats': 6,
        'clip_skip': 1, 'noise_offset': 0.05, 'min_snr_gamma': 5.0,
        'network_dropout': 0.0, 'save_every_n': 2,
        'keep_tokens': 1, 'flip_aug': True,
        'lr_scheduler': 'cosine', 'optimizer': 'AdamW8bit', 'mixed_precision': 'fp16',
    },
    # Clothing / object — tight concept binding; flip OFF so logos / asymmetric
    # items are never mirrored.
    'object': {
        'network_dim': 32, 'network_alpha': 16, 'epochs': 10,
        'unet_lr': 1e-4, 'te_lr': 2e-5, 'batch_size': 1,
        'grad_accum': 4, 'resolution': 1024, 'repeats': 12,
        'clip_skip': 1, 'noise_offset': 0.03, 'min_snr_gamma': 5.0,
        'network_dropout': 0.0, 'save_every_n': 2,
        'keep_tokens': 1, 'flip_aug': False,
        'lr_scheduler': 'cosine_with_restarts', 'optimizer': 'AdamW8bit', 'mixed_precision': 'fp16',
    },
}

_LR_SCHEDULERS = [
    'cosine_with_restarts', 'cosine', 'constant',
    'constant_with_warmup', 'linear', 'polynomial',
]
_OPTIMIZERS = [
    'AdamW8bit', 'AdamW', 'Lion8bit', 'Lion',
    'Prodigy', 'DAdaptAdam', 'SGDNesterov',
]
_PRECISIONS = ['fp16', 'bf16', 'no']


# ─────────────────────────────────────────────────────────────────────────────
# Tooltip strings — bilingual, self-contained (no translations.py dependency)
# ─────────────────────────────────────────────────────────────────────────────

_TIPS = {
    'en': {
        # Paths
        'dataset':
            "Folder containing your training images.\n"
            "Can be a flat folder (use Prepare) or already in Kohya format "
            "(numbered subfolders like 10_concept/).",
        'model':
            "Base Stable Diffusion checkpoint (.safetensors or .ckpt) "
            "the LoRA trains on top of.",
        'kohya':
            "Kohya sd-scripts install folder (must contain train_network.py).\n"
            "Click Install to download it automatically.",
        'output':
            "Where the trained LoRA (.safetensors) and config files are saved.",
        'lora_name':
            "Output filename for the LoRA (no extension, no spaces).\n"
            "Example: my_character",
        # Primary controls
        'preset':
            "Choose what you want to train — all settings are tuned for that subject.\n"
            "Character = person/identity · Style = art aesthetic · Object = clothing or item.",
        'preset_character':
            "Trains a person's appearance and identity.\n"
            "Good for OCs, real people, anime characters.\n"
            "clip_skip 2 (anime default — switch to 1 for realistic base models).",
        'preset_style':
            "Trains an art style or visual aesthetic (not a person).\n"
            "High rank captures broad style. Low TE LR so style binds to the UNet, not a trigger word.",
        'preset_object':
            "Trains a specific clothing item, outfit, or object.\n"
            "Flip OFF so asymmetric items / logos are never mirrored.",
        'epochs':
            "How many full passes over the dataset.\n"
            "Too few = underfit (weak effect). Too many = overfit (burns everything into the model).\n"
            "Typical range: 5–20.",
        'model_type':
            "SD 1.5 → use 512–768 px base models.\n"
            "SDXL → use 1024 px base models (resolution is auto-set to 1024).",
        # Network
        'network_dim':
            "LoRA rank — how much the model can learn.\n"
            "Higher = captures more detail but produces a larger file.\n"
            "Typical: 16 (fast) · 32 (balanced) · 64 (quality).",
        'network_alpha':
            "Scaling factor for LoRA weights.\n"
            "Set to dim or dim/2. Higher = stronger effect. Too high = overcooked look.",
        'network_dropout':
            "Randomly disables weights during training to prevent memorisation.\n"
            "0 = disabled. Try 0.1 if the result looks overfit.",
        # Sampling
        'batch_size':
            "Images processed in parallel per training step.\n"
            "Higher = faster but uses more VRAM. Start at 1.",
        'grad_accum':
            "Accumulate gradients over N steps before updating weights.\n"
            "Effective batch = batch_size × grad_accum.",
        'resolution':
            "Pixel size used during training.\n"
            "SD 1.5 → 512 or 768.  SDXL → 1024.",
        'repeats':
            "How many times each image repeats per epoch.\n"
            "Fewer images → increase repeats so training sees enough data.",
        'save_every_n':
            "Save a LoRA checkpoint every N epochs.\n"
            "Useful for comparing intermediate results.",
        'clip_skip':
            "Which CLIP text-encoder layer to use.\n"
            "1 = standard (SDXL, realistic models).\n"
            "2 = recommended for anime-style base models.",
        # Learning rate
        'unet_lr':
            "UNet learning rate — main control for image quality.\n"
            "Too high = burned/noisy output.\n"
            "Typical: 1e-4 (fast) down to 2e-5 (quality).",
        'te_lr':
            "Text encoder learning rate — how strongly words are tied to the concept.\n"
            "Usually ~10× smaller than UNet LR.",
        'lr_scheduler':
            "How the learning rate changes over training.\n"
            "cosine_with_restarts → best for most cases.\n"
            "cosine → smooth decay.\n"
            "constant → no decay (use if training is very short).",
        # Optimizer
        'optimizer':
            "Weight update algorithm.\n"
            "AdamW8bit → low VRAM, great default.\n"
            "Lion → faster convergence, needs lower LR (~3×).\n"
            "Prodigy / DAdaptAdam → automatic LR, set LR=1.",
        'mixed_precision':
            "GPU computation format.\n"
            "fp16 → fastest, works on all CUDA GPUs.\n"
            "bf16 → more stable, requires Ampere (RTX 30xx / 40xx).\n"
            "no → full fp32, slowest.",
        # Noise / stability
        'noise_offset':
            "Helps the LoRA learn very bright or very dark images.\n"
            "0 = disabled.  Typical: 0.05–0.1.",
        'min_snr_gamma':
            "Balances loss weighting across different noise levels — stabilises training.\n"
            "0 = disabled.  5 is the recommended value.",
        # Caption & augmentation
        'keep_tokens':
            "How many caption tokens (words) to keep at the front when shuffling.\n"
            "1 = keep the trigger word first (good for character/object).\n"
            "0 = shuffle everything freely (good for style LoRAs with no fixed trigger).",
        'flip_aug':
            "Randomly mirror training images horizontally for extra variety.\n"
            "OFF for clothing / logos / anything with left-right asymmetry.",
        # Buttons
        'install':
            "Download Kohya sd-scripts from GitHub, create a virtual environment,\n"
            "and install PyTorch (CUDA 12.1) + all requirements automatically.",
        'prepare':
            "Reorganise a flat image folder into Kohya repeats format:\n"
            "creates N_concept/ subfolders and generates dataset_config.toml.",
        'build':
            "Generate .toml config files without starting training.\n"
            "Useful for inspecting or editing the config manually.",
        'start':
            "Build the config (if needed) and start training via Kohya.",
        'stop':
            "Stop the running training process (graceful terminate).",
        'advanced_toggle':
            "Show or hide technical settings: network rank, learning rates,\n"
            "optimizer, noise offset, clip skip, and more.",
    },
    'tr': {
        # Paths
        'dataset':
            "Eğitim görsellerinin bulunduğu klasör.\n"
            "Düz klasör olabilir (Hazırla butonunu kullanın) veya zaten Kohya "
            "formatında (10_konsept/ gibi numaralı alt klasörler) olabilir.",
        'model':
            "LoRA'nın üzerine eğitileceği temel Stable Diffusion checkpoint dosyası "
            "(.safetensors veya .ckpt).",
        'kohya':
            "Kohya sd-scripts kurulum klasörü (içinde train_network.py olmalı).\n"
            "Otomatik indirmek için Install butonuna tıklayın.",
        'output':
            "Eğitilen LoRA (.safetensors) ve config dosyalarının kaydedileceği klasör.",
        'lora_name':
            "Çıktı LoRA dosyasının adı (uzantı ve boşluk olmadan).\n"
            "Örnek: karakter_adim",
        # Primary controls
        'preset':
            "Ne eğitmek istediğinizi seçin — tüm ayarlar o konuya göre ayarlanır.\n"
            "Karakter = kişi/kimlik · Stil = sanat estetiği · Eşya = kıyafet veya nesne.",
        'preset_character':
            "Bir kişinin görünüşünü ve kimliğini eğitir.\n"
            "OC'ler, gerçek kişiler, anime karakterleri için uygundur.\n"
            "clip_skip 2 (anime varsayılanı — gerçekçi modeller için 1 yapın).",
        'preset_style':
            "Bir sanat stili veya görsel estetik öğretir (kişi değil).\n"
            "Yüksek rank geniş stili yakalar. Düşük TE LR, stili kelimeye değil UNet'e bağlar.",
        'preset_object':
            "Belirli bir kıyafet, aksesuar veya nesneyi eğitir.\n"
            "Flip kapalı — asimetrik öğeler / logolar hiçbir zaman ayna görüntüsüne alınmaz.",
        'epochs':
            "Dataset üzerinden yapılan tam geçiş sayısı.\n"
            "Az = yetersiz öğrenme (zayıf etki). Fazla = ezberleme (her şey modele işlenir).\n"
            "Tipik aralık: 5–20.",
        'model_type':
            "SD 1.5 → 512–768 px temel modeller için.\n"
            "SDXL → 1024 px temel modeller için (çözünürlük otomatik 1024 yapılır).",
        # Network
        'network_dim':
            "LoRA sırası — modelin ne kadar şey öğrenebileceği.\n"
            "Yüksek = daha fazla detay ama daha büyük dosya.\n"
            "Tipik: 16 (hızlı) · 32 (dengeli) · 64 (kaliteli).",
        'network_alpha':
            "LoRA ağırlıklarının ölçek çarpanı.\n"
            "Genellikle dim veya dim/2. Yüksek = güçlü etki. Çok yüksek = yanık görüntü.",
        'network_dropout':
            "Eğitim sırasında rastgele ağırlıkları devre dışı bırakarak ezberlemeleri önler.\n"
            "0 = kapalı. Ezberleme varsa 0.1 deneyin.",
        # Sampling
        'batch_size':
            "Her eğitim adımında paralel işlenen görsel sayısı.\n"
            "Yüksek = daha hızlı ama daha fazla VRAM gerekir. 1'den başlayın.",
        'grad_accum':
            "Ağırlıkları güncellemeden önce kaç adım gradyan biriktirilir.\n"
            "Etkili batch = batch_size × grad_accum.",
        'resolution':
            "Eğitimde kullanılan görsel çözünürlüğü.\n"
            "SD 1.5 → 512 veya 768.  SDXL → 1024.",
        'repeats':
            "Her görselin epoch başına kaç kez tekrarlandığı.\n"
            "Az görsel varsa tekrar sayısını artırın — yeterli veri görmesi için.",
        'save_every_n':
            "Her N epoch'ta bir LoRA checkpoint kaydeder.\n"
            "Ara sonuçları karşılaştırmak için kullanışlıdır.",
        'clip_skip':
            "Hangi CLIP metin kodlayıcı katmanında durulacağı.\n"
            "1 = standart (SDXL, gerçekçi modeller).\n"
            "2 = anime tarzı temel modeller için önerilir.",
        # Learning rate
        'unet_lr':
            "UNet öğrenme hızı — görsel kalitesinin ana ayarı.\n"
            "Çok yüksek = yanık/gürültülü çıktı.\n"
            "Tipik: 1e-4 (hızlı) ile 2e-5 (kaliteli) arasında.",
        'te_lr':
            "Metin kodlayıcı öğrenme hızı — kelimelerin konseptle ne kadar bağlandığı.\n"
            "Genellikle UNet LR'nin ~10'da biri kadar.",
        'lr_scheduler':
            "Öğrenme hızının eğitim boyunca nasıl değiştiği.\n"
            "cosine_with_restarts → çoğu durum için en iyi.\n"
            "cosine → düzgün azalma.\n"
            "constant → sabit, azalma yok (çok kısa eğitimler için).",
        # Optimizer
        'optimizer':
            "Ağırlık güncelleme algoritması.\n"
            "AdamW8bit → az VRAM, harika varsayılan.\n"
            "Lion → daha hızlı yakınsama, ~3× daha düşük LR gerekir.\n"
            "Prodigy / DAdaptAdam → otomatik LR, LR=1 yapın.",
        'mixed_precision':
            "GPU hesaplama formatı.\n"
            "fp16 → en hızlı, tüm CUDA GPU'larda çalışır.\n"
            "bf16 → daha kararlı, Ampere (RTX 30xx / 40xx) gerektirir.\n"
            "no → tam fp32 hassasiyet, en yavaş.",
        # Noise / stability
        'noise_offset':
            "LoRA'nın çok parlak veya çok karanlık görselleri öğrenmesine yardımcı olur.\n"
            "0 = kapalı.  Tipik: 0.05–0.1.",
        'min_snr_gamma':
            "Farklı gürültü seviyeleri arasında kayıp ağırlığını dengeleyerek eğitimi stabilleştirir.\n"
            "0 = kapalı.  Önerilen değer: 5.",
        # Caption & augmentation
        'keep_tokens':
            "Karıştırma sırasında başta sabit tutulan caption token (kelime) sayısı.\n"
            "1 = tetikleyici kelimeyi başta tut (karakter/eşya için iyi).\n"
            "0 = hepsini serbestçe karıştır (sabit tetikleyicisi olmayan stil LoRA'ları için).",
        'flip_aug':
            "Ekstra çeşitlilik için eğitim görsellerini yatay olarak rastgele aynalama.\n"
            "Kıyafet / logo / sol-sağ asimetrisi olan her şey için KAPALI bırakın.",
        # Buttons
        'install':
            "Kohya sd-scripts'i GitHub'dan indirir, sanal ortam oluşturur\n"
            "ve PyTorch (CUDA 12.1) + tüm gereksinimleri otomatik yükler.",
        'prepare':
            "Düz görsel klasörünü Kohya tekrar formatına dönüştürür:\n"
            "N_konsept/ alt klasörleri oluşturur ve dataset_config.toml üretir.",
        'build':
            "Eğitimi başlatmadan .toml config dosyalarını oluşturur.\n"
            "Config'i incelemek veya elle düzenlemek istiyorsanız kullanışlıdır.",
        'start':
            "Config'i oluşturur (gerekirse) ve Kohya aracılığıyla eğitimi başlatır.",
        'stop':
            "Çalışan eğitimi durdurur (güvenli sonlandırma gönderir).",
        'advanced_toggle':
            "Network sırası, öğrenme hızları, optimizer, gürültü ofseti, clip skip\n"
            "gibi teknik ayarları göster / gizle.",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# Drag-and-drop QLineEdit
# ─────────────────────────────────────────────────────────────────────────────

class _DropLineEdit(QLineEdit):
    """QLineEdit that accepts file/folder drops and highlights on hover."""
    path_dropped = pyqtSignal(str)

    def __init__(self, accept_dirs: bool = True, accept_exts=None, parent=None):
        super().__init__(parent)
        self._accept_dirs = accept_dirs
        self._accept_exts = set(e.lower() for e in (accept_exts or []))
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                p = Path(url.toLocalFile())
                if (self._accept_dirs and p.is_dir()) or p.suffix.lower() in self._accept_exts:
                    event.accept()
                    self._highlight(True)
                    return
        event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()

    def dragLeaveEvent(self, event):
        self._highlight(False)
        super().dragLeaveEvent(event)

    def dropEvent(self, event):
        self._highlight(False)
        if not event.mimeData().hasUrls():
            event.ignore()
            return
        for url in event.mimeData().urls():
            p = Path(url.toLocalFile())
            if (self._accept_dirs and p.is_dir()) or p.suffix.lower() in self._accept_exts:
                self.setText(str(p))
                self.path_dropped.emit(str(p))
                event.accept()
                return
        event.ignore()

    def _highlight(self, active: bool):
        if active:
            self.setStyleSheet(
                f"QLineEdit {{ background: {theme.BG_SURFACE}; color: {theme.TEXT_PRIMARY}; "
                f"border: 1px solid {theme.ORANGE}; border-radius: {theme.R_SM}; padding: 0 8px; }}"
            )
        else:
            self.setStyleSheet(theme.line_edit_compact())


# ─────────────────────────────────────────────────────────────────────────────
# Background threads
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


class _InstallThread(QThread):
    log_msg      = pyqtSignal(str)
    finished_sig = pyqtSignal(bool, str)

    def __init__(self, dest_dir: Path, parent=None):
        super().__init__(parent)
        self._dest_dir = dest_dir
        self._installer = None

    def run(self):
        from src.training.kohya_installer import KohyaInstaller
        self._installer = KohyaInstaller()
        self._installer.start(
            dest_dir=self._dest_dir,
            log_callback=lambda msg: self.log_msg.emit(msg),
            finished_callback=lambda ok, s: self.finished_sig.emit(ok, s),
        )
        if self._installer._thread:
            self._installer._thread.join()

    def stop(self):
        if self._installer:
            self._installer.stop()


class _RepairDepsThread(QThread):
    """
    Installs kohya_ss training dependencies into its venv.

    Strategy (avoids common failure modes on Windows):
      1. Build a filtered package list from requirements.txt:
         - Skip already-installed packages that are at a HIGHER version
           (prevents downgrading accelerate 1.13 → 1.6, etc.)
         - Skip bitsandbytes (Windows CUDA build often fails; training works without it)
         - Skip editable installs (-e .) — handled separately in step 2
         - Skip commented-out lines
      2. Install filtered list in one pip call with --prefer-binary
      3. Install the kohya_ss package itself as editable (-e .) so
         `from library.train_util import ...` works
    """
    log_msg      = pyqtSignal(str)
    finished_sig = pyqtSignal(bool, str)

    # Packages skipped on Windows — either fail to build or are optional
    _WIN_SKIP = {"bitsandbytes", "albumentations", "pytorch-lightning"}

    def __init__(self, kohya_dir: Path, parent=None):
        super().__init__(parent)
        self._kohya_dir = Path(kohya_dir)

    def run(self):
        import subprocess, sys as _sys
        import importlib.metadata as _meta

        kohya_dir = self._kohya_dir
        req_file  = kohya_dir / "requirements.txt"
        if not req_file.exists():
            self.finished_sig.emit(False, "requirements.txt not found")
            return

        # Locate venv python (preferred) — fall back to sys.executable
        venv_py = None
        for venv_name in ("venv", ".venv"):
            for py_name in ("Scripts/python.exe", "bin/python"):
                p = kohya_dir / venv_name / py_name
                if p.exists():
                    venv_py = str(p); break
            if venv_py:
                break
        if not venv_py:
            venv_py = _sys.executable
            self.log_msg.emit(
                f"  ℹ No kohya venv found — installing into app Python ({Path(venv_py).name})"
            )

        # ── Build installed-version map from the venv ────────────────────────
        installed: dict = {}
        try:
            r = subprocess.run([venv_py, "-m", "pip", "list", "--format=freeze"],
                               capture_output=True, text=True, timeout=30)
            for line in r.stdout.splitlines():
                if "==" in line:
                    name, ver = line.split("==", 1)
                    installed[name.lower().strip()] = ver.strip()
        except Exception:
            pass  # proceed without version map

        # ── Parse requirements.txt → filtered package list ───────────────────
        to_install: list = []
        for raw in req_file.read_text("utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("-e ") or line.startswith("--"):
                continue  # editable / flag lines handled separately
            # Extract bare package name (before ==, >=, <=, [extra], etc.)
            import re
            m = re.match(r"^([A-Za-z0-9_\-]+)", line)
            if not m:
                continue
            pkg_name = m.group(1).lower().replace("_", "-")

            # Skip Windows-problematic packages
            if pkg_name in self._WIN_SKIP:
                self.log_msg.emit(f"  ⚡ skip {pkg_name} (Windows optional)")
                continue

            # Skip if pinned version is LOWER than what's already installed
            pin_m = re.search(r"==([^\s\[,]+)", line)
            if pin_m and pkg_name in installed:
                from packaging.version import Version
                try:
                    pinned  = Version(pin_m.group(1))
                    current = Version(installed[pkg_name])
                    if current >= pinned:
                        self.log_msg.emit(f"  ✓ {pkg_name} {installed[pkg_name]} (keep)")
                        continue
                except Exception:
                    pass

            to_install.append(line)

        if not to_install:
            self.log_msg.emit("  ✓ All packages already satisfied")
        else:
            self.log_msg.emit(f"📦 Installing {len(to_install)} package(s) …")
            cmd = [venv_py, "-m", "pip", "install", "--prefer-binary"] + to_install
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", env={**os.environ},
            )
            for out_line in proc.stdout:
                out_line = out_line.rstrip()
                if out_line:
                    self.log_msg.emit(f"  {out_line}")
            ret = proc.wait()
            if ret != 0:
                self.finished_sig.emit(False, f"pip failed (exit {ret}) — see log above")
                return

        # ── Step 2: install kohya_ss itself as editable package ──────────────
        self.log_msg.emit("📦 Installing kohya library (-e .) …")
        r2 = subprocess.run(
            [venv_py, "-m", "pip", "install", "-e", str(kohya_dir)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            env={**os.environ}, timeout=120,
        )
        if r2.stdout.strip():
            self.log_msg.emit("  " + r2.stdout.strip().replace("\n", "\n  "))
        if r2.returncode != 0:
            err = r2.stderr.strip().splitlines()[-1] if r2.stderr.strip() else "unknown"
            self.log_msg.emit(f"  ⚠ editable install failed: {err} (non-critical)")

        self.finished_sig.emit(True, "Dependencies installed successfully.")


# ─────────────────────────────────────────────────────────────────────────────
# Page
# ─────────────────────────────────────────────────────────────────────────────

class TrainingPage(QWidget):
    """LoRA Training page — preset-driven, beginner-friendly."""

    def __init__(self, lang: str = "en", parent=None):
        super().__init__(parent)
        self.lang = lang
        self._trainer = None
        self._thread: Optional[_TrainThread] = None
        self._install_thread: Optional[_InstallThread] = None
        self._loss_history: list = []
        self._step_estimate: dict = {}
        self._kohya_badge_state = 'detecting'
        self._kohya_struct_state = 'none'
        self._active_preset = 'character'
        self._advanced_visible = False
        self._section_lbls: dict = {}
        self._spin_refs: dict = {}
        self._combo_refs: dict = {}
        self._bool_refs: dict = {}
        # (label_widget, input_widget, tip_key) for path rows
        self._path_tip_map: list = []
        self._build_ui()
        QTimer.singleShot(200, self._detect_kohya)

    # ── Tooltip helpers ──────────────────────────────────────────────────────

    def _tip(self, key: str) -> str:
        lang_tips = _TIPS.get(self.lang, _TIPS['en'])
        return lang_tips.get(key, _TIPS['en'].get(key, ''))

    def _apply_tooltips(self):
        """Re-apply all tooltips from current self.lang (called after language change)."""
        # Spin / double-spin refs
        for key, sp in self._spin_refs.items():
            t = self._tip(key)
            sp.setToolTip(t)

        # Bool refs (checkboxes)
        for key, chk in self._bool_refs.items():
            chk.setToolTip(self._tip(key))

        # Combo refs
        for key, cb in self._combo_refs.items():
            t = self._tip(key)
            cb.setToolTip(t)

        # Path rows (label + edit)
        for lbl, edit, tip_key in self._path_tip_map:
            t = self._tip(tip_key)
            lbl.setToolTip(t)
            edit.setToolTip(t)

        # Named widgets (epochs)
        t_ep = self._tip('epochs')
        self._epochs_sp.setToolTip(t_ep)
        self._epochs_lbl.setToolTip(t_ep)

        # Preset label
        self._preset_lbl_widget.setToolTip(self._tip('preset'))
        # Per-preset buttons — each gets its own subject tip (fallback to generic)
        for k, btn in self._preset_btns.items():
            btn.setToolTip(self._tip(f'preset_{k}') or self._tip('preset'))

        # Model type
        t_mt = self._tip('model_type')
        self._model_type_lbl.setToolTip(t_mt)
        self._sd15_btn.setToolTip(t_mt)
        self._sdxl_btn.setToolTip(t_mt)

        # LoRA name
        t_ln = self._tip('lora_name')
        self._name_lbl.setToolTip(t_ln)
        self._name_edit.setToolTip(t_ln)

        # Buttons
        self._install_kohya_btn.setToolTip(self._tip('install'))
        self._prepare_btn.setToolTip(self._tip('prepare'))
        self._build_btn.setToolTip(self._tip('build'))
        self._start_btn.setToolTip(self._tip('start'))
        self._stop_btn.setToolTip(self._tip('stop'))
        self._adv_toggle.setToolTip(self._tip('advanced_toggle'))

    # ── UI construction ──────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 16, 24, 16)
        root.setSpacing(10)

        # ── Header ────────────────────────────────────────────────────────────
        self._title = QLabel(get_text('training_title', self.lang))
        self._title.setStyleSheet(theme.label_section())
        root.addWidget(self._title)

        self._subtitle = QLabel(get_text('training_subtitle', self.lang))
        self._subtitle.setStyleSheet(theme.label_muted())
        root.addWidget(self._subtitle)

        self._kohya_badge = QLabel(get_text('training_detecting', self.lang))
        self._kohya_badge.setStyleSheet(
            f"color: {theme.YELLOW}; font-size: {theme.fs(11)}; "
            f"background: transparent; border: none; padding: 2px 0 6px 0;"
        )
        root.addWidget(self._kohya_badge)

        # ── Two columns ────────────────────────────────────────────────────────
        cols = QHBoxLayout()
        cols.setSpacing(14)

        # ══ LEFT: Paths + Configuration ══════════════════════════════════════
        left = QVBoxLayout()
        left.setSpacing(10)

        # ─── Paths card ───────────────────────────────────────────────────────
        paths_card = self._make_card()
        pc = QVBoxLayout(paths_card)
        pc.setContentsMargins(14, 12, 14, 12)
        pc.setSpacing(6)

        self._paths_header = QLabel("📁  Paths")
        self._paths_header.setStyleSheet(self._section_hdr_ss())
        pc.addWidget(self._paths_header)

        self._ds_lbl, self._ds_edit = self._path_row(
            pc, 'training_dataset_lbl', self._browse_dataset, tip_key='dataset',
            drop_dirs=True)
        if isinstance(self._ds_edit, _DropLineEdit):
            self._ds_edit.path_dropped.connect(self._browse_dataset)

        # Dataset status + prepare (inline)
        ds_status_row = QHBoxLayout()
        ds_status_row.setContentsMargins(0, 0, 0, 0)
        self._kohya_struct_lbl = QLabel(get_text('training_no_folder', self.lang))
        self._kohya_struct_lbl.setStyleSheet(
            f"color: {theme.TEXT_MUTED}; font-size: {theme.fs(10)}; "
            f"background: transparent; border: none;"
        )
        ds_status_row.addWidget(self._kohya_struct_lbl)
        ds_status_row.addStretch()
        self._prepare_btn = QPushButton(get_text('training_prepare_btn', self.lang))
        self._prepare_btn.setStyleSheet(self._prepare_btn_ss())
        self._prepare_btn.setEnabled(False)
        self._prepare_btn.clicked.connect(self._prepare_kohya_structure)
        ds_status_row.addWidget(self._prepare_btn)
        pc.addLayout(ds_status_row)

        self._model_lbl, self._model_edit = self._path_row(
            pc, 'training_model_lbl', self._browse_model, tip_key='model',
            drop_exts=['.safetensors', '.ckpt', '.pt'])
        self._out_lbl, self._out_edit = self._path_row(
            pc, 'training_output_lbl', self._browse_output, tip_key='output')

        # Kohya path + Install button
        kohya_row_layout = QHBoxLayout()
        self._kohya_lbl = QLabel(get_text('training_kohya_lbl', self.lang))
        self._kohya_lbl.setStyleSheet(theme.label_frame())
        self._kohya_lbl.setMinimumWidth(130)
        self._kohya_edit = QLineEdit()
        self._kohya_edit.setStyleSheet(theme.line_edit_compact())
        kohya_browse = QPushButton("…")
        kohya_browse.setFixedWidth(28)
        kohya_browse.setStyleSheet(theme.btn_browse())
        kohya_browse.clicked.connect(self._browse_kohya)
        self._install_kohya_btn = QPushButton("⬇ Install")
        self._install_kohya_btn.setStyleSheet(self._install_btn_ss())
        self._install_kohya_btn.clicked.connect(self._start_install_kohya)
        self._repair_deps_btn = QPushButton("🔧 Fix deps")
        self._repair_deps_btn.setStyleSheet(self._install_btn_ss())
        self._repair_deps_btn.setVisible(False)
        self._repair_deps_btn.clicked.connect(self._start_repair_deps)
        kohya_row_layout.addWidget(self._kohya_lbl)
        kohya_row_layout.addWidget(self._kohya_edit)
        kohya_row_layout.addWidget(kohya_browse)
        kohya_row_layout.addWidget(self._install_kohya_btn)
        kohya_row_layout.addWidget(self._repair_deps_btn)
        pc.addLayout(kohya_row_layout)
        # register kohya path for tooltip refresh
        self._path_tip_map.append((self._kohya_lbl, self._kohya_edit, 'kohya'))
        # Auto-populate if a kohya_ss folder exists next to the project root
        if not self._kohya_edit.text():
            from src.training.kohya_installer import default_install_dir
            _local = default_install_dir()
            if _local.is_dir() and (_local / "train_network.py").exists():
                self._kohya_edit.setText(str(_local))

        # LoRA name
        sep = QFrame(); sep.setFixedHeight(1)
        sep.setStyleSheet(f"background: {theme.BORDER}; border: none;")
        pc.addWidget(sep)

        name_row = QHBoxLayout()
        self._name_lbl = QLabel(get_text('training_lora_name', self.lang))
        self._name_lbl.setStyleSheet(theme.label_frame())
        self._name_lbl.setMinimumWidth(130)
        self._name_edit = QLineEdit("my_lora")
        self._name_edit.setStyleSheet(theme.line_edit_compact())
        name_row.addWidget(self._name_lbl)
        name_row.addWidget(self._name_edit)
        pc.addLayout(name_row)

        left.addWidget(paths_card)

        # ─── Configuration card ───────────────────────────────────────────────
        cfg_card = self._make_card()
        cc = QVBoxLayout(cfg_card)
        cc.setContentsMargins(14, 12, 14, 12)
        cc.setSpacing(10)

        self._cfg_header = QLabel("⚡  Configuration")
        self._cfg_header.setStyleSheet(self._section_hdr_ss())
        cc.addWidget(self._cfg_header)

        # Preset buttons — subject-type axis
        preset_lbl = QLabel(get_text('training_type_lbl', self.lang))
        preset_lbl.setStyleSheet(theme.label_frame())
        cc.addWidget(preset_lbl)
        self._preset_lbl_widget = preset_lbl

        preset_row = QHBoxLayout()
        preset_row.setSpacing(6)
        self._preset_btns: dict[str, QPushButton] = {}
        for key, label in [('character', '👤 Character'), ('style', '🎨 Style'), ('object', '👕 Object')]:
            btn = QPushButton(label)
            btn.clicked.connect(lambda _=False, k=key: self._apply_preset(k))
            self._preset_btns[key] = btn
            preset_row.addWidget(btn)
        self._custom_btn = QPushButton("🔧 Custom")
        self._custom_btn.clicked.connect(lambda: self._apply_preset('custom'))
        self._preset_btns['custom'] = self._custom_btn
        preset_row.addWidget(self._custom_btn)
        cc.addLayout(preset_row)
        self._update_preset_btn_styles('character')

        # Epochs
        epochs_row = QHBoxLayout()
        self._epochs_lbl = QLabel("Epochs:")
        self._epochs_lbl.setStyleSheet(theme.label_frame())
        self._epochs_lbl.setMinimumWidth(130)
        self._epochs_sp = QSpinBox()
        self._epochs_sp.setRange(1, 200)
        self._epochs_sp.setValue(10)
        self._epochs_sp.setStyleSheet(self._spinbox_ss('QSpinBox'))
        self._epochs_sp.setFixedWidth(80)
        self._epochs_sp.valueChanged.connect(lambda _: self._apply_preset('custom', silent=True))
        epochs_row.addWidget(self._epochs_lbl)
        epochs_row.addStretch()
        epochs_row.addWidget(self._epochs_sp)
        cc.addLayout(epochs_row)
        self._spin_refs['epochs'] = self._epochs_sp

        # Model type toggle
        model_type_row = QHBoxLayout()
        model_type_row.setSpacing(6)
        self._model_type_lbl = QLabel("Model type:")
        self._model_type_lbl.setStyleSheet(theme.label_frame())
        self._model_type_lbl.setMinimumWidth(130)
        self._sd15_btn = QPushButton("SD 1.5")
        self._sdxl_btn = QPushButton("SDXL")
        self._sd15_btn.clicked.connect(lambda: self._set_model_type(False))
        self._sdxl_btn.clicked.connect(lambda: self._set_model_type(True))
        self._sdxl_cb = QCheckBox()
        self._sdxl_cb.hide()
        model_type_row.addWidget(self._model_type_lbl)
        model_type_row.addStretch()
        model_type_row.addWidget(self._sd15_btn)
        model_type_row.addWidget(self._sdxl_btn)
        cc.addLayout(model_type_row)
        self._set_model_type(False)

        # Advanced toggle
        self._adv_toggle = QPushButton("▶  Advanced settings")
        self._adv_toggle.setStyleSheet(self._adv_toggle_ss())
        self._adv_toggle.clicked.connect(self._toggle_advanced)
        cc.addWidget(self._adv_toggle)

        # ── Advanced frame ────────────────────────────────────────────────────
        self._adv_frame = QFrame()
        self._adv_frame.setStyleSheet(
            f"QFrame {{ background: {theme.BG_ELEVATED}; border: 1px solid {theme.BORDER}; "
            f"border-radius: 6px; }}"
        )
        adv = QVBoxLayout(self._adv_frame)
        adv.setContentsMargins(10, 8, 10, 8)
        adv.setSpacing(4)

        adv.addWidget(self._adv_section_lbl("Network"))
        adv.addLayout(self._spin_row("Network dim (rank):", "network_dim", 1, 256, 32))
        adv.addLayout(self._spin_row("Network alpha:", "network_alpha", 1, 256, 16))
        adv.addLayout(self._dspin_row("Dropout:", "network_dropout", 0.0, 0.9, 0.0, dec=2, step=0.05))

        adv.addWidget(self._adv_section_lbl("Sampling"))
        adv.addLayout(self._spin_row("Batch size:", "batch_size", 1, 16, 1))
        adv.addLayout(self._spin_row("Grad accum:", "grad_accum", 1, 32, 4))
        adv.addLayout(self._spin_row("Resolution:", "resolution", 256, 2048, 768, step=64))
        adv.addLayout(self._spin_row("Repeats:", "repeats", 1, 100, 10))
        adv.addLayout(self._spin_row("Save every N epochs:", "save_every_n", 1, 50, 1))
        adv.addLayout(self._spin_row("Clip skip:", "clip_skip", 1, 4, 1))
        adv.addLayout(self._spin_row("Keep tokens:", "keep_tokens", 0, 4, 1))

        adv.addWidget(self._adv_section_lbl("Learning Rate"))
        adv.addLayout(self._dspin_row("UNet LR:", "unet_lr", 1e-6, 1e-2, 5e-5, dec=6, step=1e-5))
        adv.addLayout(self._dspin_row("TE LR:", "te_lr", 1e-7, 1e-3, 5e-6, dec=7, step=1e-6))
        adv.addLayout(self._combo_row("LR Scheduler:", "lr_scheduler", _LR_SCHEDULERS, 0))

        adv.addWidget(self._adv_section_lbl("Optimizer"))
        adv.addLayout(self._combo_row("Optimizer:", "optimizer", _OPTIMIZERS, 0))
        adv.addLayout(self._combo_row("Mixed precision:", "mixed_precision", _PRECISIONS, 0))

        adv.addWidget(self._adv_section_lbl("Noise & Stability"))
        adv.addLayout(self._dspin_row("Noise offset:", "noise_offset", 0.0, 0.5, 0.0, dec=3, step=0.01))
        adv.addLayout(self._dspin_row("Min SNR gamma:", "min_snr_gamma", 0.0, 20.0, 0.0, dec=1, step=1.0))

        adv.addWidget(self._adv_section_lbl("Augmentation"))
        adv.addLayout(self._check_row("Flip augmentation:", "flip_aug", default=False))

        self._adv_frame.setVisible(False)
        cc.addWidget(self._adv_frame)

        left.addWidget(cfg_card)
        left.addStretch()
        cols.addLayout(left, stretch=1)

        # ══ RIGHT: Progress ═══════════════════════════════════════════════════
        right = QVBoxLayout()
        right.setSpacing(8)

        self._progress_header = QLabel("📊  Progress")
        self._progress_header.setStyleSheet(self._section_hdr_ss())
        right.addWidget(self._progress_header)

        info_row = QHBoxLayout()
        self._epoch_caption_lbl, self._epoch_lbl = self._stat_pill(
            info_row, 'training_stat_epoch', "—")
        self._loss_caption_lbl, self._loss_lbl = self._stat_pill(
            info_row, 'training_stat_loss', "—")
        self._eta_caption_lbl, self._eta_lbl = self._stat_pill(
            info_row, 'training_stat_eta', "—")
        right.addLayout(info_row)

        step_info_row = QHBoxLayout()
        self._steps_caption_lbl, self._steps_lbl = self._stat_pill(
            step_info_row, 'training_stat_steps', "—")
        self._steps_per_epoch_caption_lbl, self._steps_per_epoch_lbl = self._stat_pill(
            step_info_row, 'training_stat_steps_per_epoch', "—")
        right.addLayout(step_info_row)

        # Step-level progress bar (finer than epoch-level)
        self._step_progress = QProgressBar()
        self._step_progress.setRange(0, 100)
        self._step_progress.setValue(0)
        self._step_progress.setStyleSheet(theme.progress_bar())
        self._step_progress.setFixedHeight(4)
        right.addWidget(self._step_progress)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setStyleSheet(theme.progress_bar())
        self._progress.setFixedHeight(8)
        right.addWidget(self._progress)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setStyleSheet(theme.log_area())
        self._log.setPlaceholderText(get_text('training_log_placeholder', self.lang))
        right.addWidget(self._log, stretch=1)
        cols.addLayout(right, stretch=1)

        root.addLayout(cols, stretch=1)

        # ── Bottom buttons ─────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        self._build_btn = QPushButton(get_text('training_build_btn', self.lang))
        self._build_btn.setStyleSheet(theme.btn_secondary())
        self._build_btn.setFixedWidth(140)
        self._build_btn.clicked.connect(self._build_config)
        btn_row.addWidget(self._build_btn)

        self._start_btn = QPushButton(get_text('training_start_btn', self.lang))
        self._start_btn.setStyleSheet(theme.btn_action_start())
        self._start_btn.clicked.connect(self._start_training)
        btn_row.addWidget(self._start_btn, stretch=1)

        self._stop_btn = QPushButton(get_text('training_stop_btn', self.lang))
        self._stop_btn.setStyleSheet(theme.btn_danger())
        self._stop_btn.setEnabled(False)
        self._stop_btn.setFixedWidth(90)
        self._stop_btn.clicked.connect(self._stop_training)
        btn_row.addWidget(self._stop_btn)

        root.addLayout(btn_row)

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(1000)
        self._poll_timer.timeout.connect(self._poll_trainer)

        self._apply_preset('character')
        self._apply_tooltips()
        self._connect_step_estimate_signals()
        self._update_step_estimate()

    # ── Style helpers ────────────────────────────────────────────────────────

    def _section_hdr_ss(self) -> str:
        return (
            f"color: {theme.TEXT_MUTED}; font-size: {theme.fs(10)}; font-weight: 700; "
            f"letter-spacing: 0.07em; background: transparent; border: none;"
        )

    def _adv_section_lbl(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(
            f"color: {theme.TEXT_MUTED}; font-size: {theme.fs(10)}; font-weight: 700; "
            f"letter-spacing: 0.05em; background: transparent; border: none; "
            f"margin-top: 6px; padding-left: 0px;"
        )
        return lbl

    def _make_card(self) -> QFrame:
        f = QFrame()
        f.setStyleSheet(
            f"QFrame {{ background: {theme.BG_CARD}; border: 1px solid {theme.BORDER_LIGHT}; "
            f"border-radius: 10px; }}"
        )
        return f

    def _prepare_btn_ss(self) -> str:
        return (
            f"QPushButton {{ background: {theme.BG_ELEVATED}; color: {theme.ORANGE}; "
            f"border: 1px solid {theme.ORANGE_DIM}; border-radius: 5px; "
            f"padding: 3px 10px; font-size: {theme.fs(11)}; font-weight: 600; }}"
            f"QPushButton:hover {{ background: {theme.ORANGE_SUBTLE}; }}"
            f"QPushButton:disabled {{ color: {theme.TEXT_MUTED}; border-color: {theme.BORDER}; }}"
        )

    def _install_btn_ss(self) -> str:
        return (
            f"QPushButton {{ background: {theme.BG_ELEVATED}; color: {theme.GREEN}; "
            f"border: 1px solid {theme.GREEN}; border-radius: 5px; "
            f"padding: 3px 10px; font-size: {theme.fs(11)}; font-weight: 600; }}"
            f"QPushButton:hover {{ background: rgba(80,200,120,0.12); }}"
            f"QPushButton:disabled {{ color: {theme.TEXT_MUTED}; border-color: {theme.BORDER}; }}"
        )

    def _spinbox_ss(self, cls: str) -> str:
        return (
            f"{cls} {{ background: {theme.BG_ELEVATED}; color: {theme.TEXT_PRIMARY}; "
            f"border: 1px solid {theme.BORDER}; border-radius: 4px; padding: 2px 6px; }}"
        )

    def _combo_ss(self) -> str:
        return theme.combobox_compact() if hasattr(theme, 'combobox_compact') else (
            f"QComboBox {{ background: {theme.BG_ELEVATED}; color: {theme.TEXT_PRIMARY}; "
            f"border: 1px solid {theme.BORDER}; border-radius: 4px; padding: 2px 6px; }}"
            f"QComboBox::drop-down {{ border: none; }}"
            f"QComboBox QAbstractItemView {{ background: {theme.BG_ELEVATED}; "
            f"color: {theme.TEXT_PRIMARY}; selection-background-color: {theme.ORANGE}; }}"
        )

    def _adv_toggle_ss(self) -> str:
        return (
            f"QPushButton {{ background: transparent; color: {theme.TEXT_MUTED}; "
            f"border: none; text-align: left; font-size: {theme.fs(11)}; padding: 2px 0; }}"
            f"QPushButton:hover {{ color: {theme.TEXT_PRIMARY}; }}"
        )

    def _preset_btn_active_ss(self) -> str:
        return (
            f"QPushButton {{ background: {theme.ORANGE_SUBTLE}; color: {theme.ORANGE}; "
            f"border: 1px solid {theme.ORANGE_DIM}; border-radius: 6px; "
            f"padding: 5px 10px; font-size: {theme.fs(12)}; font-weight: 600; }}"
        )

    def _preset_btn_idle_ss(self) -> str:
        return (
            f"QPushButton {{ background: {theme.BG_ELEVATED}; color: {theme.TEXT_MUTED}; "
            f"border: 1px solid {theme.BORDER}; border-radius: 6px; "
            f"padding: 5px 10px; font-size: {theme.fs(12)}; }}"
            f"QPushButton:hover {{ color: {theme.TEXT_PRIMARY}; background: {theme.BG_HOVER}; }}"
        )

    def _model_type_btn_active_ss(self) -> str:
        return (
            f"QPushButton {{ background: {theme.BG_SURFACE}; color: {theme.TEXT_PRIMARY}; "
            f"border: 1px solid {theme.BORDER_LIGHT}; border-radius: 6px; "
            f"padding: 4px 14px; font-size: {theme.fs(12)}; font-weight: 600; }}"
        )

    def _model_type_btn_idle_ss(self) -> str:
        return (
            f"QPushButton {{ background: transparent; color: {theme.TEXT_MUTED}; "
            f"border: 1px solid {theme.BORDER}; border-radius: 6px; "
            f"padding: 4px 14px; font-size: {theme.fs(12)}; }}"
            f"QPushButton:hover {{ color: {theme.TEXT_PRIMARY}; }}"
        )

    # ── Builder helpers ──────────────────────────────────────────────────────

    def _path_row(self, layout, key: str, browse_fn, tip_key: str = '',
                  drop_dirs: bool = False, drop_exts=None):
        row = QHBoxLayout()
        lbl = QLabel(get_text(key, self.lang))
        lbl.setStyleSheet(theme.label_frame())
        lbl.setMinimumWidth(130)
        if drop_dirs or drop_exts:
            edit = _DropLineEdit(accept_dirs=drop_dirs, accept_exts=drop_exts)
        else:
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
        if tip_key:
            self._path_tip_map.append((lbl, edit, tip_key))
        return lbl, edit

    def _spin_row(self, label: str, key: str, mn: int, mx: int, default: int, step: int = 1):
        row = QHBoxLayout()
        lbl = QLabel(label)
        lbl.setStyleSheet(theme.label_frame())
        lbl.setMinimumWidth(145)
        sp = QSpinBox()
        sp.setRange(mn, mx)
        sp.setValue(default)
        sp.setSingleStep(step)
        sp.setStyleSheet(self._spinbox_ss('QSpinBox'))
        sp.setFixedWidth(80)
        t = self._tip(key)
        lbl.setToolTip(t)
        sp.setToolTip(t)
        row.addWidget(lbl)
        row.addStretch()
        row.addWidget(sp)
        self._spin_refs[key] = sp
        return row

    def _dspin_row(self, label: str, key: str, mn: float, mx: float, default: float,
                   dec: int = 6, step: float = None):
        row = QHBoxLayout()
        lbl = QLabel(label)
        lbl.setStyleSheet(theme.label_frame())
        lbl.setMinimumWidth(145)
        sp = QDoubleSpinBox()
        sp.setRange(mn, mx)
        sp.setValue(default)
        sp.setDecimals(dec)
        sp.setSingleStep(step if step is not None else default / 10 if default > 0 else 1e-5)
        sp.setStyleSheet(self._spinbox_ss('QDoubleSpinBox'))
        sp.setFixedWidth(110)
        t = self._tip(key)
        lbl.setToolTip(t)
        sp.setToolTip(t)
        row.addWidget(lbl)
        row.addStretch()
        row.addWidget(sp)
        self._spin_refs[key] = sp
        return row

    def _combo_row(self, label: str, key: str, items: list, default_idx: int = 0):
        row = QHBoxLayout()
        lbl = QLabel(label)
        lbl.setStyleSheet(theme.label_frame())
        lbl.setMinimumWidth(145)
        cb = QComboBox()
        cb.addItems(items)
        cb.setCurrentIndex(default_idx)
        cb.setStyleSheet(self._combo_ss())
        cb.setFixedWidth(180)
        cb.currentIndexChanged.connect(lambda _: self._apply_preset('custom', silent=True))
        t = self._tip(key)
        lbl.setToolTip(t)
        cb.setToolTip(t)
        row.addWidget(lbl)
        row.addStretch()
        row.addWidget(cb)
        self._combo_refs[key] = cb
        return row

    def _check_row(self, label: str, key: str, default: bool = False):
        """Build a label + QCheckBox row; store in _bool_refs[key]."""
        row = QHBoxLayout()
        lbl = QLabel(label)
        lbl.setStyleSheet(theme.label_frame())
        lbl.setMinimumWidth(145)
        cb = QCheckBox()
        cb.setChecked(default)
        cb.setStyleSheet(self._checkbox_ss())
        cb.toggled.connect(lambda _: self._apply_preset('custom', silent=True))
        t = self._tip(key)
        lbl.setToolTip(t)
        cb.setToolTip(t)
        row.addWidget(lbl)
        row.addStretch()
        row.addWidget(cb)
        self._bool_refs[key] = cb
        return row

    def _checkbox_ss(self) -> str:
        return (
            f"QCheckBox {{ color: {theme.TEXT_PRIMARY}; background: transparent; "
            f"font-size: {theme.fs(12)}; }}"
            f"QCheckBox::indicator {{ width: 16px; height: 16px; border-radius: 3px; "
            f"border: 1px solid {theme.BORDER}; background: {theme.BG_ELEVATED}; }}"
            f"QCheckBox::indicator:checked {{ background: {theme.ORANGE}; "
            f"border-color: {theme.ORANGE}; }}"
        )

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
        lbl.setStyleSheet(
            f"color: {theme.TEXT_MUTED}; font-size: {theme.fs(11)}; "
            f"background: transparent; border: none;")
        val = QLabel(value)
        val.setStyleSheet(
            f"color: {theme.TEXT_PRIMARY}; font-size: {theme.fs(13)}; "
            f"font-weight: 700; background: transparent; border: none;")
        fl.addWidget(lbl)
        fl.addWidget(val)
        layout.addWidget(frame)
        return lbl, val

    def _connect_step_estimate_signals(self):
        """Refresh estimated steps whenever dataset or training shape changes."""
        for key in ("epochs", "batch_size", "grad_accum", "repeats"):
            widget = self._spin_refs.get(key)
            if widget:
                widget.valueChanged.connect(lambda _=None: self._update_step_estimate())
        self._ds_edit.textChanged.connect(lambda _=None: self._update_step_estimate())

    def _image_count_for_steps(self, dataset_dir: str) -> int:
        p = Path(dataset_dir)
        if not p.is_dir():
            return 0
        image_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        return sum(1 for pp in p.rglob("*")
                   if pp.is_file() and pp.suffix.lower() in image_exts)

    def _calculate_step_estimate(self) -> dict:
        image_count = self._image_count_for_steps(self._ds_edit.text().strip())
        epochs = self._spin_refs.get("epochs").value() if "epochs" in self._spin_refs else 1
        repeats = self._spin_refs.get("repeats").value() if "repeats" in self._spin_refs else 10
        batch_size = self._spin_refs.get("batch_size").value() if "batch_size" in self._spin_refs else 1
        grad_accum = self._spin_refs.get("grad_accum").value() if "grad_accum" in self._spin_refs else 1
        effective_batch = max(1, batch_size * grad_accum)
        steps_per_epoch = max(0, math.ceil(image_count * repeats / effective_batch)) if image_count else 0
        total_steps = steps_per_epoch * epochs
        return {
            "image_count": image_count,
            "epochs": epochs,
            "repeats": repeats,
            "batch_size": batch_size,
            "grad_accum": grad_accum,
            "effective_batch": effective_batch,
            "steps_per_epoch": steps_per_epoch,
            "total_steps": total_steps,
        }

    def _update_step_estimate(self) -> dict:
        est = self._calculate_step_estimate()
        self._step_estimate = est
        if hasattr(self, "_steps_lbl"):
            self._steps_lbl.setText(f"{est['total_steps']:,}" if est["total_steps"] else "—")
        if hasattr(self, "_steps_per_epoch_lbl"):
            self._steps_per_epoch_lbl.setText(
                f"{est['steps_per_epoch']:,}/ep" if est["steps_per_epoch"] else "—"
            )
        return est

    def _log_step_estimate(self, est: dict):
        if not est.get("image_count"):
            self._log.append("Steps: dataset image count is 0; cannot estimate training steps.")
            return
        self._log.append(
            "Steps: "
            f"{est['image_count']} images × {est['repeats']} repeats / "
            f"effective batch {est['effective_batch']} "
            f"(batch {est['batch_size']} × grad accum {est['grad_accum']}) "
            f"= {est['steps_per_epoch']:,} steps/epoch; "
            f"{est['epochs']} epochs = {est['total_steps']:,} total steps."
        )

    # ── Preset logic ─────────────────────────────────────────────────────────

    def _apply_preset(self, key: str, silent: bool = False):
        if key == 'custom':
            self._active_preset = 'custom'
            self._update_preset_btn_styles('custom')
            if not self._advanced_visible:
                self._toggle_advanced()
            return

        data = _PRESETS.get(key, {})
        if not data:
            return
        self._active_preset = key
        self._update_preset_btn_styles(key)

        sp = self._spin_refs
        cb = self._combo_refs

        if 'epochs' in data:
            self._epochs_sp.blockSignals(True)
            self._epochs_sp.setValue(data['epochs'])
            self._epochs_sp.blockSignals(False)

        for k in ('network_dim', 'network_alpha', 'batch_size', 'grad_accum',
                  'resolution', 'repeats', 'clip_skip', 'save_every_n', 'keep_tokens'):
            if k in data and k in sp:
                sp[k].setValue(data[k])

        for k in ('unet_lr', 'te_lr', 'noise_offset', 'min_snr_gamma', 'network_dropout'):
            if k in data and k in sp:
                sp[k].setValue(data[k])

        for k, items in [('lr_scheduler', _LR_SCHEDULERS),
                          ('optimizer', _OPTIMIZERS),
                          ('mixed_precision', _PRECISIONS)]:
            if k in data and k in cb:
                try:
                    idx = items.index(data[k])
                    cb[k].blockSignals(True)
                    cb[k].setCurrentIndex(idx)
                    cb[k].blockSignals(False)
                except ValueError:
                    pass

        # Bool refs (e.g. flip_aug)
        if 'flip_aug' in data and 'flip_aug' in self._bool_refs:
            bw = self._bool_refs['flip_aug']
            bw.blockSignals(True)
            bw.setChecked(data['flip_aug'])
            bw.blockSignals(False)

        if self._sdxl_cb.isChecked() and 'resolution' in sp:
            sp['resolution'].setValue(1024)

    def _update_preset_btn_styles(self, active_key: str):
        active_ss = self._preset_btn_active_ss()
        idle_ss = self._preset_btn_idle_ss()
        for k, btn in self._preset_btns.items():
            btn.setStyleSheet(active_ss if k == active_key else idle_ss)

    def _toggle_advanced(self):
        self._advanced_visible = not self._advanced_visible
        self._adv_frame.setVisible(self._advanced_visible)
        arrow = '▼' if self._advanced_visible else '▶'
        self._adv_toggle.setText(f"{arrow}  Advanced settings")

    def _set_model_type(self, is_sdxl: bool):
        self._sdxl_cb.setChecked(is_sdxl)
        if is_sdxl:
            self._sd15_btn.setStyleSheet(self._model_type_btn_idle_ss())
            self._sdxl_btn.setStyleSheet(self._model_type_btn_active_ss())
            if 'resolution' in self._spin_refs:
                self._spin_refs['resolution'].setValue(1024)
        else:
            self._sd15_btn.setStyleSheet(self._model_type_btn_active_ss())
            self._sdxl_btn.setStyleSheet(self._model_type_btn_idle_ss())
            if self._active_preset in _PRESETS and 'resolution' in self._spin_refs:
                self._spin_refs['resolution'].setValue(
                    _PRESETS[self._active_preset]['resolution'])

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
        import re
        p = Path(folder)
        kohya_pat = re.compile(r"^\d+_.+$")
        subdirs = [d for d in p.iterdir() if d.is_dir()] if p.exists() else []
        already_kohya = any(kohya_pat.match(d.name) for d in subdirs)
        image_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        img_count = sum(1 for pp in p.rglob("*") if pp.suffix.lower() in image_exts)

        if already_kohya:
            self._kohya_struct_lbl.setText(
                f"{get_text('training_kohya_format', self.lang)}  "
                f"({img_count} {get_text('training_images_word', self.lang)})"
            )
            self._kohya_struct_state = 'kohya'
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
            self._kohya_struct_state = 'flat'
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
        from PyQt5.QtWidgets import QMessageBox
        src = self._ds_edit.text().strip()
        if not src or not Path(src).is_dir():
            QMessageBox.warning(self, get_text('training_prep_dlg_title', self.lang),
                                get_text('training_prep_need_folder', self.lang))
            return

        src_path = Path(src)
        repeats = self._spin_refs.get("repeats")
        rep_val = repeats.value() if repeats else 10
        concept_name = self._name_edit.text().strip() or src_path.name
        dest_path = src_path.parent / f"{src_path.name}_kohya"

        msg = get_text('training_prep_confirm', self.lang).format(
            source=src_path.name, dest=dest_path.name,
            repeats=rep_val, concept=concept_name,
        )
        reply = QMessageBox.question(
            self, get_text('training_prepare_btn', self.lang),
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
                source_root=src_path, dest_root=dest_path, repeats=rep_val,
                copy=True, gen_toml=True,
                resolution=self._spin_refs.get("resolution").value()
                    if "resolution" in self._spin_refs else 1024,
            )
            total = sum(counts.values())
            img_word = get_text('training_images_word', self.lang)
            details = "\n".join(f"  {c}: {n} {img_word}" for c, n in counts.items())
            self._log.append(
                get_text('training_log_done', self.lang).format(
                    total=total, details=details, folder=dest_path)
            )
            self._ds_edit.setText(str(dest_path))
            if not self._out_edit.text():
                self._out_edit.setText(str(dest_path.parent / "lora_output"))
            self._check_kohya_structure(str(dest_path))
        except Exception as exc:
            self._log.append(
                get_text('training_log_convert_error', self.lang).format(error=exc))
        finally:
            self._prepare_btn.setEnabled(True)

    # ── Kohya detection ──────────────────────────────────────────────────────

    def _detect_kohya(self):
        from src.training.trainer import KohyaTrainer
        kohya_dir = self._kohya_edit.text().strip() or None
        self._trainer = KohyaTrainer(kohya_dir=kohya_dir)
        info = self._trainer.detection_info()
        _ss = f"font-size: {theme.fs(11)}; background: transparent; border: none; padding: 2px 0 6px 0;"
        missing_deps = False
        missing_names = []
        if info["script_found"]:
            try:
                missing_names = self._trainer.missing_deps()
                missing_deps = bool(missing_names)
            except Exception:
                missing_deps = not info["accelerate_found"]

        if info["script_found"] and info["accelerate_found"]:
            if missing_deps:
                missing_label = ", ".join(missing_names[:3]) if missing_names else "missing deps"
                self._kohya_badge.setText(
                    "⚠ " + get_text('training_kohya_partial', self.lang) +
                    f" — missing {missing_label}"
                )
                self._kohya_badge_state = 'partial'
                self._kohya_badge.setStyleSheet(f"color: {theme.YELLOW}; {_ss}")
            else:
                self._kohya_badge.setText(
                    f"{get_text('training_kohya_ready', self.lang)}: "
                    f"{Path(info['script_path']).parent.name}"
                )
                self._kohya_badge_state = 'ready'
                self._kohya_badge.setStyleSheet(f"color: {theme.GREEN}; {_ss}")
        elif info["script_found"]:
            missing_label = ", ".join(missing_names[:3]) if missing_names else "accelerate"
            self._kohya_badge.setText(
                get_text('training_kohya_partial', self.lang) +
                f" — missing {missing_label}"
            )
            self._kohya_badge_state = 'partial'
            self._kohya_badge.setStyleSheet(f"color: {theme.YELLOW}; {_ss}")
        else:
            self._kohya_badge.setText(get_text('training_kohya_missing', self.lang))
            self._kohya_badge_state = 'missing'
            self._kohya_badge.setStyleSheet(f"color: {theme.RED}; {_ss}")

        # Hide Install button when kohya is detected; show when missing
        self._install_kohya_btn.setVisible(not info["script_found"])
        self._repair_deps_btn.setVisible(missing_deps)

    # ── Kohya installer ──────────────────────────────────────────────────────

    def _start_install_kohya(self):
        from PyQt5.QtWidgets import QMessageBox, QInputDialog
        from src.training.kohya_installer import default_install_dir

        default = str(default_install_dir())
        dest, ok = QInputDialog.getText(
            self, "Install Kohya sd-scripts",
            "Installation directory:\n(~2-4 GB, includes PyTorch CUDA 12.1)",
            text=default,
        )
        if not ok or not dest.strip():
            return

        dest_path = Path(dest.strip())
        reply = QMessageBox.question(
            self, "Install Kohya sd-scripts",
            f"This will:\n"
            f"  1. git clone sd-scripts → {dest_path.name}/\n"
            f"  2. Create virtual environment\n"
            f"  3. Install PyTorch (CUDA 12.1) + requirements\n"
            f"  4. Install accelerate\n\n"
            f"Requires git and ~4 GB disk space. Continue?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
        )
        if reply != QMessageBox.Yes:
            return

        self._log.clear()
        self._log.append(f"📥 Starting Kohya sd-scripts installation → {dest_path}")
        self._install_kohya_btn.setEnabled(False)
        self._install_kohya_btn.setText("⏳ Installing…")

        self._install_thread = _InstallThread(dest_path, parent=self)
        self._install_thread.log_msg.connect(self._on_log, Qt.QueuedConnection)
        self._install_thread.finished_sig.connect(self._on_install_finished, Qt.QueuedConnection)
        self._install_thread.start()

    def _on_install_finished(self, success: bool, result: str):
        self._install_kohya_btn.setEnabled(True)
        self._install_kohya_btn.setText("⬇ Install")
        if success:
            self._kohya_edit.setText(result)
            self._log.append(f"\n✅ Installation complete → {result}")
            self._log.append("Detecting Kohya…")
            self._detect_kohya()
        else:
            self._log.append(f"\n❌ Installation failed: {result}")

    def _start_repair_deps(self):
        from src.training.trainer import KohyaTrainer
        kohya_dir = self._kohya_edit.text().strip()
        if not kohya_dir:
            trainer = KohyaTrainer()
            script = trainer._find_train_script()
            if script:
                kohya_dir = str(script.parent)
            else:
                self._log.append("❌ Cannot find kohya directory to repair deps.")
                return
        self._repair_deps_btn.setEnabled(False)
        self._repair_deps_btn.setText("⏳ Installing…")
        self._log.clear()
        self._log.append(f"🔧 Repairing missing dependencies in {kohya_dir} …")
        self._log.append("This may take several minutes — diffusers, transformers, etc.\n")
        self._repair_thread = _RepairDepsThread(Path(kohya_dir), parent=self)
        self._repair_thread.log_msg.connect(self._on_log, Qt.QueuedConnection)
        self._repair_thread.finished_sig.connect(self._on_repair_finished, Qt.QueuedConnection)
        self._repair_thread.start()

    def _on_repair_finished(self, success: bool, result: str):
        self._repair_deps_btn.setEnabled(True)
        self._repair_deps_btn.setText("🔧 Fix deps")
        if success:
            self._log.append(f"\n✅ {result}")
            self._detect_kohya()
        else:
            self._log.append(f"\n❌ Repair failed: {result}")

    # ── Config building ──────────────────────────────────────────────────────

    def _get_config_overrides(self) -> dict:
        sp = self._spin_refs
        cb = self._combo_refs

        def sv(key, default=None):
            w = sp.get(key)
            return w.value() if w else default

        def cv(key, default=None):
            w = cb.get(key)
            return w.currentText() if w else default

        flip_chk = self._bool_refs.get('flip_aug')
        return {
            "network_dim":                  sv("network_dim", 32),
            "network_alpha":                sv("network_alpha", 16),
            "network_dropout":              sv("network_dropout", 0.0),
            "max_train_epochs":             sv("epochs", 10),
            "batch_size":                   sv("batch_size", 1),
            "gradient_accumulation_steps":  sv("grad_accum", 4),
            "resolution":                   sv("resolution", 768),
            "clip_skip":                    sv("clip_skip", 1),
            "save_every_n_epochs":          sv("save_every_n", 1),
            "unet_lr":                      sv("unet_lr", 5e-5),
            "text_encoder_lr":              sv("te_lr", 5e-6),
            "noise_offset":                 sv("noise_offset", 0.0),
            "min_snr_gamma":                sv("min_snr_gamma", 0.0),
            "lr_scheduler":                 cv("lr_scheduler", "cosine_with_restarts"),
            "optimizer":                    cv("optimizer", "AdamW8bit"),
            "mixed_precision":              cv("mixed_precision", "fp16"),
            "keep_tokens":                  sv("keep_tokens", 1),
            "flip_aug":                     flip_chk.isChecked() if flip_chk else False,
        }

    def _build_config(self) -> Optional[str]:
        from src.training.config_builder import TrainingConfigBuilder
        ds = self._ds_edit.text().strip()
        model = self._model_edit.text().strip()
        out = self._out_edit.text().strip()
        if not ds:
            self._log.append(get_text('training_log_need_dataset', self.lang)); return None
        if not model:
            self._log.append(get_text('training_log_need_model', self.lang)); return None
        if not out:
            self._log.append(get_text('training_log_need_output', self.lang)); return None
        try:
            builder = TrainingConfigBuilder()
            est = self._update_step_estimate()
            paths = builder.build(
                dataset_dir=ds, output_dir=out, base_model=model,
                lora_name=self._name_edit.text().strip() or "my_lora",
                repeats=self._spin_refs["repeats"].value(),
                config_overrides=self._get_config_overrides(),
                sdxl=self._sdxl_cb.isChecked(),
            )
            self._log.append(
                get_text('training_log_config_done', self.lang).format(path=paths['train_toml']))
            self._log_step_estimate(est)
            return str(paths["train_toml"])
        except Exception as exc:
            self._log.append(
                get_text('training_log_config_error', self.lang).format(error=exc))
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
        est = self._update_step_estimate()
        self._log_step_estimate(est)
        self._progress.setValue(0)
        self._step_progress.setValue(0)
        self._epoch_lbl.setText("—")
        self._loss_lbl.setText("—")
        self._eta_lbl.setText("—")
        self._steps_lbl.setText(f"0/{est['total_steps']:,}" if est.get("total_steps") else "—")
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
        sb = self._log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_finished(self, success: bool, summary: str):
        self._poll_timer.stop()
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        if success:
            self._progress.setValue(100)
            self._step_progress.setValue(100)
            self._eta_lbl.setText("Done")
            total_steps = self._trainer.total_steps if self._trainer else 0
            if not total_steps:
                total_steps = self._step_estimate.get("total_steps", 0)
            if total_steps:
                self._steps_lbl.setText(f"{total_steps:,}/{total_steps:,}")
        else:
            self._progress.setValue(0)
            self._step_progress.setValue(0)
            self._eta_lbl.setText("—")
        self._log.append(f"\n{'✅' if success else '❌'} {summary}")

    def _poll_trainer(self):
        if not self._trainer:
            return
        epoch      = self._trainer.current_epoch
        loss       = self._trainer.current_loss
        cur_step   = self._trainer.current_step
        tot_step   = self._trainer.total_steps
        estimate_total = self._step_estimate.get("total_steps", 0)
        if estimate_total and 0 < tot_step < estimate_total:
            tot_step = estimate_total
        eta        = self._trainer.eta_str

        if epoch:
            total_ep = self._spin_refs.get('epochs')
            self._epoch_lbl.setText(
                f"{epoch}/{total_ep.value()}" if total_ep else str(epoch))
            if total_ep:
                pct = int(epoch / total_ep.value() * 100)
                self._progress.setValue(min(pct, 100))

        if loss is not None:
            self._loss_lbl.setText(f"{loss:.4f}")
            self._loss_history.append(loss)

        # Step-level progress bar (updates every tqdm tick)
        if tot_step > 0:
            self._step_progress.setValue(int(cur_step / tot_step * 100))
            self._steps_lbl.setText(f"{cur_step:,}/{tot_step:,}")
        elif estimate_total:
            self._steps_lbl.setText(f"0/{estimate_total:,}")

        # ETA from tqdm remaining field
        self._eta_lbl.setText(eta if eta else "—")

    # ── External hooks ───────────────────────────────────────────────────────

    def set_dataset_path(self, path: str):
        self._browse_dataset(path)

    def update_language(self, lang: str):
        self.lang = lang
        self._title.setText(get_text('training_title', lang))
        self._subtitle.setText(get_text('training_subtitle', lang))
        self._ds_lbl.setText(get_text('training_dataset_lbl', lang))
        self._model_lbl.setText(get_text('training_model_lbl', lang))
        self._kohya_lbl.setText(get_text('training_kohya_lbl', lang))
        self._out_lbl.setText(get_text('training_output_lbl', lang))
        self._name_lbl.setText(get_text('training_lora_name', lang))
        self._preset_lbl_widget.setText(get_text('training_type_lbl', lang))
        src = self._ds_edit.text().strip()
        if src and Path(src).is_dir():
            self._check_kohya_structure(src)
        else:
            self._kohya_struct_lbl.setText(get_text('training_no_folder', lang))
            self._prepare_btn.setText(get_text('training_prepare_btn', lang))
        self._epoch_caption_lbl.setText(get_text('training_stat_epoch', lang))
        self._loss_caption_lbl.setText(get_text('training_stat_loss', lang))
        self._eta_caption_lbl.setText(get_text('training_stat_eta', lang))
        self._steps_caption_lbl.setText(get_text('training_stat_steps', lang))
        self._steps_per_epoch_caption_lbl.setText(
            get_text('training_stat_steps_per_epoch', lang))
        self._log.setPlaceholderText(get_text('training_log_placeholder', lang))
        self._build_btn.setText(get_text('training_build_btn', lang))
        self._start_btn.setText(get_text('training_start_btn', lang))
        self._stop_btn.setText(get_text('training_stop_btn', lang))
        self._detect_kohya()
        self._apply_tooltips()

    def refresh_styles(self):
        from PyQt5.QtWidgets import QDoubleSpinBox as _DSpin

        self._adv_frame.setStyleSheet(
            f"QFrame {{ background: {theme.BG_ELEVATED}; border: 1px solid {theme.BORDER}; "
            f"border-radius: 6px; }}"
        )

        self._title.setStyleSheet(theme.label_section())
        self._subtitle.setStyleSheet(theme.label_muted())
        self._name_lbl.setStyleSheet(theme.label_frame())
        self._name_edit.setStyleSheet(theme.line_edit_compact())
        self._progress.setStyleSheet(theme.progress_bar())
        self._step_progress.setStyleSheet(theme.progress_bar())
        self._log.setStyleSheet(theme.log_area())
        self._build_btn.setStyleSheet(theme.btn_secondary())
        self._start_btn.setStyleSheet(theme.btn_action_start())
        self._stop_btn.setStyleSheet(theme.btn_danger())
        self._prepare_btn.setStyleSheet(self._prepare_btn_ss())
        self._install_kohya_btn.setStyleSheet(self._install_btn_ss())
        self._repair_deps_btn.setStyleSheet(self._install_btn_ss())
        self._adv_toggle.setStyleSheet(self._adv_toggle_ss())

        for attr_lbl, attr_edit in (
            ('_ds_lbl', '_ds_edit'), ('_model_lbl', '_model_edit'),
            ('_kohya_lbl', '_kohya_edit'), ('_out_lbl', '_out_edit'),
        ):
            lbl = getattr(self, attr_lbl, None)
            edit = getattr(self, attr_edit, None)
            if lbl:  lbl.setStyleSheet(theme.label_frame())
            if edit: edit.setStyleSheet(theme.line_edit_compact())

        _spin_ss  = self._spinbox_ss('QSpinBox')
        _dspin_ss = self._spinbox_ss('QDoubleSpinBox')
        _combo_ss = self._combo_ss()
        for sp in self._spin_refs.values():
            sp.setStyleSheet(_dspin_ss if isinstance(sp, _DSpin) else _spin_ss)
        for cb in self._combo_refs.values():
            cb.setStyleSheet(_combo_ss)
        _chk_ss = self._checkbox_ss()
        for chk in self._bool_refs.values():
            chk.setStyleSheet(_chk_ss)

        for btn in self.findChildren(QPushButton):
            if btn.text() == "…":
                btn.setStyleSheet(theme.btn_browse())

        self._update_preset_btn_styles(self._active_preset)
        is_sdxl = self._sdxl_cb.isChecked()
        self._sd15_btn.setStyleSheet(
            self._model_type_btn_idle_ss() if is_sdxl else self._model_type_btn_active_ss())
        self._sdxl_btn.setStyleSheet(
            self._model_type_btn_active_ss() if is_sdxl else self._model_type_btn_idle_ss())

        _badge_color  = {'ready': theme.GREEN, 'partial': theme.YELLOW,
                         'missing': theme.RED, 'detecting': theme.YELLOW}
        _struct_color = {'kohya': theme.GREEN, 'flat': theme.YELLOW, 'none': theme.TEXT_MUTED}
        _badge_ss  = "font-size: %s; background: transparent; border: none; padding: 2px 0 6px 0;" % theme.fs(11)
        _struct_ss = "font-size: %s; background: transparent; border: none;" % theme.fs(10)
        self._kohya_badge.setStyleSheet(
            f"color: {_badge_color.get(self._kohya_badge_state, theme.YELLOW)}; {_badge_ss}")
        self._kohya_struct_lbl.setStyleSheet(
            f"color: {_struct_color.get(self._kohya_struct_state, theme.TEXT_MUTED)}; {_struct_ss}")

        _pill_frame_ss = (
            f"QFrame {{ background: {theme.BG_CARD}; border: 1px solid {theme.BORDER}; "
            f"border-radius: 6px; }}"
        )
        for caption_lbl, val_lbl in (
            (self._epoch_caption_lbl, self._epoch_lbl),
            (self._loss_caption_lbl, self._loss_lbl),
            (self._eta_caption_lbl, self._eta_lbl),
            (self._steps_caption_lbl, self._steps_lbl),
            (self._steps_per_epoch_caption_lbl, self._steps_per_epoch_lbl),
        ):
            frame = caption_lbl.parent()
            if frame:  frame.setStyleSheet(_pill_frame_ss)
            caption_lbl.setStyleSheet(
                f"color: {theme.TEXT_MUTED}; font-size: {theme.fs(11)}; "
                f"background: transparent; border: none;")
            val_lbl.setStyleSheet(
                f"color: {theme.TEXT_PRIMARY}; font-size: {theme.fs(13)}; "
                f"font-weight: 700; background: transparent; border: none;")

        _hdr_ss = self._section_hdr_ss()
        for attr in ('_paths_header', '_cfg_header', '_progress_header'):
            w = getattr(self, attr, None)
            if w:  w.setStyleSheet(_hdr_ss)

        for attr in ('_preset_lbl_widget', '_epochs_lbl', '_model_type_lbl'):
            w = getattr(self, attr, None)
            if w:  w.setStyleSheet(theme.label_frame())
