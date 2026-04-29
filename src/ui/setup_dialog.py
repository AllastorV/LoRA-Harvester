"""
First-run setup dialog.

Shown automatically on startup when required models are missing.
Downloads WD14, YOLO and InsightFace into models/ then closes.
"""
from __future__ import annotations

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout,
    QLabel, QProgressBar, QPushButton, QTextEdit,
)
from PyQt5.QtGui import QFont

from src.ui import theme


def models_ready() -> bool:
    """Return True if all required model files are already present."""
    from src.core.model_paths import WD14_DIR, YOLO_DIR
    from huggingface_hub.file_download import repo_folder_name
    import pathlib

    # WD14 — at least one model.onnx somewhere under models/wd14/
    wd14_ok = any(WD14_DIR.rglob("model.onnx"))

    # YOLO — at least one .pt file
    yolo_ok = any(YOLO_DIR.glob("*.pt"))

    return wd14_ok and yolo_ok


class SetupDialog(QDialog):
    """Modal progress dialog that downloads all default models on first run."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("LoRA-Harvester — First Run Setup")
        self.setModal(True)
        self.setMinimumWidth(580)
        self.setStyleSheet(
            f"QDialog {{ background-color: {theme.BG_DARK}; border-radius: 14px; }}"
        )
        self._thread = None
        self._done = False
        self._init_ui()

    def _init_ui(self):
        lay = QVBoxLayout(self)
        lay.setSpacing(16)
        lay.setContentsMargins(28, 24, 28, 24)

        # Title
        title = QLabel("⬇  Downloading Default Models")
        title.setFont(QFont("Inter", 20, QFont.Bold))
        title.setStyleSheet(
            f"color: {theme.TEXT_PRIMARY}; background: transparent; border: none; letter-spacing: -0.015em;"
        )
        lay.addWidget(title)

        sub = QLabel(
            "Required models are being downloaded into the  models/  folder.\n"
            "This only happens once."
        )
        sub.setWordWrap(True)
        sub.setStyleSheet(
            f"color: {theme.TEXT_SECONDARY}; background: transparent; border: none;"
            f"font-size: {theme.fs(11)};"
        )
        lay.addWidget(sub)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet(theme.progress_bar())
        self.progress_bar.setFixedHeight(10)
        lay.addWidget(self.progress_bar)

        # Status label
        self.status_lbl = QLabel("Preparing…")
        self.status_lbl.setStyleSheet(
            f"color: {theme.TEXT_MUTED}; background: transparent; border: none;"
            f"font-size: {theme.fs(10)};"
        )
        lay.addWidget(self.status_lbl)

        # Log
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMinimumHeight(140)
        self.log_box.setStyleSheet(
            f"QTextEdit {{ background-color: {theme.BG_PANEL}; "
            f"color: {theme.TEXT_SECONDARY}; "
            f"border: 1px solid {theme.BORDER}; border-radius: 6px; "
            f"font-family: {theme.FONT_MONO}; font-size: {theme.fs(10)}; "
            f"padding: 6px; }}"
        )
        lay.addWidget(self.log_box)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.skip_btn = QPushButton("Skip (offline / manual install)")
        self.skip_btn.setStyleSheet(
            f"QPushButton {{ background: transparent; border: none; "
            f"color: {theme.TEXT_MUTED}; font-size: {theme.fs(10)}; "
            f"text-decoration: underline; }}"
            f"QPushButton:hover {{ color: {theme.TEXT_SECONDARY}; }}"
        )
        self.skip_btn.clicked.connect(self._skip)
        btn_row.addWidget(self.skip_btn)
        lay.addLayout(btn_row)

    # ── Public entry point ────────────────────────────────────────────────

    def run_install(self):
        """Start download thread and exec the dialog (blocks until done/skipped)."""
        from src.core.model_installer import ModelInstallThread
        self._thread = ModelInstallThread(include_florence2=False, parent=self)
        self._thread.log_message.connect(self._append_log)
        self._thread.progress.connect(self.progress_bar.setValue)
        self._thread.finished_ok.connect(self._on_done)
        self._thread.start()
        self.exec_()

    # ── Slots ─────────────────────────────────────────────────────────────

    def _append_log(self, msg: str):
        self.log_box.append(msg)
        self.status_lbl.setText(msg.strip().lstrip("✓✅⬇📦─ ") or self.status_lbl.text())

    def _on_done(self, ok: bool, summary: str):
        self._done = True
        self.progress_bar.setValue(100)
        self.status_lbl.setText("✅ Setup complete — launching app…" if ok else
                                 "⚠️  Some downloads failed — you can retry later.")
        self.skip_btn.setText("Continue →")
        self.skip_btn.setStyleSheet(
            f"QPushButton {{ background-color: {theme.get_accent()}; "
            f"color: #ffffff; border: none; border-radius: 6px; "
            f"padding: 6px 18px; font-size: {theme.fs(11)}; font-weight: 700; }}"
        )
        # Auto-close after 1.2 s on success
        if ok:
            QTimer.singleShot(1200, self.accept)

    def _skip(self):
        if self._thread and self._thread.isRunning():
            self._thread.quit()
        self.reject()

    def closeEvent(self, event):
        if self._thread and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait(2000)
        event.accept()
