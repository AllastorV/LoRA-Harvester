"""Entry into the single bootstrap wizard; repairs require the application closed."""
from pathlib import Path
import subprocess
import sys
from PyQt5.QtWidgets import QFrame, QVBoxLayout, QLabel, QPushButton
from . import theme


def launch_setup_wizard(focus=None):
    root = Path(__file__).resolve().parents[2]
    base = Path(sys.base_prefix) / ('python.exe' if sys.platform == 'win32' else 'bin/python3')
    exe = str(base if base.is_file() else Path(sys.executable))
    command = [exe, str(root / 'scripts/setup_wizard.py')]
    if focus:
        command.extend(['--focus', focus])
    subprocess.Popen(command, cwd=root, stdin=subprocess.DEVNULL)


class MaintenanceWidget(QFrame):
    def __init__(self, lang='en', parent=None):
        super().__init__(parent)
        self.lang = lang
        layout = QVBoxLayout(self)
        self.label = QLabel(); self.label.setWordWrap(True); layout.addWidget(self.label)
        self.button = QPushButton(); self.button.clicked.connect(self.open_wizard)
        theme.bind_style(self.button, theme.btn_secondary); layout.addWidget(self.button)
        theme.bind_style(self, lambda: f'QFrame {{background:{theme.BG_CARD};border:1px solid {theme.BORDER};border-radius:10px;}}')
        self.update_language(lang)

    def update_language(self, lang):
        self.lang = lang
        self.label.setText('Kurulum ve onarım tek sihirbazda. Kontrol yapılabilir; paket kurmak için önce bu uygulamayı kapat.'
                           if lang == 'tr' else 'One setup/repair wizard. Diagnostics are available now; close this app before changing packages.')
        self.button.setText('Kurulum / onarım sihirbazını aç' if lang == 'tr' else 'Open setup / repair wizard')

    def open_wizard(self):
        from PyQt5.QtWidgets import QMessageBox
        # The runtime lock prevents changing packages until the app closes.
        try:
            launch_setup_wizard()
        except OSError as exc:
            QMessageBox.warning(self, 'Setup', str(exc))
