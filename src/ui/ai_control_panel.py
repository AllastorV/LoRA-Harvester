"""Visible, session-only control/approval panel for local MCP clients."""
from __future__ import annotations
import json
import sys
from pathlib import Path
from PyQt5.QtCore import Qt, QTimer, QObject
from PyQt5.QtWidgets import (QAction, QApplication, QCheckBox, QComboBox, QDockWidget,
    QFileDialog, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMessageBox,
    QPlainTextEdit, QPushButton, QVBoxLayout, QWidget)
from src.core.ai_control import Broker, ControlServer, ControlError, CATALOG, VERSION
from .ai_live_actions import LiveActions, DEFERRED
from . import theme

ROOT = Path(__file__).resolve().parents[2]


class AIControlPanel(QDockWidget):
    def __init__(self, window, app):
        super().__init__('AI bağlantısı · Codex / Claude', window)
        self.setObjectName('ai_control_dock')
        self.setMinimumWidth(340)
        self.window_ref = window
        self.broker = Broker(ROOT / 'logs' / 'ai-control')
        self.live = LiveActions(window, self.broker)
        self.server = None
        self.roots = []
        self.sequence = -1
        self._dispatching = False
        body = QWidget(); layout = QVBoxLayout(body)
        intro = QLabel('Ajan, açık uygulamayı adlandırılmış araçlarla kontrol eder. '
            'İzinler sadece bu oturum içindir; her açılışta kontrol kapalıdır. '
            'Caption/metin yanıtları bağlı ajana iletilir.'); intro.setWordWrap(True)
        self.intro = intro; layout.addWidget(intro)
        self.enabled = QCheckBox('AI bağlantısını aç'); layout.addWidget(self.enabled)
        self.mode = QComboBox()
        self.mode.addItem('Yalnız izle — değişiklik yok', 'observe')
        self.mode.addItem('Onaylı kontrol — yazmadan/işten önce sor', 'review')
        self.mode.addItem('Yetki devri — izinli klasörlerde otomatik', 'delegated')
        self.mode.setCurrentIndex(1); layout.addWidget(self.mode)
        self.share = QCheckBox('Görsel önizlemelerini ajana paylaş')
        self.share.setToolTip('Kapalıyken görsel bytes gönderilmez. Açılırsa bağlı Codex/Claude görseli kendi model sağlayıcısına iletebilir.')
        layout.addWidget(self.share)
        self.roots_heading = QLabel('İzinli girdi / çıktı klasörleri'); layout.addWidget(self.roots_heading)
        self.root_list = QListWidget(); self.root_list.setMaximumHeight(95); layout.addWidget(self.root_list)
        row = QHBoxLayout()
        self.add = QPushButton('Klasör ekle'); self.remove = QPushButton('Kaldır')
        row.addWidget(self.add); row.addWidget(self.remove); layout.addLayout(row)
        self.current = QPushButton('Açık veri setine izin ver'); layout.addWidget(self.current)
        self.config = QPushButton('Codex / Claude bağlantı ayarlarını kopyala'); layout.addWidget(self.config)
        self.status = QLabel('Kapalı — bağlantı noktası açılmadı.'); self.status.setWordWrap(True); layout.addWidget(self.status)
        self.requests_heading = QLabel('Komutlar / onay bekleyenler'); layout.addWidget(self.requests_heading)
        self.requests = QListWidget(); layout.addWidget(self.requests, 1)
        self.details = QPlainTextEdit(); self.details.setReadOnly(True); self.details.setMaximumHeight(180)
        layout.addWidget(self.details)
        buttons = QHBoxLayout()
        self.approve = QPushButton('Seçileni onayla'); self.reject = QPushButton('Reddet')
        buttons.addWidget(self.approve); buttons.addWidget(self.reject); layout.addLayout(buttons)
        self.stop = QPushButton('AI işlerini durdur / bağlantıyı kapat'); layout.addWidget(self.stop)
        self.setWidget(body)
        window.addDockWidget(Qt.RightDockWidgetArea, self)
        self.hide()
        self.enabled.toggled.connect(self._policy)
        self.mode.currentIndexChanged.connect(self._policy)
        self.share.toggled.connect(self._policy)
        self.add.clicked.connect(self._add_root)
        self.remove.clicked.connect(self._remove_root)
        self.current.clicked.connect(self._current_root)
        self.config.clicked.connect(self._copy_config)
        self.requests.currentItemChanged.connect(self._details)
        self.approve.clicked.connect(lambda: self._approve(True))
        self.reject.clicked.connect(lambda: self._approve(False))
        self.stop.clicked.connect(lambda: self.enabled.setChecked(False))
        self.timer = QTimer(self); self.timer.setInterval(75); self.timer.timeout.connect(self._tick)
        app.aboutToQuit.connect(self.shutdown)
        self.indicator = QPushButton('AI off' if theme.get_lang() == 'en' else 'AI pasif')
        self.indicator.setObjectName('ai_status_button')
        self.indicator.setCursor(Qt.PointingHandCursor)
        self.indicator.setToolTip('Open AI connection panel' if theme.get_lang() == 'en' else 'AI bağlantı panelini aç')
        self.indicator.clicked.connect(self._toggle_panel)
        self.warning = QLabel('Not connected' if theme.get_lang() == 'en' else 'Bağlantı yok')
        self.warning.setObjectName('ai_status_warning')
        self.warning.setToolTip('Click the AI button to connect.' if theme.get_lang() == 'en' else 'Bağlantıyı açmak için AI düğmesine tıkla.')
        self._indicator_row = QWidget(window)
        self._indicator_row.setObjectName('ai_status_row')
        theme.bind_style(self._indicator_row, lambda: 'QWidget#ai_status_row { background: transparent; border: none; }')
        indicator_layout = QHBoxLayout(self._indicator_row)
        indicator_layout.setContentsMargins(0, 0, 0, 0)
        indicator_layout.setSpacing(8)
        indicator_layout.addWidget(self.indicator)
        indicator_layout.addWidget(self.warning)
        if hasattr(window, '_studio_top_layout'):
            top = window._studio_top_layout
            top.insertWidget(top.indexOf(window._theme_button), self._indicator_row)
        else:
            window.statusBar().addPermanentWidget(self._indicator_row)
        self.toggle_action = QAction('AI bağlantı paneli', window)
        self.toggle_action.setShortcut('Ctrl+Shift+A')
        self.toggle_action.triggered.connect(self._toggle_panel)
        window.addAction(self.toggle_action)
        theme.bind_style(self.indicator, self._indicator_style)
        theme.bind_style(self.warning, lambda: f'color: {theme.RED}; font-size: {theme.fs(11)};')
        for button in (self.add, self.remove, self.current, self.config, self.approve, self.reject, self.stop):
            theme.bind_style(button, theme.btn_secondary)
        theme.bind_style(body, lambda: f'QWidget{{color:{theme.TEXT_PRIMARY};background:{theme.BG_CARD};}}')
        self.approve.setEnabled(False); self.reject.setEnabled(False)
        self.update_language()

    def _t(self, tr, en):
        return en if theme.get_lang() == 'en' else tr

    def update_language(self):
        t = self._t
        self.setWindowTitle(t('AI bağlantısı · Codex / Claude', 'AI connection · Codex / Claude'))
        self.intro.setText(t('Ajan, açık uygulamayı adlandırılmış araçlarla kontrol eder. İzinler sadece bu oturum içindir; her açılışta kontrol kapalıdır. Caption/metin yanıtları bağlı ajana iletilir.', 'An agent can control the open app through named tools. Permissions last for this session only; control starts off each time. Captions and text replies are shared with the connected agent.'))
        self.enabled.setText(t('AI bağlantısını aç', 'Enable AI connection'))
        for index, tr, en in ((0, 'Yalnız izle — değişiklik yok', 'Observe only — no changes'), (1, 'Onaylı kontrol — yazmadan/işten önce sor', 'Review control — ask before writes or jobs'), (2, 'Yetki devri — izinli klasörlerde otomatik', 'Delegated control — automatic in allowed folders')):
            self.mode.setItemText(index, t(tr, en))
        self.share.setText(t('Görsel önizlemelerini ajana paylaş', 'Share image previews with the agent'))
        self.share.setToolTip(t('Kapalıyken görsel bytes gönderilmez. Açılırsa bağlı Codex/Claude görseli kendi model sağlayıcısına iletebilir.', 'When off, image bytes are not sent. When on, the connected agent may forward previews to its model provider.'))
        self.roots_heading.setText(t('İzinli girdi / çıktı klasörleri', 'Allowed input / output folders'))
        self.add.setText(t('Klasör ekle', 'Add folder')); self.remove.setText(t('Kaldır', 'Remove'))
        self.current.setText(t('Açık veri setine izin ver', 'Allow current dataset'))
        self.config.setText(t('Codex / Claude bağlantı ayarlarını kopyala', 'Copy Codex / Claude connection settings'))
        if self.server is None and not self.enabled.isChecked():
            self.status.setText(t('Kapalı — bağlantı noktası açılmadı.', 'Off — no connection port is open.'))
        self.requests_heading.setText(t('Komutlar / onay bekleyenler', 'Commands / pending approvals'))
        self.approve.setText(t('Seçileni onayla', 'Approve selected')); self.reject.setText(t('Reddet', 'Reject'))
        self.stop.setText(t('AI işlerini durdur / bağlantıyı kapat', 'Stop AI jobs / disconnect'))
        self.toggle_action.setText(t('AI bağlantı paneli', 'AI connection panel'))
        self.indicator.setToolTip(t('AI bağlantı panelini aç', 'Open AI connection panel'))
        self.warning.setToolTip(t('Bağlantıyı açmak için AI düğmesine tıkla.', 'Click the AI button to connect.'))
        self._update_indicator()

    def _indicator_style(self):
        color = theme.GREEN if self.enabled.isChecked() and self.server is not None else theme.RED
        return (f'QPushButton{{background:{theme.BG_SURFACE};color:{color};'
                f'border:1px solid {color};border-radius:6px;padding:6px 10px;'
                f'font-weight:600;}}QPushButton:hover{{background:{theme.BG_HOVER};}}')

    def _update_indicator(self):
        active = self.enabled.isChecked() and self.server is not None
        self.indicator.setText(('AI on' if active else 'AI off') if theme.get_lang() == 'en' else ('AI etkin' if active else 'AI pasif'))
        self.warning.setText('Not connected' if theme.get_lang() == 'en' else 'Bağlantı yok')
        self.warning.setVisible(not active)
        theme.set_style_if_changed(self.indicator, theme.render_style(self.indicator))

    def _toggle_panel(self):
        self.setVisible(not self.isVisible())

    def _notice(self, exc):
        self.status.setText(str(exc))

    def _policy(self, *_):
        try:
            enabled = self.enabled.isChecked()
            # Revocation is immediate for queued commands; active writers stop cooperatively.
            self.live.stop_all()
            if enabled and not self.roots:
                raise ControlError('scope', self._t('Önce izinli klasör ekle. Proje/bilgisayar tamamı otomatik yetkilendirilmez.', 'Add an allowed folder first. The entire project or computer is not automatically authorized.'))
            if enabled and self.server is None:
                if self.live.active:
                    raise ControlError('busy', self._t('Önce durdurulan AI işinin tamamlanmasını bekle.', 'Wait for the stopped AI job to finish first.'))
                self.broker = Broker(ROOT / 'logs' / 'ai-control')
                self.live.broker = self.broker
            self.broker.configure(enabled=enabled, mode=self.mode.currentData(), roots=self.roots,
                                  share_images=self.share.isChecked())
            if enabled:
                if self.server is None:
                    self.server = ControlServer(self.broker, ROOT / '.ai-control' / 'session.json')
                    self.server.start()
                self.timer.start()
                self.status.setText(self._t('Bağlı', 'Connected') + ' · 127.0.0.1 · ' + self.mode.currentText() + '\nPort: ' + str(self.server.http.server_port))
            else:
                if self.server:
                    self.server.stop(); self.server = None
                self.status.setText(self._t('Kapalı. Başlamış yazımlar güvenli adımda durduruluyor.', 'Disconnected. Active writes will stop at a safe step.') if self.live.active else self._t('Kapalı.', 'Disconnected.'))
                if not self.live.active:
                    self.timer.stop()
            self.sequence = -1
            self._refresh()
        except Exception as exc:
            self.broker.configure(enabled=False, mode='observe', roots=[])
            if self.server:
                self.server.stop(); self.server = None
            self.enabled.blockSignals(True); self.enabled.setChecked(False); self.enabled.blockSignals(False)
            self._notice(exc)
        finally:
            self._update_indicator()

    def _add_root(self):
        folder = QFileDialog.getExistingDirectory(self, self._t('AI için izinli yerel klasör', 'Allowed local folder for AI'))
        if folder:
            self._grant(folder)

    def _grant(self, folder):
        if self.live.active:
            self._notice(self._t('İzinleri değiştirmeden önce AI işlerini durdur.', 'Stop AI jobs before changing permissions.')); return
        path = str(Path(folder).resolve(strict=True))
        if path not in self.roots:
            self.roots.append(path); self.root_list.addItem(path)
            if self.enabled.isChecked():
                self._policy()

    def _current_root(self):
        folder = self.live.edit._folder
        if folder:
            self._grant(folder)
        else:
            self._notice(self._t('Önce programda dataset klasörü aç.', 'Open a dataset folder in the app first.'))

    def _remove_root(self):
        if self.live.active:
            self._notice('İzinleri değiştirmeden önce AI işlerini durdur.'); return
        row = self.root_list.currentRow()
        if row >= 0:
            self.root_list.takeItem(row); self.roots.pop(row)
            if self.enabled.isChecked():
                self._policy()

    def _copy_config(self):
        from src.core.ai_client import connection_config
        data = connection_config(ROOT, sys.executable)
        QApplication.clipboard().setText(data)
        self._notice(self._t('Mevcut kurulum yollarına göre Codex TOML + Claude JSON panoya kopyalandı. Var olan istemci ayarlarını silmeden ilgili bloğu ekle; token içermez.', 'Codex TOML and Claude JSON connection settings were copied to the clipboard. Add the relevant block without removing existing client settings; no token is included.'))

    def _tick(self):
        if self._dispatching:
            return  # A modal dialog must not admit a re-entrant command.
        if getattr(self.window_ref, '_close_pending', False) and self.enabled.isChecked():
            self.enabled.setChecked(False)
            return
        # One action per event-loop turn avoids re-entrant UI operations.
        request = self.broker.take()
        if request:
            self._dispatching = True
            try:
                result = self.live.dispatch(request)
                if result is not DEFERRED:
                    self.broker.finish(request['id'], result)
            except Exception as exc:
                self.broker.finish(request['id'], error={'code': getattr(exc, 'code', 'action_failed'), 'message': str(exc)[:2000]})
            finally:
                self._dispatching = False
        self._refresh()
        if not self.enabled.isChecked() and not self.live.active:
            self.timer.stop()

    def _refresh(self):
        busy = bool(self.live.active)
        for control in (self.mode, self.share, self.add, self.remove, self.current):
            control.setEnabled(not busy)
        if self.sequence == self.broker.sequence:
            return
        self.sequence = self.broker.sequence
        current = self.requests.currentItem()
        selected = current.data(Qt.UserRole) if current else None
        self.requests.blockSignals(True); self.requests.clear()
        for row in reversed(self.broker.history()):
            item = QListWidgetItem(f'{row["status"]} · {row["action"]} · {row["id"][:8]}')
            item.setData(Qt.UserRole, row['id']); self.requests.addItem(item)
            if selected == row['id']:
                self.requests.setCurrentItem(item)
        self.requests.blockSignals(False)
        self._details()

    def _details(self, *_):
        item = self.requests.currentItem()
        identifier = item.data(Qt.UserRole) if item else None
        pending = False
        if identifier:
            with self.broker.lock:
                r = self.broker.requests[identifier]
                pending = r['status'] == 'pending_approval'
                text = CATALOG[r['action']].description + '\n\n' + json.dumps(
                    {'action': r['action'], 'arguments': r['arguments'], 'status': r['status'], 'error': r['error']}, ensure_ascii=False, indent=2)
            self.details.setPlainText(text[:100000])
        else:
            self.details.clear()
        self.approve.setEnabled(pending); self.reject.setEnabled(pending)

    def _approve(self, approved):
        item = self.requests.currentItem()
        if item:
            try:
                self.broker.approve(item.data(Qt.UserRole), approved)
            except Exception as exc:
                self._notice(exc)
            self._refresh()

    def shutdown(self):
        self.broker.configure(enabled=False, mode='observe', roots=[])
        self.live.stop_all()
        if self.server:
            self.server.stop(); self.server = None
        self.timer.stop()
        self._update_indicator()


def attach_ai_control(window, app):
    """No socket is opened here. A human must enable the panel each session."""
    if not hasattr(window, '_ai_control_panel'):
        window._ai_control_panel = AIControlPanel(window, app)
    return window._ai_control_panel
