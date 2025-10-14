"""
Modern UI Module for LoRA-Harvester
AI-Powered Dataset Collection Tool with PyQt5 interface
"""

import sys
import os
import subprocess
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QSlider, 
                             QComboBox, QCheckBox, QProgressBar, QFileDialog,
                             QTextEdit, QGroupBox, QSpinBox, QToolButton)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QUrl
from PyQt5.QtGui import QFont, QPalette, QColor, QDragEnterEvent, QDropEvent, QIcon
from typing import Optional
from src.ui.translations import get_text


class ProcessingThread(QThread):
    """Background thread for video processing"""
    
    progress_update = pyqtSignal(float, dict)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, processor, frame_interval, skip_text):
        super().__init__()
        self.processor = processor
        self.frame_interval = frame_interval
        self.skip_text = skip_text
        self._is_running = True
    
    def run(self):
        """Run video processing"""
        try:
            stats = self.processor.process_video(
                frame_interval=self.frame_interval,
                skip_text=self.skip_text,
                progress_callback=self.progress_callback,
                stop_callback=self.should_stop
            )
            self.finished.emit(stats)
        except Exception as e:
            if "stopped" not in str(e).lower():
                self.error.emit(str(e))
    
    def progress_callback(self, progress, stats):
        """Callback for progress updates"""
        self.progress_update.emit(progress, stats)
    
    def should_stop(self):
        """Check if processing should stop"""
        return not self._is_running
    
    def stop(self):
        """Stop processing gracefully"""
        self._is_running = False


class DropZone(QLabel):
    """Drag and drop zone for video files"""
    
    file_dropped = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumHeight(150)
        self.setStyleSheet("""
            QLabel {
                border: 3px dashed #9b59b6;
                border-radius: 10px;
                background-color: #2c2c3e;
                color: #c39bd3;
                font-size: 14px;
                padding: 20px;
            }
            QLabel:hover {
                background-color: #3d3d5c;
                border-color: #bb86fc;
            }
        """)
        self.setText("🎬 Drag & Drop Video File Here\nor click 'Browse' button")
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handle drag enter"""
        if event.mimeData().hasUrls():
            event.accept()
            self.setStyleSheet(self.styleSheet().replace('#2c2c3e', '#1a1a2e'))
        else:
            event.ignore()
    
    def dragLeaveEvent(self, event):
        """Handle drag leave"""
        self.setStyleSheet(self.styleSheet().replace('#1a1a2e', '#2c2c3e'))
    
    def dropEvent(self, event: QDropEvent):
        """Handle drop"""
        self.setStyleSheet(self.styleSheet().replace('#1a1a2e', '#2c2c3e'))
        
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        if files:
            video_file = files[0]
            # Check if it's a video file
            valid_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']
            if any(video_file.lower().endswith(ext) for ext in valid_extensions):
                self.file_dropped.emit(video_file)
                self.setText(f"✅ {Path(video_file).name}")
            else:
                self.setText("❌ Invalid file type. Please drop a video file.")


class VideoSmartCropperUI(QMainWindow):
    """Main UI window"""
    
    def __init__(self):
        super().__init__()
        self.video_path = None
        self.processor = None
        self.processing_thread = None
        self.current_lang = 'tr'  # Default to Turkish
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize UI components"""
        self.setWindowTitle(get_text('app_title', self.current_lang))
        self.setGeometry(100, 100, 900, 800)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Language selector at top-right
        lang_layout = QHBoxLayout()
        lang_layout.addStretch()
        lang_label = QLabel("�")
        lang_label.setFont(QFont('Arial', 14))
        self.lang_combo = QComboBox()
        self.lang_combo.addItems(['🇹🇷 Türkçe', '🇬🇧 English'])
        self.lang_combo.setCurrentIndex(0)  # Turkish default
        self.lang_combo.setStyleSheet("""
            QComboBox {
                padding: 5px 10px;
                border: 2px solid #9b59b6;
                border-radius: 5px;
                background-color: #2c2c3e;
                color: #ecf0f1;
                font-weight: bold;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox QAbstractItemView {
                background-color: #3d3d5c;
                color: #ecf0f1;
                selection-background-color: #9b59b6;
            }
        """)
        self.lang_combo.currentIndexChanged.connect(self.change_language)
        lang_layout.addWidget(lang_label)
        lang_layout.addWidget(self.lang_combo)
        main_layout.addLayout(lang_layout)
        
        # Title
        self.title_label = QLabel(get_text('title', self.current_lang))
        self.title_label.setFont(QFont('Arial', 24, QFont.Bold))
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("color: #ecf0f1; margin: 20px;")
        main_layout.addWidget(self.title_label)
        
        # Subtitle
        self.subtitle_label = QLabel(get_text('subtitle', self.current_lang))
        self.subtitle_label.setFont(QFont('Arial', 11))
        self.subtitle_label.setAlignment(Qt.AlignCenter)
        self.subtitle_label.setStyleSheet("color: #95a5a6; margin-bottom: 20px;")
        main_layout.addWidget(self.subtitle_label)
        
        # Drop zone
        self.drop_zone = DropZone()
        self.drop_zone.file_dropped.connect(self.on_file_dropped)
        self.update_drop_zone_text()
        main_layout.addWidget(self.drop_zone)
        
        # Browse button
        self.browse_btn = QPushButton(get_text('browse_btn', self.current_lang))
        self.browse_btn.setStyleSheet("""
            QPushButton {
                background-color: #8e44ad;
                color: white;
                border: none;
                padding: 12px;
                font-size: 14px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #9b59b6;
            }
        """)
        self.browse_btn.clicked.connect(self.browse_video)
        main_layout.addWidget(self.browse_btn)
        
        # Settings Group
        self.settings_group = QGroupBox(get_text('settings_title', self.current_lang))
        self.settings_group.setStyleSheet("""
            QGroupBox {
                font-size: 14px;
                font-weight: bold;
                border: 2px solid #9b59b6;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 15px;
                background-color: #2c2c3e;
                color: #c39bd3;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        settings_layout = QVBoxLayout()
        self.settings_group.setLayout(settings_layout)
        
        # Frame interval slider
        interval_layout = QHBoxLayout()
        self.interval_label = QLabel(get_text('frame_interval', self.current_lang))
        self.interval_label.setStyleSheet("color: #ecf0f1;")
        self.interval_help = QLabel("❓")
        self.interval_help.setStyleSheet("color: #bb86fc; font-size: 16px; font-weight: bold;")
        self.interval_help.setToolTip(get_text('frame_interval_tooltip', self.current_lang))
        self.interval_help.setMouseTracking(True)
        self.interval_slider = QSlider(Qt.Horizontal)
        self.interval_slider.setMinimum(1)
        self.interval_slider.setMaximum(120)
        self.interval_slider.setValue(30)
        self.interval_slider.setTickPosition(QSlider.TicksBelow)
        self.interval_slider.setTickInterval(10)
        self.interval_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                background: #1a1a2e;
                height: 8px;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #9b59b6;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
        """)
        self.interval_value_label = QLabel("30")
        self.interval_value_label.setStyleSheet("font-weight: bold; color: #ecf0f1; min-width: 30px;")
        self.interval_slider.valueChanged.connect(
            lambda v: self.interval_value_label.setText(str(v))
        )
        interval_layout.addWidget(self.interval_label)
        interval_layout.addWidget(self.interval_help)
        interval_layout.addWidget(self.interval_slider)
        interval_layout.addWidget(self.interval_value_label)
        settings_layout.addLayout(interval_layout)
        
        # Aspect ratio selector
        ratio_layout = QHBoxLayout()
        self.ratio_label = QLabel(get_text('output_format', self.current_lang))
        self.ratio_label.setStyleSheet("color: #ecf0f1;")
        self.ratio_help = QLabel("❓")
        self.ratio_help.setStyleSheet("color: #bb86fc; font-size: 16px; font-weight: bold;")
        self.ratio_help.setToolTip(get_text('output_format_tooltip', self.current_lang))
        self.ratio_help.setMouseTracking(True)
        self.ratio_combo = QComboBox()
        self.ratio_combo.addItems(['9:16', '3:4', '1:1', '4:5', '16:9', '4:3'])
        self.ratio_combo.setStyleSheet("""
            QComboBox {
                padding: 8px;
                border: 2px solid #9b59b6;
                border-radius: 3px;
                background-color: #2c2c3e;
                color: #ecf0f1;
                font-weight: bold;
                min-width: 100px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox QAbstractItemView {
                background-color: #3d3d5c;
                color: #ecf0f1;
                selection-background-color: #9b59b6;
            }
        """)
        ratio_layout.addWidget(self.ratio_label)
        ratio_layout.addWidget(self.ratio_help)
        ratio_layout.addWidget(self.ratio_combo)
        ratio_layout.addStretch()
        settings_layout.addLayout(ratio_layout)
        
        # Detection confidence
        conf_layout = QHBoxLayout()
        self.conf_label = QLabel(get_text('confidence', self.current_lang))
        self.conf_label.setStyleSheet("color: #ecf0f1;")
        self.conf_help = QLabel("❓")
        self.conf_help.setStyleSheet("color: #bb86fc; font-size: 16px; font-weight: bold;")
        self.conf_help.setToolTip(get_text('confidence_tooltip', self.current_lang))
        self.conf_help.setMouseTracking(True)
        self.conf_spinbox = QSpinBox()
        self.conf_spinbox.setMinimum(10)
        self.conf_spinbox.setMaximum(95)
        self.conf_spinbox.setValue(50)
        self.conf_spinbox.setSuffix("%")
        self.conf_spinbox.setStyleSheet("""
            QSpinBox {
                padding: 5px;
                border: 2px solid #9b59b6;
                border-radius: 3px;
                background-color: #2c2c3e;
                color: #ecf0f1;
                font-weight: bold;
            }
        """)
        conf_layout.addWidget(self.conf_label)
        conf_layout.addWidget(self.conf_help)
        conf_layout.addWidget(self.conf_spinbox)
        conf_layout.addStretch()
        settings_layout.addLayout(conf_layout)
        
        # Ensemble mode checkbox
        ensemble_layout_cb = QHBoxLayout()
        self.ensemble_cb = QCheckBox(get_text('ensemble_mode', self.current_lang))
        self.ensemble_cb.setChecked(False)
        self.ensemble_cb.setStyleSheet("font-size: 12px; font-weight: bold; color: #e74c3c;")
        self.ensemble_help = QLabel("❓")
        self.ensemble_help.setStyleSheet("color: #cf6679; font-size: 16px; font-weight: bold;")
        self.ensemble_help.setToolTip(get_text('ensemble_mode_tooltip', self.current_lang))
        self.ensemble_help.setMouseTracking(True)
        ensemble_layout_cb.addWidget(self.ensemble_cb)
        ensemble_layout_cb.addWidget(self.ensemble_help)
        ensemble_layout_cb.addStretch()
        settings_layout.addLayout(ensemble_layout_cb)
        
        # Ensemble settings (initially hidden)
        self.ensemble_group = QGroupBox(get_text('ensemble_settings', self.current_lang))
        self.ensemble_group.setVisible(False)
        self.ensemble_group.setStyleSheet("""
            QGroupBox {
                font-size: 12px;
                font-weight: bold;
                border: 2px solid #cf6679;
                border-radius: 5px;
                margin-top: 5px;
                padding-top: 10px;
                background-color: #2c2c3e;
                color: #cf6679;
            }
        """)
        ensemble_layout = QVBoxLayout()
        
        # Model selection checkboxes
        models_layout = QHBoxLayout()
        self.models_label = QLabel(get_text('active_models', self.current_lang))
        self.models_label.setStyleSheet("color: #ecf0f1;")
        self.yolo_cb = QCheckBox("YOLOv8")
        self.yolo_cb.setChecked(True)
        self.yolo_cb.setStyleSheet("color: #ecf0f1;")
        self.detr_cb = QCheckBox("DETR (Transformer)")
        self.detr_cb.setChecked(True)
        self.detr_cb.setStyleSheet("color: #ecf0f1;")
        self.fasterrcnn_cb = QCheckBox("Faster R-CNN")
        self.fasterrcnn_cb.setChecked(True)
        self.fasterrcnn_cb.setStyleSheet("color: #ecf0f1;")
        models_layout.addWidget(self.models_label)
        models_layout.addWidget(self.yolo_cb)
        models_layout.addWidget(self.detr_cb)
        models_layout.addWidget(self.fasterrcnn_cb)
        models_layout.addStretch()
        ensemble_layout.addLayout(models_layout)
        
        # Voting threshold
        voting_layout = QHBoxLayout()
        self.voting_label = QLabel(get_text('voting_threshold', self.current_lang))
        self.voting_label.setStyleSheet("color: #ecf0f1;")
        self.voting_help = QLabel("❓")
        self.voting_help.setStyleSheet("color: #cf6679; font-size: 16px; font-weight: bold;")
        self.voting_help.setToolTip(get_text('voting_threshold_tooltip', self.current_lang))
        self.voting_help.setMouseTracking(True)
        self.voting_spinbox = QSpinBox()
        self.voting_spinbox.setMinimum(1)
        self.voting_spinbox.setMaximum(3)
        self.voting_spinbox.setValue(2)
        self.voting_spinbox.setStyleSheet("""
            QSpinBox {
                padding: 5px;
                border: 2px solid #cf6679;
                border-radius: 3px;
                background-color: #2c2c3e;
                color: #ecf0f1;
                font-weight: bold;
            }
        """)
        voting_layout.addWidget(self.voting_label)
        voting_layout.addWidget(self.voting_help)
        voting_layout.addWidget(self.voting_spinbox)
        voting_layout.addStretch()
        ensemble_layout.addLayout(voting_layout)
        
        self.ensemble_group.setLayout(ensemble_layout)
        settings_layout.addWidget(self.ensemble_group)
        
        # Connect ensemble checkbox to show/hide settings
        self.ensemble_cb.toggled.connect(self.ensemble_group.setVisible)
        
        # Skip subtitle checkbox
        skip_layout_cb = QHBoxLayout()
        self.skip_subtitle_cb = QCheckBox(get_text('skip_subtitle', self.current_lang))
        self.skip_subtitle_cb.setChecked(True)
        self.skip_subtitle_cb.setStyleSheet("font-size: 12px; color: #ecf0f1;")
        self.skip_help = QLabel("❓")
        self.skip_help.setStyleSheet("color: #bb86fc; font-size: 16px; font-weight: bold;")
        self.skip_help.setToolTip(get_text('skip_subtitle_tooltip', self.current_lang))
        self.skip_help.setMouseTracking(True)
        skip_layout_cb.addWidget(self.skip_subtitle_cb)
        skip_layout_cb.addWidget(self.skip_help)
        skip_layout_cb.addStretch()
        settings_layout.addLayout(skip_layout_cb)
        
        # Turbo mode checkbox
        turbo_layout_cb = QHBoxLayout()
        self.turbo_cb = QCheckBox(get_text('turbo_mode', self.current_lang))
        self.turbo_cb.setChecked(True)
        self.turbo_cb.setStyleSheet("font-size: 12px; font-weight: bold; color: #f39c12;")
        self.turbo_help = QLabel("❓")
        self.turbo_help.setStyleSheet("color: #ffa726; font-size: 16px; font-weight: bold;")
        self.turbo_help.setToolTip(get_text('turbo_mode_tooltip', self.current_lang))
        self.turbo_help.setMouseTracking(True)
        turbo_layout_cb.addWidget(self.turbo_cb)
        turbo_layout_cb.addWidget(self.turbo_help)
        turbo_layout_cb.addStretch()
        settings_layout.addLayout(turbo_layout_cb)
        
        # Minimum padding
        padding_layout = QHBoxLayout()
        self.padding_label = QLabel(get_text('min_padding', self.current_lang))
        self.padding_label.setStyleSheet("color: #ecf0f1;")
        self.padding_help = QLabel("❓")
        self.padding_help.setStyleSheet("color: #bb86fc; font-size: 16px; font-weight: bold;")
        self.padding_help.setToolTip(get_text('min_padding_tooltip', self.current_lang))
        self.padding_help.setMouseTracking(True)
        self.padding_spinbox = QSpinBox()
        self.padding_spinbox.setMinimum(100)
        self.padding_spinbox.setMaximum(1000)
        self.padding_spinbox.setValue(500)
        self.padding_spinbox.setSingleStep(50)
        self.padding_spinbox.setStyleSheet("""
            QSpinBox {
                padding: 5px;
                border: 2px solid #9b59b6;
                border-radius: 3px;
                background-color: #2c2c3e;
                color: #ecf0f1;
                font-weight: bold;
            }
        """)
        padding_layout.addWidget(self.padding_label)
        padding_layout.addWidget(self.padding_help)
        padding_layout.addWidget(self.padding_spinbox)
        padding_layout.addStretch()
        settings_layout.addLayout(padding_layout)
        
        main_layout.addWidget(self.settings_group)
        
        # Process and Stop buttons
        buttons_layout = QHBoxLayout()
        
        self.process_btn = QPushButton(get_text('start_btn', self.current_lang))
        self.process_btn.setEnabled(False)
        self.process_btn.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                border: none;
                padding: 15px;
                font-size: 16px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #229954;
            }
            QPushButton:disabled {
                background-color: #7f8c8d;
            }
        """)
        self.process_btn.clicked.connect(self.start_processing)
        
        self.stop_btn = QPushButton(get_text('stop_btn', self.current_lang))
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: none;
                padding: 15px;
                font-size: 16px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
            QPushButton:disabled {
                background-color: #7f8c8d;
            }
        """)
        self.stop_btn.clicked.connect(self.stop_processing)
        
        # Open output folder button
        self.open_output_btn = QPushButton(get_text('open_output_btn', self.current_lang))
        self.open_output_btn.setStyleSheet("""
            QPushButton {
                background-color: #f39c12;
                color: white;
                border: none;
                padding: 15px;
                font-size: 16px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #e67e22;
            }
        """)
        self.open_output_btn.clicked.connect(self.open_output_folder)
        
        buttons_layout.addWidget(self.process_btn)
        buttons_layout.addWidget(self.stop_btn)
        buttons_layout.addWidget(self.open_output_btn)
        main_layout.addLayout(buttons_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #9b59b6;
                border-radius: 5px;
                text-align: center;
                height: 25px;
                background-color: #2c2c3e;
                color: #c39bd3;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #8e44ad;
            }
        """)
        main_layout.addWidget(self.progress_bar)
        
        # Status/Log area
        self.log_label = QLabel(get_text('log_title', self.current_lang))
        self.log_label.setStyleSheet("font-weight: bold; margin-top: 10px; color: #ecf0f1;")
        main_layout.addWidget(self.log_label)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #0d0d0d;
                color: #bb86fc;
                border: 2px solid #9b59b6;
                border-radius: 5px;
                padding: 10px;
                font-family: 'Consolas', monospace;
                font-size: 11px;
            }
        """)
        main_layout.addWidget(self.log_text)
        
        # Apply dark theme styling
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QLabel {
                color: #ecf0f1;
            }
        """)
        
        self.log(get_text('log_started', self.current_lang))
    
    def log(self, message: str):
        """Add message to log"""
        self.log_text.append(message)
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )
    
    def browse_video(self):
        """Browse for video file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.flv *.wmv *.webm);;All Files (*)"
        )
        
        if file_path:
            self.on_file_dropped(file_path)
    
    def open_output_folder(self):
        """Open output folder in file explorer"""
        output_path = Path("output")
        
        # Create output folder if it doesn't exist
        output_path.mkdir(exist_ok=True)
        
        # Open folder in file explorer
        try:
            if sys.platform == "win32":
                os.startfile(str(output_path))
            elif sys.platform == "darwin":  # macOS
                subprocess.call(["open", str(output_path)])
            else:  # Linux
                subprocess.call(["xdg-open", str(output_path)])
            
            self.log("📂 Output folder opened")
        except Exception as e:
            self.log(f"❌ Failed to open output folder: {str(e)}")
    
    def on_file_dropped(self, file_path: str):
        """Handle file selection"""
        self.video_path = file_path
        self.process_btn.setEnabled(True)
        self.log(get_text('log_loaded', self.current_lang).format(Path(file_path).name))
        
        # Update drop zone
        self.drop_zone.setText(get_text('drop_zone_success', self.current_lang).format(Path(file_path).name))
    
    def stop_processing(self):
        """Stop video processing"""
        if self.processing_thread and self.processing_thread.isRunning():
            self.log(get_text('log_stopping', self.current_lang))
            self.processing_thread.stop()
            self.stop_btn.setEnabled(False)
    
    def start_processing(self):
        """Start video processing"""
        if not self.video_path:
            self.log(get_text('log_no_file', self.current_lang))
            return
        
        # Disable UI during processing
        self.process_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.drop_zone.setEnabled(False)
        
        # Get settings
        frame_interval = self.interval_slider.value()
        skip_text = self.skip_subtitle_cb.isChecked()
        confidence = self.conf_spinbox.value() / 100.0
        min_padding = self.padding_spinbox.value()
        
        # Get aspect ratio - directly from combo text
        aspect_ratio = self.ratio_combo.currentText()
        
        # Get ensemble settings
        use_ensemble = self.ensemble_cb.isChecked()
        
        self.log(get_text('log_settings', self.current_lang).format(frame_interval, aspect_ratio, confidence))
        
        if use_ensemble:
            self.log(get_text('log_ensemble_on', self.current_lang))
            self.log(get_text('log_ensemble_info', self.current_lang))
        else:
            self.log(get_text('log_single_mode', self.current_lang))
        
        self.log(get_text('log_init', self.current_lang))
        
        # Import and initialize processors
        try:
            from src.core.text_detector import SubtitleDetector
            from src.core.cropper import SmartCropper
            from src.core.video_processor import VideoProcessor
            
            # Create output directory
            video_name = Path(self.video_path).stem
            mode_suffix = "ensemble" if use_ensemble else "yolo"
            output_dir = Path("output") / f"{video_name}_{aspect_ratio.replace(':', 'x')}_{mode_suffix}"
            
            # Initialize detector (ensemble or single)
            if use_ensemble:
                from src.core.ensemble_detector import EnsembleDetector
                
                # Get selected models
                models_to_use = []
                if self.yolo_cb.isChecked():
                    models_to_use.append('yolo')
                if self.detr_cb.isChecked():
                    models_to_use.append('detr')
                if self.fasterrcnn_cb.isChecked():
                    models_to_use.append('fasterrcnn')
                
                if not models_to_use:
                    self.log(get_text('log_error_model', self.current_lang))
                    self.process_btn.setEnabled(True)
                    self.drop_zone.setEnabled(True)
                    return
                
                voting_threshold = self.voting_spinbox.value()
                
                detector = EnsembleDetector(
                    models_to_use=models_to_use,
                    confidence_threshold=confidence,
                    voting_threshold=voting_threshold
                )
                
                self.log(get_text('log_models_loaded', self.current_lang).format(', '.join(models_to_use)))
                self.log(get_text('log_voting', self.current_lang).format(voting_threshold, len(models_to_use)))
            else:
                from src.core.detector import ObjectDetector
                detector = ObjectDetector(confidence=confidence)
            
            text_detector = SubtitleDetector() if skip_text else None
            cropper = SmartCropper(target_format=aspect_ratio, min_padding=min_padding)
            
            # Choose processor type
            use_turbo = self.turbo_cb.isChecked()
            
            if use_turbo:
                from src.core.optimized_processor import TurboVideoProcessor
                self.processor = TurboVideoProcessor(
                    self.video_path,
                    str(output_dir),
                    detector,
                    text_detector,
                    cropper,
                    batch_size=4  # Process 4 frames at once
                )
                self.log(get_text('log_turbo', self.current_lang))
            else:
                from src.core.video_processor import VideoProcessor
                self.processor = VideoProcessor(
                    self.video_path,
                    str(output_dir),
                    detector,
                    text_detector,
                    cropper
                )
            
            self.log(get_text('log_success', self.current_lang))
            self.log(get_text('log_processing', self.current_lang))
            
            # Start processing thread
            self.processing_thread = ProcessingThread(
                self.processor,
                frame_interval,
                skip_text
            )
            self.processing_thread.progress_update.connect(self.on_progress)
            self.processing_thread.finished.connect(self.on_finished)
            self.processing_thread.error.connect(self.on_error)
            self.processing_thread.start()
            
        except Exception as e:
            self.log(f"❌ Error: {str(e)}")
            self.process_btn.setEnabled(True)
            self.drop_zone.setEnabled(True)
    
    def on_progress(self, progress: float, stats: dict):
        """Update progress"""
        self.progress_bar.setValue(int(progress))
        
        # Update log with stats
        if stats['processed_frames'] % 10 == 0:  # Update every 10 frames
            self.log(get_text('log_progress', self.current_lang).format(
                progress, stats['saved_frames'], stats['person_frames'],
                stats['animal_frames'], stats['object_frames']
            ))
    
    def on_finished(self, stats: dict):
        """Processing finished"""
        self.progress_bar.setValue(100)
        self.log("\n" + "="*50)
        self.log(get_text('log_complete', self.current_lang))
        self.log(get_text('log_total', self.current_lang).format(stats['saved_frames']))
        self.log(get_text('log_persons', self.current_lang).format(stats['person_frames']))
        self.log(get_text('log_animals', self.current_lang).format(stats['animal_frames']))
        self.log(get_text('log_objects', self.current_lang).format(stats['object_frames']))
        self.log(get_text('log_skipped_text', self.current_lang).format(stats['skipped_text']))
        self.log(get_text('log_skipped_none', self.current_lang).format(stats['skipped_no_detection']))
        self.log("="*50)
        
        # Re-enable UI
        self.process_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.drop_zone.setEnabled(True)
    
    def on_error(self, error_msg: str):
        """Handle error"""
        self.log(get_text('log_error', self.current_lang).format(error_msg))
        self.process_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.drop_zone.setEnabled(True)
    
    def change_language(self, index):
        """Change UI language"""
        self.current_lang = 'tr' if index == 0 else 'en'
        self.update_ui_texts()
    
    def update_ui_texts(self):
        """Update all UI texts with current language"""
        self.setWindowTitle(get_text('app_title', self.current_lang))
        self.title_label.setText(get_text('title', self.current_lang))
        self.subtitle_label.setText(get_text('subtitle', self.current_lang))
        self.browse_btn.setText(get_text('browse_btn', self.current_lang))
        self.settings_group.setTitle(get_text('settings_title', self.current_lang))
        self.interval_label.setText(get_text('frame_interval', self.current_lang))
        self.interval_help.setToolTip(get_text('frame_interval_tooltip', self.current_lang))
        self.ratio_label.setText(get_text('output_format', self.current_lang))
        self.ratio_help.setToolTip(get_text('output_format_tooltip', self.current_lang))
        self.conf_label.setText(get_text('confidence', self.current_lang))
        self.conf_help.setToolTip(get_text('confidence_tooltip', self.current_lang))
        self.ensemble_cb.setText(get_text('ensemble_mode', self.current_lang))
        self.ensemble_help.setToolTip(get_text('ensemble_mode_tooltip', self.current_lang))
        self.ensemble_group.setTitle(get_text('ensemble_settings', self.current_lang))
        self.models_label.setText(get_text('active_models', self.current_lang))
        self.voting_label.setText(get_text('voting_threshold', self.current_lang))
        self.voting_help.setToolTip(get_text('voting_threshold_tooltip', self.current_lang))
        self.skip_subtitle_cb.setText(get_text('skip_subtitle', self.current_lang))
        self.skip_help.setToolTip(get_text('skip_subtitle_tooltip', self.current_lang))
        self.turbo_cb.setText(get_text('turbo_mode', self.current_lang))
        self.turbo_help.setToolTip(get_text('turbo_mode_tooltip', self.current_lang))
        self.padding_label.setText(get_text('min_padding', self.current_lang))
        self.padding_help.setToolTip(get_text('min_padding_tooltip', self.current_lang))
        self.process_btn.setText(get_text('start_btn', self.current_lang))
        self.stop_btn.setText(get_text('stop_btn', self.current_lang))
        self.open_output_btn.setText(get_text('open_output_btn', self.current_lang))
        self.log_label.setText(get_text('log_title', self.current_lang))
        self.update_drop_zone_text()
    
    def update_drop_zone_text(self):
        """Update drop zone text"""
        if not self.video_path:
            self.drop_zone.setText(get_text('drop_zone', self.current_lang))
        else:
            self.drop_zone.setText(get_text('drop_zone_success', self.current_lang).format(Path(self.video_path).name))


def create_app():
    """Create and return the application"""
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create main window
    window = VideoSmartCropperUI()
    window.show()
    
    return app, window
