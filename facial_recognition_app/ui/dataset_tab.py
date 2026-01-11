"""
Pestanya de creació de dataset per la UI.
Proporciona controls i visualització de progrés.
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QLineEdit, QProgressBar, QTextEdit, QGroupBox, QSpinBox,
    QCheckBox, QFileDialog, QFormLayout
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QFont
import os
import logging

from dataset_creator import DatasetCreator

logger = logging.getLogger(__name__)


class DatasetCreatorThread(QThread):
    """Thread per executar la creació de dataset en background."""

    # Signals
    progress_updated = pyqtSignal(int, int, float)  # processed, total, percentage
    stats_updated = pyqtSignal(dict)  # stats dictionary
    finished = pyqtSignal(bool)  # success
    error_occurred = pyqtSignal(str)  # error message

    def __init__(self, creator: DatasetCreator, root_dir: str, auto_tune: bool):
        """
        Inicialitza thread.

        Args:
            creator: Instància de DatasetCreator
            root_dir: Directori a processar
            auto_tune: Fer auto-tuning
        """
        super().__init__()
        self.creator = creator
        self.root_dir = root_dir
        self.auto_tune = auto_tune

    def run(self):
        """Executa el processament."""
        try:
            # Configura callbacks
            self.creator.set_callbacks(
                progress_callback=self._progress_callback,
                stats_callback=self._stats_callback
            )

            # Processa
            self.creator.process_folder(self.root_dir, auto_tune=self.auto_tune)

            self.finished.emit(True)

        except Exception as e:
            logger.error(f"Error en thread de creació: {str(e)}")
            self.error_occurred.emit(str(e))
            self.finished.emit(False)

    def _progress_callback(self, processed: int, total: int, percentage: float):
        """Callback per actualitzar progrés."""
        self.progress_updated.emit(processed, total, percentage)

    def _stats_callback(self, stats: dict):
        """Callback per actualitzar estadístiques."""
        self.stats_updated.emit(stats)


class DatasetTab(QWidget):
    """Pestanya per crear dataset d'embeddings."""

    def __init__(self, model_path: str):
        """
        Inicialitza la pestanya.

        Args:
            model_path: Path al model ONNX
        """
        super().__init__()
        self.model_path = model_path
        self.creator = None
        self.creator_thread = None

        self._init_ui()

    def _init_ui(self):
        """Inicialitza la interfície."""
        layout = QVBoxLayout()

        # Selector de carpeta
        folder_layout = QHBoxLayout()
        self.folder_input = QLineEdit()
        self.folder_input.setPlaceholderText("Selecciona carpeta amb imatges...")
        folder_layout.addWidget(QLabel("Carpeta:"))
        folder_layout.addWidget(self.folder_input)

        self.folder_btn = QPushButton("Selecciona")
        self.folder_btn.clicked.connect(self._select_folder)
        folder_layout.addWidget(self.folder_btn)

        layout.addLayout(folder_layout)

        # Configuració avançada (collapsable)
        self.config_group = QGroupBox("Configuració Avançada")
        self.config_group.setCheckable(True)
        self.config_group.setChecked(False)

        config_layout = QFormLayout()

        # Batch size
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(16, 512)
        self.batch_size_spin.setValue(128)
        self.batch_size_spin.setSingleStep(16)
        config_layout.addRow("Batch Size:", self.batch_size_spin)

        # Threads I/O
        self.threads_spin = QSpinBox()
        self.threads_spin.setRange(2, 16)
        self.threads_spin.setValue(6)
        config_layout.addRow("Threads I/O:", self.threads_spin)

        # FP16
        self.fp16_check = QCheckBox("Activar FP16 (Mixed Precision)")
        config_layout.addRow(self.fp16_check)

        # Skip duplicates
        self.skip_dup_check = QCheckBox("Saltar duplicats (més lent)")
        config_layout.addRow(self.skip_dup_check)

        # Auto-tune
        self.auto_tune_check = QCheckBox("Auto-tuning de batch size")
        self.auto_tune_check.setChecked(True)
        config_layout.addRow(self.auto_tune_check)

        self.config_group.setLayout(config_layout)
        layout.addWidget(self.config_group)

        # Botó d'inici
        self.start_btn = QPushButton("INICIAR PROCESSAMENT")
        self.start_btn.setStyleSheet("QPushButton { font-size: 14pt; font-weight: bold; padding: 10px; }")
        self.start_btn.clicked.connect(self._start_processing)
        layout.addWidget(self.start_btn)

        # Barra de progrés
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p% (%v / %m)")
        layout.addWidget(self.progress_bar)

        # Estadístiques
        stats_layout = QHBoxLayout()

        # Velocitat
        self.speed_label = QLabel("0 imgs/sec")
        self.speed_label.setFont(QFont("Arial", 16, QFont.Bold))
        self.speed_label.setAlignment(Qt.AlignCenter)
        stats_layout.addWidget(self._create_stat_box("Velocitat", self.speed_label))

        # Temps transcorregut
        self.elapsed_label = QLabel("00:00:00")
        self.elapsed_label.setFont(QFont("Arial", 16, QFont.Bold))
        self.elapsed_label.setAlignment(Qt.AlignCenter)
        stats_layout.addWidget(self._create_stat_box("Temps", self.elapsed_label))

        # ETA
        self.eta_label = QLabel("--:--:--")
        self.eta_label.setFont(QFont("Arial", 16, QFont.Bold))
        self.eta_label.setAlignment(Qt.AlignCenter)
        stats_layout.addWidget(self._create_stat_box("ETA", self.eta_label))

        layout.addLayout(stats_layout)

        # Errors
        error_layout = QVBoxLayout()
        error_layout.addWidget(QLabel("Log d'errors (últims 100):"))

        self.error_log = QTextEdit()
        self.error_log.setReadOnly(True)
        self.error_log.setMaximumHeight(100)
        error_layout.addWidget(self.error_log)

        layout.addLayout(error_layout)

        # Botó pausar/reprendre
        self.pause_btn = QPushButton("PAUSAR")
        self.pause_btn.setEnabled(False)
        self.pause_btn.clicked.connect(self._toggle_pause)
        layout.addWidget(self.pause_btn)

        self.setLayout(layout)

    def _create_stat_box(self, title: str, label: QLabel) -> QGroupBox:
        """Crea una caixa d'estadística."""
        box = QGroupBox(title)
        layout = QVBoxLayout()
        layout.addWidget(label)
        box.setLayout(layout)
        return box

    def _select_folder(self):
        """Obre diàleg per seleccionar carpeta."""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Selecciona carpeta amb imatges",
            "",
            QFileDialog.ShowDirsOnly
        )

        if folder:
            self.folder_input.setText(folder)

    def _start_processing(self):
        """Inicia el processament."""
        root_dir = self.folder_input.text()

        if not root_dir or not os.path.isdir(root_dir):
            self.error_log.append("ERROR: Selecciona una carpeta vàlida")
            return

        # Crea instància del creator
        self.creator = DatasetCreator(self.model_path)

        # Configura
        self.creator.configure(
            batch_size=self.batch_size_spin.value(),
            num_workers=self.threads_spin.value(),
            use_fp16=self.fp16_check.isChecked(),
            skip_duplicates=self.skip_dup_check.isChecked()
        )

        # Crea i inicia thread
        self.creator_thread = DatasetCreatorThread(
            self.creator,
            root_dir,
            auto_tune=self.auto_tune_check.isChecked()
        )

        # Connecta signals
        self.creator_thread.progress_updated.connect(self._update_progress)
        self.creator_thread.stats_updated.connect(self._update_stats)
        self.creator_thread.finished.connect(self._processing_finished)
        self.creator_thread.error_occurred.connect(self._processing_error)

        # Inicia thread
        self.creator_thread.start()

        # Actualitza UI
        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.folder_btn.setEnabled(False)
        self.config_group.setEnabled(False)

        logger.info("Processament iniciat")

    def _update_progress(self, processed: int, total: int, percentage: float):
        """Actualitza barra de progrés."""
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(processed)

    def _update_stats(self, stats: dict):
        """Actualitza estadístiques."""
        # Velocitat
        speed = stats.get('speed', 0)
        self.speed_label.setText(f"{speed:.1f} imgs/sec")

        # Temps transcorregut
        elapsed = stats.get('elapsed', 0)
        elapsed_str = self._format_time(elapsed)
        self.elapsed_label.setText(elapsed_str)

        # ETA
        eta = stats.get('eta', 0)
        eta_str = self._format_time(eta)
        self.eta_label.setText(eta_str)

        # Errors
        errors = stats.get('errors', 0)
        if errors > 0:
            self.error_log.append(f"Total errors: {errors}")

    def _format_time(self, seconds: float) -> str:
        """Formata segons en HH:MM:SS."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def _toggle_pause(self):
        """Pausa/reprèn el processament."""
        if not self.creator:
            return

        if self.creator.paused:
            self.creator.resume()
            self.pause_btn.setText("PAUSAR")
        else:
            self.creator.pause()
            self.pause_btn.setText("REPRENDRE")

    def _processing_finished(self, success: bool):
        """Callback quan el processament finalitza."""
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.folder_btn.setEnabled(True)
        self.config_group.setEnabled(True)

        if success:
            self.error_log.append("PROCESSAMENT COMPLETAT!")
            logger.info("Processament completat")
        else:
            self.error_log.append("Processament aturat")

    def _processing_error(self, error_msg: str):
        """Callback quan hi ha un error."""
        self.error_log.append(f"ERROR: {error_msg}")
        logger.error(error_msg)
