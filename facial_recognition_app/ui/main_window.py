"""
Finestra principal de l'aplicació.
Conté les pestanyes de creació de dataset i reconeixement.
"""

from PyQt5.QtWidgets import (
    QMainWindow, QTabWidget, QMessageBox, QAction, QMenuBar
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
import os
import logging

from ui.dataset_tab import DatasetTab
from ui.recognition_tab import RecognitionTab

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """Finestra principal amb pestanyes."""

    def __init__(self, model_path: str):
        """
        Inicialitza la finestra principal.

        Args:
            model_path: Path al model ONNX
        """
        super().__init__()
        self.model_path = model_path

        # Verifica que el model existeix
        if not os.path.exists(model_path):
            logger.warning(f"Model no trobat a {model_path}")
            QMessageBox.warning(
                self,
                "Model no trobat",
                f"El model ONNX no s'ha trobat a:\n{model_path}\n\n"
                "Si us plau, descarrega el model MobileFaceNet i col·loca'l a la carpeta models/."
            )

        self._init_ui()

    def _init_ui(self):
        """Inicialitza la interfície."""
        self.setWindowTitle("Sistema de Reconeixement Facial Ultra-Ràpid")
        self.setMinimumSize(1000, 700)

        # Crea menu bar
        self._create_menu_bar()

        # Crea tab widget
        self.tabs = QTabWidget()

        # Crea pestanyes
        self.dataset_tab = DatasetTab(self.model_path)
        self.recognition_tab = RecognitionTab(self.model_path)

        # Afegeix pestanyes
        self.tabs.addTab(self.dataset_tab, "Crear Dataset")
        self.tabs.addTab(self.recognition_tab, "Reconeixement Facial")

        # Configura com a widget central
        self.setCentralWidget(self.tabs)

        logger.info("Finestra principal inicialitzada")

    def _create_menu_bar(self):
        """Crea barra de menú."""
        menubar = self.menuBar()

        # Menu Fitxer
        file_menu = menubar.addMenu("Fitxer")

        # Sortir
        exit_action = QAction("Sortir", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Menu Ajuda
        help_menu = menubar.addMenu("Ajuda")

        # Sobre
        about_action = QAction("Sobre", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

        # Ajuda
        help_action = QAction("Ajuda", self)
        help_action.setShortcut("F1")
        help_action.triggered.connect(self._show_help)
        help_menu.addAction(help_action)

    def _show_about(self):
        """Mostra diàleg 'Sobre'."""
        about_text = """
        <h2>Sistema de Reconeixement Facial Ultra-Ràpid</h2>
        <p>Versió 1.0</p>
        <p>Aplicació optimitzada per màxima velocitat en creació d'embeddings
        facials i reconeixement.</p>
        <p><b>Tecnologies utilitzades:</b></p>
        <ul>
            <li>MobileFaceNet (ONNX)</li>
            <li>ONNX Runtime GPU</li>
            <li>FAISS GPU</li>
            <li>PyQt5</li>
        </ul>
        <p><b>Rendiment esperat:</b></p>
        <ul>
            <li>400-600 imatges/segon (RTX 2060)</li>
            <li>Cerca <100ms per 5M imatges</li>
        </ul>
        """

        QMessageBox.about(self, "Sobre", about_text)

    def _show_help(self):
        """Mostra diàleg d'ajuda."""
        help_text = """
        <h3>Guia d'ús</h3>

        <h4>1. Crear Dataset</h4>
        <ol>
            <li>Selecciona la carpeta amb imatges de rostres</li>
            <li>Configura paràmetres avançats si cal (opcional)</li>
            <li>Clica "INICIAR PROCESSAMENT"</li>
            <li>Espera fins que es completi</li>
        </ol>

        <h4>2. Reconeixement Facial</h4>
        <ol>
            <li>Carrega el dataset d'embeddings (facial_embeddings.h5)</li>
            <li>Carrega una imatge de consulta</li>
            <li>Selecciona el nombre de resultats (top-K)</li>
            <li>Clica "CERCAR"</li>
        </ol>

        <h4>Configuració Recomanada per RTX 2060:</h4>
        <ul>
            <li>Batch Size: 128-256</li>
            <li>Threads I/O: 6</li>
            <li>FP16: Activat</li>
            <li>Auto-tuning: Activat</li>
        </ul>

        <h4>Troubleshooting:</h4>
        <p><b>OOM Error:</b> Redueix el batch size</p>
        <p><b>CUDA not found:</b> Assegura't que tens els drivers NVIDIA i CUDA instal·lats</p>
        <p><b>Model loading error:</b> Verifica que el model ONNX està a models/mobilefacenet.onnx</p>
        """

        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Ajuda")
        msg_box.setTextFormat(Qt.RichText)
        msg_box.setText(help_text)
        msg_box.exec_()

    def closeEvent(self, event):
        """Gestiona l'event de tancament."""
        # Pregunta confirmació si hi ha processament en curs
        if hasattr(self.dataset_tab, 'creator_thread') and \
           self.dataset_tab.creator_thread and \
           self.dataset_tab.creator_thread.isRunning():

            reply = QMessageBox.question(
                self,
                "Confirmar sortida",
                "Hi ha un processament en curs. Segur que vols sortir?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )

            if reply == QMessageBox.No:
                event.ignore()
                return

            # Atura processament
            if self.dataset_tab.creator:
                self.dataset_tab.creator.stop()

        event.accept()
        logger.info("Aplicació tancada")
