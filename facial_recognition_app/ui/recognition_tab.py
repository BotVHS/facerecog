"""
Pestanya de reconeixement facial per la UI.
Carrega dataset i cerca similituds.
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QScrollArea, QGridLayout, QSpinBox, QFileDialog, QFrame, QGroupBox
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QPixmap, QFont
import os
import logging

from face_recognizer import FaceRecognizer

logger = logging.getLogger(__name__)


class RecognitionThread(QThread):
    """Thread per executar cerca en background."""

    # Signals
    search_completed = pyqtSignal(dict)  # results
    error_occurred = pyqtSignal(str)  # error message

    def __init__(self, recognizer: FaceRecognizer, query_image: str, top_k: int):
        """
        Inicialitza thread.

        Args:
            recognizer: Instància de FaceRecognizer
            query_image: Path a la imatge de consulta
            top_k: Nombre de resultats
        """
        super().__init__()
        self.recognizer = recognizer
        self.query_image = query_image
        self.top_k = top_k

    def run(self):
        """Executa la cerca."""
        try:
            results = self.recognizer.search(self.query_image, self.top_k)

            if results:
                self.search_completed.emit(results)
            else:
                self.error_occurred.emit("No s'han pogut obtenir resultats")

        except Exception as e:
            logger.error(f"Error en thread de cerca: {str(e)}")
            self.error_occurred.emit(str(e))


class LoadDatasetThread(QThread):
    """Thread per carregar dataset en background."""

    # Signals
    loading_completed = pyqtSignal(bool)  # success
    error_occurred = pyqtSignal(str)  # error message
    progress_message = pyqtSignal(str)  # progress message

    def __init__(self, recognizer: FaceRecognizer, dataset_path: str, use_gpu: bool):
        """
        Inicialitza thread.

        Args:
            recognizer: Instància de FaceRecognizer
            dataset_path: Path al dataset HDF5
            use_gpu: Utilitzar GPU per l'índex
        """
        super().__init__()
        self.recognizer = recognizer
        self.dataset_path = dataset_path
        self.use_gpu = use_gpu

    def run(self):
        """Carrega el dataset."""
        try:
            self.progress_message.emit("Carregant dataset...")
            self.recognizer.load_dataset(self.dataset_path, use_gpu=self.use_gpu)
            self.progress_message.emit("Dataset carregat correctament!")
            self.loading_completed.emit(True)

        except Exception as e:
            logger.error(f"Error carregant dataset: {str(e)}")
            self.error_occurred.emit(str(e))
            self.loading_completed.emit(False)


class ResultWidget(QFrame):
    """Widget per mostrar un resultat de cerca."""

    def __init__(self, rank: int, image_path: str, score: float):
        """
        Inicialitza widget de resultat.

        Args:
            rank: Posició en el ranking
            image_path: Path a la imatge
            score: Score de similitud
        """
        super().__init__()
        self.setFrameStyle(QFrame.Box | QFrame.Raised)
        self.setLineWidth(2)

        layout = QVBoxLayout()

        # Rank
        rank_label = QLabel(f"#{rank}")
        rank_label.setFont(QFont("Arial", 12, QFont.Bold))
        rank_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(rank_label)

        # Imatge
        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            # Resize mantenint aspect ratio
            pixmap = pixmap.scaled(150, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            image_label = QLabel()
            image_label.setPixmap(pixmap)
            image_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(image_label)
        else:
            layout.addWidget(QLabel("Error carregant imatge"))

        # Score
        score_label = QLabel(f"Score: {score:.4f}")
        score_label.setFont(QFont("Arial", 10))
        score_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(score_label)

        # Path (truncat)
        filename = os.path.basename(image_path)
        path_label = QLabel(filename)
        path_label.setFont(QFont("Arial", 8))
        path_label.setAlignment(Qt.AlignCenter)
        path_label.setToolTip(image_path)  # Full path en tooltip
        path_label.setWordWrap(True)
        layout.addWidget(path_label)

        self.setLayout(layout)


class RecognitionTab(QWidget):
    """Pestanya per reconeixement facial."""

    def __init__(self, model_path: str):
        """
        Inicialitza la pestanya.

        Args:
            model_path: Path al model ONNX
        """
        super().__init__()
        self.model_path = model_path
        self.recognizer = None
        self.query_image_path = None
        self.dataset_loaded = False

        self._init_ui()
        self._init_recognizer()

    def _init_recognizer(self):
        """Inicialitza el recognizer."""
        try:
            self.recognizer = FaceRecognizer(self.model_path)
            logger.info("FaceRecognizer inicialitzat")
        except Exception as e:
            logger.error(f"Error inicialitzant recognizer: {str(e)}")

    def _init_ui(self):
        """Inicialitza la interfície."""
        layout = QVBoxLayout()

        # Controls principals
        controls_layout = QVBoxLayout()

        # Carrega dataset
        dataset_layout = QHBoxLayout()
        self.load_dataset_btn = QPushButton("Carregar Dataset d'Embeddings")
        self.load_dataset_btn.clicked.connect(self._load_dataset)
        dataset_layout.addWidget(self.load_dataset_btn)

        self.dataset_status_label = QLabel("Dataset: No carregat")
        self.dataset_status_label.setFont(QFont("Arial", 10, QFont.Bold))
        dataset_layout.addWidget(self.dataset_status_label)

        controls_layout.addLayout(dataset_layout)

        # Carrega imatge de consulta
        query_layout = QHBoxLayout()
        self.load_query_btn = QPushButton("Carregar Imatge de Consulta")
        self.load_query_btn.clicked.connect(self._load_query_image)
        query_layout.addWidget(self.load_query_btn)

        self.query_status_label = QLabel("Imatge: No carregada")
        query_status_label_font = QFont("Arial", 10)
        self.query_status_label.setFont(query_status_label_font)
        query_layout.addWidget(self.query_status_label)

        controls_layout.addLayout(query_layout)

        # Preview de la imatge de consulta
        self.query_preview = QLabel("Preview de la imatge apareixerà aquí")
        self.query_preview.setAlignment(Qt.AlignCenter)
        self.query_preview.setMinimumHeight(200)
        self.query_preview.setFrameStyle(QFrame.Box)
        controls_layout.addWidget(self.query_preview)

        # Top-K selector
        topk_layout = QHBoxLayout()
        topk_layout.addWidget(QLabel("Top-K resultats:"))

        self.topk_spin = QSpinBox()
        self.topk_spin.setRange(1, 100)
        self.topk_spin.setValue(10)
        topk_layout.addWidget(self.topk_spin)
        topk_layout.addStretch()

        controls_layout.addLayout(topk_layout)

        # Botó de cerca
        self.search_btn = QPushButton("CERCAR")
        self.search_btn.setStyleSheet("QPushButton { font-size: 14pt; font-weight: bold; padding: 10px; }")
        self.search_btn.setEnabled(False)
        self.search_btn.clicked.connect(self._start_search)
        controls_layout.addWidget(self.search_btn)

        # Temps de cerca
        self.search_time_label = QLabel("Temps de cerca: --")
        self.search_time_label.setFont(QFont("Arial", 12))
        self.search_time_label.setAlignment(Qt.AlignCenter)
        controls_layout.addWidget(self.search_time_label)

        layout.addLayout(controls_layout)

        # Àrea de resultats
        results_group = QGroupBox("Resultats")
        results_layout = QVBoxLayout()

        # Scroll area per resultats
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(400)

        # Widget contenidor per resultats
        self.results_widget = QWidget()
        self.results_grid = QGridLayout()
        self.results_widget.setLayout(self.results_grid)

        scroll.setWidget(self.results_widget)
        results_layout.addWidget(scroll)

        results_group.setLayout(results_layout)
        layout.addWidget(results_group)

        self.setLayout(layout)

    def _load_dataset(self):
        """Carrega dataset d'embeddings."""
        dataset_path, _ = QFileDialog.getOpenFileName(
            self,
            "Selecciona dataset d'embeddings",
            "",
            "HDF5 Files (*.h5 *.hdf5)"
        )

        if not dataset_path:
            return

        # Carrega en thread
        self.dataset_status_label.setText("Carregant dataset...")
        self.load_dataset_btn.setEnabled(False)

        self.load_thread = LoadDatasetThread(self.recognizer, dataset_path, use_gpu=True)
        self.load_thread.loading_completed.connect(self._dataset_loaded)
        self.load_thread.error_occurred.connect(self._dataset_error)
        self.load_thread.progress_message.connect(self._update_dataset_status)
        self.load_thread.start()

    def _dataset_loaded(self, success: bool):
        """Callback quan el dataset s'ha carregat."""
        self.load_dataset_btn.setEnabled(True)

        if success:
            self.dataset_loaded = True
            info = self.recognizer.get_dataset_info()
            self.dataset_status_label.setText(
                f"Dataset: {info.get('total_images', 0):,} imatges carregades"
            )
            self._update_search_button_state()
        else:
            self.dataset_status_label.setText("Dataset: Error carregant")

    def _dataset_error(self, error_msg: str):
        """Callback quan hi ha error carregant dataset."""
        self.dataset_status_label.setText(f"Error: {error_msg}")
        logger.error(error_msg)

    def _update_dataset_status(self, message: str):
        """Actualitza status de càrrega."""
        self.dataset_status_label.setText(message)

    def _load_query_image(self):
        """Carrega imatge de consulta."""
        image_path, _ = QFileDialog.getOpenFileName(
            self,
            "Selecciona imatge de consulta",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp)"
        )

        if not image_path:
            return

        self.query_image_path = image_path

        # Mostra preview
        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            pixmap = pixmap.scaled(300, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.query_preview.setPixmap(pixmap)

            filename = os.path.basename(image_path)
            self.query_status_label.setText(f"Imatge: {filename}")

            self._update_search_button_state()
        else:
            self.query_status_label.setText("Error carregant imatge")

    def _update_search_button_state(self):
        """Actualitza estat del botó de cerca."""
        can_search = self.dataset_loaded and self.query_image_path is not None
        self.search_btn.setEnabled(can_search)

    def _start_search(self):
        """Inicia cerca."""
        if not self.dataset_loaded or not self.query_image_path:
            return

        # Desactiva botó
        self.search_btn.setEnabled(False)
        self.search_time_label.setText("Cercant...")

        # Neteja resultats anteriors
        self._clear_results()

        # Inicia thread de cerca
        top_k = self.topk_spin.value()
        self.search_thread = RecognitionThread(
            self.recognizer,
            self.query_image_path,
            top_k
        )
        self.search_thread.search_completed.connect(self._display_results)
        self.search_thread.error_occurred.connect(self._search_error)
        self.search_thread.start()

    def _display_results(self, results: dict):
        """Mostra resultats de cerca."""
        # Actualitza temps
        search_time = results.get('search_time', 0)
        self.search_time_label.setText(f"Temps de cerca: {search_time*1000:.1f}ms")

        # Mostra resultats en grid (4 columnes)
        matches = results.get('matches', [])
        cols = 4

        for idx, match in enumerate(matches):
            row = idx // cols
            col = idx % cols

            result_widget = ResultWidget(
                rank=match['rank'],
                image_path=match['image_path'],
                score=match['similarity_score']
            )

            self.results_grid.addWidget(result_widget, row, col)

        # Reactiva botó
        self.search_btn.setEnabled(True)

        logger.info(f"Mostrats {len(matches)} resultats")

    def _clear_results(self):
        """Neteja resultats anteriors."""
        while self.results_grid.count():
            item = self.results_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _search_error(self, error_msg: str):
        """Callback quan hi ha error en cerca."""
        self.search_time_label.setText(f"Error: {error_msg}")
        self.search_btn.setEnabled(True)
        logger.error(error_msg)
