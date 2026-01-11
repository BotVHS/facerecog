"""
Entry point de l'aplicació de reconeixement facial.
Configura logging i inicia la UI.
"""

import sys
import os
import logging
from datetime import datetime
from PyQt5.QtWidgets import QApplication

from ui.main_window import MainWindow


def setup_logging():
    """Configura el sistema de logging."""
    # Crea directori de logs si no existeix
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    # Nom del fitxer de log amb timestamp
    log_filename = os.path.join(
        log_dir,
        f"facerecog_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )

    # Configura format
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'

    # Configura logging a fitxer i consola
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("Aplicació de Reconeixement Facial Ultra-Ràpid")
    logger.info("=" * 80)
    logger.info(f"Log file: {log_filename}")

    return logger


def check_model_exists(model_path: str) -> bool:
    """
    Verifica si el model existeix.

    Args:
        model_path: Path al model

    Returns:
        True si existeix, False altrament
    """
    return os.path.exists(model_path)


def main():
    """Funció principal."""
    # Configura logging
    logger = setup_logging()

    # Path al model ONNX
    model_path = os.path.join("models", "mobilefacenet.onnx")

    # Verifica model (advertència, no error fatal)
    if not check_model_exists(model_path):
        logger.warning(f"Model ONNX no trobat a {model_path}")
        logger.warning("Si us plau, descarrega el model MobileFaceNet i col·loca'l a:")
        logger.warning("  models/mobilefacenet.onnx")
        logger.warning("")
        logger.warning("Pots descarregar-lo des de:")
        logger.warning("  - https://github.com/onnx/models/tree/main/vision/body_analysis/arcface")
        logger.warning("  - https://github.com/deepinsight/insightface")
        logger.warning("")
        logger.warning("L'aplicació s'iniciarà igualment, però no funcionarà fins que tinguis el model.")
        logger.warning("")

    # Crea aplicació Qt
    app = QApplication(sys.argv)
    app.setApplicationName("Face Recognition Ultra Fast")
    app.setOrganizationName("FaceRecog")

    # Configura estil
    app.setStyle("Fusion")

    # Crea finestra principal
    logger.info("Inicialitzant finestra principal...")
    window = MainWindow(model_path)
    window.show()

    logger.info("Aplicació iniciada correctament")

    # Executa aplicació
    exit_code = app.exec_()

    logger.info(f"Aplicació finalitzada amb codi {exit_code}")
    logger.info("=" * 80)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
