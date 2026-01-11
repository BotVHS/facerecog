"""
Multi-threaded image loader optimitzat per màxima velocitat.
Utilitza ThreadPoolExecutor per lectura concurrent i preprocessament en batch.
"""

import os
import cv2
import numpy as np
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import logging

logger = logging.getLogger(__name__)


class ImageLoader:
    """Carregador d'imatges multi-threaded amb preprocessament."""

    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp'}
    TARGET_SIZE = (112, 112)  # MobileFaceNet input size

    def __init__(self, num_workers: int = 6):
        """
        Inicialitza el carregador d'imatges.

        Args:
            num_workers: Nombre de threads per lectura concurrent
        """
        self.num_workers = num_workers
        self.executor = ThreadPoolExecutor(max_workers=num_workers)

    def scan_images(self, root_dir: str) -> List[str]:
        """
        Escaneja recursivament una carpeta per trobar totes les imatges.

        Args:
            root_dir: Directori arrel on cercar

        Returns:
            Llista de paths absoluts a les imatges trobades
        """
        image_paths = []

        logger.info(f"Escanejant imatges a {root_dir}...")

        for root, _, files in os.walk(root_dir):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in self.SUPPORTED_FORMATS:
                    image_paths.append(os.path.join(root, file))

        logger.info(f"Trobades {len(image_paths)} imatges")
        return image_paths

    def load_and_preprocess_image(self, image_path: str) -> Optional[Tuple[np.ndarray, str]]:
        """
        Carrega i preprocessa una sola imatge.

        Args:
            image_path: Path a la imatge

        Returns:
            Tuple (imatge preprocessada, path) o None si hi ha error
        """
        try:
            # Carrega imatge amb OpenCV (més ràpid que PIL)
            img = cv2.imread(image_path)

            if img is None:
                logger.warning(f"No s'ha pogut carregar: {image_path}")
                return None

            # Resize a mida esperada pel model
            img = cv2.resize(img, self.TARGET_SIZE, interpolation=cv2.INTER_LINEAR)

            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Normalitza [0, 255] -> [-1, 1]
            img = (img.astype(np.float32) - 127.5) / 128.0

            # Transpose to CHW format (Canal, Height, Width)
            img = np.transpose(img, (2, 0, 1))

            return (img, image_path)

        except Exception as e:
            logger.error(f"Error processant {image_path}: {str(e)}")
            return None

    def load_batch(self, image_paths: List[str], start_idx: int, batch_size: int) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Carrega un batch d'imatges de forma concurrent.

        Args:
            image_paths: Llista completa de paths
            start_idx: Índex inicial del batch
            batch_size: Mida del batch

        Returns:
            Tuple (batch array, paths exitosos, paths amb error)
        """
        end_idx = min(start_idx + batch_size, len(image_paths))
        batch_paths = image_paths[start_idx:end_idx]

        # Carrega imatges en paral·lel amb ThreadPoolExecutor
        results = list(self.executor.map(self.load_and_preprocess_image, batch_paths))

        # Separa resultats exitosos dels errors
        successful = [r for r in results if r is not None]

        if not successful:
            return np.array([]), [], batch_paths

        # Crea batch array i llistes de paths
        images, valid_paths = zip(*successful)
        batch_array = np.stack(images, axis=0)

        # Identifica paths amb error
        valid_paths_set = set(valid_paths)
        error_paths = [p for p in batch_paths if p not in valid_paths_set]

        return batch_array, list(valid_paths), error_paths

    def preload_batches(self, image_paths: List[str], batch_size: int,
                       num_prefetch: int = 2) -> Queue:
        """
        Precarrega batches en background per prefetching.

        Args:
            image_paths: Llista de paths a processar
            batch_size: Mida de cada batch
            num_prefetch: Nombre de batches a precarregar

        Returns:
            Queue amb batches precarregats
        """
        queue = Queue(maxsize=num_prefetch)

        def producer():
            for i in range(0, len(image_paths), batch_size):
                batch = self.load_batch(image_paths, i, batch_size)
                queue.put(batch)
            queue.put(None)  # Sentinel value

        # Inicia producer thread
        from threading import Thread
        thread = Thread(target=producer, daemon=True)
        thread.start()

        return queue

    def shutdown(self):
        """Tanca el ThreadPoolExecutor."""
        self.executor.shutdown(wait=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()


def estimate_optimal_batch_size(model_processor, image_loader: ImageLoader,
                               sample_paths: List[str],
                               test_sizes: List[int] = [64, 128, 256]) -> int:
    """
    Auto-tuning: prova diferents batch sizes i retorna el més ràpid.

    Args:
        model_processor: Instància del processador GPU
        image_loader: Instància del carregador d'imatges
        sample_paths: Paths de mostra per testejar
        test_sizes: Batch sizes a provar

    Returns:
        Batch size òptim
    """
    import time

    logger.info("Iniciant auto-tuning de batch size...")

    best_size = test_sizes[0]
    best_speed = 0

    for batch_size in test_sizes:
        try:
            # Carrega un batch de prova
            batch, paths, _ = image_loader.load_batch(sample_paths, 0, batch_size)

            if len(batch) == 0:
                continue

            # Processa i mesura temps
            start = time.time()
            model_processor.process_batch(batch)
            elapsed = time.time() - start

            speed = len(batch) / elapsed
            logger.info(f"Batch size {batch_size}: {speed:.1f} imgs/sec")

            if speed > best_speed:
                best_speed = speed
                best_size = batch_size

        except Exception as e:
            logger.warning(f"Batch size {batch_size} va fallar: {str(e)}")
            continue

    logger.info(f"Batch size òptim: {best_size} ({best_speed:.1f} imgs/sec)")
    return best_size
