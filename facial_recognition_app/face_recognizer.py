"""
Sistema de reconeixement facial amb cerca ràpida FAISS.
Carrega dataset d'embeddings i cerca similituds.
"""

import h5py
import numpy as np
import cv2
from typing import List, Tuple, Optional
import time
import logging

from utils.gpu_processor import GPUProcessor
from utils.faiss_index import FAISSIndex
from utils.image_loader import ImageLoader

logger = logging.getLogger(__name__)


class FaceRecognizer:
    """Sistema de reconeixement facial amb cerca FAISS."""

    def __init__(self, model_path: str):
        """
        Inicialitza el recognizer.

        Args:
            model_path: Path al model ONNX
        """
        self.model_path = model_path
        self.gpu_processor = None
        self.faiss_index = None
        self.dataset_loaded = False

        # Metadata del dataset
        self.dataset_info = {}

        # Inicialitza processador GPU
        self._init_gpu_processor()

    def _init_gpu_processor(self):
        """Inicialitza el processador GPU."""
        logger.info("Inicialitzant processador GPU per reconeixement...")
        self.gpu_processor = GPUProcessor(self.model_path, use_fp16=False)
        logger.info("Processador GPU inicialitzat")

    def load_dataset(self, dataset_path: str, use_gpu: bool = True):
        """
        Carrega dataset d'embeddings i crea índex FAISS.

        Args:
            dataset_path: Path al fitxer HDF5 amb embeddings
            use_gpu: Utilitzar GPU per l'índex FAISS
        """
        logger.info(f"Carregant dataset des de {dataset_path}...")

        start_time = time.time()

        # Llegeix HDF5
        with h5py.File(dataset_path, 'r') as f:
            # Carrega embeddings
            embeddings = f['embeddings'][:]
            image_paths = f['image_paths'][:]

            # Converteix bytes a strings si cal
            if isinstance(image_paths[0], bytes):
                image_paths = [p.decode('utf-8') for p in image_paths]
            else:
                image_paths = list(image_paths)

            # Metadata
            self.dataset_info = {
                'total_images': len(embeddings),
                'embedding_dim': embeddings.shape[1],
                'model_path': f.attrs.get('model_path', 'unknown'),
                'created_at': f.attrs.get('created_at', 'unknown'),
            }

        logger.info(f"Dataset carregat: {self.dataset_info['total_images']} imatges")

        # Crea índex FAISS
        logger.info("Creant índex FAISS...")
        self.faiss_index = FAISSIndex(
            dimension=self.dataset_info['embedding_dim'],
            use_gpu=use_gpu
        )

        # Afegeix embeddings a l'índex
        self.faiss_index.add(embeddings, image_paths)

        elapsed = time.time() - start_time
        logger.info(f"Índex FAISS creat en {elapsed:.2f}s")

        self.dataset_loaded = True

    def load_index(self, index_path: str, paths_path: str, use_gpu: bool = True):
        """
        Carrega índex FAISS pre-construït.

        Args:
            index_path: Path a l'índex FAISS
            paths_path: Path als image paths
            use_gpu: Utilitzar GPU
        """
        logger.info(f"Carregant índex FAISS des de {index_path}...")

        self.faiss_index = FAISSIndex(use_gpu=use_gpu)
        self.faiss_index.load(index_path, paths_path)

        self.dataset_info = {
            'total_images': len(self.faiss_index),
            'embedding_dim': self.faiss_index.dimension,
        }

        self.dataset_loaded = True
        logger.info(f"Índex carregat: {self.dataset_info['total_images']} vectors")

    def save_index(self, index_path: str, paths_path: str):
        """
        Guarda l'índex FAISS actual.

        Args:
            index_path: Path per guardar l'índex
            paths_path: Path per guardar els paths
        """
        if not self.dataset_loaded or self.faiss_index is None:
            raise RuntimeError("No hi ha índex carregat per guardar")

        logger.info(f"Guardant índex FAISS a {index_path}...")
        self.faiss_index.save(index_path, paths_path)
        logger.info("Índex guardat correctament")

    def process_query_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Processa una imatge de consulta i genera el seu embedding.

        Args:
            image_path: Path a la imatge

        Returns:
            Embedding (512,) o None si hi ha error
        """
        try:
            # Carrega i preprocessa imatge
            img = cv2.imread(image_path)

            if img is None:
                logger.error(f"No s'ha pogut carregar: {image_path}")
                return None

            # Resize a mida esperada
            img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_LINEAR)

            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Normalitza
            img = (img.astype(np.float32) - 127.5) / 128.0

            # Transpose to CHW
            img = np.transpose(img, (2, 0, 1))

            # Genera embedding
            embedding = self.gpu_processor.process_single(img)

            return embedding

        except Exception as e:
            logger.error(f"Error processant imatge de consulta: {str(e)}")
            return None

    def search(self, query_image_path: str, top_k: int = 50) -> Optional[dict]:
        """
        Cerca les imatges més similars a la consulta.

        Args:
            query_image_path: Path a la imatge de consulta
            top_k: Nombre de resultats a retornar

        Returns:
            Diccionari amb resultats o None si hi ha error
        """
        if not self.dataset_loaded:
            logger.error("Dataset no carregat!")
            return None

        logger.info(f"Cercant similituds per {query_image_path}...")

        start_time = time.time()

        # Genera embedding de la consulta
        query_embedding = self.process_query_image(query_image_path)

        if query_embedding is None:
            return None

        # Cerca en l'índex FAISS
        distances, indices, paths = self.faiss_index.search(query_embedding, top_k)

        search_time = time.time() - start_time

        # Prepara resultats
        results = {
            'query_image': query_image_path,
            'search_time': search_time,
            'num_results': len(paths),
            'matches': []
        }

        for i, (dist, idx, path) in enumerate(zip(distances, indices, paths)):
            results['matches'].append({
                'rank': i + 1,
                'image_path': path,
                'similarity_score': float(dist),  # Ja és cosine similarity
                'index': int(idx),
            })

        logger.info(f"Cerca completada en {search_time*1000:.1f}ms, {len(paths)} resultats")

        return results

    def search_by_embedding(self, embedding: np.ndarray, top_k: int = 50) -> dict:
        """
        Cerca directament amb un embedding.

        Args:
            embedding: Vector (512,)
            top_k: Nombre de resultats

        Returns:
            Diccionari amb resultats
        """
        if not self.dataset_loaded:
            raise RuntimeError("Dataset no carregat!")

        start_time = time.time()

        # Cerca en l'índex
        distances, indices, paths = self.faiss_index.search(embedding, top_k)

        search_time = time.time() - start_time

        results = {
            'search_time': search_time,
            'num_results': len(paths),
            'matches': []
        }

        for i, (dist, idx, path) in enumerate(zip(distances, indices, paths)):
            results['matches'].append({
                'rank': i + 1,
                'image_path': path,
                'similarity_score': float(dist),
                'index': int(idx),
            })

        return results

    def get_dataset_info(self) -> dict:
        """Retorna informació del dataset carregat."""
        if not self.dataset_loaded:
            return {'loaded': False}

        info = self.dataset_info.copy()
        info['loaded'] = True

        if self.faiss_index:
            info.update(self.faiss_index.get_stats())

        return info

    def batch_search(self, query_image_paths: List[str], top_k: int = 50) -> List[dict]:
        """
        Cerca múltiples imatges en batch.

        Args:
            query_image_paths: Llista de paths de consulta
            top_k: Nombre de resultats per consulta

        Returns:
            Llista de diccionaris amb resultats
        """
        results = []

        logger.info(f"Cercant {len(query_image_paths)} imatges en batch...")

        for query_path in query_image_paths:
            result = self.search(query_path, top_k)
            if result:
                results.append(result)

        return results

    def verify_match(self, image1_path: str, image2_path: str, threshold: float = 0.5) -> dict:
        """
        Verifica si dues imatges corresponen a la mateixa persona.

        Args:
            image1_path: Path a la primera imatge
            image2_path: Path a la segona imatge
            threshold: Threshold de similitud per considerar match

        Returns:
            Diccionari amb resultat de verificació
        """
        logger.info(f"Verificant match entre {image1_path} i {image2_path}...")

        # Genera embeddings
        emb1 = self.process_query_image(image1_path)
        emb2 = self.process_query_image(image2_path)

        if emb1 is None or emb2 is None:
            return {'success': False, 'error': 'No s\'han pogut processar les imatges'}

        # Calcula similitud (cosine similarity = dot product amb vectors normalitzats)
        similarity = float(np.dot(emb1, emb2))

        is_match = similarity >= threshold

        return {
            'success': True,
            'similarity': similarity,
            'is_match': is_match,
            'threshold': threshold,
            'image1': image1_path,
            'image2': image2_path,
        }
