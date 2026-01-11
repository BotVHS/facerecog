"""
Gestió d'índex FAISS per cerca ràpida de similitud.
Utilitza GPU per màxima velocitat en cerques.
"""

import faiss
import numpy as np
from typing import List, Tuple, Optional
import logging
import os

logger = logging.getLogger(__name__)


class FAISSIndex:
    """Wrapper per índex FAISS optimitzat per GPU."""

    def __init__(self, dimension: int = 512, use_gpu: bool = True):
        """
        Inicialitza índex FAISS.

        Args:
            dimension: Dimensió dels embeddings (512 per MobileFaceNet)
            use_gpu: Utilitzar GPU per cerques
        """
        self.dimension = dimension
        self.use_gpu = use_gpu
        self.index = None
        self.gpu_index = None
        self.image_paths = []

        self._create_index()

    def _create_index(self):
        """Crea índex FAISS en CPU/GPU."""
        logger.info(f"Creant índex FAISS (dimensió={self.dimension}, GPU={self.use_gpu})...")

        # Crea índex base en CPU (IndexFlatIP per Inner Product = cosine similarity)
        # Utilitzem IP perquè els embeddings estan normalitzats
        self.index = faiss.IndexFlatIP(self.dimension)

        if self.use_gpu:
            try:
                # Transfereix índex a GPU
                res = faiss.StandardGpuResources()

                # Configuració GPU
                co = faiss.GpuClonerOptions()
                co.useFloat16 = True  # Utilitzar FP16 en GPU per més velocitat

                self.gpu_index = faiss.index_cpu_to_gpu(res, 0, self.index, co)
                logger.info("Índex FAISS creat en GPU")

            except Exception as e:
                logger.warning(f"No s'ha pogut crear índex GPU: {str(e)}")
                logger.warning("Utilitzant índex en CPU")
                self.use_gpu = False
                self.gpu_index = None
        else:
            logger.info("Índex FAISS creat en CPU")

    def add(self, embeddings: np.ndarray, paths: List[str]):
        """
        Afegeix embeddings a l'índex.

        Args:
            embeddings: Array (N, D) amb embeddings normalitzats
            paths: Llista de paths corresponents
        """
        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Dimensió incorrecta: {embeddings.shape[1]} vs {self.dimension}"
            )

        # Assegura que els embeddings són float32
        embeddings = embeddings.astype(np.float32)

        # Afegeix a índex (CPU o GPU)
        if self.use_gpu and self.gpu_index is not None:
            self.gpu_index.add(embeddings)
        else:
            self.index.add(embeddings)

        # Guarda paths
        self.image_paths.extend(paths)

        logger.debug(f"Afegits {len(paths)} embeddings. Total: {len(self.image_paths)}")

    def search(self, query_embedding: np.ndarray, k: int = 50) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Cerca els K vectors més similars.

        Args:
            query_embedding: Vector de consulta (D,) o (1, D)
            k: Nombre de resultats a retornar

        Returns:
            Tuple (distances, indices, paths)
        """
        # Assegura forma correcta
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        query_embedding = query_embedding.astype(np.float32)

        # Cerca en l'índex adequat
        if self.use_gpu and self.gpu_index is not None:
            distances, indices = self.gpu_index.search(query_embedding, k)
        else:
            distances, indices = self.index.search(query_embedding, k)

        # Obté paths corresponents
        result_paths = [self.image_paths[idx] for idx in indices[0] if idx < len(self.image_paths)]

        return distances[0], indices[0], result_paths

    def save(self, index_path: str, paths_path: str):
        """
        Guarda índex i paths a disk.

        Args:
            index_path: Path per guardar l'índex
            paths_path: Path per guardar els paths
        """
        logger.info(f"Guardant índex FAISS a {index_path}...")

        # Si està en GPU, primer passa a CPU
        if self.use_gpu and self.gpu_index is not None:
            cpu_index = faiss.index_gpu_to_cpu(self.gpu_index)
            faiss.write_index(cpu_index, index_path)
        else:
            faiss.write_index(self.index, index_path)

        # Guarda paths
        import pickle
        with open(paths_path, 'wb') as f:
            pickle.dump(self.image_paths, f)

        logger.info("Índex guardat correctament")

    def load(self, index_path: str, paths_path: str):
        """
        Carrega índex i paths des de disk.

        Args:
            index_path: Path de l'índex
            paths_path: Path dels paths
        """
        logger.info(f"Carregant índex FAISS des de {index_path}...")

        # Carrega índex des de CPU
        self.index = faiss.read_index(index_path)

        # Carrega paths
        import pickle
        with open(paths_path, 'rb') as f:
            self.image_paths = pickle.load(f)

        # Transfereix a GPU si cal
        if self.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                co = faiss.GpuClonerOptions()
                co.useFloat16 = True

                self.gpu_index = faiss.index_cpu_to_gpu(res, 0, self.index, co)
                logger.info("Índex carregat en GPU")

            except Exception as e:
                logger.warning(f"No s'ha pogut carregar índex en GPU: {str(e)}")
                self.use_gpu = False
                self.gpu_index = None

        logger.info(f"Índex carregat amb {len(self.image_paths)} vectors")

    def get_stats(self) -> dict:
        """
        Retorna estadístiques de l'índex.

        Returns:
            Diccionari amb stats
        """
        if self.use_gpu and self.gpu_index is not None:
            ntotal = self.gpu_index.ntotal
        else:
            ntotal = self.index.ntotal

        return {
            'num_vectors': ntotal,
            'num_paths': len(self.image_paths),
            'dimension': self.dimension,
            'use_gpu': self.use_gpu,
            'is_trained': True,  # IndexFlat no necessita training
        }

    def __len__(self):
        """Retorna nombre de vectors en l'índex."""
        if self.use_gpu and self.gpu_index is not None:
            return self.gpu_index.ntotal
        return self.index.ntotal


class BatchedFAISSIndex(FAISSIndex):
    """
    Índex FAISS que afegeix vectors en batches grans per eficiència.
    """

    def __init__(self, dimension: int = 512, use_gpu: bool = True,
                 batch_size: int = 100000):
        """
        Inicialitza índex amb batching.

        Args:
            dimension: Dimensió dels vectors
            use_gpu: Utilitzar GPU
            batch_size: Mida del batch abans d'afegir a l'índex
        """
        super().__init__(dimension, use_gpu)
        self.batch_size = batch_size
        self.pending_embeddings = []
        self.pending_paths = []

    def add(self, embeddings: np.ndarray, paths: List[str]):
        """
        Afegeix embeddings al batch pendent.

        Args:
            embeddings: Array (N, D)
            paths: Llista de paths
        """
        self.pending_embeddings.append(embeddings)
        self.pending_paths.extend(paths)

        # Si el batch està ple, flush
        total_pending = sum(e.shape[0] for e in self.pending_embeddings)
        if total_pending >= self.batch_size:
            self.flush()

    def flush(self):
        """Afegeix tots els embeddings pendents a l'índex."""
        if not self.pending_embeddings:
            return

        # Concatena tots els embeddings pendents
        all_embeddings = np.vstack(self.pending_embeddings)

        # Crida al mètode base
        super().add(all_embeddings, self.pending_paths)

        # Neteja buffers
        self.pending_embeddings = []
        self.pending_paths = []

        logger.info(f"Flush: {len(all_embeddings)} vectors afegits a l'índex")

    def save(self, index_path: str, paths_path: str):
        """Guarda índex, fent flush primer."""
        self.flush()
        super().save(index_path, paths_path)
