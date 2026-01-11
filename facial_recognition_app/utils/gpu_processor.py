"""
Processador GPU optimitzat per batch inference amb ONNX Runtime.
Suporta FP16 mixed precision per màxima velocitat.
"""

import numpy as np
import onnxruntime as ort
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class GPUProcessor:
    """Processador d'embeddings facials amb ONNX Runtime GPU."""

    def __init__(self, model_path: str, use_fp16: bool = False):
        """
        Inicialitza el processador GPU.

        Args:
            model_path: Path al model ONNX
            use_fp16: Activar mixed precision (FP16) per més velocitat
        """
        self.model_path = model_path
        self.use_fp16 = use_fp16
        self.session = None
        self.input_name = None
        self.output_name = None

        self._load_model()

    def _load_model(self):
        """Carrega el model ONNX amb configuració GPU optimitzada."""
        logger.info(f"Carregant model ONNX des de {self.model_path}...")

        # Configura opcions de sessió per màxima velocitat
        sess_options = ort.SessionOptions()

        # Optimitzacions de graf
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Execució en paral·lel dins del graf
        sess_options.intra_op_num_threads = 2
        sess_options.inter_op_num_threads = 2

        # Enable memory pattern optimization
        sess_options.enable_mem_pattern = True
        sess_options.enable_cpu_mem_arena = True

        # Providers de CUDA amb configuració optimitzada
        providers = []

        # Configuració GPU
        cuda_provider_options = {
            'device_id': 0,
            'arena_extend_strategy': 'kSameAsRequested',  # Preallocate memory
            'gpu_mem_limit': 5 * 1024 * 1024 * 1024,  # 5GB limit per VRAM
            'cudnn_conv_algo_search': 'EXHAUSTIVE',  # Troba millor algoritme
            'do_copy_in_default_stream': True,
        }

        providers.append(('CUDAExecutionProvider', cuda_provider_options))
        providers.append('CPUExecutionProvider')  # Fallback

        # Crea sessió
        try:
            self.session = ort.InferenceSession(
                self.model_path,
                sess_options=sess_options,
                providers=providers
            )

            # Verifica que s'està utilitzant GPU
            provider = self.session.get_providers()[0]
            if provider != 'CUDAExecutionProvider':
                logger.warning(f"GPU no disponible! Utilitzant: {provider}")
            else:
                logger.info("Model carregat correctament en GPU")

            # Obté noms d'input/output
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name

            logger.info(f"Input: {self.input_name}, Output: {self.output_name}")

        except Exception as e:
            logger.error(f"Error carregant model: {str(e)}")
            raise

    def process_batch(self, batch: np.ndarray) -> np.ndarray:
        """
        Processa un batch d'imatges i retorna embeddings.

        Args:
            batch: Array de forma (N, 3, 112, 112) amb imatges normalitzades

        Returns:
            Array de forma (N, 512) amb embeddings
        """
        if self.session is None:
            raise RuntimeError("Model no carregat")

        # Converteix a FP16 si està activat
        if self.use_fp16:
            batch = batch.astype(np.float16)
        else:
            batch = batch.astype(np.float32)

        try:
            # Inferència en GPU
            outputs = self.session.run(
                [self.output_name],
                {self.input_name: batch}
            )

            embeddings = outputs[0]

            # Normalitza embeddings (L2 normalization)
            embeddings = self._normalize_embeddings(embeddings)

            return embeddings

        except Exception as e:
            logger.error(f"Error durant inferència: {str(e)}")
            raise

    def process_single(self, image: np.ndarray) -> np.ndarray:
        """
        Processa una sola imatge.

        Args:
            image: Array de forma (3, 112, 112)

        Returns:
            Embedding de forma (512,)
        """
        # Afegeix dimensió de batch
        batch = np.expand_dims(image, axis=0)
        embeddings = self.process_batch(batch)
        return embeddings[0]

    @staticmethod
    def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
        """
        Normalitza embeddings amb L2 norm.

        Args:
            embeddings: Array (N, D)

        Returns:
            Array normalitzat (N, D)
        """
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)  # Evita divisió per zero
        return embeddings / norms

    def warmup(self, num_iterations: int = 10, batch_size: int = 128):
        """
        Escalfa la GPU processant batches de prova.

        Args:
            num_iterations: Nombre d'iteracions de warming
            batch_size: Mida dels batches de prova
        """
        logger.info(f"Escalfant GPU amb {num_iterations} iteracions...")

        dummy_batch = np.random.randn(batch_size, 3, 112, 112).astype(np.float32)

        for i in range(num_iterations):
            self.process_batch(dummy_batch)

        logger.info("Warming completat")

    def get_model_info(self) -> dict:
        """
        Retorna informació sobre el model.

        Returns:
            Diccionari amb metadata del model
        """
        if self.session is None:
            return {}

        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()

        return {
            'input_name': inputs[0].name,
            'input_shape': inputs[0].shape,
            'input_type': inputs[0].type,
            'output_name': outputs[0].name,
            'output_shape': outputs[0].shape,
            'output_type': outputs[0].type,
            'provider': self.session.get_providers()[0],
            'fp16_enabled': self.use_fp16,
        }

    def __del__(self):
        """Neteja recursos."""
        if self.session:
            del self.session


class DynamicBatchProcessor(GPUProcessor):
    """
    Processador amb batch size dinàmic que s'ajusta automàticament
    en cas d'errors de memòria.
    """

    def __init__(self, model_path: str, use_fp16: bool = False,
                 initial_batch_size: int = 256):
        """
        Inicialitza processador dinàmic.

        Args:
            model_path: Path al model ONNX
            use_fp16: Activar FP16
            initial_batch_size: Batch size inicial
        """
        super().__init__(model_path, use_fp16)
        self.current_batch_size = initial_batch_size
        self.min_batch_size = 16

    def process_batch_dynamic(self, batch: np.ndarray) -> np.ndarray:
        """
        Processa batch amb ajust automàtic de mida en cas d'OOM.

        Args:
            batch: Input batch

        Returns:
            Embeddings
        """
        try:
            return self.process_batch(batch)

        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "OOM" in str(e):
                # Redueix batch size
                new_size = max(self.current_batch_size // 2, self.min_batch_size)

                if new_size < self.min_batch_size:
                    raise RuntimeError("No es pot reduir més el batch size")

                logger.warning(
                    f"OOM error! Reduint batch size: {self.current_batch_size} -> {new_size}"
                )
                self.current_batch_size = new_size

                # Divideix batch i processa en parts
                results = []
                for i in range(0, len(batch), new_size):
                    sub_batch = batch[i:i + new_size]
                    results.append(self.process_batch(sub_batch))

                return np.concatenate(results, axis=0)
            else:
                raise
