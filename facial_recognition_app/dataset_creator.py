"""
Lògica de creació de dataset d'embeddings facials.
Pipeline multi-threaded optimitzat per màxima velocitat.
"""

import os
import time
import h5py
import numpy as np
from typing import Optional, Callable
from datetime import datetime
import logging
import hashlib
import json
from tqdm import tqdm

from utils.image_loader import ImageLoader, estimate_optimal_batch_size
from utils.gpu_processor import GPUProcessor, DynamicBatchProcessor

logger = logging.getLogger(__name__)


class DatasetCreator:
    """Creador de dataset d'embeddings facials amb pipeline optimitzat."""

    def __init__(self, model_path: str, output_path: str = "facial_embeddings.h5"):
        """
        Inicialitza el creador de dataset.

        Args:
            model_path: Path al model ONNX
            output_path: Path de sortida per l'HDF5
        """
        self.model_path = model_path
        self.output_path = output_path
        self.checkpoint_path = output_path.replace('.h5', '_checkpoint.json')

        # Components
        self.image_loader = None
        self.gpu_processor = None

        # Configuració
        self.batch_size = 128
        self.num_workers = 6
        self.use_fp16 = False
        self.checkpoint_interval = 250000
        self.skip_duplicates = False

        # Estat
        self.total_images = 0
        self.processed_images = 0
        self.error_count = 0
        self.start_time = None
        self.paused = False
        self.stopped = False

        # Callbacks per UI
        self.progress_callback = None
        self.stats_callback = None

        # Buffers en memòria
        self.embeddings_buffer = []
        self.paths_buffer = []
        self.buffer_size = 500000  # Flush cada 500K imatges

        # Tracking de duplicats
        self.seen_hashes = set() if self.skip_duplicates else None

    def configure(self, batch_size: int = 128, num_workers: int = 6,
                 use_fp16: bool = False, skip_duplicates: bool = False):
        """
        Configura paràmetres del processament.

        Args:
            batch_size: Mida del batch per GPU
            num_workers: Nombre de threads I/O
            use_fp16: Activar mixed precision
            skip_duplicates: Saltar imatges duplicades (més lent)
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_fp16 = use_fp16
        self.skip_duplicates = skip_duplicates

        if skip_duplicates:
            self.seen_hashes = set()

        logger.info(f"Configuració: batch_size={batch_size}, workers={num_workers}, "
                   f"fp16={use_fp16}, skip_dup={skip_duplicates}")

    def set_callbacks(self, progress_callback: Optional[Callable] = None,
                     stats_callback: Optional[Callable] = None):
        """
        Estableix callbacks per actualitzar UI.

        Args:
            progress_callback: Funció(processed, total, percentage)
            stats_callback: Funció(stats_dict)
        """
        self.progress_callback = progress_callback
        self.stats_callback = stats_callback

    def _compute_image_hash(self, image_path: str) -> Optional[str]:
        """Calcula MD5 hash d'una imatge per detectar duplicats."""
        try:
            with open(image_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except:
            return None

    def _load_checkpoint(self) -> dict:
        """Carrega checkpoint si existeix."""
        if os.path.exists(self.checkpoint_path):
            try:
                with open(self.checkpoint_path, 'r') as f:
                    checkpoint = json.load(f)
                logger.info(f"Checkpoint carregat: {checkpoint['processed_images']} imatges")
                return checkpoint
            except Exception as e:
                logger.warning(f"Error carregant checkpoint: {str(e)}")

        return {'processed_images': 0, 'seen_hashes': []}

    def _save_checkpoint(self):
        """Guarda checkpoint actual."""
        checkpoint = {
            'processed_images': self.processed_images,
            'total_images': self.total_images,
            'error_count': self.error_count,
            'timestamp': datetime.now().isoformat(),
            'seen_hashes': list(self.seen_hashes) if self.seen_hashes else []
        }

        try:
            with open(self.checkpoint_path, 'w') as f:
                json.dump(checkpoint, f)
            logger.info(f"Checkpoint guardat: {self.processed_images} imatges")
        except Exception as e:
            logger.error(f"Error guardant checkpoint: {str(e)}")

    def _flush_to_hdf5(self, final: bool = False):
        """
        Escriu buffers a HDF5.

        Args:
            final: Si és el flush final
        """
        if not self.embeddings_buffer:
            return

        logger.info(f"Fent flush de {len(self.embeddings_buffer)} embeddings a HDF5...")

        # Concatena buffers
        all_embeddings = np.vstack(self.embeddings_buffer)
        all_paths = self.paths_buffer

        # Escriu a HDF5
        mode = 'a' if os.path.exists(self.output_path) else 'w'

        with h5py.File(self.output_path, mode) as f:
            if 'embeddings' not in f:
                # Primera escriptura: crea datasets
                f.create_dataset(
                    'embeddings',
                    data=all_embeddings,
                    maxshape=(None, 512),
                    chunks=(10000, 512),
                    compression='gzip',
                    compression_opts=1
                )

                # Paths com variable-length strings
                dt = h5py.special_dtype(vlen=str)
                f.create_dataset(
                    'image_paths',
                    data=all_paths,
                    maxshape=(None,),
                    chunks=(10000,),
                    dtype=dt,
                    compression='gzip',
                    compression_opts=1
                )

                # Metadata
                f.attrs['model_path'] = self.model_path
                f.attrs['created_at'] = datetime.now().isoformat()
                f.attrs['embedding_dim'] = 512
                f.attrs['total_images'] = 0  # Actualitzarem després

            else:
                # Append a datasets existents
                embeddings_ds = f['embeddings']
                paths_ds = f['image_paths']

                old_size = embeddings_ds.shape[0]
                new_size = old_size + len(all_embeddings)

                # Redimensiona
                embeddings_ds.resize((new_size, 512))
                paths_ds.resize((new_size,))

                # Escriu nous valors
                embeddings_ds[old_size:new_size] = all_embeddings
                paths_ds[old_size:new_size] = all_paths

            # Actualitza total
            if final:
                f.attrs['total_images'] = self.processed_images
                f.attrs['completed_at'] = datetime.now().isoformat()

        # Neteja buffers
        self.embeddings_buffer = []
        self.paths_buffer = []

        logger.info(f"Flush completat. Total en HDF5: {self.processed_images}")

    def process_folder(self, root_dir: str, auto_tune: bool = True):
        """
        Processa una carpeta completa d'imatges.

        Args:
            root_dir: Directori arrel amb imatges
            auto_tune: Fer auto-tuning del batch size
        """
        logger.info(f"Iniciant processament de {root_dir}")

        self.start_time = time.time()
        self.stopped = False

        # Inicialitza components
        self.image_loader = ImageLoader(num_workers=self.num_workers)
        self.gpu_processor = DynamicBatchProcessor(
            self.model_path,
            use_fp16=self.use_fp16,
            initial_batch_size=self.batch_size
        )

        # Warm-up GPU
        logger.info("Escalfant GPU...")
        self.gpu_processor.warmup(num_iterations=10, batch_size=self.batch_size)

        # Escaneja imatges
        image_paths = self.image_loader.scan_images(root_dir)
        self.total_images = len(image_paths)

        if self.total_images == 0:
            logger.error("No s'han trobat imatges!")
            return

        # Carrega checkpoint si existeix
        checkpoint = self._load_checkpoint()
        start_idx = checkpoint.get('processed_images', 0)

        if start_idx > 0:
            logger.info(f"Reprenent des de imatge {start_idx}")
            if self.skip_duplicates:
                self.seen_hashes = set(checkpoint.get('seen_hashes', []))

        # Auto-tuning opcional
        if auto_tune and start_idx == 0:
            sample_size = min(1000, len(image_paths))
            sample_paths = image_paths[:sample_size]

            optimal_batch = estimate_optimal_batch_size(
                self.gpu_processor,
                self.image_loader,
                sample_paths,
                test_sizes=[64, 128, 256]
            )

            self.batch_size = optimal_batch
            self.gpu_processor.current_batch_size = optimal_batch

        # Pipeline de processament
        logger.info(f"Processant {self.total_images} imatges...")

        last_update_time = time.time()
        last_checkpoint_idx = start_idx

        with tqdm(total=self.total_images, initial=start_idx, desc="Processant") as pbar:
            for i in range(start_idx, self.total_images, self.batch_size):
                # Check si està pausat/aturat
                while self.paused and not self.stopped:
                    time.sleep(0.1)

                if self.stopped:
                    logger.info("Processament aturat per l'usuari")
                    break

                # Carrega batch
                batch, valid_paths, error_paths = self.image_loader.load_batch(
                    image_paths, i, self.batch_size
                )

                self.error_count += len(error_paths)

                if len(batch) == 0:
                    continue

                # Filtra duplicats si cal
                if self.skip_duplicates:
                    filtered_batch = []
                    filtered_paths = []

                    for img, path in zip(batch, valid_paths):
                        img_hash = self._compute_image_hash(path)
                        if img_hash and img_hash not in self.seen_hashes:
                            self.seen_hashes.add(img_hash)
                            filtered_batch.append(img)
                            filtered_paths.append(path)

                    if filtered_batch:
                        batch = np.stack(filtered_batch)
                        valid_paths = filtered_paths
                    else:
                        continue

                # Processa amb GPU
                try:
                    embeddings = self.gpu_processor.process_batch_dynamic(batch)

                    # Afegeix a buffers
                    self.embeddings_buffer.append(embeddings)
                    self.paths_buffer.extend(valid_paths)

                    self.processed_images += len(embeddings)

                except Exception as e:
                    logger.error(f"Error processant batch: {str(e)}")
                    self.error_count += len(batch)

                # Actualitza progress bar
                pbar.update(len(batch))

                # Flush a HDF5 si el buffer està ple
                if len(self.paths_buffer) >= self.buffer_size:
                    self._flush_to_hdf5()

                # Checkpoint periòdic
                if self.processed_images - last_checkpoint_idx >= self.checkpoint_interval:
                    self._save_checkpoint()
                    last_checkpoint_idx = self.processed_images

                # Actualitza estadístiques cada 5 segons
                current_time = time.time()
                if current_time - last_update_time >= 5.0:
                    self._update_stats()
                    last_update_time = current_time

        # Flush final
        self._flush_to_hdf5(final=True)

        # Elimina checkpoint si s'ha completat
        if not self.stopped and os.path.exists(self.checkpoint_path):
            os.remove(self.checkpoint_path)

        # Neteja
        self.image_loader.shutdown()

        elapsed = time.time() - self.start_time
        logger.info(f"Processament completat en {elapsed/60:.1f} minuts")
        logger.info(f"Imatges processades: {self.processed_images}")
        logger.info(f"Errors: {self.error_count}")
        logger.info(f"Velocitat mitjana: {self.processed_images/elapsed:.1f} imgs/sec")

    def _update_stats(self):
        """Calcula i envia estadístiques actuals."""
        if not self.start_time:
            return

        elapsed = time.time() - self.start_time
        speed = self.processed_images / elapsed if elapsed > 0 else 0

        remaining = self.total_images - self.processed_images
        eta = remaining / speed if speed > 0 else 0

        stats = {
            'processed': self.processed_images,
            'total': self.total_images,
            'percentage': (self.processed_images / self.total_images * 100) if self.total_images > 0 else 0,
            'speed': speed,
            'elapsed': elapsed,
            'eta': eta,
            'errors': self.error_count,
        }

        # Callbacks
        if self.progress_callback:
            self.progress_callback(
                self.processed_images,
                self.total_images,
                stats['percentage']
            )

        if self.stats_callback:
            self.stats_callback(stats)

    def pause(self):
        """Pausa el processament."""
        self.paused = True
        logger.info("Processament pausat")

    def resume(self):
        """Reprèn el processament."""
        self.paused = False
        logger.info("Processament reprès")

    def stop(self):
        """Atura el processament."""
        self.stopped = True
        self._save_checkpoint()
        logger.info("Processament aturat")
