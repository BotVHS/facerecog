# facerecog
Crea una aplicació Python ULTRA-OPTIMITZADA per a màxima velocitat en creació d'embeddings facials i reconeixement, amb les següents especificacions:

## Requisits del Sistema
- Windows 10
- GPU: RTX 2060 amb CUDA instal·lat (6GB VRAM)
- RAM: 64GB (utilitza fins a 32GB agressivament)
- CPU: Ryzen 5 3600 (6 cores/12 threads)
- Dataset: ~5 milions d'imatges de rostres (format DNI)

## OBJECTIU: MÀXIMA VELOCITAT
Target: Processar 400-600 imatges/segon = ~2-3 hores per 5M imatges

## Model ULTRA-RÀPID: MobileFaceNet amb ONNX Runtime

Utilitza MobileFaceNet en format ONNX per màxima velocitat:
- Model lleuger específicament dissenyat per velocitat
- 512-dimensional embeddings
- Precisió: ~99.2% (LFW) - només 0.5% menys que models pesats
- Velocitat: 400-800 fps amb RTX 2060
- Descàrrega el model preentrenat des de: insightface models o repositoris ONNX de MobileFaceNet

## Stack Tecnològic

Instal·la aquestes dependencies:
pip install onnxruntime-gpu
pip install opencv-contrib-python
pip install faiss-gpu
pip install h5py
pip install tqdm
pip install numpy
pip install pillow
pip install PyQt5

## Optimitzacions EXTREMES

### 1. Pipeline Asíncron Multi-Etapa
Implementa un pipeline amb aquestes etapes en paral·lel:
- 6-8 threads per lectura de disk (aprofita els 6 cores CPU)
- Queue de preprocessing en CPU
- Batch processing en GPU amb CUDA
- Escriptura asíncrona a HDF5

### 2. Configuració GPU Agressiva
- Batch size: 128-256 (ajustar segons VRAM disponible)
- Mixed precision (FP16) per duplicar velocitat si el model ho suporta
- Preallocate GPU memory per evitar overhead
- Processa batches continus sense idle time

### 3. Processament d'Imatges Optimitzat
- Carrega imatges amb OpenCV
- Resize i normalize en batch
- Converteix a tensor/array en format que el model espera
- Evita còpies innecessàries de memòria

### 4. I/O Disk Multi-threaded
- ThreadPoolExecutor amb 6-8 workers per lectura concurrent
- Prefetching: mentre processa batch N, carrega batch N+2
- Buffer en RAM de 24GB per guardar embeddings abans de flush a disk

### 5. Gestió de Memòria RAM
- Prealoca arrays NumPy grans per evitar reallocacions
- Guarda embeddings en memòria, escriu a HDF5 cada 500K imatges
- Format ultra-eficient: HDF5 amb chunking optimitzat

### 6. FAISS GPU per Cerca Ràpida
- Crea índex FAISS completament en GPU: IndexFlatIP o IndexFlatL2
- Per 5M vectors de 512D això ocuparà ~10GB
- Implementa cerca ràpida de top-K similars

## Arquitectura de l'Aplicació

### Component 1: Creador de Dataset
Funcionalitats:
- Escaneja carpeta arrel recursivament
- Detecta totes les imatges (jpg, png, jpeg, bmp)
- Auto-tune inicial: prova 1000 imatges amb diferents batch sizes (64, 128, 256) i escull el més ràpid
- Pipeline multi-thread:
  - Thread pool per lectura d'imatges
  - Queue per batches preprocessats
  - GPU inference en batches
  - Escriptura asíncrona
- Checkpoint cada 250K imatges per poder reprendre
- Gestió d'errors: salta imatges corruptes sense aturar el pipeline
- Estadístiques en temps real:
  - Imatges processades / Total
  - Velocitat (imgs/segon)
  - ETA basada en últims 10K imatges
  - Ús de GPU i RAM
  - Errors trobats

Output:
- facial_embeddings.h5 amb estructura:
  - /embeddings: float32 (N, 512)
  - /image_paths: strings comprimits
  - /metadata: info del model i timestamps

### Component 2: Sistema de Reconeixement
Funcionalitats:
- Carrega dataset d'embeddings en memòria/GPU
- Crea índex FAISS per cerca ràpida
- Interface per carregar imatge de consulta
- Genera embedding de la imatge consulta
- Cerca top-K similars amb FAISS
- Mostra resultats amb:
  - Miniatura de la imatge trobada
  - Path complet
  - Score de similitud
  - Temps de cerca

### Interfície Gràfica (PyQt5)

Crea una UI amb dues pestanyes:

**Pestanya 1: "Crear Dataset"**
- QLineEdit + QPushButton per seleccionar carpeta arrel
- QLabel per mostrar carpeta seleccionada
- Configuració avançada (QGroupBox collapsable):
  - QSpinBox per batch size (64-256)
  - QSpinBox per threads I/O (4-8)
  - QCheckBox per activar FP16
- QPushButton "INICIAR PROCESSAMENT"
- QProgressBar amb percentatge
- QLCDNumber o QLabel grans per:
  - Velocitat actual (imgs/sec)
  - Temps transcorregut
  - ETA
- QTextEdit petit per log d'errors (últims 100)
- QPushButton "PAUSAR/REPRENDRE"

**Pestanya 2: "Reconeixement Facial"**
- QPushButton per carregar imatge consulta
- QLabel per preview de la imatge
- QPushButton per carregar dataset d'embeddings
- QSpinBox per seleccionar top-K (1-50)
- QPushButton "CERCAR"
- QScrollArea amb grid de resultats:
  - Cada resultat mostra: miniatura, path, score
- QLabel per mostrar temps de cerca

### Threading de la UI
- Tot el processament pesant en threads separats (QThread)
- Actualitza UI cada 5 segons (no cada batch) per evitar overhead
- Signals/Slots per comunicació thread-segura

## Gestió d'Errors Ultra-Eficient
- Try-catch al voltant de lectura/processament de cada imatge
- Salta imatges corruptes o sense rostres
- Guarda errors en buffer, escriu log al final
- NO aturis el pipeline mai per 1 error
- Mostra contador d'errors en temps real

## Característiques Extra per Velocitat

1. **Auto-tuning inicial**: Prova diferents configuracions i escull la millor
2. **Skip duplicats opcionals**: MD5 hash per detectar duplicats (checkbox a UI)
3. **Progress estimation intel·ligent**: ETA basat en velocitat recent, no global
4. **Warm-up**: Processa 100 imatges primer per escalfar GPU
5. **Batch size dinàmic**: Ajusta automàticament si hi ha OOM errors

## Codi Organitzat

Estructura de fitxers:
facial_recognition_app/
├── main.py                 # Entry point, crea la UI
├── dataset_creator.py      # Lògica de creació de dataset
├── face_recognizer.py      # Lògica de reconeixement
├── models/                 # Carpeta pels models ONNX
│   └── mobilefacenet.onnx
├── ui/
│   ├── main_window.py      # Finestra principal PyQt5
│   ├── dataset_tab.py      # Pestanya de creació
│   └── recognition_tab.py  # Pestanya de reconeixement
├── utils/
│   ├── image_loader.py     # Multi-threaded image loading
│   ├── gpu_processor.py    # Batch inference amb ONNX
│   └── faiss_index.py      # Gestió índex FAISS
├── requirements.txt
└── README.md

## README Complet

El README ha d'incloure:
1. Descripció del projecte
2. Requisits del sistema
3. Instruccions d'instal·lació pas a pas
4. Com descarregar el model MobileFaceNet ONNX
5. Ús de l'aplicació (screenshots opcionals)
6. Benchmarks esperats segons configuració
7. Troubleshooting comú:
   - OOM errors
   - CUDA not found
   - Model loading errors
8. Configuració recomanada per RTX 2060

## Mètriques de Rendiment Esperades

Amb RTX 2060 + configuració optimitzada:
- Velocitat: 400-600 imatges/segon
- Temps total per 5M imatges: 2-3 hores
- Ús GPU: 95-100% durant processament
- Ús RAM: 24-28GB durant processament
- Temps cerca (top-50 en 5M): <100ms
- Precisió: ~99% en reconeixement

## Notes Finals

- Codi ben comentat explicant les optimitzacions crítiques
- Type hints per millor mantenibilitat
- Logging amb nivells (INFO, WARNING, ERROR)
- Gestió adequada de recursos (close files, free GPU memory)
- Validació d'inputs de l'usuari

Crea una solució completa, funcional i de MÀXIM RENDIMENT. Prioritza velocitat però mantenint codi organitzat i mantenible.
