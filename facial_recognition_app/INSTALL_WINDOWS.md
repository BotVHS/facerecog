# Instal·lació a Windows

## Important: FAISS GPU a Windows

A Windows, `faiss-gpu` NO està disponible via pip. Tens dues opcions:

---

## Opció 1: FAISS CPU (Recomanat per simplicitat)

Utilitzar la versió CPU de FAISS. Encara serà ràpid per la majoria d'usos, tot i que les cerques seran més lentes.

### Instal·lació:

```powershell
# Crea entorn virtual (opcional però recomanat)
python -m venv venv
venv\Scripts\activate

# Instal·la dependencies
pip install -r requirements.txt
```

**Rendiment esperat:**
- Creació dataset: 400-600 imgs/sec (GPU encara s'utilitza)
- Cerca FAISS: 100-500ms per 5M imatges (en CPU)

---

## Opció 2: FAISS GPU amb Conda (Màxim rendiment)

Per utilitzar FAISS GPU a Windows, necessites instal·lar-lo amb Conda.

### Pas 1: Instal·la Anaconda o Miniconda

Descarrega i instal·la des de:
- Anaconda: https://www.anaconda.com/download
- Miniconda (més lleuger): https://docs.conda.io/en/latest/miniconda.html

### Pas 2: Crea entorn conda

```powershell
# Crea nou entorn amb Python 3.10
conda create -n facerecog python=3.10
conda activate facerecog
```

### Pas 3: Instal·la FAISS GPU

```powershell
# Instal·la faiss-gpu des del canal conda-forge
conda install -c conda-forge faiss-gpu

# Alternativament, des del canal pytorch
conda install -c pytorch faiss-gpu
```

### Pas 4: Instal·la altres dependencies

```powershell
pip install onnxruntime-gpu
pip install opencv-contrib-python
pip install h5py tqdm numpy pillow PyQt5
```

### Pas 5: Verifica la instal·lació

```python
import faiss
print(f"FAISS version: {faiss.__version__}")

# Verifica que GPU està disponible
res = faiss.StandardGpuResources()
print("FAISS GPU disponible!")
```

**Rendiment esperat:**
- Creació dataset: 400-600 imgs/sec
- Cerca FAISS: <100ms per 5M imatges (en GPU)

---

## Opció 3: WSL2 (Windows Subsystem for Linux)

Si vols utilitzar l'experiència completa de Linux a Windows:

### Pas 1: Activa WSL2

```powershell
wsl --install
```

### Pas 2: Instal·la Ubuntu

```powershell
wsl --install -d Ubuntu-22.04
```

### Pas 3: Dins de WSL, instal·la com a Linux

```bash
sudo apt update
sudo apt install python3-pip

pip install -r requirements_linux.txt
```

Crea `requirements_linux.txt`:
```
onnxruntime-gpu
opencv-contrib-python
faiss-gpu
h5py
tqdm
numpy
pillow
PyQt5
```

---

## Troubleshooting Windows

### Error: Microsoft Visual C++ 14.0 is required

Instal·la Visual C++ Build Tools:
https://visualstudio.microsoft.com/visual-cpp-build-tools/

### Error: CUDA not available

1. Verifica drivers NVIDIA:
   ```powershell
   nvidia-smi
   ```

2. Instal·la CUDA Toolkit:
   https://developer.nvidia.com/cuda-downloads

3. Reinicia el PC després d'instal·lar CUDA

### Error: DLL load failed

Instal·la Visual C++ Redistributables:
- https://aka.ms/vs/17/release/vc_redist.x64.exe

---

## Resum: Quina opció triar?

| Opció | Velocitat | Dificultat | Recomanat per |
|-------|-----------|------------|---------------|
| **FAISS CPU** | Ràpida (creació), Mitjana (cerca) | Fàcil | Usuaris nous, datasets petits (<1M) |
| **FAISS GPU (Conda)** | Molt ràpida | Mitjana | Usuaris amb experiència conda, datasets grans |
| **WSL2** | Molt ràpida | Mitjana-Alta | Usuaris còmodes amb Linux |

**Recomanació**: Comença amb **Opció 1 (FAISS CPU)** per provar l'aplicació. Si necessites més velocitat en cerques, passa a **Opció 2 (Conda)**.

---

## Executar l'aplicació

Després d'instal·lar les dependencies:

```powershell
# Assegura't que tens el model
# Col·loca mobilefacenet.onnx a: models/mobilefacenet.onnx

# Executa l'aplicació
python main.py
```

---

## Notes addicionals

- L'aplicació detectarà automàticament si FAISS GPU està disponible
- Si no troba GPU, utilitzarà CPU automàticament
- La creació de dataset sempre utilitzarà GPU (ONNX Runtime)
- Només les cerques FAISS es veuran afectades per CPU vs GPU
