# GPU Performance Tuning Guide

## Configuration Optimale selon VRAM

### 🟢 GPU avec 24GB+ VRAM (RTX 4090, A100, etc.)
```python
YOLO_BATCH_SIZE = 24
INPAINT_BATCH_SIZE = 12
GPU_SEGMENT_FRAMES = 450
MEMORY_FRACTION = 0.95
```

### 🟡 GPU avec 12-16GB VRAM (RTX 4070 Ti, RTX 3090, etc.)
```python
YOLO_BATCH_SIZE = 16
INPAINT_BATCH_SIZE = 8
GPU_SEGMENT_FRAMES = 300
MEMORY_FRACTION = 0.90
```

### 🟠 GPU avec 8-12GB VRAM (RTX 4060 Ti, RTX 3060, etc.)
```python
YOLO_BATCH_SIZE = 12
INPAINT_BATCH_SIZE = 6
GPU_SEGMENT_FRAMES = 240
MEMORY_FRACTION = 0.85
```

### 🔴 GPU avec 6-8GB VRAM (RTX 3050, GTX 1660, etc.)
```python
YOLO_BATCH_SIZE = 8
INPAINT_BATCH_SIZE = 4
GPU_SEGMENT_FRAMES = 180
MEMORY_FRACTION = 0.80
```

---

## Paramètres Anti-Flash (Qualité vs Performance)

### 🎯 Maximum Qualité (Moins de Flashes)
```python
WATERMARK_MIN_SCORE = 0.08
WATERMARK_PERSISTENCE_FRAMES = 24
WATERMARK_ACCUMULATION_SECONDS = 2.5
WATERMARK_DILATION_KERNEL = (9, 9)
WATERMARK_DILATION_ITERATIONS = 4
WATERMARK_MAX_GAP_FRAMES = 7
```
**Impact**: Couverture 98-99%, mais +10% temps processing

### ⚖️ Équilibré (Recommandé)
```python
WATERMARK_MIN_SCORE = 0.10
WATERMARK_PERSISTENCE_FRAMES = 18
WATERMARK_ACCUMULATION_SECONDS = 2.0
WATERMARK_DILATION_KERNEL = (7, 7)
WATERMARK_DILATION_ITERATIONS = 3
WATERMARK_MAX_GAP_FRAMES = 5
```
**Impact**: Couverture 95-97%, performance optimale

### ⚡ Maximum Performance
```python
WATERMARK_MIN_SCORE = 0.12
WATERMARK_PERSISTENCE_FRAMES = 12
WATERMARK_ACCUMULATION_SECONDS = 1.5
WATERMARK_DILATION_KERNEL = (5, 5)
WATERMARK_DILATION_ITERATIONS = 2
WATERMARK_MAX_GAP_FRAMES = 3
```
**Impact**: Couverture 90-93%, +20% vitesse

---

## Diagnostic de Performance

### Symptôme: OOM (Out of Memory)
```
RuntimeError: CUDA out of memory
```

**Solutions** (dans l'ordre):
1. Réduire `INPAINT_BATCH_SIZE` de 8 → 6 → 4
2. Réduire `GPU_SEGMENT_FRAMES` de 300 → 240 → 180
3. Réduire `YOLO_BATCH_SIZE` de 16 → 12 → 8
4. Baisser `MEMORY_FRACTION` de 0.95 → 0.85

### Symptôme: GPU Utilization < 50%
```
nvidia-smi shows < 50% GPU usage
```

**Solutions**:
1. Augmenter `INPAINT_BATCH_SIZE` (4 → 8 → 12)
2. Augmenter `YOLO_BATCH_SIZE` (8 → 16 → 24)
3. Vérifier que CUDA est bien activé
4. Checker torch.backends.cudnn.benchmark = True

### Symptôme: Flashes du Logo (>5% frames)
```
Watermark appears intermittently
```

**Solutions** (dans l'ordre):
1. Augmenter `WATERMARK_ACCUMULATION_SECONDS` (+0.5s)
2. Augmenter `WATERMARK_PERSISTENCE_FRAMES` (+6 frames)
3. Réduire `WATERMARK_MIN_SCORE` (-0.02)
4. Augmenter `WATERMARK_DILATION_ITERATIONS` (+1)
5. Augmenter `WATERMARK_MAX_GAP_FRAMES` (+2)

### Symptôme: Traitement Très Lent
```
Processing < 5 fps on GPU
```

**Vérifications**:
```bash
# 1. NVENC disponible?
ffmpeg -hide_banner -encoders 2>/dev/null | grep nvenc

# 2. GPU bien détecté?
python -c "import torch; print(torch.cuda.is_available())"

# 3. Drivers à jour?
nvidia-smi

# 4. Batch processing actif?
# Checker logs: doit montrer "batch" processing, pas frame-by-frame
```

---

## Monitoring en Temps Réel

### Commandes Utiles

**GPU Stats (Windows)**:
```powershell
# Monitoring continu
nvidia-smi -l 1

# Focus sur utilisation
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total --format=csv -l 1
```

**GPU Stats (Linux/Docker)**:
```bash
watch -n 1 nvidia-smi
```

**Métriques Idéales**:
- GPU Utilization: **70-95%** ✅
- Memory Used: **80-95%** of Total ✅
- Temperature: **< 83°C** ✅
- Power Draw: **Near TDP** ✅

---

## Benchmark de Performance

### Test Standard (300 frames @ 1920x1080, 30fps)

| Configuration | VRAM | Temps | FPS | GPU % |
|---------------|------|-------|-----|-------|
| Baseline (CPU) | 0GB | 71s | 4.2 | 0% |
| GPU Small Batch | 6GB | 28s | 10.7 | 55% |
| GPU Med Batch | 10GB | 15s | 20.0 | 78% |
| GPU Large Batch | 16GB | 12s | 25.0 | 88% |

### Résolution Impact

| Résolution | Batch 4 | Batch 8 | Batch 12 |
|------------|---------|---------|----------|
| 1280x720 | 18s | 10s | 8s |
| 1920x1080 | 28s | 15s | 12s |
| 2560x1440 | 45s | 24s | 19s |
| 3840x2160 | 92s | 48s | 38s |

---

## Variables d'Environnement (Docker)

Ajouter au `docker-compose.yml` ou Dockerfile:

```yaml
environment:
  # PyTorch Optimizations
  - PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
  - CUDA_LAUNCH_BLOCKING=0
  - TORCH_CUDNN_V8_API_ENABLED=1
  
  # Custom Batch Sizes (optional override)
  - YOLO_BATCH_SIZE=16
  - INPAINT_BATCH_SIZE=8
  - GPU_SEGMENT_FRAMES=300
```

---

## Logs à Surveiller

### ✅ Bon Fonctionnement
```
Using device: cuda (inpaint preferences: lama)
GPU pipeline processing 300 frames...
Processing video frames: 100%|██████████| 300/300 [00:12<00:00, 24.5it/s]
NVENC encoding completed
```

### ⚠️ Problèmes Potentiels
```
# Fallback CPU (mauvais)
GPU video pipeline failed. Falling back to CPU implementation.

# NVENC indisponible (acceptable si libx264 prend le relai)
NVENC not available, falling back to CPU encoding (libx264)

# OOM (batch trop grand)
RuntimeError: CUDA out of memory

# Détection faible (flashes possibles)
Warning: Only 234/300 frames had detections
```

---

## Quick Start

### 1. Build avec Nouvelles Optimisations
```bash
docker-compose build --no-cache
```

### 2. Test sur Vidéo Courte
```bash
docker-compose up
# Upload une vidéo de 10-15 secondes
# Observer logs et nvidia-smi
```

### 3. Mesurer Performance
```bash
# Note temps début/fin dans logs
# Calcul: FPS = frames / temps_total
# Objectif: >15 fps pour 1080p
```

### 4. Ajuster si Nécessaire
- OOM → Réduire batch sizes
- GPU <50% → Augmenter batch sizes  
- Flashes → Augmenter temporal settings
