# 🚨 Troubleshooting: RunPod GPU Detection Error

## ❌ Erreur Rencontrée

```
error creating container: nvidia-smi: parsing output of line 0: 
failed to parse (pcie.link.gen.max) into int: 
strconv.Atoi: parsing "": invalid syntax
```

## 🔍 Cause du Problème

Ce n'est **PAS** lié au code de l'application. C'est RunPod qui essaie de détecter les capacités GPU avant de créer le container et échoue à parser la sortie de `nvidia-smi`.

### Causes Possibles

1. **GPU Virtualisé (vGPU)** : Certaines propriétés PCIe ne sont pas exposées
2. **Drivers NVIDIA incomplets** dans l'environnement RunPod
3. **Template RunPod incompatible** avec le GPU sélectionné
4. **Version nvidia-smi obsolète** ou incompatible

---

## ✅ Solutions

### Solution 1: Workaround Dockerfile (Déjà Appliqué)

J'ai ajouté ces variables d'environnement dans le Dockerfile :

```dockerfile
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    RUNPOD_SKIP_GPU_DETECTION=1
```

**Action** : Rebuild et redeploy l'image

```bash
# Build nouvelle image
docker build -t aleou/ffmpeg-worker:0.16.1-serverless --target final_serverless .

# Push vers registry
docker push aleou/ffmpeg-worker:0.16.1-serverless

# Update RunPod template avec nouvelle version
```

---

### Solution 2: Changer de GPU Template RunPod

Le problème arrive souvent avec certains GPUs ou templates.

**Essayer dans cet ordre** :

1. **RTX 4090** (recommandé pour inference)
2. **RTX A6000** 
3. **A40**
4. **RTX 3090**

**À éviter** :
- GPU virtualisés (vGPU)
- GPUs très anciens (< Pascal)
- Templates custom mal configurés

---

### Solution 3: Utiliser Base Image Différente

Si le problème persiste, essayer avec une base image plus récente :

```dockerfile
# Au lieu de:
FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

# Essayer:
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.0-devel-ubuntu22.04
# ou
FROM nvidia/cuda:12.4.0-cudnn-devel-ubuntu22.04
```

---

### Solution 4: Contacter Support RunPod

Si rien ne fonctionne, c'est probablement un bug RunPod.

**Informations à fournir** :
- GPU Type sélectionné
- Template utilisé
- Base image Docker
- Logs complets de l'erreur

---

## 🧪 Vérifications à Faire

### 1. Vérifier GPU Disponible sur Template

Dans le pod RunPod, ouvrir un terminal et tester :

```bash
# Vérifier nvidia-smi fonctionne
nvidia-smi

# Vérifier CUDA visible
echo $CUDA_VISIBLE_DEVICES

# Vérifier PyTorch détecte GPU
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

### 2. Tester Sans Detection GPU

Si le container démarre malgré l'erreur, vérifier que le code fonctionne :

```bash
# Dans le container
python -c "
from app.services.watermark_removal_service import WatermarkRemovalService
service = WatermarkRemovalService(device='cuda')
print('Service initialized:', service.device)
"
```

### 3. Vérifier Logs RunPod

Regarder les logs détaillés dans RunPod UI :
- Container startup logs
- Application logs
- GPU detection logs

---

## 📋 Checklist de Résolution

- [ ] Rebuild image avec nouveau Dockerfile (v0.16.1)
- [ ] Push vers Docker registry
- [ ] Update RunPod template avec nouvelle image
- [ ] Tester avec différents GPU types si échec
- [ ] Vérifier nvidia-smi dans pod terminal
- [ ] Vérifier PyTorch détecte CUDA
- [ ] Contacter support RunPod si toujours échec

---

## 🔄 Rollback si Nécessaire

Si les optimisations GPU causent trop de problèmes, voici comment rollback :

```bash
# Retourner à la version précédente
docker pull aleou/ffmpeg-worker:0.15.0-serverless

# Ou désactiver GPU dans la config
AI_DEVICE=cpu
```

**Note** : Les optimisations elles-mêmes sont bonnes, c'est juste RunPod qui a un bug de détection GPU.

---

## 💡 Alternative: Tester Localement d'Abord

Pour valider que les optimisations fonctionnent avant de déployer sur RunPod :

```bash
# Build et test local (si GPU disponible)
docker build -t ffmpeg-worker:test --target final_serverless .

docker run --gpus all \
  -e RUNPOD_SKIP_GPU_DETECTION=1 \
  ffmpeg-worker:test \
  python -c "
from app.services.watermark_removal_service import WatermarkRemovalService
import torch
print('CUDA available:', torch.cuda.is_available())
service = WatermarkRemovalService(device='cuda')
print('Service device:', service.device)
"
```

---

## 📞 Contact Support

**RunPod Discord** : https://discord.gg/runpod
**GitHub Issues** : https://github.com/runpod/runpod-python/issues

**Mentionner** :
- Erreur: `nvidia-smi: parsing output of line 0: failed to parse (pcie.link.gen.max)`
- Base image: `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04`
- GPU type utilisé

---

*Note : Ce n'est PAS un bug de notre code, c'est RunPod infrastructure.*
