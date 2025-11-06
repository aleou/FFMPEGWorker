# Installation automatique de Real-ESRGAN et dépendances

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "🚀 Installation Real-ESRGAN Video Upscaling" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Vérifier Python
Write-Host "🔍 Vérification de Python..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Python non trouvé. Installez Python 3.8+ d'abord." -ForegroundColor Red
    exit 1
}
Write-Host "✅ $pythonVersion" -ForegroundColor Green
Write-Host ""

# Vérifier CUDA
Write-Host "🔍 Vérification de CUDA..." -ForegroundColor Yellow
$cudaCheck = python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}'); print(f'CUDA Version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')" 2>&1

if ($cudaCheck -match "CUDA: True") {
    Write-Host "✅ CUDA disponible" -ForegroundColor Green
    Write-Host $cudaCheck
} else {
    Write-Host "⚠️  CUDA non disponible - L'upscaling utilisera le CPU (beaucoup plus lent)" -ForegroundColor Yellow
}
Write-Host ""

# Installer les dépendances Real-ESRGAN
Write-Host "📦 Installation des dépendances Real-ESRGAN..." -ForegroundColor Yellow
Write-Host ""

$packages = @(
    "realesrgan==0.3.0",
    "basicsr==1.4.2",
    "facexlib==0.3.0",
    "gfpgan==1.3.8"
)

foreach ($package in $packages) {
    Write-Host "   Installing $package..." -ForegroundColor Cyan
    pip install $package --quiet
    if ($LASTEXITCODE -eq 0) {
        Write-Host "   ✅ $package installé" -ForegroundColor Green
    } else {
        Write-Host "   ❌ Échec installation de $package" -ForegroundColor Red
    }
}

Write-Host ""

# Vérifier l'installation
Write-Host "🔍 Vérification de l'installation..." -ForegroundColor Yellow
$verifyScript = @"
try:
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet
    import torch
    
    print('✅ Real-ESRGAN importé avec succès')
    print(f'✅ PyTorch: {torch.__version__}')
    print(f'✅ CUDA: {torch.cuda.is_available()}')
    
    if torch.cuda.is_available():
        print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f'✅ VRAM: {vram:.1f} GB')
except ImportError as e:
    print(f'❌ Erreur d import: {e}')
    exit(1)
"@

$verifyScript | python
if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "================================================" -ForegroundColor Green
    Write-Host "✅ Installation réussie !" -ForegroundColor Green
    Write-Host "================================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "📚 Prochaines étapes :" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "1. Test rapide :" -ForegroundColor White
    Write-Host "   python test_upscale.py" -ForegroundColor Gray
    Write-Host ""
    Write-Host "2. Pipeline complet (watermark + upscale) :" -ForegroundColor White
    Write-Host "   python pipeline_complete.py video.mp4" -ForegroundColor Gray
    Write-Host ""
    Write-Host "3. Via API :" -ForegroundColor White
    Write-Host "   uvicorn app.main:app --reload" -ForegroundColor Gray
    Write-Host "   curl -X POST http://localhost:8000/api/v1/upscale/ -F 'file=@video.mp4'" -ForegroundColor Gray
    Write-Host ""
    Write-Host "📖 Documentation : VIDEO_UPSCALING_GUIDE.md" -ForegroundColor Cyan
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "❌ Problème détecté lors de la vérification" -ForegroundColor Red
    Write-Host "Essayez d'installer manuellement :" -ForegroundColor Yellow
    Write-Host "   pip install realesrgan basicsr facexlib gfpgan" -ForegroundColor Gray
    Write-Host ""
}

# Télécharger les modèles (optionnel)
$downloadModels = Read-Host "Voulez-vous télécharger les modèles maintenant ? (y/N)"
if ($downloadModels -eq "y" -or $downloadModels -eq "Y") {
    Write-Host ""
    Write-Host "📥 Téléchargement des modèles..." -ForegroundColor Yellow
    
    $downloadScript = @"
from app.services.video_upscaler_service import VideoUpscalerService, UpscaleModel

print('Téléchargement RealESRGAN_x4plus...')
upscaler = VideoUpscalerService(model_name=UpscaleModel.REALESR_GENERAL_X4V3)
upscaler._load_model()
print('✅ RealESRGAN_x4plus téléchargé')

print('Téléchargement RealESRGAN_x4plus_anime...')
upscaler = VideoUpscalerService(model_name=UpscaleModel.REALESR_ANIME_X4)
upscaler._load_model()
print('✅ RealESRGAN_x4plus_anime téléchargé')

print('✅ Tous les modèles sont prêts !')
"@
    
    $downloadScript | python
}

Write-Host ""
Write-Host "Terminé !" -ForegroundColor Green
