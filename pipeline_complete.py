"""
Pipeline complet : Watermark Removal + Video Upscaling
Démontre l'utilisation combinée des deux services AI pour un traitement vidéo complet.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.services.watermark_removal_service import WatermarkRemovalService
from app.services.video_upscaler_service import VideoUpscalerService, UpscaleModel


def process_video_complete_pipeline(
    input_video: Path,
    output_dir: Path,
    use_anime_model: bool = False,
    remove_watermark: bool = True,
    upscale: bool = True,
):
    """
    Pipeline complet de traitement vidéo.
    
    Étapes :
    1. Détection et suppression des watermarks (optionnel)
    2. Upscaling 4x avec Real-ESRGAN (optionnel)
    3. Export final avec audio préservé
    
    Args:
        input_video: Chemin de la vidéo d'entrée
        output_dir: Dossier de sortie
        use_anime_model: Utiliser le modèle optimisé pour anime
        remove_watermark: Activer la suppression de watermark
        upscale: Activer l'upscaling
    """
    
    if not input_video.exists():
        print(f"❌ Vidéo introuvable : {input_video}")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("🎬 PIPELINE COMPLET DE TRAITEMENT VIDÉO")
    print("=" * 60)
    print(f"📹 Vidéo d'entrée : {input_video}")
    print(f"📁 Dossier de sortie : {output_dir}")
    print(f"🎨 Modèle anime : {'Oui' if use_anime_model else 'Non'}")
    print(f"🚫 Suppression watermark : {'Oui' if remove_watermark else 'Non'}")
    print(f"📈 Upscaling : {'Oui' if upscale else 'Non'}")
    print("=" * 60)
    print()
    
    current_video = input_video
    start_time = time.time()
    
    # ==========================================
    # ÉTAPE 1 : Suppression du watermark
    # ==========================================
    if remove_watermark:
        print("🔍 ÉTAPE 1/2 : Suppression du watermark")
        print("-" * 60)
        
        watermark_service = WatermarkRemovalService(
            device="cuda",
            preferred_models="lama,zits",
            default_detector="yolo",  # YOLO est plus rapide
        )
        
        cleaned_video = output_dir / f"{input_video.stem}_no_watermark{input_video.suffix}"
        
        try:
            step_start = time.time()
            
            current_video = watermark_service.process_file(
                input_path=current_video,
                output_path=cleaned_video,
                detector="yolo",
                overwrite=True,
            )
            
            step_duration = time.time() - step_start
            print(f"✅ Watermark supprimé en {step_duration:.1f}s")
            print(f"   Sortie : {current_video}")
            print()
            
        except Exception as e:
            print(f"❌ Échec de la suppression de watermark : {e}")
            import traceback
            traceback.print_exc()
            return
    
    # ==========================================
    # ÉTAPE 2 : Upscaling vidéo
    # ==========================================
    if upscale:
        step_number = "2/2" if remove_watermark else "1/1"
        print(f"📈 ÉTAPE {step_number} : Upscaling vidéo (4x)")
        print("-" * 60)
        
        # Choisir le modèle
        model = (
            UpscaleModel.REALESR_ANIME_X4 
            if use_anime_model 
            else UpscaleModel.REALESR_GENERAL_X4V3
        )
        
        upscaler = VideoUpscalerService(
            device="cuda",
            model_name=model,
            tile_size=0,  # Auto-détection basée sur VRAM
            half_precision=True,  # FP16 pour vitesse
        )
        
        upscaled_video = output_dir / f"{input_video.stem}_4K{input_video.suffix}"
        
        try:
            step_start = time.time()
            
            current_video = upscaler.upscale_video_gpu_optimized(
                input_path=current_video,
                output_path=upscaled_video,
                batch_size=4,
                audio=True,
            )
            
            step_duration = time.time() - step_start
            print(f"✅ Upscaling terminé en {step_duration:.1f}s")
            print(f"   Sortie : {current_video}")
            print()
            
        except Exception as e:
            print(f"❌ Échec de l'upscaling : {e}")
            import traceback
            traceback.print_exc()
            return
    
    # ==========================================
    # RÉSUMÉ FINAL
    # ==========================================
    total_duration = time.time() - start_time
    
    print("=" * 60)
    print("✨ TRAITEMENT TERMINÉ")
    print("=" * 60)
    print(f"⏱️  Durée totale : {total_duration:.1f}s ({total_duration/60:.1f} min)")
    print(f"📹 Vidéo finale : {current_video}")
    
    # Tailles de fichiers
    input_size = input_video.stat().st_size / (1024 * 1024)
    output_size = current_video.stat().st_size / (1024 * 1024)
    
    print(f"💾 Taille entrée : {input_size:.2f} MB")
    print(f"💾 Taille sortie : {output_size:.2f} MB")
    print(f"📊 Ratio : {output_size/input_size:.2f}x")
    print("=" * 60)


def main():
    """Point d'entrée principal avec exemples."""
    
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Pipeline complet : Watermark Removal + Upscaling"
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Chemin de la vidéo d'entrée"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("outputs/pipeline"),
        help="Dossier de sortie (défaut: outputs/pipeline)"
    )
    parser.add_argument(
        "--anime",
        action="store_true",
        help="Utiliser le modèle optimisé pour anime"
    )
    parser.add_argument(
        "--no-watermark-removal",
        action="store_true",
        help="Désactiver la suppression de watermark"
    )
    parser.add_argument(
        "--no-upscale",
        action="store_true",
        help="Désactiver l'upscaling"
    )
    
    args = parser.parse_args()
    
    process_video_complete_pipeline(
        input_video=args.input,
        output_dir=args.output,
        use_anime_model=args.anime,
        remove_watermark=not args.no_watermark_removal,
        upscale=not args.no_upscale,
    )


if __name__ == "__main__":
    # Exemples d'utilisation
    
    # Exemple 1 : Pipeline complet
    # python pipeline_complete.py video.mp4
    
    # Exemple 2 : Anime avec pipeline complet
    # python pipeline_complete.py anime.mp4 --anime
    
    # Exemple 3 : Uniquement upscaling
    # python pipeline_complete.py video.mp4 --no-watermark-removal
    
    # Exemple 4 : Uniquement watermark removal
    # python pipeline_complete.py video.mp4 --no-upscale
    
    main()
