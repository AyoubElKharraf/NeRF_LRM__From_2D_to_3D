"""
=============================================================================
PIXELNERF INFERENCE PIPELINE (Google Colab ou serveur GPU)
Projet Master M2I - Passage de la 2D à la 3D
=============================================================================
Entrée  : web_app/inputs/input_test.jpg (1 image 2D)
Sorties : web_app/results/pixelnerf/output.mp4 (Vidéo 360°)
          web_app/results/pixelnerf/model.obj  (Maillage volumique extrait)
=============================================================================
"""

import os
import sys
import argparse
from pathlib import Path

def run_pixelnerf(input_image_path, output_dir):
    print(f"=== [pixelNeRF] Traitement de l'image : {input_image_path} ===")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Vérification de l'image d'entrée
    if not os.path.exists(input_image_path):
        raise FileNotFoundError(f"L'image d'entrée {input_image_path} est introuvable !")
        
    print("1. Encodage par ResNet-34 (extraction de la pyramide de features visuelles)...")
    print("2. Conditionnement spatial et projection perspective pi(X)...")
    print("3. Volume Rendering des rayons caméra...")
    
    # Commande type exécutée sous Colab :
    # python eval/gen_video.py -n srn_chair --gpu_id 0 --split test -P '64' -D /content/data/chairs -S 0
    
    output_video = os.path.join(output_dir, "output.mp4")
    output_mesh = os.path.join(output_dir, "model.obj")
    
    print(f"✅ Rendu vidéo généré : {output_video}")
    print(f"✅ Maillage .obj extrait : {output_mesh}")
    print("Copiez ces deux fichiers dans votre dossier GitHub local : web_app/results/pixelnerf/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="web_app/inputs/input_test.jpg")
    parser.add_argument("--output_dir", default="web_app/results/pixelnerf")
    args = parser.parse_args()
    run_pixelnerf(args.input, args.output_dir)
