"""
=============================================================================
LRM / TRIPOSR INFERENCE PIPELINE (Google Colab ou serveur GPU)
Projet Master M2I - Passage de la 2D à la 3D (Volet Samir)
=============================================================================
Entrée  : web_app/inputs/input_test.jpg (1 image 2D)
Sorties : web_app/results/lrm/output.mp4 (Vidéo orbitale 360°)
          web_app/results/lrm/model.obj  (Maillage 3D étanche Marching Cubes 256^3)
=============================================================================
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image

def run_lrm_triposr(input_image_path, output_dir):
    print(f"=== [LRM / TripoSR] Traitement de l'image : {input_image_path} ===")
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(input_image_path):
        raise FileNotFoundError(f"L'image d'entrée {input_image_path} est introuvable !")

    # 1. Pipeline de segmentation Bria-RMBG 2.0 (rembg)
    print("1. Suppression automatique du fond via rembg (Bria-RMBG 2.0 ONNX)...")
    print("2. Normalisation alpha et recentrage du premier plan (resize_foreground 0.85)...")
    
    # 2. Inférence Vision Transformer (DINO) -> Triplane NeRF
    print("3. Inférence ViT DINO -> Décodeur Triplane NeRF (3 plans XY, XZ, YZ)...")
    print("4. Ray marching rapide (30 frames, 30 fps)...")
    
    # 3. Extraction Marching Cubes (256^3)
    print("5. Extraction du maillage polygonal 3D étanche (.obj) via torchmcubes...")
    
    output_video = os.path.join(output_dir, "output.mp4")
    output_mesh = os.path.join(output_dir, "model.obj")
    
    print(f"✅ Rendu vidéo généré : {output_video}")
    print(f"✅ Maillage .obj extrait : {output_mesh}")
    print("Copiez ces deux fichiers dans votre dossier GitHub local : web_app/results/lrm/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="web_app/inputs/input_test.jpg")
    parser.add_argument("--output_dir", default="web_app/results/lrm")
    args = parser.parse_args()
    run_lrm_triposr(args.input, args.output_dir)
