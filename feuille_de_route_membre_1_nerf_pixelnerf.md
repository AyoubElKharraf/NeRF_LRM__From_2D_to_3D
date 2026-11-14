# Feuille de Route Opérationnelle : Membre 1 (NeRF & pixelNeRF)

Ton rôle couvre le socle fondamental (**NeRF**) et le passage à la généralisation (**pixelNeRF**). Voici le plan d'action étape par étape, orienté utilisation de bibliothèques existantes et réentraînement.

---

## Phase 1 : Préparation de l'Environnement (Semaine 1)
* **Infrastructure GPU :** Configurer un environnement Python isolé (Conda ou conteneur Docker) avec PyTorch et support CUDA activé (localement, sur serveur de l'université, ou via Google Colab Pro / Kaggle).
* **Installation des outils :**
  * Installer `nerfstudio` (qui inclut les implémentations optimisées de NeRF et Instant-NGP).
  * Installer `COLMAP` pour l'extraction automatique des poses de caméras.
  * Cloner le dépôt officiel de pixelNeRF : `git clone https://github.com/sxyu/pixel-nerf.git`.

---

## Phase 2 : Implémentation et Entraînement NeRF (Semaines 2 - 3)
*Objectif : Comprendre le pipeline complet de la donnée brute au rendu 3D sur une scène fixe.*

1. **Acquisition des données :**
   * Prendre une vidéo courte (30 à 45 secondes) en tournant autour d'un objet réel bien éclairé, ou télécharger un dataset standard (ex: dataset *Lego* de NeRF).
2. **Extraction des poses (Preprocessing) :**
   * Lancer la commande `ns-process-data video --data ma_video.mp4 --output-dir ./data/mon_objet` pour extraire les images et calculer les matrices caméras via COLMAP.
3. **Entraînement du modèle :**
   * Lancer l'entraînement avec Nerfstudio : `ns-train nerfacto --data ./data/mon_objet`.
   * Observer l'apprentissage en temps réel via le viewer web de Nerfstudio (voir le nuage de bruit se transformer progressivement en objet 3D net).
4. **Export et Mesures :**
   * Générer une vidéo de trajectoire de caméra virtuelle et noter le temps d'entraînement ainsi que les métriques de qualité (PSNR, SSIM).

---

## Phase 3 : Mise en place et Réentraînement pixelNeRF (Semaines 4 - 6)
*Objectif : Passer à la reconstruction 3D à partir d'une seule image sans réentraînement par objet.*

1. **Test avec poids pré-entraînés (Inférence) :**
   * Télécharger les checkpoints officiels fournis par les auteurs de `pixel-nerf` (entraînés sur ShapeNet, catégories *Chairs* ou *Cars*).
   * Exécuter le script d'évaluation/inférence sur une image de test isolée pour valider que l'environnement fonctionne.
2. **Préparation d'un mini-dataset personnalisé (Fine-tuning) :**
   * Sélectionner une catégorie d'objets spécifique (ou utiliser un sous-ensemble de ShapeNet / DTU).
   * S'assurer que les matrices intrinsèques et extrinsèques des caméras respectent le format attendu par le dataloader de `pixel-nerf` (`transforms.json` / `.npz`).
3. **Fine-tuning (Réentraînement adapté) :**
   * Adapter le fichier de configuration (`conf/exp/...`) pour réduire le nombre d'époques et ajuster le *learning rate*.
   * Lancer le script `train/train.py` à partir des poids pré-entraînés sur vos nouvelles données.
4. **Préparation au Benchmark :**
   * Sélectionner 3 à 5 images de test (qui seront aussi données à ton binôme) et générer les vues 3D correspondantes avec pixelNeRF.