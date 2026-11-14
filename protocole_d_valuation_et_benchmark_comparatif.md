# Proposition : Protocole d'Évaluation & Structure de Comparaison

Pour que votre projet ait une vraie valeur scientifique lors de la rédaction du rapport et de la soutenance, il est essentiel de ne pas se contenter de montrer de belles images. Il faut **comparer objectivement** les trois modèles sur les mêmes bases.

## 1. Les Métriques Standards à Calculer
Dans tous les articles NeRF, pixelNeRF et LRM, trois métriques mathématiques sont utilisées pour comparer l'image générée par le modèle avec la vraie photo de référence :

1. **PSNR (Peak Signal-to-Noise Ratio) — *Plus haut = Meilleur* :**
   * Mesure la fidélité globale des pixels (basée sur l'erreur quadratique moyenne). Une valeur supérieure à 25-30 dB indique une très bonne reconstruction.
2. **SSIM (Structural Similarity Index) — *Proche de 1 = Meilleur* :**
   * Évalue la similarité structurelle (luminance, contraste, contours) telle que perçue par l'œil humain.
3. **LPIPS (Learned Perceptual Image Patch Similarity) — *Plus bas = Meilleur* :**
   * Utilise un réseau de neurones pour mesurer la distance perceptuelle (les détails fins et textures). C'est la métrique la plus fiable pour juger du réalisme visuel.
4. **Métriques Ingénierie / MLOps (Votre valeur ajoutée) :**
   * **Temps d'entraînement / d'adaptation** pour un nouvel objet.
   * **Temps d'inférence** pour générer une vue 2D (en secondes/millisecondes).
   * **Consommation VRAM GPU** requise.

---

## 2. Le Tableau Comparatif Final (À remplir avec vos résultats)

| Critère d'Évaluation | NeRF (Nerfstudio) | pixelNeRF (PyTorch) | LRM (OpenLRM / TripoSR) |
| :--- | :--- | :--- | :--- |
| **Nombre d'images d'entrée** | 30 à 100+ photos | **1 seule photo** | **1 seule photo** |
| **Temps d'adaptation à un nouvel objet** | ~15 min à plusieurs heures | **0 sec** (Inférence directe) | **0 sec** (Inférence directe) |
| **Qualité des détails visibles** | Excellente (Photoréaliste) | Moyenne à Bonne (parfois floue) | Très bonne |
| **Cohérence des faces cachées (360°)** | Parfaite (car vues en entrée) | Limitée aux catégories apprises | Très bonne (grâce au Transformer) |
| **Généralisation (Photos réelles)** | Oui, mais nécessite réentraînement | Difficile hors dataset (ShapeNet) | Excellente (*In-the-wild*) |
| **Score PSNR moyen (sur vos tests)** | *[À compléter]* | *[À compléter]* | *[À compléter]* |
| **Score SSIM / LPIPS (sur vos tests)** | *[À compléter]* | *[À compléter]* | *[À compléter]* |

---

## 3. Plan Type Recommandé pour Votre Rapport Final
1. **Introduction & Problématique :** De la modélisation 3D classique aux champs de radiance neuronaux. Le compromis entre qualité par scène et généralisation à partir d'une seule image.
2. **État de l'Art Technique :** Synthèse des 3 architectures (NeRF, pixelNeRF, LRM) — *basée sur vos fichiers 01, 02 et 03*.
3. **Architecture MLOps & Pipelines Expérimentaux :** Présentation des outils utilisés (COLMAP, Nerfstudio, ResNet, DINO/Triplane), formats de données et configuration matérielle.
4. **Résultats Expérimentaux & Benchmark :** Présentation du tableau comparatif ci-dessus, illustrations visuelles côte à côte sur les mêmes objets de test, et analyse des cas d'échec (ex: objets transparents, arrière-plans complexes).
5. **Conclusion & Perspectives :** Bilan sur l'évolution vers le *3D Gaussian Splatting* (l'étape suivante après NeRF/LRM pour le temps réel).