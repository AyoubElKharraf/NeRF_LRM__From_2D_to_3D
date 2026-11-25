# Rapport de Projet : De la 2D à la 3D
## Étude Comparative et Évolution des Champs de Radiance Neuronaux (NeRF, pixelNeRF & LRM)

**Master M2I — Semestre 3 — Module Traitement d'Images & Vision par Ordinateur**  
**Binôme :** Membre 1 (NeRF & pixelNeRF) & Samir (LRM / TripoSR)  
**Date :** Novembre 2026

---

## 1. Introduction & Contexte Scientifique

Le passage d'une ou plusieurs projections 2D (images RGB) à une représentation 3D exploitable constitue un **problème inverse mal posé** (*ill-posed inverse problem*). Lors de l'acquisition photographique, l'information de profondeur ($Z$) est projetée et perdue : une infinité de configurations 3D peuvent théoriquement produire la même projection 2D.

Ce projet explore et compare expérimentalement la rupture technologique majeure intervenue entre 2020 et 2024 dans le domaine de la **synthèse de nouvelles vues (*Novel View Synthesis - NVS*)** et de la **reconstruction 3D monoculaire** :
1. **NeRF (Mildenhall et al., ECCV 2020) :** Représentation implicite continue optimisée par descente de gradient sur une scène unique (*per-scene optimization*).
2. **pixelNeRF (Yu et al., CVPR 2021) :** Généralisation *few-view / single-view* grâce au conditionnement spatial d'un champ de radiance par un encodeur convolutif (ResNet).
3. **LRM (Hong et al., 2023 / TripoSR) :** Modèle fondation basé sur des Vision Transformers (DINO) générant un *Triplane NeRF* et un maillage 3D explicite en moins d'une seconde.

---

## 2. Modèle 1 : NeRF Classique (Mildenhall et al., 2020)

### 2.1. Fondements Mathématiques & Architecture
NeRF représente une scène 3D statique continue sous la forme d'une fonction continue vectorielle approximée par un Perceptron Multicouche (MLP) :
$$F_\Theta : (\mathbf{x}, \mathbf{d}) \longrightarrow (\mathbf{c}, \sigma)$$
- $\mathbf{x} = (x, y, z)$ : coordonnées spatiales 3D du point dans l'espace.
- $\mathbf{d} = (\theta, \phi)$ : direction de vue de la caméra (2D unitaire).
- $\mathbf{c} = (r, g, b)$ : couleur RGB émise dépendant de l'angle de vue (gestion des reflets non-lambertiens).
- $\sigma$ : densité volumique (opacité intrinsèque indépendante de la direction).

#### Positional Encoding (Encodage Positionnel)
Pour permettre au MLP d'apprendre les hautes fréquences spatiales (bords nets, textures fines), les coordonnées subissent une projection harmonique :
$$\gamma(p) = \Big( \sin(2^0 \pi p), \cos(2^0 \pi p), \dots, \sin(2^{L-1} \pi p), \cos(2^{L-1} \pi p) \Big)$$

#### Rendu Volumique Numérique (Volume Rendering)
La couleur $C(\mathbf{r})$ d'un rayon $\mathbf{r}(t) = \mathbf{o} + t\mathbf{d}$ est obtenue par quadrature numérique :
$$\hat{C}(\mathbf{r}) = \sum_{i=1}^{N} T_i \big(1 - \exp(-\sigma_i \delta_i)\big) \mathbf{c}_i \quad \text{avec} \quad T_i = \exp\left(-\sum_{j=1}^{i-1} \sigma_j \delta_j\right)$$
où $T_i$ désigne la transmittance accumulée (visibilité du rayon jusqu'au point $i$).

### 2.2. Protocole Expérimental & Résultats
- **Implémentation :** Framework Nerfstudio (`nerfacto`).
- **Dataset :** Dataset synthétique officiel *Lego* (Blender).
- **Entraînement :** Entraînement complet *from scratch* mené jusqu'à **30 000 itérations** sur GPU NVIDIA Tesla T4.
- **Rendus générés :**
  - Trajectoire orbitale à 5 000 itérations : `lego_5000_spiral.mp4` (ébauche géométrique visible, bords flous).
  - Checkpoint final : `step-000029999.ckpt` (242 Mo).
- **Métriques Officielles d'Évaluation :**
  - **PSNR :** `17.76 dB` ($\pm 3.14$)
  - **SSIM :** `0.7695` ($\pm 0.049$)
  - **LPIPS :** `0.1298` (distance perceptive basse, fidélité structurelle élevée)
  - **Débit de rayons :** 29 016 rayons / sec

---

## 3. Modèle 2 : pixelNeRF (Yu et al., CVPR 2021)

### 3.1. Rupture Conceptuelle & Mécanisme de Conditionnement Spatial
Alors que NeRF classique nécessite des dizaines de vues et des heures d'apprentissage par objet sans pouvoir généraliser, **pixelNeRF** permet la synthèse de nouvelles vues à partir d'**une seule image d'entrée (Single-View NVS)**.

1. **Extraction de Features 2D :**  
   L'image source $I$ traverse un encodeur CNN (**ResNet-34**), extrayant une pyramide de caractéristiques multi-échelles $W(I)$.
2. **Projection Perspective & Alignement Spatial :**  
   Pour chaque point 3D $X$ échantillonné le long d'un rayon de caméra virtuelle, on calcule sa projection perspective $\pi(X)$ sur le plan image source :
   $$f_X = W\big(\pi(X)\big)$$
   par interpolation bilinéaire sur la feature map.
3. **MLP Conditionné :**  
   Le MLP prédit couleur et densité en concaténant les coordonnées spatiales locales et le vecteur de caractéristiques visuelles :
   $$\text{pixelNeRF}(X, \mathbf{d}) = \text{MLP}\big(X, \mathbf{d}, f_X\big)$$

### 3.2. Expérimentation & Inférence Monoculaire
- **Méthodologie :** Inférence directe avec les poids officiels pré-entraînés sur le dataset **ShapeNet SRN Chairs** (`srn_chair`).
- **Entrée :** 1 seule vue 2D non calibrée (`-P '64'`).
- **Résultat généré :** Vidéo de rotation 360° `pixelnerf_chair_1view.mp4`.
- **Analyse critique des résultats :**
  - **Forces :** Inférence quasi instantanée sans aucun réentraînement par scène ; reconstruction fidèle de la silhouette globale et de l'assise.
  - **Limites :** Résolution native limitée ($128 \times 128$) propre au benchmark SRN ; présence de flou sur les faces masquées non visibles par la vue d'entrée (effet d'hallucination géométrique statistique).

---

## 4. Modèle 3 : LRM / TripoSR (Hong et al., 2023) — *Volet Samir*

*(Section réservée aux résultats de Samir : prétraitement `rembg`, Vision Transformer DINO, génération Triplane-NeRF instantanée < 1s et export de maillage 3D .obj/.glb).*

---

## 5. Synthèse Comparative du Benchmark

| Propriété / Métrique | 1. NeRF (Mildenhall 2020) | 2. pixelNeRF (Yu 2021) | 3. LRM / TripoSR (Hong 2023) |
| :--- | :--- | :--- | :--- |
| **Paradigme d'apprentissage** | Optimisation continue par scène (*overfitting*) | Modèle conditionné spatialement (CNN) | Modèle fondation Transformer (ViT + Triplane) |
| **Nombre de vues requises** | ~100 images denses | **1 seule image (monoculaire)** | **1 seule image (monoculaire)** |
| **Calibration caméra requise** | Oui (COLMAP SfM obligatoire) | Poses d'entraînement | **Non (In-the-wild)** |
| **Temps de reconstruction** | ~15 minutes d'entraînement GPU | ~3 minutes (volume rendering) | **< 1 seconde (Inférence directe)** |
| **Résolution native** | $800 \times 800$ | $128 \times 128$ | Haute résolution / Mesh explicite |
| **Sortie produite** | Densité continue volumique | Densité continue volumique | **Maillage 3D (.obj) + Rendu** |
| **Métriques (PSNR / SSIM)** | PSNR: **17.76 dB** / SSIM: **0.77** | Évaluation visuelle SRN | *À compléter par Samir* |

---

## 6. Conclusion Générale

Ce projet met en évidence l'évolution fondamentale du domaine de la vision 3D en 4 ans :
- D'une approche mathématique purement géométrique mais rigide (NeRF),
- À l'introduction du conditionnement par apprentissage profond convolutif (pixelNeRF),
- Jusqu'aux modèles fondation multimodaux modernes capables de générer un univers 3D complet en un battement de cil (LRM).
