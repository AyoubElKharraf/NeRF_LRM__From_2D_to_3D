# Feuille de Route Opérationnelle : Membre 2 (LRM - Large Reconstruction Model)

Ton rôle couvre l'état de l'art moderne à grande échelle (**LRM**), afin de démontrer comment les Transformers résolvent les limites de généralisation des premières approches NeRF.

---

## Phase 1 : Préparation de l'Environnement et Choix du Dépôt (Semaine 1)
*Note : Le code officiel d'Adobe (LRM original) n'étant pas entièrement open-source, il faut s'appuyer sur les implémentations open-source de référence reconnues par la communauté scientifique.*

* **Choix de l'implémentation :**
  * **Option A (Recommandée - Fidèle à l'architecture LRM) :** Cloner **OpenLRM** (`3DTopia/OpenLRM` sur GitHub), l'implémentation open-source complète de LRM en PyTorch.
  * **Option B (Alternative ultra-rapide) :** Cloner **TripoSR** (`VAST-AI-Research/TripoSR`), basé sur l'architecture LRM et optimisé pour tourner sur des GPU standards (ex: T4 sur Google Colab).
* **Configuration :** Créer un environnement Conda dédié (Python 3.10+, PyTorch + CUDA) et installer les dépendances (notamment `xformers`, `transformers` de Hugging Face, et `rembg` pour le détourage automatique d'arrière-plan).

---

## Phase 2 : Prise en Main et Inférence "In-the-Wild" (Semaines 2 - 3)
*Objectif : Tester la capacité de généralisation immédiate sur des photos réelles.*

1. **Téléchargement des poids pré-entraînés :**
   * Récupérer les modèles pré-entraînés directement via Hugging Face (intégrés dans les scripts d'OpenLRM / TripoSR).
2. **Pipeline de pré-traitement d'image :**
   * Mettre en place le script de nettoyage d'image (suppression automatique du fond via `rembg` et centrage de l'objet), étape indispensable pour les modèles de type LRM.
3. **Tests d'inférence :**
   * Tester le modèle sur des photos prises avec un smartphone (objets du quotidien) et mesurer le temps exact de génération (généralement entre 0.5 et 2 secondes).
   * Exporter les résultats sous deux formats : vidéo de rotation 360° et maillage 3D explicite (`.obj` ou `.glb`).

---

## Phase 3 : Fine-Tuning et Analyse Architecturale (Semaines 4 - 6)
*Objectif : Comprendre l'impact des données d'entraînement et préparer la comparaison.*

1. **Analyse du Triplane et de l'encodeur DINO :**
   * Isoler et documenter comment l'image passe de l'encodeur Vision Transformer (DINO) au décodeur Triplane-NeRF dans le code.
2. **Adaptation / Fine-tuning léger (Optionnel selon puissance GPU) :**
   * Si un GPU avec suffisamment de VRAM (ex: A100 ou V100) est disponible, lancer un fine-tuning sur un petit jeu de données d'objets rendus sous plusieurs angles (format Objaverse / GSO) en gelant l'encodeur DINO pour n'entraîner que les dernières couches du décodeur.
3. **Préparation au Benchmark :**
   * Passer exactement les **mêmes 3 à 5 images de test** que ton binôme (Membre 1) dans ton modèle LRM pour produire les rendus comparatifs.