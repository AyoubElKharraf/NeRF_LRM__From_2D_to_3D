# Article 1 : NeRF — Neural Radiance Fields (Mildenhall et al., ECCV 2020)

## 1. Comprendre le problème fondamental
Traditionnellement, pour modéliser un objet ou une scène en 3D sur ordinateur, on utilise des représentations explicites :
* **Les maillages (Meshes) :** Des assemblages de triangles qui deviennent très lourds et complexes à manipuler pour modéliser des éléments fins ou non rigides (cheveux, fumée, eau).
* **Les voxels :** Des grilles de "pixels 3D" dont la consommation mémoire explose dès que l'on augmente la résolution.
* **Les nuages de points :** Des ensembles de points discrets qui manquent de continuité visuelle.

**Le défi :** Comment représenter une scène 3D de manière ultra-compacte, continue et photoréaliste, tout en gérant naturellement les effets lumineux complexes comme la transparence et les reflets spéculaires ?

## 2. L'idée centrale et l'objectif
**Objectif :** Synthétiser de nouvelles vues photoréalistes d'une scène 3D complexe à partir d'un ensemble d'images 2D prises sous différents angles connus.

**Idée centrale :** Remplacer toute structure géométrique 3D explicite par une **fonction implicite continue** apprise par un simple réseau de neurones entièrement connecté (MLP). Le réseau de neurones *devient* lui-même le fichier 3D de la scène.

## 3. Le fonctionnement étape par étape
1. **Calibration des caméras (Pré-traitement) :** À partir de plusieurs photos 2D d'un objet, un outil externe (COLMAP) calcule la position $(x, y, z)$ et l'orientation $(\theta, \phi)$ exactes de la caméra pour chaque photo.
2. **Interrogation du réseau (MLP) :** Le réseau prend en entrée une coordonnée spatiale 3D $(x, y, z)$ et une direction de regard $(\theta, \phi)$. Il prédit en sortie deux valeurs :
   * La **densité de volume ($\sigma$)** : indique s'il y a de la matière solide ou du vide à cet endroit.
   * La **couleur RVB ($\mathbf{c}$)** : la couleur émise par ce point sous cet angle précis (permettant de gérer les reflets qui changent selon le point de vue).
3. **Encodage Positionnel :** Avant d'entrer dans le réseau, les coordonnées sont projetées via des fonctions sinus/cosinus pour aider le MLP à capturer les détails très fins (hautes fréquences).
4. **Rendu par Lancer de Rayons (Volume Rendering) :** Pour générer une image 2D, l'algorithme lance un rayon virtuel depuis la caméra à travers chaque pixel, échantillonne des dizaines de points le long du rayon, interroge le réseau, et accumule les couleurs et densités pour calculer la couleur finale du pixel.
5. **Optimisation :** On compare l'image générée à la vraie photo (erreur MSE) et on ajuste les poids du réseau jusqu'à ce qu'il connaisse la scène par cœur.