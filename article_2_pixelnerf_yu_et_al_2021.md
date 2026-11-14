# Article 2 : pixelNeRF (Yu et al., CVPR 2021)

## 1. Comprendre le problème fondamental
Bien que le NeRF original produise des résultats époustouflants, il souffre d'un défaut majeur en pratique : **il est strictement "personnalisé" (overfit) à une seule scène**. 
* Il exige des dizaines, voire des centaines de photos sous tous les angles pour un seul objet.
* Il nécessite plusieurs heures d'entraînement intensif sur GPU pour apprendre cette unique scène.
* Dès que l'on change d'objet, les poids du réseau sont inutilisables et il faut tout recommencer de zéro. Il est incapable de généraliser à une photo inconnue.

## 2. L'idée centrale et l'objectif
**Objectif :** Permettre la reconstruction 3D et la synthèse de nouvelles vues à partir d'**une seule image 2D** (ou de très peu de vues), de manière instantanée et sur des objets que le modèle n'a jamais vus auparavant.

**Idée centrale :** Transformer le NeRF en un modèle **conditionné par l'image**. Au lieu d'apprendre une scène par cœur dans les poids du MLP, on apprend au réseau des *a priori* géométriques généraux sur des milliers d'objets (ex: à quoi ressemble l'arrière d'une chaise ou d'une voiture) grâce à un extracteur de caractéristiques visuelles (CNN).

## 3. Le fonctionnement étape par étape
1. **Extraction des caractéristiques (Encodeur CNN) :** On donne une image 2D source en entrée à un réseau convolutif (un **ResNet34** pré-entraîné). Ce réseau produit une carte de caractéristiques visuelles (*feature map*) qui capture les textures et les formes de l'image.
2. **Projection et Alignement Spatial :** Lorsqu'on veut évaluer un point $(x, y, z)$ dans l'espace 3D pour générer une nouvelle vue, on projette mathématiquement ce point 3D sur le plan de l'image source 2D pour récupérer le vecteur visuel correspondant exactement à cet endroit.
3. **Prédiction Conditionnée :** Le MLP reçoit non seulement les coordonnées $(x, y, z)$ et la direction de regard (exprimées dans le repère de la caméra source), mais aussi ce **vecteur de caractéristiques visuelles** issu du CNN.
4. **Inférence Instantanée :** Parce que le modèle a été entraîné au préalable sur de vastes bases de données multi-objets (comme ShapeNet), il lui suffit d'une seule passe avant (*forward pass*) de quelques millisecondes sur une nouvelle photo pour prédire la densité ($\sigma$) et la couleur ($\mathbf{c}$) de l'objet en 3D.