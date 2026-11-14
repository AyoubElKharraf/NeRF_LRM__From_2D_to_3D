# Article 3 : LRM — Large Reconstruction Model (Hong et al., 2023)

## 1. Comprendre le problème fondamental
Les premières méthodes de reconstruction à partir d'une seule image (comme pixelNeRF) restent limitées à des catégories d'objets spécifiques et simples (souvent sur fond blanc, comme les chaises ou voitures de ShapeNet). Lorsqu'on leur donne une vraie photo complexe prise avec un smartphone (*in-the-wild*), la géométrie des faces cachées manque souvent de cohérence ou de détails fins, car les architectures CNN + MLP manquent de capacité de modélisation à grande échelle.

## 2. L'idée centrale et l'objectif
**Objectif :** Créer un modèle fondationnel universel capable de transformer **n'importe quelle image 2D réelle** en un modèle 3D complet, détaillé et cohérent sous 360 degrés, en moins d'une seconde.

**Idée centrale :** Appliquer la loi d'échelle (*Scaling Laws*) des grands modèles de langage (LLM) à la 3D. LRM utilise une architecture massive basée sur des **Transformers** pour traduire directement les caractéristiques visuelles d'une image 2D vers une représentation 3D structurée de type **Triplane-NeRF**, entraînée sur près d'un million d'objets 3D variés.

## 3. Le fonctionnement étape par étape
1. **Encodage Visuel Avancé (DINO) :** L'image 2D d'entrée passe dans un encodeur Vision Transformer pré-entraîné (**DINO**), reconnu pour sa capacité exceptionnelle à comprendre la structure sémantique et géométrique des objets.
2. **Le Cœur Transformer (Image-to-Triplane) :** Un grand décodeur Transformer prend des requêtes apprenables et utilise des mécanismes d'attention croisée (*Cross-Attention*) avec les caractéristiques DINO pour projeter l'information 2D dans l'espace 3D.
3. **Représentation Triplane :** Au lieu de prédire directement les pixels, le Transformer génère un **Triplane** (trois plans orthogonaux $XY, XZ, YZ$ contenant des caractéristiques 3D). Cette structure est beaucoup plus rapide et riche qu'un simple MLP.
4. **Décodage NeRF et Rendu :** Un petit MLP léger lit les informations sur ces trois plans pour prédire la densité ($\sigma$) et la couleur ($R, G, B$) de chaque point 3D, puis effectue le rendu de volume pour afficher l'objet sous n'importe quel angle (ou l'exporter en maillage 3D / Mesh).