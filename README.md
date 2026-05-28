# 🎬 Système de Recommandation LightGCN - MovieLens

> Projet ANI-IA 4068 : IA et Applications  
> Ecole Nationale Polytechnique de Yaoundé 1  
> Auteur : **MEZAGO WILFRIED AYMAR** - Matricule : 22P001  
> Superviseur : **MR BITHA JUNIOR**  
> Année académique : 2025 - 2026

---

## 📌 Description

Ce projet implémente un système de recommandation de films basé sur le modèle **LightGCN** (Light Graph Convolutional Network) en utilisant le dataset **MovieLens ml-latest-small**.

LightGCN est un modèle de filtrage collaboratif basé sur les Réseaux Neuraux Convolutifs pour Graphes (GCN). Il modélise les interactions utilisateur-film sous forme de graphe biparti et apprend des représentations (embeddings) pour chaque utilisateur et chaque film via un mécanisme de passage de messages multicouches.

> Référence : He et al., *LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation*, ACM SIGIR 2020. [arxiv.org/abs/2002.02126](https://arxiv.org/abs/2002.02126)

---

## 📊 Résultats obtenus

| Métrique | Score |
|---|---|
| Recall@10 | 0.0492 |
| NDCG@10 | 0.0587 |
| Recall@20 | 0.0856 |
| NDCG@20 | 0.0702 |

> Entraînement sur 50 epochs. Le papier original atteint Recall@20 ≈ 0.139 avec 1000 epochs.

---

## 🗂️ Structure du projet

```
LightGCN_Recom_Syst/
├── data/   # gitignore — non versionné
│   ├── ml-latest-small/   # Dataset MovieLens brut
│   ├── train.csv  # Split entraînement (79.7%)
│   ├── val.csv   # Split validation (9.7%)
│   ├── test.csv  # Split test (10.5%)
│   ├── preprocessed.pkl   # Mappings et dictionnaires
│   ├── interaction_matrix.npz   # Matrice d'interaction creuse
│   ├── distribution_notes.png   # Graphique EDA
│   ├── distribution_films.png   # Graphique EDA
│   ├── analyse_temporelle.png   # Graphique EDA
│   ├── courbes_entrainement.png   # Courbes de loss
│   ├── metriques_evaluation.png   # Recall et NDCG
│   └── tsne_embeddings.png   # Visualisation t-SNE
├── notebooks/
│   └── lightgcn_recom_syst.ipynb   # Notebook principal
├── models/  # gitignore — non versionné
│   └── lightgcn_final.pth # Modèle entraîné
├── env/    # gitignore — environnement virtuel
├── requirements.txt    # Dépendances du projet
├── README.md
└── .gitignore
```

---

## ⚙️ Installation

### Prérequis
- Python 3.11+
- GPU NVIDIA avec CUDA (recommandé)

### Étapes

**1. Cloner le dépôt**
```bash
git clone https://github.com/ton-username/LightGCN_Recom_Syst.git
cd LightGCN_Recom_Syst
```

**2. Créer et activer l'environnement virtuel**
```bash
# Windows
python -m venv env
.\env\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv env
source env/bin/activate
```

**3. Installer PyTorch avec support CUDA**
```bash
# CUDA 12.1 (compatible CUDA 13.0)
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121

# CPU uniquement
pip install torch==2.3.0 torchvision==0.18.0
```

**4. Installer PyTorch Geometric**
```bash
pip install torch-geometric --extra-index-url https://data.pyg.org/whl/torch-2.3.0+cu121.html
```

**5. Installer les autres dépendances**
```bash
pip install -r requirements.txt
```

**6. Télécharger le dataset MovieLens**

Télécharger [ml-latest-small.zip](https://files.grouplens.org/datasets/movielens/ml-latest-small.zip) et extraire dans `data/ml-latest-small/`.

---

## 🚀 Utilisation

Ouvrir et exécuter le notebook dans l'ordre :

```
notebooks/lightgcn_recom_syst.ipynb
```

Le notebook est organisé en 4 phases :

| Phase | Contenu |
|---|---|
| Phase 1 | Chargement, EDA, nettoyage, split, graphe biparti |
| Phase 2 | Architecture LightGCN, matrice d'adjacence normalisée |
| Phase 3 | Entraînement BPR, courbes de loss, sauvegarde |
| Phase 4 | Evaluation Recall/NDCG, recommandations Top-K, t-SNE |

---

## 🧠 Architecture du modèle

```
Embeddings initiaux  →  Propagation K couches  →  Moyenne des couches  →  Scores
   (602 + 3643)              (K = 3)               Embedding final        Top-K
      × 64 dims         A_norm * E par couche      (1/K+1) Σ E_k
```

**Hyperparamètres :**

| Paramètre | Valeur |
|---|---|
| Embedding dimension | 64 |
| Nombre de couches | 3 |
| Epochs | 50 |
| Batch size | 1024 |
| Learning rate | 1e-3 |
| Lambda régularisation | 1e-4 |
| Optimiseur | Adam |
| Fonction de perte | BPR |

---

## 📦 Dépendances principales

| Librairie | Version | Usage |
|---|---|---|
| torch | 2.3.0 | Framework Deep Learning |
| torch-geometric | 2.7.0 | Convolution sur graphes |
| pandas | 2.2.2 | Manipulation des données |
| numpy | 1.26.4 | Calcul numérique |
| scipy | 1.13.0 | Matrices creuses |
| scikit-learn | 1.5.0 | t-SNE, métriques |
| matplotlib | 3.9.0 | Visualisation |
| seaborn | 0.13.2 | Graphiques statistiques |

---

## 📈 Données

| Caractéristique | Brut | Après filtrage |
|---|---|---|
| Utilisateurs | 610 | 602 |
| Films | 9 724 | 3 643 |
| Interactions | 100 836 | 90 109 |
| Densité | 1.70% | 3.28% |
| Période | 1996 - 2018 | - |

**Filtrage appliqué :** utilisateurs avec ≥ 20 notes, films avec ≥ 5 notes.  
**Split temporel :** 80% train / 10% validation / 10% test.

---

## 👥 Auteur

MEZAGO Wilfried Aymar - Ecole Nationale Polytechnique de Yaoundé 1 - 2025/2026.
