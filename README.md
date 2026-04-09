# 🧠 Clustering de documents avec BERT

## 📌 Description

Ce projet propose une approche de **clustering non supervisé de documents textuels** à l’aide de modèles de type BERT.
Les documents sont transformés en vecteurs (embeddings) via des modèles issus de HuggingFace, puis regroupés à l’aide d’algorithmes de clustering comme K-Means et DBSCAN.

L’objectif est d’évaluer dans quelle mesure les embeddings capturent la structure sémantique des textes.

---

## 🎯 Objectifs

* Générer des représentations vectorielles de documents avec BERT
* Appliquer des algorithmes de clustering non supervisés
* Réduire la dimension des données (PCA / UMAP)
* Évaluer la qualité du clustering
* Visualiser les clusters obtenus

---

## ⚙️ Fonctionnalités

* Chargement de datasets textuels (ex : DBpedia_14)
* Génération d’embeddings avec modèles Transformers
* Réduction de dimension (PCA, UMAP)
* Clustering avec K-Means et DBSCAN
* Évaluation avec :

  * Silhouette Score
  * ARI (Adjusted Rand Index)
  * NMI (Normalized Mutual Information)
* Visualisation des clusters

---

## 🗂️ Structure du projet

clustering-documents-bert/
│
├── src/
│   ├── main.py
│   ├── preprocessing.py
│   ├── embedding.py
│   ├── clustering.py
│   ├── evaluation.py
│   └── visualization.py
│
├── README.md
├── LICENSE
├── requirements.txt
└── .gitignore

---

## 🧪 Dataset

Le projet utilise le dataset DBpedia_14, contenant 14 catégories de documents.

⚠️ Bien que ce dataset soit supervisé (avec labels), il est utilisé ici dans un cadre **non supervisé** pour évaluer la qualité du clustering.

---

## 📦 Installation

1. Cloner le projet :

git clone https://github.com/TON_USERNAME/clustering-documents-bert.git
cd clustering-documents-bert

2. Créer un environnement virtuel :

python -m venv venv
venv\Scripts\activate

3. Installer les dépendances :

pip install -r requirements.txt

---

## 🚀 Utilisation

Lancer le script principal :

python src/main.py

---

## 📊 Évaluation

Les performances du clustering sont mesurées avec :

* Silhouette Score → qualité interne
* ARI → comparaison avec les vrais labels
* NMI → similarité clustering / classes

Ces métriques permettent d’évaluer la pertinence des embeddings.

---

## 📈 Résultats (exemple)

| Méthode | Silhouette | ARI    | NMI  |
| ------- | ---------- | ----  | ---- |
| K-Means | 0.53       | 0.5125|0.7081|
| DBSCAN  |  0.5683    | 0.6306 |0.7643|

👉 Dbscan donne de meilleurs résultats dans ce contexte.

---

## 🧠 Choix techniques

* BERT / Sentence Transformers : représentation sémantique avancée
* K-Means : simple et efficace
* DBSCAN : détection de bruit mais sensible aux paramètres
* UMAP / PCA : réduction de dimension et visualisation

---

## 📚 Bibliothèques utilisées

Ce projet repose sur des bibliothèques open source :

* transformers (HuggingFace)
* sentence-transformers
* scikit-learn
* numpy
* pandas
* matplotlib
* umap-learn
* datasets

---

## 📜 Licence

Ce projet est distribué sous licence MIT.

Cette licence a été choisie car elle est simple, permissive et adaptée à un projet académique, permettant la réutilisation, la modification et la distribution du code.


## 🔥 Remarques

* Projet entièrement reproductible
* Utilisation exclusive de bibliothèques open source
* Code structuré et documenté
---
