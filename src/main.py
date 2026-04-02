from datasets import load_dataset
from embedding import generate_embeddings
from clustering import kmeans_clustering, dbscan_clustering
from evaluation import evaluate_clustering
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import random
from sklearn.datasets import fetch_20newsgroups

import os


def main():
    os.makedirs("data", exist_ok=True)

    # ======================
    # 1. Charger les données
    # ======================
    print("Chargement des données...")
    """
    dataset = load_dataset("ag_news")

    n_samples = 1000
    indices = random.sample(range(len(dataset["train"])), n_samples)

    docs = [dataset["train"]["text"][i] for i in indices]
    true_labels = [dataset["train"]["label"][i] for i in indices]

    print("Labels présents dans AG News :", set(dataset["train"]["label"]))
    print("Nombre de documents utilisés :", len(docs))
    """

    categories = [
        'rec.sport.baseball',
        'talk.politics.misc'
    ]

    dataset = fetch_20newsgroups(subset='train', categories=categories)

    docs = dataset.data[:300]
    true_labels = dataset.target[:300]

   

    # ======================
    # 2. Génération des embeddings
    # ======================
    print("Génération des embeddings...")
    embeddings = generate_embeddings(docs)
    print("Taille des embeddings :", embeddings.shape)

    # ======================
    # 3. Clustering K-Means
    # ======================
    print("\nClustering K-Means...")
    kmeans_labels = kmeans_clustering(embeddings)
    print("Labels K-Means :", kmeans_labels)

    # ======================
    # 4. Clustering DBSCAN
    # ======================
    """
    print("\nClustering DBSCAN...")
    dbscan_labels = dbscan_clustering(embeddings)
    print("Labels DBSCAN :", dbscan_labels)
    """


    # ======================
    # 5. Évaluation
    # ======================
    print("\nÉvaluation K-Means :")
    evaluate_clustering(embeddings, kmeans_labels, true_labels)
    """
    print("\nÉvaluation DBSCAN :")
    evaluate_clustering(embeddings, dbscan_labels, true_labels)
    """

    # ======================
    # 6. Réduction de dimension avec PCA
    # ======================
    print("\nRéduction de dimension avec PCA...")
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings)

    # ======================
    # 7. Visualisation K-Means
    # ======================
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced[:, 0], reduced[:, 1], c=kmeans_labels)
    plt.title("Clustering des documents - KMeans")
    plt.xlabel("Composante principale 1")
    plt.ylabel("Composante principale 2")
    plt.show()


    """
    # ======================
    # 8. Visualisation DBSCAN
    # ======================
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced[:, 0], reduced[:, 1], c=dbscan_labels)
    plt.title("Clustering des documents - DBSCAN")
    plt.xlabel("Composante principale 1")
    plt.ylabel("Composante principale 2")
    plt.show()
    """



if __name__ == "__main__":
    main()