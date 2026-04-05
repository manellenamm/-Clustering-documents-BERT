from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import numpy as np

def kmeans_clustering(embeddings, n_clusters):
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    labels = model.fit_predict(embeddings)
    return labels

def elbow_method(embeddings, k_range=range(2, 21)):
    inertias = []

    for k in k_range:
        model = KMeans(n_clusters=k, random_state=42)
        model.fit(embeddings)
        inertias.append(model.inertia_)
        print(f"k={k} | inertia={model.inertia_:.4f}")

    plt.figure(figsize=(8, 6))
    plt.plot(list(k_range), inertias, marker="o")
    plt.title("Elbow Method pour K-Means")
    plt.xlabel("Nombre de clusters K")
    plt.ylabel("Inertie")
    plt.xticks(list(k_range))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return inertias

def dbscan_clustering(embeddings, eps, min_samples):
    model = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
    labels = model.fit_predict(embeddings)
    return labels



def find_eps(embeddings, k=10):
    neigh = NearestNeighbors(n_neighbors=k, metric="cosine")
    neigh.fit(embeddings)

    distances, _ = neigh.kneighbors(embeddings)
    
    # prendre la distance du k-ième voisin
    k_distances = distances[:, -1]
    
    # trier
    k_distances = np.sort(k_distances)

    plt.plot(k_distances)
    plt.title("K-distance plot")
    plt.ylabel("Distance")
    plt.xlabel("Points triés")
    plt.show()