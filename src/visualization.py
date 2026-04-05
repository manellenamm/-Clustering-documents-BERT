import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import umap

"""
def reduce_for_clustering(embeddings, n_components=205):
    
    #Réduction de dimension avant clustering.
    #Ici on utilise PCA aussi pour garder le projet simple.
 
    reducer = PCA(n_components=n_components)
    reduced = reducer.fit_transform(embeddings)
    cumsum = np.cumsum(reducer.explained_variance_ratio_)
    n_95 = np.argmax(cumsum >= 0.95) + 1
    print("Nombre de composantes pour 95% variance :", n_95)
    print(f"Réduction avec PCA pour clustering : {reduced.shape}")

    return reduced

"""


def reduce_for_clustering(embeddings):
    reducer = umap.UMAP(
        n_neighbors=30,   # était 15 → plus grand = structure globale plus claire
        min_dist=0.0,     # était 0.1 → 0.0 = clusters plus compacts pour DBSCAN
        n_components=10,  # était 18 → moins de dims = moins de bruit
        metric="cosine",
        random_state=42
    )

    reduced = reducer.fit_transform(embeddings)
    print(f"Réduction avec UMAP : {reduced.shape}")
    return reduced
"""
def reduce_for_clustering(embeddings):
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=18,
        metric="cosine",
        random_state=42
    )

    reduced = reducer.fit_transform(embeddings)
    print(f"Réduction avec UMAP : {reduced.shape}")
    return reduced

"""
def reduce_for_visualization(embeddings):
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        metric="cosine",
        random_state=42
    )

    reduced = reducer.fit_transform(embeddings)
    print(f"Réduction avec UMAP : {reduced.shape}")
    return reduced




#PCA
"""
def reduce_for_visualization(embeddings):
    
    #Réduction en 2D pour visualisation avec PCA.
    
    reducer = PCA(n_components=2)
    reduced = reducer.fit_transform(embeddings)
    print(f"Réduction avec PCA pour visualisation : {reduced.shape}")
    return reduced


"""

def plot_clusters(points_2d, labels, title):
    labels = np.array(labels)
    unique_labels = sorted(set(labels))
    
    # Palette de couleurs discrètes (une couleur par cluster)
    colors = plt.cm.get_cmap("tab20", len(unique_labels))

    plt.figure(figsize=(10, 7))
    for i, label in enumerate(unique_labels):
        mask = labels == label
        color = "black" if label == -1 else colors(i)
        name = "Outliers" if label == -1 else f"Cluster {label}"
        plt.scatter(
            points_2d[mask, 0], points_2d[mask, 1],
            c=[color], label=name, alpha=0.7, s=30
        )

    plt.title(title)
    plt.xlabel("Composante principale 1")
    plt.ylabel("Composante principale 2")
    plt.legend(loc="best", markerscale=1.5, fontsize=7, ncol=2)
    plt.tight_layout()
    plt.show()