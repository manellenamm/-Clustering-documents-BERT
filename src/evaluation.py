from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
import numpy as np


def evaluate_clustering(embeddings, cluster_labels, true_labels=None):
    unique_labels = set(cluster_labels)

    if len(unique_labels) <= 1:
        print("Impossible de calculer le silhouette score : un seul cluster détecté.")
    else:
        silhouette = silhouette_score(embeddings, cluster_labels)
        print("Silhouette Score :", silhouette)

    if true_labels is not None:
        ari = adjusted_rand_score(true_labels, cluster_labels)
        nmi = normalized_mutual_info_score(true_labels, cluster_labels)
        print("ARI :", ari)
        print("NMI :", nmi)