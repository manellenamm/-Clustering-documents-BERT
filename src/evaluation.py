from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
import numpy as np


def evaluate_clustering(embeddings, predicted_labels, true_labels, model_name="Model"):
    print(f"\n===== Évaluation {model_name} =====")

    valid_mask = np.array(predicted_labels) != -1
    valid_labels = np.array(predicted_labels)[valid_mask]

    if len(set(valid_labels)) > 1:
        sil = silhouette_score(embeddings[valid_mask], valid_labels)
        print(f"Silhouette Score : {sil:.4f}")

    ari = adjusted_rand_score(true_labels, predicted_labels)
    nmi = normalized_mutual_info_score(true_labels, predicted_labels)

    print(f"ARI : {ari:.4f}")
    print(f"NMI : {nmi:.4f}")