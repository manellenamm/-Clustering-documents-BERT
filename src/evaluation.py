from sklearn.metrics import silhouette_score

def evaluate_clustering(embeddings, labels):
    if len(set(labels)) > 1:
        score = silhouette_score(embeddings, labels)
        return score
    return -1