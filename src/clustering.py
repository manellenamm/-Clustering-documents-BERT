from sklearn.cluster import KMeans, DBSCAN , AgglomerativeClustering


def kmeans_clustering(embeddings, n_clusters=2):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(embeddings)
    return labels


def dbscan_clustering(embeddings, eps=0.3, min_samples=5):
    model = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
    labels = model.fit_predict(embeddings)
    return labels

def agglomerative_clustering(embeddings, n_clusters=4):
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='euclidean',
        linkage='ward'
    )
    labels = model.fit_predict(embeddings)
    return labels