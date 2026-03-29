from sklearn.cluster import KMeans, DBSCAN , AgglomerativeClustering

def kmeans_clustering(embeddings, n_clusters=4):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(embeddings)
    return labels

def dbscan_clustering(embeddings):
    model = DBSCAN(eps=0.5, min_samples=5)
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