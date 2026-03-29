# ======================
# 5. Choix manuel de k après observation du coude
# ======================
best_k = 4   # change cette valeur après avoir vu le graphe

print(f"KMeans clustering avec k = {best_k} ...")
kmeans_labels = kmeans_clustering(embeddings, n_clusters=best_k)

# ======================
# 6. DBSCAN
# ======================
print("DBSCAN clustering...")
dbscan_labels = dbscan_clustering(embeddings)

# ======================
# 7. Réduction de dimension
# ======================
print("Réduction de dimension...")
reduced = reduce_dimensions(embeddings)

# ======================
# 8. Visualisation des clusters
# ======================
plot_clusters(reduced, kmeans_labels, f"KMeans Clustering (k={best_k})")
plot_clusters(reduced, dbscan_labels, "DBSCAN Clustering")

# ======================
# 9. Évaluation
# ======================
kmeans_score = evaluate_clustering(embeddings, kmeans_labels)
dbscan_score = evaluate_clustering(embeddings, dbscan_labels)

print("Silhouette KMeans :", kmeans_score)
print("Silhouette DBSCAN :", dbscan_score)