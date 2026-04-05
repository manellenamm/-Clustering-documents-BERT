import os
import numpy as np
import pandas as pd
import random
import plotly.graph_objects as go
import plotly.express as px
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors

from data_loader import load_dbpedia_data
from preprocessing import clean_documents
from embedding import generate_embeddings
from clustering import kmeans_clustering, elbow_method, find_eps, dbscan_clustering
from evaluation import evaluate_clustering
from visualization import reduce_for_clustering, reduce_for_visualization, plot_clusters


# ─────────────────────────────────────────────────────────────────────────────
# Fonctions de tuning DBSCAN
# ─────────────────────────────────────────────────────────────────────────────

def plot_kdistance(embeddings, k=6):
    """
    Affiche la courbe k-distance.
    Le coude de la courbe indique la valeur optimale de eps.
    k doit être égal à ton min_samples.
    """
    nbrs = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(embeddings)
    distances, _ = nbrs.kneighbors(embeddings)
    k_distances = np.sort(distances[:, -1])[::-1]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=k_distances,
        mode="lines",
        line=dict(color="royalblue", width=2),
        name=f"{k}-distance",
    ))
    fig.update_layout(
        title=f"k-Distance Plot (k={k}) — le coude = eps optimal",
        xaxis_title="Points (triés par distance décroissante)",
        yaxis_title=f"Distance au {k}e voisin",
        height=400,
    )
    fig.show()
    return k_distances


def grid_search_dbscan(embeddings, true_labels=None,
                       eps_range=None, min_samples_range=None,
                       max_outlier_ratio=0.2):
    """
    Teste toutes les combinaisons (eps, min_samples).
    Retourne un DataFrame trié par Silhouette Score.
    """
    if true_labels is not None:
        true_labels = np.array(true_labels)

    if eps_range is None:
        eps_range = np.arange(0.3, 1.5, 0.05)
    if min_samples_range is None:
        min_samples_range = range(2, 10)

    results = []
    total = len(eps_range) * len(min_samples_range)
    i = 0

    for eps in eps_range:
        for min_s in min_samples_range:
            i += 1
            labels = DBSCAN(eps=eps, min_samples=min_s, metric="euclidean").fit_predict(embeddings)

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_outliers = (labels == -1).sum()
            outlier_ratio = n_outliers / len(labels)

            if n_clusters < 2 or outlier_ratio > max_outlier_ratio:
                continue

            sil = silhouette_score(embeddings, labels)

            row = {
                "eps": round(eps, 3),
                "min_samples": min_s,
                "n_clusters": n_clusters,
                "n_outliers": n_outliers,
                "outlier_ratio": round(outlier_ratio, 3),
                "silhouette": round(sil, 4),
            }

            if true_labels is not None:
                mask = labels != -1
                if mask.sum() > 0:
                    row["ARI"] = round(adjusted_rand_score(true_labels[mask], labels[mask]), 4)
                    row["NMI"] = round(normalized_mutual_info_score(true_labels[mask], labels[mask]), 4)

            results.append(row)
            if i % 20 == 0:
                print(f"  {i}/{total} configs testées…")

    df = pd.DataFrame(results).sort_values("silhouette", ascending=False)
    return df


def plot_heatmap(results_df, metric="silhouette"):
    pivot = results_df.pivot_table(
        index="min_samples", columns="eps", values=metric, aggfunc="max"
    )
    fig = px.imshow(
        pivot,
        color_continuous_scale="Viridis",
        title=f"Grid Search DBSCAN — {metric}",
        labels={"color": metric},
        aspect="auto",
    )
    fig.update_layout(height=400)
    fig.show()


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────────────────────────────────────

def main():
    random.seed(42)
    np.random.seed(42)
    os.makedirs("outputs", exist_ok=True)

    print("=" * 60)
    print("   Clustering sémantique de documents avec BERT")
    print("   Dataset : DBPedia")
    print("=" * 60)

    # 1. Chargement des données
    print("\n[1] Chargement des données...")
    docs, true_labels, label_names = load_dbpedia_data()
    print(f"Nombre de documents chargés : {len(docs)}")
    print(f"Nombre de classes : {len(set(true_labels))}")

    # 2. Nettoyage
    print("\n[2] Nettoyage des documents...")
    docs = clean_documents(docs)
    print(f"Nombre de documents après nettoyage : {len(docs)}")

    # 3. Génération des embeddings
    print("\n[3] Génération des embeddings BERT...")
    embeddings = generate_embeddings(docs)
    print(f"Taille des embeddings : {embeddings.shape}")

    # 4. Réduction de dimension pour clustering
    print("\n[4] Réduction de dimension pour clustering...")
    reduced_embeddings = reduce_for_clustering(embeddings)

    # 5. Elbow Method (KMeans)
    print("\n[5] Elbow Method...")
    # elbow_method(reduced_embeddings, k_range=range(2, 21))
    """
    k=7 ==> Silhouette Score : 0.5312 | ARI : 0.5125 | NMI : 0.7081
    k=8 ==> Silhouette Score : 0.5246 | ARI : 0.5449 | NMI : 0.7263
    """
    #chosen_k = 8
    #kmeans_labels = kmeans_clustering(reduced_embeddings, n_clusters=chosen_k)
    """
    # 6. Tuning DBSCAN
    print("\n[6] Tuning DBSCAN...")

    # k-distance plot pour trouver la zone d'eps (k = min_samples)
    plot_kdistance(reduced_embeddings, k=6)

    # Grid search sur (eps, min_samples)
    results = grid_search_dbscan(
        reduced_embeddings,
        true_labels=true_labels,
        eps_range=np.arange(0.1, 0.8, 0.02), 
        min_samples_range=range(2, 10),
        max_outlier_ratio=0.2,
    )
    

    print("\n=== Top 10 configurations ===")
    print(results.head(10).to_string(index=False))
    print(f"\nConfig choisie → eps={chosen_eps}, min_samples={chosen_min_samples}")


    if len(results) > 0:
        plot_heatmap(results, metric="silhouette")
        best = results.iloc[0]
        chosen_eps = best["eps"]
        chosen_min_samples = int(best["min_samples"])
        print(f"\n=== Meilleure config ===")
        print(f"  eps={chosen_eps}, min_samples={chosen_min_samples}")
    else:
        # Fallback sur tes valeurs manuelles si le grid search ne trouve rien
        chosen_eps = 0.7
        chosen_min_samples = 6
"""
    # 7. Clustering DBSCAN
    print("\n[7] Clustering DBSCAN...")
    labels = dbscan_clustering(reduced_embeddings,0.24,6)

    # 8. Évaluation
    print("\n[8] Évaluation...")
    evaluate_clustering(reduced_embeddings, labels, true_labels, model_name="DBSCAN")

    # 9. Réduction pour visualisation
    print("\n[9] Réduction en 2D pour visualisation...")
    reduced_2d = reduce_for_visualization(embeddings)

    # 10. Visualisation
    print("\n[10] Affichage des clusters...")
    # plot_clusters(reduced_2d, true_labels, "Vraies classes - DBPedia")
    # plot_clusters(reduced_2d, kmeans_labels, "Clusters prédits - K-Means")
    plot_clusters(reduced_2d, labels, "Clusters prédits - DBSCAN")

    print("\nProjet terminé.")


if __name__ == "__main__":
    main()