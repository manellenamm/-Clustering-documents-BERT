import os
import numpy as np
import random
from data_loader import load_dbpedia_data
from preprocessing import clean_documents
from embedding import generate_embeddings
from clustering import kmeans_clustering, elbow_method, dbscan_clustering
from evaluation import evaluate_clustering
from visualization import reduce_for_clustering, reduce_for_visualization, plot_clusters


def choose_model() -> str:
    print("\nQuel modèle de clustering voulez-vous utiliser ?")
    print("  [1] K-Means")
    print("  [2] DBSCAN")
    choice = input("Votre choix (1 ou 2) : ").strip()
    if choice == "1":
        return "kmeans"
    elif choice == "2":
        return "dbscan"
    else:
        print("Choix invalide, DBSCAN sélectionné par défaut.")
        return "dbscan"


def run_kmeans(reduced_embeddings) -> tuple:
    print("\nVoulez-vous lancer l'Elbow Method pour choisir k ?")
    print("  [1] Oui (affiche le graphe, puis vous saisissez k)")
    print("  [2] Non (utiliser k=8 par défaut)")
    choice = input("Votre choix (1 ou 2) : ").strip()

    if choice == "1":
        elbow_method(reduced_embeddings, k_range=range(2, 21))
        try:
            k = int(input("Entrez le k souhaité : ").strip())
        except ValueError:
            print("Entrée invalide, k=8 utilisé.")
            k = 8
    else:
        k = 8
        print(f"k={k} utilisé.")

    print(f"\n[Clustering] K-Means avec k={k}...")
    labels = kmeans_clustering(reduced_embeddings, n_clusters=k)
    return labels, f"K-Means (k={k})"


def run_dbscan(reduced_embeddings) -> tuple:
    eps = 0.24
    min_samples = 6
    print(f"\n[Clustering] DBSCAN avec eps={eps}, min_samples={min_samples}...")
    labels = dbscan_clustering(reduced_embeddings, eps, min_samples)
    return labels, "DBSCAN"


def main():
    random.seed(42)
    np.random.seed(42)
    os.makedirs("outputs", exist_ok=True)

    print("=" * 60)
    print("   Clustering sémantique de documents avec BERT")
    print("   Dataset : DBPedia")
    print("=" * 60)

    # 1. Chargement
    print("\n[1] Chargement des données...")
    docs, true_labels, label_names = load_dbpedia_data()
    print(f"Documents : {len(docs)} | Classes : {len(set(true_labels))}")

    # 2. Nettoyage
    print("\n[2] Nettoyage des documents...")
    docs = clean_documents(docs)
    print(f"Documents après nettoyage : {len(docs)}")

    # 3. Embeddings
    print("\n[3] Génération des embeddings BERT...")
    embeddings = generate_embeddings(docs)
    print(f"Shape embeddings : {embeddings.shape}")

    # 4. Réduction pour clustering
    print("\n[4] Réduction de dimension pour clustering...")
    reduced_embeddings = reduce_for_clustering(embeddings)

    # 5. Choix du modèle
    model_choice = choose_model()

    if model_choice == "kmeans":
        labels, model_name = run_kmeans(reduced_embeddings)
    else:
        labels, model_name = run_dbscan(reduced_embeddings)

    # 6. Évaluation
    print(f"\n[Évaluation] Modèle : {model_name}")
    evaluate_clustering(reduced_embeddings, labels, true_labels, model_name=model_name)

    # 7. Visualisation
    print("\n[Visualisation] Réduction 2D...")
    reduced_2d = reduce_for_visualization(embeddings)
    plot_clusters(reduced_2d, labels, f"Clusters prédits - {model_name}")

    print("\nProjet terminé.")


if __name__ == "__main__":
    main()