import os
import numpy as np
import random
from data_loader import load_dbpedia_data
from preprocessing import clean_documents
from embedding import generate_embeddings
from clustering import kmeans_clustering, elbow_method,find_eps,dbscan_clustering
from evaluation import evaluate_clustering
from visualization import reduce_for_clustering, reduce_for_visualization, plot_clusters


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

    #choix de l
    print("\n[5] Elbow Method...")
    #elbow_method(reduced_embeddings, k_range=range(2, 21))
    """
     k=7 ==>Silhouette Score : 0.5312
    ARI : 0.5125
    NMI : 0.7081

    k=8==> Silhouette Score : 0.5246
    ARI : 0.5449
    NMI : 0.7263


     """
    chosen_k=8
    # 5. Clustering K-Means
    print("\n[5] Clustering K-Means...")
    kmeans_labels = kmeans_clustering(reduced_embeddings, n_clusters=chosen_k)


    
     # 5. Clustering DBSCAN
    print("\n[5] Clustering DBSCAN...")
    #find_eps(reduced_embeddings)
    chosen_eps = 0.7
    labels = dbscan_clustering(reduced_embeddings, chosen_eps,6)

    # 6. Évaluation
    print("\n[6] Évaluation...")
    evaluate_clustering(reduced_embeddings,labels,true_labels, model_name="")

    # 7. Réduction pour visualisation
    print("\n[7] Réduction en 2D pour visualisation...")
    reduced_2d = reduce_for_visualization(embeddings)


    # 8. Visualisation
    print("\n[8] Affichage des clusters...")
    #plot_clusters(reduced_2d, true_labels, "Vraies classes - DBPedia")
    #plot_clusters(reduced_2d, kmeans_labels, "Clusters prédits - K-Means")
    plot_clusters(reduced_2d, labels,"Clusters prédits - DBSCAN")
    
    
    print("\nProjet terminé.")



if __name__ == "__main__":
    main()


