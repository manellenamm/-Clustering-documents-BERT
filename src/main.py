from datasets import load_dataset
from embedding import generate_embeddings
#from clustering import kmeans_clustering
#from visualization import reduce_dimensions, plot_clusters
#from evaluation import evaluate_clustering
from bertopic_model import *
#from sklearn.cluster import KMeans
#import matplotlib.pyplot as plt
import os

def main():
    os.makedirs("data", exist_ok=True)

    # ======================
    # 1. Charger les données
    # ======================
    print("Chargement des données...")
    dataset = load_dataset("ag_news")
    docs = dataset["train"]["text"][:300]

    print("Labels présents dans AG News :", set(dataset["train"]["label"]))
    print("Nombre de documents utilisés :", len(docs))

    # ======================
    # 2. Génération des embeddings
    # ======================
    print("Génération des embeddings...")
    embeddings = generate_embeddings(docs)
    print("Taille des embeddings :", embeddings.shape)

    # ======================
    # 6. BERTopic
    # ======================
    print("\nLancement de BERTopic...")
    topic_model, topics, probs = run_bertopic(
        docs
    )

    print_main_topics(topic_model)
    save_topic_info(topic_model, output_path="data/bertopic_topics.csv")



    # Visualisation optionnelle BERTopic
    # Ouvre une visualisation interactive dans le navigateur
    topic_model.visualize_barchart().show()

    print("\nFichier des topics sauvegardé dans : data/bertopic_topics.csv")

if __name__ == "__main__":
    main()