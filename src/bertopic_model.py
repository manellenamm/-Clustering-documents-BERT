from bertopic import BERTopic
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

def run_bertopic(docs):
    embedding_model = SentenceTransformer("all-mpnet-base-v2")

    hdbscan_model = HDBSCAN(
        min_cluster_size=8,
        metric="euclidean",
        prediction_data=True
    )

    vectorizer_model = CountVectorizer(stop_words="english")

    topic_model = BERTopic(
        embedding_model=embedding_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        calculate_probabilities=True,
        verbose=True
    )

    topics, probs = topic_model.fit_transform(docs)
    return topic_model, topics, probs

def get_topic_info_dataframe(topic_model):
    return topic_model.get_topic_info()

def print_main_topics(topic_model, top_n=10):
    topic_info = topic_model.get_topic_info()
    print("\n=== Topics principaux ===")
    print(topic_info.head(top_n))

def save_topic_info(topic_model, output_path="data/bertopic_topics.csv"):
    topic_info = topic_model.get_topic_info()
    topic_info.to_csv(output_path, index=False)

def get_topic_words(topic_model, topic_id):
    return topic_model.get_topic(topic_id)