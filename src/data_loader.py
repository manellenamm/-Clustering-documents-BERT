from datasets import load_dataset
import random


def load_ag_news(n_samples=2000):
    dataset = load_dataset("ag_news")

    texts = dataset["train"]["text"][:n_samples]
    labels = dataset["train"]["label"][:n_samples]

    return texts, labels


def load_dbpedia_data(n_samples=2000):
    dataset = load_dataset("dbpedia_14")

    docs = dataset["train"]["content"]
    labels = dataset["train"]["label"]

    indices = random.sample(range(len(docs)), n_samples)

    sampled_docs = [docs[i] for i in indices]
    sampled_labels = [labels[i] for i in indices]

    label_names = [
        "Company", "EducationalInstitution", "Artist", "Athlete",
        "OfficeHolder", "MeanOfTransportation", "Building", "NaturalPlace",
        "Village", "Animal", "Plant", "Album", "Film", "WrittenWork"
    ]

    return sampled_docs, sampled_labels, label_names