from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

def generate_embeddings(docs):
    model = SentenceTransformer("all-mpnet-base-v2")
    embeddings = model.encode(docs, show_progress_bar=True)
    embeddings = normalize(embeddings)
    return embeddings