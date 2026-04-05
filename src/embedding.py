from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")

def generate_embeddings(texts):
    """
    Génère des embeddings normalisés (N, 384)
    """
    embeddings = model.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        normalize_embeddings=True  
    )
    
    return embeddings