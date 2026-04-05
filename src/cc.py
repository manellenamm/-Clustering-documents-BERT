"""
Text Clustering Pipeline
Embeddings (SentenceTransformers) → UMAP → DBSCAN → Plotly
"""
 
import numpy as np
import pandas as pd
import plotly.express as px
from sentence_transformers import SentenceTransformer
from umap import UMAP
from sklearn.cluster import DBSCAN
 
# ─── 1. Tes documents ────────────────────────────────────────────────────────
 
texts = [
    "Deep learning models require large amounts of training data.",
    "Neural networks can learn complex patterns from raw data.",
    "Transformers have revolutionized natural language processing.",
    "The stock market closed higher after the Fed's announcement.",
    "Inflation is affecting consumer spending across the country.",
    "Interest rates are expected to drop next quarter.",
    "Climate change is causing more frequent extreme weather events.",
    "Renewable energy adoption is accelerating globally.",
    "Carbon emissions need to be reduced to limit global warming.",
    "Python is the most popular language for data science.",
    "Pandas and NumPy are essential libraries for data analysis.",
    "Machine learning pipelines automate model training and deployment.",
]
 
# ─── 2. Embeddings ───────────────────────────────────────────────────────────
 
print("Loading embedding model...")
model = SentenceTransformer("all-mpnet-base-v2")  # meilleur modèle anglais
embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
 
# ─── 3. UMAP ─────────────────────────────────────────────────────────────────
 
print("Running UMAP...")
umap = UMAP(
    n_components=2,       # 2D pour la visu
    n_neighbors=5,        # ↑ = structure plus globale (ajuste selon taille corpus)
    min_dist=0.1,         # ↑ = clusters plus espacés visuellement
    metric="cosine",      # cosine est idéal pour des embeddings normalisés
    random_state=42,
)
embeddings_2d = umap.fit_transform(embeddings)
 
# ─── 4. DBSCAN ───────────────────────────────────────────────────────────────
 
print("Running DBSCAN...")
dbscan = DBSCAN(
    eps=0.5,              # rayon de voisinage — PARAMÈTRE CLÉ à ajuster
    min_samples=2,        # nb minimum de points pour former un cluster
    metric="euclidean",   # sur l'espace UMAP réduit
)
labels = dbscan.fit_predict(embeddings_2d)
 
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = (labels == -1).sum()
print(f"\n→ {n_clusters} clusters found, {n_noise} outliers")
 
# ─── 5. Visualisation Plotly ──────────────────────────────────────────────────
 
df = pd.DataFrame({
    "x": embeddings_2d[:, 0],
    "y": embeddings_2d[:, 1],
    "cluster": labels.astype(str),
    "text": texts,
})
df["cluster"] = df["cluster"].replace("-1", "Outlier")
 
fig = px.scatter(
    df,
    x="x", y="y",
    color="cluster",
    hover_data={"x": False, "y": False, "text": True},
    title=f"Document Clustering — DBSCAN ({n_clusters} clusters)",
    labels={"cluster": "Cluster"},
    color_discrete_sequence=px.colors.qualitative.Bold,
    opacity=0.85,
    width=900, height=600,
)
fig.update_traces(marker=dict(size=10))
fig.show()