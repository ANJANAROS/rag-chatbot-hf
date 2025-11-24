import os
import numpy as np
from models.embeddings import get_embedding_model


def load_documents(folder_path="docs"):
    """Load .txt files from docs folder"""
    docs = []
    if not os.path.exists(folder_path):
        return docs

    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            with open(os.path.join(folder_path, file), "r", encoding="utf-8") as f:
                docs.append({
                    "text": f.read(),
                    "source": file
                })
    return docs


def create_vector_store(docs):
    """Create vector embeddings for each document"""
    embedder = get_embedding_model()
    vector_store = []

    for doc in docs:
        emb = embedder.embed_query(doc["text"])  # embed the whole document
        vector_store.append({
            "embedding": np.array(emb, dtype=float),
            "text": doc["text"],
            "source": doc["source"]
        })

    return vector_store


def search(query, vector_store, top_k=3):
    """Search top_k similar docs using cosine similarity"""
    embedder = get_embedding_model()
    q_emb = np.array(embedder.embed_query(query), dtype=float)

    similarities = []
    for item in vector_store:
        v = item["embedding"]
        if np.linalg.norm(q_emb) == 0 or np.linalg.norm(v) == 0:
            sim = 0.0
        else:
            sim = float(np.dot(q_emb, v) / (np.linalg.norm(q_emb) * np.linalg.norm(v)))

        similarities.append((sim, item))

    similarities.sort(key=lambda x: x[0], reverse=True)
    return [item[1] for item in similarities[:top_k]]
