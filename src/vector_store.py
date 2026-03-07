import os
import faiss
import numpy as np

MODELS_DIR = "models"
EMBEDDINGS_PATH = os.path.join(MODELS_DIR, "embeddings.npy")
INDEX_PATH = os.path.join(MODELS_DIR, "faiss.index")

def build_index():
    if not os.path.exists(EMBEDDINGS_PATH):
        raise FileNotFoundError("Run embeddings.py first.")

    embeddings = np.load(EMBEDDINGS_PATH).astype(np.float32)
    
    # FAISS FlatIP (Inner Product) is equivalent to Cosine Similarity.
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)
    print(f"→ FAISS index created with {index.ntotal} vectors.")

if __name__ == "__main__":
    build_index()