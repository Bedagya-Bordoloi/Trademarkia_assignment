import os
import numpy as np
import faiss
import joblib
from sentence_transformers import SentenceTransformer

class SearchEngine:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = faiss.read_index("models/faiss.index")
        self.gmm = joblib.load("models/gmm.pkl")
        
        with open("data/cleaned_docs.txt", encoding="utf-8") as f:
            self.documents = [line.strip() for line in f]

    def search(self, query: str, top_k: int = 5):
        # 1. Embed and normalize
        emb = self.model.encode([query], normalize_embeddings=True)[0]
        emb_2d = emb.reshape(1, -1).astype(np.float32)

        # 2. Vector Search
        distances, indices = self.index.search(emb_2d, top_k)
        hits = [self.documents[i] for i in indices[0] if i < len(self.documents)]

        # 3. Fuzzy Clustering
        probs = self.gmm.predict_proba(emb_2d)[0]
        cluster_id = int(np.argmax(probs)) 

        return emb, hits, cluster_id, probs