import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import entropy

class SemanticCache:
    def __init__(self, threshold=0.84, n_clusters=12):
        self.threshold = threshold
        self.n_clusters = n_clusters
        self.buckets = {i: [] for i in range(n_clusters)}
        self.hits = 0
        self.misses = 0

    def add(self, query: str, embedding: np.ndarray, result: str, cluster_id: int):
        # Stores the result of a Miss to turn it into a potential Hit later
        self.buckets[cluster_id].append({
            "query": query,
            "embedding": embedding,
            "result": result
        })

    def lookup(self, embedding: np.ndarray, cluster_id: int):
        bucket = self.buckets.get(cluster_id, [])
        
        # Scenario A: The cluster bucket is empty
        if not bucket:
            self.misses += 1
            print(f"DEBUG: Cache Miss (Empty Bucket). Total Misses: {self.misses}")
            return None, None

        # Scenario B: Search within the bucket
        stored_embs = np.stack([e["embedding"] for e in bucket])
        sims = cosine_similarity(embedding.reshape(1, -1), stored_embs)[0]
        
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])

        if best_sim >= self.threshold:
            self.hits += 1
            print(f"DEBUG: Cache Hit! Score: {best_sim:.4f}. Total Hits: {self.hits}")
            return bucket[best_idx], best_sim

        # Scenario C: Found items in cluster, but none are "close enough"
        self.misses += 1
        print(f"DEBUG: Cache Miss (Low Similarity). Total Misses: {self.misses}")
        return None, None

    def get_stats(self):
        total = self.hits + self.misses
        return {
            "total_entries": sum(len(v) for v in self.buckets.values()),
            "hit_count": self.hits,
            "miss_count": self.misses,
            "hit_rate": round(self.hits / total if total > 0 else 0.0, 4)
        }

    def clear(self):
        self.buckets = {i: [] for i in range(self.n_clusters)}
        self.hits = 0
        self.misses = 0
        print("DEBUG: Cache cleared and stats reset.")

    @staticmethod
    def get_ambiguity(probs: np.ndarray):
        ent = float(entropy(probs))
        if ent > 1.0: level = "High (Boundary Case)"
        elif ent > 0.5: level = "Medium"
        else: level = "Low (Strong Cluster Membership)"
        return ent, level