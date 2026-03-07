from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from src.semantic_cache import SemanticCache
from src.search_engine import SearchEngine
import numpy as np

app = FastAPI(title="Trademarkia Semantic Search System")

# Initialize models
cache = SemanticCache(threshold=0.84)
engine = SearchEngine()

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    query: str
    cache_hit: bool
    matched_query: Optional[str] = None
    similarity_score: Optional[float] = None
    result: str
    dominant_cluster: int
    query_entropy: float
    ambiguity_level: str

@app.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest):
    q = req.query.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        # 1. Embed and Search
        embedding, results, cluster_id, probs = engine.search(q)

        # 2. Check Semantic Cache
        cached_entry, score = cache.lookup(embedding, cluster_id)

        # 3. Handle Entropy Analysis
        ent, amb = cache.get_ambiguity(probs)

        if cached_entry:
            return {
                "query": q,
                "cache_hit": True,
                "matched_query": str(cached_entry["query"]),
                "similarity_score": round(float(score), 4),
                "result": str(cached_entry["result"]),
                "dominant_cluster": int(cluster_id),
                "query_entropy": float(ent),
                "ambiguity_level": str(amb)
            }

        # 4. Cache Miss Path
        result_text = "\n───\n".join(results[:3]) if results else "No matches found."
        
        # Store in cache bucket for future lookups
        cache.add(q, embedding, result_text, int(cluster_id))

        return {
            "query": q,
            "cache_hit": False,
            "matched_query": None,
            "similarity_score": None,
            "result": result_text,
            "dominant_cluster": int(cluster_id),
            "query_entropy": float(ent),
            "ambiguity_level": str(amb)
        }

    except Exception as e:
        print(f"Server Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inference Error: {str(e)}")

@app.get("/cache/stats")
def cache_stats():
    return cache.get_stats()

@app.delete("/cache")
def flush_cache():
    cache.clear()
    return {"message": "Cache completely flushed and stats reset"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)