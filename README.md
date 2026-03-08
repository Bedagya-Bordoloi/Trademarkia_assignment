
# Semantic Search System: Fuzzy Clustering & Intelligent Cache  
**Built for Trademarkia AI & ML Task**

This project implements a lightweight, high-performance semantic search system using the 20 Newsgroups dataset. It features a custom-built semantic cache, a probabilistic clustering layer, and a production-ready FastAPI interface.

## 🚀 Quick Start

1. **Installation**  
   Ensure you have Python 3.9+ installed.

   ```bash
   # Clone the repository and enter the directory
   git clone <https://github.com/Bedagya-Bordoloi/Trademarkia_assignment>
   cd Trademarkia_assignment

   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Data & Model Pipeline**  
   The system follows a sequential pipeline to process the noisy corpus and build the vector search indices.

   ```bash
   python src/preprocessing.py     # Cleans ~20,000 documents
   python src/embeddings.py        # Generates 384d vectors
   python src/vector_store.py      # Builds FAISS FlatIP index
   python src/clustering.py        # Trains Fuzzy GMM model
   ```

3. **Start the API**

   ```bash
   uvicorn src.app:app --host 127.0.0.1 --port 8000
   ```

   Access the interactive documentation at:  
   http://127.0.0.1:8000/docs

## 🧠 Design Decisions & Justifications

### Part 1: Embedding & Vector Database

- **Model Choice**: `all-MiniLM-L6-v2`  
  384 dimensions, excellent semantic quality for short-to-medium text, very fast inference — ideal for keeping the system lightweight and responsive.

- **Vector DB**: FAISS with `IndexFlatIP`  
  Embeddings are L2-normalized during preprocessing → Inner Product is mathematically equivalent to Cosine Similarity but significantly faster for exact search on this corpus size.

- **Data Pruning**:  
  Removed email headers, footers, quotes, excessive whitespace, and very short documents (<120 chars) via regex + length filter. These steps reduce lexical noise and prevent low-information vectors from polluting clusters.

### Part 2: Fuzzy Clustering (GMM)

- **Gaussian Mixture Models (GMM)**  
  Provides soft, probabilistic assignments — perfect for overlapping topics (e.g. a post about "gun legislation" belongs partially to politics and firearms).

- **Cluster Count (K=12)**  
  While the dataset has 20 ground-truth labels, real semantic structure is messier with heavy overlap. BIC (Bayesian Information Criterion) curve showed a sweet spot at 12 clusters — best balance between model fit and overfitting.

- **Boundary Detection via Shannon Entropy**  
  Queries with high cluster entropy are flagged as ambiguous ("boundary cases") — exactly what the task asked for when evaluating genuine model uncertainty.

### Part 3: The Semantic Cache (Built from First Principles)

- **Cluster-Bucketed Lookup**  
  Instead of linear scan over entire cache (O(N)), queries are routed to the dominant cluster bucket → lookup complexity drops to O(N/K) where K=12.

- **Tunable Similarity Threshold (τ=0.84)**  
  - High τ → high precision, fewer false positives, but lower hit rate  
  - Low τ → high recall, more hits, but risk of semantic drift  
  0.84 was chosen after manual tuning as a strong balance for 20 Newsgroups vocabulary and downstream relevance.

- No external caching middleware (no Redis, Memcached, etc.) — pure in-memory Python implementation.

## 🛠 Project Structure

```
├── src/
│   ├── app.py                  # FastAPI service + state management
│   ├── semantic_cache.py       # Custom cluster-aware cache (core innovation)
│   ├── clustering.py           # GMM training & entropy calculation
│   ├── preprocessing.py        # Noise removal & cleaning
│   ├── embeddings.py           # SentenceTransformer encoding
│   ├── vector_store.py         # FAISS index creation
│   └── search_engine.py        # Query embedding + retrieval logic
├── analysis/
│   └── clustering_report.md    
├── tests/
│   └── test_cache.py           # Unit tests for cache hit/miss/semantic behavior
├── models/                     # Persistent artifacts (embeddings.npy, faiss.index, gmm.pkl)
├── data/                       # Raw & cleaned dataset files
├── requirements.txt
└── run.sh                      # Convenience script for pipeline + server start
```

## 📊 API Endpoints

- **POST /query**  
  ```json
  { "query": "natural language text here" }
  ```
  Returns:
  - cache_hit (bool)
  - matched_query & similarity_score (on hit)
  - result (retrieved snippets)
  - dominant_cluster
  - query_entropy + ambiguity_level

- **GET /cache/stats**  
  Real-time metrics: total_entries, hit_count, miss_count, hit_rate

- **DELETE /cache**  
  Flush in-memory cache and reset counters

## 🧪 Quality Assurance

Run unit tests to verify cache correctness:

```bash
pytest tests/
# or directly:
python tests/test_cache.py
```

Expected output:  
✅ Cache Unit Tests Passed!

## Contact & Submission

- **GitHub**: https://github.com/Bedagya-Bordoloi
- **Email**: bordoloibedagya@gmail.com
- **Recruitment contact**: recruitments@trademarkia.com 

Feel free to reach out with any questions.  
Thank you for reviewing my submission!




