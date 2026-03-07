# Clustering Analysis
- **Fuzzy Logic**: Used Gaussian Mixture Models (GMM) to provide probability distributions.
- **Ambiguity**: Shannon Entropy is calculated for every query. High entropy (e.g., > 1.0) indicates the document sits on a semantic boundary (e.g., between 'Science' and 'Law').
- **Efficiency**: The cache uses these cluster assignments to bucket entries, reducing lookup time from $O(N)$ to $O(N/K)$.