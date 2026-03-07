import os
import numpy as np
from sklearn.mixture import GaussianMixture
import joblib

MODELS_DIR = "models"
EMBEDDINGS_PATH = os.path.join(MODELS_DIR, "embeddings.npy")
GMM_PATH = os.path.join(MODELS_DIR, "gmm.pkl")

# Choice: 12 clusters. 
# Justification: While there are 20 labels, themes like 'religion' and 'atheism' 
# or 'space' and 'sci.med' overlap significantly in vector space.
N_CLUSTERS = 12 

def train():
    if not os.path.exists(EMBEDDINGS_PATH):
        raise FileNotFoundError(f"Missing: {EMBEDDINGS_PATH}. Run embeddings.py first.")

    X = np.load(EMBEDDINGS_PATH).astype(np.float32)

    print(f"Training GMM on {len(X)} samples...")
    gmm = GaussianMixture(
        n_components=N_CLUSTERS,
        covariance_type='diag', 
        max_iter=100,
        random_state=42
    )
    gmm.fit(X)

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(gmm, GMM_PATH)
    print("→ GMM training complete.")

if __name__ == "__main__":
    train()