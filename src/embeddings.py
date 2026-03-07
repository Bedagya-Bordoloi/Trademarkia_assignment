import os
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
MODELS_DIR = "models"
INPUT_FILE = "data/cleaned_docs.txt"
EMBEDDINGS_PATH = os.path.join(MODELS_DIR, "embeddings.npy")

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("Reading cleaned documents...")
    with open(INPUT_FILE, encoding="utf-8") as f:
        documents = [line.strip() for line in f if line.strip()]

    print(f"Encoding {len(documents):,} documents with {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        documents,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    np.save(EMBEDDINGS_PATH, embeddings)
    print(f"→ Saved embeddings shape {embeddings.shape} → {EMBEDDINGS_PATH}")

if __name__ == "__main__":
    main()