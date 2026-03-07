import os
import re
from sklearn.datasets import fetch_20newsgroups

DATA_DIR = "data"
OUTPUT_FILE = os.path.join(DATA_DIR, "cleaned_docs.txt")

def clean_text(text: str) -> str:
    # Remove email patterns and technical headers 
    # that 'remove=(headers)' sometimes misses.
    text = re.sub(r'\S*@\S*\s?', '', text) 
    text = re.sub(r'(Subject:|Lines:|Organization:|Reply-To:).*\n', '', text)
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)   
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def prepare_dataset():
    print("Loading 20 Newsgroups...")
    dataset = fetch_20newsgroups(
        subset='all',
        remove=('headers', 'footers', 'quotes'),
        random_state=42
    )

    cleaned = []
    for doc in dataset.data:
        cleaned_doc = clean_text(doc)
        # Deliberate choice: length > 120. This filters out very short documents that may not provide meaningful context for embedding and search, while still retaining a large portion of the dataset for training and evaluation.
        if len(cleaned_doc) > 120: 
            cleaned.append(cleaned_doc)

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for line in cleaned:
            f.write(line + "\n")

    print(f"Saved {len(cleaned):,} documents.")

if __name__ == "__main__":
    prepare_dataset()