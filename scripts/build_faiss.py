# scripts/build_faiss.py

import os
import pandas as pd
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Setup paths
RAW_DATA_PATH = '../data/raw'
FAISS_STORE_PATH = '../data/faiss_store'
os.makedirs(FAISS_STORE_PATH, exist_ok=True)

# Embedding model
EMBEDDING_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
BATCH_SIZE = 128  # control memory
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# Load all raw CSVs
def load_all_data(folder_path):
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    dfs = []
    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        dfs.append(df)
    return dfs

# Extract meaningful text from each dataset
def extract_knowledge(dfs):
    knowledge_texts = []
    metadata_records = []

    for df in dfs:
        # Check known field names
        for idx, row in df.iterrows():
            parts = []
            for col in ['incident_name', 'description', 'FIRE_NAME', 'incident_location', 'Field', 'Value', 'CAUSE']:
                if col in row and pd.notna(row[col]):
                    parts.append(str(row[col]))
            if parts:
                text = " | ".join(parts)
                knowledge_texts.append(text)
                metadata_records.append({
                    "source": "unknown",
                    "index": idx,
                    "text": text
                })
    return knowledge_texts, metadata_records

# Embed in batches
def embed_texts(texts, model, batch_size=128):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding texts"):
        batch = texts[i:i+batch_size]
        batch_embeddings = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        embeddings.append(batch_embeddings)
    return np.vstack(embeddings)

# Save mapping (for later retrieval)
def save_metadata(metadata_records, path):
    df_meta = pd.DataFrame(metadata_records)
    df_meta.to_csv(path, index=False)
    print(f"Saved metadata mapping to {path}")

# Main
if __name__ == "__main__":
    import numpy as np

    dfs = load_all_data(RAW_DATA_PATH)
    knowledge_texts, metadata_records = extract_knowledge(dfs)

    print(f"Total extracted texts for embedding: {len(knowledge_texts)}")

    embeddings = embed_texts(knowledge_texts, embedding_model)

    # Build FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    faiss.write_index(index, os.path.join(FAISS_STORE_PATH, 'faiss_index.idx'))

    # Save metadata
    save_metadata(metadata_records, os.path.join(FAISS_STORE_PATH, 'metadata.csv'))

    print("✅ FAISS index and metadata successfully built and saved!")
