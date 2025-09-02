import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# -------------------------------
# Settings
# -------------------------------
DOCS_FOLDER = "data"   
INDEX_PATH = "data/my_index.index"
MAPPING_PATH = "data/chunk_mapping.pkl"

# 🔄 Choose model carefully:
# "all-MiniLM-L6-v2" → 384-dim (fast, small)
# "all-mpnet-base-v2" → 768-dim (more accurate, matches your app)
EMBED_MODEL = "all-mpnet-base-v2"

# -------------------------------
# Load embedding model
# -------------------------------
model = SentenceTransformer(EMBED_MODEL)

# -------------------------------
# Helper: Read documents
# -------------------------------
def load_documents(folder):
    texts = []
    if not os.path.exists(folder):
        raise FileNotFoundError(f"❌ Folder not found: {folder}")

    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        if file.endswith(".txt"):   # extendable later for pdf/docx
            with open(path, "r", encoding="utf-8") as f:
                texts.append(f.read())
    return texts

# -------------------------------
# Helper: Chunk text
# -------------------------------
def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        if chunk:  # skip empty
            chunks.append(chunk)
    return chunks

# -------------------------------
# Build FAISS index
# -------------------------------
def build_faiss_index(chunks):
    embeddings = model.encode(
        chunks, 
        convert_to_numpy=True, 
        normalize_embeddings=True
    )
    dim = embeddings.shape[1]

    print(f"✅ Embedding dimension: {dim}")

    index = faiss.IndexFlatIP(dim)   # cosine similarity
    index.add(embeddings)

    return index

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    print("📖 Loading documents...")
    docs = load_documents(DOCS_FOLDER)

    print("✂️ Splitting into chunks...")
    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_text(doc))

    print(f"📊 Total chunks: {len(all_chunks)}")

    print("🔎 Creating FAISS index...")
    index = build_faiss_index(all_chunks)

    print(f"💾 Saving FAISS index → {INDEX_PATH}")
    faiss.write_index(index, INDEX_PATH)

    print(f"💾 Saving chunk mapping → {MAPPING_PATH}")
    with open(MAPPING_PATH, "wb") as f:
        pickle.dump(all_chunks, f)

    print("✅ Index and mapping saved successfully!")
