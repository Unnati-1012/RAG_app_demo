import faiss
import numpy as np
from utils.embedding import get_embedding   # your embedding function
def get_chunks(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start += chunk_size - overlap
    return chunks

# Step 1: Load your documents (replace with your own source, e.g. txt/pdf loader)
docs = [
    "This is document 1 about AI.",
    "Here is document 2 about machine learning.",
    "Document 3 talks about retrieval augmented generation."
]

# Step 2: Convert docs into smaller chunks
chunks = []
for doc in docs:
    chunks.extend(get_chunks(doc))  # split each doc into manageable pieces

print(f"🔹 Total chunks: {len(chunks)}")

# Step 3: Generate embeddings for all chunks
embeddings = [get_embedding(chunk) for chunk in chunks]
embeddings = np.array(embeddings).astype("float32")  # FAISS expects float32
print(f"🔹 Embeddings shape: {embeddings.shape}")

# Step 4: Build FAISS index
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

# Step 5: Save index
faiss.write_index(index, "faiss_index.bin")
print("✅ FAISS index built and saved as faiss_index.bin")
