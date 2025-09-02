
import faiss
import pickle

def load_faiss_index(index_path="data/my_index.index", metadata_path="data/chunk_mapping.pkl"):
    try:
        index = faiss.read_index(index_path)
        with open(metadata_path, "rb") as f:
            chunk_mapping = pickle.load(f)
        return index, chunk_mapping
    except Exception as e:
        raise RuntimeError(f"❌ Error loading FAISS index: {e}")


def retrieve_chunks(index, query_embedding, top_k=5):
    """
    Retrieve top-k closest chunks using FAISS.
    """
    distances, indices = index.search(query_embedding.reshape(1, -1), top_k)
    return indices[0], distances[0]
