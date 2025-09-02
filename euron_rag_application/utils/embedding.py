import google.generativeai as genai
import numpy as np
import os
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)
from sentence_transformers import SentenceTransformer

# Load same model used in build_index.py
model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embedding(text: str):
    return model.encode(text, convert_to_numpy=True, normalize_embeddings=True)

API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)

def get_embedding(text, model="models/embedding-001"):
    """
    Generate embeddings using Google Gemini API.
    Returns a NumPy array.
    """
    try:
        response = genai.embed_content(
            model=model,
            content=text
        )

        print("🔎 Embedding API response:", response)  # Debug print

        if "embedding" in response:
            return np.array(response["embedding"])
        else:
            raise KeyError(f"❌ No embedding found in response: {response}")

    except Exception as e:
        raise RuntimeError(f"❌ Error while generating embedding: {e}")

if __name__ == "__main__":
    # Example test run
    text = "Hello, Gemini embeddings!"
    vec = get_embedding(text)
    print("Embedding vector shape:", vec.shape)
    print("First 10 values:", vec[:10])
