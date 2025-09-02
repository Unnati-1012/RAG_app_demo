# app.py
import os
import sys
import streamlit as st
# from dotenv import load_dotenv

# -------------------------------
# Add project root to path
# -------------------------------
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

# -------------------------------
# Local imports
# -------------------------------
from utils.embedding import get_embedding
from utils.retrieval import load_faiss_index, retrieve_chunks
from utils.prompt import build_prompt
from utils.completion import generate_completion

# -------------------------------
# Load environment variables
# -------------------------------
# load_dotenv()
os.getenv("GEMINI_API_KEY")
if not API_KEY:
    st.error("❌ GEMINI_API_KEY not found in .env file.")
    st.stop()

# -------------------------------
# FAISS index paths
# -------------------------------
index_path = os.path.join(os.path.dirname(__file__), "data", "my_index.index")
metadata_path = os.path.join(os.path.dirname(__file__), "data", "chunk_mapping.pkl")

# -------------------------------
# Load FAISS index
# -------------------------------
try:
    index, chunk_mapping = load_faiss_index(
        index_path=index_path,
        metadata_path=metadata_path
    )
except Exception as e:
    st.error(f"Failed to load FAISS index or chunk mapping: {e}")
    st.stop()  # Stop execution if index is not loaded

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("RAG App: Euron Founder Story")
st.write("Ask questions grounded in the life and mission of Sudhanshu Kumar.")

# Debug info
st.sidebar.write(f"🔍 FAISS index dimension: {index.d}")

# Initialize variables to avoid NameError
response = None
top_chunks = []

# -------------------------------
# Query input
# -------------------------------
query = st.text_input("Enter your question here")

if query:
    try:
        # 1️⃣ Get query embedding
        query_embedding = get_embedding(query)

        # Debug info
        st.sidebar.write(f"🧮 Query embedding shape: {len(query_embedding)}")
        st.sidebar.write(f"⚖️ Query embedding dimension: {len(query_embedding)}")

        # Check dimension
        if len(query_embedding) != index.d:
            st.error(
                f"❌ Dimension mismatch!\n"
                f"Query embedding = {len(query_embedding)}, "
                f"FAISS index = {index.d}"
            )
        else:
            # 2️⃣ Retrieve relevant chunks
            indices, distances = retrieve_chunks(index, query_embedding, top_k=5)
            top_chunks = [chunk_mapping[i] for i in indices]

            # 3️⃣ Build prompt
            prompt = build_prompt(top_chunks, query)

            # 4️⃣ Generate completion
            try:
                response = generate_completion(prompt=prompt)
            except Exception as e:
                st.error(f"⚠️ Gemini API Error: {e}")

    except Exception as e:
        st.error(f"⚠️ Error processing query: {e}")

# -------------------------------
# Display answer
# -------------------------------
if response:
    st.subheader("Answer")
    st.write(response)

# -------------------------------
# Display retrieved chunks
# -------------------------------
if top_chunks:
    with st.expander("Retrieved Chunks"):
        for chunk in top_chunks:
            st.markdown(f"- {chunk}")
