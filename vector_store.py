import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load the embedding model once (downloads ~90MB first time)
# all-MiniLM-L6-v2 is small, fast, and good enough for most RAG use cases
MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)


def embed_chunks(chunks: list[str]) -> np.ndarray:
    """
    Converts a list of text chunks into a 2D numpy array of vectors.
    Each chunk → one vector of 384 floats (for this model).

    Shape of output: (num_chunks, 384)
    """
    print(f"Embedding {len(chunks)} chunks...")
    embeddings = model.encode(chunks, show_progress_bar=True)
    return np.array(embeddings, dtype="float32")


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    """
    Builds a FAISS index from embeddings.

    IndexFlatL2 = exact search using L2 (Euclidean) distance.
    Simple and accurate — good for small to medium datasets.

    For millions of vectors you'd use IndexIVFFlat (approximate search).
    """
    dimension = embeddings.shape[1]  # 384 for MiniLM
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    print(f"FAISS index built with {index.ntotal} vectors (dimension={dimension})")
    return index


def search(query: str, index: faiss.IndexFlatL2, chunks: list[str], top_k: int = 3) -> list[str]:
    """
    Given a query string:
    1. Embeds the query into a vector
    2. Searches FAISS for the top_k most similar chunk vectors
    3. Returns the actual text of those chunks

    Args:
        query:   The user's question
        index:   The FAISS index (holds all chunk vectors)
        chunks:  The original text chunks (so we can return actual text)
        top_k:   How many chunks to retrieve (default 3)

    Returns:
        List of the most relevant text chunks
    """
    query_vector = model.encode([query])
    query_vector = np.array(query_vector, dtype="float32")

    # D = distances, I = indices of nearest neighbors
    D, I = index.search(query_vector, top_k)

    results = [chunks[i] for i in I[0] if i < len(chunks)]
    return results
