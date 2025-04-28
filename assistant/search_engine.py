"""
search_engine.py

This module handles similarity search using a FAISS index and metadata. 
It loads the FAISS index and associated metadata from disk, 
creates embeddings for user queries, and performs vector-based similarity search 
to retrieve the most relevant text chunks.

Functions:
- load_index_and_metadata: Load FAISS index and metadata from disk.
- search_similar_chunks: Search for top-k most relevant text chunks based on query embedding.
"""

import faiss
import pickle
import numpy as np
import logging
from typing import List, Tuple, Dict, cast, Optional, Any
from openai import OpenAI
from assistant.embedding_utils import get_query_embedding

client = OpenAI()


def load_index_and_metadata(index_path: str, metadata_path: str) -> Tuple[faiss.Index, Dict[int, Dict]]:
    """
    Load the FAISS index and metadata from disk.

    Args:
        index_path (str): Path to the FAISS index file.
        metadata_path (str): Path to the metadata pickle file.

    Returns:
        Tuple[faiss.Index, Dict[int, Dict]]: The loaded FAISS index and metadata dictionary.
    """
    try:
        index = faiss.read_index(index_path)
    except Exception as e:
        logging.error(f"Failed to load FAISS index from {index_path}: {e}")
        raise

    try:
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)
    except Exception as e:
        logging.error(f"Failed to load metadata from {metadata_path}: {e}")
        raise

    return index, metadata


def search_similar_chunks(query: str, index: faiss.Index, metadata: Dict[int, Dict], top_k: int = 5) -> List[Dict]:
    """
    Search for the most relevant text chunks given a query using FAISS vector similarity.

    Args:
        query (str): The user's search query.
        index (faiss.Index): Loaded FAISS index.
        metadata (Dict[int, Dict]): Dictionary mapping vector IDs to chunk metadata.
        top_k (int): Number of top results to return.

    Returns:
        List[Dict]: List of top-k results with metadata and similarity scores.
    """
    try:
        query_embedding = get_query_embedding(query)
        query_vector = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
    except Exception as e:
        logging.error(f"Failed to generate query embedding: {e}")
        raise

    try:
        distances, indices = index.search(query_vector, top_k) # type: ignore
    except Exception as e:
        logging.error(f"FAISS search failed: {e}")
        raise

    results: List[Dict] = []
    for i, score in zip(indices[0], distances[0]):
        if i in metadata:
            result = metadata[i].copy()
            result["score"] = float(score)
            results.append(result)
        else:
            logging.warning(f"Vector index {i} not found in metadata.")

    return results


def generate_answer_from_chunks(query: str, chunks: List[str], mode: str = "study") -> str:
    """
    Use OpenAI GPT to generate an answer from a user query and a list of supporting chunks.
    Adjusts system prompt and behavior based on selected mode (e.g., study, exam, project).

    Args:
        query (str): The user's question.
        chunks (List[str]): List of relevant text chunks from FAISS.
        mode (str): Mode of operation: "study", "exam", or "project".

    Returns:
        str: GPT-generated answer.
    """
    try:
        context = "\n\n".join(chunks)

        if mode == "exam":
            system_prompt = "Είσαι αυστηρός καθηγητής και απαντάς σύντομα, με ακρίβεια και μόνο με βάση τις σημειώσεις."
        elif mode == "project":
            system_prompt = "Είσαι ένας βοηθός που βοηθάει σε υλοποίηση project. Δώσε τεχνικές και πρακτικές πληροφορίες."
        else:
            system_prompt = "Είσαι ένας εκπαιδευτικός βοηθός. Εξήγησε με απλό και φιλικό τρόπο βασισμένος στις σημειώσεις."

        prompt = (
            f"Χρήστης: {query}\n\n"
            "Παρακάτω σου δίνω αποσπάσματα από τις σημειώσεις. Χρησιμοποίησέ τα για να απαντήσεις στην ερώτηση:\n\n"
            f"--- Αποσπάσματα ---\n{context}\n--------------------\n\n"
            "Παρακαλώ απάντησε όσο πιο ακριβώς γίνεται με βάση τα παραπάνω μόνο."
        )

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )

        return response.choices[0].message.content.strip()  # type: ignore

    except Exception as e:
        logging.error(f"Failed to generate answer with GPT: {e}")
        return "Σφάλμα κατά τη δημιουργία απάντησης."

  