"""
vectorstore_utils.py

Contains utility functions for managing vector embeddings using FAISS.

Main responsibilities:
- Store and index text embeddings with FAISS
- Associate each embedding with metadata (e.g., original text chunks)
- Save and load FAISS indexes and metadata for persistent use
- Perform similarity search given a query embedding

Technologies used:
- FAISS for efficient vector similarity search (Facebook AI Similarity Search)
- NumPy for converting embeddings to NumPy arrays (FAISS only accepts np.ndarray)
- Pickle for storing metadata in binary format
"""
import faiss
import numpy as np
import pickle 
import logging
from typing import List, Dict, Optional, Any

def store_embeddings(
    embeddings: List[List[float]],
    metadata: Dict[int, Dict[str, Any]],
    index_path: str,
    metadata_path: str
) -> Optional[np.ndarray]:
    """
    Store embedding vectors and their associated metadata (e.g., text chunks) to disk.

    Converts the list of embedding vectors to a NumPy array, builds a FAISS index,
    and saves both the index and metadata for later retrieval or similarity-based search.

    Args:
        embeddings (list): List of embedding vectors (lists of floats).
        metadata (list): List of metadata items (e.g., text chunks) associated with each vector.
        index_path (str): Path where the FAISS index (.faiss file) will be saved.
        metadata_path (str): Path where the metadata (.pkl file) will be saved.

    Returns:
        np.ndarray or None: The NumPy array of embeddings if successful, otherwise None.
    """
    try:
        vectors = np.array(embeddings).astype("float32")
    except Exception as e:
        logging.error(f"[ERROR] Could not convert embeddings to NumPy: {e}")
        return
    
    try:
        build_faiss_index(vectors, save_path=index_path)
    except Exception as e:
        logging.error(f"[ERROR] Failed to build/save FAISS index: {e}")
        return

    try:
        save_metadata(metadata, path=metadata_path)
    except Exception as e:
        logging.error(f"[ERROR] Failed to save metadata: {e}")
        return

    logging.info("[INFO] Embeddings and metadata successfully stored.")
    return vectors
    


def build_faiss_index(
    vectors: np.ndarray,
    save_path: Optional[str] = None
) -> Optional[faiss.IndexFlatL2]:
    """
    Build a FAISS index from a NumPy array of embedding vectors.

    The index uses L2 (Euclidean) distance for similarity search and can be
    optionally saved to disk for future reuse.

    Args:
        vectors (np.ndarray): 2D NumPy array of shape (n_samples, embedding_dim),
                              where each row is an embedding vector.
        save_path (str, optional): Path to save the FAISS index to disk.
                                   If None, the index is not saved.

    Returns:
        faiss.IndexFlatL2 or None: A FAISS index object if successful, or None on failure.
    """
    # Get the number of dimensions from a single embedding - 1536 because we use openAI
    try: 
        dimension = vectors.shape[1]  
        logging.info(f"[INFO] Creating FAISS index with dimension: {dimension}")

        index = faiss.IndexFlatL2(dimension)
        index.add(vectors) # type: ignore[arg-type]
        logging.info(f"[INFO] FAISS index created with {index.ntotal} vectors.")

        # (Optional): Save the index to disk
        if save_path:
            faiss.write_index(index, save_path)
            logging.info(f"[INFO] FAISS index saved to '{save_path}'")

        return index
    except Exception as e:
        logging.error(f"[ERROR] Failed to build FAISS index: {e}")
        return None
    


def save_metadata(metadata_dict: Dict[int, Dict[str, Any]], path: str) -> None:
    """
    Save a metadata list (e.g., text chunks) to disk using pickle serialization.

    This ensures persistence of the original text data associated with vector embeddings,
    enabling later retrieval or use in similarity search tasks.

    Args:
        metadata_list (list): List of text chunks or any metadata corresponding to embeddings.
        path (str): Destination file path to save the metadata (.pkl format).

    Returns:
        None
    """
    try:
        with open(path,"wb") as f:
            pickle.dump(metadata_dict, f)
        logging.info(f"[INFO] Metadata successfully saved to: {path}")
    except Exception as e:
        logging.error(f"[ERROR] Failed to save metadata to '{path}': {e}")


def load_metadata(path: str) -> Optional[List[Dict[str, Any]]]:
    """
    Load a list of metadata (e.g., text chunks) from disk using pickle.

    This function is typically used to retrieve original text data that corresponds
    to previously generated vector embeddings (e.g., for search or result display).

    Args:
        path (str): Path to the pickle (.pkl) file where metadata is stored.

    Returns:
        list or None: The metadata list if loading succeeds, otherwise None on error.
    """
    try:
        with open(path,"rb") as f:
            metadata = pickle.load( f)
        return metadata
    except Exception as e:
        logging.error(f"[ERROR] Failed to load metadata from '{path}': {e}")
        return None

        
