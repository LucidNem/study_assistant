"""
build_vectorstore.py

This script processes a PDF file to extract text, clean it, split it into chunks,
create embeddings using OpenAI, and store the vectors in a FAISS index along with
associated metadata. Each file is associated with a specific course. The index and
metadata are saved separately for each course to support course-specific retrieval.
"""

import os
import logging
import pickle
import faiss
import numpy as np
from typing import List, Dict, Any

from assistant.pdf_reader import extract_text, load_pdf
from assistant.text_utils import clean_text, split_text
from assistant.embedding_utils import get_embeddings


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )


def build_vectorstore(
    pdf_path: str,
    index_output_path: str,
    metadata_output_path: str,
    course: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> None:
    """
    Builds a FAISS index and metadata store from a given PDF file.

    Args:
        pdf_path (str): Path to the PDF file.
        index_output_path (str): Output path for the FAISS index.
        metadata_output_path (str): Output path for the metadata pickle file.
        course (str): Name of the course the PDF belongs to.
        chunk_size (int): Number of characters per text chunk.
        chunk_overlap (int): Number of overlapping characters between chunks.

    Returns:
        None
    """
    ""
    try:
        doc = load_pdf(pdf_path)
        raw_text = extract_text(doc)
        cleaned_text = clean_text(raw_text)
        chunks = split_text(cleaned_text, chunk_size, chunk_overlap)

        logging.info(f"Extracted {len(chunks)} chunks from PDF.")

        embeddings = get_embeddings(chunks)
        valid_embeddings = [e for e in embeddings if e is not None]

        if not valid_embeddings:
            raise ValueError("No valid embeddings were generated.")

        dimension = len(valid_embeddings[0])
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(valid_embeddings, dtype=np.float32))  # type: ignore[arg-type]

        # Creating metadata only for the chunks that were succesfull
        metadata: Dict[int, Dict] = {}
        valid_index = 0

        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            if embedding is not None:
                metadata[valid_index] = {
                    "text": chunk,
                    "chunk_id": valid_index,
                    "filename": os.path.basename(pdf_path),
                    "course": course  # Tag course
                }
                valid_index += 1

        # Save index and metadata
        faiss.write_index(index, index_output_path)
        with open(metadata_output_path, "wb") as f:
            pickle.dump(metadata, f)

        logging.info("Vector store created and saved successfully.")

    except Exception as e:
        logging.error(f"Error building vectorstore: {e}")
        raise


if __name__ == "__main__":
    setup_logging()

    build_vectorstore(
        pdf_path="data/pdfs/os/test.pdf",
        index_output_path="data/vector_store/os_index.faiss",
        metadata_output_path="data/vector_store/os_metadata.pkl",
        course="os"
    )