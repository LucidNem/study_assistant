"""
text_utils.py

Text preprocessing utilities for preparing input to LLMs or embedding models.

Responsibilities:
- Clean text by removing noise, special characters, and normalizing content
- Perform Unicode normalization (e.g., removing accents)
- Split text into overlapping chunks for embedding or LLM input

Technologies:
- re for regular expressions
- unicodedata for character normalization
"""

import re
import unicodedata
import logging
from typing import List, Dict, Optional, Any

def clean_text(text: str) -> str:
    """
    Clean and normalize input text for NLP or embedding purposes.

    This function:
    - Removes accent marks (e.g., transforms "ά" → "α")
    - Keeps only technical and meaningful characters
    - Replaces tabs with spaces
    - Normalizes multiple newlines
    - Trims whitespace from start and end

    Args:
        text (str): Raw input text.

    Returns:
        str: Cleaned and normalized text ready for further processing.
    """
    try:
        if not text:
            return ""
        # Remove accent marks using Unicode normalization
        text = unicodedata.normalize("NFKD", text)
        text = ''.join(char for char in text if not unicodedata.combining(char))
        text = re.sub(r"[^Α-Ωα-ωA-Za-z0-9\s.,;:!?(){}\[\]\"'=+\-*/<>%&#|~^@_\\∑∫≠≤≥→⇒∈∀∃π√]","",text)
        text = text.replace('\t', ' ')
        text = re.sub(r'\n+', '\n', text)
        text = text.strip()
        return text
    except Exception as e:
        logging.error(f"[ERROR] Failed to clean text: {e}")
        return ""



def split_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Split the input text into overlapping chunks of a specified size.

    Useful for chunking long text into smaller segments suitable for LLM input or embeddings,
    while preserving some context between chunks via overlap.

    Args:
        text (str): The full input text to be chunked.
        chunk_size (int): Maximum number of characters per chunk.
        overlap (int): Number of characters to repeat between chunks to preserve context.

    Returns:
        list: List of text chunks (strings). If input is empty or fails, returns an empty list.
    """
    try:
        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += chunk_size - overlap

        return chunks

    except Exception as e:
        logging.error(f"[ERROR] Failed to split text into chunks: {e}")
        return []

