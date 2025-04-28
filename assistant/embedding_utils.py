"""
embedding_utils.py

Generates embeddings for text chunks using the OpenAI API.

Responsibilities:
- Authenticate and connect with OpenAI via environment variables
- Convert text chunks into vector embeddings using a specific model
- Track embedding progress and handle API errors with logging

Technologies:
- OpenAI Python SDK for embedding generation
- dotenv for API key loading from .env
- logging and time for monitoring and rate limit control
"""
from openai import OpenAI
from dotenv import load_dotenv
import os
import logging
import time
from typing import List, Optional


# Load environment variables and initialize OpenAI client
load_dotenv() 
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embeddings(chunks: List[str]) -> List[Optional[List[float]]]:
    """
    Generate embedding vectors for a list of text chunks using OpenAI's embedding API.

    This function sends each chunk of text to the OpenAI API and retrieves its embedding vector.
    It logs progress and errors, introduces a small delay between requests to prevent rate limits,
    and counts both successful and failed attempts.

    Args:
        chunks (list of str): A list of text strings to be converted into embeddings.

    Returns:
        list: A list of embedding vectors (list of floats). Failed chunks return None in their place.
    """
    embeddings = []
    success_count = 0
    fail_count = 0

    for i, chunk in enumerate(chunks):
        logging.info(f"[INFO] Processing chunk {i+1}/{len(chunks)}")
        try:
            response = client.embeddings.create(model = "text-embedding-ada-002", input = chunk)
            embedding = response.data[0].embedding
            embeddings.append(embedding)
            success_count += 1

        except Exception as e:
            logging.error(f"[ERROR] failed to embed chunk {i+1}: {e}")
            embeddings.append(None)
            fail_count += 1

        time.sleep(0.3)

    logging.info(f"[SUMMARY] Successfully embedded: {success_count}")
    logging.info(f"[SUMMARY] Failed to embed: {fail_count}")

    return embeddings


def get_query_embedding(query: str) -> List[float]:
    """
    Creates an embedding for the input query string using OpenAI API.
    
    Args:
        query (str): The user's query.

    Returns:
        list: The embedding vector of the query.
    """
    try:
        response = client.embeddings.create(
            input=query,
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding
    except Exception as e:
        logging.error(f"Error generating query embedding: {e}")
        raise
