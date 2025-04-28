"""
test_embedding_return.py

Test script for inspecting the structure of embedding responses from the OpenAI API.

Responsibilities:
- Send a sample text chunk to OpenAI's embedding endpoint
- Print the type and structure of the returned embedding response
- Verify the dimensions and contents of the embedding vector

Technologies:
- OpenAI Python SDK (>= 1.0)
- dotenv for loading API key
- Standard Python print and type inspection
"""
import os
from openai import OpenAI
from dotenv import load_dotenv
import logging

# Load API key from .env file
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Sample text to embed
text = "This is a test chunk for checking how OpenAI embedding response is structured."

try:
    # Request embedding from OpenAI API
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )

    # Logging response details
    logging.info(f"Type of response: {type(response)}")  # CreateEmbeddingResponse
    logging.info(f"Type of response.data: {type(response.data)}")  # list
    logging.info(f"Type of response.data[0]: {type(response.data[0])}")  # Embedding object
    logging.info(f"Type of embedding: {type(response.data[0].embedding)}")  # list
    logging.info(f"Length of embedding vector: {len(response.data[0].embedding)}")
    logging.info(f"First 5 values of embedding vector: {response.data[0].embedding[:5]}")

except Exception as e:
    logging.error(f"[ERROR] Failed to retrieve embedding: {e}")
