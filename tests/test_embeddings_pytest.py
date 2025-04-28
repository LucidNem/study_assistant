import os
import time
import pytest
from assistant.embedding_utils import get_embeddings

# Sample test data
chunks = [
    "Hello world, this is a test.",
    "OpenAI embeddings are cool!",
    "This should return a vector."
]


# Fixture that generates text embeddings once for all tests in this module.

# - Calls the `get_embeddings()` function with predefined input (`chunks`)
# - Measures execution time to allow performance testing
# - Returns both the list of embedding vectors and the duration in seconds
# - Decorated with `scope="module"` so it runs only once per test session

# This avoids redundant API calls and ensures consistent test inputs/results.
    
# Returns:
#     tuple: (embeddings: list[list[float]], duration: float)
    
@pytest.fixture(scope="module")
def embeddings_result():
    start = time.time()
    embeddings = get_embeddings(chunks)
    duration = time.time() - start
    return embeddings, duration

def test_embedding_count(embeddings_result):
    embeddings, _ = embeddings_result
    assert len(embeddings) == len(chunks), "Number of embeddings should match input chunks"

def test_no_none_embeddings(embeddings_result):
    embeddings, _ = embeddings_result
    assert all(e is not None for e in embeddings), "Some embeddings failed (returned None)"

def test_embedding_vector_type_and_length(embeddings_result):
    embeddings, _ = embeddings_result
    first = embeddings[0]
    assert isinstance(first, list), "Embedding should be of type list"
    assert len(first) > 100, "Embedding vector is unexpectedly short"

def test_log_file_created():
    log_files = [f for f in os.listdir() if f.startswith("embedding_log_") and f.endswith(".txt")]
    assert len(log_files) > 0, "No log file with timestamp was created"

def test_embedding_execution_time(embeddings_result):
    _, duration = embeddings_result
    assert duration < 10, f"Embedding took too long to run: {duration:.2f} seconds"
