"""
qa_engine.py

Simulates AI answer generation by combining the user query with retrieved chunks.
Used as a dummy fallback until GPT integration is added.

Returns a formatted answer string based on context.
"""

from typing import List, Dict

def generate_answer(query: str, chunks: List[Dict]) -> str:
    """
    Simulate an AI-generated answer based on retrieved chunks.

    Args:
        query (str): User's question.
        chunks (List[Dict]): Relevant chunks from vector store.

    Returns:
        str: Simulated answer string.
    """
    context = "\n\n".join(chunk.get("text", "") for chunk in chunks)
    return f"""[DEMO ANSWER]
Ερώτηση: {query}

🧠 Σχετικά αποσπάσματα:
{context[:700]}...
"""
