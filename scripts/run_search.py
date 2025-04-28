"""
run_search.py

This script allows you to interactively search for relevant text chunks 
based on a user query. It uses a FAISS vector index and metadata store 
to perform similarity search on previously embedded documents and generates 
answers using OpenAI GPT models.

How it works:
1. Loads a FAISS index and corresponding metadata from disk.
2. Prompts the user for a query.
3. Converts the query to an embedding.
4. Searches the index for the most similar chunks.
5. Displays the results with metadata and similarity scores.
6. Passes results to GPT to generate a natural language answer.

Usage:
    Run the script directly and enter queries when prompted.

Example:
    $ python scripts/run_search.py
"""

import logging
from assistant.user_profile import log_user_query
from assistant.search_engine import (
    load_index_and_metadata,
    search_similar_chunks,
    generate_answer_from_chunks
)


def setup_logging() -> None:
    """
    Configure logging to display errors and important messages.

    Returns:
        None
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )


def run_interactive_search(index_path: str, metadata_path: str, top_k: int = 3, course: str = "", mode: str = "study") -> None:

    """
    Run an interactive loop where the user can enter queries 
    and receive the most relevant text chunks and a GPT-generated answer.

    Args:
        index_path (str): Path to the FAISS index file.
        metadata_path (str): Path to the metadata pickle file.
        top_k (int): Number of top results to retrieve per query.

    Returns:
        None
    """
    try:
        print(f"\n--- COURSE: {course.upper()} | MODE: {mode.upper()} ---\n")
        print("Interactive Search Mode (type 'exit' to quit)\n")

        logging.info("Loading FAISS index and metadata...")
        index, metadata = load_index_and_metadata(index_path, metadata_path)
        print("FAISS index size:", index.ntotal)
        print("Metadata size:", len(metadata))

        logging.info("Index and metadata loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to initialize search engine: {e}")
        return

    print("\nInteractive Search Mode (type 'exit' to quit)\n")

    while True:
        try:
            query = input("Enter your question: ").strip()
            if query.lower() == "exit":
                print("Exiting search.")
                break

            results = search_similar_chunks(query, index, metadata, top_k=top_k)

            if not results:
                print("No results found.")
                continue

            print("\nTop Relevant Chunks:\n")
            for i, r in enumerate(results, 1):
                print(f"Result #{i}")
                print(f"Chunk ID: {r.get('chunk_id')} | Score: {r['score']:.4f}")
                if "filename" in r:
                    print(f"File: {r['filename']}")
                print(f"Text Preview:\n{r['text'][:300]}...")
                print("-" * 60)

            top_chunks = [r["text"] for r in results]
            answer = generate_answer_from_chunks(query, top_chunks, mode= mode)

            print("\nAnswer:\n", answer)
            print("=" * 80)

            
            log_user_query(
                user_id="user_001",  # προσωρινά σταθερό 
                course=course,
                mode=mode,
                query=query,
                answer=answer,
                retrieved_chunks=results)

        except KeyboardInterrupt:
            print("\nSearch interrupted by user.")
            break
        except Exception as e:
            logging.error(f"Error during search: {e}")


def main() -> None:
    """
    Entry point for the script. Defines file paths and starts the search loop.

    Returns:
        None
    """
    
    setup_logging()
    course = "os"
    mode = "study"

    index_path = f"data/vector_store/{course}_index.faiss"
    metadata_path = f"data/vector_store/{course}_metadata.pkl"

    run_interactive_search(index_path, metadata_path, top_k=3, course=course, mode=mode)


if __name__ == "__main__":
    main()
