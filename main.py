"""
main.py

Pipeline orchestrator: handles full PDF-to-embeddings workflow.

Responsibilities:
- Load, clean, and split PDF text
- Generate embeddings for each chunk
- Save embeddings to FAISS and metadata to disk
- Acts as the main entry point for local testing and development
"""
import os
import logging
import time
from assistant.pdf_reader import load_pdf, extract_text
from assistant.text_utils import split_text, clean_text
from assistant.vectorstore_utils import store_embeddings
from assistant.embedding_utils import get_embeddings

# Configure logging to output both to terminal and to a uniquely named file (with timestamp)
# This ensures that each execution has its own separate log file for better tracking and debugging.
log_filename = f"logs/embedding_log_{time.strftime('%Y-%m-%d_%H-%M-%S')}.txt"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()  
    ]
)



def main():
    pdf_path = "data/pdfs/test.pdf"
    index_pdf_path = "data/vector_store/index.faiss"
    metadata_pdf_path = "data/vector_store/metadata_store.pkl"

    #Check that the file exists
    if not os.path.isfile(pdf_path):
        logging.error(f"[ERROR] File not found: {pdf_path}")
        return
    
    try:
        doc = load_pdf(pdf_path)
    except Exception as e:
        logging.error(f"[ERROR] Failed to load PDF: {e}")
        return
    
    try:
        raw_text = extract_text(doc)
    except Exception as e:
        logging.error(f"[ERROR] Failed to extract the text : {e}")
        return
    
    try:
        cleaned_text = clean_text(raw_text)
        logging.info("Preview of cleaned text:\n" + cleaned_text[:500] + "...")
    except Exception as e:
        logging.error(f"[ERROR] Failed to clean the text: {e}")
        return

    chunks = split_text(cleaned_text, 500, 50)
    logging.info(f"Number of chunks: {len(chunks)}")
    logging.info(f"Sample chunk: {chunks[0]}")


    embeddings = get_embeddings(chunks)
    if not embeddings or all(e is None for e in embeddings):
        logging.error("[ERROR] Embedding generation failed. Aborting.")
        return

    store_embeddings(embeddings, chunks, index_pdf_path, metadata_pdf_path) # type: ignore[arg-type]



#Ensure that main will be executed only if it is called directly ant not if it is imported somwhere else
if __name__ == "__main__":
    main()




