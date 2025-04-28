# Study Assistant – PDF-to-Embeddings Pipeline

This project provides an automated pipeline for converting PDF files into vector embeddings using OpenAI's API. It includes preprocessing, chunking, embedding generation, and storage in a FAISS index for efficient similarity search.

## Features

- PDF text extraction and cleanup
- Text chunking with overlapping windows
- Embedding generation using OpenAI (text-embedding-ada-002)
- FAISS index creation for vector similarity search
- Metadata storage for chunked text using pickle
- Logging and test utilities included

## Directory Structure

See `layout.md` for a full overview.

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt


Setup
Create a .env file inside the assistant/ directory and add your OpenAI API key:
OPENAI_API_KEY=your_key_here

Ensure the following folders exist:
data/pdfs/
data/vector_store/
logs/

Usage
Run the pipeline from the root of the project:

python main.py
This will:

Read the input PDF at data/pdfs/test.pdf

Extract and clean its text

Split the text into chunks

Generate embeddings for each chunk

Save the FAISS index and metadata to data/vector_store/

Logging
Execution logs are saved in the logs/ directory and printed to the terminal. Each run generates a timestamped .txt file.

Testing
Basic tests and embedding checks can be found in the tests/ directory. Example:

pytest tests/test_embedding_return.py
Technologies Used
Python 3.12+

OpenAI API

FAISS

NumPy

PyMuPDF (for PDF parsing)

Pickle

Logging

dotenv

Future Extensions
Add support for local models (e.g., SentenceTransformers)

RAG and search functionality (query-time vector similarity)

Study assistant features (questions, mock exams, feedback)



---
