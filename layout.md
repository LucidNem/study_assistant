```
study_assistant/
│
├── assistant/                      # Main application package
│   ├── __init__.py
│   ├── pdf_reader.py              # Functions for loading and extracting text from PDFs
│   ├── text_utils.py              # Text cleaning and chunking utilities
│   ├── embedding_utils.py         # Embedding creation with OpenAI API
│   ├── vectorstore_utils.py       # FAISS index building and metadata storage
│   └── utils/                     # General-purpose helper modules
│       ├── __init__.py
│       ├── timers.py              # Execution timing functions
│       └── .gitkeep               # Keeps utils folder in Git even if empty
│
├── tests/                         # Unit tests and test utilities
│   ├── __init__.py
│   ├── test_embedding_return.py   # Test for checking OpenAI embedding structure
│   └── .gitkeep                   # Keeps tests folder tracked even if empty
│
├── data/                          # Input and output data (excluded from Git)
│   ├── pdfs/                      # Input PDFs
│   │   └── .gitkeep               # Keeps pdfs folder in Git even if no PDFs
│   └── vector_store/             # FAISS index and metadata files
│       ├── index.faiss
│       ├── metadata_store.pkl
│       └── .gitkeep               # Keeps vector_store folder tracked when empty
│
├── logs/                          # Output logs from embedding pipeline
│   └── .gitkeep                   # Keeps logs folder tracked even when empty
│
├── .gitignore                     # Specifies files/folders Git should ignore
├── requirements.txt               # Project dependencies
├── layout.md                      # This file: project structure documentation
├── README.md                      # Project overview and usage instructions
└── main.py                        # Main pipeline entry point
```
