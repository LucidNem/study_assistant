"""
pdf_reader.py

Handles reading and extracting text from PDF files using PyMuPDF.

Responsibilities:
- Load PDF documents from file
- Extract full text from all pages or per page

Technologies:
- PyMuPDF (fitz) for PDF parsing and text extraction
"""
import pymupdf
import logging
from typing import Optional, List

def load_pdf(file_path: str) -> Optional[pymupdf.Document]:
    """
    Load a PDF file and return the document object.

    Args:
        file_path (str): Path to the PDF file.

    Returns:
        PyMuPDF.Document or None: The loaded PDF document, or None if loading fails.
    """
    try:
        doc = pymupdf.open(file_path)
        return doc 
    except Exception as e:
        logging.error(f"[ERROR] Can not open PDF {e}")
        return None



def extract_text(doc: Optional[pymupdf.Document]) -> str:
    """
    Extract all text from the entire PDF document.

    Args:
        doc (PyMuPDF.Document): A loaded PDF document.

    Returns:
        str: Combined text from all pages of the PDF. Returns an empty string if doc is None.
    """
    if doc is None:
        return ""
    try:
        full_text = ""
        for page in doc:
            full_text += page.get_text()  # type: ignore[attr-defined]
        return full_text
    except Exception as e:
        logging.error(f"[ERROR] Failed to extract text: {e}")
        return ""



def extract_text_by_page(doc: Optional[pymupdf.Document]) -> List[str]:
    """
    Extract text from each page of the PDF individually.

    Args:
        doc (PyMuPDF.Document): A loaded PDF document.

    Returns:
        list: A list of strings, each representing text from one page.
              Returns an empty list if doc is None or extraction fails.
    """
    if doc is None:
        return []

    try:
        return [page.get_text() for page in doc] # type: ignore[attr-defined]
    except Exception as e:
        logging.error(f"[ERROR] Failed to extract text by page: {e}")
        return []