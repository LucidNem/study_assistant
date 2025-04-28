"""
user_profile.py

Handles logging and storage of user interactions with the assistant.
Tracks user queries, associated course, mode, retrieved chunks, and AI-generated answers.

Responsibilities:
- Maintain a persistent history of user activity
- Save user queries and results to a JSON file for later analysis or personalization

Technologies:
- JSON for structured data storage
- datetime for timestamps
- os for safe file access and creation
"""

import json
import os
import logging
from typing import List, Dict, Any
from datetime import datetime

def log_user_query(
    user_id: str,
    course: str,
    mode: str,
    query: str,
    answer: str,
    retrieved_chunks: List[Dict[str, Any]],
    profile_path: str = "data/user_data/user_profile.json"
) -> None:
    """
    Log the user's query, selected course and mode, retrieved chunks, and answer.

    Args:
        user_id (str): Unique identifier for the user (e.g., username or session ID).
        course (str): Course identifier the query was related to.
        mode (str): Current assistant mode (e.g., "study", "exam", "project").
        query (str): The question the user asked.
        answer (str): The assistant's response.
        retrieved_chunks (List[Dict[str, Any]]): Text chunks used to generate the answer.
        profile_path (str): Path to the JSON file where the profile history will be stored.

    Returns:
        None
    """
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(profile_path), exist_ok=True)

        # Load existing profile or initialize
        try:
            with open(profile_path, "r", encoding="utf-8") as f:
                profile_data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            profile_data = {}


        # Format log entry
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "course": course,
            "mode": mode,
            "query": query,
            "answer": answer,
            "chunks": [
                {
                    "chunk_id": chunk.get("chunk_id"),
                    "filename": chunk.get("filename"),
                    "text_preview": chunk.get("text", "")[:200]
                } for chunk in retrieved_chunks
            ]
        }

        # Append to user history
        if user_id not in profile_data:
            profile_data[user_id] = []

        profile_data[user_id].append(log_entry)

        # Save back to file
        with open(profile_path, "w", encoding="utf-8") as f:
            json.dump(profile_data, f, indent=2, ensure_ascii=False)

        logging.info(f"[INFO] User '{user_id}' query logged successfully.")

    except Exception as e:
        logging.error(f"[ERROR] Failed to log user query: {e}")
