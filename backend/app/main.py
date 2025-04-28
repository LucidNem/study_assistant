from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from assistant.search_engine import load_index_and_metadata, search_similar_chunks
from assistant.qa_engine import generate_answer  

app = FastAPI()

# Για να επιτρέπει το React frontend να κάνει αιτήσεις
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    course: str
    mode: str

@app.post("/query")
def query_route(payload: QueryRequest):
    try:
        index_path = f"data/vector_store/{payload.course.lower()}_index.faiss"
        metadata_path = f"data/vector_store/{payload.course.lower()}_metadata.pkl"

        index, metadata = load_index_and_metadata(index_path, metadata_path)
        chunks = search_similar_chunks(payload.query, index, metadata, top_k=3)
        answer = generate_answer(payload.query, chunks)

        return {
            "answer": answer,
            "chunks": chunks
        }

    except Exception as e:
        return {"error": str(e)}
