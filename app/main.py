from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel

from app.rag import build_index
from app.agent import run_agent

app = FastAPI(title="RAG Agent Assistant")

# Build the index once, when the server starts — not on every request
_index, _chunks, _metadata = build_index(Path("data/sample_docs"))


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str


@app.post("/ask", response_model=QueryResponse)
def ask(request: QueryRequest):
    answer = run_agent(request.question, _index, _chunks, _metadata)
    return QueryResponse(answer=answer)


@app.get("/health")
def health():
    return {"status": "ok", "chunks_indexed": len(_chunks)}