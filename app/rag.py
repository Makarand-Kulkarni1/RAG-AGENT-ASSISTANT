from pathlib import Path
import numpy as np
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq
import re
import json
from datetime import datetime

from app.config import (
    GROQ_API_KEY, EMBEDDING_MODEL, CHAT_MODEL,
    CHUNK_SIZE, CHUNK_OVERLAP, TOP_K,
)


_embed_model = SentenceTransformer(EMBEDDING_MODEL)
_groq_client = Groq(api_key=GROQ_API_KEY)


def load_documents(data_dir: Path) -> list[dict]:
    docs = []
    for path in sorted(data_dir.glob("*")):
        if path.suffix.lower() == ".txt":
            docs.append({"source": path.name, "text": path.read_text(encoding="utf-8", errors="ignore")})
        elif path.suffix.lower() == ".pdf":
            reader = PdfReader(str(path))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            docs.append({"source": path.name, "text": text})
    return docs


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = " ".join(text.split())
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks
SECTION_HEADERS = [
    "PROFESSIONAL SUMMARY", "TECHNICAL SKILLS", "PROFESSIONAL EXPERIENCE",
    "PROJECTS", "EDUCATION", "CERTIFICATIONS",
]

def split_into_sections(text: str) -> list[dict]:
    pattern = "(" + "|".join(re.escape(h) for h in SECTION_HEADERS) + ")"
    parts = re.split(pattern, text)

    sections = []
    current_header = "HEADER"
    buffer = ""
    for part in parts:
        if part in SECTION_HEADERS:
            if buffer.strip():
                sections.append({"section": current_header, "text": buffer.strip()})
            current_header = part
            buffer = ""
        else:
            buffer += part
    if buffer.strip():
        sections.append({"section": current_header, "text": buffer.strip()})
    return sections


def build_index(data_dir: Path):
    """Returns (faiss_index, all_chunks, all_metadata) — call this once at startup."""
    docs = load_documents(data_dir)

    all_chunks, all_metadata = [], []
    for doc in docs:
        sections = split_into_sections(doc["text"])
        for sec in sections:
            for i, c in enumerate(chunk_text(sec["text"])):
                all_chunks.append(c)
                all_metadata.append({
                    "source": doc["source"],
                    "section": sec["section"],
                    "chunk_index": i,
                })

    embeddings = _embed_model.encode(all_chunks, show_progress_bar=True)
    embeddings = np.array(embeddings, dtype="float32")
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index, all_chunks, all_metadata


def retrieve(query: str, index, chunks: list, metadata: list, top_k: int = TOP_K) -> list[dict]:
    query_vec = _embed_model.encode([query])
    query_vec = np.array(query_vec, dtype="float32")
    faiss.normalize_L2(query_vec)

    scores, indices = index.search(query_vec, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        results.append({"text": chunks[idx], "source": metadata[idx]["source"], "score": float(score)})
    return results


def answer_question(query: str, index, chunks: list, metadata: list) -> str:
    results = retrieve(query, index, chunks, metadata)
    context = "\n\n".join(r["text"] for r in results)

    response = _groq_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": "Answer using ONLY the provided context. If it's not there, say you don't know."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ],
        temperature=0.2,
    )

    answer = response.choices[0].message.content
    log_query(query, answer)
    return answer

LOG_PATH = Path("logs/queries.jsonl")
LOG_PATH.parent.mkdir(exist_ok=True)

def log_query(question: str, answer: str):
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "question": question,
        "answer": answer,
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")