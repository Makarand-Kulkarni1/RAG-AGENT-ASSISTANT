# RAG Agent Assistant

A Retrieval-Augmented Generation (RAG) system built from first principles — no high-level RAG frameworks — combining document retrieval, agentic tool-use, evaluation, and a live dashboard. Built as a portfolio project to demonstrate end-to-end AI engineering: from raw document ingestion through a deployed, containerized application.

## What it does

Ask natural-language questions about a set of documents (PDFs/text files) and get answers grounded in that content. The assistant can also invoke external tools (e.g., a calculator) via function-calling when a question requires it, rather than relying solely on retrieved text.

## Features

- **Retrieval-Augmented Generation (RAG)** — section-aware document chunking, local embeddings (Sentence-Transformers), and FAISS vector search for semantic retrieval
- **Agentic tool-use** — function-calling via the Groq LLM API, allowing the model to decide when to call a tool (e.g., a calculator) instead of answering from text alone
- **Evaluation pipeline** — a hand-built Q&A test set measures retrieval accuracy; used to diagnose and fix a real chunking bug that was fragmenting facts across chunk boundaries, improving retrieval accuracy from 60% to 80%
- **Query logging** — every question and answer is logged to a JSONL file for later inspection
- **FastAPI backend** — a REST API (`/ask`, `/health`) with Pydantic request/response validation and interactive Swagger docs
- **Streamlit dashboard** — live querying, query-log browsing, and on-demand evaluation runs, all in one interface
- **Dockerized** — fully containerized for reproducible deployment, with secrets managed via environment variables rather than hardcoded credentials

## Tech stack

Python · PyTorch (via Sentence-Transformers) · FAISS · Groq LLM API · FastAPI · Pydantic · Docker · Streamlit · Git/GitHub

## Architecture

```
Documents (PDF/TXT)
      │
      ▼
Section-aware chunking ──▶ Local embeddings (PyTorch/Sentence-Transformers) ──▶ FAISS index
                                                                                     │
User question ──▶ Retrieval (top-k similarity search) ◀─────────────────────────────┘
      │
      ▼
LLM (Groq) — decides: answer directly, or call a tool (function-calling)
      │
      ▼
Grounded answer ──▶ logged to logs/queries.jsonl
```

## Project structure

```
app/
├── config.py      # Environment-driven configuration (models, chunking params, paths)
├── rag.py         # Document loading, chunking, embedding, retrieval, answer generation
├── tools.py       # Agent tool definitions (e.g., calculator) and function-calling schema
├── agent.py       # Agent loop: retrieval + tool-use orchestration
└── main.py        # FastAPI app (/ask, /health endpoints)
eval/
├── eval_set.json  # Hand-built Q&A test set for retrieval evaluation
└── evaluate.py    # Runs the eval set and reports accuracy
data/
└── sample_docs/   # Source documents to index
streamlit_app.py   # Interactive dashboard (ask, query log, evaluation)
Dockerfile
requirements.txt
```

## Setup

**1. Clone and install dependencies**
```bash
git clone https://github.com/Makarand-Kulkarni1/RAG-AGENT-ASSISTANT.git
cd RAG-AGENT-ASSISTANT
pip install -r requirements.txt
```

**2. Set your Groq API key** (get one free at [console.groq.com](https://console.groq.com))
```bash
# PowerShell
$env:GROQ_API_KEY="your_key_here"

# macOS/Linux
export GROQ_API_KEY="your_key_here"
```

**3. Add documents** to `data/sample_docs/` (`.pdf` or `.txt`)

**4. Run the FastAPI server**
```bash
python -m uvicorn app.main:app --reload
```
Visit `http://127.0.0.1:8000/docs` for interactive API docs.

**5. Or run the Streamlit dashboard**
```bash
python -m streamlit run streamlit_app.py
```
Visit `http://localhost:8501`.

**6. Run the evaluation suite**
```bash
python -m eval.evaluate
```

## Running with Docker

```bash
docker build -t rag-agent-assistant .
docker run -p 8000:8000 -e GROQ_API_KEY="your_key_here" rag-agent-assistant
```

## Known limitations

- **Embedding model limitations**: the lightweight `all-MiniLM-L6-v2` model doesn't always strongly associate paraphrased questions with the correct chunk (e.g., "what company is this person interning at" vs. a resume stating "Data Science Intern — X"), even when chunking is clean. This is a documented finding from the evaluation process, not a bug.
- **Cloud hosting on free tiers**: local embedding models (via PyTorch) require more memory than typical free-tier cloud hosting provides (e.g., Render's free tier caps at 512MB). Production deployment of this architecture would either need a paid tier with more RAM or a switch to a hosted embeddings API instead of a local model.

## What this project demonstrates

Chunking strategy design (and debugging a real section-bleeding bug), retrieval evaluation methodology, agentic tool-use via function-calling, secure secrets management, containerization, and honest identification of a real production trade-off (memory vs. cost) when evaluating deployment options.
