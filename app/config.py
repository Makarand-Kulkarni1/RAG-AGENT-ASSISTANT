import os
from pathlib import Path

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable is not set")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHAT_MODEL = "openai/gpt-oss-20b"

CHUNK_SIZE = 300
CHUNK_OVERLAP = 100
TOP_K = 6

DATA_DIR = Path("data/sample_docs")