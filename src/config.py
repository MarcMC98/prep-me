import os
from dotenv import load_dotenv

load_dotenv()

# --- LLM / OpenRouter ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
RAG_MODEL = os.getenv("RAG_MODEL", "x-ai/grok-4.1-fast:free")

# --- Embeddings ---
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# --- Storage / Data ---
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_store")
DATA_DIR = os.getenv("DATA_DIR", "./data")

# --- Chunking / Retrieval ---
TOP_K = int(os.getenv("TOP_K", "4"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "400"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

if not OPENROUTER_API_KEY:
    raise ValueError(
        "Missing OPENROUTER_API_KEY in .env. "
        "Create .env and add your OpenRouter key."
    )
