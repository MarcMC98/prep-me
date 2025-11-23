# PrepMe Notes RAG Agent

A local Retrieval-Augmented Generation (RAG) system to answer questions from your own notes.

Supports:
- TXT notes
- PDF documents
- HTML exports
- Persistent Chroma vector store
- Strict context-only answering with citations

---

## Setup

1. Put notes inside `./data/`

2. Create `.env`:

```env
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxx
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
RAG_MODEL=x-ai/grok-4.1-fast:free
EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHROMA_DIR=./chroma_store
DATA_DIR=./data
TOP_K=8
CHUNK_SIZE=600
CHUNK_OVERLAP=120
