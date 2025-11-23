from typing import List, Dict, Any

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from openai import OpenAI

from .config import (
    OPENROUTER_API_KEY, OPENROUTER_BASE_URL, RAG_MODEL,
    EMBED_MODEL, CHROMA_DIR, TOP_K
)


def get_clients():
    """
    Create LLM client, embedding model, and persistent Chroma client.
    """
    llm_client = OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
    )

    embedder = SentenceTransformer(EMBED_MODEL)

    chroma_client = chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(allow_reset=False)
    )

    return llm_client, embedder, chroma_client


def index_chunks(chunks: List, embedder: SentenceTransformer, chroma_client):
    """
    Index chunks into a persistent Chroma collection with dedup.
    Dedup key: "source::chunk_index".
    """
    collection = chroma_client.get_or_create_collection("rag_collection")

    texts = [c.page_content for c in chunks]
    metadatas = [c.metadata for c in chunks]

    ids = []
    for m in metadatas:
        src = m.get("source", "unknown")
        cidx = m.get("chunk_index", -1)
        ids.append(f"{src}::chunk_{cidx}")

    # Existing ids (if any)
    existing = set()
    try:
        existing_data = collection.get(include=[])
        existing = set(existing_data["ids"])
    except Exception:
        pass

    add_texts, add_metas, add_ids = [], [], []
    for t, m, i in zip(texts, metadatas, ids):
        if i not in existing and t:
            add_texts.append(t)
            add_metas.append(m)
            add_ids.append(i)

    if not add_texts:
        print("No new chunks to index.")
        return collection

    embeddings = embedder.encode(add_texts, normalize_embeddings=True).tolist()

    collection.add(
        ids=add_ids,
        documents=add_texts,
        metadatas=add_metas,
        embeddings=embeddings
    )

    print(f"Indexed new vectors: {len(add_ids)} | Total: {collection.count()}")
    return collection


def retrieve(query: str, embedder: SentenceTransformer, collection, k: int = TOP_K):
    """
    Vector search for top-k chunks.
    Returns list of hits with text/meta/distance.
    """
    q_emb = embedder.encode([query], normalize_embeddings=True)[0].tolist()

    res = collection.query(
        query_embeddings=[q_emb],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]

    hits = []
    for doc, meta, dist in zip(docs, metas, dists):
        hits.append({
            "text": doc,
            "meta": meta or {},
            "distance": float(dist)
        })

    return hits


def format_context(hits: List[Dict[str, Any]]) -> str:
    """
    Turn retrieved hits into a numbered context block with citations.
    """
    lines = []
    for i, h in enumerate(hits, start=1):
        src = h["meta"].get("source", "unknown")
        cidx = h["meta"].get("chunk_index", -1)
        lines.append(f"[{i}] Source: {src} (chunk {cidx})\n{h['text']}")
    return "\n\n".join(lines)


def answer(llm_client: OpenAI, context: str, query: str, chat_history: List[Dict[str, str]]):
    """
    Generate answer strictly grounded in context.
    Chat history is used only for conversational continuity,
    NOT as evidence.
    """
    system_msg = (
        "You are a study-notes assistant. "
        "Answer ONLY using the provided context. "
        "If context is insufficient, say you don't know. "
        "Do not use outside knowledge."
    )

    user_msg = f"""Context:
{context}

Question:
{query}

Rules:
- Use ONLY the context above.
- If the answer isn't in context, say exactly: "I don't know from the provided documents."
- Be concise but complete.
"""

    messages = [{"role": "system", "content": system_msg}]
    messages.extend(chat_history[-6:])  # keep short to avoid drift
    messages.append({"role": "user", "content": user_msg})

    res = llm_client.chat.completions.create(
        model=RAG_MODEL,
        messages=messages,
        temperature=0.2
    )

    return res.choices[0].message.content
