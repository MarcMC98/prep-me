from typing import List, Dict

from .config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K
from .ingest import load_documents, split_documents
from .rag import get_clients, index_chunks, retrieve, format_context, answer


def run_cli():
    llm_client, embedder, chroma_client = get_clients()

    print("Loading documents...")
    docs = load_documents(DATA_DIR)
    print(f"Loaded {len(docs)} documents.")

    print("Splitting into chunks...")
    chunks = split_documents(docs, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"Created {len(chunks)} chunks.")

    print("Indexing to Chroma (persistent)...")
    collection = index_chunks(chunks, embedder, chroma_client)

    chat_history: List[Dict[str, str]] = []
    last_hits = []

    print("\nPrepMe RAG ready.")
    print("Commands:")
    print("  :reindex  -> rescan data dir and index new chunks")
    print("  :sources  -> show sources for last answer")
    print("  exit      -> quit\n")

    while True:
        query = input("> ").strip()
        if not query:
            continue

        if query.lower() in {"exit", "quit"}:
            break

        if query == ":reindex":
            docs = load_documents(DATA_DIR)
            chunks = split_documents(docs, CHUNK_SIZE, CHUNK_OVERLAP)
            collection = index_chunks(chunks, embedder, chroma_client)
            continue

        if query == ":sources":
            if not last_hits:
                print("No retrieval yet.")
            else:
                for h in last_hits:
                    src = h["meta"].get("source")
                    cidx = h["meta"].get("chunk_index")
                    dist = h["distance"]
                    print(f"- {src} (chunk {cidx}) dist={dist:.4f}")
            continue

        last_hits = retrieve(query, embedder, collection, k=TOP_K)
        context = format_context(last_hits)

        ans = answer(llm_client, context, query, chat_history)

        print("\n---- Answer ----")
        print(ans)
        print("---------------\n")

        chat_history.append({"role": "user", "content": query})
        chat_history.append({"role": "assistant", "content": ans})
