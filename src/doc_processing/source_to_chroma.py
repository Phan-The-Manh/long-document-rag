from src.doc_processing.source_to_chunks import prepare_source_for_chroma
from src.llm_clients.openai import get_chroma_embedding_function
import shutil
import os
import chromadb
from dotenv import load_dotenv

# Load the .env file containing OPENAI_API_KEY
load_dotenv()

# --- Dynamic Path Logic ---
CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_SCRIPT_DIR, "..", ".."))
UPLOAD_DIR = os.path.join(PROJECT_ROOT, "data", "uploaded_file")
STORE_DIR = os.path.join(PROJECT_ROOT, "data", "chunks_store")
CHROMA_DB_PATH = os.path.join(PROJECT_ROOT, "data", "chroma_store")

COLLECTION_NAME = "technical_docs_collection"


def _get_collection():
    os.makedirs(CHROMA_DB_PATH, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    openai_ef = get_chroma_embedding_function()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=openai_ef,
        metadata={"hnsw:space": "cosine"},
    )


def save_chunks_to_chroma(ingestion_package: dict):
    collection = _get_collection()

    # IDs are deterministic content hashes from build_enriched_chunk_and_metadata.
    # Upsert makes re-ingestion of unchanged chunks a no-op rather than a duplicate.
    print(f"Upserting {len(ingestion_package['documents'])} chunks to store...")
    collection.upsert(
        documents=ingestion_package["documents"],
        metadatas=ingestion_package["metadatas"],
        ids=ingestion_package["ids"],
    )

    return collection


def delete_document(doc_id: str, version: str = "v1", tenant_id: str = "default") -> int:
    """Delete every chunk for one (tenant, doc_id, version). Returns chunks removed."""
    collection = _get_collection()
    where = {"$and": [
        {"tenant_id": tenant_id},
        {"doc_id": doc_id},
        {"version": version},
    ]}
    existing = collection.get(where=where, include=[])
    ids = existing.get("ids", [])
    if not ids:
        print(f"[INFO] No chunks found for {tenant_id}/{doc_id}/{version}.")
        return 0
    collection.delete(ids=ids)
    print(f"[OK] Deleted {len(ids)} chunks for {tenant_id}/{doc_id}/{version}.")
    return len(ids)


def list_documents(tenant_id: str | None = None) -> list[dict]:
    """List distinct documents in the store, keyed by (tenant, doc_id, version)."""
    collection = _get_collection()
    where = {"tenant_id": tenant_id} if tenant_id else None
    result = collection.get(where=where, include=["metadatas"])
    metadatas = result.get("metadatas", []) or []

    aggregated: dict[tuple, dict] = {}
    for meta in metadatas:
        key = (
            meta.get("tenant_id", "default"),
            meta.get("doc_id", "unknown"),
            meta.get("version", "v1"),
        )
        entry = aggregated.setdefault(key, {
            "tenant_id": key[0],
            "doc_id": key[1],
            "version": key[2],
            "source": meta.get("source", "unknown"),
            "chunk_count": 0,
        })
        entry["chunk_count"] += 1
    return sorted(aggregated.values(), key=lambda d: (d["tenant_id"], d["doc_id"], d["version"]))


def process_and_save_full_pipeline(source_path: str, source_name: str):
    """
    Orchestrates the full RAG ingestion pipeline.
    """
    print(f"--- Phase 1: Processing {source_name} ---")
    ingestion_package = prepare_source_for_chroma(source_path, source_name)

    if not ingestion_package or "documents" not in ingestion_package:
        raise ValueError("Ingestion package is empty or invalid.")

    print(f"--- Phase 2: Storing in ChromaDB (Fresh Start) ---")
    collection = save_chunks_to_chroma(ingestion_package)

    print(f"--- Pipeline Complete: {source_name} is ready for retrieval ---")

def clear_chroma_database():
    """Wipes the entire collection. Use only for tests / dev REPL — prefer
    delete_document() on real app paths so other docs survive."""
    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        client.delete_collection(name=COLLECTION_NAME)
        print("[OK] Database collection cleared successfully.")
        return True
    except Exception as e:
        print(f"[ERROR] Error clearing database: {e}")
        return False

# --- Usage ---
if __name__ == "__main__":
    process_and_save_full_pipeline("https://arxiv.org/pdf/2408.09869", "arxiv_paper")