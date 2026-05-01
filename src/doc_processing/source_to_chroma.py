from src.doc_processing.source_to_chunks import prepare_source_for_chroma
import shutil
import os
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

# Load the .env file containing OPENAI_API_KEY
load_dotenv()

# --- Dynamic Path Logic ---
CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_SCRIPT_DIR, "..", ".."))
UPLOAD_DIR = os.path.join(PROJECT_ROOT, "data", "uploaded_file")
STORE_DIR = os.path.join(PROJECT_ROOT, "data", "chunks_store")
CHROMA_DB_PATH = os.path.join(PROJECT_ROOT, "data", "chroma_store")

import uuid  # Add this to your imports at the top

def save_chunks_to_chroma(ingestion_package: dict):
    db_path = CHROMA_DB_PATH

    # 1. REMOVED PHYSICAL WIPE (The WinError 32 Fix)
    # We no longer delete the folder. Chroma handles persistence for us.
    os.makedirs(db_path, exist_ok=True)

    # 2. Setup Client
    client = chromadb.PersistentClient(path=db_path)

    # 3. Setup Embedding Function
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.environ.get("OPENAI_API_KEY"),
        model_name="text-embedding-3-large"
    )

    # 4. Use get_or_create_collection (Prevents "Collection already exists" errors)
    collection = client.get_or_create_collection(
        name="technical_docs_collection",
        embedding_function=openai_ef,
        metadata={"hnsw:space": "cosine"}
    )

    # 5. Add Randomness to IDs to prevent collisions
    # This generates a unique suffix like 'arxiv_paper_chunk1_a4b2'
    unique_suffix = str(uuid.uuid4())[:4]
    new_ids = [f"{id_str}_{unique_suffix}" for id_str in ingestion_package["ids"]]

    # 6. Add the Data
    print(f"Adding {len(ingestion_package['documents'])} chunks to store...")
    collection.add(
        documents=ingestion_package["documents"],
        metadatas=ingestion_package["metadatas"],
        ids=new_ids
    )
    
    return collection


def process_and_save_full_pipeline(source_path: str, source_name: str):
    """
    Orchestrates the full RAG ingestion pipeline.
    """
    print(f"--- Phase 1: Processing {source_name} ---")
    ingestion_package = prepare_source_for_chroma(source_path, source_name)

    if not ingestion_package or "documents" not in ingestion_package:
        print("Error: Ingestion package is empty or invalid.")
        return

    print(f"--- Phase 2: Storing in ChromaDB (Fresh Start) ---")
    collection = save_chunks_to_chroma(ingestion_package)

    print(f"--- Pipeline Complete: {source_name} is ready for retrieval ---")

def clear_chroma_database():
    """Safely wipes the vector database collection."""
    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        client.delete_collection(name="technical_docs_collection")
        print("[OK] Database collection cleared successfully.")
        return True
    except Exception as e:
        print(f"[ERROR] Error clearing database: {e}")
        return False

# --- Usage ---
if __name__ == "__main__":
    process_and_save_full_pipeline("https://arxiv.org/pdf/2408.09869", "arxiv_paper")