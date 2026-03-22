from src.doc_processing.source_to_chunks import prepare_source_for_chroma
import shutil
import os
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

# Load the .env file containing OPENAI_API_KEY
load_dotenv()

def save_chunks_to_chroma(ingestion_package: dict):
    db_path = "D:/long_doc_agent/data/chroma_store"

    # 1. PHYSICAL WIPE
    # Instead of just client.reset(), we delete the actual folder
    if os.path.exists(db_path):
        print(f"🗑️ Physically deleting all files in {db_path}...")
        # Use shutil.rmtree to delete the folder and all subfolders (UUID folders)
        shutil.rmtree(db_path)
    
    # Re-create the empty directory
    os.makedirs(db_path)

    # 2. Setup Client
    # Since the folder is gone, this starts a brand-new database
    client = chromadb.PersistentClient(path=db_path)

    # 3. Setup Embedding Function
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.environ.get("OPENAI_API_KEY"),
        model_name="text-embedding-3-large"
    )

    # 4. Create fresh collection
    collection = client.create_collection(
        name="technical_docs_collection",
        embedding_function=openai_ef,
        metadata={"hnsw:space": "cosine"}
    )

    # 5. Add the Data
    print(f"📥 Saving {len(ingestion_package['documents'])} chunks to fresh store...")
    collection.add(
        documents=ingestion_package["documents"],
        metadatas=ingestion_package["metadatas"],
        ids=ingestion_package["ids"]
    )
    
    return collection


def process_and_save_full_pipeline(source_path: str, source_name: str):
    """
    Orchestrates the full RAG ingestion pipeline.
    """
    print(f"--- Phase 1: Processing {source_name} ---")
    ingestion_package = prepare_source_for_chroma(source_path, source_name)

    print(f"--- Phase 2: Storing in ChromaDB (Fresh Start) ---")
    # We no longer pass source_name here because we use a fixed session name
    collection = save_chunks_to_chroma(ingestion_package)

    print(f"--- Pipeline Complete: {source_name} is ready for retrieval ---")

# --- Usage ---
if __name__ == "__main__":
    process_and_save_full_pipeline("https://arxiv.org/pdf/2408.09869", "arxiv_paper")