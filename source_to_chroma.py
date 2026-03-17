from source_to_chunks import prepare_source_for_chroma

import os
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

# Load the .env file containing OPENAI_API_KEY
load_dotenv()

def save_chunks_to_chroma(ingestion_package: dict, source_name: str):
    """
    Initializes ChromaDB in D:/doc_processing and saves chunks.
    """
    # 1. Define the specific D: drive path
    db_path = "D:/doc_processing/chromadb_store"
    
    # Ensure the directory exists
    if not os.path.exists(db_path):
        os.makedirs(db_path)

    # 2. Initialize the Persistent Client
    client = chromadb.PersistentClient(path=db_path)

    # 3. Setup the Embedding Function
    # It automatically looks for 'OPENAI_API_KEY' in os.environ
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.environ.get("OPENAI_API_KEY"),
        model_name="text-embedding-3-large"
    )

    # 4. Create or Get the Collection
    collection = client.get_or_create_collection(
        name="technical_docs_collection",
        embedding_function=openai_ef,
        metadata={"hnsw:space": "cosine"}
    )

    # 5. Upsert the Data
    print(f"Saving {len(ingestion_package['documents'])} chunks to {db_path}...")
    
    collection.upsert(
        documents=ingestion_package["documents"],
        metadatas=ingestion_package["metadatas"],
        ids=ingestion_package["ids"]
    )

    print(f"Success! ChromaDB is persistent in {db_path}")
    return collection

def process_and_save_full_pipeline(source_path: str, source_name: str):
    """
    Orchestrates the full RAG ingestion pipeline by calling sub-functions.
    """
    # 1. Step 1: Parse, Chunk, and Enrich (Returns the ingestion package)
    print(f"--- Phase 1: Processing {source_name} ---")
    ingestion_package = prepare_source_for_chroma(source_path, source_name)

    # 2. Step 2: Save to ChromaDB (Persists to D:/doc_processing)
    print(f"--- Phase 2: Storing in ChromaDB ---")
    collection = save_chunks_to_chroma(ingestion_package, source_name)

    print(f"--- Pipeline Complete: {source_name} is ready for retrieval ---")

# --- Usage ---
if __name__ == "__main__":
    process_and_save_full_pipeline("https://arxiv.org/pdf/2408.09869", "arxiv_paper")