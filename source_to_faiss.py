from source_to_chunks import source_to_langchain_docs
import os
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()

def save_langchain_docs_to_faiss(docs: List[Document], save_dir = "faiss_store"):
    """
    Takes a list of LangChain Documents, generates embeddings, 
    and saves them into a local FAISS vector store.
    """
    if not docs:
        print("⚠️ No documents provided. Skipping FAISS build.")
        return None

    # 1. Initialize Embeddings 
    # Ensure your OPENAI_API_KEY is in your environment variables
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 2. Build FAISS vector store
    print(f"Generating embeddings for {len(docs)} documents... (This may take a moment)")
    vectorstore = FAISS.from_documents(docs, embeddings)

    # 3. Save locally
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(save_dir)
    
    print(f"✅ Success! FAISS index saved to: {os.path.abspath(save_dir)}")

def source_to_vectorstore(source: str, save_dir: str = "faiss_store"):
    """
    Complete Pipeline:
    1. Converts PDF/URL to Docling chunks.
    2. Maps chunks to LangChain Document objects.
    3. Embeds documents and saves them to a FAISS vector store.
    """
    # Step 1 & 2: Convert source to LangChain Documents
    # (This uses the 'source_to_langchain_docs' function we defined earlier)
    print(f"\n--- Phase 1: Processing {source} ---")
    docs = source_to_langchain_docs(source)

    # Step 3: Embed and Save to FAISS
    # (This uses your 'save_langchain_docs_to_faiss' function)
    print(f"\n--- Phase 2: Building Vector Store in {save_dir} ---")
    vectorstore = save_langchain_docs_to_faiss(docs, save_dir)

    print("\n✅ Full Pipeline Complete!")
    return vectorstore

source_to_vectorstore("https://arxiv.org/pdf/2408.09869")