import json
import shutil
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker import HybridChunker
from typing import Dict, Any, Tuple, List
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from pypdf import PdfReader
import time
import os
import requests

# --- Dynamic Path Logic ---
# Sets the base directory to the folder where this script resides
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "data", "uploaded_file")
STORE_DIR = os.path.join(BASE_DIR, "data", "chunks_store")

def ensure_local_path(source: str, target_dir: str = UPLOAD_DIR):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)
        
    local_destination = os.path.join(target_dir, "user_upload.pdf")

    # Force delete existing file to prevent metadata "ghosting"
    if os.path.exists(local_destination):
        os.remove(local_destination)

    if source.startswith(("http://", "https://")):
        # ... (keep your existing URL download logic here)
        return local_destination
    else:
        clean_source = source.strip().strip('"').strip("'")
        if os.path.exists(clean_source):
            print(f"📂 Fresh Copy: {clean_source} -> {local_destination}")
            shutil.copy2(clean_source, local_destination)
            return local_destination
        else:
            raise FileNotFoundError(f"❌ File not found at: {clean_source}")
        
def parse_pdf_to_docs(source_path: str, window_size: int = 10, overlap: int = 2):
    """
    Step 1: Converts a PDF into a list of Docling Document objects 
    using a sliding window to manage memory.
    """
    pipeline_options = PdfPipelineOptions(
        do_ocr=True,
        do_table_structure=True,
    )
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )

    # Get page count
    try:
        reader = PdfReader(source_path)
        total_pages = len(reader.pages)
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return [converter.convert(source_path).document]

    documents = []
    current_start = 1
    step = window_size - overlap

    while current_start <= total_pages:
        current_end = min(current_start + window_size - 1, total_pages)
        print(f"--- Parsing Window: {current_start} to {current_end} ---")
        
        result = converter.convert(source_path, page_range=(current_start, current_end))
        documents.append(result.document)
        
        if current_end == total_pages:
            break
        current_start += step

    return documents

def chunk_documents(documents: list, max_tokens: int = 512):
    """
    Step 2: Takes a list of Docling Documents and breaks them 
    into individual text chunks.
    """
    # merge_peers=False keeps chunks small for 512-token context windows
    chunker = HybridChunker(max_tokens=max_tokens, merge_peers=False)
    
    all_chunks = []
    for doc in documents:
        chunks = list(chunker.chunk(doc))
        all_chunks.extend(chunks)
        
    print(f"Extraction complete. Created {len(all_chunks)} total chunks.")
    return all_chunks

def build_enriched_chunk_and_metadata(chunk, source_name: str, chunk_index: int) -> Tuple[str, Dict[str, Any]]:
    """
    Prepends context to the chunk text and extracts retrieval metadata.
    Returns: (new_chunk_text, metadata_dict)
    """
    # 1. Extract and format Page Numbers
    pages_list = sorted({
        prov.page_no
        for item in getattr(chunk.meta, "doc_items", [])
        for prov in getattr(item, "prov", [])
        if getattr(prov, "page_no", None) is not None
    })
    pages_str = ", ".join(map(str, pages_list)) if pages_list else "Unknown"
    first_page = pages_list[0] if pages_list else 0

    # 2. Handle Heading Hierarchy (Breadcrumbs)
    headings_list = getattr(chunk.meta, "headings", [])
    breadcrumb = " > ".join(headings_list) if headings_list else "General"

    # 3. Create the New Chunk Text (Prepend Logic)
    new_chunk_text = (
        f"SECTION: {breadcrumb}\n"
        f"PAGES: {pages_str}\n"
        f"CONTENT: {chunk.text}"
    )

    # 4. Final Metadata for Chroma
    metadata = {
        "source": source_name,
        "chunk_id": f"{source_name}_chunk_{chunk_index}",
        "first_page": first_page,      # Useful for 'Sort by Page'
        "pages_label": pages_str,     # Useful for 'Display in UI'
        "section_path": breadcrumb,    # Useful for 'Filter by Section'
    }

    return new_chunk_text, metadata

def prepare_source_for_chroma(input_source: str, source_name: str) -> Dict[str, List[Any]]:
    """
    Orchestrates the full pipeline.
    """
    # 1. Resolve Path
    try:
        source_path = ensure_local_path(input_source, target_dir=UPLOAD_DIR)
    except Exception as e:
        print(f"Critical Error resolving source: {e}")
        return {}

    # 2. Setup storage
    os.makedirs(STORE_DIR, exist_ok=True)
    
    # 3. Step 1: Parse PDF
    print(f"Starting PDF Conversion for: {source_name}")
    doc_objects = parse_pdf_to_docs(source_path, window_size=10, overlap=2)
    
    # 4. Step 2: Chunk
    print(f"Chunking documents...")
    raw_chunks = chunk_documents(doc_objects, max_tokens=480)
    
    # 5. Prepare containers
    enriched_documents = []
    metadatas = []
    ids = []

    print(f"✨ Enriching {len(raw_chunks)} total chunks...")

    # 6. Process and Enrich
    for i, chunk in enumerate(raw_chunks):
        enriched_text, metadata = build_enriched_chunk_and_metadata(
            chunk, 
            source_name=source_name, 
            chunk_index=i
        )
        
        enriched_documents.append(enriched_text)
        metadatas.append(metadata)
        ids.append(metadata["chunk_id"])

    # 7. Save as JSON
    output_data = {
        "documents": enriched_documents,
        "metadatas": metadatas,
        "ids": ids
    }

    save_path = os.path.join(STORE_DIR, f"{source_name}_enriched.json")
    
    try:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)
        print(f"Successfully prepared {source_name}. Total chunks: {len(enriched_documents)}")
        print(f"Saved to: {save_path}")
    except Exception as e:
        print(f"Error saving JSON: {e}")

    return output_data