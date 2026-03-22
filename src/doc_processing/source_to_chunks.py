import json

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker import HybridChunker
from typing import Dict, Any, Tuple, List
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from pypdf import PdfReader
import time
import os
import requests

def save_url_to_local(url: str, target_dir: str = r"D:\long_doc_agent\data\uploaded_file"):
    """
    Downloads a file from a URL and saves it as 'user_upload.pdf'.
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)
    
    local_path = os.path.join(target_dir, "user_upload.pdf")
    
    print(f"🌐 Downloading: {url}")
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    
    with open(local_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            
    return local_path

def convert_source_to_chunks(source_path: str, window_size: int = 10, overlap: int = 2):
    """
    source_path should be the LOCAL path
    """
    start_time = time.perf_counter()
    
    # 1. Setup Converter & Chunker
    # TIP: For 512-token models, use max_tokens=480 to avoid indexing errors
    pipeline_options = PdfPipelineOptions(
        do_ocr=True,
        do_table_structure=True,
        generate_page_images=False,
        generate_picture_images=False,
    )
    
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )
    
    # merge_peers=False helps prevent chunks from growing too large for the 512 limit
    chunker = HybridChunker(max_tokens=480, merge_peers=False)

    # 2. Get page length using pypdf from the LOCAL file
    total_pages = 0
    if os.path.exists(source_path) and source_path.lower().endswith(".pdf"):
        try:
            reader = PdfReader(source_path)
            total_pages = len(reader.pages)
        except Exception as e:
            print(f"pypdf error: {e}. Falling back to one-shot.")
    else:
        print(f"File not found or not a PDF at: {source_path}")

    all_chunks = []

    # 3. Execution Strategy
    if total_pages > 0:
        print(f"Sliding Window active: {total_pages} pages.")
        current_start = 1
        step = window_size - overlap

        while current_start <= total_pages:
            current_end = min(current_start + window_size - 1, total_pages)
            print(f"--- Window: {current_start} to {current_end} ---")
            
            # Docling processes only this specific slice
            result = converter.convert(source_path, page_range=(current_start, current_end))
            all_chunks.extend(list(chunker.chunk(result.document)))
            
            if current_end == total_pages:
                break
            current_start += step
    else:
        # One-shot fallback
        print(f"Processing in one shot...")
        result = converter.convert(source_path)
        all_chunks = list(chunker.chunk(result.document))

    duration = time.perf_counter() - start_time
    print(f"Completed: {len(all_chunks)} chunks in {duration:.2f}s.")
    return all_chunks

def build_enriched_chunk_and_metadata(chunk, source_name: str, chunk_index: int) -> Tuple[str, Dict[str, Any]]:
    """
    Prepends context to the chunk text and extracts retrieval metadata.
    Returns: (new_chunk_text, metadata_dict)
    """
    # 1. Extract and format Page Numbers
    # We convert the list to a comma-separated string because Chroma filters 
    # prefer primitive types (str, int, float) over lists.
    pages_list = sorted({
        prov.page_no
        for item in getattr(chunk.meta, "doc_items", [])
        for prov in getattr(item, "prov", [])
        if getattr(prov, "page_no", None) is not None
    })
    pages_str = ", ".join(map(str, pages_list)) if pages_list else "Unknown"
    first_page = pages_list[0] if pages_list else 0

    # 2. Handle Heading Hierarchy (Breadcrumbs)
    # Docling stores hierarchy in headings; we join them for a 'path'
    headings_list = getattr(chunk.meta, "headings", [])
    breadcrumb = " > ".join(headings_list) if headings_list else "General"

    # 3. Create the New Chunk Text (Prepend Logic)
    # This is what gets embedded. It ensures the 'context' is weighted by the model.
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

def prepare_source_for_chroma(url: str, source_name: str) -> Dict[str, List[Any]]:
    """
    Orchestrates the full pipeline: Download -> Chunk -> Enrich -> Save.
    """
    # 1. Download the file using your function
    # This ensures pypdf (inside convert_source_to_chunks) can calculate page length
    UPLOAD_DIR = r"D:\long_doc_agent\data\uploaded_file"
    try:
        source_path = save_url_to_local(url, target_dir=UPLOAD_DIR)
    except Exception as e:
        print(f"❌ Critical Error during download: {e}")
        return {}

    # 2. Setup storage for enriched results
    STORE_DIR = r"D:\long_doc_agent\data\chunks_store"
    os.makedirs(STORE_DIR, exist_ok=True)
    
    # 3. Get raw chunks from your Sliding Window logic
    # This now receives the local path 'D:\...\user_upload.pdf'
    raw_chunks = convert_source_to_chunks(source_path)
    
    # 4. Prepare containers for Chroma format
    enriched_documents = []
    metadatas = []
    ids = []

    print(f"✨ Enriching {len(raw_chunks)} chunks...")

    # 5. Process and Enrich
    for i, chunk in enumerate(raw_chunks):
        enriched_text, metadata = build_enriched_chunk_and_metadata(
            chunk, 
            source_name=source_name, 
            chunk_index=i
        )
        
        enriched_documents.append(enriched_text)
        metadatas.append(metadata)
        ids.append(metadata["chunk_id"])

    # 6. Save as JSON
    output_data = {
        "documents": enriched_documents,
        "metadatas": metadatas,
        "ids": ids
    }

    save_path = os.path.join(STORE_DIR, f"{source_name}_enriched.json")
    
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

    print(f"✅ Successfully prepared {source_name} and saved to: {save_path}")

    return output_data
