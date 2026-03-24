import json
import shutil
import time
import os
import requests
from typing import Dict, Any, Tuple, List

# Docling & PDF imports
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker import HybridChunker
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from pypdf import PdfReader

# --- DYNAMIC PATH SETUP ---
# Gets the directory where this script is saved
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Navigates up two levels to reach the project root
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

# Define folder paths
UPLOAD_DIR = os.path.join(PROJECT_ROOT, "data", "uploaded_file")
STORE_DIR = os.path.join(PROJECT_ROOT, "data", "chunks_store")

def ensure_directories():
    """Ensures all required project folders exist."""
    for folder in [UPLOAD_DIR, STORE_DIR]:
        if not os.path.exists(folder):
            print(f"📁 Creating missing directory: {folder}")
            os.makedirs(folder, exist_ok=True)

def ensure_local_path(source: str, target_dir: str = UPLOAD_DIR):
    # CRITICAL FIX: Ensure target directory exists before file operations
    os.makedirs(target_dir, exist_ok=True)
        
    local_destination = os.path.join(target_dir, "user_upload.pdf")

    # Force delete existing file to prevent metadata "ghosting"
    if os.path.exists(local_destination):
        os.remove(local_destination)

    if source.startswith(("http://", "https://")):
        print(f"🌐 Downloading from URL: {source}")
        response = requests.get(source, timeout=30)
        response.raise_for_status()
        with open(local_destination, "wb") as f:
            f.write(response.content)
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
    pipeline_options = PdfPipelineOptions(
        do_ocr=True,
        do_table_structure=True,
    )
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )

    try:
        reader = PdfReader(source_path)
        total_pages = len(reader.pages)
    except Exception as e:
        print(f"⚠️ Error reading PDF metadata: {e}. Falling back to standard conversion.")
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
    chunker = HybridChunker(max_tokens=max_tokens, merge_peers=False)
    all_chunks = []
    for doc in documents:
        chunks = list(chunker.chunk(doc))
        all_chunks.extend(chunks)
        
    print(f"✅ Extraction complete. Created {len(all_chunks)} total chunks.")
    return all_chunks

def build_enriched_chunk_and_metadata(chunk, source_name: str, chunk_index: int) -> Tuple[str, Dict[str, Any]]:
    # 1. Extract Page Numbers
    pages_list = sorted({
        prov.page_no
        for item in getattr(chunk.meta, "doc_items", [])
        for prov in getattr(item, "prov", [])
        if getattr(prov, "page_no", None) is not None
    })
    pages_str = ", ".join(map(str, pages_list)) if pages_list else "Unknown"
    first_page = pages_list[0] if pages_list else 0

    # 2. Heading Hierarchy
    headings_list = getattr(chunk.meta, "headings", [])
    breadcrumb = " > ".join(headings_list) if headings_list else "General"

    # 3. Prepend Context
    new_chunk_text = (
        f"SECTION: {breadcrumb}\n"
        f"PAGES: {pages_str}\n"
        f"CONTENT: {chunk.text}"
    )

    # 4. Final Metadata
    metadata = {
        "source": source_name,
        "chunk_id": f"{source_name}_chunk_{chunk_index}",
        "first_page": first_page,
        "pages_label": pages_str,
        "section_path": breadcrumb,
    }

    return new_chunk_text, metadata

def prepare_source_for_chroma(input_source: str, source_name: str) -> Dict[str, List[Any]]:
    # 0. Safety Check: Ensure folders exist
    ensure_directories()

    # 1. Resolve Path
    try:
        source_path = ensure_local_path(input_source, target_dir=UPLOAD_DIR)
    except Exception as e:
        print(f"❌ Critical Error resolving source: {e}")
        return {}

    # 2. Parse PDF
    print(f"🚀 Starting PDF Conversion: {source_name}")
    doc_objects = parse_pdf_to_docs(source_path, window_size=10, overlap=2)
    
    # 3. Chunk
    raw_chunks = chunk_documents(doc_objects, max_tokens=480)
    
    enriched_documents = []
    metadatas = []
    ids = []

    print(f"✨ Enriching {len(raw_chunks)} chunks...")

    # 4. Enrich
    for i, chunk in enumerate(raw_chunks):
        enriched_text, metadata = build_enriched_chunk_and_metadata(
            chunk, 
            source_name=source_name, 
            chunk_index=i
        )
        enriched_documents.append(enriched_text)
        metadatas.append(metadata)
        ids.append(metadata["chunk_id"])

    # 5. Save Results
    output_data = {
        "documents": enriched_documents,
        "metadatas": metadatas,
        "ids": ids
    }

    save_path = os.path.join(STORE_DIR, f"{source_name}_enriched.json")
    
    try:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)
        print(f"🎉 Success! Total chunks: {len(enriched_documents)}")
        print(f"📂 Saved to: {save_path}")
    except Exception as e:
        print(f"❌ Error saving JSON: {e}")

    return output_data