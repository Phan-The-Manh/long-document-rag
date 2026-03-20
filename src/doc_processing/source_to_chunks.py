from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker import HybridChunker
from typing import Dict, Any, Tuple, List
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions
from docling.datamodel.base_models import InputFormat
import time
import pypdfium2 as pdfium
import os

def convert_source_to_chunks(source: str, total_pages: int = 121, window_size: int = 10, overlap: int = 2):
    # Enforce 1 thread to prevent memory spikes on your 20-thread CPU
    os.environ["DOCLING_NUM_THREADS"] = "1"
    start_time = time.perf_counter()

    # 1. Minimalist Pipeline
    pipeline_options = PdfPipelineOptions(
        do_ocr=True,
        do_table_structure=True,
        generate_page_images=False,
        generate_picture_images=False,
    )

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )
    chunker = HybridChunker(max_tokens=512, merge_peers=True)

    all_chunks = []

    # 2. String Slicing: Trigger Window only for PDFs
    if source[-4:].lower() == ".pdf":
        print(f"📄 PDF Detected. Processing {total_pages} pages in slices...")
        
        current_start = 1
        step = window_size - overlap

        while current_start <= total_pages:
            current_end = min(current_start + window_size - 1, total_pages)
            print(f"--- Window: {current_start} to {current_end} ---")
            
            # Convert only this specific range
            result = converter.convert(source, page_range=(current_start, current_end))
            all_chunks.extend(list(chunker.chunk(result.document)))
            
            if current_end == total_pages:
                break
            
            current_start += step
            
    else:
        # For Word, PPT, etc. (where total_pages is ignored by Docling)
        print(f"📝 Non-PDF ({source[-5:]}) detected. Processing in one shot...")
        result = converter.convert(source)
        all_chunks = list(chunker.chunk(result.document))

    duration = time.perf_counter() - start_time
    print(f"✅ Created {len(all_chunks)} chunks in {duration:.2f} seconds.")
    
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

def prepare_source_for_chroma(source_path: str, source_name: str) -> Dict[str, List[Any]]:
    """
    Orchestrates the full pipeline:
    1. Parses & Chunks the file (Docling)
    2. Enriches text with Breadcrumbs/Page context
    3. Formats data for direct ChromaDB ingestion
    """
    # 1. Get raw chunks from Docling
    raw_chunks = convert_source_to_chunks(source_path)
    
    # 2. Prepare containers for Chroma
    enriched_documents = []
    metadatas = []
    ids = []

    print(f"Enriching {len(raw_chunks)} chunks...")

    # 3. Process each chunk
    for i, chunk in enumerate(raw_chunks):
        # Use your enrichment function
        enriched_text, metadata = build_enriched_chunk_and_metadata(
            chunk, 
            source_name=source_name, 
            chunk_index=i
        )
        
        enriched_documents.append(enriched_text)
        metadatas.append(metadata)
        ids.append(metadata["chunk_id"])

    print(f"Successfully prepared {source_name} for ingestion.")

    return {
        "documents": enriched_documents,
        "metadatas": metadatas,
        "ids": ids
    }
