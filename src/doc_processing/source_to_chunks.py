from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker import HybridChunker
from typing import Dict, Any, Tuple, List
from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions
from docling.datamodel.base_models import InputFormat

def convert_source_to_chunks(source: str):
    """
    Converts a source document into a list of chunks using a HybridChunker.
    """

    # 1. Setup Pipeline Options
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True # OCR runs only where needed
    
    # 2. Use RapidOCR with ONNX (Best for Intel CPUs/iGPUs)
    # Ensure you have 'rapidocr_onnxruntime' installed
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )
    
    print(f"Converting document from {source}...")
    result = converter.convert(source)
    doc = result.document
    print("Conversion complete!")

    # 3. Initialize the chunker (adjust max_tokens as needed)
    chunker = HybridChunker(max_tokens=512)
    
    # 4. Generate chunks and return as a list
    print("Generating chunks...")
    chunks = list(chunker.chunk(doc))
    print(f"Created {len(chunks)} chunks.")
    
    return chunks


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
