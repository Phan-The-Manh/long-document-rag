import os
import json
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions
from flashrank import Ranker, RerankRequest
from dotenv import load_dotenv

# Load the .env file containing OPENAI_API_KEY
load_dotenv()

# --- Dynamic Path Logic ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
CHROMA_DB_PATH = os.path.join(PROJECT_ROOT, "data", "chroma_store")
RERANK_CACHE_DIR = os.path.join(PROJECT_ROOT, "data", "rerank_cache")
UPLOAD_DIR = os.path.join(PROJECT_ROOT, "data", "uploaded_file")
STORE_DIR = os.path.join(PROJECT_ROOT, "data", "chunks_store")

def rewrite_query_triad(raw_query: str):
    """
    Refined Triad: Transforms raw query into Semantic, Keyword, and Metadata.
    Specifically parses for Page Numbers and Section Headers.
    """
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    prompt = f"""
    You are a technical search expert. Categorize the user's question into three search formats.
    
    1. SEMANTIC: A formal, descriptive sentence for vector similarity search (ONLY using keywords in the question, just add filling word and rephrase).
    2. KEYWORD: A list of unique technical terms, part numbers, or error codes.
    3. METADATA: A JSON object containing:
        - "section": A likely header name (e.g., 'Maintenance', 'Troubleshooting').
        - "pages": A LIST of integers for every page number mentioned. Empty list if none.
    
    User Question: "{raw_query}"
    
    Output ONLY valid JSON:
    {{
        "semantic": "string",
        "keyword": "string",
        "metadata": {{
            "section": "string or null",
            "pages": [int] 
        }}
    }}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        response_format={"type": "json_object"}
    )

    query_triad = json.loads(response.choices[0].message.content)
    return query_triad

def stage_1_hybrid_retriever(triad: dict, top_n: int = 15):
    """
    Tiered Fallback Search:
    Tier 1: Semantic + Keyword + Metadata (The "Golden" Search)
    Tier 2: Semantic + Metadata (Keyword dropped)
    Tier 3: Semantic Only (Pure Vector Fallback)
    """
    # Use the dynamic path
    db_path = CHROMA_DB_PATH
    client = chromadb.PersistentClient(path=db_path)
    
    # 1. Setup Embedding Function
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.environ.get("OPENAI_API_KEY"),
        model_name="text-embedding-3-large"
    )
    
    collection = client.get_collection(
        name="technical_docs_collection", 
        embedding_function=openai_ef
    )

    # 2. Prepare and Sanitize Inputs
    semantic_q = triad.get('semantic', "")
    
    keyword_list = triad.get('keyword', "").replace(',', ' ').split()
    or_filters = [{"$contains": word} for word in keyword_list]
    
    where_doc = None
    if len(or_filters) > 1:
        where_doc = {"$or": or_filters}
    elif len(or_filters) == 1:
        where_doc = or_filters[0]

    section = triad['metadata'].get('section')
    if str(section).lower() in ["null", "none", ""]: section = None
    
    pages = triad['metadata'].get('pages', [])

    # 3. Build Metadata Filter Object
    filters = []
    if pages:
        filters.append({"first_page": {"$in": pages}})
    if section:
        filters.append({"section_path": {"$contains": section}})

    where_meta = None
    if len(filters) > 1:
        where_meta = {"$and": filters}
    elif len(filters) == 1:
        where_meta = filters[0]

    # 4. Define Search Execution Logic
    def execute_query(use_meta=True, use_keyword=True):
        return collection.query(
            query_texts=[semantic_q],
            n_results=top_n,
            where=where_meta if use_meta else None,
            where_document=where_doc if use_keyword else None, 
            include=["documents", "metadatas"]
        )

    # --- THE FALLBACK LOOP ---
    print("DEBUG: Tier 1 - Full Hybrid (Semantic + Meta + Keyword)")
    results = execute_query(use_meta=True, use_keyword=True)

    if not results["documents"] or not results["documents"][0]:
        print("DEBUG: Tier 2 - Technical Focus (Dropping Metadata, Keeping Keyword)")
        results = execute_query(use_meta=False, use_keyword=True)

    if not results["documents"] or not results["documents"][0]:
        print("DEBUG: Tier 3 - Location Focus (Dropping Keyword, Keeping Metadata)")
        results = execute_query(use_meta=True, use_keyword=False)

    if not results["documents"] or not results["documents"][0]:
        print("DEBUG: Tier 4 - Pure Semantic Fallback")
        results = execute_query(use_meta=False, use_keyword=False)

    # 5. Final Assembly of Candidates
    candidates = []
    if results["documents"] and results["documents"][0]:
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            candidates.append({
                "id": meta["chunk_id"], 
                "text": doc, 
                "meta": meta
            })
    
    print(f"DEBUG: Successfully retrieved {len(candidates)} candidates.")
    return candidates

# Initialize the Ranker globally with a dynamic cache path.
ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir=RERANK_CACHE_DIR)

def stage_2_reranker(query: str, candidates: list, final_k: int = 5):
    """
    Refines the Stage 1 results by re-sorting them based on exact relevancy.
    """
    if not candidates:
        print("No candidates found in Stage 1 to rerank.")
        return []

    rerank_request = RerankRequest(query=query, passages=candidates)
    reranked_results = ranker.rerank(rerank_request)

    # Filter by a Confidence Threshold
    top_results = [res for res in reranked_results if res['score'] > 0.05]

    return top_results[:final_k]

def query_retrieval(user_input: str):
    # 0. Rewrite
    triad = rewrite_query_triad(user_input)
    print("Query Triad:", triad)
    # 1. Retrieve (Hybrid)
    candidates = stage_1_hybrid_retriever(triad)
    print(f"Retrieved {len(candidates)} candidates from Stage 1.")
    # 2. Rerank (Precision)
    final_context = stage_2_reranker(triad['semantic'], candidates)
    print(f"Reranked to {len(final_context)} final candidates.")
    return final_context

# --- Usage ---
if __name__ == "__main__":
    result = query_retrieval("what is ocr?")
    if result:
        print("Final RAG Candidates:", result[0])