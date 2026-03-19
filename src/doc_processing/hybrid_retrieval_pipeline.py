import os
import json
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions
from flashrank import Ranker, RerankRequest

from dotenv import load_dotenv
load_dotenv()

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
    db_path = "D:/long_doc_agent/data_store/chroma_store"
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
    
    # Sanitize Keyword: Use only the first word to avoid phrase-match failure
    keyword_list = triad.get('keyword', "").replace(',', ' ').split()
    or_filters = [{"$contains": word} for word in keyword_list]
    
    # Sanitize Metadata
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
            where_document={"$or": or_filters} if (use_keyword and or_filters) else None,
            include=["documents", "metadatas"]
        )

    # --- THE FALLBACK LOOP ---
    
    # ATTEMPT 1: The "Everything" Search
    print("DEBUG: Tier 1 - Full Hybrid (Semantic + Meta + Keyword)")
    results = execute_query(use_meta=True, use_keyword=True)

    # ATTEMPT 2: The "Technical Content" Search (Semantic + Keyword)
    # This ignores the Page/Section but keeps the strict keyword requirement.
    if not results["documents"] or not results["documents"][0]:
        print("DEBUG: Tier 2 - Technical Focus (Dropping Metadata, Keeping Keyword)")
        results = execute_query(use_meta=False, use_keyword=True)

    # ATTEMPT 3: The "Location" Search (Semantic + Metadata)
    # Keeps the page/section but allows for fuzzy semantic matching.
    if not results["documents"] or not results["documents"][0]:
        print("DEBUG: Tier 3 - Location Focus (Dropping Keyword, Keeping Metadata)")
        results = execute_query(use_meta=True, use_keyword=False)

    # ATTEMPT 4: The "Emergency" Search (Semantic Only)
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

# Initialize the Ranker globally so it only loads into memory once.
# This uses a light but powerful model optimized for technical RAG.
ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="D:/long_doc_agent/data_store/rerank_cache")

def stage_2_reranker(query: str, candidates: list, final_k: int = 5):
    """
    Refines the Stage 1 results by re-sorting them based on exact relevancy.
    
    Args:
        query: The raw user question or the optimized semantic query.
        candidates: The list of dicts returned by Stage 1.
        final_k: How many top results to return to the LLM.
    """
    if not candidates:
        print("No candidates found in Stage 1 to rerank.")
        return []

    # 1. Prepare the Rerank Request
    # FlashRank expects a list of dicts with 'id', 'text', and 'meta'
    rerank_request = RerankRequest(query=query, passages=candidates)
    
    # 2. Execute Reranking
    # This compares the query + text together for every candidate
    reranked_results = ranker.rerank(rerank_request)

    # 3. Filter by a Confidence Threshold
    # In 2026, we don't just take the top results; we check if they are actually good.
    # Scores > 0.1 are generally relevant; scores > 0.5 are high confidence.
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
    # We use the 'semantic' query from the triad for the best reranking match
    final_context = stage_2_reranker(triad['semantic'], candidates)
    print(f"Reranked to {len(final_context)} final candidates.")
    return final_context

# result = query_retrieval("what is ocr?")
# print("Final RAG Candidates:", result[0])