import json
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Ragas imports
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.transforms import HeadlinesExtractor, apply_transforms
from ragas.testset.transforms.extractors import NERExtractor
from ragas.testset.transforms.relationship_builders.traditional import JaccardSimilarityBuilder
from ragas.llms import LangchainLLMWrapper

load_dotenv()

# --- 1. CORE UTILITIES ---

def load_chunks(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find chunks file at: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_node_map(kg: KnowledgeGraph):
    """Creates a lookup table for nodes by their ID."""
    return {node.id: node for node in kg.nodes}

# --- 2. TRANSFORMS & CLEANING ---

def clean_node_entities(kg: KnowledgeGraph):
    """Standardizes entity format and handles LLM output inconsistencies."""
    print("Cleaning up entity data...")
    for node in kg.nodes:
        entities = node.properties.get("entities", [])
        
        # Convert dict {"Entity": "Type"} to ["Entity"]
        if isinstance(entities, dict):
            entities = list(entities.keys())
        
        # Ensure list and strip whitespace
        if not entities:
            node.properties["entities"] = []
        else:
            node.properties["entities"] = [str(e).strip() for e in entities]

def resolve_synonyms(kg: KnowledgeGraph, llm_wrapper):
    """Groups similar entities (e.g., 'USA' and 'United States') using the LLM."""
    all_entities = set()
    for node in kg.nodes:
        all_entities.update(node.properties.get("entities", []))
    
    if not all_entities:
        return

    print(f"Resolving synonyms for {len(all_entities)} entities...")
    prompt = (
        f"Extract synonyms from this list: {list(all_entities)}. "
        "Return a JSON object where the key is the synonym and the value is the master name."
    )
    
    try:
        # Note: adjust method call based on your specific Ragas/Langchain wrapper version
        mapping = llm_wrapper.generate_json(prompt) 
        if mapping:
            for node in kg.nodes:
                ents = node.properties.get("entities", [])
                # Map to master name and deduplicate
                resolved = {mapping.get(e, e) for e in ents}
                node.properties["entities"] = list(resolved)
    except Exception as e:
        print(f"Synonym resolution skipped: {e}")

# --- 3. RELATIONSHIPS & ENRICHMENT ---

def enrich_relationships(kg: KnowledgeGraph):
    """Manually adds overlapped entity lists to relationships."""
    print("Enriching relationships with shared entities...")
    node_map = get_node_map(kg)
    enriched_count = 0

    for rel in kg.relationships:
        # Handle cases where Ragas uses ID strings vs Node objects
        src = rel.source if hasattr(rel.source, 'properties') else node_map.get(rel.source)
        tgt = rel.target if hasattr(rel.target, 'properties') else node_map.get(rel.target)
        
        if src and tgt:
            ents_a = set(src.properties.get("entities", []))
            ents_b = set(tgt.properties.get("entities", []))
            overlap = list(ents_a.intersection(ents_b))
            
            if overlap:
                rel.properties["overlapped_items"] = overlap
                enriched_count += 1
    
    print(f"Enriched {enriched_count}/{len(kg.relationships)} relationships.")
    return enriched_count

# --- 4. PERSISTENCE ---

def save_and_verify_kg(kg: KnowledgeGraph, output_path: str, expected_count: int):
    """Saves the KG and checks if custom properties survived serialization."""
    # Ensure Ragas saves as UTF-8
    kg.save(output_path)
    
    if expected_count > 0:
        # CRITICAL FIX: Add encoding="utf-8" here
        with open(output_path, 'r', encoding="utf-8") as f:
            try:
                saved_data = json.load(f)
                rels = saved_data.get("relationships", [])
                if rels and "overlapped_items" in rels[0].get("properties", {}):
                    print(f"Success: Knowledge Graph saved to {output_path}")
                else:
                    print("ALERT: Ragas .save() stripped your custom properties!")
            except json.JSONDecodeError:
                print("ALERT: Saved file is not valid JSON!")
# --- 5. MAIN WORKFLOW ---

def build_kg_from_chunks_path(chunks_path: str, llm_wrapper):
    data = load_chunks(chunks_path)
    kg = KnowledgeGraph()

    # Step 1: Initialize Nodes
    for text, meta in zip(data["documents"], data["metadatas"]):
        node = Node(
            type=NodeType.DOCUMENT,
            properties={"page_content": text, "document_metadata": meta}
        )
        kg.nodes.append(node)

    # Step 2: Extraction
    node_filter = lambda n: len(n.properties.get("page_content", "")) > 300
    transforms = [
        HeadlinesExtractor(llm=llm_wrapper, filter_nodes=node_filter),
        NERExtractor(llm=llm_wrapper, filter_nodes=node_filter)
    ]
    
    print("Running Extractors...")
    apply_transforms(kg, transforms=transforms)

    # Step 3: Cleanup & Synonyms
    clean_node_entities(kg)
    resolve_synonyms(kg, llm_wrapper)

    # Step 4: Building Relationships
    print("Building similarity links...")
    rel_builder = JaccardSimilarityBuilder(
        property_name="entities", 
        new_property_name="entity_overlap_similarity",
        threshold=0.1
    )
    apply_transforms(kg, transforms=[rel_builder])

    # Step 5: Enrichment & Export
    enriched_count = enrich_relationships(kg)
    output_path = chunks_path.replace(".json", "_kg.json")
    save_and_verify_kg(kg, output_path, enriched_count)

if __name__ == "__main__":
    eval_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    evaluator_wrapper = LangchainLLMWrapper(eval_llm)

    CHUNKS_PATH = "D:/long_doc_agent/data/chunks_store/user_upload_enriched.json"
    build_kg_from_chunks_path(CHUNKS_PATH, evaluator_wrapper)