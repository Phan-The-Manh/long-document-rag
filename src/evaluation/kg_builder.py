import json
import os
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.transforms import HeadlinesExtractor, apply_transforms
from ragas.testset.transforms.extractors import NERExtractor
from ragas.testset.transforms.relationship_builders.traditional import JaccardSimilarityBuilder
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper
from dotenv import load_dotenv
load_dotenv()

def build_kg_from_chunks_path(chunks_path: str, llm_wrapper) -> KnowledgeGraph:
    """
    Manual workflow: 
    1. Extract -> 2. Clean Data (Manual Loop) -> 3. Build Links
    """
    if not os.path.exists(chunks_path):
        raise FileNotFoundError(f"Could not find the chunks file at: {chunks_path}")

    with open(chunks_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"📂 Loaded {len(data['documents'])} chunks.")
    kg = KnowledgeGraph()

    # Create the nodes
    for text, meta in zip(data["documents"], data["metadatas"]):
        node = Node(
            type=NodeType.DOCUMENT,
            properties={
                "page_content": text, 
                "document_metadata": meta,
            }
        )
        kg.nodes.append(node)

    # 1. Setup components
    def node_filter(node):
        return len(node.properties.get("page_content", "")) > 300 

    headline_ext = HeadlinesExtractor(llm=llm_wrapper, filter_nodes=node_filter)
    ner_ext = NERExtractor(llm=llm_wrapper, filter_nodes=node_filter)
    
    rel_builder = JaccardSimilarityBuilder(
        property_name="entities", 
        new_property_name="entity_overlap_similarity",
        threshold=0.1
    )

    # --- STEP 1: RUN EXTRACTORS ---
    print("🧠 Step 1: Extracting Headlines and Entities...")
    apply_transforms(kg, transforms=[headline_ext, ner_ext])

    # --- STEP 2: MANUAL DATA CLEANUP ---
    # We fix the 'entities' format right here before the builder looks at it
    print("🧹 Step 2: Cleaning up entity data...")
    for node in kg.nodes:
        entities = node.properties.get("entities", {})
        
        # If NER gave us a dict like {"GPT-4": "MODEL"}, convert to ["GPT-4"]
        if isinstance(entities, dict):
            node.properties["entities"] = list(entities.keys())
        
        # Ensure it's never None or missing
        if not node.properties.get("entities"):
            node.properties["entities"] = []

    # --- STEP 3: BUILD RELATIONSHIPS ---
    print("🔗 Step 3: Building similarity links...")
    apply_transforms(kg, transforms=[rel_builder])

# --- STEP 4: ENRICH RELATIONSHIPS ---
    print("\n🔍 Step 4: Enriching relationships with shared entities...")
    
    enriched_count = 0
    total_rels = len(kg.relationships)

    for rel in kg.relationships:
        # 1. Access the source and target nodes directly
        # Some versions of Ragas store IDs, others store the Node objects.
        # This check handles both.
        src = rel.source if hasattr(rel.source, 'properties') else node_map.get(rel.source)
        tgt = rel.target if hasattr(rel.target, 'properties') else node_map.get(rel.target)
        
        if src and tgt:
            # 2. Extract entities
            ents_a = set(src.properties.get("entities", []))
            ents_b = set(tgt.properties.get("entities", []))
            
            # 3. Find intersection
            overlap = list(ents_a.intersection(ents_b))
            
            if overlap:
                # 4. Inject into relationship properties
                rel.properties["overlapped_items"] = overlap
                enriched_count += 1
        else:
            # This shouldn't happen now, but good for debugging
            src_id = getattr(rel.source, 'id', rel.source)
            tgt_id = getattr(rel.target, 'id', rel.target)
            print(f"❌ Still missing link for: {src_id} -> {tgt_id}")

    print(f"✨ Successfully enriched {enriched_count} / {total_rels} relationships.")
    
    # --- STEP 5: THE "FORCE SAVE" (If kg.save() fails you) ---
    kg_output_path = chunks_path.replace(".json", "_kg.json")
    
    # Check if the properties actually exist before saving
    if enriched_count > 0:
        # Try standard save first
        kg.save(kg_output_path)
        
        # Verify the save by reading it back immediately
        with open(kg_output_path, 'r') as f:
            saved_data = json.load(f)
            # Check the first relationship in the file
            if "relationships" in saved_data and len(saved_data["relationships"]) > 0:
                first_rel_props = saved_data["relationships"][0].get("properties", {})
                if "overlapped_items" in first_rel_props:
                    print(f"💾 Verification Success: 'overlapped_items' found in {kg_output_path}")
                else:
                    print("🚨 ALERT: Ragas .save() is stripping your custom properties! Use manual json.dump.")
    else:
        print("🚫 No overlaps were found, so nothing was added to the file.")


if __name__ == "__main__":
    eval_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    evaluator_llm = LangchainLLMWrapper(eval_llm)

    CHUNKS_PATH = "D:/long_doc_agent/data_store/chunks_store/user_upload_enriched.json"
    
    # Run the builder
    build_kg_from_chunks_path(CHUNKS_PATH, evaluator_llm)