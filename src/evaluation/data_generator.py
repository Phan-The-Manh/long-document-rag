import asyncio
import os
import typing as t
from dataclasses import dataclass

import pandas as pd
from dotenv import load_dotenv

# --- IMPORT YOUR CUSTOM CLIENT ---
from src.llm_clients.openai import client as openai_client

# MODERN RAGAS CORE IMPORTS
from ragas import EvaluationDataset, SingleTurnSample, MultiTurnSample
from ragas.llms import llm_factory
from ragas.testset.graph import KnowledgeGraph
from ragas.testset.persona import Persona
from ragas.testset.synthesizers.multi_hop.base import MultiHopQuerySynthesizer, MultiHopScenario
from ragas.testset.synthesizers.prompts import ThemesPersonasInput, ThemesPersonasMatchingPrompt
from ragas.testset.synthesizers.single_hop import SingleHopQuerySynthesizer, SingleHopScenario

# Load environment variables
load_dotenv()

# --- DYNAMIC PATH SETUP ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
KG_STORE_DIR = os.path.join(PROJECT_ROOT, "data", "chunks_store")
EVAL_STORE_DIR = os.path.join(PROJECT_ROOT, "data", "eval_dataset_store")

# Verification (Optional)
# print(f"Targeting global data folder at: {PROJECT_ROOT}/data")

# ==========================================
# 1. CUSTOM SYNTHESIZERS
# ==========================================

@dataclass
class MyMultiHopQuery(MultiHopQuerySynthesizer):
    theme_persona_matching_prompt = ThemesPersonasMatchingPrompt()

    async def _generate_scenarios(
        self, n: int, knowledge_graph, persona_list, callbacks
    ) -> t.List[MultiHopScenario]:

        results = knowledge_graph.find_two_nodes_single_rel(
            relationship_condition=lambda rel: rel.type == "entity_overlap_similarity"
        )

        if not results:
            print("No Multi-Hop relationships found. Check your KG relationship types.")
            return []

        num_sample_per_triplet = max(1, n // len(results))
        scenarios = []

        for triplet in results:
            if len(scenarios) >= n:
                break
            
            node_a, rel, node_b = triplet[0], triplet[1], triplet[2]
            overlapped_keywords = rel.properties.get("overlapped_items", [])
            
            if overlapped_keywords:
                try:
                    if isinstance(overlapped_keywords[0], (list, tuple)):
                        themes = list(dict(overlapped_keywords).keys())
                    else:
                        themes = list(overlapped_keywords)
                except Exception:
                    themes = list(overlapped_keywords)

                prompt_input = ThemesPersonasInput(themes=themes, personas=persona_list)
                
                persona_concepts_output = await self.theme_persona_matching_prompt.generate(
                    data=prompt_input, llm=self.llm, callbacks=callbacks
                )

                # THE FIX: 100% Positional Arguments (Immune to naming changes)
                base_scenarios = self.prepare_combinations(
                    [node_a, node_b],                   # Arg 1: nodes
                    [themes],                           # Arg 2: terms/themes
                    persona_list,                       # Arg 3: personas
                    persona_concepts_output.mapping,    # Arg 4: persona_concepts
                    "entities"                          # Arg 5: property_name
                )

                sampled_scenarios = self.sample_diverse_combinations(
                    base_scenarios, num_sample_per_triplet
                )
                scenarios.extend(sampled_scenarios)

        return scenarios[:n]


@dataclass
class MySingleHopQuerySynthesizer(SingleHopQuerySynthesizer):
    theme_persona_matching_prompt = ThemesPersonasMatchingPrompt()

    async def _generate_scenarios(
        self, n: int, knowledge_graph, persona_list, callbacks
    ) -> t.List[SingleHopScenario]:
        
        nodes = [node for node in knowledge_graph.nodes if node.properties.get("entities")]
        if not nodes:
            print("No Single-Hop nodes found with the 'entities' property.")
            return []
        
        scenarios = []
        num_sample_per_node = max(1, n // len(nodes))

        for node in nodes:
            if len(scenarios) >= n: 
                break
            
            themes = node.properties.get("entities", [])
            if not themes:
                continue

            prompt_input = ThemesPersonasInput(themes=themes, personas=persona_list)
            persona_concepts_output = await self.theme_persona_matching_prompt.generate(
                data=prompt_input, llm=self.llm, callbacks=callbacks
            )
            
            # THE FIX: 100% Positional Arguments
            base_scenarios = self.prepare_combinations(
                node,                               # Arg 1: node
                themes,                             # Arg 2: terms/themes
                persona_list,                       # Arg 3: personas
                persona_concepts_output.mapping     # Arg 4: persona_concepts
            )
            
            scenarios.extend(self.sample_combinations(base_scenarios, num_sample_per_node))
            
        return scenarios[:n]

# ==========================================
# 2. GENERATION LOGIC
# ==========================================

async def generate_ragas_dataset(kg, personas, llm, n_total=20):
    m_hop = MyMultiHopQuery(llm=llm)
    s_hop = MySingleHopQuerySynthesizer(llm=llm)

    print(f"Phase 1: Planning {n_total} scenarios...")
    m_scenarios = await m_hop._generate_scenarios(n_total // 2, kg, personas, [])
    s_scenarios = await s_hop._generate_scenarios(n_total // 2, kg, personas, [])
    all_scenarios = m_scenarios + s_scenarios

    print(f"Phase 2: Writing Q&A pairs for {len(all_scenarios)} scenarios...")
    tasks = [
        (m_hop if isinstance(s, MultiHopScenario) else s_hop)._generate_sample(s, callbacks=[])
        for s in all_scenarios
    ]
    
    results = await asyncio.gather(*tasks)
    
    valid_samples = []
    for r in results:
        # Extract the core sample out of the Ragas TestsetSample wrapper
        if isinstance(r, (SingleTurnSample, MultiTurnSample)):
            valid_samples.append(r)
        elif hasattr(r, 'eval_sample'):
            valid_samples.append(r.eval_sample)
            
    return EvaluationDataset(samples=valid_samples)

# ==========================================
# 3. RUNTIME & STORAGE
# ==========================================

async def run_generation(my_kg):
    # --- USE CENTRALIZED CLIENT ---
    llm = llm_factory("gpt-4o-mini", client=openai_client)

    # Define Personas
    personas = [
        Persona(name="Junior Developer", role_description="Needs simple 'how-to' guides and definitions."),
        Persona(name="Solutions Architect", role_description="Asks about system scalability and deep dependencies."),
        Persona(
            name="Quantitative Auditor", 
            role_description="Performs cross-page data validation, trend analysis (CAGR), and multi-step calculations. Does not accept surface-level answers; requires raw numbers and the logic used to derive the result."
        ),
        Persona(
            name="Project Manager", 
            role_description="Asks for high-level summaries, key milestones, and 'who is responsible for what' without getting into technical weeds."
        ),
        Persona(
            name="Equity Research Analyst", 
            role_description="Focuses on precise numeric trends, year-over-year (YoY) comparisons, specific fiscal risks, and footnotes in financial tables."
        )
    ]

    # Setup directories securely using dynamic path
    os.makedirs(EVAL_STORE_DIR, exist_ok=True)
    output_file = os.path.join(EVAL_STORE_DIR, "testset_output.csv")

    # Generate Dataset
    dataset = await generate_ragas_dataset(my_kg, personas, llm, n_total=20)

    # Export
    df = dataset.to_pandas()
    df.to_csv(output_file, index=False)
    
    print(f"\nSuccess! Saved {len(df)} questions to: {output_file}")
    return df

async def generate_test_set():
    # Use dynamic path for loading the KG
    kg_path = os.path.join(KG_STORE_DIR, "user_upload_enriched_kg.json")
    
    if not os.path.exists(kg_path):
        print(f"Error: Could not find the KG file at {kg_path}")
        return

    print(f"Loading enriched Knowledge Graph from JSON...")
    try:
        my_kg = KnowledgeGraph.load(kg_path)
        print(f"KG Loaded: {len(my_kg.nodes)} nodes and {len(my_kg.relationships)} relationships found.")
    except Exception as e:
        print(f"Failed to parse JSON KG: {e}")
        return

    await run_generation(my_kg)

if __name__ == "__main__":
    asyncio.run(generate_test_set())