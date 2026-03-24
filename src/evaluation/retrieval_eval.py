import asyncio
import os
import ast
import warnings
from dotenv import load_dotenv
import pandas as pd

# --- THE FIX: IMPORT YOUR CUSTOM CLIENT ---
from src.llm_clients.openai import client as openai_client

# Silence deprecation warnings 
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["HF_HOME"] = os.environ.get("HF_HOME", ".huggingface_cache")

# Ragas Imports
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import ContextPrecision, ContextRecall
from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings
from src.llm_clients.openai import async_client as openai_async_client
# Import YOUR full retrieval pipeline
from src.doc_processing.hybrid_retrieval_pipeline import query_retrieval

load_dotenv()

# --- DYNAMIC PATH SETUP ---
# Gets the directory where this script is saved
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Builds paths relative to the script location
INPUT_CSV = os.path.join(BASE_DIR, "data", "eval_dataset_store", "testset_output.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "data", "eval_dataset_store", "retrieval_evaluation_results.csv")

async def calculate_retrieval_metrics(eval_dataset: EvaluationDataset) -> pd.DataFrame:
    print("\nInitializing Ragas Grader Models...")
    
    # Use the ASYNC client
    evaluator_llm = llm_factory("gpt-4o-mini", client=openai_async_client)
    
    evaluator_embeddings = OpenAIEmbeddings(
        client=openai_async_client, # Use ASYNC client here
        model="text-embedding-3-small", 
    )

    metrics = [ContextPrecision(), ContextRecall()]

    # evaluate() will now run smoothly in the async loop
    results = evaluate(
        dataset=eval_dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings
    )

    return results.to_pandas()


# ==========================================
# FUNCTION 2: The Orchestrator
# ==========================================
async def run_retrieval_and_evaluate():
    """
    Loads the golden dataset, runs queries through your custom pipeline,
    and calls the evaluator.
    """
    print("Starting End-to-End Retrieval Pipeline Execution...")

    # 1. Load the Golden Dataset using dynamic paths
    input_csv = INPUT_CSV
    output_csv = OUTPUT_CSV
    
    if not os.path.exists(input_csv):
        print(f"Error: Cannot find dataset at {input_csv}")
        return
        
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    df = pd.read_csv(input_csv)
    samples = []
    
    print(f"Testing {len(df)} queries against your hybrid retriever...")

    # 2. Process each query
    for index, row in df.iterrows():
        user_input = row['user_input']
        reference_answer = row['reference']
        
        try:
            # Handle potential string-representation of lists in CSV
            ctx = row['reference_contexts']
            reference_contexts = ast.literal_eval(ctx) if isinstance(ctx, str) and ctx.startswith('[') else [ctx]
        except (ValueError, SyntaxError):
            reference_contexts = [row['reference_contexts']]

        print(f"Query {index + 1}/{len(df)}: {user_input}")
        
        # --- EXECUTE YOUR PIPELINE ---
        final_docs = query_retrieval(user_input)
        
        # Extract raw text safely
        retrieved_contexts = [
            doc.page_content if hasattr(doc, 'page_content') else str(doc) 
            for doc in final_docs
        ]

        # Package the sample
        sample = SingleTurnSample(
            user_input=user_input,
            reference=reference_answer,               
            reference_contexts=reference_contexts,    
            retrieved_contexts=retrieved_contexts     
        )
        samples.append(sample)

    # 3. Create Dataset and Evaluate
    eval_dataset = EvaluationDataset(samples=samples)
    results_df = await calculate_retrieval_metrics(eval_dataset)

    # 4. Export results
    results_df.to_csv(output_csv, index=False)
    
    print("\nEvaluation Complete!")
    print("================ FINAL SCORES ================")
    
    for metric in ['context_precision', 'context_recall']:
        if metric in results_df.columns:
            mean_score = results_df[metric].mean()
            print(f"{metric.replace('_', ' ').title()}: {mean_score:.4f} / 1.0000")
        
    print(f"\nDetailed results saved to: {output_csv}")


if __name__ == "__main__":
    # Run the async orchestrator
    asyncio.run(run_retrieval_and_evaluate())