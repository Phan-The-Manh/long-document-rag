import asyncio
import os
import ast
import uuid
import warnings
import logging
import pandas as pd
from dotenv import load_dotenv

# --- 1. LOGGING & WARNING CONFIGURATION ---
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("ragas").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# --- 2. IMPORT CUSTOM CLIENTS & AGENT ---
from src.llm_clients.openai import async_client as openai_async_client
from src.agent.graph import app  # Your LangGraph application
from langchain_core.messages import HumanMessage

# --- 3. RAGAS CORE IMPORTS ---
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import Faithfulness, AnswerRelevancy, AnswerCorrectness
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings as LC_OpenAIEmbeddings

load_dotenv()

# --- DYNAMIC PATH SETUP ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
INPUT_CSV_PATH = os.path.join(PROJECT_ROOT, "data", "eval_dataset_store", "testset_output.csv")
OUTPUT_CSV_PATH = os.path.join(PROJECT_ROOT, "data", "eval_dataset_store", "agent_generation_results.csv")

async def calculate_generation_metrics(eval_dataset: EvaluationDataset) -> pd.DataFrame:
    print("\n⚖️ Initializing Ragas Generation Graders...")
    
    # Use the LangChain objects which Ragas supports natively
    eval_llm = ChatOpenAI(model="gpt-4o-mini")
    eval_embeddings = LC_OpenAIEmbeddings(model="text-embedding-3-small")

    # Wrap them for Ragas
    evaluator_llm = LangchainLLMWrapper(eval_llm)
    evaluator_embeddings = LangchainEmbeddingsWrapper(eval_embeddings)

    metrics = [
        Faithfulness(),      
        AnswerRelevancy(),   
        AnswerCorrectness()  
    ]

    print("🚀 Starting Ragas Evaluation Loop (Async)...")
    
    results = evaluate(
        dataset=eval_dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        max_workers=10
    )

    return results.to_pandas()

async def run_agent_generation_evaluation():
    """
    Orchestrator: Runs the Agent, packages samples, and triggers evaluation.
    """
    print("🤖 Starting LangGraph Agent Generation Evaluation...")

    # Use the dynamic paths defined above
    input_csv = INPUT_CSV_PATH
    output_csv = OUTPUT_CSV_PATH
    
    if not os.path.exists(input_csv):
        print(f"❌ Error: Cannot find dataset at {input_csv}")
        return
        
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    df = pd.read_csv(input_csv)
    samples = []
    
    print(f"📊 Testing {len(df)} queries against the LangGraph Agent...")

    # --- PHASE 1: EXECUTE AGENT ---
    for index, row in df.iterrows():
        user_input = row['user_input']
        reference_answer = row['reference']
        
        # Safely handle reference contexts
        raw_ref_ctx = row.get('reference_contexts', "[]")
        try:
            reference_contexts = ast.literal_eval(raw_ref_ctx) if isinstance(raw_ref_ctx, str) else raw_ref_ctx
        except:
            reference_contexts = [str(raw_ref_ctx)]

        print(f"[{index + 1}/{len(df)}] Agent Thinking: {user_input[:50]}...")
        
        try:
            # Use a fresh Thread ID to prevent conversation leakage
            config = {"configurable": {"thread_id": f"eval_task_{uuid.uuid4()}"}}
            
            # Invoke LangGraph Agent
            final_state = app.invoke(
                {"messages": [HumanMessage(content=user_input)]}, 
                config=config
            )
            
            # Extract content from last AI Message
            response_text = final_state["messages"][-1].content
            
            # Extract docs
            retrieved_docs = final_state.get("retrieved_docs", []) 
            retrieved_contexts = [
                doc.page_content if hasattr(doc, 'page_content') else str(doc) 
                for doc in retrieved_docs
            ]

            # Build the Ragas Sample
            sample = SingleTurnSample(
                user_input=user_input,
                response=response_text,
                retrieved_contexts=retrieved_contexts,
                reference=reference_answer,
                reference_contexts=reference_contexts
            )
            samples.append(sample)

        except Exception as e:
            print(f"⚠️ Error processing query {index}: {e}")
            continue

    if not samples:
        print("🚫 No samples were successfully generated. Ending run.")
        return

    # --- PHASE 2: CALCULATE METRICS ---
    eval_dataset = EvaluationDataset(samples=samples)
    results_df = await calculate_generation_metrics(eval_dataset)

    # --- PHASE 3: EXPORT & REPORT ---
    results_df.to_csv(output_csv, index=False)
    
    print("\n" + "="*40)
    print("✅ EVALUATION COMPLETE")
    print("="*40)
    
    metrics_to_print = ['faithfulness', 'answer_relevancy', 'answer_correctness']
    for metric in metrics_to_print:
        if metric in results_df.columns:
            mean_score = results_df[metric].mean()
            print(f"{metric.replace('_', ' ').title():<20}: {mean_score:.4f} / 1.0000")
    
    print("="*40)
    print(f"📄 Detailed report saved to: {output_csv}")

if __name__ == "__main__":
    asyncio.run(run_agent_generation_evaluation())