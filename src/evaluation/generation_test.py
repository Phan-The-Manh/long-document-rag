import os
import pandas as pd
import asyncio
from dotenv import load_dotenv

# Ragas & LangChain Imports
from ragas import evaluate, EvaluationDataset
from ragas.metrics import Faithfulness, AnswerRelevancy, AnswerCorrectness
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

# --- 1. SETUP PATHS ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
INPUT_CSV = os.path.join(PROJECT_ROOT, "data", "eval_dataset_store", "answer_generation.csv")
OUTPUT_CSV = os.path.join(PROJECT_ROOT, "data", "eval_dataset_store", "final_eval_results.csv")

import ast # Make sure this is imported at the top

async def main():
    if not os.path.exists(INPUT_CSV):
        print(f"❌ Error: {INPUT_CSV} not found!")
        return

    print(f"📂 Loading generated answers from: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)

    # --- THE CRITICAL FIX ---
    # Convert string-represented lists back into actual Python lists
    for col in ['retrieved_contexts', 'reference_contexts']:
        if col in df.columns:
            print(f"🔧 Converting {col} from string to list...")
            df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    # ------------------------

    # 1. RENAME THE COLUMN DIRECTLY
    if 'generated_answer' in df.columns:
        df = df.rename(columns={'generated_answer': 'response'})
    
    # 2. CREATE DATASET (Now it has the 'response' column Ragas wants)
    dataset = EvaluationDataset.from_pandas(df)

    # --- 3. INITIALIZE EVALUATORS ---
    eval_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
    eval_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))

    metrics = [
        Faithfulness(),
        AnswerRelevancy(),
        AnswerCorrectness()
    ]

    # --- 4. RUN EVALUATION ---
    print("⚖️ Starting Evaluation (Using stable settings)...")
    
    # Using the safer config we discussed to avoid timeouts
    config = RunConfig(max_workers=4, timeout=180)

    results = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=eval_llm,
        embeddings=eval_embeddings,
        run_config=config
        # column_map is no longer needed because we renamed it above
    )

    # --- 5. SAVE & DISPLAY ---
    results_df = results.to_pandas()
    results_df.to_csv(OUTPUT_CSV, index=False)
    
    print("\n" + "="*40)
    print("✅ EVALUATION COMPLETE")
    print(f"📊 Results saved to: {OUTPUT_CSV}")
    print("="*40)
    print(results)

if __name__ == "__main__":
    asyncio.run(main())