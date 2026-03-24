import pandas as pd
import asyncio
import ast
import os
from dotenv import load_dotenv

# Ragas & Langchain Imports
from ragas import EvaluationDataset, SingleTurnSample, evaluate, RunConfig
from ragas.metrics import Faithfulness, AnswerRelevancy, AnswerCorrectness
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings as LC_OpenAIEmbeddings

# Load environment variables (API Keys)
load_dotenv()

# Gets the directory of the current script
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
# Builds the path (much cleaner syntax!)
RESULTS_FILE_PATH = os.path.join(PROJECT_ROOT, "data", "eval_dataset_store", "testset_output.csv")
async def run_grading_only():
    # 1. Load the CSV using the dynamic path
    file_path = RESULTS_FILE_PATH
    
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found at {file_path}")
        return

    df = pd.read_csv(file_path)
    print(f"📂 Loaded {len(df)} samples from {file_path}. Starting Grading Phase...")

    # 2. Convert DataFrame rows back into Ragas Samples
    samples = []
    for _, row in df.iterrows():
        # Handle string-to-list conversion for retrieved contexts
        try:
            retrieved_ctx = ast.literal_eval(row['retrieved_contexts']) if isinstance(row['retrieved_contexts'], str) else row['retrieved_contexts']
        except (ValueError, SyntaxError):
            retrieved_ctx = [row['retrieved_contexts']]
        
        samples.append(SingleTurnSample(
            user_input=row['user_input'],
            response=row['response'],
            retrieved_contexts=retrieved_ctx,
            reference=row['reference']
        ))

    eval_dataset = EvaluationDataset(samples=samples)

    # 3. Setup Grader Models with Langchain Wrappers
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
    evaluator_embeddings = LangchainEmbeddingsWrapper(LC_OpenAIEmbeddings(model="text-embedding-3-small"))

    # 4. Run the Evaluation with Backpressure (max_workers) and higher Timeout
    run_config = RunConfig(max_workers=4, timeout=120) 
    
    print("🚀 Running Ragas Grader (LLM-as-a-judge)...")
    results = evaluate(
        dataset=eval_dataset,
        metrics=[Faithfulness(), AnswerRelevancy(), AnswerCorrectness()],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        run_config=run_config
    )

    # 5. Save and PRINT the results
    results_df = results.to_pandas()
    results_df.to_csv(file_path, index=False)
    
    print("\n" + "="*50)
    print("✅ GRADING COMPLETE")
    print("="*50)
    
    # Calculate and print mean scores
    for metric_name, score in results.items():
        print(f"{metric_name:20}: {score:.4f} / 1.0000")
    
    print("="*50)
    print(f"📄 Detailed results saved back to: {file_path}")

if __name__ == "__main__":
    asyncio.run(run_grading_only())