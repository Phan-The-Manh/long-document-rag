# CLAUDE.md

## 🛠 Common Development Commands
- **Install dependencies**: `pip install -r requirements.txt`
- **Run application**: `python main.py`
- **Retrieval Evaluation**: `python -m src.evaluation.retrieval_eval`
- **Generation Evaluation (Ragas)**: `python -m src.evaluation.generation_test`
- **Knowledge Graph Builder**: `python -m src.evaluation.kg_builder`
- **Synthetic Data Generation**: `python -m src.evaluation.data_generator`
- **Run Tests**: `python -m pytest`

## 🏗 High-Level Architecture
- **State Machine**: Uses LangGraph `StateGraph` in `src/agent/graph.py` to orchestrate nodes.
- **Router Node (`src/agent/nodes/router.py`)**: Uses LLM-structured output to decide between document search or general chat.
- **Retriever Node (`src/agent/nodes/retriever.py`)**: Executes hybrid retrieval via `src/doc_processing/hybrid_retrieval_pipeline.py`.
- **Generator Node (`src/agent/nodes/generator.py`)**: Processes context to produce final responses with citations.
- **Ingestion Pipeline (`src/doc_processing/source_to_chroma.py`)**: Wipes and re-indexes ChromaDB store in `data/chroma_store`.

## ⚖️ Behavioral Guidelines

### 1. Think & Surface Tradeoffs
- **State Assumptions**: Before modifying a LangGraph node, explicitly state how it affects the `State` schema.
- **Simplify**: If a RAG improvement can be done via prompt engineering instead of a new code module, suggest it first.

### 2. Surgical Changes & Style
- **Touch Only What You Must**: Do not refactor `src/evaluation/` scripts unless specifically requested.
- **Match Style**: Follow the pattern of converting raw dicts into LangChain `Document` objects.

### 3. Verification Protocol (The "Single-Pass" Rule)
- **Don't Over-Evaluate**: Do NOT run the full scripts in `src/evaluation/` for minor logic updates; they are too slow and expensive.
- **Live Data Flow Test**: After any change to the nodes or pipeline, you MUST verify the fix by running `python main.py` for a single query. 
- **Success Criteria**:
  1. The data flow completes from `router` → `retriever` → `generator` without errors.
  2. The generated result is manually inspected for logical correctness and citation presence.
- **Final Evaluation**: Only run the full `generation_test` or `retrieval_eval` when specifically asked to "benchmark" the system or after completing a major feature.