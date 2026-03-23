# Long-Doc QA: Technical Implementation

An AI system engineered for high-precision Q&A on documents exceeding 100 pages using structured parsing and agentic retrieval.

---

## 1. Project Overview

### 🎯 The Challenge
Building a system for 100+ page documents that is:
* **Accurate:** Directly derived from the source.
* **Grounded:** Every answer includes page and section citations.
* **Robust:** Handles dense text, tables and narrative repetition.
* **Hallucination-Resistant:** Strict retrieval boundaries to prevent "creative" AI gaps.

### 🛠️ Core Strategy

To ensure a robust, grounded, and hallucination-resistant system for 100+ page documents, I implemented the following high-level design:

* **Advanced Parsing:** Used **Docling** to transform complex content (scanned PDFs, multi-page tables, and images) into structured data, preserving the document's original hierarchy.
* **Hybrid Chunking:** Applied **Hierarchical Chunking** with strict token limitations to maintain logical boundaries, enriched with "Contextual DNA" (metadata for section, page number, and table coordinates).
* **Storage & Hybrid Search:** Leveraged **ChromaDB** for lightweight local storage, utilizing a **Search Triad** (Semantic + Keyword + Metadata) to capture both conceptual and exact-match data.
* **Precision Re-ranking:** Integrated **FlashRank** to distill initial retrievals into the top 5 high-signal contexts, significantly reducing noise for the LLM.
* **Stateful Agent Logic:** Built with **LangGraph** to create a stateful agent capable of iterative reasoning, query refinement, and self-correction if initial evidence is insufficient.
* **Automated Evaluation:** Utilized **RAGAS** to quantitatively measure **Faithfulness**, **Answer Relevance**, and **Context Precision**, **Context Recall** across the long-form document.
---
## 2. System Design & Architecture

The system is architected as a **Stateful RAG Agent** optimized for high-precision retrieval from dense, structural documents.

### 📄 Ingestion & Parsing
Instead of standard PDF parsers that lose formatting, I utilized **Docling** to transform the document into a structured representation.
* **Multimodal Support:** Robustly handles scanned PDFs, complex tables, and embedded images.
* **Memory-Optimized Parsing:** Implemented a **Sliding Page Window** (10-page range with a 2-page overlap). This ensures that structural elements (like headers or tables) that bridge across page boundaries are not "severed," maintaining context continuity while keeping the memory footprint low.

### ✂️ Chunking
To solve the "lost context" problem, the system uses a **Hybrid Hierarchical Chunker**:
* **Structural Awareness:** Chunks are aligned with document sections and page numbers rather than arbitrary character counts.
* **Metadata Enrichment:** Every chunk is "enriched" with its parent section header and page number. When the LLM reads a chunk, it knows exactly its location within the 100+ page document.
* **Token Optimization:** Chunks are constrained to a specific token limit to ensure the final context window doesn't become oversaturated with noise.

### 🧠 Embedding & Storage
* **Vector Store:** **ChromaDB** was selected for its lightweight, local-first footprint and its robust support for **Metadata Filtering**.
* **Local Efficiency:** Chroma provides fast semantic searching and filtering even with long documents in a lightweight setup, making it ideal for rapid development and testing without cloud overhead.

### 🔍 Retrieval Strategy
The system employs a multi-stage retrieval pipeline to maximize "groundedness":
* **Query Transformation:** The system rewrites the user's natural language query into a **Search Triad** (Semantic, Keyword, and Metadata filters) to capture both conceptual and exact-match data.
* **Factor-Based Searching:** The retriever searches across all factors first; it includes **fallback logic** where it relaxes constraints (e.g., broadening the metadata filter) if the primary search returns empty.
* **Precision Re-ranking:** We initially retrieve up to 15 potential contexts but use **FlashRank** to select the **Top 5** most relevant snippets. This ensures the LLM receives only high-signal data.

### 🤖 Agentic Logic
The system is orchestrated using **LangGraph** to manage state and complex decision-making:
* **Stateful Iteration:** Using a directed graph, the agent can maintain the state of the search and "loop back" to refine its query if the initial retrieval is insufficient.
* **Self-Correction:** The agent evaluates the retrieved evidence against the user's question before generating a response, significantly reducing the risk of hallucination in dense, 100-page contexts.

3. System Architecture
   Use a clear flow (or a simple ASCII diagram) to show how data moves:

4. Key Design Decisions & Trade-offs

5. Implementation Details (The "How-To")

6. Evaluation & Testing

7. Assumptions & Limitations
   
8. Future Roadmap (The "If I had more time" section)
