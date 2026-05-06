# CLAUDE.md

## 🏗 High-Level Architecture
- **State Machine**: Uses LangGraph `StateGraph` in `src/agent/graph.py` to orchestrate nodes.
- **Router Node (`src/agent/nodes/router.py`)**: Uses LLM-structured output to decide between document search or general chat.
- **Retriever Node (`src/agent/nodes/retriever.py`)**: Executes hybrid retrieval via `src/doc_processing/hybrid_retrieval_pipeline.py`.
- **Generator Node (`src/agent/nodes/generator.py`)**: Processes context to produce final responses with citations.
- **Ingestion Pipeline (`src/doc_processing/source_to_chroma.py`)**: Idempotent upsert into ChromaDB store in `data/chroma_store`. Exposes `process_and_save_full_pipeline`, `delete_document(doc_id, version, tenant_id)`, `list_documents(tenant_id)`, and `clear_chroma_database` (test/dev only).

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

## 🗄 Production Vector DB System Design (1,000+ docs × 100+ pages)

### Why the current store fails at scale
- **Single-node Chroma**: `chromadb.PersistentClient` in `src/doc_processing/source_to_chroma.py` cannot shard. At ~74 chunks per 10 pages, the 1,000-doc target is **~750K–1.5M chunks** (~30–60 GB indexed with HNSW + payload). *Still open.*
- **`$contains` keyword filter** in `hybrid_retrieval_pipeline.py` is a linear scan — fine at thousands, ruinous at millions. *Still open — next priority.*
- ~~**Random-suffix IDs** (`uuid.uuid4()[:4]`) break idempotent re-ingestion.~~ **Fixed**: deterministic content-hash IDs + `collection.upsert` in `source_to_chroma.py`.
- ~~**Single collection** — no per-doc delete, no versioning, no tenant isolation.~~ **Partially fixed**: still one collection, but `delete_document(doc_id)` / `list_documents()` work, and `tenant_id` + `version` live in the payload (logical isolation only — no server-side enforcement yet).

### Target storage tier
- **Vector store → Qdrant** (self-hosted) or Milvus at very high scale.
  - Shard by `tenant_id`; payload indexes on `doc_id`, `version`, `first_page`, `section_path`.
  - `on_disk: true` segments + scalar/product quantization → ~4× RAM reduction with < 2% recall loss.
  - Optional native sparse vectors (BM42 / SPLADE) for hybrid in one engine.
- **Lexical index → OpenSearch / Elasticsearch.** Replaces the `$contains` substring filter with BM25, stemming, synonyms, field boosts.
- **Canonical metadata → Postgres.** Source of truth for `doc_id`, `tenant_id`, `version`, `source_uri`, `sha256`, `ingestion_status`, `chunk_count`, `embedding_model`. Drives idempotency and audit.
- **Object storage → S3 / Azure Blob.** Holds raw PDFs and parsed Docling JSON. Enables cheap re-embed when models change (re-parse is the expensive step).
- **Cache → Redis.** Query-triad results and reranker scores keyed by `sha256(tenant_id | normalized_query | doc_filter)`, TTL ~1h.

### Schema & identity model *(implemented)*

Implemented in `build_enriched_chunk_and_metadata` (`src/doc_processing/source_to_chunks.py`):

```
doc_id   = sha256(file_bytes)[:16]                                    # at upload boundary
chunk_id = sha256(tenant_id | doc_id | version | chunk_index | content_hash)[:16]
payload  = {
  source,            # human-readable label (e.g. "user_upload")
  doc_id,            # canonical content-addressable identity
  tenant_id,         # default = "default" — server-side enforcement TODO
  version,           # default = "v1"
  chunk_id, chunk_index, content_hash,
  first_page, pages_label, section_path,
}
```

- **On-disk file**: `data/uploaded_file/{doc_id}.pdf` — same bytes never re-downloaded or re-written.
- **Sidecar JSON**: `data/chunks_store/{doc_id}_enriched.json`.
- **Idempotency**: same `(tenant_id, doc_id, version, chunk_index, content_hash)` → same `chunk_id` → `upsert` is a true no-op. Re-ingesting an unchanged PDF does not re-embed.
- **Known edge case**: same bytes ingested with two different `source_name` values → second call's label silently overwrites the first (last-write-wins on the row). `source_name` is **not** in the chunk_id seed by design (see "Identity vs label" trade-off below).
- **Not yet captured**: `embedding_model` field — needed before the embedding model is ever swapped.

### Multi-tenant isolation (pick one)
| Pattern | Isolation | When |
| :--- | :--- | :--- |
| Filter by `tenant_id` payload | Logical | Internal use, < 50 tenants |
| Collection-per-tenant *(default)* | Strong | SaaS, 50–500 tenants |
| Cluster-per-tenant | Hard | Regulated data (HIPAA, SOC 2 Type II) |

Tenant scoping is enforced **server-side** in the retrieval API, never trusted from the client.

### Async ingestion pipeline
Replace synchronous `process_and_save_full_pipeline()` with four idempotent, checkpointed stages (state in Postgres):
1. **Upload & register** → write PDF to S3, insert row `status=PENDING`, enqueue.
2. **Parse (Docling worker pool)** → sliding-window parse, write JSON to S3, `status=PARSED`. *Fixes the "no checkpointing" limitation: a crash at page 90 resumes from the last completed window.*
3. **Chunk + enrich** → reuse `build_enriched_chunk_and_metadata`; persist chunks to S3 + Postgres.
4. **Embed + index** → batched embed calls with content-hash cache; transactional dual-write to Qdrant + OpenSearch; `status=INDEXED`.

A pool of 8–16 parsers sustains ~10–15 docs/hour each → **1,000-doc backfill in ~6–10 hours**, dominated by Docling.

### Retrieval at scale
Preserve the current triad-rewrite → hybrid → rerank structure in `hybrid_retrieval_pipeline.py`; swap backends:
- Issue **dense (Qdrant) and BM25 (OpenSearch) queries in parallel**; fuse via Reciprocal Rank Fusion.
- Tenant filter injected at the API gateway, never optional.
- Move FlashRank into a **stateless gRPC reranker service** so the agent doesn't reload the model per request and the reranker scales independently.

### Versioning & deletion
- New version = new rows; old version stays queryable until new is fully indexed, then a background job flips `latest=true` and tombstones old vectors.
- Soft delete (`status=DELETED` filter) → nightly hard-delete compaction. Supports GDPR erasure with audit trail.

### Observability SLOs
| Signal | Tool | SLO |
| :--- | :--- | :--- |
| Ingestion lag (upload → indexed) | Postgres + Grafana | P95 < 30 min for 100-page doc |
| Retrieval latency (P95, no LLM) | Prometheus + OTel | < 400 ms |
| Recall@5 vs golden set | Nightly RAGAS job | ≥ 0.85 |
| Embedding-cache hit rate | Redis stats | ≥ 60% on re-ingests |
| Qdrant RAM utilization | Qdrant `/metrics` | < 70% sustained |

### Migration path

**Done (Steps 1–4):**
1. ✅ Deterministic content-hash `chunk_id` + `collection.upsert` (`source_to_chroma.py`).
2. ✅ `tenant_id` + `version` + `content_hash` on every chunk payload.
3. ✅ `delete_document(doc_id)` / `list_documents()` + `DELETE /documents/{doc_id}` and `GET /documents` endpoints in `src/backend/api.py`.
4. ✅ Content-addressable upload filenames: `data/uploaded_file/{doc_id}.pdf`. Two PDFs with the same `source_name` no longer overwrite each other.

**Next, in order of payoff:**

5. **Replace `$contains` keyword tier with proper BM25** (`hybrid_retrieval_pipeline.py:86-92`). The current substring filter is the dominant retrieval-latency risk past ~10K chunks. Two paths:
   - *Lite (no new infra)*: precompute a token-set per chunk and filter via `$in`/payload index — buys an order of magnitude before forcing OpenSearch.
   - *Real fix*: add OpenSearch / Elasticsearch alongside Chroma; issue dense + BM25 in parallel and fuse with RRF. Bigger lift, but matches the target architecture.

6. **Enforce `tenant_id` server-side in the API gateway.** Today the field exists in the payload but every request hardcodes `"default"`. Inject it from auth context in `src/backend/api.py` ingest/chat/list/delete endpoints, and pass it down into `query_retrieval` so the retriever filters by tenant. Required before any multi-user deployment.

7. **Decide the "same bytes, two labels" semantic.** Today it silently dedupes (last label wins). Either accept that and document it in the API, or add `source_name` to the `chunk_id` seed to make labels first-class (loses dedup savings — measure before choosing).

8. **Postgres registry behind a thin DAO** (CLAUDE.md old Step 3). Source of truth for `(tenant_id, doc_id, version, source_uri, sha256, status, chunk_count)`. Unlocks audit, soft-delete, version replacement workflow ("publish v2 of doc X without breaking live queries"), and reliable retries. Mirror-write alongside Chroma.

9. **Swap Chroma → Qdrant (single-node).** Only when Chroma's HNSW genuinely caps out (~50–100K chunks on a single box). Changes contained to `_get_collection()` in `source_to_chroma.py` and the `collection.query(...)` call in `hybrid_retrieval_pipeline.py`. Add `embedding_model` to the payload before this step so future model swaps are tracked.

10. **Async ingestion with queue + workers + Postgres-tracked stages** (Celery / Ray / K8s Jobs). Replaces synchronous `process_and_save_full_pipeline()`. Critical at >100-doc backfills; not before.

11. **Shard by tenant** once multi-tenant traffic justifies it (collection-per-tenant or cluster-per-tenant — see Multi-tenant isolation table).

> **State-schema impact (per Behavioral Guideline 1)**: Steps 5–11 don't change `AgentState`. Step 9 changes the retriever-node *backend* but preserves its return shape (`documents: List[Document]`), so `src/agent/nodes/retriever.py` stays untouched and the LangGraph flow is unaffected.

### Trade-offs
- **Complexity vs. scale**: Qdrant + OpenSearch + Postgres + Redis + S3 + queue is overkill below ~50 docs; it is the floor at 1,000+.
- **Cost vs. recall**: Quantization and Matryoshka dim-reduction (3072 → 1024) cut storage and latency materially with < 2% recall loss — measure on the existing 20-record golden set before committing (do **not** trigger full evaluation per the Single-Pass Rule).
- **Build vs. buy**: Managed (Qdrant Cloud, Pinecone, Weaviate Cloud) removes ops burden but limits payload-index flexibility. Self-hosted Qdrant on one 32-core / 128 GB box handles the full target with headroom.
- **Identity vs label**: `chunk_id` seed includes `doc_id` (content hash) but not `source_name`. Same bytes always dedupe to one row regardless of label — saves embedding cost, but the user-facing label of the second uploader silently wins. Acceptable for a single-tenant tool, must be revisited before multi-user.