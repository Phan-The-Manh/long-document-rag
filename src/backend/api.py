import sys
import json
import time
from contextlib import asynccontextmanager
from typing import Optional

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from starlette.concurrency import run_in_threadpool

from langchain_core.messages import HumanMessage, AIMessageChunk
from src.agent.graph import app as graph_app
from src.doc_processing.source_to_chroma import (
    process_and_save_full_pipeline,
    delete_document,
    list_documents,
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    docs = await run_in_threadpool(list_documents)
    if docs:
        latest = docs[-1]
        app.state.ingested = True
        app.state.ingested_source = latest["source"]
        print(f"[STARTUP] Restored ingested state: {len(docs)} document(s) found in ChromaDB.")
    else:
        app.state.ingested = False
        app.state.ingested_source = None
    yield

app = FastAPI(title="Local Chatbot Backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MODELS ---

class IngestRequest(BaseModel):
    doc_link: str
    source_name: Optional[str] = "user_upload"

class ChatRequest(BaseModel):
    message: str
    session_id: str

class IngestAndChatRequest(BaseModel):
    doc_link: str
    source_name: Optional[str] = "user_upload"
    message: str
    session_id: str

# --- HELPERS ---

def _evt(type_: str, msg: str = "") -> dict:
    return {"event": "message", "data": json.dumps({"type": type_, "msg": msg})}

# --- ROUTES ---

@app.get("/status")
async def status():
    return {"ingested": app.state.ingested, "source_name": app.state.ingested_source}

@app.post("/reset")
async def reset():
    app.state.ingested = False
    app.state.ingested_source = None
    return {"status": "ok"}

@app.get("/documents")
async def documents(tenant_id: Optional[str] = None):
    docs = await run_in_threadpool(list_documents, tenant_id)
    return {"documents": docs}


@app.delete("/documents/{doc_id}")
async def delete_doc(doc_id: str, version: str = "v1", tenant_id: str = "default"):
    removed = await run_in_threadpool(delete_document, doc_id, version, tenant_id)
    if removed == 0:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"status": "deleted", "removed_chunks": removed}


@app.post("/ingest")
async def ingest(payload: IngestRequest):
    """Accepts JSON body: {"doc_link": "URL", "source_name": "NAME"}"""
    try:
        start_time = time.time()
        await run_in_threadpool(
            process_and_save_full_pipeline,
            payload.doc_link,
            payload.source_name
        )
        app.state.ingested = True
        app.state.ingested_source = payload.source_name
        duration = round(time.time() - start_time, 2)
        return {"status": "success", "message": "Document processed", "duration": f"{duration}s"}
    except Exception as e:
        print(f"Ingestion Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat(payload: ChatRequest):
    if not app.state.ingested:
        raise HTTPException(status_code=400, detail="No document ingested yet.")
    if not payload.message:
        raise HTTPException(status_code=400, detail="Message is required")

    config = {"configurable": {"thread_id": payload.session_id}}
    inputs = {"messages": [HumanMessage(content=payload.message)]}

    async def event_generator():
        try:
            async for chunk, metadata in graph_app.astream(
                inputs, config=config, stream_mode="messages"
            ):
                if (isinstance(chunk, AIMessageChunk)
                        and chunk.content
                        and metadata.get("langgraph_node") == "generator"):
                    yield _evt("token", chunk.content)
            yield _evt("done")
        except Exception as e:
            print(f"[ERROR] Chat Graph failed: {e}")
            yield _evt("error", str(e))

    return EventSourceResponse(event_generator())

@app.post("/ingest-and-chat")
async def ingest_and_chat(payload: IngestAndChatRequest):
    async def event_generator():
        # Phase 1 — Ingest
        yield _evt("progress", "Downloading and processing document...")
        try:
            await run_in_threadpool(
                process_and_save_full_pipeline,
                payload.doc_link,
                payload.source_name
            )
            app.state.ingested = True
            app.state.ingested_source = payload.source_name
        except Exception as e:
            print(f"[ERROR] Ingest failed: {e}")
            yield _evt("error", str(e))
            return

        # Phase 2 — Stream answer tokens
        yield _evt("progress", "Document ready. Generating answer...")
        config = {"configurable": {"thread_id": payload.session_id}}
        inputs = {"messages": [HumanMessage(content=payload.message)]}
        try:
            async for chunk, metadata in graph_app.astream(
                inputs, config=config, stream_mode="messages"
            ):
                if (isinstance(chunk, AIMessageChunk)
                        and chunk.content
                        and metadata.get("langgraph_node") == "generator"):
                    yield _evt("token", chunk.content)
            yield _evt("done")
        except Exception as e:
            print(f"[ERROR] Generation failed: {e}")
            yield _evt("error", str(e))

    return EventSourceResponse(event_generator())
