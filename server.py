"""
FastAPI Server and REST API for OKF Document Studio & CAG In-Memory Engine.
"""

import os
import re
import sys
import time
from pathlib import Path

# Force UTF-8 encoding on standard streams to avoid Windows cp1252 charmap errors
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
from pydantic import BaseModel, Field

import uvicorn
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Ensure cag-okf is in sys.path
sys.path.insert(0, str(Path(__file__).parent))

from cag_okf.models import OKFDocument, OKFMetadata
from cag_okf.okf_parser import OKFParser
from cag_okf.composer import OKFComposer
from cag_okf.fast_memory_cache import FastInMemoryCache
from cag_okf.engine import OKFCAGEngine


app = FastAPI(
    title="CAG-OKF Knowledge Studio & In-Memory Engine",
    description="High-performance in-memory cache and editor for Open Knowledge Format documents with Cache-Augmented Generation.",
    version="0.2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://127.0.0.1",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:3000",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Global in-memory cache and CAG engine instances
memory_cache = FastInMemoryCache()
engine: Optional[OKFCAGEngine] = None
bundle_dir_path = str(Path(__file__).parent / "examples" / "sample_okf_bundle")


def sanitize_doc_id(doc_id: str) -> str:
    """Validates and sanitizes a document identifier to prevent path traversal."""
    if not doc_id or ".." in doc_id or "/" in doc_id or "\\" in doc_id:
        raise HTTPException(
            status_code=400,
            detail="Invalid doc_id: Path traversal characters ('..', '/', '\\') are strictly forbidden."
        )
    sanitized = re.sub(r"[^a-zA-Z0-9_\-\.]", "", doc_id)
    if not sanitized:
        raise HTTPException(status_code=400, detail="Invalid doc_id: must contain valid alphanumeric characters.")
    return sanitized


def get_safe_bundle_path(doc_id: str, requested_path: Optional[str] = None) -> str:
    """Resolves and verifies that a target file path resides strictly inside the bundle directory."""
    base_dir = Path(bundle_dir_path).resolve()
    if requested_path:
        p = Path(requested_path).resolve()
        try:
            p.relative_to(base_dir)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Security violation: Requested file path escapes the bundle root directory."
            )
        return str(p)

    safe_id = sanitize_doc_id(doc_id)
    target = (base_dir / "rules" / f"{safe_id}.md").resolve()
    try:
        target.relative_to(base_dir)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Security violation: Target file path escapes the bundle root directory."
        )
    return str(target)


def get_engine() -> OKFCAGEngine:
    global engine
    if engine is None:
        print("[Server] Initializing OKFCAGEngine with SmolLM2-135M-Instruct...")
        engine = OKFCAGEngine(model_id="HuggingFaceTB/SmolLM2-135M-Instruct", lazy_load=False)
        memory_cache.set_tokenizer(engine.tokenizer)
    return engine


@app.on_event("startup")
def startup_event():
    """Preloads bundle documents and initial KV cache into RAM."""
    global bundle_dir_path
    print(f"[Server Startup] Loading OKF documents into RAM from {bundle_dir_path}...")
    memory_cache.load_from_directory(bundle_dir_path)

    try:
        eng = get_engine()
        # Compile initial in-memory KV cache
        t0 = time.perf_counter()
        docs = memory_cache.list_all()
        composer = OKFComposer()
        prefix = composer.compose_prefix(eng.bundle or OKFParser.parse_bundle(bundle_dir_path), selected_docs=docs)
        
        cache, prefix_ids, seq_len = eng.precompute_cache_from_prefix(prefix) if hasattr(eng, "precompute_cache_from_prefix") else (None, None, 0)
        # Using build_cache
        eng.load_bundle(bundle_dir_path)
        profile = eng.build_cache()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        
        memory_cache.set_kv_cache(
            cache=eng.active_cache,
            prefix_ids=eng.active_prefix_ids,
            profile=profile,
            duration_ms=elapsed_ms
        )
        print(f"[Server Startup] Initialized in-memory KV cache ({profile.total_tokens} tokens) in {elapsed_ms:.1f}ms.")
    except Exception as e:
        print(f"[Server Startup Warning] KV Cache pre-compilation deferred: {e}")


# --- Pydantic Schemas for API Requests ---

class DocumentPayload(BaseModel):
    id: Optional[str] = Field(default=None, max_length=128)
    type: str = Field(default="rule", max_length=64)
    title: str = Field(..., min_length=1, max_length=256)
    description: Optional[str] = Field(default=None, max_length=1024)
    category: Optional[str] = Field(default=None, max_length=128)
    target_services: List[str] = Field(default_factory=list)
    target_languages: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    status: str = Field(default="APPROVED", max_length=32)
    valid_until: Optional[str] = Field(default=None, max_length=32)
    approved_by: Optional[str] = Field(default=None, max_length=128)
    body: str = Field(..., max_length=1000000)
    persist_to_disk: bool = True
    auto_rebuild_cache: bool = False


class PatchDocumentPayload(BaseModel):
    title: Optional[str] = Field(default=None, max_length=256)
    type: Optional[str] = Field(default=None, max_length=64)
    status: Optional[str] = Field(default=None, max_length=32)
    valid_until: Optional[str] = Field(default=None, max_length=32)
    category: Optional[str] = Field(default=None, max_length=128)
    target_services: Optional[List[str]] = None
    target_languages: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    approved_by: Optional[str] = Field(default=None, max_length=128)
    body: Optional[str] = Field(default=None, max_length=1000000)
    persist_to_disk: bool = True
    auto_rebuild_cache: bool = False


class AppendDocumentPayload(BaseModel):
    heading: Optional[str] = Field(default="Exceção Aprovada", max_length=128)
    content: str = Field(..., min_length=1, max_length=100000)
    approved_by: Optional[str] = Field(default=None, max_length=128)
    persist_to_disk: bool = True
    auto_rebuild_cache: bool = False


class RawDocumentPayload(BaseModel):
    raw_content: str = Field(..., min_length=1, max_length=1000000)
    file_path: Optional[str] = Field(default=None, max_length=512)
    persist_to_disk: bool = True
    auto_rebuild_cache: bool = False


class QueryPayload(BaseModel):
    query: str = Field(..., min_length=1, max_length=4096)
    max_tokens: int = Field(default=40, ge=1, le=512)
    use_cache: bool = True
    services: Optional[List[str]] = None


class ContextRequestPayload(BaseModel):
    service: Optional[str] = None
    language: Optional[str] = None
    tags: List[str] = Field(default_factory=list)


class OpenAIChatMessage(BaseModel):
    role: str
    content: str


class OpenAIChatPayload(BaseModel):
    model: Optional[str] = "cag-okf"
    messages: List[OpenAIChatMessage]
    max_tokens: Optional[int] = Field(default=40, ge=1, le=512)
    temperature: Optional[float] = Field(default=0.3, ge=0.0, le=2.0)


# --- Document CRUD Endpoints (In-Memory $O(1)$) ---

@app.get("/api/documents")
def list_documents(
    service: Optional[str] = None,
    language: Optional[str] = None,
    tag: Optional[str] = None,
    status: Optional[str] = None,
    doc_type: Optional[str] = None,
    search: Optional[str] = None,
):
    """Lists cached documents with indexed filtering."""
    docs = memory_cache.filter_docs(
        service=service,
        language=language,
        tag=tag,
        status=status,
        doc_type=doc_type,
        search_query=search
    )
    return [
        {
            "id": d.metadata.id,
            "type": d.metadata.type,
            "title": d.metadata.title,
            "description": d.metadata.description,
            "status": d.metadata.status,
            "target_services": d.metadata.target_services,
            "target_languages": d.metadata.target_languages,
            "tags": d.metadata.tags,
            "token_count": d.token_count or 0,
            "content_hash": d.metadata.content_hash,
            "file_path": d.file_path,
        }
        for d in docs
    ]


@app.get("/api/documents/{doc_id}")
def get_document(doc_id: str):
    """Retrieves document body and frontmatter directly from RAM."""
    doc = memory_cache.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found in memory cache.")
    return {
        "metadata": doc.metadata.model_dump(),
        "body": doc.body,
        "raw_content": doc.raw_content,
        "token_count": doc.token_count or 0,
        "file_path": doc.file_path,
    }


@app.post("/api/documents")
@app.post("/v1/documents")
def create_document(payload: DocumentPayload, background_tasks: BackgroundTasks):
    """Creates a new document directly in memory cache."""
    raw_id = payload.id or f"WIKI-{int(time.time())}"
    doc_id = sanitize_doc_id(raw_id)
    yaml_lines = [
        "---",
        f"id: \"{doc_id}\"",
        f"type: \"{payload.type}\"",
        f"title: \"{payload.title}\"",
    ]
    if payload.description:
        yaml_lines.append(f"description: \"{payload.description}\"")
    if payload.category:
        yaml_lines.append(f"category: \"{payload.category}\"")
    if payload.target_services:
        yaml_lines.append(f"target_services: {payload.target_services}")
    if payload.target_languages:
        yaml_lines.append(f"target_languages: {payload.target_languages}")
    if payload.tags:
        yaml_lines.append(f"tags: {payload.tags}")
    if payload.status:
        yaml_lines.append(f"status: \"{payload.status}\"")
    if payload.valid_until:
        yaml_lines.append(f"valid_until: \"{payload.valid_until}\"")
    if payload.approved_by:
        yaml_lines.append(f"approved_by: \"{payload.approved_by}\"")
    yaml_lines.append("---\n")

    raw_content = "\n".join(yaml_lines) + payload.body

    file_path = None
    if payload.persist_to_disk:
        file_path = get_safe_bundle_path(doc_id)

    doc = OKFParser.parse_text(raw_content, file_path=file_path)
    stored = memory_cache.put(doc, persist_to_disk=payload.persist_to_disk, file_path=file_path)

    if payload.auto_rebuild_cache:
        background_tasks.add_task(rebuild_cache)

    return {
        "status": "created",
        "id": stored.metadata.id,
        "title": stored.metadata.title,
        "token_count": stored.token_count,
        "is_dirty": memory_cache.is_dirty()
    }


@app.post("/api/documents/raw")
@app.post("/v1/documents/raw")
def create_or_update_raw_document(payload: RawDocumentPayload, background_tasks: BackgroundTasks):
    """Parses raw Markdown + YAML frontmatter and writes to memory cache."""
    safe_path = get_safe_bundle_path("temp", payload.file_path) if payload.file_path else None
    doc = OKFParser.parse_text(payload.raw_content, file_path=safe_path)
    if not doc.metadata.id:
        doc.metadata.id = f"DOC-{int(time.time())}"
    doc.metadata.id = sanitize_doc_id(doc.metadata.id)

    target_path = safe_path
    if payload.persist_to_disk and not target_path:
        target_path = get_safe_bundle_path(doc.metadata.id)

    stored = memory_cache.put(doc, persist_to_disk=payload.persist_to_disk, file_path=target_path)

    if payload.auto_rebuild_cache:
        background_tasks.add_task(rebuild_cache)

    return {
        "status": "saved",
        "id": stored.metadata.id,
        "title": stored.metadata.title,
        "token_count": stored.token_count,
        "is_dirty": memory_cache.is_dirty()
    }


@app.put("/api/documents/{doc_id}")
@app.put("/v1/documents/{doc_id}")
def update_document(doc_id: str, payload: DocumentPayload, background_tasks: BackgroundTasks):
    """Full update of an existing document in RAM."""
    safe_id = sanitize_doc_id(doc_id)
    existing = memory_cache.get(safe_id)
    target_path = get_safe_bundle_path(safe_id, existing.file_path if existing else None)

    yaml_lines = [
        "---",
        f"id: \"{safe_id}\"",
        f"type: \"{payload.type}\"",
        f"title: \"{payload.title}\"",
    ]
    if payload.description:
        yaml_lines.append(f"description: \"{payload.description}\"")
    if payload.category:
        yaml_lines.append(f"category: \"{payload.category}\"")
    if payload.target_services:
        yaml_lines.append(f"target_services: {payload.target_services}")
    if payload.target_languages:
        yaml_lines.append(f"target_languages: {payload.target_languages}")
    if payload.tags:
        yaml_lines.append(f"tags: {payload.tags}")
    if payload.status:
        yaml_lines.append(f"status: \"{payload.status}\"")
    if payload.valid_until:
        yaml_lines.append(f"valid_until: \"{payload.valid_until}\"")
    if payload.approved_by:
        yaml_lines.append(f"approved_by: \"{payload.approved_by}\"")
    yaml_lines.append("---\n")

    raw_content = "\n".join(yaml_lines) + payload.body
    doc = OKFParser.parse_text(raw_content, file_path=target_path)
    stored = memory_cache.put(doc, persist_to_disk=payload.persist_to_disk, file_path=target_path)

    if payload.auto_rebuild_cache:
        background_tasks.add_task(rebuild_cache)

    return {
        "status": "updated",
        "id": stored.metadata.id,
        "token_count": stored.token_count,
        "is_dirty": memory_cache.is_dirty()
    }


@app.patch("/api/documents/{doc_id}")
@app.patch("/v1/documents/{doc_id}")
def patch_document(doc_id: str, payload: PatchDocumentPayload, background_tasks: BackgroundTasks):
    """
    Partial update of a document in RAM:
    Allows AIs to alter specific fields (e.g. extending valid_until, changing status, or updating tags)
    without sending the whole document.
    """
    safe_id = sanitize_doc_id(doc_id)
    existing = memory_cache.get(safe_id)
    if not existing:
        raise HTTPException(status_code=404, detail=f"Document '{safe_id}' not found.")

    meta = existing.metadata

    if payload.title is not None:
        meta.title = payload.title
    if payload.type is not None:
        meta.type = payload.type
    if payload.status is not None:
        meta.status = payload.status
    if payload.valid_until is not None:
        meta.valid_until = payload.valid_until
    if payload.category is not None:
        meta.category = payload.category
    if payload.target_services is not None:
        meta.target_services = payload.target_services
    if payload.target_languages is not None:
        meta.target_languages = payload.target_languages
    if payload.tags is not None:
        meta.tags = payload.tags
    if payload.approved_by is not None:
        meta.approved_by = payload.approved_by

    new_body = payload.body if payload.body is not None else existing.body

    yaml_lines = [
        "---",
        f"id: \"{safe_id}\"",
        f"type: \"{meta.type}\"",
        f"title: \"{meta.title or ''}\"",
    ]
    if meta.description:
        yaml_lines.append(f"description: \"{meta.description}\"")
    if meta.category:
        yaml_lines.append(f"category: \"{meta.category}\"")
    if meta.target_services:
        yaml_lines.append(f"target_services: {meta.target_services}")
    if meta.target_languages:
        yaml_lines.append(f"target_languages: {meta.target_languages}")
    if meta.tags:
        yaml_lines.append(f"tags: {meta.tags}")
    if meta.status:
        yaml_lines.append(f"status: \"{meta.status}\"")
    if meta.valid_until:
        yaml_lines.append(f"valid_until: \"{meta.valid_until}\"")
    if meta.approved_by:
        yaml_lines.append(f"approved_by: \"{meta.approved_by}\"")
    yaml_lines.append("---\n")

    raw_content = "\n".join(yaml_lines) + new_body
    target_path = get_safe_bundle_path(safe_id, existing.file_path)
    updated_doc = OKFParser.parse_text(raw_content, file_path=target_path)
    stored = memory_cache.put(updated_doc, persist_to_disk=payload.persist_to_disk, file_path=target_path)

    if payload.auto_rebuild_cache:
        background_tasks.add_task(rebuild_cache)

    return {
        "status": "patched",
        "id": stored.metadata.id,
        "token_count": stored.token_count,
        "is_dirty": memory_cache.is_dirty(),
        "metadata": stored.metadata.model_dump()
    }


@app.post("/api/documents/{doc_id}/append")
@app.post("/v1/documents/{doc_id}/append")
def append_to_document(doc_id: str, payload: AppendDocumentPayload, background_tasks: BackgroundTasks):
    """
    Appends a new section or approved exception to the body of an existing document.
    Enables autonomous agents to register approved RFC exceptions or security waivers in real time.
    """
    safe_id = sanitize_doc_id(doc_id)
    existing = memory_cache.get(safe_id)
    if not existing:
        raise HTTPException(status_code=404, detail=f"Document '{safe_id}' not found.")

    section_heading = f"\n\n## {payload.heading}\n"
    if payload.approved_by:
        section_heading += f"> Aprovado por: `{payload.approved_by}` em {datetime.now(timezone.utc).strftime('%Y-%m-%d')}\n\n"
    else:
        section_heading += "\n"

    appended_body = existing.body + section_heading + payload.content

    patch_payload = PatchDocumentPayload(
        body=appended_body,
        persist_to_disk=payload.persist_to_disk,
        auto_rebuild_cache=payload.auto_rebuild_cache
    )
    return patch_document(doc_id=safe_id, payload=patch_payload, background_tasks=background_tasks)


@app.delete("/api/documents/{doc_id}")
@app.delete("/v1/documents/{doc_id}")
def delete_document(
    doc_id: str,
    background_tasks: BackgroundTasks,
    delete_from_disk: bool = False,
    auto_rebuild_cache: bool = False
):
    """Deletes a document from RAM and optional disk deletion."""
    safe_id = sanitize_doc_id(doc_id)
    existing = memory_cache.get(safe_id)
    if not existing:
        raise HTTPException(status_code=404, detail=f"Document '{safe_id}' not found.")

    if delete_from_disk and existing.file_path:
        base_dir = Path(bundle_dir_path).resolve()
        p = Path(existing.file_path).resolve()
        try:
            p.relative_to(base_dir)
        except ValueError:
            raise HTTPException(status_code=403, detail="Forbidden: Cannot delete file outside bundle directory.")

    deleted = memory_cache.delete(safe_id, remove_from_disk=delete_from_disk)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Document '{safe_id}' not found.")

    if auto_rebuild_cache:
        background_tasks.add_task(rebuild_cache)

    return {"status": "deleted", "id": safe_id, "is_dirty": memory_cache.is_dirty()}


# --- AI Agent Integration Endpoints (/v1) ---

@app.post("/v1/context")
def get_ai_context(payload: ContextRequestPayload):
    """
    AI Integration Endpoint: Returns pre-filtered, structured organizational context
    ready for direct injection into any external AI agent's system prompt (e.g., Cursor, Claude, LangChain).
    """
    eng = get_engine()
    docs = memory_cache.list_all()
    bundle = eng.load_bundle(bundle_dir_path)
    for d in docs:
        if d.metadata.id:
            bundle.documents[d.metadata.id] = d

    composer = OKFComposer()
    selected_docs = composer.filter_documents(
        bundle,
        target_services=[payload.service] if payload.service else None,
        target_languages=[payload.language] if payload.language else None,
        tags=payload.tags if payload.tags else None
    )

    context_str = composer.compose_prefix(bundle, selected_docs=selected_docs)
    doc_ids = [d.metadata.id for d in selected_docs]

    return {
        "status": "success",
        "service": payload.service,
        "language": payload.language,
        "matched_documents_count": len(selected_docs),
        "document_ids": doc_ids,
        "context": context_str
    }


@app.post("/v1/chat/completions")
def openai_chat_completions(payload: OpenAIChatPayload):
    """
    OpenAI-Compatible Chat Completion API:
    Allows any OpenAI SDK, LangChain, LlamaIndex, or AutoGen bot to point to this server
    as an LLM backend (e.g. base_url="http://localhost:8000/v1") and answer queries
    leveraging the zero-prefill in-memory Key-Value cache.
    """
    eng = get_engine()
    if eng.active_cache is None:
        eng.load_bundle(bundle_dir_path)
        eng.build_cache()

    # Extract last user query
    user_messages = [m for m in payload.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message provided in messages array.")

    last_query = user_messages[-1].content

    answer, metrics = eng.generate(
        query=last_query,
        max_new_tokens=payload.max_tokens or 40,
        temperature=payload.temperature or 0.3,
        use_cache=True
    )

    created_timestamp = int(time.time())
    return {
        "id": f"chatcmpl-cag-okf-{created_timestamp}",
        "object": "chat.completion",
        "created": created_timestamp,
        "model": payload.model or "cag-okf",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": answer
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": metrics.prefill_tokens_evaluated,
            "completion_tokens": metrics.output_tokens_generated,
            "total_tokens": metrics.prefill_tokens_evaluated + metrics.output_tokens_generated,
            "prefill_token_savings_percent": metrics.token_savings_percent
        }
    }


# --- In-Memory Key-Value Cache & CAG Operations ---

@app.post("/api/cache/rebuild")
def rebuild_cache(services: Optional[List[str]] = Query(None)):
    """
    Rebuilds the active in-memory Key-Value DynamicCache using resident documents.
    """
    eng = get_engine()
    t0 = time.perf_counter()

    # Re-build bundle from in-memory cache
    docs = memory_cache.list_all()
    bundle = eng.load_bundle(bundle_dir_path)
    for d in docs:
        if d.metadata.id:
            bundle.documents[d.metadata.id] = d

    profile = eng.build_cache(target_services=services)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    memory_cache.set_kv_cache(
        cache=eng.active_cache,
        prefix_ids=eng.active_prefix_ids,
        profile=profile,
        duration_ms=elapsed_ms
    )

    return {
        "status": "rebuilt",
        "total_tokens": profile.total_tokens,
        "documents_count": len(profile.document_ids),
        "duration_ms": round(elapsed_ms, 2),
        "content_hash": profile.content_hash,
    }


@app.get("/api/cache/stats")
def get_cache_stats():
    """Returns memory statistics, hit rates, and KV cache status."""
    return memory_cache.get_stats()


@app.post("/api/query")
def execute_query(payload: QueryPayload):
    """
    Performs inference against the in-memory KV cache with zero prefill overhead on knowledge.
    """
    eng = get_engine()
    if eng.active_cache is None:
        eng.load_bundle(bundle_dir_path)
        eng.build_cache(target_services=payload.services)

    answer, metrics = eng.generate(
        query=payload.query,
        max_new_tokens=payload.max_tokens,
        use_cache=payload.use_cache
    )

    return {
        "query": payload.query,
        "answer": answer,
        "prefill_tokens_evaluated": metrics.prefill_tokens_evaluated,
        "total_context_tokens": eng.active_prefix_len + metrics.prefill_tokens_evaluated,
        "tokens_generated": metrics.output_tokens_generated,
        "token_savings_percent": metrics.token_savings_percent,
        "latency_seconds": metrics.total_time_s,
        "tokens_per_second": metrics.tokens_per_second,
    }


@app.post("/api/benchmark")
def execute_benchmark(payload: QueryPayload):
    """
    Compares Zero-Cache Baseline vs in-memory OKF-CAG side-by-side.
    """
    eng = get_engine()
    if eng.active_cache is None:
        eng.load_bundle(bundle_dir_path)
        eng.build_cache(target_services=payload.services)

    _, baseline = eng.generate(payload.query, max_new_tokens=payload.max_tokens, use_cache=False)
    _, cag = eng.generate(payload.query, max_new_tokens=payload.max_tokens, use_cache=True)

    return {
        "query": payload.query,
        "baseline": {
            "prefill_tokens": baseline.prefill_tokens_evaluated,
            "duration_s": baseline.total_time_s,
            "answer": baseline.response_text,
        },
        "okf_cag": {
            "prefill_tokens": cag.prefill_tokens_evaluated,
            "duration_s": cag.total_time_s,
            "savings_percent": cag.token_savings_percent,
            "answer": cag.response_text,
        }
    }


# Static files mount
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", response_class=HTMLResponse)
def index_page():
    """Serves the web studio frontend."""
    index_html = static_dir / "index.html"
    if index_html.exists():
        with open(index_html, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>CAG-OKF Studio: static/index.html not found.</h1>"


def start(port: int = 8000, host: str = "127.0.0.1"):
    print("\n=======================================================")
    print("[CAG-OKF] Knowledge Studio & In-Memory Cache Server")
    print(f"[CAG-OKF] URL: http://{host}:{port}")
    print("=======================================================\n")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CAG-OKF Server")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--host", default="0.0.0.0", help="Host address to bind")
    args = parser.parse_args()
    start(port=args.port, host=args.host)
