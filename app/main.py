import asyncio
import logging
import os
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.agent.react import ReActAgent
from app.agent.tools import ToolRegistry
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.ingestion.parser import SUPPORTED_EXTENSIONS
from app.ingestion.service import DocumentIngestionService
from app.llm.client import LLMClient, LLMUnavailable
from app.retrieval.hybrid import HybridRetriever
from app.schemas import AgentRequest, AgentResponse, Citation, HealthResponse, IngestionResponse, QueryRequest, QueryResponse
from app.security.guardrails import GuardrailEngine

configure_logging()
logger = logging.getLogger(__name__)
settings = get_settings()
_rate_limit_buckets: dict[str, tuple[int, int]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.settings = settings
    app.state.retriever = HybridRetriever(settings)
    app.state.ingestion = DocumentIngestionService(settings, app.state.retriever)
    app.state.llm = LLMClient(settings)
    app.state.guardrails = GuardrailEngine()
    app.state.tools = ToolRegistry(settings.sqlite_path, app.state.retriever)
    app.state.agent = ReActAgent(settings, app.state.llm, app.state.tools)
    yield


app = FastAPI(
    title="Production-Ready LLM Capstone API",
    description="Hybrid RAG, bounded ReAct tools, safety guardrails, and vLLM-backed generation.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",") if origin.strip()],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-Request-ID"],
)


@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))[:128]
    request.state.request_id = request_id
    started = time.perf_counter()
    if request.url.path not in {"/healthz", "/readyz", "/docs", "/openapi.json", "/redoc"}:
        client_ip = request.client.host if request.client else "unknown"
        current_window = int(time.time() // 60)
        previous_window, previous_count = _rate_limit_buckets.get(client_ip, (current_window, 0))
        count = previous_count + 1 if previous_window == current_window else 1
        _rate_limit_buckets[client_ip] = (current_window, count)
        if len(_rate_limit_buckets) > 10_000:
            for key, (window, _) in list(_rate_limit_buckets.items()):
                if window < current_window:
                    _rate_limit_buckets.pop(key, None)
        if count > settings.rate_limit_per_minute:
            response = JSONResponse(status_code=429, content={"detail": "Rate limit exceeded", "request_id": request_id})
            response.headers["X-Request-ID"] = request_id
            response.headers["Cache-Control"] = "no-store"
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            return response
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("Unhandled request failure request_id=%s path=%s", request_id, request.url.path)
        response = JSONResponse(status_code=500, content={"detail": "Internal server error", "request_id": request_id})
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Cache-Control"] = "no-store"
    logger.info("request_id=%s method=%s path=%s status=%s latency_ms=%.2f", request_id, request.method, request.url.path, response.status_code, (time.perf_counter() - started) * 1000)
    return response


def _check_input(request: Request, question: str) -> str:
    result = request.app.state.guardrails.inspect_input(question)
    if not result.allowed:
        logger.warning("Blocked request request_id=%s reason=%s", request.state.request_id, result.reason)
        raise HTTPException(status_code=400, detail={"code": "guardrail_block", "message": "Request blocked by safety controls."})
    if result.redactions:
        logger.info("PII redacted request_id=%s redaction_count=%s", request.state.request_id, result.redactions)
    return result.text


@app.get("/healthz", response_model=HealthResponse, tags=["Operations"])
async def healthz(request: Request):
    return HealthResponse(status="ok", service="production-llm-capstone", dependencies={"api": "up"})


@app.get("/readyz", response_model=HealthResponse, tags=["Operations"])
async def readyz(request: Request):
    qdrant_ok, inference_ok = await asyncio.gather(
        asyncio.to_thread(request.app.state.retriever.health),
        request.app.state.llm.health(),
    )
    dependencies = {"qdrant": "up" if qdrant_ok else "down", "vllm": "up" if inference_ok else "down"}
    is_ready = qdrant_ok and inference_ok
    result = HealthResponse(status="ready" if is_ready else "degraded", service="production-llm-capstone", dependencies=dependencies)
    if not is_ready:
        raise HTTPException(status_code=503, detail=result.model_dump())
    return result


@app.post("/v1/ingest", response_model=IngestionResponse, tags=["Retrieval"])
async def ingest_documents(request: Request, files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="Upload at least one PDF, DOCX, or Markdown file")
    if len(files) > 10:
        raise HTTPException(status_code=413, detail="At most 10 files may be uploaded in one request")
    total_chunks = 0
    processed: list[str] = []
    with tempfile.TemporaryDirectory(prefix="llm-capstone-") as temp_dir:
        for upload in files:
            filename = Path(upload.filename or "upload").name
            suffix = Path(filename).suffix.lower()
            if suffix not in SUPPORTED_EXTENSIONS:
                raise HTTPException(status_code=415, detail=f"Unsupported document type: {suffix or 'unknown'}")
            data = await upload.read(settings.max_upload_mb * 1024 * 1024 + 1)
            if len(data) > settings.max_upload_mb * 1024 * 1024:
                raise HTTPException(status_code=413, detail=f"{filename} exceeds the {settings.max_upload_mb} MB upload limit")
            temp_path = Path(temp_dir) / f"{uuid.uuid4().hex}{suffix}"
            temp_path.write_bytes(data)
            try:
                count = await asyncio.to_thread(request.app.state.ingestion.ingest_path, temp_path, filename)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            except Exception as exc:
                logger.warning("Ingestion failed request_id=%s error=%s", request.state.request_id, type(exc).__name__)
                raise HTTPException(status_code=503, detail="Document ingestion failed. Check Qdrant and embedding configuration.") from exc
            total_chunks += count
            processed.append(filename)
    return IngestionResponse(files_processed=len(processed), chunks_added=total_chunks, sources=processed)


@app.post("/v1/query", response_model=QueryResponse, tags=["Retrieval"])
async def query_documents(request: Request, body: QueryRequest):
    started = time.perf_counter()
    question = _check_input(request, body.question)
    top_k = min(body.top_k or settings.default_top_k, settings.max_top_k)
    try:
        retrieved = await asyncio.to_thread(request.app.state.retriever.search, question, top_k)
        answer = await request.app.state.llm.answer_from_context(question, retrieved)
    except LLMUnavailable as exc:
        raise HTTPException(status_code=503, detail="Inference service is unavailable. Check the vLLM service.") from exc
    except Exception as exc:
        logger.warning("Query failed request_id=%s error=%s", request.state.request_id, type(exc).__name__)
        raise HTTPException(status_code=503, detail="Retrieval service is unavailable. Check Qdrant and the embedding model.") from exc
    answer = request.app.state.guardrails.inspect_output(answer)
    citations = [Citation(source=row["source"], chunk_id=row["id"], score=row["score"]) for row in retrieved]
    return QueryResponse(
        answer=answer,
        citations=citations,
        retrieved_contexts=[request.app.state.guardrails.inspect_output(row["text"]) for row in retrieved],
        latency_ms=round((time.perf_counter() - started) * 1000, 2),
        model=settings.vllm_model,
        request_id=request.state.request_id,
    )


@app.post("/v1/agent", response_model=AgentResponse, tags=["Agent"])
async def run_agent(request: Request, body: AgentRequest):
    question = _check_input(request, body.question)
    result = await request.app.state.agent.run(question)
    safe_steps = []
    for step in result["steps"]:
        safe_steps.append({**step, "observation": request.app.state.guardrails.inspect_output(str(step["observation"]))})
    answer = request.app.state.guardrails.inspect_output(str(result["answer"]))
    return AgentResponse(answer=answer, steps=safe_steps, final_action=result["final_action"], request_id=request.state.request_id)


@app.get("/v1/tools", tags=["Agent"])
async def list_tools():
    return {
        "tools": [
            {"name": "rag_search", "description": "Search indexed files using dense+sparse hybrid retrieval"},
            {"name": "sql_query", "description": "Read-only SELECT queries against the seeded SQLite sales database"},
            {"name": "python_exec", "description": "Restricted Python expression tool for arithmetic and basic data operations"},
        ],
        "loop_limit": settings.max_agent_steps,
    }
