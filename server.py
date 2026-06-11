import os
import time
import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from cache import SemanticCache
from guardrails import is_prompt_injection, is_on_topic, verify_response_grounded
from rag import RAGController

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Lifespan: init RAG + cache once ──
@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag, cache

    api_key = os.environ.get("OPENAI_API_KEY", "")
    supabase_url = os.environ.get("SUPABASE_URL", "")
    supabase_key = os.environ.get("SUPABASE_API_KEY", "")

    db_connection = os.environ.get("SUPABASE_DB_CONNECTION", "")
    cache = SemanticCache(db_connection=db_connection) if db_connection else None
    if cache is None:
        logger.warning("Missing SUPABASE_DB_CONNECTION — cache not initialized")

    if api_key and supabase_url and supabase_key:
        try:
            rag = RAGController(
                api_key=api_key,
                supabase_url=supabase_url,
                supabase_key=supabase_key,
            )
            logger.info("RAG controller initialized")
        except Exception as e:
            logger.error(f"Failed to init RAG: {e}")
            rag = None
    else:
        logger.warning("Missing credentials — RAG not initialized")
        rag = None

    yield

    rag = None
    cache = None


app = FastAPI(title="GitLab Handbook API", lifespan=lifespan)

_origins_raw = os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:5173")
ALLOWED_ORIGINS = ["*"] if _origins_raw.strip() == "*" else [o.strip() for o in _origins_raw.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag: RAGController | None = None
cache: SemanticCache | None = None


# ── Models ──
class ChatRequest(BaseModel):
    query: str


class SourceChunk(BaseModel):
    title: str
    url: str
    content: str


class ChatResponse(BaseModel):
    id: str
    role: str = "assistant"
    content: str
    cache: str
    latency: float
    ttft: float
    sources: list[SourceChunk]
    guardrail_status: str = "pending"


# ── Background guardrail ──
def _run_guardrail_async(response_text: str, chunks: list, api_key: str, response_id: str):
    """Run output guardrail in background. Logs result."""
    try:
        is_safe = verify_response_grounded(response_text, chunks, api_key)
        if is_safe:
            logger.info(f"[Guardrail] Response {response_id}: SAFE")
        else:
            logger.warning(f"[Guardrail] Response {response_id}: UNSAFE — flagged")
    except Exception as e:
        logger.error(f"[Guardrail] Response {response_id}: error — {e}")


# ── Endpoints ──
@app.get("/")
async def root():
    return {"status": "ok"}


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "rag_initialized": rag is not None,
        "cache_entries": cache.get_stats()["total_entries"] if cache else 0,
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, background_tasks: BackgroundTasks):
    query = req.query.strip()
    if not query:
        raise HTTPException(400, "Query cannot be empty")

    if rag is None or cache is None:
        raise HTTPException(503, "RAG controller not initialized. Check server credentials.")

    api_key = os.environ.get("OPENAI_API_KEY", "")

    # 1. Input guardrails
    if is_prompt_injection(query):
        raise HTTPException(400, "Unsafe query pattern detected.")

    if not is_on_topic(query):
        raise HTTPException(
            400,
            "I'm specialized in the GitLab Handbook. Try asking about GitLab's values, culture, hiring, or product direction.",
        )

    start = time.time()

    try:
        # 2. Cache lookup (embedding is already generated for cache check)
        query_embedding = rag.get_query_embedding(query)
        cached_res = cache.lookup(query_embedding)

        if cached_res:
            return ChatResponse(
                id=str(int(time.time() * 1000)),
                content=cached_res,
                cache="HIT",
                latency=round(time.time() - start, 3),
                ttft=round(time.time() - start, 3),
                sources=[],
                guardrail_status="safe",
            )

        # 3. RAG retrieval & generation
        result = rag.query(query, query_embedding=query_embedding)
        response_text = result["response"]
        response_id = str(int(time.time() * 1000))

        # 4. Format sources
        sources = []
        for chunk in result.get("retrieved_chunks", []):
            meta = chunk.metadata if hasattr(chunk, "metadata") else {}
            sources.append(
                SourceChunk(
                    title=meta.get("title", "Handbook Page"),
                    url=meta.get("url", "https://handbook.gitlab.com/"),
                    content=chunk.get_content()[:300],
                )
            )

        # 5. Cache store
        cache.store(query, query_embedding, response_text)

        # 6. Fire output guardrail in background — don't block response
        background_tasks.add_task(
            _run_guardrail_async,
            response_text,
            result["retrieved_chunks"],
            api_key,
            response_id,
        )

        return ChatResponse(
            id=response_id,
            content=response_text,
            cache="MISS",
            latency=round(result.get("latency", time.time() - start), 3),
            ttft=round(result.get("time_to_first_token", 0), 3),
            sources=sources,
            guardrail_status="pending",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Chat error")
        raise HTTPException(500, str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
