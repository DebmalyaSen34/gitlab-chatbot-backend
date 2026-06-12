# GitLab Chatbot

A RAG chatbot that answers questions related to Gitlab workings, policies, directions, etc. Built with a focus on answer quality, response speed, and cost efficiency.

**The problem:** New employees face information overload. The data corpus is huge, constantly updated, and hard to search.

**The solution:** A chatbot that lets employees ask questions in plain English and get answers sourced directly from the handbook.

---

## Table of Contents

- [Setup & Installation](#setup--installation)
- [Architecture Overview](#architecture-overview)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
  - [Ingestion Pipeline](#ingestion-pipeline)
  - [Query Pipeline](#query-pipeline)
  - [Semantic Cache](#semantic-cache)
  - [Guardrails](#guardrails)
- [API Reference](#api-reference)
- [Testing](#testing)
- [Deployment](#deployment)
- [Performance & Design Decisions](#performance--design-decisions)
- [License](#license)

---

## Setup & Installation

### Prerequisites

- Python 3.12+
- A Supabase project (free tier works)
- An OpenAI-compatible API key
- (Optional) GitLab API token for ingestion via API

### 1. Clone the Repository

```bash
git clone https://github.com/DebmalyaSen34/gitlab-chatbot-backend.git
cd gitlab-chatbot-backend
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file in the project root:

```env
# Required
OPENAI_API_KEY=your-api-key-here
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_API_KEY=your-supabase-anon-or-service-key
SUPABASE_DB_CONNECTION=postgresql://postgres:password@db.your-project.supabase.co:5432/postgres

# Optional
OPENAI_API_BASE=https://api.openai.com/v1
OPENAI_MODEL=gpt-3.5-turbo
GITLAB_TOKEN=glpat-xxxxxxxxxxxxxxxxxxxx
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5173
```

**Environment Variable Reference:**

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes | — | API key for the OpenAI-compatible LLM endpoint |
| `SUPABASE_URL` | Yes | — | Supabase project URL |
| `SUPABASE_API_KEY` | Yes | — | Supabase publishable or service role key |
| `SUPABASE_DB_CONNECTION` | Yes | — | Direct PostgreSQL connection string (used by vecs and psycopg2) |
| `OPENAI_API_BASE` | No | OpenAI default | Custom base URL for OpenAI-compatible API |
| `OPENAI_MODEL` | No | `gpt-3.5-turbo` | Model name for generation |
| `GITLAB_TOKEN` | No | —  | GitLab API token to avoid rate limits during ingestion |
| `ALLOWED_ORIGINS` | No | `http://localhost:3000,http://localhost:5173` | Comma-separated CORS origins for FastAPI (use `*` to allow all) |

### 5. Database Setup

Run the SQL setup script in your Supabase SQL Editor:

```sql
-- Enable pgvector
CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA extensions;

-- Vectors table
CREATE TABLE IF NOT EXISTS vecs.handbook_embeddings (
    id TEXT PRIMARY KEY,
    vec VECTOR(384) NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

-- HNSW index for fast cosine similarity
CREATE INDEX IF NOT EXISTS ix_vector_cosine_ops_hnsw_m16_efc64
    ON vecs.handbook_embeddings
    USING hnsw (vec vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Semantic cache table
CREATE TABLE IF NOT EXISTS semantic_cache (
    id BIGSERIAL PRIMARY KEY,
    query TEXT NOT NULL,
    embedding BYTEA NOT NULL,
    response TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_semantic_cache_created_at
    ON semantic_cache (created_at DESC);
```

### 6. Clone the GitLab Handbook (for local ingestion)

```bash
git clone https://gitlab.com/gitlab-com/content-sites/handbook.git handbook
```

### 7. Run Ingestion

```bash
# From local clone (default, faster)
python ingest.py --db-connection "$SUPABASE_DB_CONNECTION"

# From GitLab API (no local clone needed)
python ingest.py --use-api --gitlab-token "$GITLAB_TOKEN" --db-connection "$SUPABASE_DB_CONNECTION"
```

### 8. Run the App

**Streamlit frontend:**
```bash
streamlit run app.py
```

**FastAPI backend:**
```bash
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

---

## Architecture Overview

The system is a full-stack RAG application with a Streamlit or React frontend, a FastAPI backend, Supabase for vector storage, and an OpenAI-compatible LLM for generation. The Streamlit frontend includes Chat, Analytics, and Admin tabs. The React frontend provides only the Chat interface.

```mermaid
graph TB
    User(["User"]) --> FE["Frontend<br/>Streamlit / React"]

    subgraph FE_Streamlit["Streamlit Frontend"]
        S_Chat["Chat Tab"]
        S_Analytics["Analytics Tab"]
        S_Admin["Admin Tab"]
    end

    subgraph FE_React["React Frontend"]
        R_Chat["Chat Tab"]
    end

    FE --> FE_Streamlit
    FE --> FE_React

    FE --> API["FastAPI Backend<br/>server.py"]

    subgraph Core["Core Pipeline"]
        direction TB
        GR["Guardrails<br/>guardrails.py"]
        RAG["RAG Controller<br/>rag.py"]
        CACHE["Semantic Cache<br/>cache.py"]
    end

    API --> GR
    GR --> RAG
    GR --> CACHE

    subgraph Storage["Storage Layer"]
        direction LR
        SUPA["Supabase<br/>PostgreSQL + pgvector"]
        SQLITE["SQLite / Postgres<br/>Cache Persistence"]
        MEMORY["In-Memory<br/>numpy Matrix"]
    end

    RAG --> SUPA
    CACHE --> SQLITE
    CACHE --> MEMORY

    subgraph External["External Services"]
        direction LR
        LLM["OpenAI-Compatible<br/>LLM API"]
        EMBED["fastembed<br/>Local CPU"]
    end

    RAG --> LLM
    RAG --> EMBED

    subgraph Data["Data Source"]
        HANDBOOK["GitLab Handbook<br/>Hugo Repository<br/>2,200+ pages"]
    end

    HANDBOOK -->|"ingest.py"| SUPA

    style User fill:#555,stroke:#333,color:#fff
    style FE fill:#888,stroke:#555,color:#fff
    style FE_Streamlit fill:#999,stroke:#666,color:#000
    style FE_React fill:#bbb,stroke:#888,color:#000
    style Core fill:#888,stroke:#555,color:#fff
    style Storage fill:#999,stroke:#666,color:#000
    style External fill:#888,stroke:#555,color:#fff
    style Data fill:#999,stroke:#666,color:#000
```

### Component Interaction

```mermaid
graph LR
    Streamlit["Streamlit Frontend<br/>app.py"] --> FastAPI["FastAPI Backend<br/>server.py"]
    FastAPI --> Supabase["Supabase<br/>PostgreSQL + pgvector"]
    Streamlit --> Guardrails["Guardrails<br/>regex + LLM check"]
    FastAPI --> LLM["LLM API<br/>OpenAI-compatible"]
    Streamlit --> Cache["Semantic Cache<br/>numpy in-memory"]

    style Streamlit fill:#888,stroke:#555,color:#fff
    style FastAPI fill:#999,stroke:#666,color:#000
    style Supabase fill:#aaa,stroke:#777,color:#000
    style Guardrails fill:#888,stroke:#555,color:#fff
    style LLM fill:#999,stroke:#666,color:#000
    style Cache fill:#aaa,stroke:#777,color:#000
```

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | Streamlit (Chat, Analytics, Admin) / React (Chat only) | Interactive web UI |
| **Backend API** | FastAPI + Uvicorn | REST API with async support, CORS, and background tasks |
| **Vector DB** | Supabase (PostgreSQL + pgvector) | Vector storage with HNSW indexing for fast similarity search |
| **Embeddings** | SentenceTransformer (ingestion) / fastembed (query) | 384-dim vectors using `BAAI/bge-small-en-v1.5` |
| **LLM** | OpenAI-compatible API | Generation |
| **Cache** | SQLite (dev) / PostgreSQL (prod) + numpy | Semantic similarity cache with in-memory matrix |
| **Deployment** | Render.com | Container-based deployment with auto-scaling |

---

## Project Structure

```
gitlab-chatbot-backend/
├── app.py                   
├── server.py                
├── rag.py                   
├── ingest.py                
├── cache.py                 
├── guardrails.py            
├── style.css                
├── requirements.txt         
├── render.yaml              
│
└── tests/                   
   ├── conftest.py          
   ├── test_cache.py        
   ├── test_guardrails.py   
   ├── test_ingestion.py    
   └── test_rag.py                   
```

---

## How It Works

### Ingestion Pipeline

The ingestion pipeline is an **offline batch process** that fetches, cleans, chunks, embeds, and stores handbook content. It runs once (or on-demand via the Streamlit Admin tab) to populate the vector database.

```mermaid
flowchart LR
    subgraph SOURCE["Data Source"]
        LOCAL["Local Clone<br/>handbook/"]
        API["GitLab API<br/>fallback"]
    end

    subgraph FETCH["1. Fetch"]
        READ["Read .md files<br/>from 37 directories"]
    end

    subgraph CLEAN["2. Clean"]
        STRIP["Strip Hugo frontmatter"]
        HTML["Remove HTML tags"]
        WHITESPACE["Collapse whitespace"]
    end

    subgraph CHUNK["3. Chunk"]
        SPLIT["Split into<br/>paragraphs"]
        FILTER["Filter tiny<br/>paragraphs < 20 chars"]
        MERGE["Merge paragraphs<br/>min: 500 chars<br/>max: 1,500 chars"]
    end

    subgraph EMBED["4. Embed"]
        MODEL["SentenceTransformer<br/>BAAI/bge-small-en-v1.5"]
        BATCH["Batch encode<br/>512 chunks/batch"]
    end

    subgraph STORE["5. Store"]
        UPSERT["Upsert to Supabase<br/>via vecs (500/batch)"]
        INDEX["Create HNSW Index<br/>m=16, ef_construction=64"]
    end

    LOCAL --> READ
    API --> READ
    READ --> STRIP --> HTML --> WHITESPACE
    WHITESPACE --> SPLIT --> FILTER --> MERGE
    MERGE --> MODEL --> BATCH
    BATCH --> UPSERT --> INDEX

    style SOURCE fill:#999,stroke:#666,color:#000
    style FETCH fill:#888,stroke:#555,color:#fff
    style CLEAN fill:#aaa,stroke:#777,color:#000
    style CHUNK fill:#888,stroke:#555,color:#fff
    style EMBED fill:#aaa,stroke:#777,color:#000
    style STORE fill:#888,stroke:#555,color:#fff
```

**Stage 1 — Fetch:**
- **Default mode:** Reads `.md` files from a local `handbook/` directory (clone of `gitlab-com/content-sites/handbook`)
- **API mode** (`--use-api`): Uses GitLab Repository API to recursively list and fetch files with base64 decoding
- Covers 37 handbook directories (values, engineering, hiring, product, etc.)

**Stage 2 — Clean:**
- Strips Hugo YAML frontmatter (`---` delimiters)
- Removes HTML tags
- Collapses excessive newlines
- Filters out files with < 100 characters of cleaned content

**Stage 3 — Chunk:**
- Splits into paragraphs by `\n\n`
- Filters paragraphs under 20 characters
- Merges small paragraphs into chunks between **500-1,500 characters**
- Each chunk gets a `doc_id` (`{file_path}::{chunk_index}`) and metadata (title, source_path, url, chunk_index, content)

**Stage 4 — Embed & Store:**
- Generates 384-dimensional embeddings via `SentenceTransformer("BAAI/bge-small-en-v1.5")` in batches of 512
- Upserts to Supabase `handbook_embeddings` table via `vecs` in batches of 500
- Creates HNSW index (`m=16, ef_construction=64`) for fast cosine similarity search

---

### Query Pipeline

When a user asks a question, the query pipeline handles embedding, cache lookup, retrieval, and generation.

```mermaid
flowchart TD
    START(["User sends query"]) --> INJECT{"Prompt Injection<br/>Check (regex)"}

    INJECT -->|"Match detected"| REJECT["Reject:<br/>Unsafe query pattern"]
    INJECT -->|"Clean"| TOPIC{"Topic Check<br/>(permissive)"}

    TOPIC -->|"Off-topic"| WARN["Warn:<br/>Specialized in GitLab Handbook"]
    TOPIC -->|"On-topic"| PARALLEL

    subgraph PARALLEL["Parallel Execution"]
        direction LR
        EMBED["Generate Embedding<br/>fastembed CPU<br/>~100ms"]
        LOOKUP["Cache Lookup<br/>numpy matrix @ vector<br/>~5ms"]
    end

    PARALLEL --> HIT{"Cache Hit?<br/>similarity >= 0.95"}

    HIT -->|"Cache HIT"| RETURN_FAST["Return cached response<br/>~100-200ms total"]
    HIT -->|"Cache MISS"| SEARCH

    subgraph SEARCH["Retrieval"]
        direction TB
        VS["Vector Search<br/>Supabase pgvector<br/>Top-15 results<br/>~200-500ms"]
        FILTER["Filter by similarity<br/>threshold: 0.3"]
        SELECT["Select Top-8<br/>by similarity score"]
        VS --> FILTER --> SELECT
    end

    SEARCH --> GENERATE["LLM Generation<br/>OpenAI-compatible API<br/>~15-25s"]

    GENERATE --> RESPOND["Return response<br/>+ sources + metrics"]

    RESPOND --> CACHE_STORE["Store in Cache"]
    RESPOND --> GUARDRAIL["Output Guardrail<br/>async background"]

    GUARDRAIL --> SAFE{"SAFE or UNSAFE?"}
    SAFE -->|"SAFE"| OK["Response verified"]
    SAFE -->|"UNSAFE"| FLAG["Flag response<br/>with warning badge"]

    RETURN_FAST --> END(["Done"])
    CACHE_STORE --> END
    OK --> END
    FLAG --> END

    style START fill:#555,stroke:#333,color:#fff
    style REJECT fill:#777,stroke:#555,color:#fff
    style RETURN_FAST fill:#999,stroke:#666,color:#000
    style OK fill:#999,stroke:#666,color:#000
    style FLAG fill:#777,stroke:#555,color:#fff
    style PARALLEL fill:#888,stroke:#555,color:#fff
    style SEARCH fill:#999,stroke:#666,color:#000
```

**Step-by-step:**

1. **Input Guardrails** — Regex-based prompt injection detection (14 patterns). Blocks immediately if a match is found.
2. **Parallel Embedding + Cache Lookup** — `ThreadPoolExecutor` runs both simultaneously. Embedding takes ~100ms, cache lookup takes ~5ms.
3. **Cache Check** — If similarity >= 0.95, return cached response immediately (~100-200ms total).
4. **Vector Search** — Queries Supabase pgvector for top-15 similar chunks (~200-500ms).
5. **Filter & Select** — Discards chunks below 0.3 similarity, keeps top-8 by score.
6. **LLM Generation** — Builds prompt with context chunks and source URLs, calls OpenAI-compatible API (~15-25s).
7. **Output Guardrail** — Asynchronous LLM-based grounding check runs in background thread.

---

### Semantic Cache

The semantic cache avoids redundant LLM calls by recognizing semantically similar queries.

```mermaid
graph TB
    subgraph QUERY["Query Flow"]
        Q["User Query"] --> E["fastembed<br/>~100ms"]
        E --> V["Query Vector<br/>384 dims"]
    end

    subgraph CACHE_LOOKUP["Cache Lookup (in-memory)"]
        V --> DOT["Matrix-Vector Product<br/>matrix @ query_vector<br/>numpy BLAS ~5ms"]
        MATRIX[("numpy Matrix<br/>shape: N x 384<br/>float32")]
        MATRIX --> DOT
        DOT --> BEST{"Best similarity<br/>>= 0.95?"}
    end

    BEST -->|"HIT"| RETURN["Return cached response<br/>~100ms total"]
    BEST -->|"MISS"| RAG["Continue to<br/>RAG pipeline"]

    RAG --> STORE["Store new entry"]
    STORE --> APPEND_DB["Append to DB"]
    STORE --> APPEND_MEM["Append to<br/>in-memory matrix"]

    subgraph PERSISTENCE["Persistent Storage"]
        DB[("SQLite (dev)<br/>or Postgres (prod)<br/>query TEXT<br/>embedding BYTEA<br/>response TEXT")]
    end

    APPEND_DB --> DB

    STARTUP["App Startup"] --> LOAD["Load all embeddings<br/>into numpy matrix"]
    DB --> LOAD
    LOAD --> MATRIX

    style QUERY fill:#888,stroke:#555,color:#fff
    style CACHE_LOOKUP fill:#999,stroke:#666,color:#000
    style PERSISTENCE fill:#aaa,stroke:#777,color:#000
    style RETURN fill:#999,stroke:#666,color:#000
```

**How it works:**
- On startup, all cached embeddings are loaded into a numpy matrix of shape `[N, 384]`
- Lookup: matrix-vector dot product (`matrix @ query_vector`) computes similarity against all cached queries in one operation (~5ms)
- **Threshold: 0.95** — only near-exact semantic matches trigger a cache hit
- Write-through: new entries are appended to both the persistent store and the in-memory matrix
- **Memory:** ~15 MB for 10,000 cached queries (384 x 4 bytes x 10,000)

---

### Guardrails

Three-layer safety system to prevent misuse and ensure answer quality.

```mermaid
flowchart TD
    Q["User Query"] --> LAYER1

    subgraph LAYER1["Layer 1: Input Guardrails"]
        direction TB
        PI{"Prompt Injection<br/>Detection"}
        PI -->|"14 regex patterns:<br/>ignore previous, jailbreak,<br/>DAN mode, system prompt..."| PI_RESULT

        PI_RESULT{"Match?"}
        PI_RESULT -->|"Yes"| BLOCK["Block query<br/>immediate rejection"]
        PI_RESULT -->|"No"| TOPIC{"Topic Check"}

        TOPIC -->|"Permissive:<br/>always True"| PASS["Proceed"]
    end

    PASS --> PIPELINE["RAG Pipeline<br/>embed -> search -> generate"]

    PIPELINE --> RESPONSE["Response Generated"]

    RESPONSE --> LAYER3

    subgraph LAYER3["Layer 3: Output Guardrail (Async)"]
        direction TB
        SHOW["Show response<br/>to user immediately"]
        BG["Background Task:<br/>verify_response_grounded"]

        SHOW --> BG

        BG --> EVAL["LLM evaluates:<br/>Are claims grounded<br/>in context chunks?"]
        EVAL --> VERDICT{"Verdict"}
        VERDICT -->|"SAFE"| MARK_SAFE["Mark as verified"]
        VERDICT -->|"UNSAFE"| MARK_FLAG["Flag with<br/>warning badge"]
        VERDICT -->|"ERROR"| MARK_ERR["Mark as error"]
    end

    style LAYER1 fill:#888,stroke:#555,color:#fff
    style LAYER3 fill:#999,stroke:#666,color:#000
    style BLOCK fill:#777,stroke:#555,color:#fff
    style MARK_SAFE fill:#aaa,stroke:#777,color:#000
    style MARK_FLAG fill:#777,stroke:#555,color:#fff
    style MARK_ERR fill:#999,stroke:#666,color:#000
```

**Design rationale:**
- **Prompt injection** is caught immediately with regex — no LLM call needed
- **Topic filtering** is intentionally permissive because the RAG similarity threshold naturally rejects off-topic queries
- **Output grounding** runs asynchronously so it doesn't block the user's response. The response is shown immediately, then flagged if ungrounded

---

## API Reference

### `GET /api/health`

Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "rag_initialized": true,
  "cache_stats": {
    "total_entries": 42,
    "memory_mb": 0.06
  }
}
```

### `POST /api/chat`

Send a question and get an answer sourced from the GitLab Handbook.

**Request:**
```json
{
  "query": "What is GitLab's PTO policy?"
}
```

**Response (200):**
```json
{
  "response": "GitLab offers a flexible PTO policy...",
  "sources": [
    {
      "title": "Paid Time Off (PTO)",
      "url": "https://handbook.gitlab.com/handbook/total-rewards/benefits/pto/",
      "snippet": "Team members are entitled to..."
    }
  ],
  "cache_hit": false,
  "latency_ms": 2340.5,
  "ttft_ms": 890.2,
  "guardrail_status": "safe"
}
```

**Error Responses:**
- `400` — Prompt injection detected or off-topic query
- `503` — RAG not initialized (ingestion hasn't run)
- `500` — Internal server error

---

## Testing

Run the full test suite:

```bash
pytest tests/ -v
```

Run specific test files:

```bash
pytest tests/test_cache.py -v          # cache init, store, lookup, similarity
pytest tests/test_guardrails.py -v     # injection patterns, topic, grounding
pytest tests/test_ingestion.py -v      # URL mapping, cleaning, hashing, API
pytest tests/test_rag.py -v            # embedding, search, selection, query
```

**Test coverage:**
- **Cache:** Initialization, store, lookup (exact match, threshold sensitivity, orthogonal vectors), cosine similarity, clear, stats
- **Guardrails:** All 14 prompt injection patterns, permissive topic filtering, output grounding verification (mocked LLM)
- **Ingestion:** URL mapping, markdown cleaning, file hashing, GitLab API fetching (mocked)
- **RAG:** Controller initialization, query embedding, node building, vector search, top-N selection, full query pipeline (mocked)

---

## Deployment

```mermaid
graph TB
    subgraph CLIENT["Client"]
        BROWSER["Browser"]
        API_CLIENT["API Client"]
    end

    subgraph RENDER["Render.com"]
        direction TB
        FASTAPI["FastAPI Server<br/>server.py"]
    end

    subgraph SUPABASE["Supabase (Free Tier)"]
        direction TB
        PG["PostgreSQL"]
        PGV["pgvector Extension"]
        VEC_TABLE[("handbook_embeddings<br/>id TEXT<br/>vec VECTOR 384<br/>metadata JSONB")]
        CACHE_TABLE[("semantic_cache<br/>query TEXT<br/>embedding BYTEA<br/>response TEXT")]
        HNSW["HNSW Index<br/>m=16, ef_construction=64"]
        PG --> PGV
        PGV --> VEC_TABLE
        PGV --> CACHE_TABLE
        VEC_TABLE --> HNSW
    end

    subgraph LLM_PROVIDER["LLM Provider"]
        OPENAI["OpenAI API<br/>or compatible endpoint"]
    end

    BROWSER --> FASTAPI
    API_CLIENT --> FASTAPI

    FASTAPI -->|"Vector search<br/>vecs RPC"| PG
    FASTAPI -->|"Cache read/write"| PG
    FASTAPI -->|"Generation +<br/>Guardrail"| OPENAI

    style CLIENT fill:#888,stroke:#555,color:#fff
    style RENDER fill:#999,stroke:#666,color:#000
    style SUPABASE fill:#aaa,stroke:#777,color:#000
    style LLM_PROVIDER fill:#888,stroke:#555,color:#fff
```

### Render.com

The project includes a `render.yaml` for one-click deployment:

```yaml
services:
  - type: web
    name: gitlab-chatbot-api
    runtime: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn server:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: OPENAI_API_KEY
        sync: false
      - key: OPENAI_API_BASE
        sync: false
      - key: OPENAI_MODEL
        sync: false
      - key: SUPABASE_URL
        sync: false
      - key: SUPABASE_API_KEY
        sync: false
      - key: SUPABASE_DB_CONNECTION
        sync: false
```

**Steps:**
1. Connect your GitHub repo to Render
2. Render detects `render.yaml` and creates the service
3. Set environment variables in the Render dashboard
4. Deploy — ingestion must be run separately (via Streamlit Admin tab or CLI)

## Performance & Design Decisions

### Response Time Breakdown

The query pipeline has two execution paths depending on whether the cache hits. The cache hit path skips retrieval and generation entirely, returning in under 200ms.

```mermaid
flowchart TD
    Q(["Query arrives"]) --> G{"Guardrails<br/>< 1ms"}
    G -->|"Blocked"| X(["Reject"])
    G -->|"Passed"| E["Embed query<br/>~100ms"]

    E --> C{"Cache lookup<br/>~5ms"}
    C -->|"HIT"| CR(["Return cached response<br/>Total: ~100ms"])
    C -->|"MISS"| V["Vector search<br/>~200-500ms"]

    V --> F["Filter + select<br/>~1ms"]
    F --> L["LLM generation<br/>~15-25s"]
    L --> R(["Return response<br/>Total: ~16-26s"])

    R --> CA["Store in cache"]
    R --> GU["Output guardrail<br/>(async, non-blocking)"]

    style G fill:#888,stroke:#555,color:#fff
    style E fill:#999,stroke:#666,color:#000
    style C fill:#888,stroke:#555,color:#fff
    style V fill:#999,stroke:#666,color:#000
    style F fill:#aaa,stroke:#777,color:#000
    style L fill:#777,stroke:#555,color:#fff
    style CR fill:#aaa,stroke:#666,color:#000
    style R fill:#999,stroke:#666,color:#000
    style X fill:#777,stroke:#555,color:#fff
    style CA fill:#aaa,stroke:#777,color:#000
    style GU fill:#aaa,stroke:#777,color:#000
```

**Per-component latency:**

| Step | Latency | What happens |
|------|---------|-------------|
| Guardrails | < 1ms | 14 regex patterns checked against the query |
| Embed query | ~100ms | fastembed generates 384-dim vector on CPU |
| Cache lookup | ~5ms | numpy matrix-vector dot product against all cached embeddings |
| Vector search | ~200-500ms | Supabase pgvector HNSW index returns top-15 chunks |
| Filter + select | ~1ms | Drop below 0.3 similarity, keep top-8 |
| LLM generation | ~30-40s | OpenAI-compatible API call with context prompt |
| Output guardrail | Non-blocking | LLM verifies grounding in background thread |

**Path comparison:**

| Metric | Cache Hit | Cache Miss |
|--------|-----------|------------|
| Steps executed | Guardrails, embed, cache lookup | Full pipeline |
| LLM calls | 0 | 1 (generation) + 1 (async guardrail) |
| DB queries | 0 | 1 (vector search) |
| Total latency | ~100ms | ~16-26s |
| Speedup | **160-260x faster** than cache miss | Baseline |

The cache hit path avoids the two most expensive operations — vector search and LLM generation — which together account for 99.5% of cache miss latency. This is why the semantic cache (threshold: 0.95) is the single highest-impact optimization in the system.

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **No LLM-based reranking** | Tested with reranker — added ~7.5s latency with minimal quality improvement. Vector similarity alone provides good ranking. |
| **Semantic cache threshold: 0.95** | High threshold ensures only truly similar queries get cached responses. Prevents stale/wrong answers. |
| **fastembed for query, SentenceTransformer for ingestion** | Same model (`BAAI/bge-small-en-v1.5`), different libraries. fastembed is optimized for single-query latency; SentenceTransformer is optimized for batch throughput. |
| **Chunk size 500-1500 chars** | Tested 200-2000, 500-1500, 1000-3000. Middle range provides best balance of context and precision. |
| **Top-8 chunks to LLM** | More chunks = more context but higher token cost. 8 provides sufficient coverage for most handbook questions. |
| **Async output guardrail** | Grounding check runs after response is shown. Doesn't block UX. Unsafe responses get flagged retroactively. |
| **Permissive topic filtering** | RAG's similarity threshold naturally rejects off-topic queries. Explicit topic classification adds latency without value. |

### Data Source

- **Repository:** `gitlab-com/content-sites/handbook` (public, Apache 2.0 license) + `GitLab Direction Pages`
- **Size:** 2,200+ files making it ~300Mb
