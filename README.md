# Customer Support RAG

A customer support chatbot powered by a Retrieval-Augmented Generation (RAG) pipeline. It answers natural language questions against a knowledge base, fetches live order and invoice data from Supabase, handles self-service actions (order cancellation, profile updates), and verifies refund claims from product photos using a vision model.

---

## Tech Stack

| Tool | Why we used it |
|---|---|
| **FastAPI** | We needed a lightweight async API layer to expose the RAG pipeline, auth, and admin routes — FastAPI's dependency injection made JWT auth and request validation clean to add |
| **Chainlit** | We wanted an admin + user chat interface without building a frontend — Chainlit gave us auth, thread history, file upload, and action buttons out of the box |
| **Azure OpenAI** | Our primary inference and embedding backend — `gpt-4.1` for answer generation and `text-embedding-3-small` for consistent embeddings across ingestion and retrieval |
| **Google Gemini** (`gemini-2.0-flash`) | We needed a fast, cheap model for HyDE generation and answer relevancy checks — Gemini 2.0 Flash is free-tier and low-latency, perfect for sub-second judgment calls |
| **Groq** (`llama-3.1-8b-instant` / `llama-4-scout`) | Used as a zero-cost fallback when Gemini is unavailable, and as our vision model for refund image analysis since `llama-4-scout` supports multimodal input |
| **Qdrant** | We needed a vector store with payload filtering to store and search FAQ chunks by metadata — Qdrant's scored point results fit our retrieval gate logic directly |
| **Supabase** | We used it as a single managed Postgres backend for all structured data — users, chat history, orders, invoices, customers, and refund records — avoiding a separate DB service |
| **Langfuse** | We needed per-call token tracking and cost visibility across multiple LLMs — Langfuse's generation spans give us this without instrumenting each call individually |
| **RAGAS** | We needed a way to catch regressions when we change prompts or retrieval settings — RAGAS runs the live pipeline against a golden question set and scores faithfulness, recall, and relevancy |
| **LangChain Text Splitters** | We used `MarkdownHeaderTextSplitter` to preserve section structure from FAQ docs (headers become chunk metadata) and `RecursiveCharacterTextSplitter` for PDF/DOCX fallback |
| **PyMuPDF / python-docx** | Local text extraction so ingestion doesn't call any external API for standard PDFs and DOCX files — keeps ingestion fast and cost-free for most documents |

---

## Architecture

The system is organized into four pipelines:

**Pipeline 1 — Ingestion** (admin-only): Load documents (PDF/DOCX/MD) → chunk → embed → upsert to Qdrant.

```
User Query
    │
    ▼
Pipeline 2: Query Processing  (Azure gpt-4.1)
  Load chat history → rewrite query → detect intent → extract entities
    │
    ▼
Pipeline 3: Retrieval & Response  (Azure gpt-4.1 + Gemini)
  HyDE → Qdrant vector search → GATE 1 (score check)
  → LLM generation → GATE 2 (answer relevancy check)
  → confidence score → citations
    │
    ▼
Pipeline 4: Refund Verification  (Groq Vision, image upload)
  Authenticity check → issue classification → claim match
  → auto-approve / auto-reject / Human-in-the-Loop
```

### Model Routing

| Task | Model |
|---|---|
| Answer generation | Azure `gpt-4.1` |
| Embeddings | Azure `text-embedding-3-small` |
| HyDE + relevancy gates | Gemini `gemini-2.0-flash` → Groq fallback |
| Refund image analysis | Groq `llama-4-scout-17b-16e-instruct` |
| Multi-intent splitting | Gemini `gemini-2.0-flash` → Groq fallback |

---

## Setup & Run

### 1. Environment Configuration

See [`.env.example`](.env.example) for all required and optional variables.

### 2. Run Locally

```bash
# Install dependencies
uv sync

# Terminal 1: FastAPI backend (port 8000)
uv run uvicorn app.app:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Chainlit frontend (port 8501)
uv run chainlit run chainlit_app.py --port 8501
```

### 3. Ingest Knowledge Base Documents

Log in as admin in the Chainlit UI and use slash commands:

```
/ingest /path/to/docs    # ingest a folder by path
/browse                  # interactive folder browser
/status                  # view last 10 ingestion runs
```

Or drag and drop `.md`, `.pdf`, or `.docx` files directly into the chat.

---

## Testing & Evaluation

### Unit Tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run a specific file
uv run pytest tests/test_retrieval.py -v
```

Tests run without any real credentials — `conftest.py` injects dummy env vars so no Azure/Qdrant/Supabase connection is needed.

### RAG Evaluation (RAGAS)

Runs the live pipeline against a golden question set and scores it:

```bash
# Full evaluation
uv run python evaluation/ragas_eval.py

# Smoke test — 2 cases only
uv run python evaluation/ragas_eval.py --limit 2

# Filter by category
uv run python evaluation/ragas_eval.py --categories shipping,payment
```

Metrics: `answer_relevancy`, `faithfulness`, `context_recall`, `context_precision`, `answer_correctness`.

Results are saved to `evaluation/results/` as CSV and JSON.

---

## Project Structure

```
Customer_Support-RAG/
├── app/                   # Core application (FastAPI, pipelines, LLM, helpers)
│   ├── pipelines/         # Four pipelines: ingestion, query, retrieval, vision
│   ├── routes/            # API route handlers
│   ├── services/          # Business logic (chat, cancellation, profile)
│   ├── llm/               # LLM clients and prompt templates
│   ├── helpers/           # Supabase, date facts, tokens, logger
│   └── vectorstore/       # Qdrant client wrapper
├── chainlit_app.py        # Chainlit UI
├── evaluation/            # RAGAS evaluation harness + golden question set
├── tests/                 # Unit tests
├── .env.example
└── pyproject.toml
```

---

## API Endpoints

### Auth
| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/auth/register` | Create account, returns JWT |
| `POST` | `/api/v1/auth/login` | Login, returns JWT |
| `POST` | `/api/v1/auth/refresh` | Refresh access token |

### Chat
| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/chat/query` | RAG query — returns answer + citations |
| `POST` | `/api/v1/chat/analyze-image` | Upload product photo for refund verification |
| `POST` | `/api/v1/chat/cancel-order` | Execute order cancellation |

### Profile
| Method | Path | Description |
|---|---|---|
| `GET` | `/api/v1/profile/contact` | Get phone and address |
| `PUT` | `/api/v1/profile/contact` | Update phone or address |

### Admin
| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/admin/ingest` | Ingest a folder of documents |
| `GET` | `/api/v1/admin/ingestion/status` | View recent ingestion run history |

### Health
| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Returns `{"status": "healthy"}` |
