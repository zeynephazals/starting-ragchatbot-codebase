# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Application

```bash
# Quick start (from repo root)
./run.sh

# Manual start
cd backend && uv run uvicorn app:app --reload --port 8000
```

The app serves at `http://localhost:8000` (web UI) and `http://localhost:8000/docs` (FastAPI Swagger).

The server must be started from the `backend/` directory because `app.py` uses relative paths (`../docs`, `../frontend`) to locate documents and static files.

## Dependencies

```bash
uv sync   # install dependencies
```

Python 3.13+ is required. Uses `uv` as the package manager (`pyproject.toml`). No test suite exists.

## Environment

Copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY`. The `run.sh` script also sets Azure Foundry env vars (`ANTHROPIC_FOUNDRY_BASE_URL`, `ANTHROPIC_FOUNDRY_API_KEY`) for routing Claude API calls through an Azure endpoint.

## Architecture

This is a full-stack RAG chatbot: a FastAPI backend (`backend/`) serves a static HTML/JS/CSS frontend (`frontend/`) and exposes two API endpoints (`POST /api/query`, `GET /api/courses`).

### Request Flow

1. Frontend sends a query to `POST /api/query`
2. `app.py` delegates to `RAGSystem.query()` — the central orchestrator in `rag_system.py`
3. `RAGSystem` builds a prompt and calls `AIGenerator.generate_response()` with tool definitions
4. Claude (`ai_generator.py`) decides whether to call the `search_course_content` tool
5. If tool use is triggered, `ToolManager` dispatches to `CourseSearchTool.execute()` → `VectorStore.search()`
6. The search result is fed back to Claude for a final response
7. The session exchange is saved in `SessionManager` (in-memory, resets on server restart)

### Key Components

- **`rag_system.py`** — orchestrates all subsystems; entry point for all queries and document ingestion
- **`vector_store.py`** — wraps ChromaDB with two collections: `course_catalog` (course titles/metadata) and `course_content` (chunked lesson text); uses `all-MiniLM-L6-v2` embeddings via `sentence-transformers`
- **`ai_generator.py`** — wraps the Anthropic SDK; handles the two-turn tool-use loop (initial response → tool execution → final response)
- **`search_tools.py`** — `Tool` ABC + `CourseSearchTool` + `ToolManager`; defines the Anthropic tool schema for `search_course_content`; new tools must implement `Tool` and be registered with `ToolManager`
- **`document_processor.py`** — parses `.txt`/`.pdf`/`.docx` course files and splits them into `CourseChunk` objects
- **`session_manager.py`** — in-memory conversation history keyed by session ID; stores last `MAX_HISTORY` exchanges
- **`config.py`** — single `Config` dataclass loaded from `.env`; tune `CHUNK_SIZE`, `CHUNK_OVERLAP`, `MAX_RESULTS`, `MAX_HISTORY`, and `CHROMA_PATH` here

### Course Document Format

Documents in `docs/` must follow this structure for correct parsing:

```
Course Title: <title>
Course Link: <url>
Course Instructor: <name>

Lesson 1: <lesson title>
Lesson Link: <url>
<lesson content...>

Lesson 2: <lesson title>
...
```

On startup, `app.py` auto-loads all `.txt`/`.pdf`/`.docx` files from `../docs`. Already-indexed courses (matched by title) are skipped. ChromaDB data persists in `backend/chroma_db/`.

### Data Model

`models.py` defines three Pydantic models: `Course` (title is the unique ID in ChromaDB), `Lesson` (lesson_number + title + optional link), and `CourseChunk` (the unit stored in the vector store, tagged with course_title and lesson_number for filtered search).
