# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAG chatbot that answers questions about educational course materials. FastAPI backend serves a static HTML/JS/CSS frontend. Users ask questions, Claude decides whether to use a search tool to query ChromaDB, then synthesizes an answer from retrieved course content.

## Commands

**Always use `uv` for dependency management and running scripts. Never use `pip`.**

```bash
# Install dependencies
uv sync

# Run the server (from project root)
./run.sh
# Or manually:
cd backend && uv run uvicorn app:app --reload --port 8000

# App available at http://localhost:8000, API docs at http://localhost:8000/docs
```

## Architecture

**Query flow involves two Claude API calls:**
1. User question is sent to Claude with the `search_course_content` tool definition
2. Claude calls the tool → `CourseSearchTool` → `VectorStore.search()` (ChromaDB semantic search)
3. Tool results are sent back to Claude (without tools) for a second call that synthesizes the final answer

**Key design decisions:**
- Course names are resolved semantically: a `course_catalog` ChromaDB collection maps fuzzy names (e.g., "MCP") to exact course titles via vector search, then filters the `course_content` collection
- Document processing expects a specific format: `Course Title:`, `Course Link:`, `Course Instructor:` header lines, then `Lesson N: Title` markers separating content
- Conversation history is passed as a formatted string in the system prompt (not as message array), capped at 2 exchanges
- The `Tool` ABC in `search_tools.py` allows registering new tools via `ToolManager`; each tool self-describes its Anthropic tool definition
- Sources are tracked as side-effects on `CourseSearchTool.last_sources` and reset after each query

**Backend modules (all in `backend/`):**
- `app.py` — FastAPI routes (`POST /api/query`, `GET /api/courses`), loads docs from `../docs` on startup
- `rag_system.py` — Orchestrator wiring all components
- `ai_generator.py` — Anthropic client, system prompt, tool execution loop
- `vector_store.py` — ChromaDB with two collections: `course_catalog` and `course_content`
- `document_processor.py` — Parses course files, sentence-based chunking (800 chars, 100 overlap)
- `search_tools.py` — Tool abstraction + `CourseSearchTool` implementation
- `session_manager.py` — In-memory session/history storage
- `config.py` — Dataclass config from env vars
- `models.py` — Pydantic models: `Course`, `Lesson`, `CourseChunk`

**Frontend (`frontend/`):** Vanilla HTML/JS/CSS chat UI. Uses `marked.js` for markdown rendering. No build step.

## Environment

- Python 3.13+, managed with `uv`
- Requires `ANTHROPIC_API_KEY` in `.env` at project root
- ChromaDB persists to `backend/chroma_db/`
- Course documents go in `docs/` (`.txt` files with structured format)
