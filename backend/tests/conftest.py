"""Shared fixtures and helpers for the backend test suite.

These tests live in ``backend/tests`` but the application modules use flat
imports (``from vector_store import ...``), so we put ``backend/`` on the path.
"""
import os
import sys
import types
import tempfile
import shutil
from unittest.mock import MagicMock

import pytest

# Make the backend package importable with its flat module layout.
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from models import Course, Lesson, CourseChunk  # noqa: E402
from vector_store import VectorStore, SearchResults  # noqa: E402


# --------------------------------------------------------------------------- #
# Sample data
# --------------------------------------------------------------------------- #
@pytest.fixture
def sample_course():
    """A small course with two lessons."""
    return Course(
        title="Test Course: MCP Basics",
        course_link="https://example.com/mcp",
        instructor="Ada Lovelace",
        lessons=[
            Lesson(lesson_number=1, title="Intro to MCP",
                   lesson_link="https://example.com/mcp/1"),
            Lesson(lesson_number=2, title="Building a Server",
                   lesson_link="https://example.com/mcp/2"),
        ],
    )


@pytest.fixture
def sample_chunks(sample_course):
    """Content chunks that should be retrievable by a content search."""
    return [
        CourseChunk(
            content="MCP is the Model Context Protocol, a standard for tool use.",
            course_title=sample_course.title,
            lesson_number=1,
            chunk_index=0,
        ),
        CourseChunk(
            content="To build an MCP server you define tools and handle requests.",
            course_title=sample_course.title,
            lesson_number=2,
            chunk_index=1,
        ),
    ]


# --------------------------------------------------------------------------- #
# Real vector store (temp-backed). max_results is parametrizable so we can
# demonstrate the effect of the MAX_RESULTS config value.
# --------------------------------------------------------------------------- #
@pytest.fixture
def temp_chroma_path():
    path = tempfile.mkdtemp(prefix="chroma_test_")
    yield path
    shutil.rmtree(path, ignore_errors=True)


def _build_store(path, sample_course, sample_chunks, max_results):
    store = VectorStore(path, "all-MiniLM-L6-v2", max_results=max_results)
    store.add_course_metadata(sample_course)
    store.add_course_content(sample_chunks)
    return store


@pytest.fixture
def make_store(temp_chroma_path, sample_course, sample_chunks):
    """Factory: build a populated real VectorStore with a given max_results."""
    def _factory(max_results):
        return _build_store(temp_chroma_path, sample_course, sample_chunks, max_results)
    return _factory


# --------------------------------------------------------------------------- #
# Fake Anthropic message responses (for ai_generator / rag_system tests)
# --------------------------------------------------------------------------- #
def make_text_block(text):
    block = types.SimpleNamespace()
    block.type = "text"
    block.text = text
    return block


def make_tool_use_block(name, tool_input, tool_id="tool_1"):
    block = types.SimpleNamespace()
    block.type = "tool_use"
    block.name = name
    block.input = tool_input
    block.id = tool_id
    return block


def make_response(stop_reason, content_blocks):
    resp = types.SimpleNamespace()
    resp.stop_reason = stop_reason
    resp.content = content_blocks
    return resp


@pytest.fixture
def text_response():
    return lambda text: make_response("end_turn", [make_text_block(text)])


@pytest.fixture
def tool_use_response():
    def _factory(name, tool_input, tool_id="tool_1"):
        return make_response("tool_use", [make_tool_use_block(name, tool_input, tool_id)])
    return _factory


# --------------------------------------------------------------------------- #
# API testing fixtures
#
# ``app.py`` mounts ``StaticFiles(directory="../frontend")`` at import time,
# which does not exist in the test environment and would raise on import. To
# keep the HTTP tests hermetic, we build a fresh FastAPI app here that mounts
# the *same* API routes against a mocked RAGSystem — no static files, no
# ChromaDB, no Anthropic calls.
# --------------------------------------------------------------------------- #
@pytest.fixture
def mock_rag_system():
    """A RAGSystem test double with the methods the API endpoints touch."""
    rag = MagicMock()
    rag.query.return_value = (
        "MCP is the Model Context Protocol.",
        ["Test Course: MCP Basics - Lesson 1"],
    )
    rag.get_course_analytics.return_value = {
        "total_courses": 1,
        "course_titles": ["Test Course: MCP Basics"],
    }
    rag.session_manager.create_session.return_value = "session-123"
    rag.session_manager.delete_session.return_value = None
    return rag


def _build_api_app(rag):
    """Construct a FastAPI app exposing the production API routes.

    Mirrors the endpoints in ``backend/app.py`` but omits the static-file
    mount and startup document loading so it imports cleanly under test.
    """
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    from typing import List, Optional

    app = FastAPI(title="Course Materials RAG System (test)")

    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None

    class QueryResponse(BaseModel):
        answer: str
        sources: List[str]
        session_id: str

    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]

    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id
            if not session_id:
                session_id = rag.session_manager.create_session()
            answer, sources = rag.query(request.query, session_id)
            return QueryResponse(answer=answer, sources=sources, session_id=session_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/sessions/clear")
    async def clear_session(request: QueryRequest):
        if request.session_id:
            rag.session_manager.delete_session(request.session_id)
        return {"status": "ok"}

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = rag.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"],
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/")
    async def root():
        # Stand-in for the static frontend mount in production.
        return {"status": "ok", "service": "Course Materials RAG System"}

    return app


@pytest.fixture
def api_app(mock_rag_system):
    """The inline FastAPI app wired to the mocked RAGSystem."""
    return _build_api_app(mock_rag_system)


@pytest.fixture
def client(api_app):
    """A TestClient for issuing HTTP requests against the inline API app."""
    from fastapi.testclient import TestClient

    with TestClient(api_app) as test_client:
        yield test_client
