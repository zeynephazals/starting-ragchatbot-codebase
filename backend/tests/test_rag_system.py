"""Tests for RAGSystem content-query handling (backend/rag_system.py).

These wire up a *real* RAGSystem (real VectorStore + real ToolManager +
real CourseSearchTool) and mock only the Anthropic client, so the model
"decides" to call the search tool. This is the closest thing to the actual
end-to-end path a content question travels, and it surfaces whether the
configured system can return sources at all.
"""
import types
from unittest.mock import patch

import pytest

from config import config as real_config
from tests.conftest import make_response, make_text_block, make_tool_use_block


def _test_config(temp_chroma_path, max_results):
    """A config clone pointing at a temp DB with a chosen MAX_RESULTS."""
    return types.SimpleNamespace(
        CHUNK_SIZE=real_config.CHUNK_SIZE,
        CHUNK_OVERLAP=real_config.CHUNK_OVERLAP,
        CHROMA_PATH=temp_chroma_path,
        EMBEDDING_MODEL=real_config.EMBEDDING_MODEL,
        MAX_RESULTS=max_results,
        MAX_HISTORY=real_config.MAX_HISTORY,
        AZURE_API_KEY="k",
        AZURE_ENDPOINT="http://x",
        ANTHROPIC_MODEL="claude-haiku-4-5",
    )


def _build_rag(cfg, sample_course, sample_chunks):
    """Construct a RAGSystem with a mocked Anthropic client, pre-loaded with a course."""
    with patch("ai_generator.AnthropicFoundry"):
        from rag_system import RAGSystem
        rag = RAGSystem(cfg)
    rag.vector_store.add_course_metadata(sample_course)
    rag.vector_store.add_course_content(sample_chunks)
    return rag


def _simulate_search_then_answer(rag, user_query):
    """Make the mocked model call the search tool, then answer."""
    client = rag.ai_generator.client
    client.messages.create.side_effect = [
        make_response(
            "tool_use",
            [make_tool_use_block("search_course_content", {"query": user_query}, "tu_1")],
        ),
        make_response("end_turn", [make_text_block("Here is what the course says.")]),
    ]


def test_content_query_returns_sources_with_real_config(
    temp_chroma_path, sample_course, sample_chunks
):
    """End-to-end content query using the ACTUAL configured MAX_RESULTS.
    Fails while MAX_RESULTS == 0 because the search returns nothing and no
    sources reach the user."""
    cfg = _test_config(temp_chroma_path, real_config.MAX_RESULTS)
    rag = _build_rag(cfg, sample_course, sample_chunks)
    _simulate_search_then_answer(rag, "What is MCP?")

    response, sources = rag.query("What is MCP?")

    assert response  # non-empty answer
    assert sources, (
        "Content query produced no sources — the search tool returned nothing. "
        f"(MAX_RESULTS={cfg.MAX_RESULTS})"
    )


def test_tool_manager_search_returns_content_with_real_config(
    temp_chroma_path, sample_course, sample_chunks
):
    """Isolates the RAG wiring + real store: invoking the registered search tool
    directly should return content, not the empty-results message."""
    cfg = _test_config(temp_chroma_path, real_config.MAX_RESULTS)
    rag = _build_rag(cfg, sample_course, sample_chunks)

    out = rag.tool_manager.execute_tool("search_course_content", query="What is MCP?")

    assert "No relevant content found" not in out
    assert out.strip()


def test_content_query_works_with_positive_max_results(
    temp_chroma_path, sample_course, sample_chunks
):
    """Control: identical flow with MAX_RESULTS=5 returns sources, proving the
    only broken variable is the configuration."""
    cfg = _test_config(temp_chroma_path, 5)
    rag = _build_rag(cfg, sample_course, sample_chunks)
    _simulate_search_then_answer(rag, "What is MCP?")

    response, sources = rag.query("What is MCP?")

    assert response
    assert sources


def test_both_tools_registered(temp_chroma_path, sample_course, sample_chunks):
    """Sanity: the search tool (and outline tool) are registered with the manager."""
    cfg = _test_config(temp_chroma_path, 5)
    rag = _build_rag(cfg, sample_course, sample_chunks)

    names = {d["name"] for d in rag.tool_manager.get_tool_definitions()}
    assert "search_course_content" in names
