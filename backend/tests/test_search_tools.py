"""Tests for CourseSearchTool.execute (backend/search_tools.py).

Two layers:
  * Unit tests with a mocked VectorStore — prove the tool's own logic
    (formatting, error pass-through, empty handling, source tracking) is correct.
  * Integration tests with a real VectorStore wired to the *actual* config value
    — these reveal whether the system as configured can return content at all.
"""
from unittest.mock import MagicMock

import pytest

from search_tools import CourseSearchTool
from vector_store import SearchResults
from config import config


# --------------------------------------------------------------------------- #
# Unit tests: CourseSearchTool logic in isolation (mocked store)
# --------------------------------------------------------------------------- #
def _results(docs, metas, distances=None):
    return SearchResults(
        documents=docs,
        metadata=metas,
        distances=distances if distances is not None else [0.1] * len(docs),
    )


def test_execute_formats_results_with_headers():
    store = MagicMock()
    store.search.return_value = _results(
        ["MCP is a protocol."],
        [{"course_title": "MCP Basics", "lesson_number": 1}],
    )
    store.get_lesson_link.return_value = "https://example.com/1"

    tool = CourseSearchTool(store)
    out = tool.execute(query="what is mcp")

    assert "MCP Basics" in out
    assert "Lesson 1" in out
    assert "MCP is a protocol." in out


def test_execute_passes_course_and_lesson_filters():
    store = MagicMock()
    store.search.return_value = _results(["doc"], [{"course_title": "X", "lesson_number": 2}])
    store.get_lesson_link.return_value = None

    tool = CourseSearchTool(store)
    tool.execute(query="q", course_name="MCP", lesson_number=2)

    store.search.assert_called_once_with(query="q", course_name="MCP", lesson_number=2)


def test_execute_returns_error_from_store():
    store = MagicMock()
    store.search.return_value = SearchResults.empty("No course found matching 'Bogus'")

    tool = CourseSearchTool(store)
    out = tool.execute(query="q", course_name="Bogus")

    assert out == "No course found matching 'Bogus'"


def test_execute_handles_empty_results():
    store = MagicMock()
    store.search.return_value = _results([], [])

    tool = CourseSearchTool(store)
    out = tool.execute(query="q", course_name="MCP", lesson_number=3)

    assert "No relevant content found" in out
    assert "course 'MCP'" in out
    assert "lesson 3" in out


def test_execute_tracks_sources():
    store = MagicMock()
    store.search.return_value = _results(
        ["doc"], [{"course_title": "MCP Basics", "lesson_number": 1}]
    )
    store.get_lesson_link.return_value = "https://example.com/1"

    tool = CourseSearchTool(store)
    tool.execute(query="q")

    assert len(tool.last_sources) == 1
    assert "MCP Basics - Lesson 1" in tool.last_sources[0]
    assert "https://example.com/1" in tool.last_sources[0]


# --------------------------------------------------------------------------- #
# Integration tests: real VectorStore. These exercise the actual config.
# --------------------------------------------------------------------------- #
def test_config_max_results_is_positive():
    """The vector store is told how many results to return via MAX_RESULTS.
    If this is 0, every content search returns nothing. This test pins the
    root cause directly."""
    assert config.MAX_RESULTS > 0, (
        f"config.MAX_RESULTS is {config.MAX_RESULTS}; with 0 the vector store "
        "requests zero results and all content queries come back empty."
    )


def test_real_search_with_config_value_returns_content(make_store):
    """Build a real store using the ACTUAL configured MAX_RESULTS and confirm a
    matching content query returns results. Fails while MAX_RESULTS == 0."""
    store = make_store(config.MAX_RESULTS)
    tool = CourseSearchTool(store)

    out = tool.execute(query="What is MCP?")

    assert "No relevant content found" not in out
    assert out.strip() != ""
    assert tool.last_sources, "expected at least one source from a real search"


def test_real_search_with_positive_max_results_works(make_store):
    """Control: with a sane max_results the same query clearly returns content,
    proving the tool + store are correct and the only variable is the config."""
    store = make_store(5)
    tool = CourseSearchTool(store)

    out = tool.execute(query="What is MCP?")

    assert "MCP" in out
    assert tool.last_sources
