"""HTTP-level tests for the FastAPI endpoints (backend/app.py).

The production ``app.py`` mounts ``StaticFiles(directory="../frontend")`` at
import time, which does not exist in the test environment. Rather than import
that module, these tests run against an equivalent inline app (see the
``api_app``/``client`` fixtures in ``conftest.py``) wired to a mocked
``RAGSystem``. That keeps the request/response contract under test while
avoiding filesystem, ChromaDB, and Anthropic dependencies.
"""
import pytest

pytestmark = pytest.mark.api


# --------------------------------------------------------------------------- #
# POST /api/query
# --------------------------------------------------------------------------- #
class TestQueryEndpoint:
    def test_query_with_session_id_returns_answer_and_sources(self, client, mock_rag_system):
        resp = client.post(
            "/api/query",
            json={"query": "What is MCP?", "session_id": "existing-session"},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["answer"] == "MCP is the Model Context Protocol."
        assert body["sources"] == ["Test Course: MCP Basics - Lesson 1"]
        # An explicit session id is passed straight through, not regenerated.
        assert body["session_id"] == "existing-session"
        mock_rag_system.query.assert_called_once_with("What is MCP?", "existing-session")
        mock_rag_system.session_manager.create_session.assert_not_called()

    def test_query_without_session_id_creates_one(self, client, mock_rag_system):
        resp = client.post("/api/query", json={"query": "What is MCP?"})

        assert resp.status_code == 200
        body = resp.json()
        assert body["session_id"] == "session-123"
        mock_rag_system.session_manager.create_session.assert_called_once()
        mock_rag_system.query.assert_called_once_with("What is MCP?", "session-123")

    def test_query_response_shape(self, client):
        body = client.post("/api/query", json={"query": "hi"}).json()
        assert set(body.keys()) == {"answer", "sources", "session_id"}
        assert isinstance(body["sources"], list)

    def test_query_missing_query_field_is_422(self, client):
        resp = client.post("/api/query", json={"session_id": "s1"})
        assert resp.status_code == 422

    def test_query_empty_body_is_422(self, client):
        resp = client.post("/api/query", json={})
        assert resp.status_code == 422

    def test_query_rag_failure_returns_500(self, client, mock_rag_system):
        mock_rag_system.query.side_effect = RuntimeError("vector store down")
        resp = client.post("/api/query", json={"query": "boom"})
        assert resp.status_code == 500
        assert "vector store down" in resp.json()["detail"]

    def test_query_with_empty_sources(self, client, mock_rag_system):
        mock_rag_system.query.return_value = ("No sources here.", [])
        resp = client.post("/api/query", json={"query": "obscure"})
        assert resp.status_code == 200
        assert resp.json()["sources"] == []


# --------------------------------------------------------------------------- #
# GET /api/courses
# --------------------------------------------------------------------------- #
class TestCoursesEndpoint:
    def test_courses_returns_stats(self, client):
        resp = client.get("/api/courses")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total_courses"] == 1
        assert body["course_titles"] == ["Test Course: MCP Basics"]

    def test_courses_response_shape(self, client):
        body = client.get("/api/courses").json()
        assert set(body.keys()) == {"total_courses", "course_titles"}
        assert isinstance(body["course_titles"], list)

    def test_courses_empty_catalog(self, client, mock_rag_system):
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": [],
        }
        body = client.get("/api/courses").json()
        assert body["total_courses"] == 0
        assert body["course_titles"] == []

    def test_courses_failure_returns_500(self, client, mock_rag_system):
        mock_rag_system.get_course_analytics.side_effect = RuntimeError("no db")
        resp = client.get("/api/courses")
        assert resp.status_code == 500
        assert "no db" in resp.json()["detail"]


# --------------------------------------------------------------------------- #
# POST /api/sessions/clear
# --------------------------------------------------------------------------- #
class TestClearSessionEndpoint:
    def test_clear_session_deletes_history(self, client, mock_rag_system):
        resp = client.post("/api/sessions/clear", json={"query": "", "session_id": "s1"})
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}
        mock_rag_system.session_manager.delete_session.assert_called_once_with("s1")

    def test_clear_session_without_id_is_noop(self, client, mock_rag_system):
        resp = client.post("/api/sessions/clear", json={"query": ""})
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}
        mock_rag_system.session_manager.delete_session.assert_not_called()


# --------------------------------------------------------------------------- #
# GET /  (frontend root stand-in)
# --------------------------------------------------------------------------- #
class TestRootEndpoint:
    def test_root_is_reachable(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
