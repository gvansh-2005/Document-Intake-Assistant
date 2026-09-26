"""
Tests for the FastAPI API endpoints — integration tests.

Covers:
  - Health check
  - Session creation
  - Chat endpoint with mock LLM
  - Multi-turn API flow
  - Error handling (missing session_id, empty message)
  - Response schema validation
  - State ownership verification (backend-owned)
"""

import os

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(autouse=True)
def force_mock(monkeypatch):
    """Ensure mock LLM is used for all API tests."""
    monkeypatch.setenv("USE_MOCK_LLM", "true")


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class TestHealthEndpoint:

    def test_health_returns_200(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

class TestSessionEndpoint:

    def test_create_session(self, client):
        resp = client.post("/api/session")
        assert resp.status_code == 200
        data = resp.json()
        assert "session_id" in data
        assert len(data["session_id"]) > 0


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

class TestChatEndpoint:

    def test_basic_chat(self, client):
        # Create session
        session = client.post("/api/session").json()
        sid = session["session_id"]

        # Send message
        resp = client.post("/api/chat", json={
            "session_id": sid,
            "message": "My name is Jane Smith.",
        })
        assert resp.status_code == 200
        data = resp.json()

        assert data["session_id"] == sid
        assert data["state"]["full_name"] == "Jane Smith"
        assert "FICTIONAL DOCUMENT" in data["draft_document"]
        assert len(data["assistant_reply"]) > 0

    def test_multi_turn_flow(self, client):
        session = client.post("/api/session").json()
        sid = session["session_id"]

        # Turn 1: name
        r1 = client.post("/api/chat", json={
            "session_id": sid, "message": "My name is Jane Smith."
        }).json()
        assert r1["state"]["full_name"] == "Jane Smith"

        # Turn 2: address — name should persist
        r2 = client.post("/api/chat", json={
            "session_id": sid, "message": "I live at 42 Elm Street, London."
        }).json()
        assert r2["state"]["full_name"] == "Jane Smith"  # preserved
        assert r2["state"]["home_address"] is not None

    def test_state_owned_by_backend(self, client):
        """
        The client sends only session_id + message.
        The backend should maintain state independently.
        """
        session = client.post("/api/session").json()
        sid = session["session_id"]

        client.post("/api/chat", json={
            "session_id": sid, "message": "My name is Jane Smith."
        })

        # Second request — no state sent — backend should still know the name
        r2 = client.post("/api/chat", json={
            "session_id": sid, "message": "I live at London."
        })
        assert r2.json()["state"]["full_name"] == "Jane Smith"

    def test_multiple_fields_one_message(self, client):
        session = client.post("/api/session").json()
        sid = session["session_id"]

        resp = client.post("/api/chat", json={
            "session_id": sid,
            "message": "My name is Jane Smith and I don't have children.",
        }).json()
        assert resp["state"]["full_name"] == "Jane Smith"
        assert resp["state"]["has_children"] is False

    def test_response_includes_status(self, client):
        session = client.post("/api/session").json()
        sid = session["session_id"]

        resp = client.post("/api/chat", json={
            "session_id": sid, "message": "Hello"
        }).json()
        assert "status" in resp
        assert resp["status"] in ("collecting", "needs_clarification",
                                   "needs_confirmation", "complete", "error")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestChatValidation:

    def test_missing_message_returns_422(self, client):
        resp = client.post("/api/chat", json={"session_id": "abc"})
        assert resp.status_code == 422

    def test_empty_message_returns_422(self, client):
        resp = client.post("/api/chat", json={"session_id": "abc", "message": ""})
        assert resp.status_code == 422

    def test_missing_session_id_returns_422(self, client):
        resp = client.post("/api/chat", json={"message": "hello"})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------

class TestFrontend:

    def test_index_returns_html(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "Document Intake Assistant" in resp.text
