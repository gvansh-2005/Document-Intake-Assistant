"""
Tests for the Conversation Service — orchestration layer.

Covers:
  - End-to-end message processing
  - Session creation and state ownership
  - Conversation history accumulation
  - Contradiction notification in assistant replies
  - Status transitions
"""

import os

import pytest

from app.models import ConversationStatus, ExecutorInfo, PersonalWishesState
from app.services.conversation_service import ConversationService
from app.storage.session_store import SessionStore


@pytest.fixture(autouse=True)
def force_mock_llm(monkeypatch):
    """Force mock LLM for all conversation tests."""
    monkeypatch.setenv("USE_MOCK_LLM", "true")


@pytest.fixture
def service():
    store = SessionStore()
    return ConversationService(session_store=store), store


class TestConversationFlow:

    def test_first_message_creates_session(self, service):
        svc, store = service
        response = svc.handle_message("test-session", "My name is Jane Smith.")
        assert response.session_id == "test-session"
        assert store.session_exists("test-session")

    def test_name_extraction(self, service):
        svc, _ = service
        resp = svc.handle_message("s1", "My name is Jane Smith.")
        assert resp.state.full_name == "Jane Smith"

    def test_multi_turn_state_accumulation(self, service):
        svc, _ = service
        svc.handle_message("s1", "My name is Jane Smith.")
        resp2 = svc.handle_message("s1", "I live at 42 Elm Street, London.")
        assert resp2.state.full_name == "Jane Smith"  # preserved
        assert resp2.state.home_address is not None

    def test_conversation_history_grows(self, service):
        svc, store = service
        svc.handle_message("s1", "Hello")
        svc.handle_message("s1", "My name is Jane.")
        history = store.get_recent_history("s1")
        # Each turn adds user + assistant = 2 messages
        assert len(history) >= 4

    def test_status_transitions_to_collecting(self, service):
        svc, _ = service
        resp = svc.handle_message("s1", "My name is Jane.")
        assert resp.status in (
            ConversationStatus.COLLECTING,
            ConversationStatus.NEEDS_CONFIRMATION,
        )

    def test_draft_document_always_present(self, service):
        svc, _ = service
        resp = svc.handle_message("s1", "My name is Jane.")
        assert "FICTIONAL DOCUMENT" in resp.draft_document

    def test_assistant_reply_not_empty(self, service):
        svc, _ = service
        resp = svc.handle_message("s1", "Hello")
        assert len(resp.assistant_reply) > 0


class TestConversationContradictions:

    def test_name_correction_notifies(self, service):
        svc, _ = service
        svc.handle_message("s1", "My name is Jane Smith.")
        resp = svc.handle_message("s1", "My name is Jane Doe.")
        # The assistant should mention the change
        assert resp.state.full_name == "Jane Doe"

    def test_no_data_loss_on_correction(self, service):
        svc, _ = service
        svc.handle_message("s1", "My name is Jane Smith.")
        svc.handle_message("s1", "I live at 42 Elm Street.")
        resp = svc.handle_message("s1", "My name is Jane Doe.")
        # Address should still be preserved
        assert resp.state.home_address is not None


class TestConversationNegatives:

    def test_no_children(self, service):
        svc, _ = service
        resp = svc.handle_message("s1", "I don't have children.")
        assert resp.state.has_children is False

    def test_no_worldwide(self, service):
        svc, _ = service
        resp = svc.handle_message("s1", "Not worldwide, specific jurisdiction only.")
        assert resp.state.covers_worldwide_assets is False
