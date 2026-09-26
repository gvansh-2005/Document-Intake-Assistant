"""
Tests for Pydantic domain models.

Verifies:
  - Schema validation (valid, invalid, edge-case inputs)
  - Correct default semantics (None vs. empty list)
  - Rejection of malformed LLM outputs
  - Serialisation round-trips
"""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.models import (
    ChatRequest,
    ChatResponse,
    ConversationStatus,
    ExecutorInfo,
    PartialStateUpdate,
    PendingConfirmation,
    PersonalWishesState,
)

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# PersonalWishesState
# ---------------------------------------------------------------------------

class TestPersonalWishesState:

    def test_default_state_is_fully_unknown(self):
        """A fresh state should have all fields as None (unknown)."""
        state = PersonalWishesState()
        assert state.full_name is None
        assert state.home_address is None
        assert state.covers_worldwide_assets is None
        assert state.has_children is None
        assert state.children_names is None  # None = not yet asked
        assert state.executor is None
        assert state.specific_gifts is None  # None = not yet asked
        assert state.additional_wishes is None

    def test_none_vs_empty_list_semantics(self):
        """None means 'not asked yet'; [] means 'confirmed empty'."""
        # Not asked
        state_a = PersonalWishesState(children_names=None, specific_gifts=None)
        assert state_a.children_names is None
        assert state_a.specific_gifts is None

        # Confirmed empty
        state_b = PersonalWishesState(children_names=[], specific_gifts=[])
        assert state_b.children_names == []
        assert state_b.specific_gifts == []

    def test_full_state_construction(self):
        """A fully populated state should round-trip cleanly."""
        state = PersonalWishesState(
            full_name="Jane Doe",
            home_address="42 Elm Street",
            covers_worldwide_assets=True,
            has_children=True,
            children_names=["John", "Sarah"],
            executor=ExecutorInfo(name="James", relationship="brother"),
            specific_gifts=["Car to John"],
            additional_wishes="Scatter ashes at sea",
        )
        data = state.model_dump()
        rebuilt = PersonalWishesState(**data)
        assert rebuilt == state

    def test_serialisation_round_trip(self):
        """JSON serialisation → deserialisation preserves all values."""
        state = PersonalWishesState(
            full_name="Test User",
            executor=ExecutorInfo(name="Bob"),
        )
        json_str = state.model_dump_json()
        rebuilt = PersonalWishesState.model_validate_json(json_str)
        assert rebuilt == state


# ---------------------------------------------------------------------------
# PartialStateUpdate
# ---------------------------------------------------------------------------

class TestPartialStateUpdate:

    def test_minimal_update(self):
        """An update with only assistant_reply and no extractions is valid."""
        update = PartialStateUpdate(assistant_reply="Could you provide your name?")
        assert update.full_name is None
        assert update.assistant_reply == "Could you provide your name?"

    def test_assistant_reply_is_required(self):
        """PartialStateUpdate must always include assistant_reply."""
        with pytest.raises(ValidationError):
            PartialStateUpdate()

    def test_multi_field_update(self):
        """Multiple fields can be extracted in a single update."""
        update = PartialStateUpdate(
            full_name="Jane Smith",
            home_address="London",
            has_children=False,
            assistant_reply="Got it!",
        )
        assert update.full_name == "Jane Smith"
        assert update.home_address == "London"
        assert update.has_children is False

    def test_malformed_fixture(self):
        """Loading the malformed fixture should cause a Pydantic ValidationError."""
        fixture = json.loads((FIXTURES / "malformed.json").read_text())
        raw = fixture["llm_response_raw"]
        with pytest.raises(ValidationError):
            PartialStateUpdate(**raw)

    def test_valid_fixture_loads(self):
        """The valid fixture should parse cleanly into a PartialStateUpdate."""
        fixture = json.loads((FIXTURES / "valid.json").read_text())
        update = PartialStateUpdate(**fixture["llm_response"])
        assert update.full_name == "Jane Smith"
        assert update.has_children is False


# ---------------------------------------------------------------------------
# API Contracts
# ---------------------------------------------------------------------------

class TestChatRequest:

    def test_requires_session_id_and_message(self):
        with pytest.raises(ValidationError):
            ChatRequest()

    def test_rejects_empty_message(self):
        with pytest.raises(ValidationError):
            ChatRequest(session_id="abc", message="")

    def test_valid_request(self):
        req = ChatRequest(session_id="abc123", message="Hello")
        assert req.session_id == "abc123"


class TestChatResponse:

    def test_response_construction(self):
        resp = ChatResponse(
            session_id="abc",
            assistant_reply="Hi!",
            state=PersonalWishesState(),
            draft_document="Draft",
            status=ConversationStatus.COLLECTING,
        )
        assert resp.status == ConversationStatus.COLLECTING
        assert resp.pending_confirmation is None
