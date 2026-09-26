"""
Tests for the State Service — merge logic, contradiction detection, missing fields.

Covers:
  - Basic field merging
  - Multi-field updates in one pass
  - Corrections (overwriting existing values)
  - Contradiction detection
  - Children logic (has_children ↔ children_names consistency)
  - Executor partial updates
  - Missing field detection
  - Completeness calculation
  - Fixture-driven scenarios
"""

import json
from pathlib import Path

import pytest

from app.models import (
    ConversationStatus,
    ExecutorInfo,
    PartialStateUpdate,
    PersonalWishesState,
)
from app.services.state_service import (
    compute_status,
    get_missing_fields,
    merge_state_update,
)

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Basic merge
# ---------------------------------------------------------------------------

class TestMergeBasic:

    def test_merge_single_field(self):
        state = PersonalWishesState()
        update = PartialStateUpdate(full_name="Jane Smith", assistant_reply="Thanks!")
        new_state, pending = merge_state_update(state, update)
        assert new_state.full_name == "Jane Smith"
        assert pending is None

    def test_merge_preserves_existing_fields(self):
        state = PersonalWishesState(full_name="Jane", home_address="London")
        update = PartialStateUpdate(
            covers_worldwide_assets=True,
            assistant_reply="Noted.",
        )
        new_state, _ = merge_state_update(state, update)
        assert new_state.full_name == "Jane"
        assert new_state.home_address == "London"
        assert new_state.covers_worldwide_assets is True

    def test_merge_multiple_fields(self):
        """From valid fixture: multiple fields in one message."""
        fixture = json.loads((FIXTURES / "valid.json").read_text())
        state = PersonalWishesState()
        update = PartialStateUpdate(**fixture["llm_response"])
        new_state, _ = merge_state_update(state, update)

        assert new_state.full_name == "Jane Smith"
        assert new_state.home_address == "42 Elm Street, London"
        assert new_state.has_children is False
        assert new_state.children_names == []

    def test_null_update_fields_are_ignored(self):
        state = PersonalWishesState(full_name="Jane")
        update = PartialStateUpdate(
            full_name=None,  # should not clear the name
            assistant_reply="OK",
        )
        new_state, _ = merge_state_update(state, update)
        assert new_state.full_name == "Jane"


# ---------------------------------------------------------------------------
# Corrections & Contradictions
# ---------------------------------------------------------------------------

class TestCorrections:

    def test_name_correction(self):
        """From correction fixture: user changes name from Jane Smith to Jane Doe."""
        fixture = json.loads((FIXTURES / "correction.json").read_text())
        initial = PersonalWishesState(**fixture["initial_state"])
        update = PartialStateUpdate(**fixture["llm_response"])
        new_state, pending = merge_state_update(initial, update)

        assert new_state.full_name == "Jane Doe"
        assert new_state.home_address == "42 Elm Street"  # preserved
        assert pending is not None
        assert pending.field == "full_name"
        assert pending.old_value == "Jane Smith"
        assert pending.new_value == "Jane Doe"

    def test_executor_correction_triggers_contradiction(self):
        state = PersonalWishesState(
            executor=ExecutorInfo(name="James", relationship="brother"),
        )
        update = PartialStateUpdate(
            executor_name="John",
            assistant_reply="Updated.",
        )
        new_state, pending = merge_state_update(state, update)
        assert new_state.executor.name == "John"
        assert pending is not None
        assert pending.field == "executor.name"

    def test_no_contradiction_on_first_entry(self):
        state = PersonalWishesState()
        update = PartialStateUpdate(full_name="Jane", assistant_reply="Hi!")
        _, pending = merge_state_update(state, update)
        assert pending is None

    def test_worldwide_correction(self):
        state = PersonalWishesState(covers_worldwide_assets=True)
        update = PartialStateUpdate(
            covers_worldwide_assets=False,
            assistant_reply="Updated.",
        )
        new_state, pending = merge_state_update(state, update)
        assert new_state.covers_worldwide_assets is False
        assert pending is not None
        assert pending.field == "covers_worldwide_assets"


# ---------------------------------------------------------------------------
# Children logic
# ---------------------------------------------------------------------------

class TestChildrenLogic:

    def test_has_children_false_clears_names(self):
        state = PersonalWishesState(
            has_children=True,
            children_names=["John", "Sarah"],
        )
        update = PartialStateUpdate(has_children=False, assistant_reply="OK")
        new_state, _ = merge_state_update(state, update)
        assert new_state.has_children is False
        assert new_state.children_names == []

    def test_children_names_implies_has_children(self):
        state = PersonalWishesState()
        update = PartialStateUpdate(
            children_names=["Alice", "Bob"],
            assistant_reply="Noted.",
        )
        new_state, _ = merge_state_update(state, update)
        assert new_state.has_children is True
        assert new_state.children_names == ["Alice", "Bob"]

    def test_has_children_true_without_names(self):
        state = PersonalWishesState()
        update = PartialStateUpdate(has_children=True, assistant_reply="Names?")
        new_state, _ = merge_state_update(state, update)
        assert new_state.has_children is True
        assert new_state.children_names is None  # still need names


# ---------------------------------------------------------------------------
# Executor logic
# ---------------------------------------------------------------------------

class TestExecutorLogic:

    def test_ambiguous_executor_name_only(self):
        """From ambiguous fixture: executor name without relationship."""
        fixture = json.loads((FIXTURES / "ambiguous.json").read_text())
        state = PersonalWishesState()
        update = PartialStateUpdate(**fixture["llm_response"])
        new_state, _ = merge_state_update(state, update)

        assert new_state.executor is not None
        assert new_state.executor.name == "James"
        assert new_state.executor.relationship is None

    def test_executor_relationship_added_later(self):
        state = PersonalWishesState(
            executor=ExecutorInfo(name="James", relationship=None),
        )
        update = PartialStateUpdate(
            executor_relationship="brother",
            assistant_reply="Thanks!",
        )
        new_state, _ = merge_state_update(state, update)
        assert new_state.executor.name == "James"
        assert new_state.executor.relationship == "brother"


# ---------------------------------------------------------------------------
# Missing fields & completeness
# ---------------------------------------------------------------------------

class TestMissingFields:

    def test_empty_state_all_missing(self):
        state = PersonalWishesState()
        missing = get_missing_fields(state)
        assert "full_name" in missing
        assert "home_address" in missing
        assert "covers_worldwide_assets" in missing
        assert "has_children" in missing
        assert "executor_name" in missing

    def test_complete_state_no_required_missing(self):
        state = PersonalWishesState(
            full_name="Jane",
            home_address="London",
            covers_worldwide_assets=True,
            has_children=False,
            children_names=[],
            executor=ExecutorInfo(name="James", relationship="brother"),
            specific_gifts=[],
            additional_wishes="",
        )
        missing = get_missing_fields(state)
        # All required are filled
        required_missing = [
            f for f in missing
            if f in ("full_name", "home_address", "covers_worldwide_assets",
                     "has_children", "executor_name")
        ]
        assert len(required_missing) == 0

    def test_children_names_missing_when_has_children(self):
        state = PersonalWishesState(has_children=True)
        missing = get_missing_fields(state)
        assert "children_names" in missing

    def test_children_names_not_missing_when_no_children(self):
        state = PersonalWishesState(has_children=False, children_names=[])
        missing = get_missing_fields(state)
        assert "children_names" not in missing

    def test_executor_relationship_missing(self):
        state = PersonalWishesState(
            executor=ExecutorInfo(name="James", relationship=None),
        )
        missing = get_missing_fields(state)
        assert "executor_relationship" in missing


class TestComputeStatus:

    def test_empty_state_is_collecting(self):
        assert compute_status(PersonalWishesState()) == ConversationStatus.COLLECTING

    def test_complete_state_is_complete(self):
        state = PersonalWishesState(
            full_name="Jane",
            home_address="London",
            covers_worldwide_assets=True,
            has_children=False,
            children_names=[],
            executor=ExecutorInfo(name="James", relationship="brother"),
            specific_gifts=[],
            additional_wishes="",
        )
        assert compute_status(state) == ConversationStatus.COMPLETE


# ---------------------------------------------------------------------------
# Fixture-driven: contradiction and multi-field scenarios
# ---------------------------------------------------------------------------

class TestContradictionFixture:

    def test_children_contradiction_detected(self):
        """From contradiction fixture: user changes has_children from false to true."""
        fixture = json.loads((FIXTURES / "contradiction.json").read_text())
        initial = PersonalWishesState(**fixture["initial_state"])
        update = PartialStateUpdate(**fixture["llm_response"])
        new_state, pending = merge_state_update(initial, update)

        assert new_state.has_children is True
        assert new_state.children_names == ["John", "Sarah"]
        assert pending is not None
        assert pending.field == "has_children"

    def test_children_contradiction_preserves_other_fields(self):
        """Other fields should remain unchanged during a contradiction."""
        fixture = json.loads((FIXTURES / "contradiction.json").read_text())
        initial = PersonalWishesState(**fixture["initial_state"])
        update = PartialStateUpdate(**fixture["llm_response"])
        new_state, _ = merge_state_update(initial, update)

        assert new_state.full_name == "Jane Smith"
        assert new_state.home_address == "42 Elm Street"


class TestMultiFieldFixture:

    def test_multi_field_extraction(self):
        """From multi-field fixture: multiple fields in one message."""
        fixture = json.loads((FIXTURES / "multi-field.json").read_text())
        state = PersonalWishesState()
        update = PartialStateUpdate(**fixture["llm_response"])
        new_state, _ = merge_state_update(state, update)

        assert new_state.full_name == "Nishant Shukla"
        assert new_state.home_address == "Mumbai"
        assert new_state.has_children is True
        assert new_state.children_names == ["Rahul", "Ananya"]
        assert new_state.executor is not None
        assert new_state.executor.name == "James"
        assert new_state.executor.relationship == "brother"

