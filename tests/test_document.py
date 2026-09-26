"""
Tests for the Document Service — deterministic draft generation.

Verifies:
  - Document always includes fictional disclaimer
  - Unknown fields render as [UNKNOWN / UNCONFIRMED]
  - Known fields render correctly
  - None vs [] semantics are reflected in the document text
  - Document is consistent with state (not with conversation)
"""

from app.models import ExecutorInfo, PersonalWishesState
from app.services.document_service import generate_draft_document


class TestDocumentDisclaimer:

    def test_disclaimer_always_present(self):
        doc = generate_draft_document(PersonalWishesState())
        assert "FICTIONAL DOCUMENT" in doc
        assert "NOT LEGAL ADVICE" in doc

    def test_draft_status_footer(self):
        doc = generate_draft_document(PersonalWishesState())
        assert "DRAFT PREVIEW" in doc


class TestDocumentUnknownFields:

    def test_empty_state_shows_unknown_markers(self):
        doc = generate_draft_document(PersonalWishesState())
        assert "[UNKNOWN / UNCONFIRMED]" in doc
        assert "Children Status: [UNKNOWN / UNCONFIRMED]" in doc

    def test_gifts_none_shows_not_yet_discussed(self):
        state = PersonalWishesState(specific_gifts=None)
        doc = generate_draft_document(state)
        assert "[Not yet discussed]" in doc

    def test_gifts_empty_shows_none_specified(self):
        state = PersonalWishesState(specific_gifts=[])
        doc = generate_draft_document(state)
        assert "None specified" in doc

    def test_wishes_none_shows_not_yet_discussed(self):
        state = PersonalWishesState(additional_wishes=None)
        doc = generate_draft_document(state)
        assert "[Not yet discussed]" in doc

    def test_wishes_empty_shows_none_specified(self):
        state = PersonalWishesState(additional_wishes="")
        doc = generate_draft_document(state)
        assert "None specified" in doc


class TestDocumentPopulatedFields:

    def test_name_appears_in_document(self):
        state = PersonalWishesState(full_name="Jane Doe")
        doc = generate_draft_document(state)
        assert "Jane Doe" in doc

    def test_worldwide_yes(self):
        state = PersonalWishesState(covers_worldwide_assets=True)
        doc = generate_draft_document(state)
        assert "Yes (Applies globally)" in doc

    def test_worldwide_no(self):
        state = PersonalWishesState(covers_worldwide_assets=False)
        doc = generate_draft_document(state)
        assert "No (Specific jurisdiction only)" in doc

    def test_children_with_names(self):
        state = PersonalWishesState(
            has_children=True,
            children_names=["John", "Sarah"],
        )
        doc = generate_draft_document(state)
        assert "Children: Yes" in doc
        assert "John, Sarah" in doc

    def test_children_pending_names(self):
        state = PersonalWishesState(has_children=True, children_names=[])
        doc = generate_draft_document(state)
        assert "[Names pending confirmation]" in doc

    def test_no_children(self):
        state = PersonalWishesState(has_children=False)
        doc = generate_draft_document(state)
        assert "None declared" in doc

    def test_executor_with_relationship(self):
        state = PersonalWishesState(
            executor=ExecutorInfo(name="James", relationship="brother"),
        )
        doc = generate_draft_document(state)
        assert "James (brother)" in doc

    def test_executor_without_relationship(self):
        state = PersonalWishesState(
            executor=ExecutorInfo(name="James", relationship=None),
        )
        doc = generate_draft_document(state)
        assert "James" in doc

    def test_specific_gifts_listed(self):
        state = PersonalWishesState(specific_gifts=["Car to John", "£10,000 to Sarah"])
        doc = generate_draft_document(state)
        assert "Car to John" in doc
        assert "£10,000 to Sarah" in doc

    def test_additional_wishes_rendered(self):
        state = PersonalWishesState(additional_wishes="Scatter ashes at sea")
        doc = generate_draft_document(state)
        assert "Scatter ashes at sea" in doc


class TestDocumentConsistency:

    def test_document_reflects_state_not_conversation(self):
        """
        The document should be generated purely from state.
        Given a state with full_name='Jane' but no address,
        the document must show Jane and [UNKNOWN] for address.
        """
        state = PersonalWishesState(full_name="Jane", home_address=None)
        doc = generate_draft_document(state)
        assert "Jane" in doc
        assert "[UNKNOWN / UNCONFIRMED]" in doc
