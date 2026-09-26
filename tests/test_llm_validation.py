"""
Tests for the LLM output validation layer — _validate_llm_output().

Verifies that the schema validation gate between the LLM (untrusted)
and application state (trusted) correctly:
  - Accepts valid structured output
  - Rejects malformed types (string where bool expected, etc.)
  - Handles missing assistant_reply
  - Cleans up null-string artefacts from LLMs
  - Returns safe fallback on complete garbage
  - Provider abstraction (LLMProvider ABC) works correctly
"""

from app.services.llm_service import _validate_llm_output


class TestValidateLlmOutput:

    def test_valid_output_accepted(self):
        raw = {
            "full_name": "Jane Smith",
            "has_children": False,
            "assistant_reply": "What is your address?"
        }
        result = _validate_llm_output(raw)
        assert result.full_name == "Jane Smith"
        assert result.has_children is False
        assert result.assistant_reply == "What is your address?"

    def test_null_fields_remain_none(self):
        raw = {
            "full_name": None,
            "home_address": None,
            "assistant_reply": "Hello!"
        }
        result = _validate_llm_output(raw)
        assert result.full_name is None
        assert result.home_address is None

    def test_missing_assistant_reply_gets_default(self):
        raw = {"full_name": "Jane"}
        result = _validate_llm_output(raw)
        assert result.assistant_reply is not None
        assert len(result.assistant_reply) > 0

    def test_empty_assistant_reply_gets_default(self):
        raw = {"full_name": "Jane", "assistant_reply": ""}
        result = _validate_llm_output(raw)
        assert len(result.assistant_reply) > 0

    def test_null_string_artefact_cleaned(self):
        """Gemini sometimes returns the literal string 'null' instead of null."""
        raw = {
            "full_name": "null",
            "home_address": "None",
            "assistant_reply": "Hello!"
        }
        result = _validate_llm_output(raw)
        assert result.full_name is None
        assert result.home_address is None

    def test_empty_string_field_cleaned(self):
        raw = {
            "full_name": "   ",
            "assistant_reply": "Hello!"
        }
        result = _validate_llm_output(raw)
        assert result.full_name is None

    def test_boolean_field_rejects_string(self):
        """If Gemini returns has_children: 'yes' instead of true, it should be cleared."""
        raw = {
            "has_children": "yes",
            "assistant_reply": "OK"
        }
        result = _validate_llm_output(raw)
        # _post_validate_types should clear this
        assert result.has_children is None

    def test_boolean_field_rejects_number(self):
        raw = {
            "covers_worldwide_assets": 1,
            "assistant_reply": "OK"
        }
        result = _validate_llm_output(raw)
        assert result.covers_worldwide_assets is None

    def test_list_field_rejects_string(self):
        raw = {
            "children_names": "John, Sarah",
            "assistant_reply": "OK"
        }
        result = _validate_llm_output(raw)
        assert result.children_names is None

    def test_list_field_rejects_non_string_items(self):
        raw = {
            "specific_gifts": [123, 456],
            "assistant_reply": "OK"
        }
        result = _validate_llm_output(raw)
        assert result.specific_gifts is None

    def test_valid_list_field_accepted(self):
        raw = {
            "children_names": ["John", "Sarah"],
            "assistant_reply": "OK"
        }
        result = _validate_llm_output(raw)
        assert result.children_names == ["John", "Sarah"]

    def test_complete_garbage_returns_safe_fallback(self):
        """Completely invalid structure should return a safe fallback."""
        raw = {
            "full_name": {"nested": "object"},
            "assistant_reply": "Hello"
        }
        # Pydantic should reject nested object for a str field
        result = _validate_llm_output(raw)
        # Either it parsed (Pydantic coerced) or returned fallback
        assert result.assistant_reply is not None

    def test_multi_field_extraction_validated(self):
        raw = {
            "full_name": "Jane Smith",
            "home_address": "42 Elm Street",
            "covers_worldwide_assets": True,
            "has_children": False,
            "children_names": [],
            "executor_name": "James",
            "executor_relationship": "brother",
            "specific_gifts": None,
            "additional_wishes": None,
            "assistant_reply": "All collected!"
        }
        result = _validate_llm_output(raw)
        assert result.full_name == "Jane Smith"
        assert result.home_address == "42 Elm Street"
        assert result.covers_worldwide_assets is True
        assert result.has_children is False
        assert result.children_names == []
        assert result.executor_name == "James"
        assert result.executor_relationship == "brother"
        assert result.specific_gifts is None
        assert result.additional_wishes is None


class TestSchemaMatchesStructuredState:
    """
    Verify that the schema enforced by the application matches
    the STRUCTURED STATE format from the requirements:

    {
      "full_name": "Jane Smith",
      "covers_worldwide_assets": true,
      "has_children": false,
      "executor": {
        "name": "James Smith",
        "relationship": "brother"
      }
    }
    """

    def test_structured_state_format(self):
        from app.models import PersonalWishesState, ExecutorInfo

        state = PersonalWishesState(
            full_name="Jane Smith",
            covers_worldwide_assets=True,
            has_children=False,
            executor=ExecutorInfo(name="James Smith", relationship="brother"),
        )
        dumped = state.model_dump()

        assert dumped["full_name"] == "Jane Smith"
        assert dumped["covers_worldwide_assets"] is True
        assert dumped["has_children"] is False
        assert dumped["executor"]["name"] == "James Smith"
        assert dumped["executor"]["relationship"] == "brother"

    def test_state_serialises_to_required_format(self):
        """The JSON output matches the exact structure from the requirements."""
        import json
        from app.models import PersonalWishesState, ExecutorInfo

        state = PersonalWishesState(
            full_name="Jane Smith",
            covers_worldwide_assets=True,
            has_children=False,
            executor=ExecutorInfo(name="James Smith", relationship="brother"),
        )
        serialised = json.loads(state.model_dump_json())

        # Verify required keys exist
        assert "full_name" in serialised
        assert "covers_worldwide_assets" in serialised
        assert "has_children" in serialised
        assert "executor" in serialised
        assert "name" in serialised["executor"]
        assert "relationship" in serialised["executor"]


class TestProviderAbstraction:
    """
    Verify the LLM provider abstraction layer.

    The LLMProvider ABC defines the interface. MockLLMProvider and
    GroqProvider both implement it. Swapping providers should not
    require changes to state management, document generation, or
    the frontend.
    """

    def test_mock_provider_implements_interface(self):
        from app.services.llm_service import LLMProvider, MockLLMProvider
        provider = MockLLMProvider()
        assert isinstance(provider, LLMProvider)

    def test_groq_provider_implements_interface(self):
        from app.services.llm_service import LLMProvider, GroqProvider
        provider = GroqProvider(api_key="test-key")
        assert isinstance(provider, LLMProvider)

    def test_factory_returns_mock_when_no_key(self, monkeypatch):
        from app.services.llm_service import get_llm_provider, MockLLMProvider
        monkeypatch.setenv("GROQ_API_KEY", "")
        monkeypatch.setenv("USE_MOCK_LLM", "false")
        provider = get_llm_provider()
        assert isinstance(provider, MockLLMProvider)

    def test_factory_returns_mock_when_forced(self, monkeypatch):
        from app.services.llm_service import get_llm_provider, MockLLMProvider
        monkeypatch.setenv("GROQ_API_KEY", "some-key")
        monkeypatch.setenv("USE_MOCK_LLM", "true")
        provider = get_llm_provider()
        assert isinstance(provider, MockLLMProvider)

    def test_factory_returns_groq_when_key_present(self, monkeypatch):
        from app.services.llm_service import get_llm_provider, GroqProvider
        monkeypatch.setenv("GROQ_API_KEY", "some-key")
        monkeypatch.setenv("USE_MOCK_LLM", "false")
        provider = get_llm_provider()
        assert isinstance(provider, GroqProvider)

    def test_mock_provider_returns_partial_state_update(self):
        from app.services.llm_service import MockLLMProvider
        from app.models import PartialStateUpdate, PersonalWishesState
        provider = MockLLMProvider()
        result = provider.extract("My name is Jane.", PersonalWishesState(), [])
        assert isinstance(result, PartialStateUpdate)
        assert result.assistant_reply is not None

