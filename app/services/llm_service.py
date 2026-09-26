"""
LLM Service — handles all LLM interaction with structured output enforcement.

Responsibilities (Single Responsibility):
  - Format prompts with current state context
  - Call Groq with structured JSON output
  - Validate LLM output against Pydantic schema BEFORE accepting it
  - Provide a robust deterministic mock for offline/test execution
  - Handle API errors gracefully without corrupting state

Provider Abstraction (Dependency Inversion):
  LLMProvider (ABC)
      ├── GroqProvider       — calls Groq Cloud API
      └── MockLLMProvider    — deterministic regex-based extraction

  Swapping providers requires only changing the factory function or
  environment variable — no changes to state management, document
  generation, or the frontend.
"""

from __future__ import annotations

import json
import logging
import os
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from pydantic import ValidationError

from app.models import PartialStateUpdate, PersonalWishesState
from app.services.state_service import get_missing_fields

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "intake_prompt.txt"


# ---------------------------------------------------------------------------
# Abstract LLM provider interface
# ---------------------------------------------------------------------------

class LLMProvider(ABC):
    """
    Abstract interface for LLM providers.

    Any concrete implementation must:
      1. Accept a user message, current state, and conversation history
      2. Return a validated PartialStateUpdate
      3. Handle its own errors gracefully (never raise to caller)

    To swap providers (e.g., MockLLMProvider → GroqProvider),
    change only the factory function or environment variable.
    No changes are required to state management, document generation,
    or the frontend.
    """

    @abstractmethod
    def extract(
        self,
        user_message: str,
        current_state: PersonalWishesState,
        conversation_history: List[dict],
    ) -> PartialStateUpdate:
        """Extract structured data from a user message."""
        ...


# ---------------------------------------------------------------------------
# Groq provider
# ---------------------------------------------------------------------------

class GroqProvider(LLMProvider):
    """
    Calls Groq Cloud API with structured JSON output, returning a validated
    PartialStateUpdate.

    Steps:
      1. Build the prompt with current state context
      2. Send to Groq with response_format=json_object
      3. Parse the raw JSON response
      4. Validate against PartialStateUpdate schema (Pydantic gate)
      5. Return the validated update OR a safe fallback on any error
    """

    def __init__(self, api_key: str, model: str = "qwen/qwen3.8-27b") -> None:
        self._api_key = api_key
        self._model = model

    def extract(
        self,
        user_message: str,
        current_state: PersonalWishesState,
        conversation_history: List[dict],
    ) -> PartialStateUpdate:
        try:
            from groq import Groq
        except ImportError:
            logger.error("groq package not installed — falling back to mock")
            return MockLLMProvider().extract(user_message, current_state, conversation_history)

        # Load and format the system prompt
        prompt_template = _PROMPT_PATH.read_text(encoding="utf-8")
        system_prompt = prompt_template.format(
            current_state=json.dumps(current_state.model_dump(), indent=2, default=str)
        )

        # Build the conversation messages for Groq (OpenAI-compatible format)
        messages = [{"role": "system", "content": system_prompt}]

        for msg in conversation_history[-8:]:
            role = msg.get("role", "user")
            messages.append({
                "role": role,
                "content": msg.get("content", ""),
            })

        # Add the current user message
        messages.append({"role": "user", "content": user_message})

        try:
            client = Groq(api_key=self._api_key)

            response = client.chat.completions.create(
                model=self._model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.2,
                max_tokens=1024,
            )

            # Parse the JSON response
            raw_text = response.choices[0].message.content.strip()
            logger.debug("Raw Groq response: %s", raw_text[:500])

            # Qwen models wrap output in <think>...</think> tags — strip them
            raw_json = self._parse_response(raw_text)
            if raw_json is None:
                return PartialStateUpdate(
                    assistant_reply=(
                        "I had trouble understanding the response format. "
                        "Could you please rephrase?"
                    )
                )

            # Gate: validate the raw JSON against our Pydantic schema
            validated = _validate_llm_output(raw_json)

            logger.info(
                "Groq extraction complete — fields: %s",
                [k for k, v in validated.model_dump().items()
                 if v is not None and k != "assistant_reply"],
            )
            return validated

        except Exception as e:
            error_str = str(e)

            # If json_object mode failed, retry without it (Qwen sometimes can't do json_object)
            if "json_validate_failed" in error_str or "Failed to generate JSON" in error_str:
                logger.warning("Groq json_object mode failed — retrying without response_format")
                try:
                    client = Groq(api_key=self._api_key)
                    response = client.chat.completions.create(
                        model=self._model,
                        messages=messages,
                        temperature=0.2,
                        max_tokens=1024,
                    )
                    raw_text = response.choices[0].message.content.strip()
                    raw_json = self._parse_response(raw_text)
                    if raw_json is not None:
                        validated = _validate_llm_output(raw_json)
                        logger.info(
                            "Groq extraction complete (fallback) — fields: %s",
                            [k for k, v in validated.model_dump().items()
                             if v is not None and k != "assistant_reply"],
                        )
                        return validated
                except Exception:
                    logger.exception("Groq fallback also failed")

            logger.exception("Groq API call failed — returning safe fallback")
            return PartialStateUpdate(
                assistant_reply=(
                    "I'm sorry, I experienced a temporary issue processing your message. "
                    "Could you please rephrase or try again?"
                )
            )

    @staticmethod
    def _parse_response(raw_text: str) -> dict | None:
        """
        Parse JSON from Groq/Qwen response, handling:
          - <think>...</think> tags (Qwen reasoning)
          - Markdown ```json ... ``` code blocks
          - Bare JSON objects
        """
        # Strip <think>...</think> blocks
        text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()

        # Try direct JSON parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try extracting from markdown code block
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try finding the first { ... } block
        brace_start = text.find("{")
        if brace_start != -1:
            # Find matching closing brace
            depth = 0
            for i in range(brace_start, len(text)):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[brace_start:i + 1])
                        except json.JSONDecodeError:
                            break

        logger.error("Could not parse JSON from Groq response: %s", text[:500])
        return None


# ---------------------------------------------------------------------------
# Deterministic mock provider
# ---------------------------------------------------------------------------

class MockLLMProvider(LLMProvider):
    """
    Deterministic mock for local testing without an API key.

    Implements the same LLMProvider interface as GroqProvider.
    Therefore, replacing MockLLMProvider with GroqProvider does not
    require changes to state management, document generation, or
    the frontend.

    Handles:
      - name extraction ("my name is ...")
      - address extraction ("I live at ..." / "my address is ...")
      - worldwide assets (yes/no/global/worldwide)
      - children (yes/no with name extraction)
      - executor with relationship
      - specific gifts
      - additional wishes
      - corrections ("actually ...", "no, ...")
      - multiple fields in one message
      - negative forms ("don't", "no", "none")
    """

    def extract(
        self,
        user_message: str,
        current_state: PersonalWishesState,
        conversation_history: List[dict],
    ) -> PartialStateUpdate:
        return _mock_llm_response(user_message, current_state)


# ---------------------------------------------------------------------------
# Provider factory
# ---------------------------------------------------------------------------

def get_llm_provider() -> LLMProvider:
    """
    Factory function that returns the appropriate LLM provider based on
    environment configuration.

    Routing:
      - USE_MOCK_LLM=true  OR  GROQ_API_KEY missing  → MockLLMProvider
      - Otherwise → GroqProvider
    """
    use_mock = os.getenv("USE_MOCK_LLM", "false").lower() == "true"
    api_key = os.getenv("GROQ_API_KEY", "").strip()

    if use_mock or not api_key:
        return MockLLMProvider()
    return GroqProvider(api_key=api_key)


# ---------------------------------------------------------------------------
# Public API (delegates to the provider)
# ---------------------------------------------------------------------------

def process_user_input(
    user_message: str,
    current_state: PersonalWishesState,
    conversation_history: List[dict],
) -> PartialStateUpdate:
    """
    Process a user message and return a structured partial state update.

    Routing:
      - USE_MOCK_LLM=true  OR  GROQ_API_KEY missing  → deterministic mock
      - Otherwise → Groq structured output call
    """
    provider = get_llm_provider()
    provider_name = type(provider).__name__
    logger.info("LLM provider: %s", provider_name)
    return provider.extract(user_message, current_state, conversation_history)


# ---------------------------------------------------------------------------
# Schema validation layer
# ---------------------------------------------------------------------------

def _validate_llm_output(raw_json: dict) -> PartialStateUpdate:
    """
    Validate and parse raw LLM JSON output against the PartialStateUpdate schema.

    This is the critical gate between the LLM (untrusted) and the application
    state (trusted). If validation fails, the raw data is rejected entirely.

    Checks performed:
      1. Pydantic type coercion and constraint enforcement
      2. assistant_reply is present and non-empty
      3. No unexpected field types (e.g., string where bool expected)
      4. List fields contain only strings
    """
    # Pre-validation: strip null-string artefacts from LLMs
    cleaned = {}
    # Fields where "" or [] is a valid "user said no" value (distinct from null = "not asked")
    preserve_empty = {"additional_wishes", "specific_gifts", "children_names", "executor_name", "executor_relationship"}
    for key, value in raw_json.items():
        if value == "null" or value == "None":
            cleaned[key] = None
        elif isinstance(value, str) and value.strip() == "" and key not in preserve_empty:
            cleaned[key] = None
        else:
            cleaned[key] = value

    # Ensure assistant_reply exists
    if "assistant_reply" not in cleaned or not cleaned.get("assistant_reply"):
        cleaned["assistant_reply"] = "Could you tell me more?"

    try:
        update = PartialStateUpdate(**cleaned)
    except ValidationError as e:
        logger.warning("LLM output failed Pydantic validation: %s", e.errors())
        logger.warning("Raw LLM output was: %s", json.dumps(raw_json, indent=2))
        # Return safe fallback — state is never corrupted
        return PartialStateUpdate(
            assistant_reply=(
                "I had a little trouble processing that. "
                "Could you please rephrase your response?"
            )
        )

    # Post-validation: verify field types match schema expectations
    _post_validate_types(update, raw_json)

    logger.debug("Validated LLM output: %s non-null fields extracted",
                 sum(1 for v in update.model_dump().values() if v is not None) - 1)
    return update


def _post_validate_types(update: PartialStateUpdate, raw: dict) -> None:
    """
    Additional type-safety checks beyond Pydantic coercion.

    For example, the LLM might return has_children: "yes" (a string) which
    Pydantic would reject, but it could also return a number. We clear
    any field whose raw type doesn't match expectations.
    """
    bool_fields = ["covers_worldwide_assets", "has_children"]
    for field in bool_fields:
        raw_val = raw.get(field)
        if raw_val is not None and not isinstance(raw_val, bool):
            logger.warning("Field '%s' had unexpected type %s (value: %r) — clearing",
                           field, type(raw_val).__name__, raw_val)
            setattr(update, field, None)

    list_fields = ["children_names", "specific_gifts"]
    for field in list_fields:
        raw_val = raw.get(field)
        if raw_val is not None and not isinstance(raw_val, list):
            logger.warning("Field '%s' had unexpected type %s — clearing",
                           field, type(raw_val).__name__)
            setattr(update, field, None)
        elif isinstance(raw_val, list):
            # Ensure all list items are strings
            if not all(isinstance(item, str) for item in raw_val):
                logger.warning("Field '%s' contains non-string items — clearing", field)
                setattr(update, field, None)


# ---------------------------------------------------------------------------
# Deterministic mock (robust keyword-based extraction)
# ---------------------------------------------------------------------------

# Mapping of question text keyed by missing field name
_FOLLOW_UP_QUESTIONS = {
    "full_name": "Could you please provide your full legal name?",
    "home_address": "Thank you! What is your current residential address?",
    "covers_worldwide_assets": (
        "Should this document cover your worldwide assets, "
        "or only those in a specific jurisdiction?"
    ),
    "has_children": "Do you have any children?",
    "children_names": "Could you share the names of your children?",
    "executor_name": (
        "Who would you like to appoint as your executor? "
        "Please also mention their relationship to you."
    ),
    "executor_relationship": (
        "What is your executor's relationship to you (e.g., brother, friend, solicitor)?"
    ),
    "specific_gifts": (
        "Are there any specific items or monetary gifts you'd like to include?"
    ),
    "additional_wishes": (
        "Finally, do you have any additional wishes or instructions to record?"
    ),
}


def _mock_llm_response(
    user_message: str,
    current_state: PersonalWishesState,
) -> PartialStateUpdate:
    """
    Deterministic mock for local testing without an API key.

    Handles:
      - name extraction ("my name is ...")
      - address extraction ("I live at ..." / "my address is ...")
      - worldwide assets (yes/no/global/worldwide)
      - children (yes/no with name extraction)
      - executor with relationship
      - specific gifts
      - additional wishes
      - corrections ("actually ...", "no, ...")
      - multiple fields in one message
      - negative forms ("don't", "no", "none")
    """
    msg = user_message.strip()
    msg_lower = msg.lower()
    update = PartialStateUpdate(assistant_reply="")

    # ── Name extraction ─────────────────────────────────────────────────
    # Pre-split at sentence/clause boundaries to isolate the name clause
    _clause_splitters = re.compile(
        r"(?:\.\s+|\s+and\s+(?:i|my|we)\s|\s+but\s+|\s+,\s*(?:i|my|we)\s)",
        re.IGNORECASE,
    )
    name_clause = _clause_splitters.split(msg)[0]  # take first clause
    name_patterns = [
        r"(?:my\s+)?name\s+is\s+(.+?)(?:\.|,|;|$)",
        r"i(?:'m|\s+am)\s+(.+?)(?:\.|,|;|$)",
        r"call\s+me\s+(.+?)(?:\.|,|;|$)",
    ]
    for pattern in name_patterns:
        match = re.search(pattern, name_clause, re.IGNORECASE)
        if match:
            extracted = match.group(1).strip().rstrip(".,;!\"'")
            # Remove any trailing conjunction fragment
            extracted = re.sub(
                r"\s+(?:and|but|or)\s*$", "", extracted, flags=re.IGNORECASE
            )
            extracted = extracted.strip()
            # Skip filler phrases
            if extracted.lower() not in ("here", "ready", "fine", "good", "ok", ""):
                update.full_name = extracted.title()
            break

    # ── Address extraction ───────────────────────────────────────────────
    addr_patterns = [
        r"(?:i\s+)?live\s+(?:at|in)\s+(.+?)(?:\.|$)",
        r"(?:my\s+)?address\s+is\s+(.+?)(?:\.|$)",
        r"(?:my\s+)?residence\s+is\s+(.+?)(?:\.|$)",
        r"(?:i\s+)?reside\s+(?:at|in)\s+(.+?)(?:\.|$)",
    ]
    for pattern in addr_patterns:
        match = re.search(pattern, msg, re.IGNORECASE)
        if match:
            update.home_address = match.group(1).strip().rstrip(".,;!")
            break

    # ── Worldwide assets ─────────────────────────────────────────────────
    neg_worldwide = re.search(
        r"(not?\s+(?:worldwide|global)|don'?t.*(?:worldwide|global)|"
        r"specific\s+jurisdiction|only\s+(?:in|for)|no(?:,?\s+(?:it|this))?.*worldwide)",
        msg_lower,
    )
    pos_worldwide = re.search(
        r"(worldwide|global(?:ly)?|all\s+(?:my\s+)?assets|everywhere|yes.*worldwide)",
        msg_lower,
    )
    if neg_worldwide:
        update.covers_worldwide_assets = False
    elif pos_worldwide:
        update.covers_worldwide_assets = True

    # ── Children ─────────────────────────────────────────────────────────
    neg_children = re.search(
        r"(no\s+(?:children|kids)|don'?t\s+have\s+(?:any\s+)?(?:children|kids)|"
        r"no,?\s+i\s+don'?t|childless|without\s+children)",
        msg_lower,
    )
    pos_children = re.search(
        r"((?:i\s+)?have\s+(?:\d+\s+)?(?:children|kids)|"
        r"my\s+(?:children|kids)(?:\s+are)?|"
        r"(?:children|kids)\s+(?:named|are|called))",
        msg_lower,
    )
    if neg_children:
        update.has_children = False
        update.children_names = []
    elif pos_children:
        update.has_children = True
        # Try to extract names after keywords
        names_match = re.search(
            r"(?:named|are|called|:)\s+(.+?)(?:\.|$)", msg, re.IGNORECASE
        )
        if names_match:
            raw = names_match.group(1)
            names = [n.strip().title() for n in re.split(r"[,]|\band\b", raw) if n.strip()]
            if names:
                update.children_names = names

    # ── Executor ─────────────────────────────────────────────────────────
    relationship_words = {
        "brother", "sister", "friend", "wife", "husband", "spouse", "partner",
        "solicitor", "lawyer", "attorney", "colleague", "father", "mother",
        "son", "daughter", "uncle", "aunt", "nephew", "niece", "cousin",
    }
    exec_match = re.search(
        r"(?:executor|representative)\s+(?:is|should\s+be|will\s+be)\s+(?:my\s+)?(.+?)(?:\.|$)",
        msg, re.IGNORECASE,
    )
    rel_match = re.search(
        r"my\s+(\w+)\s+(\w+)", msg, re.IGNORECASE,
    )

    if exec_match:
        parts = exec_match.group(1).strip().split()
        for part in parts:
            if part.lower().rstrip(".,;!") in relationship_words:
                update.executor_relationship = part.lower().rstrip(".,;!")
            else:
                if not update.executor_name:
                    update.executor_name = part.strip(".,;!").title()
    elif rel_match:
        rel_word = rel_match.group(1).lower()
        name_word = rel_match.group(2).strip(".,;!")
        if rel_word in relationship_words:
            update.executor_relationship = rel_word
            update.executor_name = name_word.title()

    # Additional executor pattern: "brother James is my executor"
    for rel in relationship_words:
        pattern = rf"{rel}\s+(\w+)"
        m = re.search(pattern, msg_lower)
        if m and not update.executor_name:
            update.executor_relationship = rel
            update.executor_name = m.group(1).strip(".,;!").title()
            break

    # ── Specific gifts ───────────────────────────────────────────────────
    gift_match = re.search(
        r"(?:give|leave|bequeath|gift)\s+(.+?)(?:to\s+\w+)?(?:\.|$)",
        msg, re.IGNORECASE,
    )
    no_gifts = re.search(
        r"(no\s+(?:specific\s+)?gifts|none|nothing\s+specific|no\s+items)",
        msg_lower,
    )
    if no_gifts and not gift_match:
        update.specific_gifts = []
    elif gift_match:
        raw = gift_match.group(0).strip().rstrip(".")
        update.specific_gifts = [raw]

    # ── Additional wishes ────────────────────────────────────────────────
    wish_match = re.search(
        r"(?:wish(?:es)?|additional|also\s+want|note|instruction)[:\s]+(.+?)(?:\.|$)",
        msg, re.IGNORECASE,
    )
    no_wishes = re.search(
        r"(no\s+(?:additional\s+)?wish|none|nothing\s+(?:else|more)|that'?s?\s+(?:all|it))",
        msg_lower,
    )
    if no_wishes and not wish_match:
        update.additional_wishes = ""
    elif wish_match:
        update.additional_wishes = wish_match.group(1).strip()

    # ── Determine follow-up question ─────────────────────────────────────
    # Build a temporary merged state to check what's still missing
    temp_dict = current_state.model_dump()
    if update.full_name:
        temp_dict["full_name"] = update.full_name
    if update.home_address:
        temp_dict["home_address"] = update.home_address
    if update.covers_worldwide_assets is not None:
        temp_dict["covers_worldwide_assets"] = update.covers_worldwide_assets
    if update.has_children is not None:
        temp_dict["has_children"] = update.has_children
    if update.children_names:
        temp_dict["children_names"] = update.children_names
    if update.executor_name:
        if not temp_dict.get("executor"):
            temp_dict["executor"] = {"name": update.executor_name, "relationship": update.executor_relationship}
        else:
            temp_dict["executor"]["name"] = update.executor_name
    if update.executor_relationship and temp_dict.get("executor"):
        temp_dict["executor"]["relationship"] = update.executor_relationship
    if update.specific_gifts is not None:
        temp_dict["specific_gifts"] = update.specific_gifts
    if update.additional_wishes is not None:
        temp_dict["additional_wishes"] = update.additional_wishes

    temp_state = PersonalWishesState(**temp_dict)
    missing = get_missing_fields(temp_state)

    if missing:
        first_missing = missing[0]
        update.assistant_reply = _FOLLOW_UP_QUESTIONS.get(
            first_missing,
            "Could you provide more details?"
        )
    else:
        update.assistant_reply = (
            "Wonderful! I have all the information needed for your Personal Wishes Document. "
            "Please review the draft on the right. Would you like to change anything?"
        )

    return update
