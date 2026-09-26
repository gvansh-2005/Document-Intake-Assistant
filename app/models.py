"""
Domain models for the Document Intake Assistant.

Defines the canonical state schema, partial update schema for LLM extraction,
and API request/response contracts. All state semantics are explicit:
  - None  → field not yet collected (unknown)
  - False → user explicitly said no
  - []    → user explicitly confirmed empty list (no items)

Follows the Single Responsibility Principle: this module is the single authority
on what the data looks like.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional, List

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------

class ConversationStatus(str, Enum):
    """Tracks the overall intake workflow phase."""
    COLLECTING = "collecting"
    NEEDS_CLARIFICATION = "needs_clarification"
    NEEDS_CONFIRMATION = "needs_confirmation"
    COMPLETE = "complete"
    ERROR = "error"


class ExecutorInfo(BaseModel):
    """Represents the designated executor of the personal wishes document."""
    name: Optional[str] = Field(None, description="Full name of the designated executor")
    relationship: Optional[str] = Field(
        None, description="Relationship to the user (e.g., brother, friend, solicitor)"
    )


# ---------------------------------------------------------------------------
# Canonical application state  (the single source of truth)
# ---------------------------------------------------------------------------

class PersonalWishesState(BaseModel):
    """
    The authoritative state of collected information for the personal wishes document.

    Semantic conventions:
      - Optional[T] = None   → not yet collected / unknown
      - Optional[bool] False → user explicitly declined
      - Optional[List] = None → not yet asked; [] → asked and confirmed empty
    """
    full_name: Optional[str] = Field(None, description="Full legal name of the user")
    home_address: Optional[str] = Field(None, description="Primary residential address")
    covers_worldwide_assets: Optional[bool] = Field(
        None, description="True if document applies to assets globally"
    )
    has_children: Optional[bool] = Field(None, description="True if the user has children")
    children_names: Optional[List[str]] = Field(
        None,
        description="List of children's names. None=not yet asked, []=confirmed no names"
    )
    executor: Optional[ExecutorInfo] = Field(None, description="Executor appointment details")
    specific_gifts: Optional[List[str]] = Field(
        None,
        description="Specific item/monetary gifts. None=not yet asked, []=confirmed none"
    )
    additional_wishes: Optional[str] = Field(None, description="Extra custom notes or wishes")


# ---------------------------------------------------------------------------
# Pending confirmation tracking
# ---------------------------------------------------------------------------

class PendingConfirmation(BaseModel):
    """Tracks a value change that requires explicit user confirmation before applying."""
    field: str = Field(..., description="Name of the field being changed")
    old_value: Optional[str] = Field(None, description="Previous confirmed value (human-readable)")
    new_value: str = Field(..., description="Proposed replacement value (human-readable)")


# ---------------------------------------------------------------------------
# LLM extraction schema (what the LLM is forced to return)
# ---------------------------------------------------------------------------

class PartialStateUpdate(BaseModel):
    """
    Schema enforced on LLM structured output for incremental state extraction.

    Each field is Optional — the LLM only populates fields that the user's
    latest message actually mentions. Fields left None are not touched.
    """
    full_name: Optional[str] = None
    home_address: Optional[str] = None
    covers_worldwide_assets: Optional[bool] = None
    has_children: Optional[bool] = None
    children_names: Optional[List[str]] = None
    executor_name: Optional[str] = None
    executor_relationship: Optional[str] = None
    specific_gifts: Optional[List[str]] = None
    additional_wishes: Optional[str] = None
    assistant_reply: str = Field(
        ..., description="The follow-up conversational message to the user"
    )


# ---------------------------------------------------------------------------
# API contracts
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    """Inbound request from the frontend."""
    session_id: str = Field(..., description="Unique session identifier")
    message: str = Field(..., min_length=1, description="User's chat message")


class ChatResponse(BaseModel):
    """Outbound response to the frontend."""
    session_id: str
    assistant_reply: str
    state: PersonalWishesState
    draft_document: str
    status: ConversationStatus
    pending_confirmation: Optional[PendingConfirmation] = None
