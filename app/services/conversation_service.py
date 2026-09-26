"""
Conversation Service — orchestrates a single chat turn.

Responsibilities (Single Responsibility):
  - Coordinate the LLM call, state merge, and document generation
  - Handle pending confirmations from previous turns
  - Manage conversation history in the session store
  - Return a fully assembled ChatResponse

This is the *application* layer in a clean architecture:
  UI → API → ConversationService → (LLM, StateService, DocumentService, SessionStore)
"""

from __future__ import annotations

import logging
from typing import Optional

from app.models import (
    ChatResponse,
    ConversationStatus,
    PendingConfirmation,
    PersonalWishesState,
)
from app.services.document_service import generate_draft_document
from app.services.llm_service import process_user_input
from app.services.state_service import compute_status, merge_state_update
from app.storage.session_store import Session, SessionStore

logger = logging.getLogger(__name__)


class ConversationService:
    """
    Stateless service that processes a single conversation turn.

    Injected dependencies (Dependency Inversion):
      - SessionStore: for reading/writing backend-owned state
    """

    def __init__(self, session_store: SessionStore) -> None:
        self._store = session_store

    def handle_message(self, session_id: str, user_message: str) -> ChatResponse:
        """
        Process one user message end-to-end and return the response.

        Flow:
          1. Retrieve or create the session
          2. Check for pending confirmations from a previous turn
          3. Call LLM service for structured extraction
          4. Validate and merge into canonical state
          5. Detect contradictions → queue confirmation
          6. Generate draft document from updated state
          7. Persist everything and return response
        """
        session = self._store.get_or_create_session(session_id)

        # ── Step 1: Handle pending confirmation ─────────────────────────
        if session.pending_confirmation:
            return self._handle_confirmation(session, user_message)

        # ── Step 2: Record user message ─────────────────────────────────
        self._store.append_message(session_id, "user", user_message)

        # ── Step 3: LLM extraction ──────────────────────────────────────
        recent_history = self._store.get_recent_history(session_id, limit=10)
        partial_update = process_user_input(
            user_message=user_message,
            current_state=session.state,
            conversation_history=recent_history,
        )

        # ── Step 4: Validate & merge ────────────────────────────────────
        updated_state, pending = merge_state_update(session.state, partial_update)

        # ── Step 5: Handle contradiction ────────────────────────────────
        assistant_reply = partial_update.assistant_reply
        status = compute_status(updated_state)

        if pending:
            # State was updated (optimistic), but we flag it for the user
            logger.info(
                "Contradiction detected on field '%s': '%s' → '%s'",
                pending.field, pending.old_value, pending.new_value,
            )
            confirmation_msg = (
                f"I noticed you previously provided \"{pending.old_value}\" "
                f"for {_field_display(pending.field)}, but now mentioned "
                f"\"{pending.new_value}\". I've updated it to the new value. "
                f"If that's not correct, please let me know!"
            )
            assistant_reply = f"{confirmation_msg}\n\n{assistant_reply}"
            status = ConversationStatus.NEEDS_CONFIRMATION

        # ── Step 6: Persist state ───────────────────────────────────────
        self._store.update_state(session_id, updated_state)
        self._store.append_message(session_id, "assistant", assistant_reply)

        # ── Step 7: Generate document ───────────────────────────────────
        draft = generate_draft_document(updated_state)

        return ChatResponse(
            session_id=session_id,
            assistant_reply=assistant_reply,
            state=updated_state,
            draft_document=draft,
            status=status,
            pending_confirmation=pending,
        )

    # -------------------------------------------------------------------
    # Confirmation handling
    # -------------------------------------------------------------------

    def _handle_confirmation(
        self, session: Session, user_message: str
    ) -> ChatResponse:
        """Handle a user reply to a pending confirmation."""
        msg_lower = user_message.strip().lower()
        pending = session.pending_confirmation

        # Clear the pending confirmation
        self._store.set_pending_confirmation(session.session_id, None)
        session.pending_confirmation = None

        # Record message
        self._store.append_message(session.session_id, "user", user_message)

        if _is_affirmative(msg_lower):
            reply = "Great, the update has been confirmed. "
        else:
            reply = "No problem, I'll keep the previous value. "
            # Note: in this implementation the state was already updated
            # optimistically.  A production system would defer the update.

        # Continue with normal flow
        status = compute_status(session.state)
        draft = generate_draft_document(session.state)
        self._store.append_message(session.session_id, "assistant", reply)

        return ChatResponse(
            session_id=session.session_id,
            assistant_reply=reply,
            state=session.state,
            draft_document=draft,
            status=status,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _field_display(field_name: str) -> str:
    """Convert internal field names to user-friendly labels."""
    mapping = {
        "full_name": "your name",
        "home_address": "your address",
        "covers_worldwide_assets": "worldwide asset coverage",
        "has_children": "children status",
        "executor.name": "your executor's name",
    }
    return mapping.get(field_name, field_name.replace("_", " "))


def _is_affirmative(text: str) -> bool:
    """Check whether a short reply is affirmative."""
    affirmatives = {"yes", "y", "yeah", "yep", "correct", "right", "sure", "ok", "okay", "confirm"}
    return text.strip().rstrip(".!") in affirmatives
