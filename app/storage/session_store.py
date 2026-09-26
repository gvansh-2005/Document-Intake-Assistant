"""
In-memory session store — the backend's authoritative state repository.

Design decision:
    We deliberately use an in-memory dictionary rather than a database because
    the exercise does not require persistence across restarts.  In production
    this would be swapped for Redis or PostgreSQL via the same interface
    (Open/Closed Principle).

Thread-safety:
    FastAPI uses a single event loop with async handlers; dict operations on
    CPython are GIL-protected, so an in-memory dict is safe for this use case.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from app.models import PersonalWishesState, PendingConfirmation


@dataclass
class Session:
    """Holds all server-side state for a single user session."""
    session_id: str
    state: PersonalWishesState = field(default_factory=PersonalWishesState)
    conversation_history: List[dict] = field(default_factory=list)
    pending_confirmation: Optional[PendingConfirmation] = None


class SessionStore:
    """
    Simple in-memory session repository.

    Follows the Interface Segregation Principle — exposes only the operations
    that the rest of the application needs.
    """

    def __init__(self) -> None:
        self._sessions: Dict[str, Session] = {}

    # -- lifecycle -----------------------------------------------------------

    def create_session(self) -> Session:
        """Create and return a new session with a unique ID."""
        session_id = uuid.uuid4().hex[:12]
        session = Session(session_id=session_id)
        self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        """Retrieve an existing session or None."""
        return self._sessions.get(session_id)

    def get_or_create_session(self, session_id: str) -> Session:
        """Return the session if it exists, otherwise create a new one with the given ID."""
        if session_id not in self._sessions:
            session = Session(session_id=session_id)
            self._sessions[session_id] = session
        return self._sessions[session_id]

    # -- mutations -----------------------------------------------------------

    def update_state(self, session_id: str, state: PersonalWishesState) -> None:
        """Replace the canonical state for a session."""
        session = self._sessions.get(session_id)
        if session:
            session.state = state

    def append_message(self, session_id: str, role: str, content: str) -> None:
        """Append a message to the session's conversation history."""
        session = self._sessions.get(session_id)
        if session:
            session.conversation_history.append({"role": role, "content": content})

    def set_pending_confirmation(
        self, session_id: str, pending: Optional[PendingConfirmation]
    ) -> None:
        """Set or clear a pending confirmation on a session."""
        session = self._sessions.get(session_id)
        if session:
            session.pending_confirmation = pending

    def get_recent_history(self, session_id: str, limit: int = 10) -> List[dict]:
        """
        Return the most recent conversation messages, capped at *limit*.

        Optimisation: instead of sending the entire history to the LLM, we
        send only a sliding window.  This keeps token usage bounded and
        reduces latency without losing conversational coherence.
        """
        session = self._sessions.get(session_id)
        if not session:
            return []
        return session.conversation_history[-limit:]

    # -- introspection -------------------------------------------------------

    def session_exists(self, session_id: str) -> bool:
        return session_id in self._sessions

    def session_count(self) -> int:
        return len(self._sessions)

    def list_sessions(self) -> List[dict]:
        """
        Return summary metadata for all sessions (for the sidebar).
        Sorted by most recently active (last message) first.
        """
        summaries = []
        for sid, session in self._sessions.items():
            # Derive a display label from the user's name or first message
            label = "New Conversation"
            if session.state.full_name:
                label = session.state.full_name
            elif session.conversation_history:
                first_user = next(
                    (m["content"][:40] for m in session.conversation_history
                     if m.get("role") == "user"),
                    "New Conversation",
                )
                label = first_user

            summaries.append({
                "session_id": sid,
                "label": label,
                "message_count": len(session.conversation_history),
                "has_name": session.state.full_name is not None,
            })

        # Most messages = most recent activity → show first
        summaries.sort(key=lambda s: s["message_count"], reverse=True)
        return summaries

    def get_full_history(self, session_id: str) -> List[dict]:
        """Return the complete conversation history for a session."""
        session = self._sessions.get(session_id)
        if not session:
            return []
        return list(session.conversation_history)

