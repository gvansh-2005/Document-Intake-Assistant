"""
FastAPI application entry point.

Routes:
  GET  /              → serves the single-page UI
  POST /api/chat      → processes a chat message (session_id + message)
  POST /api/session   → creates a new session and returns its ID
  GET  /api/health    → health check

Design decisions:
  - Session state is owned entirely by the backend (in-memory store).
  - The frontend only sends session_id + message; it never sends state.
  - ConversationService is injected as an application-scoped singleton
    (Dependency Inversion via FastAPI's app.state).
  - Logging is configured at startup to make LLM provider mode visible.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.models import ChatRequest, ChatResponse
from app.services.conversation_service import ConversationService
from app.storage.session_store import SessionStore

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Application lifespan (startup / shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise shared resources on startup; clean up on shutdown."""
    store = SessionStore()
    service = ConversationService(session_store=store)

    # Attach to app.state so route handlers can access them
    app.state.session_store = store
    app.state.conversation_service = service

    # Log configuration
    use_mock = os.getenv("USE_MOCK_LLM", "false").lower() == "true"
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    mode = "deterministic mock" if (use_mock or not api_key) else "Groq (qwen/qwen3.8-27b)"
    logger.info("Starting Document Intake Assistant")
    logger.info("LLM provider: %s", mode)

    yield  # ← application runs

    logger.info("Shutting down Document Intake Assistant")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Document Intake Assistant",
    description=(
        "A structured document intake web application powered by FastAPI, "
        "Groq LLM structured outputs, and Pydantic validation."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

templates = Jinja2Templates(directory="app/templates")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def render_ui(request: Request):
    """Serve the single-page frontend."""
    return templates.TemplateResponse(request, "index.html")


@app.post("/api/session")
async def create_session(request: Request):
    """Create a new session and return its ID."""
    store: SessionStore = request.app.state.session_store
    session = store.create_session()
    logger.info("New session created: %s", session.session_id)
    return {
        "session_id": session.session_id,
        "message": "Session created. Begin by telling me your full legal name.",
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest, request: Request):
    """
    Process a single chat turn.

    Request:  { session_id, message }
    Response: { session_id, assistant_reply, state, draft_document, status }
    """
    service: ConversationService = request.app.state.conversation_service

    try:
        response = service.handle_message(
            session_id=payload.session_id,
            user_message=payload.message,
        )
        return response

    except Exception:
        logger.exception("Unhandled error in chat endpoint")
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred. Your state has been preserved.",
        )


@app.get("/api/sessions")
async def list_sessions(request: Request):
    """Return all session summaries for the sidebar."""
    store: SessionStore = request.app.state.session_store
    return {"sessions": store.list_sessions()}


@app.get("/api/session/{session_id}")
async def get_session(session_id: str, request: Request):
    """Load a full session — state, history, and draft document."""
    store: SessionStore = request.app.state.session_store
    session = store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    from app.services.document_service import generate_draft_document
    from app.services.state_service import compute_status

    return {
        "session_id": session.session_id,
        "state": session.state.model_dump(),
        "conversation_history": store.get_full_history(session_id),
        "draft_document": generate_draft_document(session.state),
        "status": compute_status(session.state).value,
    }


@app.get("/api/session/{session_id}/pdf")
async def download_pdf(session_id: str, request: Request):
    """Generate and download the Personal Wishes Document as a PDF."""
    store: SessionStore = request.app.state.session_store
    session = store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    from io import BytesIO
    from fastapi.responses import StreamingResponse
    from app.services.document_service import generate_pdf

    pdf_bytes = generate_pdf(session.state)
    filename = f"personal_wishes_{session_id[:8]}.pdf"

    return StreamingResponse(
        BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/health")
async def health_check():
    """Simple liveness probe."""
    return {"status": "healthy", "service": "Document Intake Assistant"}
