# Document Intake Assistant

A structured document intake web application that collects personal wishes information through a conversational AI interface and generates a live-preview draft document.

**Core Philosophy**: *The LLM understands language. Pydantic validates structure. Application code controls state. Deterministic code generates the document.*

---

## Table of Contents

- [Demo Video](#demo-video)
- [Quick Start](#quick-start)
- [Architecture Overview](#architecture-overview)
- [Project Structure](#project-structure)
- [Structured State Schema](#structured-state-schema)
- [Implementation Details](#implementation-details)
- [API Contract](#api-contract)
- [Design Decisions](#design-decisions)
- [SOLID Principles Applied](#solid-principles-applied)
- [Requirement Coverage](#requirement-coverage)
- [Testing](#testing)
- [Performance & Internals](#performance--internals)
  - [API Response Optimisations](#api-response-optimisations)
  - [Conversation History Mechanism](#conversation-history-mechanism)
- [Known Limitations & Production Improvements](#known-limitations--production-improvements)
- [License](#license)

---

## Demo Video

A short walkthrough of the application in action — covering the conversational intake flow, live state preview, and live document generation.

📹 **Watch the demo**: https://drive.google.com/file/d/1lvb8BnKinoek8pgINsTRHYOeeNlwoURo/view?usp=sharing

---

## Quick Start

### Prerequisites
- Python 3.10+
- pip

### Setup

```bash
# 1. Clone and enter the project
cd document-intake-assistant

# 2. Create virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
copy .env.example .env
# Edit .env:
#   GROQ_API_KEY=your_key_here    (or leave empty for mock)
#   USE_MOCK_LLM=true             (set true for offline mode)

# 5. Run the application
uvicorn app.main:app --reload

# 6. Open in browser
# Navigate to http://127.0.0.1:8000
```

### Run Tests

```bash
pytest -v
```

> **Note**: Set `USE_MOCK_LLM=true` to run completely offline without an API key. The deterministic mock provides full conversational functionality for testing and development.

---

## Architecture Overview

```
                 ┌───────────────────┐
                 │    Simple UI      │
                 │ HTML + JS         │
                 │ (sends only       │
                 │  session_id +     │
                 │  message)         │
                 └─────────┬─────────┘
                           │
                    POST /api/chat
                           │
                           ▼
                 ┌───────────────────┐
                 │      FastAPI      │
                 │      main.py      │
                 └─────────┬─────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │  Session Store    │  ← Backend owns state
                 │  (in-memory)      │
                 └─────────┬─────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │ Conversation      │  ← Orchestration layer
                 │ Service           │
                 └─────────┬─────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │   LLM Provider    │  ← Abstracted interface
                 │ (Groq / Mock)     │
                 └─────────┬─────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │ Pydantic          │  ← Schema enforcement
                 │ Validation        │
                 └─────────┬─────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │ State Service     │  ← Merge + contradiction
                 │ (State Machine)   │     detection
                 └─────────┬─────────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
              ▼            ▼            ▼
           Missing      Ambiguous   Contradiction
              │            │            │
              └────────────┼────────────┘
                           ▼
                 ┌───────────────────┐
                 │ Document Service  │  ← Deterministic template
                 └─────────┬─────────┘
                           │
                           ▼
                  Live State + Draft
```

### Data Flow 

1. **Frontend** sends `{ session_id, message }` via `POST /api/chat`
2. **FastAPI** routes to `ConversationService.handle_message()`
3. **SessionStore** retrieves the canonical state for the session
4. **LLM Provider** (Groq or Mock) extracts structured data from the user message → `PartialStateUpdate`
5. **Pydantic** validates the structured output against the schema
6. **State Service** merges the update into canonical state, detects contradictions
7. **Document Service** generates the draft document deterministically from state
8. **Response** returns `{ session_id, assistant_reply, state, draft_document, status }`

---

## Project Structure

```
document-intake-assistant/
│
├── app/
│   ├── __init__.py                 # Package docstring
│   ├── main.py                     # FastAPI routes + lifespan
│   ├── models.py                   # All Pydantic schemas & API contracts
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── llm_service.py          # LLM provider abstraction + Groq + Mock
│   │   ├── state_service.py        # State merge, contradictions, missing fields
│   │   ├── conversation_service.py # Orchestrates a full chat turn
│   │   └── document_service.py     # Deterministic draft document generator
│   │
│   ├── storage/
│   │   ├── __init__.py
│   │   └── session_store.py        # In-memory backend-owned session store
│   │
│   ├── templates/
│   │   └── index.html              # Single-page frontend (vanilla HTML/CSS/JS)
│   │
│   └── prompts/
│       └── intake_prompt.txt       # Externalised LLM system prompt
│
├── tests/
│   ├── __init__.py
│   ├── test_models.py              # Schema validation, serialisation, malformed rejection
│   ├── test_state.py               # Merge logic, contradictions, children/executor logic
│   ├── test_document.py            # Draft generation, disclaimers, consistency
│   ├── test_conversation.py        # Orchestration, multi-turn, session lifecycle
│   ├── test_api.py                 # HTTP endpoint integration tests
│   ├── test_llm_validation.py      # LLM output validation + provider abstraction
│   │
│   └── fixtures/
│       ├── valid.json              # Multi-field extraction
│       ├── ambiguous.json          # Executor without relationship
│       ├── malformed.json          # Invalid type → Pydantic rejection
│       ├── correction.json         # Name correction → contradiction
│       ├── contradiction.json      # Children status contradiction
│       └── multi-field.json        # All fields in one message
│
├── .env.example                    # Environment template
├── .gitignore                      # Keeps secrets/artifacts out of VCS
├── requirements.txt                # Python dependencies
├── pytest.ini                      # Test configuration
├── README.md                       # This file
└── AI_LOG.md                       # Design iterations & LLM correction log
```

---

## Structured State Schema

```python
class PersonalWishesState(BaseModel):
    full_name: Optional[str] = None               # Full legal name
    home_address: Optional[str] = None             # Residential address
    covers_worldwide_assets: Optional[bool] = None # True/False/None(unknown)
    has_children: Optional[bool] = None            # True/False/None(unknown)
    children_names: Optional[List[str]] = None     # None=not asked, []=confirmed empty
    executor: Optional[ExecutorInfo] = None         # {name, relationship}
    specific_gifts: Optional[List[str]] = None     # None=not asked, []=confirmed none
    additional_wishes: Optional[str] = None        # Free-text wishes
```

### Value Semantics

| Value | Meaning |
|-------|---------|
| `None` | Unknown / not yet collected — assistant should ask |
| `False` | User explicitly declined (e.g., "I don't have children") |
| `[]` | User confirmed empty (e.g., "No specific gifts") |
| `""` | User confirmed none (for string fields) |

This distinction is reflected throughout the codebase: state service, document generator, and follow-up question logic all branch on it.

---

## Implementation Details

### 1. Backend-Owned State (Session Management)

The **backend is the single source of truth** for all session state. The frontend never sends state — only `session_id` and `message`.

**Implementation**: `app/storage/session_store.py`

```python
# Session dataclass holds:
# - session_id: str
# - state: PersonalWishesState      ← canonical state
# - conversation_history: List[dict] ← full history
# - pending_confirmation: Optional   ← contradiction tracking
```

**Why**: Prevents stale client state from overwriting newer server state. The brief explicitly says "conversation history alone is not sufficient as the application's source of truth."

### 2. Contradiction Detection

When the LLM extracts a value that differs from an existing non-null value, the state service returns a `PendingConfirmation`:

```python
# state_service.py → merge_state_update()
PendingConfirmation(
    field="full_name",
    old_value="Jane Smith",
    new_value="Jane Doe",
)
```

The conversation service includes a notification in the assistant reply and sets the response status to `needs_confirmation`.

### 3. Ambiguity Handling

The state service detects partially-filled structures that need follow-up:

- **Executor name without relationship**: `get_missing_fields()` detects `executor_relationship` is missing → triggers: "What is James's relationship to you?"
- **Has children without names**: `has_children=True` with empty `children_names` → triggers: "Could you share the names of your children?"

### 4. Deterministic Document Generation

The draft document is generated from **state, not conversation**:

```python
# document_service.py
State → deterministic template → document
```

Not:
```python
# ❌ What we DON'T do
Conversation → LLM → document
```

This eliminates hallucination risk and ensures the document is always consistent with the validated state.

### 5. LLM Provider Abstraction

The LLM interaction is hidden behind an abstract interface (`LLMProvider` ABC):

```python
# llm_service.py
class LLMProvider(ABC):
    @abstractmethod
    def extract(self, user_message, current_state, conversation_history) -> PartialStateUpdate:
        ...

class GroqProvider(LLMProvider):
    """Calls Groq Cloud API with structured JSON output."""
    ...

class MockLLMProvider(LLMProvider):
    """Deterministic regex-based extraction for offline/testing."""
    ...
```

**MockLLMProvider implements the same interface as GroqProvider. Therefore, replacing MockLLMProvider with GroqProvider does not require changes to state management, document generation, or the frontend.**

The mock handles:

| Capability | Examples |
|-----------|----------|
| Name extraction | "My name is Jane Smith" |
| Address extraction | "I live at 42 Elm Street" |
| Negation handling | "No children", "Don't have kids", "Not worldwide" |
| Executor + relationship | "My brother James", "executor is my friend Bob" |
| Multiple fields | "My name is Jane and I don't have children" |
| Follow-up questions | Based on which fields are still missing |
| Corrections | "Actually my name is Jane Doe" |

### 6. Frontend Security (XSS Prevention)

All user and assistant messages are rendered using safe DOM methods:

```javascript
// ✅ Safe — treats content as text
div.textContent = message;

// ❌ Removed — vulnerable to XSS
chatBox.innerHTML += `<div>${message}</div>`;
```

### 7. Structured LLM Output Enforcement

When using Groq, the system uses `response_format={"type": "json_object"}` to force JSON output:

```python
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=messages,
    response_format={"type": "json_object"},
    temperature=0.2,
)
```

The raw JSON is then validated against the Pydantic schema. If validation fails, the output is rejected and a safe fallback is returned.

### 8. Externalised Prompt

The system prompt is stored in `app/prompts/intake_prompt.txt` rather than inline in Python code. This:
- Separates prompt engineering from application logic
- Makes prompt iteration easier
- Allows independent version control and review

### 9. Graceful Error Handling

| Error Type | Handling |
|-----------|---------|
| Groq API failure | Falls back to a safe `PartialStateUpdate` with error message; state unchanged |
| Network timeout | Same fallback; state preserved |
| Malformed LLM output | Pydantic rejects it; state unchanged |
| Missing API key | Automatically switches to deterministic mock |
| Invalid user input | FastAPI returns 422 with validation details |
| Unhandled exception | Returns 500 with generic message; state preserved |

---

## API Contract

### `POST /api/session` — Create Session

**Response**:
```json
{
    "session_id": "a1b2c3d4e5f6",
    "message": "Session created. Begin by telling me your full legal name."
}
```

### `POST /api/chat` — Process Chat Message

**Request**:
```json
{
    "session_id": "a1b2c3d4e5f6",
    "message": "My name is Jane Smith and my executor is my brother James."
}
```

**Response**:
```json
{
    "session_id": "a1b2c3d4e5f6",
    "assistant_reply": "Thank you, Jane! What is your current residential address?",
    "state": {
        "full_name": "Jane Smith",
        "home_address": null,
        "covers_worldwide_assets": null,
        "has_children": null,
        "children_names": null,
        "executor": {
            "name": "James",
            "relationship": "brother"
        },
        "specific_gifts": null,
        "additional_wishes": null
    },
    "draft_document": "=====================...DRAFT...",
    "status": "collecting",
    "pending_confirmation": null
}
```

**Status values**:

| Status | Meaning |
|--------|---------|
| `collecting` | Still gathering required information |
| `needs_clarification` | Ambiguous input detected |
| `needs_confirmation` | Contradiction detected — user notified |
| `complete` | All required fields collected |
| `error` | Processing error occurred |

### `GET /api/health` — Health Check

**Response**: `{ "status": "healthy" }`

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| **FastAPI** | Native Pydantic integration, async support, automatic OpenAPI docs, easy testing |
| **Backend-owned state** | Prevents client from accidentally or maliciously overwriting state |
| **In-memory session store** | Simple, no external dependencies; production would use Redis/PostgreSQL |
| **Deterministic document generation** | Eliminates hallucination risk; document always matches validated state |
| **LLM Provider ABC** | Formal interface enables swapping Groq ↔ Mock without touching business logic |
| **Structured LLM output** | JSON mode forces schema compliance; Pydantic gate catches malformed output |
| **No database** | Deliberate choice — persistence was not required; documented as known limitation |
| **No React/Vue** | Vanilla HTML/CSS/JS keeps the frontend simple; the brief asks for a "simple UI" |
| **No LangChain/LangGraph** | Unnecessary abstraction for this use case; direct SDK is simpler |
| **Externalised prompt** | Separates prompt engineering from application logic |
| **Service layer pattern** | Clean separation of concerns following SOLID principles |

---

## SOLID Principles Applied

### Single Responsibility (SRP)
Each module has one reason to change:
- `models.py` — data shape definitions only
- `llm_service.py` — LLM communication only
- `state_service.py` — state transition logic only
- `document_service.py` — document rendering only
- `conversation_service.py` — turn orchestration only
- `session_store.py` — session persistence only

### Open/Closed (OCP)
- `SessionStore` can be swapped for Redis/PostgreSQL without changing business logic
- LLM provider can be swapped (Groq → Anthropic → OpenAI) by implementing the `LLMProvider` ABC
- New document sections can be added without modifying existing section builders

### Liskov Substitution (LSP)
- `MockLLMProvider` and `GroqProvider` are interchangeable — both implement `LLMProvider`
- `PartialStateUpdate` and `PersonalWishesState` share field semantics — updates are a proper subset of state

### Interface Segregation (ISP)
- `SessionStore` exposes only the operations needed (create, get, update, append, recent_history)
- Services don't depend on the full FastAPI request — they receive only the data they need

### Dependency Inversion (DIP)
- `ConversationService` depends on `SessionStore` (abstraction), not on `dict` (implementation)
- `state_service` depends on `PartialStateUpdate` (schema), not on the LLM provider
- Services are injected via `app.state` at startup, not hard-coded
- LLM interaction goes through `LLMProvider` (ABC), not a concrete provider

---

## Requirement Coverage

| # | Requirement | Implementation | File |
|---|------------|---------------|------|
| 1 | Backend + simple UI | FastAPI + vanilla HTML/JS | `main.py`, `index.html` |
| 2 | Multi-turn conversation | Conversation history in session store | `session_store.py` |
| 3 | Conversation history not source of truth | Backend-owned `PersonalWishesState` | `session_store.py`, `state_service.py` |
| 4 | Structured schema | Pydantic `PersonalWishesState` | `models.py` |
| 5 | Live structured-state preview | JSON state in frontend panel | `index.html` |
| 6 | Live document preview | Draft document in frontend panel | `index.html`, `document_service.py` |
| 7 | User can correct information | State merge overwrites + contradiction detection | `state_service.py` |
| 8 | Ask follow-up questions | `get_missing_fields()` → question mapping | `state_service.py`, `llm_service.py` |
| 9 | Handle ambiguity | Executor without relationship → follow-up | `state_service.py` |
| 10 | Handle contradictions | `PendingConfirmation` + notification | `state_service.py`, `conversation_service.py` |
| 11 | Don't invent facts | Prompt constraints + None semantics | `intake_prompt.txt`, `models.py` |
| 12 | Unknown = unconfirmed | `Optional[T] = None` throughout | `models.py` |
| 13 | Don't re-ask captured info | State-aware follow-up logic | `llm_service.py` |
| 14 | Multiple fields in one answer | LLM extracts all; state merges all | `llm_service.py`, `state_service.py` |
| 15 | Validate LLM output | Pydantic structured output | `llm_service.py` |
| 16 | Validate before applying state | `merge_state_update()` after Pydantic | `state_service.py` |
| 17 | Preview consistent with state | Document generated from state, not conversation | `document_service.py` |
| 18 | Generate draft document | Deterministic template generator | `document_service.py` |
| 19 | Fictional disclaimer | Always present in document header | `document_service.py` |
| 20 | Provider abstracted | `LLMProvider` ABC with `GroqProvider` + `MockLLMProvider` | `llm_service.py` |
| 21 | Defined API contract | Pydantic request/response models | `models.py` |
| 22 | LLM error handling | Try/except → safe fallback | `llm_service.py` |
| 23 | Malformed response handling | Pydantic validation rejects invalid types | `models.py`, `test_models.py` |
| 24 | Missing configuration handling | Auto-switch to mock | `llm_service.py` |
| 25 | Automated tests | 6 test modules, 50+ test cases | `tests/` |
| 26 | Test fixtures | Valid, ambiguous, malformed, correction, contradiction, multi-field | `tests/fixtures/` |
| 27 | Local setup instructions | README Quick Start | This file |
| 28 | Secrets out of source control | `.env` + `.gitignore` | `.env.example`, `.gitignore` |
| 29 | AI log | 11 documented iterations | `AI_LOG.md` |
| 30 | Deterministic mock | Regex-based extraction with negation | `llm_service.py` |

---

## Testing

### Test Modules

| Module | Focus | Count |
|--------|-------|-------|
| `test_models.py` | Schema validation, serialisation, malformed rejection | 10 |
| `test_state.py` | Merge logic, contradictions, children/executor, missing fields, fixtures | 20 |
| `test_document.py` | Draft generation, disclaimers, None vs [] rendering | 14 |
| `test_conversation.py` | Multi-turn flow, session lifecycle, corrections, negatives | 10 |
| `test_api.py` | HTTP endpoints, state ownership, validation, frontend serving | 10 |
| `test_llm_validation.py` | LLM output validation, provider abstraction | 20 |

### Test Fixtures

| Fixture | Scenario |
|---------|----------|
| `valid.json` | Multi-field extraction from one message |
| `ambiguous.json` | Executor name without relationship → triggers follow-up |
| `malformed.json` | Invalid type (`has_children: "banana"`) → Pydantic rejects |
| `correction.json` | Name change → contradiction detection |
| `contradiction.json` | Children status change → contradiction detection |
| `multi-field.json` | Name, address, children, executor in one message |

### Running Tests

```bash
# All tests
pytest -v

# Specific module
pytest tests/test_state.py -v

# With coverage (if pytest-cov installed)
pytest --cov=app --cov-report=term-missing
```

---

## Performance & Internals

### API Response Optimisations

**1. Minimal Payload — Session-Based Architecture**
The request payload is minimal: `{ session_id, message }`. State is not sent from the client, reducing request size and eliminating redundant data transfer.

**2. Sliding Window Conversation History**
Only the last 10 messages are sent to the LLM (`get_recent_history(limit=10)`). This bounds:
- **Token cost**: Prevents unbounded growth of prompt tokens
- **Latency**: Shorter prompts = faster LLM response
- **State safety**: Older messages are not needed because extracted state is already persisted

**3. Lazy Document Generation**
The draft document is generated only once per request, after state merge. It is not regenerated on read — only on write (state change).

**4. Response Compression via Status Enum**
The `status` field (`collecting`, `needs_confirmation`, `complete`, etc.) allows the frontend to make UI decisions without parsing the full state object.

**5. Structured Output — Zero Post-Processing**
Using Groq's JSON mode with `response_format={"type": "json_object"}` means the LLM response is already JSON. Pydantic then validates and parses it. No regex parsing, no JSON extraction from text, no retry loops.

### Conversation History Mechanism

**How it works**:

1. **Storage**: Each session maintains a `conversation_history: List[dict]` in the backend session store
2. **Append**: After each turn, both the user message and assistant reply are appended
3. **Retrieval**: `get_recent_history(session_id, limit=10)` returns a sliding window
4. **LLM Context**: The sliding window is injected into the LLM prompt as prior messages

**Why a sliding window?**

| Approach | Token Cost | Latency | State Safety |
|----------|-----------|---------|-------------|
| Full history | O(n) — grows per turn | Increases | Risk of context confusion |
| **Sliding window** | **O(1) — constant** | **Bounded** | **State already extracted** |
| No history | O(1) | Fastest | Loses conversational context |

The sliding window is the best trade-off: it maintains enough context for coherent conversation while keeping costs bounded. Since all extracted information is persisted in the canonical state, no *data* is lost — only conversational context beyond the window.

**History never controls state.** The conversation history is used for:
- ✅ Providing conversational context to the LLM
- ✅ Helping the LLM understand follow-up references ("yes", "that's correct")

The conversation history is **not** used for:
- ❌ Determining what the current state is
- ❌ Deciding which fields are filled
- ❌ Generating the document

State is always read from `SessionStore.state`, never derived from history.

---

## Known Limitations & Production Improvements

### Current Limitations

| Limitation | Reason | Production Solution |
|-----------|--------|-------------------|
| In-memory state | Simplicity for exercise | Redis or PostgreSQL |
| No authentication | Not required by brief | JWT / OAuth2 |
| No rate limiting | Not required | FastAPI middleware or API gateway |
| Single-process | In-memory store | Sticky sessions or shared store |
| No audit logging | Not required | Structured logging + audit trail |
| Mock has limitations | Regex-based, not AI | Groq API for production |
| No i18n | English only | gettext or similar |

### Production Improvements

1. **Persistent Session Store**: Replace `SessionStore` with Redis (for TTL/expiry) or PostgreSQL (for audit trail)
2. **Authentication**: Add user accounts with JWT tokens; associate sessions with users
3. **Observability**: Structured logging, request tracing, LLM call metrics (latency, token usage)
4. **Rate Limiting**: Per-session and per-IP rate limits to prevent abuse
5. **Retry Logic**: Exponential backoff for transient Groq API failures
6. **Confirmation Workflow**: Deferred state updates for contradictions (currently optimistic)
7. **PDF Generation**: Export the draft document as a downloadable PDF
8. **Field-Level Confidence**: Track extraction confidence scores per field
9. **Multi-Language Support**: Internationalise prompts and UI
10. **Deployment**: Docker containerisation + health checks + graceful shutdown
11. **PII Handling**: Encrypt sensitive data at rest, implement data retention policies
12. **Provider Fallback**: Chain multiple LLM providers (Groq → OpenAI → Mock) for resilience
13. **Document Versioning**: Track document revisions across state changes
14. **Prompt/Version Management**: Version prompts alongside code releases

---

## License

This is a fictional demonstration project for educational and evaluation purposes only. Not legal advice.
