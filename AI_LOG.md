# AI Log & Design Decision Record

This log documents key prompts, notable design iterations, LLM output issues encountered, and corrections applied during development.

---

## Iteration 1 — Initial Architecture: Conversation-Driven State

**Prompt**: "Design a conversational document intake application that collects user information through chat."

**Initial approach**: Relied on the LLM's conversation history as the primary source of truth. The system prompt contained all previously collected values and the LLM was asked to maintain them.

**Problem discovered**: The LLM would occasionally "forget" or alter previously confirmed values when the conversation grew long. For example, after 6 turns, the model dropped `covers_worldwide_assets: true` from its response even though the user had confirmed it earlier.

**Decision**: Moved to an explicit Pydantic state object (`PersonalWishesState`) owned by the backend. The LLM is now responsible *only* for extracting changes from the latest user message, while application code (`state_service.py`) validates and merges those changes.

> **Principle**: The LLM understands language. Application code controls state.

---

## Iteration 2 — Client-Owned vs Backend-Owned State

**Initial approach**: The frontend maintained `currentState` in JavaScript and sent it to the backend with every request.

**Problem**: This meant the browser was the source of truth. A stale tab, network error, or race condition could cause the client to send an outdated state, silently overwriting newer values.

**Decision**: Introduced a `SessionStore` on the backend. The frontend now sends only `session_id` + `message`. The backend retrieves the canonical state from its store, processes the update, and returns the new state.

**Trade-off**: State is lost on server restart (in-memory store). Documented as a known limitation with Redis/PostgreSQL as the production upgrade path.

---

## Iteration 3 — None vs Empty List Semantics

**Problem**: The original schema used `children_names: List[str] = []` and `specific_gifts: List[str] = []`. This created an ambiguity:

```json
{ "specific_gifts": [] }
```

Could mean:
1. User hasn't been asked about gifts yet → should ask
2. User explicitly said "no gifts" → should not ask again

**Fix**: Changed to `Optional[List[str]] = None`:
- `None` → field not yet discussed
- `[]` → user explicitly confirmed no items

This distinction is reflected in the document generator:
- `None` → "[Not yet discussed]"
- `[]` → "None specified"

---

## Iteration 4 — Handling Ambiguity (Executor Without Relationship)

**Example LLM interaction**:

```
User: "My executor is James."

Model output:
{
    "executor_name": "James",
    "executor_relationship": "friend"   ← HALLUCINATED
}
```

**Problem**: The model invented "friend" without any evidence from the user's message.

**Correction applied**:
1. Strengthened the system prompt: "Do NOT invent, assume, or hallucinate any facts."
2. Added `get_missing_fields()` in `state_service.py` which detects that `executor.relationship` is `None` when `executor.name` is set, and triggers a follow-up question.

**Result**: The system now stores `executor.name = "James"`, `executor.relationship = None`, and asks: "What is James's relationship to you?"

---

## Iteration 5 — Contradiction Detection

**Scenario**:
```
Turn 1: "My name is Jane Smith."     → state: full_name = "Jane Smith"
Turn 2: "My name is Jane Doe."      → state: full_name = "Jane Doe"
```

**Initial behaviour**: The state silently replaced "Jane Smith" with "Jane Doe". No indication to the user that a change occurred.

**Improvement**: Added `_check_contradiction()` in the state merge logic. When an existing non-null value is replaced with a different value, a `PendingConfirmation` object is returned. The conversation service includes a notification in the assistant's reply:

> "I noticed you previously provided 'Jane Smith' for your name, but now mentioned 'Jane Doe'. I've updated it to the new value. If that's not correct, please let me know!"

---

## Iteration 6 — Mock LLM Robustness

**Problem**: The initial mock used simplistic keyword matching:
```python
if "worldwide" in msg:
    update.covers_worldwide_assets = True
```

This failed for:
- "No, I don't want worldwide coverage" → incorrectly set `True`
- "No kids" → not handled
- "My brother James is my executor" → executor name not extracted

**Fix**: Rewrote the mock with regex patterns that handle:
- Negation ("don't", "no", "not")
- Multiple relationship words (brother, sister, friend, solicitor, etc.)
- Multiple extraction patterns per field
- Multi-field extraction in a single message
- Follow-up question selection based on which fields are still missing

---

## Iteration 7 — Frontend XSS Prevention

**Problem**: The original frontend used `innerHTML` to render messages:
```javascript
chatBox.innerHTML += `<div>${message}</div>`;
```

This allowed HTML injection. If a user typed `<img src=x onerror=alert(1)>`, the script would execute.

**Fix**: Replaced all `innerHTML` usage with safe DOM construction:
```javascript
const div = document.createElement('div');
div.textContent = message;  // safe — treats content as text, not HTML
chatMessages.appendChild(div);
```

---

## Iteration 8 — Conversation History Optimisation

**Problem**: Sending the entire conversation history to the LLM would grow unbounded, increasing latency and token cost.

**Solution**: Implemented a sliding window via `get_recent_history(limit=10)` in the session store. Only the last 10 messages are sent to the LLM. This keeps token usage bounded while preserving enough context for coherent conversation.

**Trade-off**: Very early messages may be lost from the LLM's context. However, since extracted state is preserved in the backend store, no *information* is lost — only conversational context.

---

## Iteration 9 — Prompt Externalisation

**Decision**: Moved the system prompt from inline Python string to `app/prompts/intake_prompt.txt`.

**Rationale**:
1. Easier to iterate on prompt wording without touching Python code
2. Cleaner separation of concerns (prompt engineering vs. application logic)
3. Can be version-controlled and reviewed independently

---

## Iteration 10 — Test Fixture Design

**Approach**: Created four fixture files covering the testing scenarios required by the brief:

| Fixture | Purpose |
|---------|---------|
| `valid.json` | Multi-field extraction from one message |
| `ambiguous.json` | Executor name without relationship → triggers follow-up |
| `malformed.json` | Invalid type (`has_children: "banana"`) → Pydantic rejects |
| `correction.json` | Name change → contradiction detection |

Tests load these fixtures and verify both the parsing and the state machine behaviour.
