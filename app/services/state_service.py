"""
State Service — owns all state transition logic.

Responsibilities (Single Responsibility):
  - Merge validated partial updates into canonical state
  - Detect contradictions requiring user confirmation
  - Determine which fields are still missing
  - Evaluate whether intake is complete

The LLM extracts.  This module *decides*.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from app.models import (
    ConversationStatus,
    ExecutorInfo,
    PendingConfirmation,
    PartialStateUpdate,
    PersonalWishesState,
)


# ---------------------------------------------------------------------------
# Required-field definitions
# ---------------------------------------------------------------------------

REQUIRED_FIELDS: List[str] = [
    "full_name",
    "home_address",
    "covers_worldwide_assets",
    "has_children",
    "executor",
]

CONDITIONAL_FIELDS = {
    # field → condition function on state
    "children_names": lambda s: s.has_children is True,
}

OPTIONAL_FIELDS: List[str] = [
    "specific_gifts",
    "additional_wishes",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def merge_state_update(
    current_state: PersonalWishesState,
    update: PartialStateUpdate,
) -> Tuple[PersonalWishesState, Optional[PendingConfirmation]]:
    """
    Apply a validated partial update to the canonical state.

    Returns:
        (new_state, pending_confirmation)
        If a contradiction is detected the update is still applied, but a
        PendingConfirmation object is returned so the conversation layer
        can ask the user to confirm.
    """
    state_dict = current_state.model_dump()
    pending: Optional[PendingConfirmation] = None

    # -- scalar fields -------------------------------------------------------
    if update.full_name is not None:
        pending = _check_contradiction(
            current_state.full_name, update.full_name, "full_name", pending
        )
        state_dict["full_name"] = update.full_name

    if update.home_address is not None:
        pending = _check_contradiction(
            current_state.home_address, update.home_address, "home_address", pending
        )
        state_dict["home_address"] = update.home_address

    if update.covers_worldwide_assets is not None:
        old_str = _bool_display(current_state.covers_worldwide_assets)
        new_str = _bool_display(update.covers_worldwide_assets)
        pending = _check_contradiction(old_str, new_str, "covers_worldwide_assets", pending)
        state_dict["covers_worldwide_assets"] = update.covers_worldwide_assets

    # -- children ------------------------------------------------------------
    if update.has_children is not None:
        old_str = _bool_display(current_state.has_children)
        new_str = _bool_display(update.has_children)
        pending = _check_contradiction(old_str, new_str, "has_children", pending)
        state_dict["has_children"] = update.has_children
        if update.has_children is False:
            state_dict["children_names"] = []

    if update.children_names is not None and len(update.children_names) > 0:
        state_dict["children_names"] = update.children_names
        state_dict["has_children"] = True

    # -- executor ------------------------------------------------------------
    exec_info = current_state.executor or ExecutorInfo()
    exec_dict = exec_info.model_dump()

    if update.executor_name is not None:
        pending = _check_contradiction(
            exec_dict.get("name"), update.executor_name, "executor.name", pending
        )
        exec_dict["name"] = update.executor_name

    if update.executor_relationship is not None:
        exec_dict["relationship"] = update.executor_relationship

    # Create executor object if any field was set (including empty string = declined)
    if exec_dict["name"] is not None or exec_dict["relationship"]:
        state_dict["executor"] = exec_dict

    # -- list / optional fields ----------------------------------------------
    if update.specific_gifts is not None:
        state_dict["specific_gifts"] = update.specific_gifts

    if update.additional_wishes is not None:
        state_dict["additional_wishes"] = update.additional_wishes

    return PersonalWishesState(**state_dict), pending


def get_missing_fields(state: PersonalWishesState) -> List[str]:
    """Return a list of field names that still need to be collected."""
    missing: List[str] = []

    if state.full_name is None:
        missing.append("full_name")
    if state.home_address is None:
        missing.append("home_address")
    if state.covers_worldwide_assets is None:
        missing.append("covers_worldwide_assets")
    if state.has_children is None:
        missing.append("has_children")
    if state.has_children is True and (state.children_names is None or len(state.children_names) == 0):
        missing.append("children_names")
    if state.executor is None or state.executor.name is None:
        missing.append("executor_name")
    if state.executor is not None and state.executor.name and state.executor.relationship is None:
        missing.append("executor_relationship")
    if state.specific_gifts is None:
        missing.append("specific_gifts")
    if state.additional_wishes is None:
        missing.append("additional_wishes")

    return missing


def compute_status(state: PersonalWishesState) -> ConversationStatus:
    """Determine the overall intake workflow status from current state."""
    missing = get_missing_fields(state)
    # Required + conditional fields determine completeness
    required_missing = [
        f for f in missing
        if f in (
            "full_name", "home_address", "covers_worldwide_assets",
            "has_children", "children_names", "executor_name",
        )
    ]
    if not required_missing:
        return ConversationStatus.COMPLETE
    return ConversationStatus.COLLECTING


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _check_contradiction(
    old_value, new_value, field_name: str,
    existing_pending: Optional[PendingConfirmation],
) -> Optional[PendingConfirmation]:
    """
    If the old value is not None/empty and differs from the new value,
    return a PendingConfirmation.  Only the *first* contradiction is reported
    to keep the UX simple.
    """
    if existing_pending is not None:
        return existing_pending  # already have one queued

    if old_value is not None and str(old_value).strip() and str(old_value) != str(new_value):
        return PendingConfirmation(
            field=field_name,
            old_value=str(old_value),
            new_value=str(new_value),
        )
    return None


def _bool_display(value: Optional[bool]) -> Optional[str]:
    """Convert Optional[bool] to a display string for contradiction checks."""
    if value is None:
        return None
    return "Yes" if value else "No"
