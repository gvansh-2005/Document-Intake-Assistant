"""
Document Service — deterministic draft document generation.

Responsibilities (Single Responsibility):
  - Transform PersonalWishesState into a human-readable draft document
  - Apply prominent fictional disclaimer
  - Represent unknown fields explicitly as [UNKNOWN / UNCONFIRMED]

Design decision:
    We generate the document deterministically from validated state rather
    than asking the LLM to write it.  This eliminates hallucination risk
    and ensures the document is always consistent with the state.
"""

from __future__ import annotations

from app.models import PersonalWishesState


_DISCLAIMER = (
    "=====================================================================\n"
    " [FICTIONAL DOCUMENT - NOT LEGAL ADVICE - FOR TESTING PURPOSES ONLY] \n"
    "=====================================================================\n\n"
)

_FOOTER = (
    "---------------------------------------------------------------------\n"
    "Status: DRAFT PREVIEW (Updates dynamically as conversation proceeds)\n"
)


def generate_draft_document(state: PersonalWishesState) -> str:
    """
    Generate a formatted draft personal wishes document from canonical state.

    Every section maps directly to a state field.  Unknown values are
    explicitly rendered as [UNKNOWN / UNCONFIRMED] so the user can see
    exactly what information is still missing.
    """
    sections = [
        _DISCLAIMER,
        "DRAFT PERSONAL WISHES DOCUMENT\n",
        "---------------------------------------------------------------------\n\n",
        _section_declarant(state),
        _section_jurisdiction(state),
        _section_family(state),
        _section_executor(state),
        _section_gifts(state),
        _section_wishes(state),
        _FOOTER,
    ]
    return "".join(sections)


# ---------------------------------------------------------------------------
# Section builders (each is a pure function of state)
# ---------------------------------------------------------------------------

def _section_declarant(state: PersonalWishesState) -> str:
    return (
        "1. DECLARANT INFORMATION\n"
        f"   - Name: {state.full_name or '[UNKNOWN / UNCONFIRMED]'}\n"
        f"   - Residence: {state.home_address or '[UNKNOWN / UNCONFIRMED]'}\n\n"
    )


def _section_jurisdiction(state: PersonalWishesState) -> str:
    if state.covers_worldwide_assets is True:
        scope = "Yes (Applies globally)"
    elif state.covers_worldwide_assets is False:
        scope = "No (Specific jurisdiction only)"
    else:
        scope = "[UNKNOWN / UNCONFIRMED]"

    return (
        "2. JURISDICTION & ASSET SCOPE\n"
        f"   - Covers Worldwide Assets: {scope}\n\n"
    )


def _section_family(state: PersonalWishesState) -> str:
    lines = "3. FAMILY & DEPENDENTS\n"

    if state.has_children is True:
        lines += "   - Children: Yes\n"
        if state.children_names:
            lines += f"   - Children Names: {', '.join(state.children_names)}\n\n"
        else:
            lines += "   - Children Names: [Names pending confirmation]\n\n"
    elif state.has_children is False:
        lines += "   - Children: None declared\n\n"
    else:
        lines += "   - Children Status: [UNKNOWN / UNCONFIRMED]\n\n"

    return lines


def _section_executor(state: PersonalWishesState) -> str:
    lines = "4. EXECUTOR APPOINTMENT\n"

    if state.executor and (state.executor.name or state.executor.relationship):
        name = state.executor.name or "[Name Unspecified]"
        rel = f" ({state.executor.relationship})" if state.executor.relationship else ""
        lines += f"   - Appointed Executor: {name}{rel}\n\n"
    else:
        lines += "   - Appointed Executor: [UNKNOWN / UNCONFIRMED]\n\n"

    return lines


def _section_gifts(state: PersonalWishesState) -> str:
    lines = "5. SPECIFIC GIFTS & BEQUESTS\n"

    if state.specific_gifts is None:
        lines += "   - [Not yet discussed]\n\n"
    elif len(state.specific_gifts) == 0:
        lines += "   - None specified\n\n"
    else:
        for gift in state.specific_gifts:
            lines += f"   - Item/Gift: {gift}\n"
        lines += "\n"

    return lines


def _section_wishes(state: PersonalWishesState) -> str:
    lines = "6. ADDITIONAL WISHES & INSTRUCTIONS\n"

    if state.additional_wishes is None:
        lines += "   - [Not yet discussed]\n\n"
    elif state.additional_wishes == "":
        lines += "   - None specified\n\n"
    else:
        lines += f"   - {state.additional_wishes}\n\n"

    return lines


# ---------------------------------------------------------------------------
# PDF generation
# ---------------------------------------------------------------------------

def generate_pdf(state: PersonalWishesState) -> bytes:
    """
    Generate a professional PDF of the Personal Wishes Document.

    Uses the same deterministic logic as generate_draft_document — the PDF
    is just a presentation layer on top of the same validated state.
    No LLM involvement.
    """
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=25)
    pdf.add_page()

    # ── Header / Disclaimer ──
    pdf.set_fill_color(220, 38, 38)  # Red banner
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 8, "FICTIONAL DOCUMENT - NOT LEGAL ADVICE - FOR TESTING PURPOSES ONLY", new_x="LMARGIN", new_y="NEXT", align="C", fill=True)
    pdf.ln(6)

    # ── Title ──
    pdf.set_text_color(30, 41, 59)
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, "Personal Wishes Document", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(2)
    pdf.set_draw_color(99, 102, 241)
    pdf.set_line_width(0.8)
    pdf.line(30, pdf.get_y(), 180, pdf.get_y())
    pdf.ln(8)

    # ── Helper to add a section ──
    def add_section(number: str, title: str, items: list):
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(55, 65, 81)
        pdf.cell(0, 8, f"{number}. {title}", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)
        pdf.set_text_color(75, 85, 99)
        for label, value in items:
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(8)  # indent
            pdf.cell(45, 6, f"{label}:", new_x="END")
            pdf.set_font("Helvetica", "", 10)
            # Use cell for short values, multi_cell for long ones
            val = str(value)
            if len(val) < 60:
                pdf.cell(0, 6, val, new_x="LMARGIN", new_y="NEXT")
            else:
                pdf.multi_cell(0, 6, val)
        pdf.ln(4)

    # ── 1. Declarant ──
    add_section("1", "DECLARANT INFORMATION", [
        ("Name", state.full_name or "[UNKNOWN / UNCONFIRMED]"),
        ("Residence", state.home_address or "[UNKNOWN / UNCONFIRMED]"),
    ])

    # ── 2. Jurisdiction ──
    if state.covers_worldwide_assets is True:
        scope = "Yes (Applies globally)"
    elif state.covers_worldwide_assets is False:
        scope = "No (Specific jurisdiction only)"
    else:
        scope = "[UNKNOWN / UNCONFIRMED]"
    add_section("2", "JURISDICTION & ASSET SCOPE", [
        ("Worldwide Assets", scope),
    ])

    # ── 3. Family ──
    family_items = []
    if state.has_children is True:
        family_items.append(("Has Children", "Yes"))
        if state.children_names:
            family_items.append(("Children Names", ", ".join(state.children_names)))
        else:
            family_items.append(("Children Names", "[Names pending confirmation]"))
    elif state.has_children is False:
        family_items.append(("Has Children", "None declared"))
    else:
        family_items.append(("Children Status", "[UNKNOWN / UNCONFIRMED]"))
    add_section("3", "FAMILY & DEPENDENTS", family_items)

    # ── 4. Executor ──
    if state.executor and (state.executor.name or state.executor.relationship):
        name = state.executor.name or "[Name Unspecified]"
        rel = f" ({state.executor.relationship})" if state.executor.relationship else ""
        executor_val = f"{name}{rel}"
    else:
        executor_val = "[UNKNOWN / UNCONFIRMED]"
    add_section("4", "EXECUTOR APPOINTMENT", [
        ("Appointed Executor", executor_val),
    ])

    # ── 5. Gifts ──
    gifts_items = []
    if state.specific_gifts is None:
        gifts_items.append(("Gifts", "[Not yet discussed]"))
    elif len(state.specific_gifts) == 0:
        gifts_items.append(("Gifts", "None specified"))
    else:
        for i, gift in enumerate(state.specific_gifts, 1):
            gifts_items.append((f"Item {i}", gift))
    add_section("5", "SPECIFIC GIFTS & BEQUESTS", gifts_items)

    # ── 6. Additional Wishes ──
    if state.additional_wishes is None:
        wishes_val = "[Not yet discussed]"
    elif state.additional_wishes == "":
        wishes_val = "None specified"
    else:
        wishes_val = state.additional_wishes
    add_section("6", "ADDITIONAL WISHES & INSTRUCTIONS", [
        ("Wishes", wishes_val),
    ])

    # ── Footer ──
    pdf.ln(6)
    pdf.set_draw_color(156, 163, 175)
    pdf.set_line_width(0.3)
    pdf.line(20, pdf.get_y(), 190, pdf.get_y())
    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(156, 163, 175)
    pdf.cell(0, 5, "Generated by Document Intake Assistant | Fictional demonstration only", new_x="LMARGIN", new_y="NEXT", align="C")

    return pdf.output()

