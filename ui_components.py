"""Reusable V2.0 Design System components for ReFind AI.

Centralizes headers, metric ribbons, restrained status badges, metadata strips,
comparison chips, integrated native card interiors, empty states, and academic
AI reference content so every screen shares one cohesive visual language.
"""

from __future__ import annotations

import html
from datetime import date
from typing import Any, Optional

import streamlit as st

from ai_matcher import MatchConfig, MatchResult
from utils import format_display_date, parse_iso_date, score_tier


def esc(value: Any) -> str:
    """Safely HTML-escape any dynamic text prior to HTML rendering."""
    return html.escape("" if value is None else str(value))


def format_relative_date(date_str: Optional[str]) -> str:
    """Return a concise, human-friendly date label (e.g., Today, Yesterday, Nov 20)."""
    parsed = parse_iso_date(date_str)
    if parsed is None:
        return "Date unknown"
    delta = (date.today() - parsed).days
    if delta == 0:
        return "Today"
    if delta == 1:
        return "Yesterday"
    if 2 <= delta <= 6:
        return f"{delta} days ago"
    return format_display_date(date_str)


def render_page_header(
    title: str,
    subtitle: str,
    eyebrow: Optional[str] = None,
) -> None:
    """Render a calm, restrained page header with consistent spacing and hierarchy."""
    eyebrow_html = f'<div class="rf-eyebrow">{esc(eyebrow)}</div>' if eyebrow else ""
    st.markdown(
        f"""
        <div class="rf-page-header">
            {eyebrow_html}
            <h1 class="rf-page-title">{esc(title)}</h1>
            <p class="rf-page-subtitle">{esc(subtitle)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_label(title: str, caption: Optional[str] = None) -> None:
    """Render a clean section title with optional muted helper text."""
    caption_html = f'<p class="rf-section-caption">{esc(caption)}</p>' if caption else ""
    st.markdown(
        f"""
        <div class="rf-section-head">
            <h2 class="rf-section-title">{esc(title)}</h2>
            {caption_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_ribbon(stats: dict[str, int], compact: bool = False) -> None:
    """Render a sleek 4-metric summary ribbon replacing bulky spreadsheet KPI boxes."""
    cls = "rf-metric-ribbon compact" if compact else "rf-metric-ribbon"
    st.markdown(
        f"""
        <div class="{cls}" role="region" aria-label="Campus lost and found summary">
            <div class="rf-metric-cell">
                <div class="rf-metric-label">Active Reports</div>
                <div class="rf-metric-num accent-blue">{int(stats["active_reports"])}</div>
                <div class="rf-metric-sub">Open for AI matching</div>
            </div>
            <div class="rf-metric-cell">
                <div class="rf-metric-label">Lost Items</div>
                <div class="rf-metric-num">{int(stats["total_lost"])}</div>
                <div class="rf-metric-sub">{int(stats["active_lost"])} currently active</div>
            </div>
            <div class="rf-metric-cell">
                <div class="rf-metric-label">Found Items</div>
                <div class="rf-metric-num">{int(stats["total_found"])}</div>
                <div class="rf-metric-sub">{int(stats["active_found"])} currently active</div>
            </div>
            <div class="rf-metric-cell">
                <div class="rf-metric-label">Recovered</div>
                <div class="rf-metric-num accent-green">{int(stats["total_recovered"])}</div>
                <div class="rf-metric-sub">Resolved campus cases</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state(title: str, description: str) -> None:
    """Render a calm, centered empty-state surface."""
    st.markdown(
        f"""
        <div class="rf-empty-state">
            <div class="rf-empty-title">{esc(title)}</div>
            <div class="rf-empty-desc">{esc(description)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_report_badges_html(
    report_type: str,
    status: str,
    *,
    show_active_badge: bool = False,
) -> str:
    """Return restrained status pills for report type and lifecycle state."""
    rtype = str(report_type).lower()
    st_val = str(status).lower()
    type_cls = "pill-lost" if rtype == "lost" else "pill-found"
    pills = [
        f'<span class="rf-pill {type_cls}"><span class="rf-pill-dot"></span>{esc(rtype.capitalize())}</span>'
    ]
    if st_val == "recovered":
        pills.append(
            '<span class="rf-pill pill-recovered"><span class="rf-pill-dot"></span>Recovered</span>'
        )
    elif show_active_badge:
        pills.append('<span class="rf-pill pill-neutral">Active</span>')
    return "".join(pills)


def build_metadata_html(
    *,
    category: str,
    location: str,
    report_date: str,
    contact: Optional[str] = None,
    date_suffix: Optional[str] = None,
) -> str:
    """Build a clean, scannable horizontal metadata strip."""
    date_text = format_display_date(report_date)
    if date_suffix:
        date_text = f"{date_text} ({date_suffix})"
    items = [
        f'<span class="rf-meta-chip"><strong>Category:</strong> {esc(category)}</span>',
        f'<span class="rf-meta-chip"><strong>Location:</strong> {esc(location)}</span>',
        f'<span class="rf-meta-chip"><strong>Date:</strong> {esc(date_text)}</span>',
    ]
    if contact:
        items.append(
            f'<span class="rf-meta-chip"><strong>Ref:</strong> {esc(contact)}</span>'
        )
    return f'<div class="rf-meta-strip">{"".join(items)}</div>'


def render_report_body(
    report: dict[str, Any],
    *,
    compact: bool = False,
    show_contact: bool = True,
    show_active_badge: bool = True,
) -> None:
    """Render report header, description, and metadata INSIDE a native st.container(border=True)."""
    desc = str(report.get("description") or "")
    if compact and len(desc) > 135:
        desc = desc[:132].rstrip() + "..."

    badges = build_report_badges_html(
        report["report_type"],
        report["status"],
        show_active_badge=show_active_badge,
    )
    meta_html = build_metadata_html(
        category=str(report.get("category") or "Other"),
        location=str(report.get("location") or "Unspecified"),
        report_date=str(report.get("report_date") or ""),
        contact=str(report.get("contact_label") or "") if (show_contact and not compact) else None,
    )

    notes_html = ""
    if not compact and report.get("additional_details"):
        notes_html = (
            f'<div class="rf-notes-line">'
            f'<strong>Additional notes:</strong> {esc(report["additional_details"])}'
            f"</div>"
        )

    st.markdown(
        f"""
        <div class="rf-card-top">
            <div class="rf-card-heading-group">
                <span class="rf-id-tag">#{int(report["id"])}</span>
                <span class="rf-item-title">{esc(report["title"])}</span>
            </div>
            <div class="rf-pill-row">{badges}</div>
        </div>
        <p class="rf-item-desc">{esc(desc)}</p>
        {meta_html}
        {notes_html}
        """,
        unsafe_allow_html=True,
    )


def build_comparison_chips_html(
    source_report: dict[str, Any],
    result: MatchResult,
) -> str:
    """Render concise visual cues highlighting both agreements AND differences."""
    chips: list[str] = []

    # Category comparison
    if result.category_score >= 1.0:
        chips.append(
            f'<span class="rf-signal-chip chip-match">Same category &middot; {esc(result.candidate["category"])}</span>'
        )
    else:
        chips.append(
            f'<span class="rf-signal-chip chip-diff">Different category ({esc(source_report["category"])} vs. {esc(result.candidate["category"])})</span>'
        )

    # Location comparison
    if result.location_match_reason == "exact":
        chips.append('<span class="rf-signal-chip chip-match">Exact location match</span>')
    elif result.location_match_reason in ("token_subset", "token_overlap") and result.location_score > 0:
        shared = ", ".join(result.shared_location_tokens[:2])
        detail = f" ({shared})" if shared else ""
        chips.append(
            f'<span class="rf-signal-chip chip-match">Compatible location{esc(detail)}</span>'
        )
    else:
        chips.append('<span class="rf-signal-chip chip-diff">Different locations</span>')

    # Date proximity comparison
    if result.days_apart is None:
        chips.append('<span class="rf-signal-chip chip-neutral">Date unavailable</span>')
    elif result.days_apart == 0:
        chips.append('<span class="rf-signal-chip chip-match">Reported same day</span>')
    elif result.days_apart <= 3:
        unit = "day" if result.days_apart == 1 else "days"
        chips.append(
            f'<span class="rf-signal-chip chip-match">{result.days_apart} {unit} apart</span>'
        )
    elif result.days_apart <= 7:
        chips.append(
            f'<span class="rf-signal-chip chip-neutral">{result.days_apart} days apart</span>'
        )
    else:
        chips.append(
            f'<span class="rf-signal-chip chip-diff">{result.days_apart} days apart</span>'
        )

    return f'<div class="rf-signal-chip-row">{"".join(chips)}</div>'


def render_score_breakdown_grid(result: MatchResult, config: MatchConfig) -> None:
    """Render the exact V1.2 weighted contribution breakdown inside a progressive-disclosure drawer."""
    sem_pct = round(result.semantic_score * 100)
    cat_pct = round(result.category_score * 100)
    loc_pct = round(result.location_score * 100)
    date_pct = round(result.date_score * 100)

    max_sem = round(config.weight_semantic * 100.0, 1)
    max_cat = round(config.weight_category * 100.0, 1)
    max_loc = round(config.weight_location * 100.0, 1)
    max_date = round(config.weight_date * 100.0, 1)

    sem_pts = result.semantic_contribution
    cat_pts = result.category_contribution
    loc_pts = result.location_contribution
    date_pts = result.date_contribution

    st.markdown(
        f"""
        <div class="rf-breakdown-summary">
            <span><strong>Weighted Sum:</strong> {sem_pts:.1f} + {cat_pts:.1f} + {loc_pts:.1f} + {date_pts:.1f} = <strong>{result.final_score:.1f} / 100</strong></span>
            <span>Raw Cosine Similarity: <code>{result.raw_cosine_similarity:.3f}</code></span>
        </div>
        <div class="rf-signal-grid">
            <div class="rf-signal-tile">
                <div class="rf-signal-top"><span>Semantic ({max_sem:.0f}%)</span><span>{sem_pct}%</span></div>
                <div class="rf-signal-pts">+{sem_pts:.1f} <span>/ {max_sem:.0f} pts</span></div>
                <div class="rf-bar-track" role="progressbar" aria-label="Semantic similarity" aria-valuenow="{sem_pct}" aria-valuemin="0" aria-valuemax="100">
                    <div class="rf-bar-fill {'fill-strong' if sem_pct >= 70 else ''}" style="width:{sem_pct}%;"></div>
                </div>
            </div>
            <div class="rf-signal-tile">
                <div class="rf-signal-top"><span>Category ({max_cat:.0f}%)</span><span>{cat_pct}%</span></div>
                <div class="rf-signal-pts">+{cat_pts:.1f} <span>/ {max_cat:.0f} pts</span></div>
                <div class="rf-bar-track" role="progressbar" aria-label="Category match" aria-valuenow="{cat_pct}" aria-valuemin="0" aria-valuemax="100">
                    <div class="rf-bar-fill {'fill-strong' if cat_pct >= 100 else ''}" style="width:{cat_pct}%;"></div>
                </div>
            </div>
            <div class="rf-signal-tile">
                <div class="rf-signal-top"><span>Location ({max_loc:.0f}%)</span><span>{loc_pct}%</span></div>
                <div class="rf-signal-pts">+{loc_pts:.1f} <span>/ {max_loc:.0f} pts</span></div>
                <div class="rf-bar-track" role="progressbar" aria-label="Location similarity" aria-valuenow="{loc_pct}" aria-valuemin="0" aria-valuemax="100">
                    <div class="rf-bar-fill {'fill-strong' if loc_pct >= 70 else ''}" style="width:{loc_pct}%;"></div>
                </div>
            </div>
            <div class="rf-signal-tile">
                <div class="rf-signal-top"><span>Date Proximity ({max_date:.0f}%)</span><span>{date_pct}%</span></div>
                <div class="rf-signal-pts">+{date_pts:.1f} <span>/ {max_date:.0f} pts</span></div>
                <div class="rf-bar-track" role="progressbar" aria-label="Date proximity" aria-valuenow="{date_pct}" aria-valuemin="0" aria-valuemax="100">
                    <div class="rf-bar-fill {'fill-strong' if date_pct >= 70 else ''}" style="width:{date_pct}%;"></div>
                </div>
            </div>
        </div>
        <div class="rf-prob-footnote">
            <strong>Academic Transparency:</strong> {result.final_score:.1f} / 100 is a deterministic weighted
            similarity score for ranking candidates — <em>not</em> a statistically calibrated probability that
            two reports refer to the identical physical object. Always verify ownership before handover.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_how_ai_works_content() -> None:
    """Render concise, viva-ready explanation of ReFind AI's NLP pipeline & math."""
    st.markdown(
        """
        #### How ReFind AI Matches Reports

        1. **Text Preparation:** Combines item title, description, and optional identifying notes while strictly isolating opposite report types (`Lost` ↔ `Found`).
        2. **Dense Sentence Embeddings (`all-MiniLM-L6-v2`):** A 6-layer MiniLM transformer running locally on CPU encodes natural language into a 384-dimensional semantic vector $\\mathbf{v} \\in \\mathbb{R}^{384}$ so paraphrases like *"blue water bottle"* and *"navy reusable drinking bottle"* map close together.
        3. **Cosine Similarity (Scikit-learn):** Measures angular alignment between vectors, clamped to $[0, 1]$:
           $$\\cos(\\theta) = \\frac{\\mathbf{u} \\cdot \\mathbf{v}}{\\|\\mathbf{u}\\|_2 \\|\\mathbf{v}\\|_2}$$
        4. **Hybrid Contextual Fusion (0–100):**
           $$\\text{Final Score} = 100 \\times (0.70 \\cdot S_{\\text{sem}} + 0.15 \\cdot S_{\\text{cat}} + 0.10 \\cdot S_{\\text{loc}} + 0.05 \\cdot S_{\\text{date}})$$
           - **Category ($S_{\\text{cat}}$):** Exact normalized category match (`1.0` or `0.0`).
           - **Location ($S_{\\text{loc}}$):** Whole-token lexical comparison (exact `1.0`, token subset `0.85`, Jaccard overlap; no fabricated GPS distances).
           - **Date Proximity ($S_{\\text{date}}$):** Exponential decay $\\exp(-\\Delta d / 14)$ over calendar days apart.

        > **Similarity Score ≠ Probability:** A score such as **85.9 / 100** indicates strong relative similarity across text and context — it does **not** mean an 85.9% statistical probability that the items are identical. Human verification is always required.
        """
    )


def match_tier_display(score: float) -> tuple[str, str]:
    """Return calm tier label and CSS class for match badges."""
    label, old_cls = score_tier(score)
    mapping = {
        "badge-strong": "pill-strong",
        "badge-moderate": "pill-moderate",
        "badge-possible": "pill-possible",
        "badge-low": "pill-neutral",
    }
    return label, mapping.get(old_cls, "pill-neutral")
