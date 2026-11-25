"""ReFind AI V2.0 — Intelligent AI-Powered Campus Lost & Found Platform.

Built on the verified V1.2 engineering baseline with a unified V2.0 Design System:
- Action-first Dashboard with sleek metric ribbon, lazy AI highlights, and activity feed
- Split-workspace AI Matching studio with progressive-disclosure score breakdowns
- Guided Lost & Found reporting forms with above-the-fold post-submit AI results
- Browse Campus Reports directory with keyword search, sorting, pagination, editing,
  and two-step delete confirmation with safe upload cleanup
"""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Optional

import streamlit as st

import database
from ai_matcher import (
    DEFAULT_MODEL_NAME,
    EmbeddingEngine,
    MatchConfig,
    MatchResult,
    ModelLoadError,
    find_matches,
    find_top_match_highlights,
)
from database import (
    count_seed_reports,
    delete_report_with_cleanup,
    delete_seed_reports_with_cleanup,
    find_duplicate_report,
    get_all_reports,
    get_dashboard_stats,
    get_report_by_id,
    get_reports_version,
    init_db,
    insert_report,
    update_report,
    update_report_status,
)
from seed_data import ensure_seed_data, seed_database
from ui_components import (
    build_comparison_chips_html,
    build_metadata_html,
    build_report_badges_html,
    esc,
    format_relative_date,
    match_tier_display,
    render_empty_state,
    render_how_ai_works_content,
    render_metric_ribbon,
    render_page_header,
    render_report_body,
    render_score_breakdown_grid,
    render_section_label,
)
from utils import (
    BASE_DIR,
    CAMPUS_LOCATION_SUGGESTIONS,
    CATEGORIES,
    parse_iso_date,
    resolve_image_path,
    save_uploaded_image,
)

PAGES: list[str] = [
    "Dashboard",
    "AI Matching",
    "Report Lost Item",
    "Report Found Item",
    "Browse Reports",
]

DEMO_PRESETS: list[tuple[str, str]] = [
    ("Wallet Near Library", "seed_lost_wallet_library"),
    ("Water Bottle (Paraphrase)", "seed_lost_blue_water_bottle"),
    ("Casio Calculator", "seed_lost_scientific_calculator"),
    ("Red vs. Blue Bag", "seed_lost_red_backpack"),
]

SORT_OPTIONS: dict[str, str] = {
    "Newest Submitted": "newest",
    "Oldest Submitted": "oldest",
    "Most Recent Item Date": "date_desc",
    "Earliest Item Date": "date_asc",
    "Title (A–Z)": "title_asc",
}

BROWSE_PAGE_SIZE = 8
CUSTOM_LOCATION_OPTION = "Custom location (type below)"


# ---------------------------------------------------------------------------
# Styling & AI Engine State Management (Preserves V1.2 Lazy Loading & Caching)
# ---------------------------------------------------------------------------


def load_stylesheet() -> None:
    """Inject the V2.0 Design System CSS from assets/styles.css."""
    css_path = BASE_DIR / "assets" / "styles.css"
    if css_path.exists():
        st.markdown(
            f"<style>{css_path.read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True,
        )


@dataclass
class EngineState:
    """Process-scoped AI engine lifecycle state & versioned highlights cache."""

    engine: Optional[EmbeddingEngine] = None
    status: str = "not_initialized"  # "not_initialized" | "ready" | "unavailable"
    error_message: Optional[str] = None
    highlights_cache: dict[tuple[str, str], list[tuple[dict[str, Any], MatchResult]]] = field(
        default_factory=dict
    )


@st.cache_resource
def get_engine_state() -> EngineState:
    """Return process-wide AI engine state WITHOUT loading the transformer model."""
    return EngineState()


def ensure_engine_loaded(force_retry: bool = False) -> Optional[EmbeddingEngine]:
    """Load the Sentence Transformer model lazily on demand and track honest status."""
    state = get_engine_state()
    if state.status == "ready" and state.engine is not None:
        return state.engine
    if state.status == "unavailable" and not force_retry:
        return None

    with st.spinner("Loading ReFind AI semantic model (all-MiniLM-L6-v2)..."):
        try:
            engine = EmbeddingEngine(model_name=DEFAULT_MODEL_NAME)
        except ModelLoadError as exc:
            state.engine = None
            state.status = "unavailable"
            state.error_message = str(exc)
            return None
        state.engine = engine
        state.status = "ready"
        state.error_message = None
        return engine


def try_get_engine(force_retry: bool = False) -> Optional[EmbeddingEngine]:
    """Return ready EmbeddingEngine or render a clear error without keyword fallback."""
    engine = ensure_engine_loaded(force_retry=force_retry)
    if engine is not None:
        return engine

    state = get_engine_state()
    st.error(
        "**AI Embedding Model Unavailable**\n\n"
        f"{state.error_message or 'Unknown model initialization error.'}\n\n"
        "ReFind AI does not silently substitute keyword search for semantic embeddings. "
        "Ensure `sentence-transformers` is installed and internet access is available "
        "for the initial download of `sentence-transformers/all-MiniLM-L6-v2`."
    )
    return None


def get_cached_dashboard_highlights(
    engine: EmbeddingEngine,
) -> list[tuple[dict[str, Any], MatchResult]]:
    """Return vectorized Top AI Match Highlights cached by current database version."""
    state = get_engine_state()
    db_key = str(database._resolve_db_path())
    version_key = get_reports_version()
    cache_key = (db_key, version_key)

    if cache_key in state.highlights_cache:
        return state.highlights_cache[cache_key]

    for existing_key in [k for k in state.highlights_cache if k[0] == db_key]:
        state.highlights_cache.pop(existing_key, None)

    active_lost = get_all_reports(report_type="lost", status="active")
    active_found = get_all_reports(report_type="found", status="active")
    highlights = find_top_match_highlights(
        active_lost,
        active_found,
        engine,
        config=MatchConfig(min_score_threshold=45.0, max_results=1),
        max_highlights=4,
    )
    state.highlights_cache[cache_key] = highlights
    return highlights


def navigate_to(page: str, report_id: Optional[int] = None) -> None:
    """Queue a clean navigation transition for the next rerun."""
    st.session_state["nav_target"] = page
    if report_id is not None:
        st.session_state["pending_match_report_id"] = int(report_id)
    st.rerun()


# ---------------------------------------------------------------------------
# Integrated Match Card Component (Progressive Disclosure + Deliberate Recovery)
# ---------------------------------------------------------------------------


def _concise_explanation(full_text: str) -> str:
    """Return a crisp 1-2 sentence summary for primary card display."""
    sentences = [s.strip() for s in full_text.split(". ") if s.strip()]
    if len(sentences) <= 2:
        return full_text
    joined = ". ".join(sentences[:2])
    return joined if joined.endswith(".") else joined + "."


def render_integrated_match_card(
    *,
    rank: int,
    result: MatchResult,
    source_report: dict[str, Any],
    config: MatchConfig,
    key_prefix: str,
    allow_recovery_actions: bool = True,
    expand_breakdown_default: bool = False,
) -> None:
    """Render an integrated native-container AI match card with progressive disclosure."""
    cand = result.candidate
    cand_id = int(cand["id"])
    src_id = int(source_report["id"]) if source_report.get("id") is not None else None

    tier_label, tier_cls = match_tier_display(result.final_score)
    is_strong = result.final_score >= 75.0
    overall_pct = max(0.0, min(100.0, result.final_score))
    bar_cls = "fill-strong" if is_strong else ""
    rationale_cls = "strong" if is_strong else ""

    rtype = str(cand["report_type"]).lower()
    type_cls = "pill-lost" if rtype == "lost" else "pill-found"
    status = str(cand["status"]).lower()
    status_pill = (
        '<span class="rf-pill pill-recovered"><span class="rf-pill-dot"></span>Recovered</span>'
        if status == "recovered"
        else ""
    )
    rank_pill = (
        '<span class="rf-pill pill-rank">Top Match</span>'
        if rank == 1 and result.final_score >= 60.0
        else f'<span class="rf-pill pill-neutral">Rank #{rank}</span>'
    )

    days_suffix = (
        "same day"
        if result.days_apart == 0
        else (f"{result.days_apart}d apart" if result.days_apart is not None else None)
    )
    meta_html = build_metadata_html(
        category=str(cand.get("category") or "Other"),
        location=str(cand.get("location") or "Unspecified"),
        report_date=str(cand.get("report_date") or ""),
        contact=str(cand.get("contact_label") or "Campus Desk"),
        date_suffix=days_suffix,
    )
    comparison_chips_html = build_comparison_chips_html(source_report, result)

    with st.container(border=True):
        st.markdown(
            f"""
            <div class="rf-match-header">
                <div style="min-width:0; flex:1;">
                    <div class="rf-pill-row" style="margin-bottom:0.4rem;">
                        {rank_pill}
                        <span class="rf-pill {type_cls}"><span class="rf-pill-dot"></span>{esc(rtype.capitalize())} #{cand_id}</span>
                        <span class="rf-pill {tier_cls}">{esc(tier_label)}</span>
                        {status_pill}
                    </div>
                    <div class="rf-item-title">{esc(cand["title"])}</div>
                </div>
                <div class="rf-score-block">
                    <div class="rf-score-value">{result.final_score:.1f}<span> / 100</span></div>
                    <div class="rf-score-caption">Similarity score</div>
                </div>
            </div>

            <div class="rf-bar-track" role="progressbar" aria-label="Overall similarity score"
                 aria-valuenow="{overall_pct:.0f}" aria-valuemin="0" aria-valuemax="100">
                <div class="rf-bar-fill {bar_cls}" style="width:{overall_pct:.1f}%;"></div>
            </div>

            <div class="rf-match-rationale {rationale_cls}">
                {esc(result.explanation)}
            </div>

            {comparison_chips_html}

            <p class="rf-item-desc">{esc(cand["description"])}</p>
            {meta_html}
            """,
            unsafe_allow_html=True,
        )

        with st.expander(
            "Score breakdown & technical details",
            expanded=expand_breakdown_default,
        ):
            render_score_breakdown_grid(result, config)

        img_path = resolve_image_path(cand.get("image_path"))
        if img_path is not None:
            with st.expander(f"View uploaded photo — Report #{cand_id}"):
                st.image(str(img_path), width=320)

        # Deliberate two-step recovery confirmation inside the card
        if allow_recovery_actions and src_id is not None and status == "active":
            confirm_key = f"confirm_recover_{key_prefix}_{cand_id}"
            pending_mode = st.session_state.get(confirm_key)

            if pending_mode == "pair":
                st.warning(
                    f"**Confirm recovery:** Mark both **Report #{src_id} ({source_report['title']})** "
                    f"and **Report #{cand_id} ({cand['title']})** as recovered? "
                    "Please verify ownership details before closing both cases."
                )
                rc1, rc2, _ = st.columns([1.3, 1.0, 2.2])
                with rc1:
                    if st.button(
                        "Confirm Pair Recovery",
                        key=f"{key_prefix}_confirm_pair_{cand_id}",
                        type="primary",
                        width="stretch",
                    ):
                        try:
                            update_report_status(src_id, "recovered")
                            update_report_status(cand_id, "recovered")
                        except (sqlite3.Error, ValueError) as exc:
                            st.error(f"Could not update recovery status: {exc}")
                        else:
                            st.session_state.pop(confirm_key, None)
                            st.session_state["match_flash_success"] = (
                                f"Marked Reports **#{src_id}** and **#{cand_id}** as recovered."
                            )
                            st.rerun()
                with rc2:
                    if st.button(
                        "Cancel",
                        key=f"{key_prefix}_cancel_pair_{cand_id}",
                        width="stretch",
                    ):
                        st.session_state.pop(confirm_key, None)
                        st.rerun()
            elif pending_mode == "single":
                st.warning(
                    f"**Confirm recovery:** Mark **Report #{cand_id} ({cand['title']})** as recovered?"
                )
                rc1, rc2, _ = st.columns([1.3, 1.0, 2.2])
                with rc1:
                    if st.button(
                        f"Confirm #{cand_id} Recovered",
                        key=f"{key_prefix}_confirm_single_{cand_id}",
                        type="primary",
                        width="stretch",
                    ):
                        try:
                            update_report_status(cand_id, "recovered")
                        except (sqlite3.Error, ValueError) as exc:
                            st.error(f"Could not update recovery status: {exc}")
                        else:
                            st.session_state.pop(confirm_key, None)
                            st.session_state["match_flash_success"] = (
                                f"Marked Report **#{cand_id}** as recovered."
                            )
                            st.rerun()
                with rc2:
                    if st.button(
                        "Cancel",
                        key=f"{key_prefix}_cancel_single_{cand_id}",
                        width="stretch",
                    ):
                        st.session_state.pop(confirm_key, None)
                        st.rerun()
            else:
                ac1, ac2, _ = st.columns([1.5, 1.35, 1.65])
                with ac1:
                    if source_report.get("status") == "active":
                        if st.button(
                            "Mark Pair as Recovered",
                            key=f"{key_prefix}_recover_pair_{cand_id}",
                            type="primary",
                            width="stretch",
                        ):
                            st.session_state[confirm_key] = "pair"
                            st.rerun()
                with ac2:
                    if st.button(
                        f"Mark #{cand_id} Recovered",
                        key=f"{key_prefix}_recover_cand_{cand_id}",
                        width="stretch",
                    ):
                        st.session_state[confirm_key] = "single"
                        st.rerun()


# ---------------------------------------------------------------------------
# View 1: Action-First Dashboard
# ---------------------------------------------------------------------------


def render_dashboard_page() -> None:
    """Render the redesigned action-first campus dashboard."""
    # SECTION A — HERO / PRIMARY ACTIONS
    with st.container(border=True):
        st.markdown(
            """
            <div class="rf-eyebrow">Campus Lost &amp; Found Intelligence</div>
            <h1 class="rf-hero-title">Lost something on campus?</h1>
            <p class="rf-hero-sub">
                Report a lost item or help someone recover theirs. ReFind AI uses local
                semantic NLP to surface meaningful connections across campus reports — even
                when descriptions use different words.
            </p>
            """,
            unsafe_allow_html=True,
        )
        hc1, hc2, hc3, _ = st.columns([1.3, 1.3, 1.15, 1.25], gap="small")
        with hc1:
            if st.button(
                "Report Lost Item",
                key="hero_report_lost",
                type="primary",
                width="stretch",
            ):
                navigate_to("Report Lost Item")
        with hc2:
            if st.button(
                "Report Found Item",
                key="hero_report_found",
                width="stretch",
            ):
                navigate_to("Report Found Item")
        with hc3:
            if st.button(
                "Browse Reports",
                key="hero_browse",
                width="stretch",
            ):
                navigate_to("Browse Reports")

    # SECTION B — COMPACT CAMPUS ACTIVITY SUMMARY RIBBON
    stats = get_dashboard_stats()
    render_metric_ribbon(stats)

    # SECTION C & D — AI MATCH OPPORTUNITIES + RECENT CAMPUS ACTIVITY
    col_matches, col_activity = st.columns([1.15, 1.0], gap="large")

    with col_matches:
        render_section_label(
            "AI Match Opportunities",
            "Active Lost and Found pairs ranked by semantic and contextual similarity.",
        )

        active_lost = get_all_reports(report_type="lost", status="active")
        active_found = get_all_reports(report_type="found", status="active")

        if not active_lost or not active_found:
            render_empty_state(
                "Waiting for Opposite-Type Reports",
                "Submit at least one active Lost report and one active Found report to scan for semantic matches.",
            )
        else:
            engine_state = get_engine_state()
            if engine_state.status == "not_initialized":
                with st.container(border=True):
                    st.markdown(
                        """
                        <div class="rf-item-title" style="margin-bottom:0.3rem;">
                            Scan active campus reports with AI
                        </div>
                        <p class="rf-item-desc" style="margin-bottom:0.75rem;">
                            To keep the Dashboard fast on startup, the local
                            <code>all-MiniLM-L6-v2</code> model loads on demand. Click below to
                            initialize the AI engine and discover top matching pairs.
                        </p>
                        """,
                        unsafe_allow_html=True,
                    )
                    lc1, lc2 = st.columns([1.45, 1.0], gap="small")
                    with lc1:
                        if st.button(
                            "Load AI & Scan Matches",
                            key="dash_load_ai_highlights",
                            type="primary",
                            width="stretch",
                        ):
                            if try_get_engine() is not None:
                                st.rerun()
                    with lc2:
                        if st.button(
                            "Open AI Matching",
                            key="dash_open_ai_matching",
                            width="stretch",
                        ):
                            navigate_to("AI Matching")
            elif engine_state.status == "unavailable":
                st.error(
                    f"AI Engine is currently unavailable: {engine_state.error_message or 'Model failed to load.'}"
                )
                if st.button(
                    "Retry AI Engine Load",
                    key="dash_retry_ai_highlights",
                    width="stretch",
                ):
                    if try_get_engine(force_retry=True) is not None:
                        st.rerun()
            elif engine_state.engine is not None:
                top_highlights = get_cached_dashboard_highlights(engine_state.engine)
                if not top_highlights:
                    render_empty_state(
                        "No Strong Active Pairs Yet",
                        "No active Lost ↔ Found pairs currently score above 45.0 / 100. Open AI Matching to inspect lower-threshold candidates.",
                    )
                else:
                    for lost_item, best_match in top_highlights[:3]:
                        cand = best_match.candidate
                        tier_label, tier_cls = match_tier_display(best_match.final_score)
                        bar_cls = "fill-strong" if best_match.final_score >= 75.0 else ""
                        short_why = _concise_explanation(best_match.explanation)

                        with st.container(border=True):
                            st.markdown(
                                f"""
                                <div class="rf-activity-row-top">
                                    <span class="rf-pill {tier_cls}">{best_match.final_score:.1f} / 100 &middot; {esc(tier_label)}</span>
                                    <span style="font-size:0.76rem;color:#64748B;">Semantic {round(best_match.semantic_score*100)}%</span>
                                </div>
                                <div class="rf-bar-track" style="height:5px; margin:0.45rem 0 0.5rem 0;"
                                     role="progressbar" aria-label="Match similarity score"
                                     aria-valuenow="{best_match.final_score:.0f}" aria-valuemin="0" aria-valuemax="100">
                                    <div class="rf-bar-fill {bar_cls}" style="width:{best_match.final_score:.1f}%;"></div>
                                </div>
                                <div class="rf-highlight-pair">
                                    <strong>Lost #{int(lost_item["id"])}:</strong> {esc(lost_item["title"])}<br/>
                                    <strong>Found #{int(cand["id"])}:</strong> {esc(cand["title"])}
                                </div>
                                <div class="rf-highlight-why">{esc(short_why)}</div>
                                """,
                                unsafe_allow_html=True,
                            )
                            if st.button(
                                "Review Match",
                                key=f"highlight_open_{lost_item['id']}",
                                width="stretch",
                            ):
                                navigate_to("AI Matching", report_id=int(lost_item["id"]))

    with col_activity:
        render_section_label(
            "Recent Campus Activity",
            "Latest reported lost and found items across campus.",
        )

        recent = get_all_reports(limit=5)
        if not recent:
            render_empty_state(
                "No Campus Reports Yet",
                "Use Report Lost Item or Report Found Item above to record the first item.",
            )
        else:
            for item in recent:
                item_id = int(item["id"])
                badges = build_report_badges_html(
                    item["report_type"], item["status"], show_active_badge=False
                )
                rel_date = format_relative_date(item.get("report_date"))

                with st.container(border=True):
                    c_info, c_act = st.columns([2.2, 0.95], vertical_alignment="center")
                    with c_info:
                        st.markdown(
                            f"""
                            <div class="rf-activity-row-top">
                                <span class="rf-item-title" style="font-size:0.93rem;">
                                    <span class="rf-id-tag">#{item_id}</span> {esc(item["title"])}
                                </span>
                            </div>
                            <div class="rf-pill-row" style="margin-top:0.3rem;">
                                {badges}
                            </div>
                            <div class="rf-activity-meta">
                                {esc(item["location"])} &middot; {esc(rel_date)}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    with c_act:
                        if item["status"] == "active":
                            if st.button(
                                "Match",
                                key=f"dash_match_{item_id}",
                                width="stretch",
                            ):
                                navigate_to("AI Matching", report_id=item_id)
                        else:
                            if st.button(
                                "View",
                                key=f"dash_view_{item_id}",
                                width="stretch",
                            ):
                                navigate_to("Browse Reports")


# ---------------------------------------------------------------------------
# View 2: AI Matching — Split Comparison Workspace
# ---------------------------------------------------------------------------


def _reset_matching_weights() -> None:
    """Callback to restore the default 70/15/10/5 academic weights & threshold."""
    st.session_state["ai_w_sem"] = 70
    st.session_state["ai_w_cat"] = 15
    st.session_state["ai_w_loc"] = 10
    st.session_state["ai_w_date"] = 5
    st.session_state["ai_min_threshold"] = 35
    st.session_state["ai_show_below"] = False
    st.session_state["ai_include_rec_cand"] = False


def render_ai_matching_page() -> None:
    """Render the centrepiece split-comparison AI Matching workspace."""
    # Compact header with unobtrusive How AI Works & Scoring Settings popovers
    hdr_left, hdr_right = st.columns([2.4, 1.3], vertical_alignment="bottom")
    with hdr_left:
        render_page_header(
            "AI Matching",
            "Discover potential connections between lost and found reports using semantic AI.",
        )
    with hdr_right:
        p1, p2 = st.columns(2, gap="small")
        with p1:
            with st.popover("How AI Works", width="stretch"):
                render_how_ai_works_content()
        with p2:
            with st.popover("Scoring Settings", width="stretch"):
                st.caption(
                    "Default formula: 70% Semantic, 15% Category, 10% Location, 5% Date. "
                    "Weights normalize automatically to 100%."
                )
                raw_sem = st.slider("Semantic Weight", 0, 100, 70, step=5, key="ai_w_sem")
                raw_cat = st.slider("Category Weight", 0, 100, 15, step=5, key="ai_w_cat")
                raw_loc = st.slider("Location Weight", 0, 100, 10, step=5, key="ai_w_loc")
                raw_date = st.slider("Date Weight", 0, 100, 5, step=5, key="ai_w_date")
                min_threshold = st.slider(
                    "Minimum Display Score",
                    min_value=0,
                    max_value=85,
                    value=35,
                    step=5,
                    key="ai_min_threshold",
                )
                show_below_threshold = st.checkbox(
                    "Show candidates below threshold",
                    value=False,
                    key="ai_show_below",
                )
                include_recovered_candidates = st.checkbox(
                    "Include recovered items in candidates",
                    value=False,
                    key="ai_include_rec_cand",
                )
                st.button(
                    "Reset Default Weights",
                    key="ai_reset_weights",
                    on_click=_reset_matching_weights,
                    width="stretch",
                )

    flash_success = st.session_state.pop("match_flash_success", None)
    if flash_success:
        st.success(flash_success)

    engine = try_get_engine()
    if engine is None:
        return

    all_reports = get_all_reports()

    # Source selection bar
    sel_col, tog_col = st.columns([2.5, 1.0], vertical_alignment="bottom")
    with tog_col:
        include_recovered_source = st.checkbox(
            "Include recovered reports",
            value=False,
            key="ai_include_rec_source",
        )

    source_pool = (
        all_reports
        if include_recovered_source
        else [r for r in all_reports if r["status"] == "active"]
    )
    if not source_pool:
        render_empty_state(
            "No Reports Available for Matching",
            "Submit a Lost or Found report first, or enable 'Include recovered reports' above.",
        )
        return

    report_by_id = {int(r["id"]): r for r in source_pool}
    option_ids = [int(r["id"]) for r in source_pool]

    # V1.2 one-time pending navigation selection (never overwrites user dropdown choices!)
    pending_id = st.session_state.pop("pending_match_report_id", None)
    if pending_id is not None and int(pending_id) in report_by_id:
        st.session_state["ai_match_select_id"] = int(pending_id)
    elif st.session_state.get("ai_match_select_id") not in report_by_id:
        st.session_state["ai_match_select_id"] = option_ids[0]

    def _format_id_option(rep_id: int) -> str:
        rep = report_by_id[int(rep_id)]
        return (
            f"[{rep['report_type'].upper()} #{rep['id']}] {rep['title']} "
            f"— {rep['location']} ({format_relative_date(rep['report_date'])})"
        )

    with sel_col:
        selected_id = st.selectbox(
            "Selected report to match",
            options=option_ids,
            format_func=_format_id_option,
            key="ai_match_select_id",
        )
    selected_report = report_by_id[int(selected_id)]

    # Compact Academic Demo Presets (preserved for viva walkthrough)
    seed_lookup = {
        str(r["seed_key"]): int(r["id"])
        for r in all_reports
        if r.get("seed_key") and r.get("status") == "active"
    }
    available_presets = [
        (label, seed_lookup[key])
        for label, key in DEMO_PRESETS
        if key in seed_lookup
    ]
    if available_presets:
        st.caption("Quick sample cases:")
        preset_cols = st.columns(len(available_presets), gap="small")
        for col, (label, rep_id) in zip(preset_cols, available_presets):
            with col:
                if st.button(label, key=f"demo_preset_{rep_id}", width="stretch"):
                    st.session_state["pending_match_report_id"] = rep_id
                    st.rerun()

    # Build normalized MatchConfig from settings popover
    total_raw = raw_sem + raw_cat + raw_loc + raw_date
    if total_raw == 0:
        w_sem, w_cat, w_loc, w_date = 0.70, 0.15, 0.10, 0.05
        st.warning("All weights were 0; reverted to default 70/15/10/5 weights.")
    else:
        w_sem = raw_sem / total_raw
        w_cat = raw_cat / total_raw
        w_loc = raw_loc / total_raw
        w_date = raw_date / total_raw

    config = MatchConfig(
        weight_semantic=w_sem,
        weight_category=w_cat,
        weight_location=w_loc,
        weight_date=w_date,
        min_score_threshold=float(min_threshold),
        max_results=10,
    )

    opposite_label = "Found" if selected_report["report_type"] == "lost" else "Lost"
    matches = find_matches(
        source_report=selected_report,
        candidate_reports=all_reports,
        engine=engine,
        config=config,
        include_recovered=include_recovered_candidates,
        apply_threshold=not show_below_threshold,
    )

    st.divider()

    # SPLIT COMPARISON WORKSPACE (Left: Source Item | Right: Ranked AI Candidates)
    col_source, col_candidates = st.columns([0.92, 1.48], gap="large")

    with col_source:
        render_section_label(
            "Selected Source Report",
            f"Comparing against active {opposite_label.lower()} campus reports.",
        )
        with st.container(border=True):
            render_report_body(selected_report, compact=False, show_contact=True)
            source_img = resolve_image_path(selected_report.get("image_path"))
            if source_img is not None:
                with st.expander("View uploaded photo"):
                    st.image(str(source_img), width=280)

        st.markdown(
            """
            <div class="rf-prob-footnote" style="margin-top:0.65rem;">
                <strong>Human verification required:</strong> Semantic similarity can surface
                similar-looking items of different brands or owners. Compare location, date,
                and distinguishing marks before confirming recovery.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_candidates:
        render_section_label(
            f"Ranked {opposite_label} Candidates ({len(matches)})",
            f"Evaluated with {engine.model_name.split('/')[-1]} ({engine.embedding_dim}-D cosine similarity + context).",
        )

        if not matches:
            render_empty_state(
                f"No {opposite_label} Candidates Above {min_threshold:.0f} / 100",
                "Lower the minimum threshold or enable 'Show candidates below threshold' inside Scoring Settings to inspect lower-ranked comparisons.",
            )
        else:
            for rank, result in enumerate(matches, start=1):
                render_integrated_match_card(
                    rank=rank,
                    result=result,
                    source_report=selected_report,
                    config=config,
                    key_prefix=f"matcher_{selected_report['id']}_{rank}",
                    expand_breakdown_default=(rank == 1),
                )


# ---------------------------------------------------------------------------
# Views 3 & 4: Guided Report Lost Item & Report Found Item
# ---------------------------------------------------------------------------


def _init_form_state(report_type: str, force_reset: bool = False) -> None:
    """Initialize or cleanly reset controlled session-state keys for a report form."""
    defaults: dict[str, Any] = {
        f"{report_type}_title": "",
        f"{report_type}_category": CATEGORIES[0],
        f"{report_type}_description": "",
        f"{report_type}_location_preset": CUSTOM_LOCATION_OPTION,
        f"{report_type}_location": "",
        f"{report_type}_date": date.today(),
        f"{report_type}_extra": "",
        f"{report_type}_contact": "",
    }
    for key, default_val in defaults.items():
        if force_reset or key not in st.session_state:
            st.session_state[key] = default_val

    nonce_key = f"{report_type}_upload_nonce"
    if nonce_key not in st.session_state:
        st.session_state[nonce_key] = 0
    elif force_reset:
        st.session_state[nonce_key] = int(st.session_state[nonce_key]) + 1


def _render_report_form_widget(report_type: str, is_lost: bool) -> None:
    """Render the guided 3-section reporting form and process verified submissions."""
    loc_label = "Last Seen Location *" if is_lost else "Location Found *"
    date_label = "Date Lost *" if is_lost else "Date Found *"
    extra_label = (
        "Distinguishing Marks or Identifying Details (Optional)"
        if is_lost
        else "Current Custody & Observations (Optional)"
    )
    extra_placeholder = (
        "E.g., small dent near base, blue sticker, initials inside flap..."
        if is_lost
        else "E.g., handed in to Central Library security desk; minor scratches on back..."
    )
    desc_placeholder = (
        "E.g., Dark brown leather bifold wallet lost near the library entrance. Slim design with stitched edges."
        if is_lost
        else "E.g., Brown leather purse-style wallet found on a bench outside the central library."
    )
    upload_nonce = int(st.session_state[f"{report_type}_upload_nonce"])

    with st.form(
        key=f"form_report_{report_type}",
        clear_on_submit=False,
        enter_to_submit=False,
    ):
        # SECTION 1 — ITEM DETAILS
        st.markdown(
            '<div class="rf-form-section-title">1. Item Details</div>',
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns([1.45, 1.0])
        with c1:
            title = st.text_input(
                "Item Title *",
                placeholder="E.g., Dark Brown Leather Wallet",
                max_chars=120,
                key=f"{report_type}_title",
            )
        with c2:
            category = st.selectbox(
                "Category *",
                options=CATEGORIES,
                key=f"{report_type}_category",
            )

        description = st.text_area(
            "Detailed Description *",
            placeholder=desc_placeholder,
            height=115,
            key=f"{report_type}_description",
        )
        st.markdown(
            """
            <div class="rf-field-hint">
                Include details such as colour, brand, material, size or distinguishing marks.
                The more descriptive your report, the better AI can identify potential matches.
            </div>
            """,
            unsafe_allow_html=True,
        )

        # SECTION 2 — WHERE AND WHEN
        st.markdown(
            '<div class="rf-form-section-title">2. Where and When</div>',
            unsafe_allow_html=True,
        )
        c3, c4 = st.columns([1.35, 1.0])
        with c3:
            location_preset = st.selectbox(
                "Common Campus Area (Optional quick-select)",
                options=[CUSTOM_LOCATION_OPTION, *CAMPUS_LOCATION_SUGGESTIONS],
                key=f"{report_type}_location_preset",
            )
        with c4:
            report_date = st.date_input(
                date_label,
                max_value=date.today() + timedelta(days=1),
                key=f"{report_type}_date",
            )

        location = st.text_input(
            loc_label,
            placeholder="Type exact spot or room (e.g., Central Library Entrance, Room 204)",
            help="Leave blank only if you selected a Common Campus Area above; typing here overrides the quick-select.",
            key=f"{report_type}_location",
        )

        # SECTION 3 — ADDITIONAL DETAILS
        st.markdown(
            '<div class="rf-form-section-title">3. Additional Details</div>',
            unsafe_allow_html=True,
        )
        if not is_lost:
            st.markdown(
                """
                <div class="rf-field-hint">
                    <strong>Finder guidance:</strong> Mention where the item is currently kept
                    (e.g., turned in to a campus security desk). Avoid posting sensitive ID numbers
                    or personal phone numbers publicly.
                </div>
                """,
                unsafe_allow_html=True,
            )

        additional_details = st.text_input(
            extra_label,
            placeholder=extra_placeholder,
            max_chars=240,
            key=f"{report_type}_extra",
        )

        c5, c6 = st.columns([1.15, 1.25])
        with c5:
            contact_label = st.text_input(
                "Campus Desk / Reference (Optional)",
                placeholder="E.g., Library Desk Ref #108 (Demo)",
                help="Use a campus desk reference or synthetic contact label rather than private personal data.",
                key=f"{report_type}_contact",
            )
        with c6:
            uploaded_photo = st.file_uploader(
                "Optional Photograph (PNG, JPG, WEBP)",
                type=["png", "jpg", "jpeg", "webp"],
                key=f"{report_type}_photo_{upload_nonce}",
            )

        submitted = st.form_submit_button(
            f"Submit {'Lost' if is_lost else 'Found'} Report & Find Matches",
            type="primary",
            width="stretch",
        )

    if submitted:
        errors: list[str] = []
        clean_title = (title or "").strip()
        clean_desc = (description or "").strip()
        typed_loc = (location or "").strip()
        preset_loc = (
            "" if location_preset == CUSTOM_LOCATION_OPTION else str(location_preset).strip()
        )
        effective_loc = typed_loc or preset_loc

        if not clean_title:
            errors.append("Please enter an item title.")
        if not clean_desc or len(clean_desc) < 5:
            errors.append("Please provide a descriptive summary (at least 5 characters).")
        if not effective_loc:
            errors.append("Please enter a campus location or choose a common campus area.")

        valid_date: Optional[date] = None
        if isinstance(report_date, date):
            valid_date = report_date
        elif (
            isinstance(report_date, (tuple, list))
            and report_date
            and isinstance(report_date[0], date)
        ):
            valid_date = report_date[0]

        if valid_date is None:
            errors.append("Please select a valid date.")
        elif valid_date > date.today() + timedelta(days=1):
            errors.append("The selected date cannot be in the future.")

        if not errors and valid_date is not None:
            existing_dup = find_duplicate_report(
                title=clean_title,
                description=clean_desc,
                report_type=report_type,
                location=effective_loc,
                report_date=valid_date.isoformat(),
            )
            if existing_dup is not None:
                st.session_state[f"last_submitted_{report_type}_id"] = int(existing_dup["id"])
                st.warning(
                    f"An identical active {report_type} report (**#{existing_dup['id']} — {existing_dup['title']}**) "
                    "already exists. Displaying its AI matches without creating a duplicate."
                )
            else:
                image_rel_path: Optional[str] = None
                if uploaded_photo is not None:
                    try:
                        image_rel_path = save_uploaded_image(uploaded_photo)
                    except (ValueError, OSError) as exc:
                        errors.append(f"Image upload error: {exc}")

                if not errors:
                    try:
                        new_id = insert_report(
                            title=clean_title,
                            description=clean_desc,
                            category=category,
                            report_type=report_type,
                            location=effective_loc,
                            report_date=valid_date.isoformat(),
                            additional_details=(additional_details or "").strip(),
                            contact_label=(contact_label or "").strip()
                            or "Campus Lost & Found Desk (Demo)",
                            image_path=image_rel_path,
                        )
                    except (sqlite3.Error, ValueError) as exc:
                        errors.append(f"Database error while saving report: {exc}")
                    else:
                        st.session_state[f"last_submitted_{report_type}_id"] = int(new_id)
                        st.session_state[f"flash_success_{report_type}"] = (
                            f"Report **#{new_id} ({clean_title})** saved to SQLite. "
                            "Your submission is live across the Dashboard, Browse Reports, and AI Matching."
                        )
                        st.session_state[f"reset_form_{report_type}"] = True
                        st.rerun()

        if errors:
            for err in errors:
                st.error(err)


def _render_post_submit_results_panel(saved_report: dict[str, Any], report_type: str) -> None:
    """Render saved report confirmation summary and instant AI matches prominently."""
    rep_id = int(saved_report["id"])
    render_section_label(
        f"Submitted Report #{rep_id} & Instant AI Matches",
        "Semantic matches against active opposite-type campus reports.",
    )
    with st.container(border=True):
        render_report_body(saved_report, compact=False, show_contact=True)
        b1, b2, b3 = st.columns([1.25, 1.1, 1.05], gap="small")
        with b1:
            if st.button(
                "Open in AI Matching",
                key=f"jump_after_submit_{rep_id}",
                type="primary",
                width="stretch",
            ):
                navigate_to("AI Matching", report_id=rep_id)
        with b2:
            if st.button(
                "View in Browse",
                key=f"browse_after_submit_{rep_id}",
                width="stretch",
            ):
                navigate_to("Browse Reports")
        with b3:
            if st.button(
                "Report Another",
                key=f"dismiss_after_submit_{report_type}",
                width="stretch",
            ):
                st.session_state.pop(f"last_submitted_{report_type}_id", None)
                st.session_state[f"reset_form_{report_type}"] = True
                st.rerun()

    engine = try_get_engine()
    if engine is not None:
        cfg = MatchConfig(max_results=5)
        candidates = get_all_reports()
        matches = find_matches(saved_report, candidates, engine, config=cfg)
        if not matches:
            render_empty_state(
                "No Immediate Matches Above 35 / 100",
                "Your report is saved in SQLite and will automatically appear as a candidate when new opposite-type reports are submitted.",
            )
        else:
            for idx, res in enumerate(matches, start=1):
                render_integrated_match_card(
                    rank=idx,
                    result=res,
                    source_report=saved_report,
                    config=cfg,
                    key_prefix=f"instant_{rep_id}_{idx}",
                    expand_breakdown_default=(idx == 1),
                )


def render_report_form_page(report_type: str) -> None:
    """Render guided Lost or Found reporting with focused width & above-the-fold results."""
    is_lost = report_type == "lost"
    render_page_header(
        "Report Lost Item" if is_lost else "Report Found Item",
        (
            "Share clear details about your missing belonging so ReFind AI can check active found reports."
            if is_lost
            else "Log an item found on campus so ReFind AI can connect it with active lost reports."
        ),
    )

    should_reset = bool(st.session_state.pop(f"reset_form_{report_type}", False))
    _init_form_state(report_type, force_reset=should_reset)

    flash_msg = st.session_state.pop(f"flash_success_{report_type}", None)
    if flash_msg:
        st.success(flash_msg)

    last_id = st.session_state.get(f"last_submitted_{report_type}_id")
    saved_report = get_report_by_id(int(last_id)) if last_id is not None else None
    if last_id is not None and saved_report is None:
        st.session_state.pop(f"last_submitted_{report_type}_id", None)

    if saved_report is None:
        # Focused ~780px centered form column for a calm consumer reporting experience.
        _, center_col, _ = st.columns([0.11, 0.78, 0.11])
        with center_col:
            _render_report_form_widget(report_type, is_lost)
    else:
        # Post-submission split view: instant AI matches appear immediately above the fold,
        # while the fresh form remains ready on the left (and keeps submit as button[0]).
        col_form, col_results = st.columns([0.92, 1.08], gap="large")
        with col_form:
            render_section_label(
                "Submit Another Report",
                "The form has been cleared for a new entry.",
            )
            _render_report_form_widget(report_type, is_lost)
        with col_results:
            _render_post_submit_results_panel(saved_report, report_type)


# ---------------------------------------------------------------------------
# View 5: Browse Campus Reports (Directory, Filters, Sorting, Pagination & Edit)
# ---------------------------------------------------------------------------


@st.dialog("Edit Campus Report")
def open_edit_report_dialog(report: dict[str, Any]) -> None:
    """Focused modal dialog for safely editing an existing SQLite report."""
    rep_id = int(report["id"])
    st.caption(
        f"Editing {str(report['report_type']).capitalize()} Report **#{rep_id}**"
    )
    parsed_date = parse_iso_date(report.get("report_date")) or date.today()
    cat_index = (
        CATEGORIES.index(report["category"])
        if report.get("category") in CATEGORIES
        else 0
    )

    with st.form(key=f"edit_report_form_{rep_id}", enter_to_submit=False):
        e_title = st.text_input("Item Title *", value=str(report.get("title") or ""), max_chars=120)
        e_cat = st.selectbox("Category *", options=CATEGORIES, index=cat_index)
        e_desc = st.text_area(
            "Detailed Description *",
            value=str(report.get("description") or ""),
            height=110,
        )
        ec1, ec2 = st.columns([1.3, 1.0])
        with ec1:
            e_loc = st.text_input(
                "Campus Location *", value=str(report.get("location") or ""), max_chars=120
            )
        with ec2:
            e_date = st.date_input(
                "Date *",
                value=parsed_date,
                max_value=date.today() + timedelta(days=1),
            )
        e_extra = st.text_input(
            "Additional Notes (Optional)",
            value=str(report.get("additional_details") or ""),
            max_chars=240,
        )
        e_contact = st.text_input(
            "Campus Desk / Reference",
            value=str(report.get("contact_label") or ""),
            max_chars=120,
        )
        save_clicked = st.form_submit_button("Save Changes", type="primary", width="stretch")

    if save_clicked:
        errs: list[str] = []
        if not (e_title or "").strip():
            errs.append("Item title cannot be empty.")
        if not (e_desc or "").strip() or len(e_desc.strip()) < 5:
            errs.append("Description must be at least 5 characters.")
        if not (e_loc or "").strip():
            errs.append("Location cannot be empty.")
        if not isinstance(e_date, date):
            errs.append("Please choose a valid date.")

        if errs:
            for err in errs:
                st.error(err)
            return

        try:
            updated = update_report(
                rep_id,
                title=e_title,
                description=e_desc,
                category=e_cat,
                location=e_loc,
                report_date=e_date.isoformat(),
                additional_details=e_extra,
                contact_label=e_contact,
            )
        except (sqlite3.Error, ValueError) as exc:
            st.error(f"Could not save changes: {exc}")
        else:
            if updated:
                st.session_state["browse_flash_success"] = (
                    f"Updated Report **#{rep_id} ({e_title.strip()})**."
                )
            else:
                st.session_state["browse_flash_warning"] = (
                    f"Report #{rep_id} could not be found."
                )
            st.rerun()


def _clear_browse_filters() -> None:
    """Callback to reset all Browse search/filter/sort/page controls cleanly."""
    st.session_state["browse_search"] = ""
    st.session_state["browse_type"] = "All"
    st.session_state["browse_category"] = "All"
    st.session_state["browse_status"] = "All"
    st.session_state["browse_sort"] = "Newest Submitted"
    st.session_state["browse_page"] = 1


def _render_browse_report_card(
    report: dict[str, Any],
    *,
    compact_list: bool,
    pending_delete_id: Optional[int],
) -> None:
    """Render one integrated Browse report card with inside-card actions & safe delete."""
    rep_id = int(report["id"])
    with st.container(border=True):
        render_report_body(
            report,
            compact=compact_list,
            show_contact=not compact_list,
            show_active_badge=True,
        )

        img_path = resolve_image_path(report.get("image_path"))
        if img_path is not None:
            with st.expander(f"View uploaded photograph — #{rep_id}"):
                st.image(str(img_path), width=280)

        a1, a2, a3, a4 = st.columns([1.2, 1.25, 0.78, 0.82], gap="small")
        with a1:
            if st.button(
                "Find Matches",
                key=f"browse_match_{rep_id}",
                width="stretch",
            ):
                navigate_to("AI Matching", report_id=rep_id)
        with a2:
            if report["status"] == "active":
                if st.button(
                    "Mark Recovered",
                    key=f"browse_recover_{rep_id}",
                    type="primary",
                    width="stretch",
                ):
                    try:
                        update_report_status(rep_id, "recovered")
                        st.session_state["browse_flash_success"] = (
                            f"Report **#{rep_id}** marked as recovered."
                        )
                        st.rerun()
                    except (sqlite3.Error, ValueError) as exc:
                        st.error(f"Could not update report status: {exc}")
            else:
                if st.button(
                    "Reopen",
                    key=f"browse_reopen_{rep_id}",
                    width="stretch",
                ):
                    try:
                        update_report_status(rep_id, "active")
                        st.session_state["browse_flash_info"] = (
                            f"Report **#{rep_id}** reopened as active."
                        )
                        st.rerun()
                    except (sqlite3.Error, ValueError) as exc:
                        st.error(f"Could not reopen report: {exc}")
        with a3:
            if st.button(
                "Edit",
                key=f"browse_edit_{rep_id}",
                width="stretch",
            ):
                open_edit_report_dialog(report)
        with a4:
            if st.button(
                "Delete",
                key=f"browse_delete_{rep_id}",
                width="stretch",
            ):
                st.session_state["confirm_delete_report_id"] = rep_id
                st.rerun()

        # V1.2 two-step deletion confirmation integrated directly inside the card
        if pending_delete_id == rep_id:
            photo_note = (
                " Its uploaded image file will also be removed."
                if report.get("image_path")
                else ""
            )
            st.warning(
                f"**Confirm permanent deletion:** Report **#{rep_id} — {report['title']}**?"
                f"{photo_note} This action cannot be undone."
            )
            cd1, cd2 = st.columns([1.4, 1.0], gap="small")
            with cd1:
                if st.button(
                    "Confirm Permanent Delete",
                    key=f"browse_confirm_delete_{rep_id}",
                    type="primary",
                    width="stretch",
                ):
                    try:
                        db_deleted, img_deleted, img_warning = delete_report_with_cleanup(rep_id)
                    except sqlite3.Error as exc:
                        st.error(f"Could not delete report #{rep_id}: {exc}")
                    else:
                        st.session_state.pop("confirm_delete_report_id", None)
                        if st.session_state.get("ai_match_select_id") == rep_id:
                            st.session_state.pop("ai_match_select_id", None)
                        if st.session_state.get("pending_match_report_id") == rep_id:
                            st.session_state.pop("pending_match_report_id", None)
                        for rtype in ("lost", "found"):
                            if st.session_state.get(f"last_submitted_{rtype}_id") == rep_id:
                                st.session_state.pop(f"last_submitted_{rtype}_id", None)

                        if db_deleted:
                            suffix = (
                                " Associated uploaded photo was also removed."
                                if img_deleted
                                else ""
                            )
                            st.session_state["browse_flash_success"] = (
                                f"Deleted Report **#{rep_id} ({report['title']})**.{suffix}"
                            )
                        else:
                            st.session_state["browse_flash_warning"] = (
                                f"Report #{rep_id} was already removed."
                            )
                        if img_warning:
                            st.session_state["browse_flash_warning"] = img_warning
                        st.rerun()
            with cd2:
                if st.button(
                    "Cancel",
                    key=f"browse_cancel_delete_{rep_id}",
                    width="stretch",
                ):
                    st.session_state.pop("confirm_delete_report_id", None)
                    st.session_state["browse_flash_info"] = (
                        f"Cancelled deletion of Report #{rep_id} ({report['title']})."
                    )
                    st.rerun()


def render_browse_reports_page() -> None:
    """Render the redesigned campus report discovery directory."""
    render_page_header(
        "Browse Campus Reports",
        "Explore reported lost and found items across campus.",
    )

    stats = get_dashboard_stats()
    render_metric_ribbon(stats, compact=True)

    flash_success = st.session_state.pop("browse_flash_success", None)
    if flash_success:
        st.success(flash_success)
    flash_warning = st.session_state.pop("browse_flash_warning", None)
    if flash_warning:
        st.warning(flash_warning)
    flash_info = st.session_state.pop("browse_flash_info", None)
    if flash_info:
        st.info(flash_info)

    # Filter & discovery toolbar inside an integrated container
    with st.container(border=True):
        f1, f2, f3, f4 = st.columns([1.85, 0.95, 1.25, 0.95], gap="small")
        with f1:
            search_query = st.text_input(
                "Keyword search",
                placeholder="Search title, description, or location...",
                key="browse_search",
            )
        with f2:
            type_choice = st.selectbox("Type", ["All", "Lost", "Found"], key="browse_type")
        with f3:
            category_choice = st.selectbox(
                "Category", ["All", *CATEGORIES], key="browse_category"
            )
        with f4:
            status_choice = st.selectbox(
                "Status", ["All", "Active", "Recovered"], key="browse_status"
            )

        s1, s2, s3 = st.columns([1.4, 1.15, 0.95], vertical_alignment="bottom", gap="small")
        with s1:
            sort_label = st.selectbox(
                "Sort by",
                options=list(SORT_OPTIONS.keys()),
                key="browse_sort",
            )
        with s2:
            view_mode = st.selectbox(
                "Layout",
                options=["Cards (2-column)", "Compact List"],
                key="browse_view",
            )
        with s3:
            st.button(
                "Clear Filters",
                key="browse_clear_filters",
                on_click=_clear_browse_filters,
                width="stretch",
            )

        st.caption(
            "Browse search uses exact SQL keyword matching. Use **AI Matching** for semantic NLP search across paraphrased descriptions."
        )

    reports = get_all_reports(
        report_type=None if type_choice == "All" else type_choice.lower(),
        category=None if category_choice == "All" else category_choice,
        status=None if status_choice == "All" else status_choice.lower(),
        search_query=search_query,
        sort_by=SORT_OPTIONS[sort_label],
    )

    total_matches = len(reports)
    total_pages = max(1, math.ceil(total_matches / BROWSE_PAGE_SIZE))
    current_page = int(st.session_state.get("browse_page", 1))
    if current_page > total_pages:
        current_page = total_pages
        st.session_state["browse_page"] = current_page
    if current_page < 1:
        current_page = 1
        st.session_state["browse_page"] = 1

    start_idx = (current_page - 1) * BROWSE_PAGE_SIZE
    end_idx = min(start_idx + BROWSE_PAGE_SIZE, total_matches)
    page_reports = reports[start_idx:end_idx]

    if total_matches == 0:
        render_empty_state(
            "No Reports Match These Filters",
            "Try clearing filters or broadening your keyword search.",
        )
    else:
        st.caption(
            f"Showing **{start_idx + 1}–{end_idx}** of **{total_matches}** report(s) · Page {current_page} of {total_pages}"
        )

        pending_delete_id = st.session_state.get("confirm_delete_report_id")
        is_compact = view_mode == "Compact List"

        if is_compact:
            for report in page_reports:
                _render_browse_report_card(
                    report,
                    compact_list=True,
                    pending_delete_id=pending_delete_id,
                )
        else:
            # Clean 2-column responsive directory grid
            for idx in range(0, len(page_reports), 2):
                row_items = page_reports[idx : idx + 2]
                cols = st.columns(2, gap="medium")
                for col, rep in zip(cols, row_items):
                    with col:
                        _render_browse_report_card(
                            rep,
                            compact_list=False,
                            pending_delete_id=pending_delete_id,
                        )

        if total_pages > 1:
            p_prev, p_info, p_next = st.columns([1.0, 2.0, 1.0], vertical_alignment="center")
            with p_prev:
                if st.button(
                    "Previous Page",
                    key="browse_prev_page",
                    disabled=(current_page <= 1),
                    width="stretch",
                ):
                    st.session_state["browse_page"] = current_page - 1
                    st.rerun()
            with p_info:
                st.markdown(
                    f'<div style="text-align:center;font-size:0.84rem;color:#64748B;">Page {current_page} of {total_pages}</div>',
                    unsafe_allow_html=True,
                )
            with p_next:
                if st.button(
                    "Next Page",
                    key="browse_next_page",
                    disabled=(current_page >= total_pages),
                    width="stretch",
                ):
                    st.session_state["browse_page"] = current_page + 1
                    st.rerun()

    st.divider()
    with st.expander("Demo Dataset Administration"):
        st.caption(
            "Manage synthetic demonstration records without affecting your own submitted reports."
        )
        seed_count = count_seed_reports()
        dc1, dc2 = st.columns(2, gap="small")
        with dc1:
            if st.button("Restore Missing Demo Records", key="demo_restore_btn", width="stretch"):
                added = seed_database()
                st.session_state["browse_flash_success"] = (
                    f"Restored {added} synthetic demonstration record(s)."
                )
                st.rerun()
        with dc2:
            if st.button(
                "Remove Synthetic Demo Records",
                key="demo_remove_request",
                disabled=(seed_count == 0),
                width="stretch",
            ):
                st.session_state["confirm_delete_seed_reports"] = True
                st.rerun()

        if st.session_state.get("confirm_delete_seed_reports"):
            st.warning(
                f"**Confirm bulk removal:** Permanently remove **{seed_count}** synthetic "
                "demonstration record(s)? Your own user-submitted reports will not be affected."
            )
            sd1, sd2, _ = st.columns([1.6, 1.0, 2.4])
            with sd1:
                if st.button(
                    "Confirm Remove Demo Records",
                    key="demo_remove_confirm",
                    type="primary",
                    width="stretch",
                ):
                    try:
                        removed, removed_imgs, img_warnings = delete_seed_reports_with_cleanup()
                    except sqlite3.Error as exc:
                        st.error(f"Could not remove synthetic demo records: {exc}")
                    else:
                        st.session_state.pop("confirm_delete_seed_reports", None)
                        img_note = (
                            f" Cleaned up {removed_imgs} linked image file(s)."
                            if removed_imgs
                            else ""
                        )
                        st.session_state["browse_flash_success"] = (
                            f"Removed {removed} synthetic demonstration record(s).{img_note}"
                        )
                        if img_warnings:
                            st.session_state["browse_flash_warning"] = " ".join(img_warnings)
                        st.rerun()
            with sd2:
                if st.button("Cancel", key="demo_remove_cancel", width="stretch"):
                    st.session_state.pop("confirm_delete_seed_reports", None)
                    st.session_state["browse_flash_info"] = (
                        "Cancelled synthetic demo record removal."
                    )
                    st.rerun()


# ---------------------------------------------------------------------------
# Global Application Shell & Honest Dynamic Sidebar Status
# ---------------------------------------------------------------------------


def _render_sidebar_status(placeholder: Any) -> None:
    """Render honest, non-blocking System Status reflecting real DB & AI engine state."""
    stats = get_dashboard_stats()
    engine_state = get_engine_state()

    if engine_state.status == "ready" and engine_state.engine is not None:
        dot_cls = "dot-ready"
        ai_label = f"Ready ({engine_state.engine.model_name.split('/')[-1]})"
    elif engine_state.status == "unavailable":
        dot_cls = "dot-error"
        ai_label = "Unavailable"
    else:
        dot_cls = "dot-idle"
        ai_label = "Not initialized"

    placeholder.markdown(
        f"""
        <div class="rf-sidebar-status">
            <div class="rf-status-heading">
                <span class="rf-status-dot {dot_cls}"></span>System Status
            </div>
            <div><strong>AI Engine:</strong> {esc(ai_label)}</div>
            <div><strong>Database:</strong> SQLite (Persistent)</div>
            <div><strong>Active:</strong> {stats["active_reports"]} ({stats["active_lost"]} lost &middot; {stats["active_found"]} found)</div>
            <div><strong>Recovered:</strong> {stats["total_recovered"]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="ReFind AI — Smart Campus Lost & Found",
        page_icon="🔎",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_db()
    ensure_seed_data()
    load_stylesheet()

    if "nav_target" in st.session_state:
        st.session_state["nav_radio"] = st.session_state.pop("nav_target")
    if "nav_radio" not in st.session_state:
        st.session_state["nav_radio"] = "Dashboard"

    with st.sidebar:
        st.markdown(
            """
            <div class="rf-brand">
                <div class="rf-brand-row">
                    <span class="rf-brand-mark">RF</span>
                    <span class="rf-brand-name">ReFind AI</span>
                </div>
                <p class="rf-brand-tag">Lost something? Let AI help you find it.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        selected_page = st.radio(
            "Workspace",
            options=PAGES,
            key="nav_radio",
        )

        st.divider()
        sidebar_status_placeholder = st.empty()

    if selected_page == "Dashboard":
        render_dashboard_page()
    elif selected_page == "AI Matching":
        render_ai_matching_page()
    elif selected_page == "Report Lost Item":
        render_report_form_page("lost")
    elif selected_page == "Report Found Item":
        render_report_form_page("found")
    elif selected_page == "Browse Reports":
        render_browse_reports_page()

    _render_sidebar_status(sidebar_status_placeholder)


if __name__ == "__main__":
    main()
