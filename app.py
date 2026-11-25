"""ReFind AI — Intelligent AI-Powered Lost & Found Management System (v1.1).

Streamlit web application providing:
- Responsive Campus Dashboard with live KPIs, recent reports, and top AI match highlights
- Academic-grade AI Matching Dashboard powered by sentence-transformers/all-MiniLM-L6-v2
- Reliable Report Lost Item & Report Found Item forms with duplicate guard & instant AI matching
- Browse & Manage Reports with search, multi-filter, and persistent recovery management
"""

from __future__ import annotations

import html
import sqlite3
from datetime import date, timedelta
from typing import Any, Optional

import streamlit as st

from ai_matcher import (
    DEFAULT_MODEL_NAME,
    EmbeddingEngine,
    MatchConfig,
    MatchResult,
    ModelLoadError,
    find_matches,
)
from database import (
    delete_report,
    delete_seed_reports,
    find_duplicate_report,
    get_all_reports,
    get_dashboard_stats,
    get_report_by_id,
    init_db,
    insert_report,
    update_report_status,
)
from seed_data import ensure_seed_data, seed_database
from utils import (
    BASE_DIR,
    CAMPUS_LOCATION_SUGGESTIONS,
    CATEGORIES,
    format_display_date,
    resolve_image_path,
    save_uploaded_image,
    score_tier,
)

PAGES = [
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


def load_stylesheet() -> None:
    """Inject the custom SaaS dashboard CSS from assets/styles.css."""
    css_path = BASE_DIR / "assets" / "styles.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading ReFind AI semantic model (all-MiniLM-L6-v2)...")
def get_embedding_engine(model_name: str = DEFAULT_MODEL_NAME) -> EmbeddingEngine:
    """Load the Sentence Transformer model once per process via Streamlit cache."""
    return EmbeddingEngine(model_name=model_name)


def try_get_engine() -> Optional[EmbeddingEngine]:
    """Return cached EmbeddingEngine or render a clear error without keyword fallback."""
    try:
        return get_embedding_engine(DEFAULT_MODEL_NAME)
    except ModelLoadError as exc:
        st.error(
            "**AI Embedding Model Unavailable**\n\n"
            f"{exc}\n\n"
            "ReFind AI does not silently substitute keyword search for semantic embeddings. "
            "Please ensure `sentence-transformers` is installed and internet access is available "
            "for the initial download of `sentence-transformers/all-MiniLM-L6-v2`."
        )
        return None


def navigate_to(page: str, report_id: Optional[int] = None) -> None:
    """Queue a clean navigation transition for the next rerun."""
    st.session_state["nav_target"] = page
    if report_id is not None:
        st.session_state["selected_match_report_id"] = int(report_id)
    st.rerun()


def esc(value: Any) -> str:
    """Escape dynamic text safely before rendering inside HTML templates."""
    return html.escape("" if value is None else str(value))


def render_page_header(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="page-header">
            <h1 class="page-title">{esc(title)}</h1>
            <p class="page-subtitle">{esc(subtitle)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_report_card_html(report: dict[str, Any], compact: bool = False) -> str:
    """Build clean, responsive HTML markup for a lost or found report card."""
    rtype = str(report["report_type"]).lower()
    status = str(report["status"]).lower()
    type_cls = "badge-lost" if rtype == "lost" else "badge-found"
    status_cls = "badge-recovered" if status == "recovered" else "badge-active"
    card_cls = "card-recovered" if status == "recovered" else (
        "card-lost" if rtype == "lost" else "card-found"
    )

    desc = str(report.get("description") or "")
    if compact and len(desc) > 155:
        desc = desc[:152].rstrip() + "..."

    extra_html = ""
    if not compact and report.get("additional_details"):
        extra_html = (
            f'<div class="rf-meta-row" style="margin-top:0.45rem;">'
            f'<span class="rf-meta-item"><strong>Additional Notes:</strong> {esc(report["additional_details"])}</span>'
            f"</div>"
        )

    contact_html = ""
    if not compact and report.get("contact_label"):
        contact_html = (
            f'<span class="rf-meta-item"><strong>Reference:</strong> {esc(report["contact_label"])}</span>'
        )

    return f"""
    <div class="rf-card {card_cls}">
        <div class="rf-card-header">
            <div style="min-width:0; flex:1;">
                <div class="rf-card-title">#{int(report["id"])} &middot; {esc(report["title"])}</div>
            </div>
            <div style="display:flex; gap:0.35rem; flex-wrap:wrap; justify-content:flex-end;">
                <span class="rf-badge {type_cls}"><span class="rf-badge-dot"></span>{esc(rtype.upper())}</span>
                <span class="rf-badge {status_cls}"><span class="rf-badge-dot"></span>{esc(status.upper())}</span>
            </div>
        </div>
        <p class="rf-card-desc">{esc(desc)}</p>
        <div class="rf-meta-row">
            <span class="rf-meta-item"><strong>Category:</strong> {esc(report["category"])}</span>
            <span class="rf-meta-item"><strong>Location:</strong> {esc(report["location"])}</span>
            <span class="rf-meta-item"><strong>Date:</strong> {esc(format_display_date(report["report_date"]))}</span>
            {contact_html}
        </div>
        {extra_html}
    </div>
    """


def render_match_result_card(
    rank: int,
    result: MatchResult,
    source_report: dict[str, Any],
    config: MatchConfig,
    key_prefix: str,
    allow_recovery_actions: bool = True,
) -> None:
    """Render an Explainable AI match card with weighted contribution breakdown."""
    cand = result.candidate
    tier_label, tier_cls = score_tier(result.final_score)
    is_strong = result.final_score >= 75.0
    card_accent = "tier-strong" if is_strong else ""
    bar_accent = "strong" if is_strong else ""

    rtype = str(cand["report_type"]).lower()
    type_cls = "badge-lost" if rtype == "lost" else "badge-found"
    status = str(cand["status"]).lower()
    status_cls = "badge-recovered" if status == "recovered" else "badge-active"

    top_pick_html = (
        '<span class="rf-badge badge-top-pick">Top Candidate</span>'
        if rank == 1 and result.final_score >= 60.0
        else ""
    )

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

    sem_bar_cls = "green" if sem_pct >= 70 else ""
    cat_bar_cls = "green" if cat_pct >= 100 else ""
    loc_bar_cls = "green" if loc_pct >= 70 else ""
    date_bar_cls = "green" if date_pct >= 70 else ""

    days_label = (
        "Same day"
        if result.days_apart == 0
        else (f"{result.days_apart}d apart" if result.days_apart is not None else "Date N/A")
    )
    overall_pct = max(0.0, min(100.0, result.final_score))

    st.markdown(
        f"""
        <div class="match-card {card_accent}">
            <div class="rf-card-header">
                <div style="min-width:0; flex:1;">
                    <div style="display:flex; align-items:center; gap:0.4rem; flex-wrap:wrap; margin-bottom:0.4rem;">
                        {top_pick_html}
                        <span class="rf-badge {type_cls}"><span class="rf-badge-dot"></span>{esc(rtype.upper())} #{int(cand["id"])}</span>
                        <span class="rf-badge {tier_cls}">Rank #{rank} &middot; {esc(tier_label)}</span>
                        <span class="rf-badge {status_cls}">{esc(status.upper())}</span>
                    </div>
                    <div class="rf-card-title">{esc(cand["title"])}</div>
                </div>
                <div class="match-score-box">
                    <div class="match-score-number">{result.final_score:.1f}<span style="font-size:0.95rem;color:#64748B;">/100</span></div>
                    <div class="match-score-sub">Similarity Score</div>
                </div>
            </div>

            <div class="overall-bar-wrap">
                <div class="overall-bar-meta">
                    <span><strong>Weighted Similarity Index</strong> (Cosine: {result.raw_cosine_similarity:.3f})</span>
                    <span>{sem_pts:.1f} + {cat_pts:.1f} + {loc_pts:.1f} + {date_pts:.1f} = <strong>{result.final_score:.1f} / 100</strong></span>
                </div>
                <div class="overall-bar-track">
                    <div class="overall-bar-fill {bar_accent}" style="width:{overall_pct:.1f}%;"></div>
                </div>
            </div>

            <p class="rf-card-desc">{esc(cand["description"])}</p>

            <div class="rf-meta-row">
                <span class="rf-meta-item"><strong>Category:</strong> {esc(cand["category"])}</span>
                <span class="rf-meta-item"><strong>Location:</strong> {esc(cand["location"])}</span>
                <span class="rf-meta-item"><strong>Date:</strong> {esc(format_display_date(cand["report_date"]))} ({esc(days_label)})</span>
                <span class="rf-meta-item"><strong>Reference:</strong> {esc(cand.get("contact_label") or "Campus Desk")}</span>
            </div>

            <div class="signal-grid">
                <div class="signal-box">
                    <div class="signal-label"><span>Semantic Similarity</span><span>{sem_pct}%</span></div>
                    <div class="signal-contrib">+{sem_pts:.1f} <span>/ {max_sem:.0f} pts</span></div>
                    <div class="signal-bar-track"><div class="signal-bar-fill {sem_bar_cls}" style="width:{sem_pct}%;"></div></div>
                </div>
                <div class="signal-box">
                    <div class="signal-label"><span>Category Match</span><span>{cat_pct}%</span></div>
                    <div class="signal-contrib">+{cat_pts:.1f} <span>/ {max_cat:.0f} pts</span></div>
                    <div class="signal-bar-track"><div class="signal-bar-fill {cat_bar_cls}" style="width:{cat_pct}%;"></div></div>
                </div>
                <div class="signal-box">
                    <div class="signal-label"><span>Location Match</span><span>{loc_pct}%</span></div>
                    <div class="signal-contrib">+{loc_pts:.1f} <span>/ {max_loc:.0f} pts</span></div>
                    <div class="signal-bar-track"><div class="signal-bar-fill {loc_bar_cls}" style="width:{loc_pct}%;"></div></div>
                </div>
                <div class="signal-box">
                    <div class="signal-label"><span>Date Proximity</span><span>{date_pct}%</span></div>
                    <div class="signal-contrib">+{date_pts:.1f} <span>/ {max_date:.0f} pts</span></div>
                    <div class="signal-bar-track"><div class="signal-bar-fill {date_bar_cls}" style="width:{date_pct}%;"></div></div>
                </div>
            </div>

            <div class="xai-box">
                <div class="xai-title">Explainable AI Rationale</div>
                {esc(result.explanation)}
                <div class="xai-footnote">
                    Similarity score ({result.final_score:.1f}/100) reflects weighted vector &amp; contextual closeness — not a statistical probability of item identity.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    img_path = resolve_image_path(cand.get("image_path"))
    if img_path is not None:
        with st.expander(f"View uploaded photo for Report #{cand['id']}"):
            st.image(str(img_path), width=320)

    if allow_recovery_actions and source_report.get("id") and cand.get("status") == "active":
        col_a, col_b, _ = st.columns([1.45, 1.45, 2.1])
        with col_a:
            if source_report.get("status") == "active":
                if st.button(
                    "Mark Pair as Recovered",
                    key=f"{key_prefix}_recover_pair_{cand['id']}",
                    type="primary",
                    width="stretch",
                ):
                    try:
                        update_report_status(int(source_report["id"]), "recovered")
                        update_report_status(int(cand["id"]), "recovered")
                        st.toast(
                            f"Reports #{source_report['id']} and #{cand['id']} marked as recovered."
                        )
                        st.rerun()
                    except (sqlite3.Error, ValueError) as exc:
                        st.error(f"Could not update recovery status: {exc}")
        with col_b:
            if st.button(
                f"Mark #{cand['id']} Recovered",
                key=f"{key_prefix}_recover_cand_{cand['id']}",
                width="stretch",
            ):
                try:
                    update_report_status(int(cand["id"]), "recovered")
                    st.toast(f"Report #{cand['id']} marked as recovered.")
                    st.rerun()
                except (sqlite3.Error, ValueError) as exc:
                    st.error(f"Could not update recovery status: {exc}")


def render_dashboard_page() -> None:
    """Render the responsive campus overview dashboard."""
    render_page_header(
        "ReFind AI Dashboard",
        "Campus-wide lost & found overview powered by local semantic NLP matching.",
    )

    stats = get_dashboard_stats()
    recovery_rate = (
        round((stats["total_recovered"] / stats["total_reports"]) * 100)
        if stats["total_reports"] > 0
        else 0
    )

    # Responsive CSS Grid ensures all 4 KPI cards fit comfortably with zero horizontal overflow.
    st.markdown(
        f"""
        <div class="kpi-grid">
            <div class="kpi-card kpi-navy">
                <div class="kpi-label">Total Lost Items</div>
                <div class="kpi-value">{stats["total_lost"]}</div>
                <div class="kpi-caption">{stats["active_lost"]} currently active</div>
            </div>
            <div class="kpi-card kpi-slate">
                <div class="kpi-label">Total Found Items</div>
                <div class="kpi-value">{stats["total_found"]}</div>
                <div class="kpi-caption">{stats["active_found"]} currently active</div>
            </div>
            <div class="kpi-card kpi-green">
                <div class="kpi-label">Items Recovered</div>
                <div class="kpi-value accent-green">{stats["total_recovered"]}</div>
                <div class="kpi-caption">{recovery_rate}% of all reports resolved</div>
            </div>
            <div class="kpi-card kpi-blue">
                <div class="kpi-label">Active Reports</div>
                <div class="kpi-value accent-blue">{stats["active_reports"]}</div>
                <div class="kpi-caption">Open for AI semantic matching</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-heading">Quick Actions</div>', unsafe_allow_html=True)
    qa1, qa2, qa3 = st.columns(3, gap="small")
    with qa1:
        if st.button("Report a Lost Item", type="primary", width="stretch"):
            navigate_to("Report Lost Item")
    with qa2:
        if st.button("Report a Found Item", width="stretch"):
            navigate_to("Report Found Item")
    with qa3:
        if st.button("Open AI Matching Dashboard", width="stretch"):
            navigate_to("AI Matching")

    st.divider()

    left_col, right_col = st.columns([1.3, 1.0], gap="large")

    with left_col:
        st.markdown('<div class="section-heading">Recent Campus Reports</div>', unsafe_allow_html=True)
        recent = get_all_reports(limit=6)
        if not recent:
            st.info("No reports recorded yet. Use the buttons above to submit a lost or found report.")
        else:
            for item in recent:
                st.markdown(render_report_card_html(item, compact=True), unsafe_allow_html=True)
                if item["status"] == "active":
                    if st.button(
                        f"Find AI Matches for #{item['id']}",
                        key=f"dash_match_{item['id']}",
                    ):
                        navigate_to("AI Matching", report_id=int(item["id"]))

    with right_col:
        st.markdown(
            '<div class="section-heading">Top AI Match Highlights</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "Strongest active Lost ↔ Found pairs discovered by `all-MiniLM-L6-v2` (similarity scores, not probabilities)."
        )
        active_lost = get_all_reports(report_type="lost", status="active")
        active_found = get_all_reports(report_type="found", status="active")

        if not active_lost or not active_found:
            st.info("Submit at least one active Lost report and one active Found report to see AI match highlights.")
        else:
            engine = try_get_engine()
            if engine is not None:
                default_cfg = MatchConfig(min_score_threshold=45.0, max_results=1)
                highlights: list[tuple[dict[str, Any], MatchResult]] = []
                for lost_item in active_lost:
                    matches = find_matches(
                        lost_item, active_found, engine, config=default_cfg
                    )
                    if matches:
                        highlights.append((lost_item, matches[0]))

                highlights.sort(key=lambda pair: pair[1].final_score, reverse=True)
                top_highlights = highlights[:4]

                if not top_highlights:
                    st.info("No active pairs currently cross the 45.0 similarity highlight threshold.")
                else:
                    for lost_item, best_match in top_highlights:
                        cand = best_match.candidate
                        tier_label, tier_cls = score_tier(best_match.final_score)
                        bar_cls = "strong" if best_match.final_score >= 75.0 else ""
                        st.markdown(
                            f"""
                            <div class="rf-card" style="border-left: 3px solid {'#16A34A' if best_match.final_score >= 75.0 else '#2563EB'};">
                                <div class="rf-card-header">
                                    <span class="rf-badge {tier_cls}">{best_match.final_score:.1f}/100 &middot; {esc(tier_label)}</span>
                                    <span style="font-size:0.78rem;color:#64748B;">Semantic: {round(best_match.semantic_score*100)}%</span>
                                </div>
                                <div class="overall-bar-track" style="height:6px; margin:0.35rem 0 0.55rem 0;">
                                    <div class="overall-bar-fill {bar_cls}" style="width:{best_match.final_score:.1f}%;"></div>
                                </div>
                                <div style="font-size:0.89rem;line-height:1.45;">
                                    <strong>Lost #{int(lost_item["id"])}:</strong> {esc(lost_item["title"])}<br/>
                                    <strong>Found #{int(cand["id"])}:</strong> {esc(cand["title"])}
                                </div>
                                <div style="font-size:0.79rem;color:#475569;margin-top:0.4rem;line-height:1.45;">
                                    {esc(best_match.explanation)}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        if st.button(
                            f"Inspect Match for #{lost_item['id']}",
                            key=f"highlight_open_{lost_item['id']}",
                            width="stretch",
                        ):
                            navigate_to("AI Matching", report_id=int(lost_item["id"]))


def render_ai_matching_page() -> None:
    """Render the dedicated AI Semantic Matching & Academic Demonstration Dashboard."""
    render_page_header(
        "AI Semantic Matching Engine",
        "Natural-language item matching powered by sentence-transformers/all-MiniLM-L6-v2 and explainable hybrid scoring.",
    )

    # Academic workflow overview strip
    st.markdown(
        """
        <div class="workflow-grid">
            <div class="workflow-step">
                <div class="workflow-num">Step 1</div>
                <div class="workflow-title">Text Preprocessing</div>
                <div class="workflow-desc">Combines title, description, and notes into clean natural language while filtering to opposite report types.</div>
            </div>
            <div class="workflow-step">
                <div class="workflow-num">Step 2</div>
                <div class="workflow-title">Sentence Embeddings</div>
                <div class="workflow-desc">Encodes text into 384-D semantic vectors using local <code>all-MiniLM-L6-v2</code> so synonyms sit close together.</div>
            </div>
            <div class="workflow-step">
                <div class="workflow-num">Step 3</div>
                <div class="workflow-title">Cosine Similarity</div>
                <div class="workflow-desc">Measures angular vector closeness via Scikit-learn, independent of exact keyword overlap.</div>
            </div>
            <div class="workflow-step">
                <div class="workflow-num">Step 4</div>
                <div class="workflow-title">Hybrid Fusion &amp; XAI</div>
                <div class="workflow-desc">Combines Semantic (70%), Category (15%), Location (10%), and Date (5%) with evidence-grounded explanations.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("How Sentence Embeddings & Cosine Similarity Work (Academic Reference)", expanded=False):
        st.markdown(
            """
            - **Why Not Keyword Search?** A student may report *"blue water bottle"* while a finder logs *"navy reusable drinking bottle"*. Exact keyword matching misses synonymy and paraphrasing.
            - **Sentence Embeddings (`all-MiniLM-L6-v2`):** A 6-layer MiniLM transformer trained via contrastive Siamese learning maps each item description into a 384-dimensional vector $\\mathbf{v} \\in \\mathbb{R}^{384}$ representing semantic meaning.
            - **Cosine Similarity:** Measures the cosine of the angle between two embedding vectors:
              $$\\cos(\\theta) = \\frac{\\mathbf{u} \\cdot \\mathbf{v}}{\\|\\mathbf{u}\\|_2 \\|\\mathbf{v}\\|_2}$$
              Values are clamped to $[0, 1]$ for weighted scoring, where higher values indicate closer semantic alignment.
            - **Hybrid Scoring Formula (0–100):**
              $$\\text{Final Score} = 100 \\times (0.70 \\cdot S_{\\text{sem}} + 0.15 \\cdot S_{\\text{cat}} + 0.10 \\cdot S_{\\text{loc}} + 0.05 \\cdot S_{\\text{date}})$$
            - **Similarity Score vs. Probability:** A score of **82.4 / 100** indicates strong relative similarity across semantic and contextual signals — it does **not** represent an 82.4% probability that two reports describe the identical physical object.
            """
        )

    st.markdown(
        """
        <div class="academic-note">
            <strong>Important Evaluation Note:</strong> ReFind AI scores (0–100) represent a
            <strong>weighted similarity index</strong>, <strong>not</strong> a statistical probability.
            Always verify distinguishing details before returning an item.
        </div>
        """,
        unsafe_allow_html=True,
    )

    engine = try_get_engine()
    if engine is None:
        return

    all_reports = get_all_reports()

    # Quick academic demo preset selector when seed reports are present
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
        st.caption("Quick Academic Demo Cases (click to inspect realistic sample reports):")
        preset_cols = st.columns(len(available_presets), gap="small")
        for col, (label, rep_id) in zip(preset_cols, available_presets):
            with col:
                if st.button(label, key=f"demo_preset_{rep_id}", width="stretch"):
                    st.session_state["selected_match_report_id"] = rep_id
                    st.session_state["ai_match_select_id"] = rep_id
                    st.rerun()

    include_recovered_source = st.checkbox(
        "Include already-recovered reports in source selector",
        value=False,
    )
    source_pool = (
        all_reports
        if include_recovered_source
        else [r for r in all_reports if r["status"] == "active"]
    )

    if not source_pool:
        st.info("No reports available for matching. Submit a Lost or Found report first.")
        return

    report_by_id = {int(r["id"]): r for r in source_pool}
    option_ids = [int(r["id"]) for r in source_pool]

    # Deterministically synchronize integer report ID selection across navigation and reruns.
    requested_id = st.session_state.get("selected_match_report_id")
    if requested_id in report_by_id:
        st.session_state["ai_match_select_id"] = int(requested_id)
    elif st.session_state.get("ai_match_select_id") not in report_by_id:
        st.session_state["ai_match_select_id"] = option_ids[0]

    def _format_id_option(rep_id: int) -> str:
        rep = report_by_id[int(rep_id)]
        return (
            f"[{rep['report_type'].upper()} #{rep['id']}] {rep['title']} "
            f"— {rep['location']} ({format_display_date(rep['report_date'])})"
        )

    selected_id = st.selectbox(
        "Select a report to analyze with ReFind AI:",
        options=option_ids,
        format_func=_format_id_option,
        key="ai_match_select_id",
    )
    st.session_state["selected_match_report_id"] = int(selected_id)
    selected_report = report_by_id[int(selected_id)]

    # Configurable algorithm weights & threshold
    with st.expander("Configure Hybrid Scoring Weights & Threshold", expanded=False):
        st.caption(
            "Default formula: 70% Semantic + 15% Category + 10% Location + 5% Date Proximity. "
            "Adjust sliders to test sensitivity; weights normalize automatically to 100%."
        )
        wc1, wc2, wc3, wc4 = st.columns(4)
        raw_sem = wc1.slider("Semantic Weight", 0, 100, 70, step=5)
        raw_cat = wc2.slider("Category Weight", 0, 100, 15, step=5)
        raw_loc = wc3.slider("Location Weight", 0, 100, 10, step=5)
        raw_date = wc4.slider("Date Weight", 0, 100, 5, step=5)

        total_raw = raw_sem + raw_cat + raw_loc + raw_date
        if total_raw == 0:
            w_sem, w_cat, w_loc, w_date = 0.70, 0.15, 0.10, 0.05
            st.warning("All weights were 0; reverted to default 70/15/10/5 weights.")
        else:
            w_sem = raw_sem / total_raw
            w_cat = raw_cat / total_raw
            w_loc = raw_loc / total_raw
            w_date = raw_date / total_raw

        tc1, tc2 = st.columns([2, 1.5])
        min_threshold = tc1.slider(
            "Minimum Display Score (0–100)",
            min_value=0,
            max_value=85,
            value=35,
            step=5,
        )
        show_below_threshold = tc2.checkbox(
            "Show all opposite-type candidates (ignore threshold)",
            value=False,
        )
        include_recovered_candidates = st.checkbox(
            "Include recovered items in candidate pool",
            value=False,
        )

    config = MatchConfig(
        weight_semantic=w_sem,
        weight_category=w_cat,
        weight_location=w_loc,
        weight_date=w_date,
        min_score_threshold=float(min_threshold),
        max_results=10,
    )

    st.markdown('<div class="section-heading">Selected Source Report</div>', unsafe_allow_html=True)
    st.markdown(render_report_card_html(selected_report, compact=False), unsafe_allow_html=True)

    source_img = resolve_image_path(selected_report.get("image_path"))
    if source_img is not None:
        with st.expander("View uploaded photo for selected report"):
            st.image(str(source_img), width=320)

    opposite_label = "FOUND" if selected_report["report_type"] == "lost" else "LOST"
    matches = find_matches(
        source_report=selected_report,
        candidate_reports=all_reports,
        engine=engine,
        config=config,
        include_recovered=include_recovered_candidates,
        apply_threshold=not show_below_threshold,
    )

    st.markdown(
        f'<div class="section-heading">Ranked AI Recommendations ({opposite_label} Reports)</div>',
        unsafe_allow_html=True,
    )

    if not matches:
        st.info(
            f"No {opposite_label.lower()} reports scored at or above {min_threshold:.0f}/100. "
            "You can lower the threshold or toggle 'Show all opposite-type candidates' above to inspect lower-ranked comparisons."
        )
    else:
        st.caption(
            f"Showing {len(matches)} ranked {opposite_label.lower()} candidate(s) evaluated with "
            f"`{engine.model_name}` ({engine.embedding_dim}-D embeddings, Scikit-learn cosine similarity)."
        )
        for rank, result in enumerate(matches, start=1):
            render_match_result_card(
                rank=rank,
                result=result,
                source_report=selected_report,
                config=config,
                key_prefix=f"matcher_{selected_report['id']}_{rank}",
            )


def _init_form_state(report_type: str, force_reset: bool = False) -> None:
    """Initialize or cleanly reset controlled session-state keys for a report form."""
    defaults: dict[str, Any] = {
        f"{report_type}_title": "",
        f"{report_type}_category": CATEGORIES[0],
        f"{report_type}_description": "",
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


def render_report_form_page(report_type: str) -> None:
    """Render either the Report Lost Item or Report Found Item form with reliable state handling."""
    is_lost = report_type == "lost"
    page_title = "Report a Lost Item" if is_lost else "Report a Found Item"
    page_subtitle = (
        "Describe what you misplaced in natural language. ReFind AI saves your report and immediately checks active found items."
        if is_lost
        else "Log an item you found on campus. ReFind AI saves your report and immediately checks active lost reports."
    )
    render_page_header(page_title, page_subtitle)

    # Reset form inputs ONLY after a previously verified successful submission.
    should_reset = bool(st.session_state.pop(f"reset_form_{report_type}", False))
    _init_form_state(report_type, force_reset=should_reset)

    # Display flash confirmation banner persisted across post-submit rerun.
    flash_msg = st.session_state.pop(f"flash_success_{report_type}", None)
    if flash_msg:
        st.success(flash_msg)

    loc_label = "Last Seen Location *" if is_lost else "Location Found *"
    date_label = "Date Lost *" if is_lost else "Date Found *"
    extra_label = (
        "Additional Identifying Details (Optional)"
        if is_lost
        else "Additional Observations (Optional)"
    )
    extra_placeholder = (
        "E.g., small scratch on back, specific sticker, keychain, or engraving..."
        if is_lost
        else "E.g., handed in to security desk, condition of item, notable markings..."
    )
    desc_placeholder = (
        "E.g., Lost my dark brown leather wallet near the library entrance. Slim bifold style with stitched edges."
        if is_lost
        else "E.g., Found a brown leather purse-like wallet on a bench outside the central library."
    )

    upload_nonce = int(st.session_state[f"{report_type}_upload_nonce"])

    # clear_on_submit=False and enter_to_submit=False prevent premature Enter-key submits
    # and preserve user input if any validation error occurs.
    with st.form(
        key=f"form_report_{report_type}",
        clear_on_submit=False,
        enter_to_submit=False,
    ):
        col1, col2 = st.columns([1.4, 1.0])
        with col1:
            title = st.text_input(
                "Item Name / Title *",
                placeholder="E.g., Dark Brown Leather Wallet",
                max_chars=120,
                key=f"{report_type}_title",
            )
        with col2:
            category = st.selectbox(
                "Item Category *",
                options=CATEGORIES,
                key=f"{report_type}_category",
            )

        description = st.text_area(
            "Detailed Description *",
            placeholder=desc_placeholder,
            height=115,
            help="Include color, material, brand, size, or distinctive features so the NLP model can identify semantic matches.",
            key=f"{report_type}_description",
        )

        col3, col4 = st.columns([1.4, 1.0])
        with col3:
            location = st.text_input(
                loc_label,
                placeholder="E.g., Central Library, Room 204, Main Cafeteria",
                help="Common campus areas: " + ", ".join(CAMPUS_LOCATION_SUGGESTIONS[:5]),
                key=f"{report_type}_location",
            )
        with col4:
            report_date = st.date_input(
                date_label,
                max_value=date.today() + timedelta(days=1),
                key=f"{report_type}_date",
            )

        additional_details = st.text_input(
            extra_label,
            placeholder=extra_placeholder,
            max_chars=240,
            key=f"{report_type}_extra",
        )

        col5, col6 = st.columns([1.2, 1.2])
        with col5:
            contact_label = st.text_input(
                "Campus Desk / Demo Reference (Optional)",
                placeholder="E.g., Library Desk Ref #108 (Demo)",
                help="For privacy during academic demos, use a synthetic desk reference rather than personal phone numbers or emails.",
                key=f"{report_type}_contact",
            )
        with col6:
            uploaded_photo = st.file_uploader(
                "Optional Photograph (PNG, JPG, WEBP)",
                type=["png", "jpg", "jpeg", "webp"],
                key=f"{report_type}_photo_{upload_nonce}",
            )

        submitted = st.form_submit_button(
            f"Submit {'Lost' if is_lost else 'Found'} Report & Run AI Match",
            type="primary",
            width="stretch",
        )

    if submitted:
        errors: list[str] = []
        clean_title = (title or "").strip()
        clean_desc = (description or "").strip()
        clean_loc = (location or "").strip()

        if not clean_title:
            errors.append("Please enter an item name/title.")
        if not clean_desc or len(clean_desc) < 5:
            errors.append("Please provide a descriptive summary (at least 5 characters).")
        if not clean_loc:
            errors.append("Please enter the campus location.")

        # Defensively handle date picker returning None or a tuple if cleared.
        valid_date: Optional[date] = None
        if isinstance(report_date, date):
            valid_date = report_date
        elif isinstance(report_date, (tuple, list)) and report_date and isinstance(report_date[0], date):
            valid_date = report_date[0]

        if valid_date is None:
            errors.append("Please select a valid date.")
        elif valid_date > date.today() + timedelta(days=1):
            errors.append("The selected date cannot be in the future.")

        if not errors and valid_date is not None:
            # Prevent accidental duplicate submissions (e.g., double-click or repeat submit).
            existing_dup = find_duplicate_report(
                title=clean_title,
                description=clean_desc,
                report_type=report_type,
                location=clean_loc,
                report_date=valid_date.isoformat(),
            )
            if existing_dup is not None:
                st.session_state[f"last_submitted_{report_type}_id"] = int(existing_dup["id"])
                st.warning(
                    f"An identical active {report_type} report (**#{existing_dup['id']} — {existing_dup['title']}**) "
                    "is already saved in the database. Showing its AI match results below without creating a duplicate."
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
                            location=clean_loc,
                            report_date=valid_date.isoformat(),
                            additional_details=(additional_details or "").strip(),
                            contact_label=(contact_label or "").strip()
                            or "Campus Lost & Found Desk (Demo)",
                            image_path=image_rel_path,
                        )
                    except (sqlite3.Error, ValueError) as exc:
                        errors.append(f"Database error while saving report: {exc}")
                    else:
                        # Sync sidebar counts, clear form cleanly, and show confirmation + AI matches.
                        st.session_state[f"last_submitted_{report_type}_id"] = int(new_id)
                        st.session_state[f"flash_success_{report_type}"] = (
                            f"Report **#{new_id} ({clean_title})** has been saved to SQLite. "
                            "Your report is now live on the Dashboard, Browse Reports, and AI Matching below."
                        )
                        st.session_state[f"reset_form_{report_type}"] = True
                        st.rerun()

        if errors:
            for err in errors:
                st.error(err)

    # Show immediate AI matches for the most recently submitted report on this form.
    last_id = st.session_state.get(f"last_submitted_{report_type}_id")
    if last_id is not None:
        saved_report = get_report_by_id(int(last_id))
        if saved_report is None:
            st.session_state.pop(f"last_submitted_{report_type}_id", None)
        else:
            st.divider()
            st.markdown(
                f'<div class="section-heading">Instant AI Matches for Report #{saved_report["id"]}: {esc(saved_report["title"])}</div>',
                unsafe_allow_html=True,
            )
            ac1, ac2, _ = st.columns([1.45, 1.35, 1.2])
            with ac1:
                if st.button(
                    "Open in AI Matching Dashboard",
                    key=f"jump_after_submit_{saved_report['id']}",
                    type="primary",
                    width="stretch",
                ):
                    navigate_to("AI Matching", report_id=int(saved_report["id"]))
            with ac2:
                if st.button(
                    "View in Browse Reports",
                    key=f"browse_after_submit_{saved_report['id']}",
                    width="stretch",
                ):
                    navigate_to("Browse Reports")

            engine = try_get_engine()
            if engine is not None:
                cfg = MatchConfig(max_results=5)
                candidates = get_all_reports()
                matches = find_matches(saved_report, candidates, engine, config=cfg)
                if not matches:
                    st.info(
                        "No opposite-type active reports crossed the 35.0 similarity threshold yet. "
                        "Your report is stored in SQLite and will be matched automatically as new opposite-type reports arrive."
                    )
                else:
                    for idx, res in enumerate(matches, start=1):
                        render_match_result_card(
                            rank=idx,
                            result=res,
                            source_report=saved_report,
                            config=cfg,
                            key_prefix=f"instant_{saved_report['id']}_{idx}",
                        )


def render_browse_reports_page() -> None:
    """Render searchable, filterable report catalog with recovery & demo controls."""
    render_page_header(
        "Browse & Manage Reports",
        "Search, filter, inspect details, run AI matching, and manage item recovery status.",
    )

    f1, f2, f3, f4 = st.columns([1.8, 1.0, 1.2, 1.0])
    with f1:
        search_query = st.text_input(
            "Search reports",
            placeholder="Search title, description, or location...",
        )
    with f2:
        type_choice = st.selectbox("Report Type", ["All", "Lost", "Found"])
    with f3:
        category_choice = st.selectbox("Category", ["All", *CATEGORIES])
    with f4:
        status_choice = st.selectbox("Status", ["All", "Active", "Recovered"])

    reports = get_all_reports(
        report_type=None if type_choice == "All" else type_choice.lower(),
        category=None if category_choice == "All" else category_choice,
        status=None if status_choice == "All" else status_choice.lower(),
        search_query=search_query,
    )

    st.caption(f"Displaying **{len(reports)}** matching report(s), newest submitted first.")

    if not reports:
        st.info("No reports match the current filters.")
    else:
        for report in reports:
            st.markdown(render_report_card_html(report, compact=False), unsafe_allow_html=True)

            img_path = resolve_image_path(report.get("image_path"))
            if img_path is not None:
                with st.expander(f"View uploaded photograph — Report #{report['id']}"):
                    st.image(str(img_path), width=320)

            b1, b2, b3, _ = st.columns([1.25, 1.35, 0.9, 2.0])
            with b1:
                if st.button(
                    "Find AI Matches",
                    key=f"browse_match_{report['id']}",
                    width="stretch",
                ):
                    navigate_to("AI Matching", report_id=int(report["id"]))
            with b2:
                if report["status"] == "active":
                    if st.button(
                        "Mark as Recovered",
                        key=f"browse_recover_{report['id']}",
                        type="primary",
                        width="stretch",
                    ):
                        try:
                            update_report_status(int(report["id"]), "recovered")
                            st.toast(f"Report #{report['id']} marked as recovered.")
                            st.rerun()
                        except (sqlite3.Error, ValueError) as exc:
                            st.error(f"Could not update report status: {exc}")
                else:
                    if st.button(
                        "Reopen as Active",
                        key=f"browse_reopen_{report['id']}",
                        width="stretch",
                    ):
                        try:
                            update_report_status(int(report["id"]), "active")
                            st.toast(f"Report #{report['id']} reopened as active.")
                            st.rerun()
                        except (sqlite3.Error, ValueError) as exc:
                            st.error(f"Could not reopen report: {exc}")
            with b3:
                if st.button(
                    "Delete",
                    key=f"browse_delete_{report['id']}",
                    width="stretch",
                ):
                    try:
                        rep_id = int(report["id"])
                        delete_report(rep_id)
                        if st.session_state.get("selected_match_report_id") == rep_id:
                            st.session_state.pop("selected_match_report_id", None)
                        for rtype in ("lost", "found"):
                            if st.session_state.get(f"last_submitted_{rtype}_id") == rep_id:
                                st.session_state.pop(f"last_submitted_{rtype}_id", None)
                        st.toast(f"Report #{rep_id} deleted.")
                        st.rerun()
                    except sqlite3.Error as exc:
                        st.error(f"Could not delete report: {exc}")

    st.divider()
    with st.expander("Demo Dataset Administration"):
        st.caption(
            "Manage synthetic demonstration records without affecting your own submitted reports."
        )
        dc1, dc2 = st.columns(2)
        with dc1:
            if st.button("Restore Missing Demo Records", width="stretch"):
                added = seed_database()
                st.toast(f"Restored {added} synthetic demonstration record(s).")
                st.rerun()
        with dc2:
            if st.button("Remove Synthetic Demo Records", width="stretch"):
                removed = delete_seed_reports()
                st.toast(f"Removed {removed} synthetic demonstration record(s).")
                st.rerun()


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

    # Handle programmatic navigation requests safely before instantiating sidebar radio.
    if "nav_target" in st.session_state:
        st.session_state["nav_radio"] = st.session_state.pop("nav_target")
    if "nav_radio" not in st.session_state:
        st.session_state["nav_radio"] = "Dashboard"

    with st.sidebar:
        st.markdown(
            """
            <div class="brand-box">
                <div class="brand-title"><span class="brand-dot"></span>ReFind AI</div>
                <p class="brand-tagline">Lost something? Let AI help you find it.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        selected_page = st.radio(
            "Navigation",
            options=PAGES,
            key="nav_radio",
        )

        st.divider()
        stats = get_dashboard_stats()
        st.markdown(
            f"""
            <div class="sidebar-status-card">
                <div class="sidebar-status-title"><span class="status-pulse"></span>System Status</div>
                <div><strong>AI Model:</strong> all-MiniLM-L6-v2</div>
                <div><strong>Storage:</strong> SQLite (Persistent)</div>
                <div><strong>Active Cases:</strong> {stats["active_reports"]} ({stats["active_lost"]} lost / {stats["active_found"]} found)</div>
                <div><strong>Recovered:</strong> {stats["total_recovered"]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

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


if __name__ == "__main__":
    main()
