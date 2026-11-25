"""Automated verification suite for ReFind AI matching engine, UI workflows, and SQLite persistence.

Covers all required academic validation scenarios and v1.1 workflow tests:
1. Strong semantic match (wallet near library)
2. Different wording / paraphrased description (blue water bottle vs. navy drinking bottle)
3. Different categories & unrelated items (wallet vs. wireless headphones)
4. Same category, different item contrast (red backpack vs. blue school bag)
5. Strict opposite report-type enforcement (no lost↔lost or found↔found matches)
6. SQLite persistence across reopened connections & idempotent seeding
7. End-to-end Lost Item submission via Streamlit AppTest (SQLite, Dashboard, Browse, AI Matching, duplicate guard)
8. End-to-end Found Item submission via Streamlit AppTest (SQLite, Browse, AI Matching)
9. Form validation preserving user inputs (no wipeout on error)
10. Recovery management & non-resurrecting demo record removal
11. Weighted contribution breakdown accuracy & all five views rendering cleanly
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import database
from ai_matcher import (
    DEFAULT_MODEL_NAME,
    EmbeddingEngine,
    MatchConfig,
    compute_category_score,
    compute_date_proximity_score,
    compute_hybrid_score,
    compute_location_score,
    find_matches,
)
from database import (
    count_reports,
    delete_seed_reports,
    get_all_reports,
    get_dashboard_stats,
    get_report_by_id,
    init_db,
    insert_report,
    update_report_status,
)
from seed_data import ensure_seed_data, seed_database


APP_PATH = Path(__file__).resolve().parent.parent / "app.py"


@pytest.fixture(scope="module")
def engine() -> EmbeddingEngine:
    """Load the genuine pretrained Sentence Transformer model once for the test module."""
    return EmbeddingEngine(model_name=DEFAULT_MODEL_NAME)


@pytest.fixture()
def isolated_app_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point database.DEFAULT_DB_PATH to an isolated temp SQLite file for UI AppTest runs."""
    db_file = tmp_path / "app_test_refind.db"
    monkeypatch.setattr(database, "DEFAULT_DB_PATH", db_file)
    init_db(db_file)
    ensure_seed_data(db_file)
    return db_file


# ---------------------------------------------------------------------------
# Required Core Academic Validation Tests (Tests 1 - 6)
# ---------------------------------------------------------------------------


def test_1_strong_semantic_match(engine: EmbeddingEngine) -> None:
    """Test 1: Strong semantic match should produce high semantic and final scores."""
    lost_wallet = {
        "id": 1,
        "title": "Black Leather Wallet",
        "description": "Black leather wallet lost near the library.",
        "category": "Wallet & Cards",
        "report_type": "lost",
        "location": "Near the Library",
        "report_date": "2026-11-24",
        "status": "active",
    }
    found_wallet = {
        "id": 2,
        "title": "Dark Black Leather Wallet",
        "description": "Found a dark black leather wallet outside the central library.",
        "category": "Wallet & Cards",
        "report_type": "found",
        "location": "Outside the Central Library",
        "report_date": "2026-11-24",
        "status": "active",
    }

    matches = find_matches(lost_wallet, [found_wallet], engine, apply_threshold=False)
    assert len(matches) == 1
    top = matches[0]
    assert top.semantic_score >= 0.75
    assert top.category_score == 1.0
    assert top.location_score > 0.0
    assert top.date_score == 1.0
    assert top.final_score >= 75.0
    assert "semantic similarity" in top.explanation.lower()


def test_2_different_wording_paraphrase(engine: EmbeddingEngine) -> None:
    """Test 2: Different wording ('blue water bottle' vs 'navy reusable drinking bottle')."""
    lost_bottle = {
        "id": 10,
        "title": "Blue Water Bottle",
        "description": "My blue water bottle is missing from the classroom.",
        "category": "Bottles & Drinkware",
        "report_type": "lost",
        "location": "Classroom",
        "report_date": "2026-11-23",
        "status": "active",
    }
    found_bottle = {
        "id": 11,
        "title": "Navy Reusable Drinking Bottle",
        "description": "Found a navy reusable drinking bottle in Room 204.",
        "category": "Bottles & Drinkware",
        "report_type": "found",
        "location": "Room 204",
        "report_date": "2026-11-24",
        "status": "active",
    }
    unrelated_headphones = {
        "id": 12,
        "title": "Silver Wireless Headphones",
        "description": "Found silver over-ear Bluetooth headphones near the gym.",
        "category": "Electronics",
        "report_type": "found",
        "location": "Sports Complex",
        "report_date": "2026-11-24",
        "status": "active",
    }

    matches = find_matches(
        lost_bottle,
        [unrelated_headphones, found_bottle],
        engine,
        apply_threshold=False,
    )
    assert len(matches) == 2
    assert matches[0].candidate["id"] == 11
    assert matches[0].semantic_score >= 0.50
    assert matches[0].final_score >= 50.0
    assert matches[0].final_score > matches[1].final_score + 25.0


def test_3_different_categories_unrelated_items(engine: EmbeddingEngine) -> None:
    """Test 3: Completely different items & categories must receive a low score."""
    lost_wallet = {
        "id": 20,
        "title": "Black Leather Wallet",
        "description": "Black leather wallet.",
        "category": "Wallet & Cards",
        "report_type": "lost",
        "location": "Central Library",
        "report_date": "2026-11-20",
        "status": "active",
    }
    found_headphones = {
        "id": 21,
        "title": "Silver Wireless Headphones",
        "description": "Silver wireless headphones.",
        "category": "Electronics",
        "report_type": "found",
        "location": "Sports Complex",
        "report_date": "2026-11-25",
        "status": "active",
    }

    matches = find_matches(
        lost_wallet, [found_headphones], engine, apply_threshold=False
    )
    assert len(matches) == 1
    result = matches[0]
    assert result.category_score == 0.0
    assert result.location_score == 0.0
    assert result.semantic_score < 0.35
    assert result.final_score < 35.0
    assert not result.meets_threshold


def test_4_same_category_different_item(engine: EmbeddingEngine) -> None:
    """Test 4: Same category alone must not override description & contextual differences."""
    lost_red_backpack = {
        "id": 30,
        "title": "Red Backpack",
        "description": "Red backpack lost near the cafeteria.",
        "category": "Bags & Backpacks",
        "report_type": "lost",
        "location": "Main Cafeteria",
        "report_date": "2026-11-24",
        "status": "active",
    }
    found_blue_bag = {
        "id": 31,
        "title": "Blue School Bag",
        "description": "Blue school bag found at the sports field.",
        "category": "Bags & Backpacks",
        "report_type": "found",
        "location": "Sports Complex",
        "report_date": "2026-11-10",
        "status": "active",
    }
    found_red_backpack = {
        "id": 32,
        "title": "Red Campus Backpack",
        "description": "Found a red backpack with laptop sleeve outside the main cafeteria.",
        "category": "Bags & Backpacks",
        "report_type": "found",
        "location": "Main Cafeteria",
        "report_date": "2026-11-24",
        "status": "active",
    }

    matches = find_matches(
        lost_red_backpack,
        [found_blue_bag, found_red_backpack],
        engine,
        apply_threshold=False,
    )
    assert len(matches) == 2
    assert matches[0].candidate["id"] == 32
    assert matches[1].candidate["id"] == 31
    assert matches[0].category_score == 1.0
    assert matches[1].category_score == 1.0
    assert matches[0].final_score > matches[1].final_score + 20.0


def test_5_opposite_report_types_only(engine: EmbeddingEngine) -> None:
    """Test 5: Lost reports never match Lost; Found reports never match Found."""
    lost_a = {
        "id": 40,
        "title": "Black Leather Wallet",
        "description": "Lost black leather wallet near library.",
        "category": "Wallet & Cards",
        "report_type": "lost",
        "location": "Central Library",
        "report_date": "2026-11-24",
        "status": "active",
    }
    lost_b = {
        "id": 41,
        "title": "Black Leather Wallet",
        "description": "Lost black leather wallet near library.",
        "category": "Wallet & Cards",
        "report_type": "lost",
        "location": "Central Library",
        "report_date": "2026-11-24",
        "status": "active",
    }
    found_a = {
        "id": 42,
        "title": "Black Leather Wallet",
        "description": "Found black leather wallet near library.",
        "category": "Wallet & Cards",
        "report_type": "found",
        "location": "Central Library",
        "report_date": "2026-11-24",
        "status": "active",
    }
    found_b = {
        "id": 43,
        "title": "Black Leather Wallet",
        "description": "Found black leather wallet near library.",
        "category": "Wallet & Cards",
        "report_type": "found",
        "location": "Central Library",
        "report_date": "2026-11-24",
        "status": "active",
    }

    lost_matches = find_matches(lost_a, [lost_a, lost_b, found_a], engine, apply_threshold=False)
    assert [m.candidate["id"] for m in lost_matches] == [42]

    found_matches = find_matches(found_a, [found_a, found_b, lost_a], engine, apply_threshold=False)
    assert [m.candidate["id"] for m in found_matches] == [40]


def test_6_sqlite_persistence_and_seed_idempotency(tmp_path: Path) -> None:
    """Test 6: Inserted reports and status updates persist across restarts without duplicate seeds."""
    db_file = tmp_path / "test_refind.db"
    init_db(db_file)

    first_inserted = seed_database(db_file)
    assert first_inserted >= 12
    second_inserted = seed_database(db_file)
    assert second_inserted == 0
    initial_total = count_reports(db_file)

    # Even with an older report_date, newly inserted report must rank first in recent reports.
    older_date = (date.today() - timedelta(days=10)).isoformat()
    new_id = insert_report(
        title="Green Mechanical Pencil Case",
        description="Lost a dark green zippered canvas pencil pouch in Lecture Hall B.",
        category="Books & Stationery",
        report_type="lost",
        location="Lecture Hall B",
        report_date=older_date,
        db_path=db_file,
    )

    reloaded = get_report_by_id(new_id, db_path=db_file)
    assert reloaded is not None
    assert reloaded["title"] == "Green Mechanical Pencil Case"
    assert reloaded["status"] == "active"
    assert count_reports(db_file) == initial_total + 1

    recent = get_all_reports(limit=6, db_path=db_file)
    assert recent[0]["id"] == new_id

    stats_before = get_dashboard_stats(db_file)
    assert update_report_status(new_id, "recovered", db_path=db_file) is True
    stats_after = get_dashboard_stats(db_file)
    assert stats_after["total_recovered"] == stats_before["total_recovered"] + 1
    assert stats_after["active_reports"] == stats_before["active_reports"] - 1


# ---------------------------------------------------------------------------
# Unit Tests for Contextual Signals, Contributions, Edge Cases & Validation
# ---------------------------------------------------------------------------


def test_contextual_scoring_components() -> None:
    """Verify transparent category, location, date decay, and hybrid scoring formulas."""
    assert compute_category_score("Electronics", "electronics ") == 1.0
    assert compute_category_score("Electronics", "Eyewear") == 0.0

    exact_score, exact_reason, _ = compute_location_score("Central Library", "central library")
    assert exact_score == 1.0
    assert exact_reason == "exact"

    sub_score, sub_reason, shared = compute_location_score("Library", "Outside Central Library")
    assert sub_score == 0.85
    assert sub_reason == "substring"
    assert "library" in shared

    diff_score, diff_reason, _ = compute_location_score("Sports Complex", "Computer Science Lab")
    assert diff_score == 0.0
    assert diff_reason == "different"

    same_day, delta0 = compute_date_proximity_score("2026-11-25", "2026-11-25")
    assert same_day == 1.0 and delta0 == 0

    two_weeks, delta14 = compute_date_proximity_score("2026-11-25", "2026-11-11", decay_days=14.0)
    assert delta14 == 14
    assert math.isclose(two_weeks, math.exp(-1.0), rel_tol=1e-4)

    missing_date, delta_none = compute_date_proximity_score("", "2026-11-25")
    assert missing_date == 0.0 and delta_none is None

    score = compute_hybrid_score(
        semantic_score=0.80,
        category_score=1.0,
        location_score=0.50,
        date_score=1.0,
    )
    assert score == 81.0


def test_weighted_contributions_match_final_score(engine: EmbeddingEngine) -> None:
    """Ensure exposed weighted contributions sum accurately to final_score."""
    lost_item = {
        "id": 70,
        "title": "Black Leather Wallet",
        "description": "Lost dark leather bifold wallet near central library.",
        "category": "Wallet & Cards",
        "report_type": "lost",
        "location": "Central Library",
        "report_date": "2026-11-25",
        "status": "active",
    }
    found_item = {
        "id": 71,
        "title": "Dark Leather Bifold Wallet",
        "description": "Found brown-black leather wallet outside central library.",
        "category": "Wallet & Cards",
        "report_type": "found",
        "location": "Outside Central Library",
        "report_date": "2026-11-24",
        "status": "active",
    }

    matches = find_matches(lost_item, [found_item], engine, apply_threshold=False)
    assert len(matches) == 1
    res = matches[0]
    contrib_sum = (
        res.semantic_contribution
        + res.category_contribution
        + res.location_contribution
        + res.date_contribution
    )
    assert abs(contrib_sum - res.final_score) <= 0.25


def test_empty_description_and_missing_fields_graceful(engine: EmbeddingEngine) -> None:
    """Ensure empty descriptions or missing metadata never crash the AI pipeline."""
    sparse_lost = {
        "id": 90,
        "title": "",
        "description": "",
        "category": "",
        "report_type": "lost",
        "location": "",
        "report_date": "",
        "status": "active",
    }
    normal_found = {
        "id": 91,
        "title": "Black Umbrella",
        "description": "Found a compact folding black umbrella near main gate.",
        "category": "Other",
        "report_type": "found",
        "location": "Main Gate",
        "report_date": "2026-11-25",
        "status": "active",
    }

    matches = find_matches(sparse_lost, [normal_found], engine, apply_threshold=False)
    assert len(matches) == 1
    assert matches[0].semantic_score == 0.0
    assert matches[0].final_score == 0.0


def test_match_config_rejects_invalid_weights() -> None:
    """MatchConfig must enforce valid non-negative weights summing to 1.0."""
    with pytest.raises(ValueError):
        MatchConfig(weight_semantic=0.5, weight_category=0.5, weight_location=0.2, weight_date=0.0)
    with pytest.raises(ValueError):
        MatchConfig(weight_semantic=-0.1, weight_category=0.5, weight_location=0.4, weight_date=0.2)


def test_remove_demo_records_does_not_auto_resurrect(tmp_path: Path) -> None:
    """Removing synthetic demo records must remain removed across app reruns."""
    db_file = tmp_path / "demo_remove.db"
    init_db(db_file)
    inserted = ensure_seed_data(db_file)
    assert inserted > 0

    removed = delete_seed_reports(db_file)
    assert removed == inserted
    assert count_reports(db_file) == 0

    # Simulating next Streamlit rerun calling ensure_seed_data() must NOT re-insert seeds.
    assert ensure_seed_data(db_file) == 0
    assert count_reports(db_file) == 0


# ---------------------------------------------------------------------------
# End-to-End Streamlit AppTest Workflow Tests (v1.1)
# ---------------------------------------------------------------------------


def test_lost_item_submission_workflow_end_to_end(isolated_app_db: Path) -> None:
    """Verify Report Lost Item inserts into SQLite, resets form, guards duplicates, and reaches Dashboard, Browse & AI Matching."""
    before_stats = get_dashboard_stats(isolated_app_db)

    at = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
    assert not at.exception

    # Navigate to Report Lost Item
    at.sidebar.radio[0].set_value("Report Lost Item").run()
    assert not at.exception

    title_val = "Matte Black Leather Bi-Fold Wallet"
    desc_val = "Lost my matte black leather wallet outside the central library entrance."
    loc_val = "Central Library"

    at.text_input(key="lost_title").set_value(title_val)
    at.selectbox(key="lost_category").set_value("Wallet & Cards")
    at.text_area(key="lost_description").set_value(desc_val)
    at.text_input(key="lost_location").set_value(loc_val)
    at.date_input(key="lost_date").set_value(date.today() - timedelta(days=5))
    at.button[0].click().run()

    assert not at.exception
    assert len(at.error) == 0
    assert any(title_val in msg.value for msg in at.success)

    # Form must have cleanly reset after successful submission
    assert at.text_input(key="lost_title").value == ""
    assert at.text_area(key="lost_description").value == ""

    # Verify SQLite persistence and KPI update
    after_stats = get_dashboard_stats(isolated_app_db)
    assert after_stats["total_lost"] == before_stats["total_lost"] + 1
    assert after_stats["active_reports"] == before_stats["active_reports"] + 1

    # Duplicate submission must be blocked cleanly without adding a second row
    at.text_input(key="lost_title").set_value(title_val)
    at.selectbox(key="lost_category").set_value("Wallet & Cards")
    at.text_area(key="lost_description").set_value(desc_val)
    at.text_input(key="lost_location").set_value(loc_val)
    at.date_input(key="lost_date").set_value(date.today() - timedelta(days=5))
    at.button[0].click().run()
    assert not at.exception
    assert len(at.warning) >= 1
    assert get_dashboard_stats(isolated_app_db)["total_lost"] == after_stats["total_lost"]

    # Verify report appears immediately on Dashboard Recent Campus Reports (even with 5-day-old date)
    at.sidebar.radio[0].set_value("Dashboard").run()
    assert not at.exception
    dashboard_md = "\n".join(m.value for m in at.markdown)
    assert title_val in dashboard_md

    # Verify report appears in Browse Reports
    at.sidebar.radio[0].set_value("Browse Reports").run()
    assert not at.exception
    browse_md = "\n".join(m.value for m in at.markdown)
    assert title_val in browse_md

    # Verify report is available in AI Matching and matches opposite-type Found wallet
    at.sidebar.radio[0].set_value("AI Matching").run()
    assert not at.exception
    select_options = at.selectbox(key="ai_match_select_id").options
    newest_report = get_all_reports(report_type="lost", db_path=isolated_app_db)[0]
    assert any(f"[LOST #{newest_report['id']}]" in opt for opt in select_options)
    at.selectbox(key="ai_match_select_id").select(newest_report["id"]).run()
    assert not at.exception
    assert at.selectbox(key="ai_match_select_id").value == newest_report["id"]
    matcher_md = "\n".join(m.value for m in at.markdown)
    assert "Brown Leather Purse-Style Wallet" in matcher_md

    # Verify Recovery Management from Browse Reports persists to SQLite & updates KPIs
    at.sidebar.radio[0].set_value("Browse Reports").run()
    assert not at.exception
    at.button(key=f"browse_recover_{newest_report['id']}").click().run()
    assert not at.exception
    recovered_record = get_report_by_id(newest_report["id"], db_path=isolated_app_db)
    assert recovered_record is not None and recovered_record["status"] == "recovered"
    final_stats = get_dashboard_stats(isolated_app_db)
    assert final_stats["total_recovered"] == after_stats["total_recovered"] + 1
    assert final_stats["active_reports"] == after_stats["active_reports"] - 1


def test_found_item_submission_workflow_end_to_end(isolated_app_db: Path) -> None:
    """Verify Report Found Item inserts into SQLite and becomes immediately matchable against Lost reports."""
    before_stats = get_dashboard_stats(isolated_app_db)

    at = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
    at.sidebar.radio[0].set_value("Report Found Item").run()
    assert not at.exception

    title_val = "Found Casio FX-991EX Scientific Calculator"
    desc_val = "Found a black Casio scientific engineering calculator left near a CS Lab computer."
    loc_val = "Computer Science Lab"

    at.text_input(key="found_title").set_value(title_val)
    at.selectbox(key="found_category").set_value("Electronics")
    at.text_area(key="found_description").set_value(desc_val)
    at.text_input(key="found_location").set_value(loc_val)
    at.button[0].click().run()

    assert not at.exception
    assert len(at.error) == 0
    assert any(title_val in msg.value for msg in at.success)

    after_stats = get_dashboard_stats(isolated_app_db)
    assert after_stats["total_found"] == before_stats["total_found"] + 1

    # Instant AI matching below form should surface the matching Lost Casio calculator
    page_md = "\n".join(m.value for m in at.markdown)
    assert "Casio Scientific Calculator" in page_md


def test_form_validation_preserves_user_input_on_error(isolated_app_db: Path) -> None:
    """Invalid submission must show clear errors without inserting or wiping user input."""
    before_count = count_reports(isolated_app_db)

    at = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
    at.sidebar.radio[0].set_value("Report Lost Item").run()

    at.text_input(key="lost_title").set_value("Blue Keychain")
    at.text_area(key="lost_description").set_value("Key")  # Too short (<5 chars)
    at.text_input(key="lost_location").set_value("")  # Missing required location
    at.button[0].click().run()

    assert not at.exception
    assert len(at.error) == 2
    assert count_reports(isolated_app_db) == before_count
    # Previously typed values must still be preserved so user can fix and resubmit
    assert at.text_input(key="lost_title").value == "Blue Keychain"
    assert at.text_area(key="lost_description").value == "Key"


def test_all_five_views_render_without_exceptions(isolated_app_db: Path) -> None:
    """Confirm all five ReFind AI views load cleanly with zero exceptions."""
    at = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
    assert not at.exception
    for page in ["Dashboard", "AI Matching", "Report Lost Item", "Report Found Item", "Browse Reports"]:
        at.sidebar.radio[0].set_value(page).run()
        assert not at.exception, f"View '{page}' raised: {at.exception}"
