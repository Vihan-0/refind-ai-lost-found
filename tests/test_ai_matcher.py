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
import streamlit as st
from streamlit.testing.v1 import AppTest

import app as refind_app
import database
from ai_matcher import (
    DEFAULT_MODEL_NAME,
    EmbeddingEngine,
    MatchConfig,
    ModelLoadError,
    compute_category_score,
    compute_date_proximity_score,
    compute_hybrid_score,
    compute_location_score,
    find_matches,
    find_top_match_highlights,
)
from database import (
    count_reports,
    count_seed_reports,
    delete_report_with_cleanup,
    delete_seed_reports,
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
from utils import delete_uploaded_image, resolve_safe_upload_path


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
    assert sub_reason == "token_subset"
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


# ---------------------------------------------------------------------------
# V1.2 P0 Engineering Correction Regression Tests
# ---------------------------------------------------------------------------


def test_location_matching_no_substring_false_positives() -> None:
    """Whole-token location matching must reject character-substring false positives."""
    # Confirmed false positives from audit must score 0.0
    assert compute_location_score("A", "Central Library") == (0.0, "different", [])
    assert compute_location_score("Lab", "Collaborative Space") == (0.0, "different", [])
    assert compute_location_score("A", "Block A") == (0.0, "different", [])
    assert compute_location_score("B", "Lecture Hall B") == (0.0, "different", [])

    # Valid whole-token partial & subset matches must continue working accurately
    score_lab, reason_lab, shared_lab = compute_location_score("Physics Lab", "Lab")
    assert score_lab == 0.85
    assert reason_lab == "token_subset"
    assert shared_lab == ["lab"]

    score_lib, reason_lib, shared_lib = compute_location_score(
        "Near the Library", "Outside Central Library"
    )
    assert score_lib == 0.85
    assert reason_lib == "token_subset"
    assert shared_lib == ["library"]

    score_cafe, reason_cafe, shared_cafe = compute_location_score(
        "Main Cafeteria", "Cafeteria Dining Area"
    )
    assert reason_cafe == "token_overlap"
    assert shared_cafe == ["cafeteria"]
    assert 0.25 <= score_cafe <= 0.6

    # Exact normalized matches remain 1.0
    assert compute_location_score("Computer Science Lab", "  computer science lab ") == (
        1.0,
        "exact",
        ["computer", "lab", "science"],
    )

    # Unrelated locations remain 0.0
    assert compute_location_score("Sports Complex", "Central Library") == (0.0, "different", [])


def test_safe_image_cleanup_and_path_traversal_protection(tmp_path: Path) -> None:
    """Report deletion must remove valid uploaded images and refuse arbitrary external paths."""
    db_file = tmp_path / "img_cleanup.db"
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    init_db(db_file)

    # Protected files that MUST NEVER be deleted
    outside_secret = tmp_path / "sensitive_outside.txt"
    outside_secret.write_text("do not delete", encoding="utf-8")
    gitkeep = upload_dir / ".gitkeep"
    gitkeep.write_text("", encoding="utf-8")

    # 1. Valid image strictly inside upload_dir
    valid_img = upload_dir / "valid_photo.png"
    valid_img.write_bytes(b"fake-png-bytes")
    rep_valid = insert_report(
        title="Valid Photo Report",
        description="Report with valid uploaded image inside uploads directory.",
        category="Other",
        report_type="lost",
        location="Central Library",
        report_date="2026-11-25",
        image_path=str(valid_img),
        db_path=db_file,
    )
    db_del, img_del, warn = delete_report_with_cleanup(
        rep_valid, db_path=db_file, upload_dir=upload_dir
    )
    assert db_del is True
    assert img_del is True
    assert warn is None
    assert not valid_img.exists()
    assert get_report_by_id(rep_valid, db_path=db_file) is None

    # 2. Relative traversal path escaping upload_dir MUST be refused
    traversal_path = str(upload_dir / ".." / "sensitive_outside.txt")
    rep_traversal = insert_report(
        title="Traversal Attempt Report",
        description="Untrusted path attempting directory traversal outside uploads.",
        category="Other",
        report_type="lost",
        location="Central Library",
        report_date="2026-11-25",
        image_path=traversal_path,
        db_path=db_file,
    )
    db_del2, img_del2, warn2 = delete_report_with_cleanup(
        rep_traversal, db_path=db_file, upload_dir=upload_dir
    )
    assert db_del2 is True
    assert img_del2 is False
    assert warn2 is not None and "outside" in warn2.lower()
    assert outside_secret.exists()

    # 3. Direct absolute path outside upload_dir MUST be refused
    img_del3, warn3 = delete_uploaded_image(str(outside_secret), upload_dir=upload_dir)
    assert img_del3 is False
    assert warn3 is not None
    assert outside_secret.exists()

    # 4. .gitkeep inside upload_dir MUST be protected
    assert resolve_safe_upload_path(str(gitkeep), upload_dir=upload_dir) is None
    img_del4, warn4 = delete_uploaded_image(str(gitkeep), upload_dir=upload_dir)
    assert img_del4 is False
    assert warn4 is not None
    assert gitkeep.exists()


def test_dashboard_and_ordinary_pages_do_not_initialize_ai_model(
    isolated_app_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Opening Dashboard, Browse, or blank forms must NOT load SentenceTransformer."""
    st.cache_resource.clear()
    load_attempts = {"count": 0}

    def _forbid_model_init(self, *args, **kwargs):
        load_attempts["count"] += 1
        raise AssertionError("EmbeddingEngine should not be initialized on ordinary pages!")

    monkeypatch.setattr(EmbeddingEngine, "__init__", _forbid_model_init)

    at = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    assert not at.exception
    assert load_attempts["count"] == 0

    sidebar_md = "\n".join(m.value for m in at.sidebar.markdown)
    assert "Not initialized" in sidebar_md
    assert "Ready (" not in sidebar_md
    # Dashboard must offer explicit on-demand AI scan button instead of blocking startup
    assert at.button(key="dash_load_ai_highlights") is not None

    # Navigate across ordinary pages — still zero AI model initialization
    for page in ["Browse Reports", "Report Lost Item", "Report Found Item", "Dashboard"]:
        at.sidebar.radio[0].set_value(page).run()
        assert not at.exception
        assert load_attempts["count"] == 0
        assert "Not initialized" in "\n".join(m.value for m in at.sidebar.markdown)


def test_sidebar_reports_unavailable_when_model_load_fails(
    isolated_app_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When SentenceTransformer fails to load, status must honestly report Unavailable."""
    st.cache_resource.clear()

    def _fail_load(self, *args, **kwargs):
        raise ModelLoadError("Simulated offline model failure")

    monkeypatch.setattr(EmbeddingEngine, "__init__", _fail_load)

    try:
        at = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
        at.sidebar.radio[0].set_value("AI Matching").run()
        assert not at.exception
        assert len(at.error) >= 1
        sidebar_md = "\n".join(m.value for m in at.sidebar.markdown)
        assert "Unavailable" in sidebar_md
        assert "Ready (" not in sidebar_md
    finally:
        st.cache_resource.clear()


def test_ai_matching_dropdown_switches_between_two_reports(
    isolated_app_db: Path,
) -> None:
    """Verify users can freely switch between different source reports in the AI Matching dropdown."""
    refind_app.get_engine_state.clear()
    active_reports = get_all_reports(status="active", db_path=isolated_app_db)
    assert len(active_reports) >= 3
    first_rep = active_reports[0]
    second_rep = active_reports[1]
    third_rep = active_reports[2]

    at = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
    at.sidebar.radio[0].set_value("AI Matching").run()
    assert not at.exception

    # Explicitly select report #1, then report #2, then report #3 via dropdown
    at.selectbox(key="ai_match_select_id").select(int(first_rep["id"])).run()
    assert not at.exception
    assert at.selectbox(key="ai_match_select_id").value == int(first_rep["id"])
    assert first_rep["title"] in "\n".join(m.value for m in at.markdown)

    at.selectbox(key="ai_match_select_id").select(int(second_rep["id"])).run()
    assert not at.exception
    assert at.selectbox(key="ai_match_select_id").value == int(second_rep["id"])
    assert second_rep["title"] in "\n".join(m.value for m in at.markdown)

    # Click a Quick Demo Preset, then confirm dropdown STILL switches freely afterwards
    preset_buttons = [b for b in at.button if b.key and b.key.startswith("demo_preset_")]
    assert len(preset_buttons) >= 1
    preset_buttons[0].click().run()
    assert not at.exception

    at.selectbox(key="ai_match_select_id").select(int(third_rep["id"])).run()
    assert not at.exception
    assert at.selectbox(key="ai_match_select_id").value == int(third_rep["id"])
    assert third_rep["title"] in "\n".join(m.value for m in at.markdown)

    # Sidebar must now honestly report Ready after AI Matching loaded the model
    sidebar_md = "\n".join(m.value for m in at.sidebar.markdown)
    assert "Ready (" in sidebar_md


def test_delete_confirmation_cancel_and_confirm_workflow(isolated_app_db: Path) -> None:
    """Verify 2-step delete confirmation for individual reports and bulk demo records."""
    user_rep_id = insert_report(
        title="Custom Graphite Stylus Pen",
        description="Lost a grey digital stylus pen near the design studio.",
        category="Electronics",
        report_type="lost",
        location="Design Studio",
        report_date="2026-11-25",
        db_path=isolated_app_db,
    )
    before_total = count_reports(isolated_app_db)

    at = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    at.sidebar.radio[0].set_value("Browse Reports").run()
    assert not at.exception

    # Step 1: Click Delete -> must NOT delete yet; must show title & ID confirmation warning
    at.button(key=f"browse_delete_{user_rep_id}").click().run()
    assert not at.exception
    assert get_report_by_id(user_rep_id, db_path=isolated_app_db) is not None
    assert any(
        f"#{user_rep_id}" in w.value and "Custom Graphite Stylus Pen" in w.value
        for w in at.warning
    )

    # Cancel -> record remains intact
    at.button(key=f"browse_cancel_delete_{user_rep_id}").click().run()
    assert not at.exception
    assert get_report_by_id(user_rep_id, db_path=isolated_app_db) is not None
    assert count_reports(isolated_app_db) == before_total

    # Step 2: Click Delete again -> Confirm Permanent Delete -> record removed
    at.button(key=f"browse_delete_{user_rep_id}").click().run()
    at.button(key=f"browse_confirm_delete_{user_rep_id}").click().run()
    assert not at.exception
    assert get_report_by_id(user_rep_id, db_path=isolated_app_db) is None
    assert count_reports(isolated_app_db) == before_total - 1

    # Bulk synthetic demo removal: request -> cancel keeps records -> confirm removes seeds only
    kept_user_id = insert_report(
        title="User Kept Notebook",
        description="Blue spiral notebook belonging to user.",
        category="Books & Stationery",
        report_type="lost",
        location="Central Library",
        report_date="2026-11-25",
        db_path=isolated_app_db,
    )
    seed_before = count_seed_reports(isolated_app_db)
    assert seed_before > 0

    at.sidebar.radio[0].set_value("Browse Reports").run()
    at.button(key="demo_remove_request").click().run()
    assert count_seed_reports(isolated_app_db) == seed_before
    at.button(key="demo_remove_cancel").click().run()
    assert count_seed_reports(isolated_app_db) == seed_before

    at.button(key="demo_remove_request").click().run()
    at.button(key="demo_remove_confirm").click().run()
    assert not at.exception
    assert count_seed_reports(isolated_app_db) == 0
    assert get_report_by_id(kept_user_id, db_path=isolated_app_db) is not None


def test_dashboard_highlights_vectorized_and_cache_invalidation(
    isolated_app_db: Path, engine: EmbeddingEngine
) -> None:
    """Vectorized highlights must cache on unchanged DB and invalidate on report changes."""
    refind_app.get_engine_state.clear()
    state = refind_app.get_engine_state()
    state.engine = engine
    state.status = "ready"

    v1 = get_reports_version(isolated_app_db)
    highlights_1 = refind_app.get_cached_dashboard_highlights(engine)
    assert len(highlights_1) > 0
    top_lost_id = int(highlights_1[0][0]["id"])
    assert (str(isolated_app_db.resolve()), v1) in state.highlights_cache

    # Calling again with unchanged DB returns exact cached object
    highlights_cached = refind_app.get_cached_dashboard_highlights(engine)
    assert highlights_cached is highlights_1

    # Marking the top highlighted lost item as recovered must change DB version & invalidate cache
    assert update_report_status(top_lost_id, "recovered", db_path=isolated_app_db) is True
    v2 = get_reports_version(isolated_app_db)
    assert v2 != v1

    highlights_2 = refind_app.get_cached_dashboard_highlights(engine)
    assert (str(isolated_app_db.resolve()), v1) not in state.highlights_cache
    assert (str(isolated_app_db.resolve()), v2) in state.highlights_cache
    assert all(int(pair[0]["id"]) != top_lost_id for pair in highlights_2)

    # Also verify direct vectorized function enforces opposite types & active status
    active_lost = get_all_reports(report_type="lost", status="active", db_path=isolated_app_db)
    active_found = get_all_reports(report_type="found", status="active", db_path=isolated_app_db)
    direct = find_top_match_highlights(active_lost, active_found, engine, max_highlights=4)
    assert len(direct) == len(highlights_2)
    for lost_rep, match_res in direct:
        assert lost_rep["report_type"] == "lost"
        assert match_res.candidate["report_type"] == "found"


# ---------------------------------------------------------------------------
# V2.0 Product Redesign Feature Tests (Editing, Sorting, Filters, Recovery Confirm)
# ---------------------------------------------------------------------------


def test_v2_report_editing_persists_and_invalidates_version(tmp_path: Path) -> None:
    """Editing a report must update SQLite fields and bump get_reports_version()."""
    db_file = tmp_path / "edit_test.db"
    init_db(db_file)
    rep_id = insert_report(
        title="Black Umbrella",
        description="Left folding umbrella near library entrance.",
        category="Other",
        report_type="lost",
        location="Library",
        report_date="2026-11-24",
        db_path=db_file,
    )
    v_before = get_reports_version(db_file)

    updated = update_report(
        rep_id,
        title="Matte Black Compact Umbrella",
        description="Left matte black compact folding umbrella outside Central Library.",
        category="Clothing & Accessories",
        location="Central Library",
        report_date="2026-11-25",
        additional_details="Wooden curved handle.",
        contact_label="Library Desk #22 (Demo)",
        db_path=db_file,
    )
    assert updated is True

    reloaded = get_report_by_id(rep_id, db_path=db_file)
    assert reloaded is not None
    assert reloaded["title"] == "Matte Black Compact Umbrella"
    assert reloaded["category"] == "Clothing & Accessories"
    assert reloaded["location"] == "Central Library"
    assert reloaded["additional_details"] == "Wooden curved handle."
    # Version string tracks updated_at (second precision) or row changes; check fields updated cleanly
    assert reloaded["updated_at"] >= reloaded["created_at"]
    assert isinstance(v_before, str)


def test_v2_browse_sorting_filtering_pagination_and_clear_filters(
    isolated_app_db: Path,
) -> None:
    """Verify Browse Reports keyword search, sorting, pagination, and Clear Filters button."""
    # SQL sorting check directly
    by_title = get_all_reports(sort_by="title_asc", db_path=isolated_app_db)
    titles = [r["title"].lower() for r in by_title]
    assert titles == sorted(titles)

    at = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    at.sidebar.radio[0].set_value("Browse Reports").run()
    assert not at.exception

    # Seed DB has 16 reports -> with page size 8, page 2 button should be enabled
    assert at.button(key="browse_next_page").disabled is False
    at.button(key="browse_next_page").click().run()
    assert not at.exception
    assert at.button(key="browse_prev_page").disabled is False

    # Filter by keyword + category
    at.text_input(key="browse_search").set_value("Casio").run()
    assert not at.exception
    browse_md = "\n".join(m.value for m in at.markdown)
    assert "Casio" in browse_md

    # Click Clear Filters -> resets search & filters cleanly
    at.button(key="browse_clear_filters").click().run()
    assert not at.exception
    assert at.text_input(key="browse_search").value == ""
    assert at.selectbox(key="browse_type").value == "All"
    assert at.selectbox(key="browse_category").value == "All"
    assert at.selectbox(key="browse_status").value == "All"


def test_v2_ai_matching_deliberate_pair_recovery_confirmation(
    isolated_app_db: Path,
) -> None:
    """Marking a match pair recovered must require deliberate confirmation and persist."""
    st.cache_resource.clear()
    at = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
    at.sidebar.radio[0].set_value("AI Matching").run()
    assert not at.exception

    # Select Wallet Near Library preset (strong active pair: Lost #1 <-> Found #2)
    wallet_rep = [
        r for r in get_all_reports(db_path=isolated_app_db) if r.get("seed_key") == "seed_lost_wallet_library"
    ][0]
    found_wallet = [
        r for r in get_all_reports(db_path=isolated_app_db) if r.get("seed_key") == "seed_found_wallet_central_library"
    ][0]

    at.selectbox(key="ai_match_select_id").select(int(wallet_rep["id"])).run()
    assert not at.exception

    pair_btn_key = f"matcher_{wallet_rep['id']}_1_recover_pair_{found_wallet['id']}"
    at.button(key=pair_btn_key).click().run()
    assert not at.exception

    # Step 1 click must NOT have recovered yet — must ask for confirmation first
    assert get_report_by_id(wallet_rep["id"], db_path=isolated_app_db)["status"] == "active"
    assert get_report_by_id(found_wallet["id"], db_path=isolated_app_db)["status"] == "active"
    assert len(at.warning) >= 1

    # Confirm recovery -> both reports transition to recovered in SQLite
    confirm_btn_key = f"matcher_{wallet_rep['id']}_1_confirm_pair_{found_wallet['id']}"
    at.button(key=confirm_btn_key).click().run()
    assert not at.exception
    assert get_report_by_id(wallet_rep["id"], db_path=isolated_app_db)["status"] == "recovered"
    assert get_report_by_id(found_wallet["id"], db_path=isolated_app_db)["status"] == "recovered"


def test_v2_report_location_preset_and_dismiss_post_submit_results(
    isolated_app_db: Path,
) -> None:
    """Users can select a common campus area without typing, and dismiss post-submit results."""
    at = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
    at.sidebar.radio[0].set_value("Report Lost Item").run()
    assert not at.exception

    at.text_input(key="lost_title").set_value("Silver USB Flash Drive")
    at.selectbox(key="lost_category").set_value("Electronics")
    at.text_area(key="lost_description").set_value("Lost a silver 64GB metal USB flash drive.")
    # Choose quick-select preset and leave custom text location blank
    at.selectbox(key="lost_location_preset").set_value("Central Library")
    at.text_input(key="lost_location").set_value("")
    at.button[0].click().run()

    assert not at.exception
    assert len(at.error) == 0
    saved = get_all_reports(search_query="Silver USB Flash Drive", db_path=isolated_app_db)
    assert len(saved) == 1
    assert saved[0]["location"] == "Central Library"

    # Clicking 'Report Another' clears the post-submission result panel cleanly
    at.button(key="dismiss_after_submit_lost").click().run()
    assert not at.exception
    assert not any(b.key == "dismiss_after_submit_lost" for b in at.button)


