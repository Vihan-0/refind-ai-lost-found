"""Automated verification suite for ReFind AI matching engine and SQLite persistence.

Covers all required academic validation scenarios:
1. Strong semantic match (wallet near library)
2. Different wording / paraphrased description (blue water bottle vs. navy drinking bottle)
3. Different categories & unrelated items (wallet vs. wireless headphones)
4. Same category, different item contrast (red backpack vs. blue school bag)
5. Strict opposite report-type enforcement (no lost↔lost or found↔found matches)
6. SQLite persistence across reopened connections & idempotent seeding
Plus contextual scoring unit tests, weight validation, edge cases, and XAI checks.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

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
    get_dashboard_stats,
    get_report_by_id,
    init_db,
    insert_report,
    update_report_status,
)
from seed_data import seed_database


@pytest.fixture(scope="module")
def engine() -> EmbeddingEngine:
    """Load the genuine pretrained Sentence Transformer model once for the test module."""
    return EmbeddingEngine(model_name=DEFAULT_MODEL_NAME)


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
    # Bottle must rank #1 over unrelated item and show clear semantic recognition.
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
    # Both share the same category (15% weight), yet the closer description + context wins clearly.
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

    # First seed inserts records; second seed must insert zero duplicates.
    first_inserted = seed_database(db_file)
    assert first_inserted >= 12
    second_inserted = seed_database(db_file)
    assert second_inserted == 0
    initial_total = count_reports(db_file)

    # Insert a brand-new user report and simulate reopening the app via fresh DB queries.
    new_id = insert_report(
        title="Green Mechanical Pencil Case",
        description="Lost a dark green zippered canvas pencil pouch in Lecture Hall B.",
        category="Books & Stationery",
        report_type="lost",
        location="Lecture Hall B",
        report_date="2026-11-25",
        db_path=db_file,
    )

    reloaded = get_report_by_id(new_id, db_path=db_file)
    assert reloaded is not None
    assert reloaded["title"] == "Green Mechanical Pencil Case"
    assert reloaded["status"] == "active"
    assert count_reports(db_file) == initial_total + 1

    # Mark recovered and verify dashboard stats update cleanly.
    stats_before = get_dashboard_stats(db_file)
    assert update_report_status(new_id, "recovered", db_path=db_file) is True
    stats_after = get_dashboard_stats(db_file)
    assert stats_after["total_recovered"] == stats_before["total_recovered"] + 1
    assert stats_after["active_reports"] == stats_before["active_reports"] - 1


# ---------------------------------------------------------------------------
# Unit Tests for Contextual Signals, Formula Math, Edge Cases & Validation
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

    # Exact weighted formula check: 0.70*0.8 + 0.15*1.0 + 0.10*0.5 + 0.05*1.0 = 0.81 -> 81.0
    score = compute_hybrid_score(
        semantic_score=0.80,
        category_score=1.0,
        location_score=0.50,
        date_score=1.0,
    )
    assert score == 81.0


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
