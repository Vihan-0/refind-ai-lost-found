"""Idempotent synthetic demonstration dataset for ReFind AI.

Provides 16 realistic campus lost-and-found reports (14 active + 2 recovered) spanning multiple categories:
- Strong semantic matches with varied wording (wallet, water bottle, earbuds,
  scientific calculator, prescription eyeglasses).
- Same-category / different-attribute contrast (red backpack vs. blue school bag).
- Distinct non-matching items (bike lock key, linear algebra textbook, USB-C hub).
- One already-recovered pair so dashboard recovery metrics display immediately.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Optional

from database import PathLike, count_reports, get_meta, init_db, insert_seed_report, set_meta


def _days_ago(days: int) -> str:
    return (date.today() - timedelta(days=days)).isoformat()


def get_seed_records() -> list[dict[str, Any]]:
    """Return synthetic campus reports with unique seed_key identifiers."""
    return [
        # --- PAIR 1: Strong semantic match (Wallet near library) ---
        {
            "seed_key": "seed_lost_wallet_library",
            "title": "Dark Brown Leather Wallet",
            "description": (
                "I lost my dark brown leather wallet near the library. "
                "It is a slim bifold wallet containing a few transit cards and receipts."
            ),
            "category": "Wallet & Cards",
            "report_type": "lost",
            "location": "Near the Library",
            "report_date": _days_ago(2),
            "additional_details": "Subtle stitched border and small brass button clasp.",
            "contact_label": "Student Desk Ref #L-101 (Demo)",
            "status": "active",
        },
        {
            "seed_key": "seed_found_wallet_central_library",
            "title": "Brown Leather Purse-Style Wallet",
            "description": (
                "Found a brown leather purse-like wallet on a bench outside the central library. "
                "Slim folding design with card slots inside."
            ),
            "category": "Wallet & Cards",
            "report_type": "found",
            "location": "Outside Central Library",
            "report_date": _days_ago(1),
            "additional_details": "Turned in to Central Library security desk.",
            "contact_label": "Library Security Desk #F-201 (Demo)",
            "status": "active",
        },
        # --- PAIR 2: Paraphrased wording match (Water bottle in classroom) ---
        {
            "seed_key": "seed_lost_blue_water_bottle",
            "title": "Blue Insulated Water Bottle",
            "description": (
                "My blue water bottle is missing from the classroom after morning lecture. "
                "Matte metal flask with a black screw-on cap."
            ),
            "category": "Bottles & Drinkware",
            "report_type": "lost",
            "location": "Academic Block Classroom 204",
            "report_date": _days_ago(3),
            "additional_details": "Has a small dent near the base.",
            "contact_label": "Student Desk Ref #L-102 (Demo)",
            "status": "active",
        },
        {
            "seed_key": "seed_found_navy_drinking_bottle",
            "title": "Navy Reusable Drinking Bottle",
            "description": (
                "Found a navy reusable drinking bottle left under a desk in Room 204. "
                "Dark blue stainless steel flask with black lid."
            ),
            "category": "Bottles & Drinkware",
            "report_type": "found",
            "location": "Room 204",
            "report_date": _days_ago(2),
            "additional_details": "Slightly scratched bottom edge.",
            "contact_label": "Academic Block Office #F-202 (Demo)",
            "status": "active",
        },
        # --- PAIR 3: Paraphrased electronics match (Wireless earbuds at cafeteria) ---
        {
            "seed_key": "seed_lost_wireless_earbuds",
            "title": "White Wireless Earbuds in Charging Case",
            "description": (
                "Misplaced my white Bluetooth earbuds inside a compact oval charging case "
                "while having lunch at the cafeteria."
            ),
            "category": "Electronics",
            "report_type": "lost",
            "location": "Main Cafeteria",
            "report_date": _days_ago(1),
            "additional_details": "Small green LED indicator on the front of the case.",
            "contact_label": "Student Desk Ref #L-103 (Demo)",
            "status": "active",
        },
        {
            "seed_key": "seed_found_bluetooth_earphones",
            "title": "White Bluetooth Earphones with Pod Case",
            "description": (
                "Picked up a pair of white wireless earphones in a small rounded charging pod "
                "left on a dining table near the cafeteria counter."
            ),
            "category": "Electronics",
            "report_type": "found",
            "location": "Cafeteria Dining Area",
            "report_date": _days_ago(1),
            "additional_details": "Both left and right buds are inside the case.",
            "contact_label": "Cafeteria Counter Desk #F-203 (Demo)",
            "status": "active",
        },
        # --- PAIR 4: Academic equipment match (Scientific calculator in CS Lab) ---
        {
            "seed_key": "seed_lost_scientific_calculator",
            "title": "Casio Scientific Calculator",
            "description": (
                "Left my black and silver Casio scientific engineering calculator at a workstation "
                "in the Computer Science Lab after the statistics exam."
            ),
            "category": "Electronics",
            "report_type": "lost",
            "location": "Computer Science Lab",
            "report_date": _days_ago(4),
            "additional_details": "Slide-on protective grey cover included.",
            "contact_label": "Student Desk Ref #L-104 (Demo)",
            "status": "active",
        },
        {
            "seed_key": "seed_found_engineering_calculator",
            "title": "Black Casio Engineering Calculator",
            "description": (
                "Found a black Casio FX-991EX mathematical calculator with hard slide cover "
                "next to terminal 14 inside the Computer Science Lab."
            ),
            "category": "Electronics",
            "report_type": "found",
            "location": "Computer Science Lab",
            "report_date": _days_ago(4),
            "additional_details": "Stored with the CS Lab assistant.",
            "contact_label": "CS Lab Coordinator #F-204 (Demo)",
            "status": "active",
        },
        # --- PAIR 5: Eyewear match (Spectacles at auditorium) ---
        {
            "seed_key": "seed_lost_eyeglasses_auditorium",
            "title": "Black Frame Prescription Eyeglasses",
            "description": (
                "Lost my black rectangular prescription eyeglasses inside a matte grey hard case "
                "during the guest seminar at the main auditorium."
            ),
            "category": "Eyewear",
            "report_type": "lost",
            "location": "Main Auditorium",
            "report_date": _days_ago(5),
            "additional_details": "Microfiber cleaning cloth inside the case.",
            "contact_label": "Student Desk Ref #L-105 (Demo)",
            "status": "active",
        },
        {
            "seed_key": "seed_found_spectacles_auditorium",
            "title": "Dark Rimmed Spectacles in Protective Box",
            "description": (
                "Found dark black-framed reading glasses stored in a rigid grey spectacle case "
                "in the middle row seats of the auditorium."
            ),
            "category": "Eyewear",
            "report_type": "found",
            "location": "Main Auditorium",
            "report_date": _days_ago(4),
            "additional_details": "Handed over to the auditorium front desk.",
            "contact_label": "Auditorium Desk #F-205 (Demo)",
            "status": "active",
        },
        # --- CONTRAST PAIR: Same category, different color/item ---
        {
            "seed_key": "seed_lost_red_backpack",
            "title": "Red Canvas Laptop Backpack",
            "description": (
                "Lost a bright red canvas backpack with padded laptop compartment and two side pockets "
                "near the student union lounge."
            ),
            "category": "Bags & Backpacks",
            "report_type": "lost",
            "location": "Student Union Building",
            "report_date": _days_ago(6),
            "additional_details": "Silver zipper pulls, no keychain attached.",
            "contact_label": "Student Desk Ref #L-106 (Demo)",
            "status": "active",
        },
        {
            "seed_key": "seed_found_blue_school_bag",
            "title": "Blue Nylon School Bag",
            "description": (
                "Found an empty light blue nylon school daypack with drawstring front pocket "
                "on the bleachers at the sports complex."
            ),
            "category": "Bags & Backpacks",
            "report_type": "found",
            "location": "Sports Complex",
            "report_date": _days_ago(11),
            "additional_details": "Lightweight gym-style bag.",
            "contact_label": "Sports Complex Office #F-206 (Demo)",
            "status": "active",
        },
        # --- UNMATCHED NEGATIVE CONTROLS ---
        {
            "seed_key": "seed_lost_linear_algebra_book",
            "title": "Hardcover Linear Algebra Textbook",
            "description": (
                "Left my hardcover Introduction to Linear Algebra textbook with yellow sticky tabs "
                "in the science block study corridor."
            ),
            "category": "Books & Stationery",
            "report_type": "lost",
            "location": "Science Block Corridor",
            "report_date": _days_ago(7),
            "additional_details": "Notes written in pencil on chapter 4.",
            "contact_label": "Student Desk Ref #L-107 (Demo)",
            "status": "active",
        },
        {
            "seed_key": "seed_found_bicycle_key_fob",
            "title": "Silver Bicycle U-Lock Key with Green Tag",
            "description": (
                "Found a single silver metallic bicycle U-lock key attached to a neon green plastic tag "
                "near the main campus entrance bike rack."
            ),
            "category": "Keys & Access Cards",
            "report_type": "found",
            "location": "University Main Gate",
            "report_date": _days_ago(3),
            "additional_details": "Held at Main Gate security booth.",
            "contact_label": "Main Gate Security #F-207 (Demo)",
            "status": "active",
        },
        # --- RECOVERED DEMO PAIR (for realistic dashboard KPIs) ---
        {
            "seed_key": "seed_recovered_usbc_hub_lost",
            "title": "Grey Aluminum USB-C Multiport Adapter",
            "description": (
                "Lost a compact space-grey aluminum USB-C hub with HDMI and two USB-A ports "
                "in Lecture Hall B."
            ),
            "category": "Electronics",
            "report_type": "lost",
            "location": "Lecture Hall B",
            "report_date": _days_ago(9),
            "additional_details": "Successfully verified and returned to owner.",
            "contact_label": "Campus Lost & Found Desk (Demo)",
            "status": "recovered",
        },
        {
            "seed_key": "seed_recovered_usbc_hub_found",
            "title": "Silver-Grey USB-C HDMI Dongle",
            "description": (
                "Found a metallic grey USB-C to HDMI multiport dongle near the projector podium "
                "in Lecture Hall B."
            ),
            "category": "Electronics",
            "report_type": "found",
            "location": "Lecture Hall B",
            "report_date": _days_ago(9),
            "additional_details": "Matched and reunited with student owner.",
            "contact_label": "Campus Lost & Found Desk (Demo)",
            "status": "recovered",
        },
    ]


def seed_database(db_path: Optional[PathLike] = None) -> int:
    """Insert all missing synthetic demo records idempotently. Returns count inserted."""
    init_db(db_path)
    inserted = 0
    for record in get_seed_records():
        if insert_seed_report(record, db_path=db_path):
            inserted += 1
    set_meta("initial_seed_completed", "1", db_path=db_path)
    return inserted


def ensure_seed_data(db_path: Optional[PathLike] = None) -> int:
    """Populate demo records only once on first-ever database initialization.

    If a user later removes synthetic demo records intentionally, this flag
    prevents the demo data from unexpectedly reappearing on the next rerun.
    """
    init_db(db_path)
    if get_meta("initial_seed_completed", db_path=db_path) == "1":
        return 0
    if count_reports(db_path) > 0:
        set_meta("initial_seed_completed", "1", db_path=db_path)
        return 0
    return seed_database(db_path)


if __name__ == "__main__":
    count = seed_database()
    print(f"Seeded {count} new demonstration reports into ReFind AI.")
