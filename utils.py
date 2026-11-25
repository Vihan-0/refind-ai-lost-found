"""Shared utilities, constants, and helper functions for ReFind AI."""

from __future__ import annotations

import os
import re
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Optional, Union

from PIL import Image, UnidentifiedImageError

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = Path(os.getenv("REFIND_DB_PATH", str(BASE_DIR / "data" / "refind_ai.db")))
DEFAULT_UPLOAD_DIR = Path(os.getenv("REFIND_UPLOAD_DIR", str(BASE_DIR / "uploads")))

CATEGORIES: list[str] = [
    "Wallet & Cards",
    "Electronics",
    "Bottles & Drinkware",
    "Bags & Backpacks",
    "Keys & Access Cards",
    "Eyewear",
    "Books & Stationery",
    "ID & Documents",
    "Clothing & Accessories",
    "Other",
]

REPORT_TYPES: tuple[str, str] = ("lost", "found")
STATUS_OPTIONS: tuple[str, str] = ("active", "recovered")

CAMPUS_LOCATION_SUGGESTIONS: list[str] = [
    "Central Library",
    "Main Cafeteria",
    "Computer Science Lab",
    "Lecture Hall B",
    "Room 204 - Academic Block",
    "Main Auditorium",
    "Student Union Building",
    "Science Block Corridor",
    "Sports Complex",
    "University Main Gate",
]

ALLOWED_IMAGE_EXTENSIONS: set[str] = {".png", ".jpg", ".jpeg", ".webp"}

# Generic spatial prepositions/articles ignored when comparing normalized location tokens.
LOCATION_STOPWORDS: set[str] = {
    "a",
    "an",
    "the",
    "in",
    "on",
    "at",
    "by",
    "to",
    "of",
    "for",
    "from",
    "near",
    "outside",
    "inside",
    "around",
    "beside",
    "behind",
    "next",
    "front",
    "area",
    "side",
    "section",
    "spot",
}


def clean_text(text: Optional[str]) -> str:
    """Trim and collapse consecutive whitespace in user-supplied text."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def normalize_location(location: Optional[str]) -> str:
    """Normalize a campus location string to lowercase alphanumeric words."""
    if not location:
        return ""
    lowered = str(location).lower()
    cleaned = re.sub(r"[^a-z0-9\s]", " ", lowered)
    return re.sub(r"\s+", " ", cleaned).strip()


def tokenize_location(location: Optional[str]) -> set[str]:
    """Extract meaningful normalized location tokens for lexical comparison."""
    normalized = normalize_location(location)
    if not normalized:
        return set()
    all_tokens = normalized.split()
    meaningful = {tok for tok in all_tokens if tok not in LOCATION_STOPWORDS}
    return meaningful if meaningful else set(all_tokens)


def parse_iso_date(value: Optional[Union[str, date, datetime]]) -> Optional[date]:
    """Parse an ISO YYYY-MM-DD string, date, or datetime into a date object."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = clean_text(str(value))
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def days_between_dates(
    date_a: Optional[Union[str, date, datetime]],
    date_b: Optional[Union[str, date, datetime]],
) -> Optional[int]:
    """Return absolute calendar days between two dates, or None if either is invalid."""
    parsed_a = parse_iso_date(date_a)
    parsed_b = parse_iso_date(date_b)
    if parsed_a is None or parsed_b is None:
        return None
    return abs((parsed_a - parsed_b).days)


def format_display_date(date_str: Optional[str]) -> str:
    """Format an ISO date string into a clean human-readable date."""
    parsed = parse_iso_date(date_str)
    if parsed is None:
        return "Unknown date"
    return parsed.strftime("%b %d, %Y")


def score_tier(score: float) -> tuple[str, str]:
    """Classify a 0-100 hybrid similarity score into a human-readable tier and CSS tag."""
    if score >= 75.0:
        return "Strong Match", "badge-strong"
    if score >= 55.0:
        return "Potential Match", "badge-moderate"
    if score >= 35.0:
        return "Possible Match", "badge-possible"
    return "Low Similarity", "badge-low"


def save_uploaded_image(uploaded_file, upload_dir: Path = DEFAULT_UPLOAD_DIR) -> Optional[str]:
    """Validate and save a Streamlit UploadedFile locally, returning its relative path.

    Images are stored on disk under `uploads/` rather than as SQLite BLOBs.
    """
    if uploaded_file is None:
        return None

    original_name = getattr(uploaded_file, "name", "") or ""
    ext = Path(original_name).suffix.lower()
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        raise ValueError(
            f"Unsupported image format '{ext or 'unknown'}'. "
            f"Please upload PNG, JPG, JPEG, or WEBP."
        )

    raw_bytes = uploaded_file.getvalue()
    if not raw_bytes:
        raise ValueError("Uploaded image file is empty.")

    # Verify image integrity using Pillow before writing to disk.
    from io import BytesIO

    try:
        with Image.open(BytesIO(raw_bytes)) as img:
            img.verify()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("The uploaded file is not a valid image.") from exc

    target_dir = Path(upload_dir)
    if not target_dir.is_absolute():
        target_dir = BASE_DIR / target_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{uuid.uuid4().hex}{ext}"
    destination = target_dir / filename
    destination.write_bytes(raw_bytes)

    try:
        rel_path = destination.relative_to(BASE_DIR).as_posix()
    except ValueError:
        rel_path = destination.as_posix()
    return rel_path


def resolve_image_path(stored_path: Optional[str]) -> Optional[Path]:
    """Resolve a stored relative image path safely and verify the file exists."""
    if not stored_path:
        return None
    candidate = Path(stored_path)
    if not candidate.is_absolute():
        candidate = BASE_DIR / candidate
    try:
        resolved = candidate.resolve()
    except OSError:
        return None
    if resolved.exists() and resolved.is_file():
        return resolved
    return None


def resolve_safe_upload_path(
    stored_path: Optional[str], upload_dir: Path = DEFAULT_UPLOAD_DIR
) -> Optional[Path]:
    """Resolve stored_path ONLY if it strictly resides inside the designated uploads directory.

    Protects against path traversal (e.g., '../app.py'), arbitrary absolute paths,
    symlinks escaping the upload directory, and deleting '.gitkeep'.
    Returns the resolved Path if safe, or None if untrusted/empty.
    """
    if not stored_path or not str(stored_path).strip():
        return None

    root = Path(upload_dir)
    if not root.is_absolute():
        root = BASE_DIR / root
    try:
        allowed_root = root.resolve()
    except OSError:
        return None

    candidate = Path(stored_path)
    if not candidate.is_absolute():
        candidate = BASE_DIR / candidate
    try:
        resolved = candidate.resolve()
    except OSError:
        return None

    if resolved == allowed_root or resolved.name == ".gitkeep":
        return None
    if not resolved.is_relative_to(allowed_root):
        return None
    return resolved


def delete_uploaded_image(
    stored_path: Optional[str], upload_dir: Path = DEFAULT_UPLOAD_DIR
) -> tuple[bool, Optional[str]]:
    """Safely delete an uploaded image file strictly inside upload_dir.

    Returns:
        (deleted: bool, warning_message: Optional[str])
        - (False, None): no image path stored or file already absent.
        - (True, None): image file safely validated and deleted.
        - (False, msg): path rejected as unsafe or filesystem deletion failed.
    """
    if not stored_path or not str(stored_path).strip():
        return False, None

    safe_path = resolve_safe_upload_path(stored_path, upload_dir=upload_dir)
    if safe_path is None:
        return (
            False,
            "Skipped image cleanup because the stored image path is outside the allowed uploads directory.",
        )

    if not safe_path.exists():
        return False, None
    if not safe_path.is_file():
        return False, "Skipped image cleanup because the target path is not a regular file."

    try:
        safe_path.unlink()
        return True, None
    except OSError as exc:
        return False, f"Could not delete associated image file: {exc}"
