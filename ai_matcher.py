"""Genuine Semantic NLP & Hybrid Contextual Matching Engine for ReFind AI.

Uses `sentence-transformers/all-MiniLM-L6-v2` to generate 384-dimensional dense
embeddings and Scikit-learn `cosine_similarity` to compare lost and found item
reports, combined with transparent category, location, and date signals.

Academic note:
    Scores output by this module (0-100) are relative weighted similarity scores,
    NOT statistical probabilities that two reports describe the identical object.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from utils import (
    clean_text,
    days_between_dates,
    normalize_location,
    tokenize_location,
)

DEFAULT_MODEL_NAME = os.getenv(
    "REFIND_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)


class ModelLoadError(RuntimeError):
    """Raised when the pretrained Sentence Transformer model cannot be loaded.

    ReFind AI intentionally surfaces this error rather than silently replacing
    semantic NLP with keyword matching.
    """


@dataclass(frozen=True)
class MatchConfig:
    """Configurable weights, decay parameter, and threshold for hybrid matching."""

    model_name: str = DEFAULT_MODEL_NAME
    weight_semantic: float = float(os.getenv("REFIND_WEIGHT_SEMANTIC", "0.70"))
    weight_category: float = float(os.getenv("REFIND_WEIGHT_CATEGORY", "0.15"))
    weight_location: float = float(os.getenv("REFIND_WEIGHT_LOCATION", "0.10"))
    weight_date: float = float(os.getenv("REFIND_WEIGHT_DATE", "0.05"))
    date_decay_days: float = 14.0
    min_score_threshold: float = float(os.getenv("REFIND_MIN_MATCH_SCORE", "35.0"))
    max_results: int = 10

    def __post_init__(self) -> None:
        weights = (
            self.weight_semantic,
            self.weight_category,
            self.weight_location,
            self.weight_date,
        )
        if any(w < 0.0 for w in weights):
            raise ValueError("All matching weights must be non-negative.")
        total = sum(weights)
        if not math.isclose(total, 1.0, abs_tol=1e-4):
            raise ValueError(
                f"Matching weights must sum to 1.0 (received {total:.4f})."
            )
        if self.date_decay_days <= 0.0:
            raise ValueError("date_decay_days must be greater than 0.")
        if not (0.0 <= self.min_score_threshold <= 100.0):
            raise ValueError("min_score_threshold must be between 0 and 100.")
        if self.max_results < 1:
            raise ValueError("max_results must be at least 1.")


@dataclass
class MatchResult:
    """Structured output for a single ranked candidate match."""

    candidate: dict[str, Any]
    final_score: float
    semantic_score: float
    raw_cosine_similarity: float
    category_score: float
    location_score: float
    date_score: float
    days_apart: Optional[int]
    location_match_reason: str
    shared_location_tokens: list[str]
    explanation: str
    meets_threshold: bool


class EmbeddingEngine:
    """Local CPU wrapper around `SentenceTransformer` with in-memory vector caching."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        self.model_name = model_name
        self._cache: dict[str, np.ndarray] = {}
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(model_name, device="cpu")
        except Exception as exc:
            raise ModelLoadError(
                f"Could not load Sentence Transformer model '{model_name}'. "
                "Verify that `sentence-transformers` is installed and that an "
                "internet connection is available for the initial model download. "
                f"Underlying error: {exc}"
            ) from exc

        get_dim = getattr(
            self.model,
            "get_embedding_dimension",
            getattr(self.model, "get_sentence_embedding_dimension", None),
        )
        dim = get_dim() if callable(get_dim) else None
        self.embedding_dim: int = int(dim) if dim is not None else 384

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Encode a sequence of strings into a 2D float32 numpy array (N, D).

        Empty strings receive a zero vector so cosine similarity evaluates to 0.0.
        """
        if not texts:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        cleaned = [clean_text(t) for t in texts]
        to_encode_indices: list[int] = []
        to_encode_texts: list[str] = []

        for idx, text in enumerate(cleaned):
            if text and text not in self._cache:
                to_encode_indices.append(idx)
                to_encode_texts.append(text)

        if to_encode_texts:
            # Encode unique uncached texts in one batch for CPU efficiency.
            unique_texts = list(dict.fromkeys(to_encode_texts))
            vectors = self.model.encode(
                unique_texts,
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            )
            vectors = np.asarray(vectors, dtype=np.float32)
            if vectors.ndim == 1:
                vectors = vectors.reshape(1, -1)
            for text, vec in zip(unique_texts, vectors):
                self._cache[text] = vec

        result = np.zeros((len(cleaned), self.embedding_dim), dtype=np.float32)
        for idx, text in enumerate(cleaned):
            if text:
                result[idx] = self._cache[text]
        return result


def build_item_text(report: dict[str, Any]) -> str:
    """Construct clean natural-language text from a report's title and description."""
    title = clean_text(report.get("title"))
    description = clean_text(report.get("description"))
    additional = clean_text(report.get("additional_details"))

    parts: list[str] = []
    if title:
        parts.append(title)
    if description:
        # Avoid repeating the exact title if description already starts identically.
        if not title or description.lower() != title.lower():
            parts.append(description)
    if additional:
        parts.append(additional)

    return ". ".join(parts)


def compute_semantic_similarity(
    vec_a: np.ndarray, vec_b: np.ndarray
) -> tuple[float, float]:
    """Return (bounded_score_0_1, raw_cosine_similarity) using Scikit-learn."""
    a = np.asarray(vec_a, dtype=np.float32).reshape(1, -1)
    b = np.asarray(vec_b, dtype=np.float32).reshape(1, -1)

    if np.linalg.norm(a) == 0.0 or np.linalg.norm(b) == 0.0:
        return 0.0, 0.0

    raw = float(cosine_similarity(a, b)[0, 0])
    bounded = float(np.clip(raw, 0.0, 1.0))
    return bounded, raw


def compute_category_score(
    category_a: Optional[str], category_b: Optional[str]
) -> float:
    """Return 1.0 when normalized categories match exactly, otherwise 0.0."""
    norm_a = clean_text(category_a).lower()
    norm_b = clean_text(category_b).lower()
    if not norm_a or not norm_b:
        return 0.0
    return 1.0 if norm_a == norm_b else 0.0


def compute_location_score(
    location_a: Optional[str], location_b: Optional[str]
) -> tuple[float, str, list[str]]:
    """Evaluate lexical location alignment transparently without inventing geography.

    Returns:
        (score in [0.0, 1.0], reason_tag, sorted_shared_tokens)
    """
    norm_a = normalize_location(location_a)
    norm_b = normalize_location(location_b)
    if not norm_a or not norm_b:
        return 0.0, "missing", []

    tokens_a = tokenize_location(location_a)
    tokens_b = tokenize_location(location_b)
    shared = sorted(tokens_a & tokens_b)

    if norm_a == norm_b:
        return 1.0, "exact", shared

    if (norm_a in norm_b) or (norm_b in norm_a):
        return 0.85, "substring", shared

    union = tokens_a | tokens_b
    if not union or not shared:
        return 0.0, "different", []

    jaccard = len(shared) / len(union)
    return float(np.clip(jaccard, 0.0, 1.0)), "token_overlap", shared


def compute_date_proximity_score(
    date_a: Optional[str], date_b: Optional[str], decay_days: float = 14.0
) -> tuple[float, Optional[int]]:
    """Compute exponential date-proximity score exp(-delta_days / decay_days)."""
    delta = days_between_dates(date_a, date_b)
    if delta is None:
        return 0.0, None
    score = math.exp(-float(delta) / float(decay_days))
    return float(np.clip(score, 0.0, 1.0)), delta


def compute_hybrid_score(
    *,
    semantic_score: float,
    category_score: float,
    location_score: float,
    date_score: float,
    config: MatchConfig = MatchConfig(),
) -> float:
    """Compute the normalized 0-100 hybrid matching score."""
    weighted = (
        config.weight_semantic * float(np.clip(semantic_score, 0.0, 1.0))
        + config.weight_category * float(np.clip(category_score, 0.0, 1.0))
        + config.weight_location * float(np.clip(location_score, 0.0, 1.0))
        + config.weight_date * float(np.clip(date_score, 0.0, 1.0))
    )
    return round(float(np.clip(weighted * 100.0, 0.0, 100.0)), 1)


def generate_match_explanation(
    *,
    source_report: dict[str, Any],
    candidate_report: dict[str, Any],
    semantic_score: float,
    category_score: float,
    location_score: float,
    location_reason: str,
    shared_location_tokens: list[str],
    days_apart: Optional[int],
) -> str:
    """Generate a concise, factual XAI explanation grounded strictly in computed signals."""
    sem_pct = round(semantic_score * 100)
    if semantic_score >= 0.70:
        sem_phrase = (
            f"High semantic similarity ({sem_pct}%): both descriptions align closely in meaning."
        )
    elif semantic_score >= 0.45:
        sem_phrase = (
            f"Moderate semantic similarity ({sem_pct}%): the item descriptions share related phrasing or attributes."
        )
    else:
        sem_phrase = (
            f"Low semantic similarity ({sem_pct}%): the item descriptions appear largely distinct."
        )

    cat_src = clean_text(source_report.get("category")) or "Uncategorized"
    cat_cand = clean_text(candidate_report.get("category")) or "Uncategorized"
    if category_score >= 1.0:
        cat_phrase = f"Both reports belong to the '{cat_src}' category."
    else:
        cat_phrase = f"Categories differ ('{cat_src}' vs. '{cat_cand}')."

    loc_src = clean_text(source_report.get("location"))
    loc_cand = clean_text(candidate_report.get("location"))
    if location_reason == "exact":
        loc_phrase = f"Reported locations match exactly ('{loc_cand}')."
    elif location_reason in ("substring", "token_overlap") and location_score > 0.0:
        if shared_location_tokens:
            terms = ", ".join(f"'{t}'" for t in shared_location_tokens[:3])
            loc_phrase = (
                f"Locations share overlapping terms ({terms}) across '{loc_src}' and '{loc_cand}'."
            )
        else:
            loc_phrase = f"Reported locations partially overlap ('{loc_src}' vs. '{loc_cand}')."
    elif location_reason == "missing":
        loc_phrase = "Location information is incomplete for one or both reports."
    else:
        loc_phrase = f"Reported locations do not share common terms ('{loc_src}' vs. '{loc_cand}')."

    if days_apart is None:
        date_phrase = "Report dates could not be compared."
    elif days_apart == 0:
        date_phrase = "Both items were reported on the same day."
    elif days_apart == 1:
        date_phrase = "The reports are 1 day apart."
    elif days_apart <= 7:
        date_phrase = f"The reports are {days_apart} days apart."
    else:
        date_phrase = f"The reports are {days_apart} days apart (wider time gap)."

    return f"{sem_phrase} {cat_phrase} {loc_phrase} {date_phrase}"


def find_matches(
    source_report: dict[str, Any],
    candidate_reports: Sequence[dict[str, Any]],
    engine: EmbeddingEngine,
    config: Optional[MatchConfig] = None,
    *,
    include_recovered: bool = False,
    apply_threshold: bool = True,
) -> list[MatchResult]:
    """Rank opposite-type candidate reports against a source report using hybrid AI scoring.

    Strictly enforces opposite report types:
        - 'lost' reports ONLY match 'found' reports.
        - 'found' reports ONLY match 'lost' reports.
    """
    cfg = config or MatchConfig()
    source_type = clean_text(source_report.get("report_type")).lower()
    if source_type not in ("lost", "found"):
        raise ValueError("source_report must specify report_type as 'lost' or 'found'.")

    opposite_type = "found" if source_type == "lost" else "lost"
    source_id = source_report.get("id")

    eligible: list[dict[str, Any]] = []
    for cand in candidate_reports:
        cand_type = clean_text(cand.get("report_type")).lower()
        if cand_type != opposite_type:
            continue
        if source_id is not None and cand.get("id") == source_id:
            continue
        if not include_recovered and clean_text(cand.get("status") or "active").lower() != "active":
            continue
        eligible.append(cand)

    if not eligible:
        return []

    source_text = build_item_text(source_report)
    candidate_texts = [build_item_text(c) for c in eligible]

    embeddings = engine.encode_texts([source_text, *candidate_texts])
    source_vec = embeddings[0:1]
    cand_vecs = embeddings[1:]

    # Compute full cosine similarity row vector using Scikit-learn.
    source_norm = float(np.linalg.norm(source_vec))
    cand_norms = np.linalg.norm(cand_vecs, axis=1)

    if source_norm == 0.0:
        raw_sims = np.zeros(len(eligible), dtype=np.float32)
    else:
        sim_matrix = cosine_similarity(source_vec, cand_vecs)[0]
        raw_sims = np.where(cand_norms > 0.0, sim_matrix, 0.0)

    results: list[MatchResult] = []
    for cand, raw_sim in zip(eligible, raw_sims):
        raw_val = float(raw_sim)
        sem_score = float(np.clip(raw_val, 0.0, 1.0))
        cat_score = compute_category_score(
            source_report.get("category"), cand.get("category")
        )
        loc_score, loc_reason, shared_tokens = compute_location_score(
            source_report.get("location"), cand.get("location")
        )
        date_score, delta_days = compute_date_proximity_score(
            source_report.get("report_date"),
            cand.get("report_date"),
            decay_days=cfg.date_decay_days,
        )
        final_score = compute_hybrid_score(
            semantic_score=sem_score,
            category_score=cat_score,
            location_score=loc_score,
            date_score=date_score,
            config=cfg,
        )
        explanation = generate_match_explanation(
            source_report=source_report,
            candidate_report=cand,
            semantic_score=sem_score,
            category_score=cat_score,
            location_score=loc_score,
            location_reason=loc_reason,
            shared_location_tokens=shared_tokens,
            days_apart=delta_days,
        )
        meets = final_score >= cfg.min_score_threshold
        if apply_threshold and not meets:
            continue

        results.append(
            MatchResult(
                candidate=cand,
                final_score=final_score,
                semantic_score=sem_score,
                raw_cosine_similarity=raw_val,
                category_score=cat_score,
                location_score=loc_score,
                date_score=date_score,
                days_apart=delta_days,
                location_match_reason=loc_reason,
                shared_location_tokens=shared_tokens,
                explanation=explanation,
                meets_threshold=meets,
            )
        )

    results.sort(key=lambda item: (item.final_score, item.semantic_score), reverse=True)
    return results[: cfg.max_results]
