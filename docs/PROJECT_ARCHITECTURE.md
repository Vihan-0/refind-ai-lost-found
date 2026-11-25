# ReFind AI — Software Architecture Documentation

## 1. Architectural Overview

ReFind AI uses a clean, modular, single-process Python architecture optimized for rapid academic development, local CPU execution, zero paid external dependencies, and one-command startup (`streamlit run app.py`).

Instead of introducing unnecessary distributed complexity (separate REST APIs, React build chains, Docker containers, or cloud brokers), responsibilities are cleanly separated across focused Python modules.

---

## 2. Module Responsibilities

```text
ReFind AI/
├── app.py                   # Streamlit UI, navigation, caching, and user workflows
├── ai_matcher.py            # Sentence-Transformer embeddings, cosine similarity, hybrid scoring & XAI
├── database.py              # SQLite schema initialization and parameterized CRUD operations
├── seed_data.py             # Idempotent synthetic campus demonstration dataset (16 reports)
├── utils.py                 # Shared constants, text/location/date helpers, safe image storage
├── requirements.txt         # Pinned compatible Python dependencies
├── PROJECT_BLUEPRINT.md     # Master academic & product specification
├── README.md                # Setup, Windows PowerShell guide, usage, and testing instructions
├── .gitignore               # Excludes .venv, local DBs, private uploads, caches, and secrets
├── .env.example             # Documented environment variable template
├── .streamlit/
│   └── config.toml          # Cohesive SaaS theme & Streamlit server settings
├── assets/
│   └── styles.css           # Custom CSS for metric cards, report cards, badges, and hierarchy
├── data/                    # Runtime SQLite database directory (DB file gitignored)
├── uploads/                 # Runtime local image storage (uploaded photos gitignored)
├── docs/
│   ├── AI_EXPLANATION.md    # Deep dive into NLP, hybrid formula, XAI, and score vs. probability
│   └── PROJECT_ARCHITECTURE.md  # System architecture and data-flow documentation
└── tests/
    └── test_ai_matcher.py   # Automated pytest verification suite (semantic, hybrid, persistence)
```

---

## 3. Data Flow Architecture

### 3.1 Report Submission Flow
```text
User Form (app.py)
  │  1. Validate required fields & date
  │  2. Save optional photo via utils.save_uploaded_image() -> uploads/<uuid>.<ext>
  ▼
database.insert_report(...)
  │  3. Parameterized SQL INSERT into data/refind_ai.db
  ▼
ai_matcher.find_matches(...)
  │  4. Query opposite-type active candidates from SQLite
  │  5. Encode texts with cached all-MiniLM-L6-v2 (EmbeddingEngine)
  │  6. Compute cosine similarity + category + location + date scores
  │  7. Build MatchResult list with grounded explanations
  ▼
Streamlit UI renders ranked AI match cards immediately below confirmation
```

### 3.2 AI Matching Dashboard Flow
```text
Selected Active Report (app.py)
  ├──► Configurable MatchConfig (weights + minimum score threshold)
  ├──► database.get_all_reports(report_type=opposite, status="active")
  └──► ai_matcher.find_matches(source, candidates, cached_engine, config)
         └──► Returns sorted list[MatchResult] with:
                • final_score (0–100)
                • semantic_score, raw_cosine_similarity
                • category_score, location_score, date_score
                • days_apart, location_match_reason, shared_location_tokens
                • explanation (human-readable XAI rationale)
```

---

## 4. Key Engineering Decisions

| Decision | Choice Made | Rationale |
|---|---|---|
| **Application Framework** | Streamlit (Python) | Single-language full-stack execution; native caching (`@st.cache_resource`) ideal for ML models. |
| **Database** | SQLite (`sqlite3`) | Zero setup, ACID-compliant local persistence across restarts, ideal for campus MVP. |
| **Image Storage** | Local filesystem (`uploads/`) + relative DB path | Prevents database bloat from large binary BLOBs while keeping backup and rendering simple. |
| **Model Lifecycle** | Singleton via `@st.cache_resource` + text embedding cache | Loads `all-MiniLM-L6-v2` once at startup and avoids redundant inference on unchanged descriptions. |
| **Seed Idempotency** | Unique `seed_key` column + `is_seed` flag | Guarantees demo data is populated cleanly on first run and never duplicated across restarts. |
| **Security & Privacy** | Parameterized SQL + synthetic campus contact labels | Prevents SQL injection and avoids exposing private personal phone numbers or addresses. |
| **Failure Transparency** | Explicit `ModelLoadError` | Never degrades silently to keyword search if the transformer model fails to load. |

---

## 5. Persistence & Lifecycle

1. **First Launch:** `init_db()` creates `data/refind_ai.db` and the `reports` table with `CHECK` constraints and indexes on `report_type`, `status`, and `category`. `ensure_seed_data()` inserts 16 realistic synthetic records (14 active + 2 recovered) only when the database is empty (and uses `INSERT OR IGNORE` on `seed_key` as a second safety guard).
2. **Subsequent Launches:** Existing user reports, recovery statuses, and uploaded image references remain intact.
3. **Recovery Workflow:** Updating a report's status calls `update_report_status(report_id, "recovered")`, updating `updated_at` and immediately recalculating Dashboard KPIs.
4. **Demo Reset:** Administrators can optionally remove or restore synthetic seed records (`delete_seed_reports()` / `seed_database()`) without losing newly submitted user reports.
