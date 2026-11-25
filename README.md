# ReFind AI — Intelligent AI-Powered Lost & Found Management System

> **Lost something? Let AI help you find it.**

**ReFind AI** is a campus lost-and-found web platform powered by Natural Language Processing (NLP) and semantic similarity. Built for university and college campuses, it matches lost and found item reports even when users describe the same belonging in completely different words.

---

## Key Features

- **Genuine Semantic NLP Matching:** Uses the pretrained `sentence-transformers/all-MiniLM-L6-v2` transformer locally on CPU to convert natural-language descriptions into 384-dimensional embeddings and compare them with Scikit-learn cosine similarity.
- **Hybrid Contextual Scoring:** Combines semantic similarity (**70%**) with item category match (**15%**), normalized location overlap (**10%**), and exponential date proximity (**5%**) on a transparent `0–100` scale.
- **Explainable AI (XAI):** Every recommended match displays a component score breakdown and a factual, plain-English explanation grounded strictly in computed signals.
- **Modern Campus Dashboard:** Live metrics for Lost, Found, Recovered, and Active reports, plus quick actions and recent activity.
- **Persistent SQLite Storage:** Reports, recovery status updates, and local upload paths persist across restarts, preloaded with 16 idempotent synthetic demonstration records (14 active + 2 recovered).
- **Zero Paid APIs:** Runs 100% locally after the initial pretrained model download.

---

## Tech Stack

- **Frontend / App:** Streamlit (Python)
- **AI / NLP:** `sentence-transformers` (`all-MiniLM-L6-v2`)
- **Similarity:** `scikit-learn` (`cosine_similarity`)
- **Database:** SQLite (`sqlite3`)
- **Data & Imaging:** `pandas`, `numpy`, `Pillow`
- **Testing:** `pytest`

---

## Project Structure

```text
ReFind AI/
├── app.py                      # Main Streamlit web application
├── ai_matcher.py               # Transformer embeddings, cosine similarity, hybrid scoring & XAI
├── database.py                 # SQLite database initialization & parameterized CRUD
├── seed_data.py                # 16 realistic synthetic campus reports (idempotent)
├── utils.py                    # Shared categories, text/location/date & upload utilities
├── requirements.txt            # Python package dependencies
├── PROJECT_BLUEPRINT.md        # Comprehensive 22-section project documentation
├── .env.example                # Example configuration template
├── .gitignore                  # Excludes .venv, local DBs, uploads, caches, secrets
├── .streamlit/config.toml      # Clean SaaS visual theme
├── assets/styles.css           # Custom dashboard & card styling
├── data/                       # Local SQLite storage (ignored by Git)
├── uploads/                    # Local item photograph uploads (ignored by Git)
├── docs/
│   ├── AI_EXPLANATION.md       # Detailed NLP, scoring formula & XAI documentation
│   └── PROJECT_ARCHITECTURE.md # System design & module responsibilities
└── tests/
    └── test_ai_matcher.py      # Automated pytest test suite
```

---

## Local Setup & Running (Windows PowerShell)

### 1. Create a Virtual Environment

```powershell
python -m venv .venv
```

### 2. Activate the Virtual Environment

```powershell
.\.venv\Scripts\Activate.ps1
```

> **If PowerShell blocks script activation (`ExecutionPolicy` error):**  
> You do **not** need to change system-wide security settings. Either bypass for the current terminal session only:
> ```powershell
> Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
> .\.venv\Scripts\Activate.ps1
> ```
> Or run Python/Streamlit directly from `.venv` without activating:
> ```powershell
> .\.venv\Scripts\python.exe -m pip install -r requirements.txt
> .\.venv\Scripts\python.exe -m streamlit run app.py
> ```

### 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 4. Launch ReFind AI

```powershell
streamlit run app.py
```

The app will open in your browser (typically at `http://localhost:8501`). On the very first launch, Sentence Transformers will download `all-MiniLM-L6-v2` (~90 MB) once and cache it locally for fast subsequent runs.

---

## Running Automated Tests

Run the full `pytest` suite (verifying semantic matching, paraphrased wording, category isolation, opposite report-type filtering, hybrid weights, explanations, and SQLite persistence):

```powershell
pytest -v
```

---

## Important Academic Note: Similarity Score vs. Probability

A ReFind AI score (e.g., **86 / 100**) is a **relative weighted similarity score**, **not** a statistical probability that two reports describe the exact same physical item. It ranks candidate matches to support human verification. See [`docs/AI_EXPLANATION.md`](docs/AI_EXPLANATION.md) and [`PROJECT_BLUEPRINT.md`](PROJECT_BLUEPRINT.md) for the full technical explanation.

---

## Documentation Index

- **[PROJECT_BLUEPRINT.md](PROJECT_BLUEPRINT.md)** — Vision, mission, objectives, architecture, methodology, roadmap, limitations, and academic outcomes.
- **[docs/AI_EXPLANATION.md](docs/AI_EXPLANATION.md)** — Embedding pipeline, cosine similarity, hybrid formula, XAI rationale rules, and similarity vs. probability.
- **[docs/PROJECT_ARCHITECTURE.md](docs/PROJECT_ARCHITECTURE.md)** — Module breakdown, data flow, and engineering decisions.
