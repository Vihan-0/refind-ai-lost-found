# PROJECT BLUEPRINT: ReFind AI

**Tagline:** *Lost something? Let AI help you find it.*  
**Category:** Artificial Intelligence / Natural Language Processing / Web Application  
**Target Environment:** University and College Campuses  
**Academic Context:** B.Sc. Artificial Intelligence End-Semester Project

---

## 1. Project Overview

**ReFind AI** is an intelligent digital lost-and-found management platform engineered for university and college campuses. Rather than relying on physical logbooks, scattered messaging groups, or fragile keyword searches, ReFind AI applies modern Natural Language Processing (NLP) and dense sentence embeddings to match natural-language descriptions of lost and found belongings.

Users submit structured and free-text reports for either **Lost** or **Found** items. ReFind AI encodes each report into a 384-dimensional semantic vector using the pretrained transformer model `sentence-transformers/all-MiniLM-L6-v2`, compares it against opposite-type reports stored in a local SQLite database using cosine similarity, and fuses semantic similarity with three transparent contextual signals (category, location, and date proximity). The system presents ranked candidate matches accompanied by plain-language, evidence-grounded explanations so users and campus staff can quickly verify and reunite items with their owners.

---

## 2. Problem Statement

Conventional campus lost-and-found workflows depend on paper registers at security desks, notice boards, fragmented WhatsApp/Telegram groups, or basic substring search boxes. These approaches fail systematically for five reasons:

1. **Vocabulary Mismatch:** Two people describing the same object rarely use identical words. A student might write *"dark brown leather wallet near the library"*, while the finder logs *"brown leather purse-like wallet outside the central library."* Exact keyword search misses or under-ranks the connection.
2. **Location Imprecision:** A person who lost an item often remembers only a general area (e.g., *"near the library"* or *"second floor classroom"*), whereas the finder records a more specific room or landmark (*"central library entrance"* or *"Room 204"*).
3. **Unstructured Fragmentation:** Lost reports and found reports live in separate lists or chat threads and are rarely cross-referenced systematically.
4. **High Manual Effort:** Campus administrators or students must manually skim dozens of unstructured posts to spot potential matches.
5. **Lack of Transparency and Persistence:** Ad-hoc chat messages scroll out of view quickly, lack status tracking (active vs. recovered), and provide no ranked prioritization.

---

## 3. Vision Statement

To create an intelligent, accessible, and efficient lost-and-found ecosystem that uses artificial intelligence to help people recover their belongings through meaningful connections between lost and found reports.

---

## 4. Mission Statement

To simplify the process of reporting and recovering lost belongings by developing an AI-driven platform capable of understanding natural-language descriptions, identifying semantic similarities, and recommending potential matches with explainable results.

---

## 5. Project Objectives

- Develop an intuitive web platform for reporting lost and found campus belongings.
- Implement genuine NLP-based semantic matching using a pretrained Sentence Transformer model running locally on CPU.
- Identify relevant matches even when lost and found descriptions use different phrasing or synonyms.
- Combine semantic similarity with explicit, transparent contextual features (category, location, and date proximity) using a configurable hybrid scoring formula.
- Present ranked match recommendations with clear component score breakdowns.
- Generate explainable AI rationales grounded strictly in computed signals—never claiming unverified evidence.
- Persist all reports, image paths, and recovery statuses reliably in a local SQLite database.
- Demonstrate how applied artificial intelligence solves a practical, everyday campus problem cleanly and reproducibly.

---

## 6. Target Audience

| Audience Segment | Primary Needs | How ReFind AI Helps |
|---|---|---|
| **University Students** | Fast reporting of misplaced belongings (wallets, bottles, earphones, calculators, IDs) and instant feedback on whether someone found them. | Simple reporting form + immediate ranked AI match suggestions. |
| **Faculty & Staff** | Quick way to log items found in classrooms, labs, and offices without spending time searching old lists. | Structured Found-Item form + automatic matching against active Lost reports. |
| **Campus Lost-and-Found / Security Desk** | Centralized, searchable inventory with status tracking and clear match prioritization. | Unified dashboard, filterable report browser, and one-click recovery status updates. |
| **Academic Evaluators** | Verifiable, genuine AI engineering with clear methodology, reproducible tests, and architectural rigor. | Modular Python codebase, pretrained transformer embeddings, transparent scoring, and automated test suite. |

---

## 7. Proposed Solution

ReFind AI replaces manual comparison and naive keyword lookup with a lightweight, locally runnable, full-stack Python web application built on **Streamlit**, **Sentence Transformers**, **Scikit-learn**, and **SQLite**:

1. **Structured + Free-Text Intake:** Users submit item title, natural-language description, category, campus location, date, optional identifying notes, and an optional photograph.
2. **Semantic Encoding:** The `all-MiniLM-L6-v2` transformer maps natural-language item text into dense numerical embeddings where semantically similar descriptions sit close together in vector space.
3. **Hybrid Contextual Ranking:** Opposite-type reports (Lost ↔ Found) are scored via cosine similarity combined with category match, normalized location token overlap, and exponential date-proximity decay.
4. **Explainable Recommendations:** Every candidate match displays its 0–100 hybrid score, per-component breakdown, and a factual explanation summarizing why the algorithm ranked it.
5. **End-to-End Lifecycle:** Items remain persisted in SQLite across restarts and can be marked as **Recovered**, automatically updating campus KPI metrics.

---

## 8. Core Features

1. **Executive Campus Dashboard:** Live metrics for Total Lost, Total Found, Recovered Items, and Active Reports, paired with quick-action workflows, recent reports, and top AI match highlights.
2. **Report Lost Item Workflow:** Clean validated form with optional local photo upload, immediate database persistence, and automatic AI matching against existing Found items.
3. **Report Found Item Workflow:** Parallel validated form for finders that immediately checks active Lost reports for potential owners.
4. **AI Matching Dashboard:** Dedicated matching workspace allowing users to select any active report, inspect ranked matches, adjust hybrid weights/threshold for experimentation, view component score charts, read Explainable AI rationales, and complete a demonstration recovery workflow.
5. **Browse & Filter Reports:** Full catalog of lost and found reports with keyword search, category filter, report-type filter (`Lost` / `Found`), status filter (`Active` / `Recovered`), and photo previews.
6. **Recovery Management:** Mark any active item (or matched Lost + Found pair) as recovered—or reopen if needed—with immediate KPI updates and no exposure of private personal PII.
7. **Idempotent Synthetic Seed Data:** Preloaded with 16 realistic campus reports (14 active + 2 recovered, covering strong matches, paraphrased matches, partial matches, unrelated items, and recovered records) so the AI matcher works immediately upon first launch without duplicating data across restarts.

---

## 9. AI/ML Methodology

### 9.1 Why Semantic Embeddings Instead of Keyword Matching?

Traditional keyword matching (e.g., SQL `LIKE`, TF-IDF bag-of-words) treats words as independent discrete tokens. Consequently, *"blue water bottle"* and *"navy reusable drinking flask"* share almost no tokens despite describing the same object.

**Sentence Transformers** (Siamese/bi-encoder BERT architectures fine-tuned with contrastive learning on massive sentence-pair corpora) project entire sentences into a shared continuous semantic space ($\mathbb{R}^{384}$). Synonyms, paraphrases, and related concepts map to nearby vectors, enabling accurate comparison via **cosine similarity**.

### 9.2 Selected Model: `sentence-transformers/all-MiniLM-L6-v2`

| Property | Specification |
|---|---|
| Architecture | 6-layer MiniLM Transformer (distilled from BERT-style models) |
| Embedding Dimension | 384 floats per text input |
| Max Sequence Length | 256 word pieces |
| Model Footprint | ~80–90 MB (runs smoothly on standard laptop CPU) |
| License & Cost | Apache 2.0 / Free, 100% local after initial Hugging Face download |
| Similarity Metric | Cosine Similarity (via Scikit-learn) |

### 9.3 Important Academic Distinction: Similarity Score vs. Probability

> **Critical Evaluation Note:** A ReFind AI matching score (for example, **89 / 100**) is a **normalized weighted similarity index**, **NOT** a statistical probability.
>
> - A score of **89%** means the algorithm computed a high relative match based on semantic vector closeness (70%) plus contextual alignment (30%).
> - It does **not** mean there is an 89% mathematical probability that the two reports refer to the exact same physical object.
> - True probability estimation would require calibrated posterior modeling over labeled historical ground-truth match outcomes ($P(\text{same item} \mid x_1, x_2)$), which is outside the scope of an unlabeled campus MVP.
> - Therefore, ReFind AI frames all outputs as **ranked decision-support recommendations** requiring human verification.

---

## 10. Technical Architecture

ReFind AI follows a clean, single-process layered architecture designed for reliability, clarity, and one-command local execution:

```text
┌──────────────────────────────────────────────────────────────────────┐
│                   Presentation Layer (Streamlit UI)                  │
│          app.py  +  assets/styles.css  +  .streamlit/config.toml     │
│  [Dashboard] [AI Matching] [Report Lost] [Report Found] [Browse]     │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
          ┌─────────────────────┴──────────────────────┐
          ▼                                            ▼
┌────────────────────────────────────┐  ┌──────────────────────────────┐
│        AI Matching Engine          │  │    Data Persistence Layer    │
│           ai_matcher.py            │  │         database.py          │
│  • Text preprocessing              │  │  • Parameterized SQLite CRUD │
│  • all-MiniLM-L6-v2 embeddings     │  │  • Status & KPI queries      │
│  • Scikit-learn cosine similarity  │  │  • Idempotent seeding        │
│  • Hybrid contextual scoring       │  │    (seed_data.py)            │
│  • Explainable AI rationale gen    │  │                              │
└─────────────────┬──────────────────┘  └──────────────┬───────────────┘
                  │                                    │
                  ▼                                    ▼
┌────────────────────────────────────┐  ┌──────────────────────────────┐
│     Local Model Cache (CPU)        │  │  SQLite DB & Local Storage   │
│  Hugging Face / SentenceTransf.    │  │  • data/refind_ai.db         │
│  (cached once via st.cache_resource)  │  • uploads/<uuid>.<ext>      │
└────────────────────────────────────┘  └──────────────────────────────┘
```

Shared utilities (`utils.py`) provide category definitions, text/location normalization, date calculations, safe image storage, and score formatting across layers.

---

## 11. Technology Stack

| Layer | Technology | Role in ReFind AI |
|---|---|---|
| **Language** | Python 3.13 (compatible with 3.11+) | Core application, AI, database, and testing language |
| **Web UI Framework** | Streamlit | Interactive multipage-style SaaS dashboard, forms, filters, resource caching |
| **NLP & Embeddings** | `sentence-transformers` (`all-MiniLM-L6-v2`) | Local dense semantic embedding generation |
| **Similarity Engine** | `scikit-learn` (`cosine_similarity`) | Fast pairwise vector cosine similarity calculation |
| **Numerical / Tabular** | `numpy`, `pandas` | Vector operations, score clamping, tabular summaries |
| **Database** | SQLite (`sqlite3` standard library) | Zero-configuration persistent relational database |
| **Image Handling** | `Pillow` (PIL) | Validating and rendering optional uploaded item photographs |
| **Automated Testing** | `pytest` | Unit and integration verification of AI matching and persistence |
| **Version Control** | Git & GitHub | Source control, reproducible history, private remote backup |

No external paid APIs, cloud databases, Docker containers, or separate backend microservices are required.

---

## 12. Database Design

ReFind AI stores structured records in a local SQLite database (`data/refind_ai.db`). Uploaded images are stored on the local filesystem inside `uploads/` with unique filenames, and only the relative file path is stored in SQLite (avoiding heavy BLOBs).

### Table: `reports`

| Column | Type | Constraints | Description |
|---|---|---|---|
| `id` | `INTEGER` | `PRIMARY KEY AUTOINCREMENT` | Unique numeric identifier for each report |
| `title` | `TEXT` | `NOT NULL` | Concise item title (e.g., *"Dark Brown Leather Wallet"*) |
| `description` | `TEXT` | `NOT NULL` | Detailed natural-language description |
| `category` | `TEXT` | `NOT NULL` | Standardized item category from controlled list |
| `report_type` | `TEXT` | `NOT NULL`, `CHECK(report_type IN ('lost','found'))` | Whether the item was lost or found |
| `location` | `TEXT` | `NOT NULL` | Last seen (lost) or found campus location |
| `report_date` | `TEXT` | `NOT NULL` | ISO-8601 date (`YYYY-MM-DD`) when lost or found |
| `additional_details` | `TEXT` | `DEFAULT ''` | Optional identifying features or finder observations |
| `contact_label` | `TEXT` | `DEFAULT ''` | Synthetic campus desk / demonstration reference (no private PII) |
| `status` | `TEXT` | `NOT NULL DEFAULT 'active'`, `CHECK(status IN ('active','recovered'))` | Current lifecycle status |
| `image_path` | `TEXT` | `DEFAULT NULL` | Relative local path to optional uploaded photo (`uploads/...`) |
| `is_seed` | `INTEGER` | `NOT NULL DEFAULT 0` | Flag identifying synthetic demo records (`1` = seed, `0` = user) |
| `seed_key` | `TEXT` | `UNIQUE DEFAULT NULL` | Idempotency key preventing duplicate seed rows across restarts |
| `created_at` | `TEXT` | `NOT NULL` | UTC creation timestamp (ISO-8601) |
| `updated_at` | `TEXT` | `NOT NULL` | UTC last-modified timestamp (ISO-8601) |

All SQL queries use **parameterized placeholders (`?`)** to prevent SQL injection and ensure type safety.

---

## 13. AI Matching Algorithm

### 13.1 End-to-End Matching Pipeline

1. **Input Acquisition:** Accept a target report (either newly submitted or selected on the AI Matching Dashboard).
2. **Text Construction & Cleaning:** Concatenate `title`, `description`, and optional `additional_details`, normalize whitespace, and handle empty fields gracefully.
3. **Candidate Filtering:** Retrieve reports from SQLite and strictly filter to the **opposite** `report_type` (`lost` ↔ `found`), excluding the source report itself and optionally filtering to `active` records.
4. **Embedding Generation:** Encode the source text and all candidate texts into 384-D vectors using the cached `all-MiniLM-L6-v2` transformer.
5. **Cosine Similarity Computation:** Compute pairwise cosine similarity via `sklearn.metrics.pairwise.cosine_similarity` and clip values to $[0.0, 1.0]$ for weighted scoring (while preserving raw cosine values for transparency).
6. **Contextual Signal Evaluation:** Independently calculate Category Match, Location Match, and Date Proximity.
7. **Weighted Hybrid Fusion:** Compute the combined score on a `0–100` scale.
8. **Explainable AI Generation:** Construct a concise, evidence-grounded explanation summarizing each active signal.
9. **Ranking & Filtering:** Sort candidates in descending order of `final_score` and present top matches above the configurable threshold.

### 13.2 Hybrid Scoring Formula

$$\text{Final Score} = 100 \times \Big( 0.70 \cdot S_{\text{sem}} + 0.15 \cdot S_{\text{cat}} + 0.10 \cdot S_{\text{loc}} + 0.05 \cdot S_{\text{date}} \Big)$$

Where every component score lies in $[0.0, 1.0]$:

- **Semantic Similarity ($S_{\text{sem}}$, weight `0.70`):**
  $$S_{\text{sem}} = \text{clip}\!\left(\frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|},\ 0.0,\ 1.0\right)$$
  Computed using `sentence-transformers/all-MiniLM-L6-v2` embeddings and Scikit-learn `cosine_similarity`.
- **Category Match ($S_{\text{cat}}$, weight `0.15`):**
  $$S_{\text{cat}} = \begin{cases} 1.0 & \text{if } \text{norm}(\text{category}_A) = \text{norm}(\text{category}_B) \\ 0.0 & \text{otherwise} \end{cases}$$
- **Location Match ($S_{\text{loc}}$, weight `0.10`):**
  Evaluated transparently on normalized location strings without inventing GPS coordinates or campus distances:
  - `1.0` if normalized location strings are identical.
  - `0.85` if one normalized location string contains the other (e.g., `"library"` inside `"central library"`).
  - Otherwise, Jaccard overlap $\frac{|T_A \cap T_B|}{|T_A \cup T_B|}$ over meaningful location tokens (excluding generic prepositions/stop-words such as *near, outside, at, in, the, by, around*).
- **Date Proximity ($S_{\text{date}}$, weight `0.05`):**
  Exponential decay over the absolute calendar day difference $\Delta d = |\text{date}_A - \text{date}_B|$ with configurable half-scale parameter $\tau = 14$ days:
  $$S_{\text{date}} = e^{-\Delta d / 14}$$
  Yields `1.00` for same-day reports, `0.93` for 1 day apart, `0.81` for 3 days apart, `0.61` for 7 days apart, and `0.37` for 14 days apart. Returns `0.0` if either date is missing or invalid.

All four weights and the minimum display threshold are encapsulated in `MatchConfig` and can be tuned dynamically in the UI or configuration.

### 13.3 Explainable AI Design

Every recommendation includes a short, human-readable rationale strictly derived from measured signals—for example:

> *"Strong semantic similarity (84%): Both item descriptions align closely in meaning. Both reports share the 'Wallet & Cards' category. Reported locations share overlapping campus terms ('central library' vs 'near the library'). Reported 1 day apart."*

The explanation engine never fabricates visual observations from photos or geographic distances that were not computed.

---

## 14. System Workflow

1. **Application Startup:** `app.py` initializes SQLite (`database.init_db()`), seeds synthetic demo records idempotently if needed (`seed_data.ensure_seed_data()`), loads custom CSS (`assets/styles.css`), and caches the Sentence Transformer model (`@st.cache_resource`).
2. **Report Submission:** User fills out **Report Lost Item** or **Report Found Item** → inputs validated → optional photo saved to `uploads/` → record inserted via parameterized SQL → AI Matcher automatically runs against opposite-type active reports and displays immediate top matches.
3. **Interactive Exploration:** User navigates to **AI Matching Dashboard** → picks any active report → inspects ranked matches, component bars, side-by-side descriptions, and explanations.
4. **Recovery Resolution:** Once a user confirms a match in the demo workflow, clicking **Mark as Recovered** updates `status = 'recovered'` in SQLite and immediately refreshes dashboard counts.

---

## 15. User Journey

### Scenario A: Student Loses a Wallet
1. Opens **ReFind AI** landing dashboard and sees live campus statistics.
2. Clicks **Report Lost Item**.
3. Enters *"Dark Brown Leather Wallet"*, description *"Lost my dark brown leather wallet near the library"*, category `Wallet & Cards`, location *"Near the Library"*, and date.
4. Submits the form. ReFind AI saves the report and immediately displays a top AI match: *"Brown Leather Bifold Purse-Wallet"* found *"Outside Central Library"* with a high hybrid score and clear explanation.
5. Clicks **Mark as Recovered** (demo claim workflow) to close the case.

### Scenario B: Staff Member Finds a Water Bottle
1. Clicks **Report Found Item**, enters *"Navy Reusable Drinking Bottle"* found in *"Room 204"*, category `Bottles & Drinkware`.
2. Submits the form. ReFind AI surfaces the matching student report (*"Blue water bottle missing from classroom"*) despite completely different phrasing (`navy drinking bottle` vs. `blue water bottle`).

---

## 16. Development Roadmap

| Phase | Milestone | Status |
|---|---|---|
| **Phase 1** | Environment audit, workspace setup, `PROJECT_BLUEPRINT.md`, architecture & AI documentation | Completed |
| **Phase 2** | Python virtual environment (`.venv`), dependency installation, model load verification | Completed |
| **Phase 3** | SQLite persistence layer (`database.py`), synthetic dataset (`seed_data.py`), utility module (`utils.py`) | Completed |
| **Phase 4** | Genuine Sentence Transformer matching engine (`ai_matcher.py`) with hybrid scoring and XAI | Completed |
| **Phase 5** | Polished SaaS-style Streamlit UI (`app.py`, `assets/styles.css`) across all 5 modules | Completed |
| **Phase 6** | Automated unit & integration tests (`tests/test_ai_matcher.py`), persistence check, live launch check | Completed |
| **Phase 7** | Git repository initialization, clean `.gitignore`, commit, and GitHub private repository push | Completed / Handover |

---

## 17. Testing Strategy

Testing combines automated `pytest` verification with live UI smoke verification:

1. **Test 1 — Strong Semantic Match:** *"Black leather wallet lost near the library"* vs. *"Found a dark black leather wallet outside the central library"* → verifies high semantic similarity and top hybrid ranking.
2. **Test 2 — Paraphrased / Different Wording:** *"My blue water bottle is missing from the classroom"* vs. *"Found a navy reusable drinking bottle in Room 204"* → verifies NLP recognizes synonymy (`blue`/`navy`, `water bottle`/`drinking bottle`) beyond exact keywords.
3. **Test 3 — Different Categories & Items:** *"Black leather wallet"* vs. *"Silver wireless headphones"* → verifies low semantic and final hybrid score.
4. **Test 4 — Same Category, Different Object:** *"Red backpack"* vs. *"Blue school bag"* → verifies score reflects nuanced description and contextual differences rather than blindly awarding a high score for matching category alone.
5. **Test 5 — Opposite Report-Type Enforcement:** Verifies `lost` reports never match other `lost` reports and `found` reports never match other `found` reports.
6. **Test 6 — Database Persistence & Idempotency:** Inserts a new report, closes and reopens a fresh SQLite connection (simulating restart), confirms the record persists, and verifies repeated seeding creates zero duplicates.
7. **Edge Cases:** Empty descriptions, missing dates, non-matching locations, custom weight validation, and explanation faithfulness checks.

---

## 18. Expected Outcomes

- A fully working, locally hosted AI web application launchable with `streamlit run app.py`.
- Demonstrably superior matching over keyword search on paraphrased campus lost-and-found descriptions.
- Clean, transparent explainable AI cards suitable for live faculty demonstration.
- Reliable local SQLite storage with realistic pre-seeded campus records.
- Reproducible automated tests verifying every core requirement.
- Comprehensive academic documentation ready for viva and report submission.

---

## 19. Project Limitations

1. **Text-Only AI Matching (Current MVP):** Uploaded images are stored and displayed for human verification, but visual image embeddings (e.g., CLIP) are not yet incorporated into the similarity score.
2. **Lexical Location Matching:** Location similarity compares normalized strings and tokens rather than true campus GIS coordinates or building-to-building walking distances.
3. **Unauthenticated Demo Claim Flow:** To avoid unnecessary complexity and PII exposure in an academic MVP, user login, institutional SSO, and private messaging are omitted; recovery uses a simplified demonstration status action.
4. **Local Single-Node Storage:** SQLite is ideal for a single-campus MVP or demo, but high-concurrency multi-campus deployment would require PostgreSQL + vector indexing (e.g., `pgvector`).
5. **Cold-Start Model Download:** The first run requires internet access to download `all-MiniLM-L6-v2` (~90 MB) from Hugging Face; subsequent runs operate offline using the local cache.
6. **Similarity vs. Probability:** Output scores are heuristic weighted similarity indices, not calibrated match probabilities.

---

## 20. Future Enhancements

- **Multimodal Image + Text Matching:** Integrate CLIP (`clip-ViT-B-32`) to match uploaded item photographs alongside natural-language descriptions.
- **Campus Spatial Graph:** Map campus buildings and rooms to a structured campus graph or coordinates so *"Library"* and *"Reading Hall"* receive true proximity scores.
- **Persistent Vector Indexing:** Cache precomputed embeddings in SQLite or FAISS/`pgvector` to scale smoothly to tens of thousands of reports.
- **Campus SSO & Verified Claim Handover:** Add university email authentication, proof-of-ownership verification questions, and email/push notifications when a high-confidence match appears.
- **Feedback-Driven Calibration:** Collect confirmed match/non-match feedback from campus desks to train a calibrated logistic regression layer that converts raw signals into true match probabilities.

---

## 21. Academic Learning Outcomes

Through designing and building ReFind AI, the student demonstrates mastery of:

- **Natural Language Processing & Representation Learning:** Understanding dense transformer embeddings, tokenization, bi-encoder architectures, and semantic vector spaces.
- **Information Retrieval & Similarity Metrics:** Applying cosine similarity, bounding scores, and designing hybrid multi-signal ranking functions.
- **Explainable AI (XAI):** Designing transparent, faithful explanations and rigorously distinguishing similarity metrics from statistical probabilities.
- **Software Architecture & Data Engineering:** Separating presentation, AI inference, and relational persistence layers; preventing SQL injection; managing file storage cleanly.
- **Verification & DevOps Practices:** Writing automated tests with `pytest`, managing reproducible Python virtual environments, and maintaining clean Git/GitHub repositories.

---

## 22. Project Success Criteria

| Criterion | Measurable Target |
|---|---|
| **Genuine AI Execution** | `sentence-transformers/all-MiniLM-L6-v2` loads cleanly and computes real embeddings + cosine similarity with zero keyword fallback. |
| **Semantic Accuracy** | Paraphrased lost/found pairs (Tests 1 & 2) outscore unrelated pairs (Test 3) by a wide, verifiable margin. |
| **Strict Type Isolation** | 0% same-type (`lost↔lost` or `found↔found`) matches ever returned. |
| **Persistence** | Reports and recovery status updates survive application restarts; seed data never duplicates. |
| **Explainability** | Every recommended match includes component scores and a grounded plain-English explanation. |
| **Usability & Polish** | Clean SaaS-inspired Streamlit interface with zero runtime errors across all five views. |
| **Reproducibility** | Automated `pytest` suite passes completely and app launches via `streamlit run app.py`. |
