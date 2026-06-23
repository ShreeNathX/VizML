# VizML: Desktop Automated Data Curation & Model Selection Engine
 
An installable, zero-cloud-cost desktop application that bridges the gap between manual data cleaning and full-scale automated machine learning (AutoML). The platform runs entirely inside the client machine's own memory and CPU/GPU — no data leaves the device, no cloud bill, no API key, no internet dependency once installed.
 
> **Elevator pitch:** Drop in a messy CSV → get a profiled, cleaned dataset, an interactive 2D/3D exploration view, and a best-fit trained model (.pkl) — all in one local desktop session.
 
---
 
## Table of Contents
 
1. [Why This Project Exists](#why-this-project-exists)
2. [Core Features](#core-features)
3. [Architecture & System Flow](#architecture--system-flow)
4. [End-to-End Pipeline Logic](#end-to-end-pipeline-logic)
5. [Project Directory Structure](#project-directory-structure)
6. [Tech Stack & Dependencies](#tech-stack--dependencies)
7. [Building the Project From Scratch](#building-the-project-from-scratch)
8. [Module-by-Module Implementation Guide](#module-by-module-implementation-guide)
9. [Running the Application](#running-the-application)
10. [Configuration & Customization](#configuration--customization)
11. [Packaging as a Standalone Desktop App](#packaging-as-a-standalone-desktop-app)
12. [Roadmap](#roadmap)
13. [Contributing](#contributing)
14. [License](#license)
---
 
## Why This Project Exists
 
Most data scientists and analysts repeat the same three steps before any real modeling happens:
 
1. Load a CSV and figure out what's wrong with it (missing values, wrong types, outliers, duplicates).
2. Clean it by hand in a notebook, re-running cells over and over.
3. Try a handful of baseline models just to see what's even feasible before investing in tuning.
VizML automates steps 1–3 in a single local desktop session, with a UI instead of notebook cells, so the feedback loop is seconds instead of minutes — and because everything happens in local RAM, there is no concern about uploading sensitive or regulated data anywhere.
 
---
 
## Core Features
 
| Feature | Description |
|---|---|
| **Local-only execution** | No network calls for data processing; all computation happens on the user's CPU/RAM. |
| **Automated data profiling** | Detects column types, missing-value ratios, cardinality, and outliers on ingestion. |
| **Non-destructive cleaning** | Every cleaning operation works on a copy held in `st.session_state`; the original upload is never overwritten on disk. |
| **Interactive 2D/3D visualization** | Plotly-based exploration matrix for spotting correlations and clusters before modeling. |
| **Hardware-aware task detection** | Automatically infers classification vs. regression vs. clustering from the target column. |
| **Baseline AutoML** | Cross-validates a curated shortlist of scikit-learn models and ranks them by performance. |
| **One-click export** | Download the cleaned CSV and the serialized best model (`.pkl`) directly from the UI. |
 
---
 
## Architecture & System Flow
 
```text
+-------------------------------------------------------------------------------------+
|                                 USER HARDWARE (RAM)                                 |
|                                                                                     |
|  +------------------+      Memory Ingestion      +-----------------------------+    |
|  | Local File (.csv)| -------------------------> |  st.session_state['df']     |    |
|  +------------------+                            +-----------------------------+    |
|                                                                |                    |
|                                                                v                    |
|                                                  +-----------------------------+    |
|                                                  |   DataCleaner Engine        |    |
|                                                  +-----------------------------+    |
|                                                                |                    |
|                                                                v                    |
|                                                  +-----------------------------+    |
|                                                  |   2D / 3D Plotly Matrix     |    |
|                                                  +-----------------------------+    |
|                                                                |                    |
|                                                                v                    |
|                                                  +-----------------------------+    |
|                                                  |   ModelSelector Engine      |    |
|                                                  +-----------------------------+    |
|                                                                |                    |
|                                                                v                    |
|                                                  +-----------------------------+    |
|                                                  | Baseline Cross-Validation   |    |
|                                                  +-----------------------------+    |
|                                                                |                    |
|                +------------------+      Export Artifacts      |                    |
|                | Cleaned CSV Disk | <--------------------------+                    |
|                +------------------+                            |                    |
|                | Final Model .pkl | <--------------------------+                    |
|                +------------------+                                                 |
+-------------------------------------------------------------------------------------+
```
 
**Read the diagram top to bottom:** a file is read into memory once, every subsequent stage (cleaning, visualization, model selection) mutates only the in-memory object, and disk writes happen only at the very end, on explicit user action ("Export").
 
---
 
## End-to-End Pipeline Logic
 
### Stage 1 — Ingestion
- User uploads a `.csv` (or `.xlsx`) through a Streamlit `file_uploader` widget.
- The file is read with `pandas.read_csv` directly from the in-memory buffer Streamlit provides — it is **never written to a temp path on disk**.
- The resulting `DataFrame` is stored in `st.session_state['df']` so it survives Streamlit's re-run-on-every-interaction model.
- A profiling pass runs immediately and populates `st.session_state['profile']` with: dtypes, null counts/percentages, unique-value counts, and basic descriptive statistics (`df.describe(include='all')`).
### Stage 2 — Cleaning (`src/cleaner.py`)
The `DataCleaner` class exposes one method per operation so the UI can call them independently and show a live diff:
 
- **Missing-value handling** — numeric columns imputed with median (robust to outliers) or mean (toggle), categorical columns imputed with mode or a constant `"Unknown"` token.
- **Type coercion** — strings that are actually numbers/dates get auto-cast; the engine never silently drops a column it can't parse, it flags it instead.
- **Outlier handling** — IQR-based detection (`Q1 - 1.5*IQR`, `Q3 + 1.5*IQR`) with a user choice of clip, remove, or flag-only.
- **Duplicate removal** — exact-row duplicate detection with a count shown before removal is confirmed.
- **Categorical encoding** — optional one-hot or label encoding, deferred until just before modeling so the visualization stage still works on human-readable categories.
Every method returns a **new DataFrame** rather than mutating in place, so the UI can offer an "Undo" by simply not committing the result back to `st.session_state`.
 
### Stage 3 — Visualization
- A Plotly Express scatter/scatter-3d matrix is generated from the cleaned numeric columns.
- Correlation heatmap (`plotly.figure_factory` or `px.imshow`) helps the user (and the model selector) spot redundant features.
- This stage is purely diagnostic — it does not mutate `st.session_state['df']`.
### Stage 4 — Model Selection (`src/selector.py`)
The `ModelSelector` class:
 
1. **Task detection** — inspects the target column the user selects: if dtype is object/category or has low cardinality relative to row count → classification; if numeric with high cardinality → regression.
2. **Baseline shortlist** — for classification: Logistic Regression, Random Forest, Gradient Boosting, SVM; for regression: Linear Regression, Random Forest Regressor, Gradient Boosting Regressor.
3. **Cross-validation** — `sklearn.model_selection.cross_val_score` with stratified k-fold (classification) or standard k-fold (regression), default `k=5`.
4. **Scoring** — accuracy/F1 for classification, RMSE/R² for regression; results are ranked and displayed in a sortable table.
5. **Hardware awareness** — checks available CPU core count (`os.cpu_count()`) and sets `n_jobs=-1` where supported so training uses all local cores without configuration.
### Stage 5 — Export
- Cleaned DataFrame → `df.to_csv()` → served via Streamlit's `download_button` (in-memory bytes, no disk write needed on the server side).
- Best model → `pickle.dumps()` (or `joblib.dump`) → served the same way.
---
 
## Project Directory Structure
 
```text
vizml/
│
├── app.py                  # Desktop UI application entrypoint (Streamlit interface)
├── requirements.txt        # Frozen local dependency list
├── README.md                # This file
│
├── src/                     # Core algorithmic framework
│   ├── __init__.py          # Marks src/ as an importable package
│   ├── cleaner.py            # DataCleaner: profiling + non-destructive cleaning
│   └── selector.py           # ModelSelector: task detection + baseline AutoML
│
└── tests/                   # Unit tests (recommended addition)
    ├── test_cleaner.py
    └── test_selector.py
```
 
---
 
## Tech Stack & Dependencies
 
| Layer | Choice | Why |
|---|---|---|
| UI framework | **Streamlit** | Fastest way to ship a desktop-like local web UI in pure Python; runs on `localhost`, no separate frontend build. |
| Data handling | **pandas / numpy** | Industry-standard in-memory tabular processing. |
| Visualization | **Plotly** | Interactive 2D/3D plots that render natively inside Streamlit. |
| Modeling | **scikit-learn** | Mature, well-documented baseline models and cross-validation utilities. |
| Serialization | **joblib** (or `pickle`) | Efficient persistence of trained scikit-learn estimators. |
 
### `requirements.txt`
 
```text
streamlit>=1.32.0
pandas>=2.2.0
numpy>=1.26.0
plotly>=5.20.0
scikit-learn>=1.4.0
joblib>=1.3.0
openpyxl>=3.1.0
```
 
---
 
## Building the Project From Scratch
 
### Prerequisites
- Python 3.10 or later installed and on your `PATH`.
- `pip` available (bundled with modern Python installers).
- (Optional but recommended) `git` for version control.
### Step 1 — Create the project folder and virtual environment
 
```bash
mkdir vizml && cd vizml
python -m venv venv
 
# Activate the environment
# macOS / Linux:
source venv/bin/activate
# Windows (PowerShell):
venv\Scripts\Activate.ps1
```
 
### Step 2 — Scaffold the directory structure
 
```bash
mkdir src tests
touch app.py requirements.txt README.md
touch src/__init__.py src/cleaner.py src/selector.py
touch tests/test_cleaner.py tests/test_selector.py
```
 
### Step 3 — Add dependencies and install
 
Paste the `requirements.txt` contents from the section above, then:
 
```bash
pip install -r requirements.txt
```
 
### Step 4 — Implement the modules
 
Follow the [Module-by-Module Implementation Guide](#module-by-module-implementation-guide) below to fill in `src/cleaner.py`, `src/selector.py`, and `app.py`.
 
### Step 5 — Run and verify
 
```bash
streamlit run app.py
```
 
Streamlit will open `http://localhost:8501` in your default browser — this is the "desktop window" for the app.
 
---
 
## Module-by-Module Implementation Guide
 
### `src/cleaner.py` — `DataCleaner`
 
Responsibilities and suggested method signatures:
 
```python
class DataCleaner:
    def __init__(self, df: "pd.DataFrame"):
        self.df = df.copy()
 
    def profile(self) -> dict:
        """Return dtypes, null %, unique counts, describe() summary."""
 
    def handle_missing(self, strategy: str = "median") -> "pd.DataFrame":
        """strategy: 'median' | 'mean' | 'mode' | 'constant'."""
 
    def detect_outliers(self, method: str = "iqr") -> "pd.DataFrame":
        """Return a boolean mask DataFrame flagging outlier cells."""
 
    def handle_outliers(self, action: str = "clip") -> "pd.DataFrame":
        """action: 'clip' | 'remove' | 'flag'."""
 
    def remove_duplicates(self) -> "pd.DataFrame":
        ...
 
    def encode_categoricals(self, method: str = "onehot") -> "pd.DataFrame":
        """method: 'onehot' | 'label'."""
```
 
Design rule: **every public method returns a new DataFrame**; the caller (the Streamlit app) decides whether to commit the result to `st.session_state['df']`.
 
### `src/selector.py` — `ModelSelector`
 
```python
class ModelSelector:
    def __init__(self, df: "pd.DataFrame", target: str):
        self.df = df
        self.target = target
        self.task = self._detect_task()
 
    def _detect_task(self) -> str:
        """Return 'classification' or 'regression' based on target dtype/cardinality."""
 
    def get_candidate_models(self) -> dict:
        """Return {name: sklearn_estimator} shortlist based on self.task."""
 
    def run_baseline(self, cv: int = 5) -> "pd.DataFrame":
        """Cross-validate every candidate, return a ranked results table."""
 
    def best_model(self):
        """Fit and return the top-ranked estimator on the full dataset."""
```
 
### `app.py` — Streamlit Entrypoint
 
High-level skeleton:
 
```python
import streamlit as st
import pandas as pd
from src.cleaner import DataCleaner
from src.selector import ModelSelector
 
st.set_page_config(page_title="VizML", layout="wide")
st.title("VizML: Local Data Curation & Model Selection")
 
uploaded = st.file_uploader("Upload a CSV file", type=["csv", "xlsx"])
if uploaded:
    if "df" not in st.session_state:
        st.session_state["df"] = pd.read_csv(uploaded)
 
    cleaner = DataCleaner(st.session_state["df"])
    st.subheader("Data Profile")
    st.json(cleaner.profile())
 
    if st.button("Apply Cleaning"):
        cleaned = cleaner.handle_missing()
        cleaned = DataCleaner(cleaned).remove_duplicates()
        st.session_state["df"] = cleaned
 
    st.subheader("Explore")
    # Plotly scatter / scatter_3d / heatmap calls go here
 
    target = st.selectbox("Select target column", st.session_state["df"].columns)
    if st.button("Run Baseline Model Selection"):
        selector = ModelSelector(st.session_state["df"], target)
        results = selector.run_baseline()
        st.dataframe(results)
 
    st.download_button(
        "Download Cleaned CSV",
        st.session_state["df"].to_csv(index=False),
        file_name="cleaned_data.csv",
    )
```
 
This is intentionally a skeleton — fill in the Plotly figures, error handling (empty uploads, all-null columns, single-class targets), and progress spinners as you flesh it out.
 
---
 
## Running the Application
 
```bash
# From the project root, with the virtual environment activated
streamlit run app.py
```
 
The app opens automatically in your browser at `http://localhost:8501`. Closing the terminal process shuts the app down — nothing persists outside the session unless the user explicitly clicks an export/download button.
 
---
 
## Configuration & Customization
 
| What to change | Where |
|---|---|
| Add a new cleaning strategy | Add a method to `DataCleaner` in `src/cleaner.py`, then wire a UI control for it in `app.py`. |
| Add/remove candidate models | Edit `get_candidate_models()` in `src/selector.py`. |
| Change cross-validation folds | Pass a different `cv` value to `run_baseline()`. |
| Theme the UI | Add a `.streamlit/config.toml` with `[theme]` settings. |
 
---
 
## Packaging as a Standalone Desktop App
 
Streamlit apps run as local web servers, but they can be wrapped to feel like a native desktop install:
 
1. **PyInstaller**: bundle `app.py` plus a small launcher script that calls `streamlit run` programmatically, then package with PyInstaller into a single executable.
2. **pywebview**: launch the Streamlit server as a background process and open it inside a lightweight native window (no browser chrome) using `pywebview`.
3. **Electron wrapper**: point an Electron shell at `http://localhost:8501` for a cross-platform installer with a proper app icon and taskbar entry.
Any of the three keeps the "zero-cloud-cost" property intact since the Streamlit server still only binds to `localhost`.
 
---
 
## Roadmap
 
- [ ] Hyperparameter tuning pass (Optuna or `GridSearchCV`) on the selected best model.
- [ ] Support for Parquet and JSON ingestion.
- [ ] Pluggable cleaning "recipes" that can be saved/reused across datasets.
- [ ] Multi-target / multi-output modeling support.
- [ ] Dark-mode native theme out of the box.
---
 
## Contributing
 
1. Fork the repository and create a feature branch.
2. Keep `DataCleaner` and `ModelSelector` methods pure (no `st.session_state` access inside `src/`) so they stay independently testable.
3. Add or update tests under `tests/` for any new method.
4. Open a pull request with a clear description of the change and before/after behavior.
---
 
## License
 
Specify a license (e.g., MIT) in a `LICENSE` file at the project root before distributing this publicly.