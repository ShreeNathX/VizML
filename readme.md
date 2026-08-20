# VizML Studio

**A local-first, end-to-end data intelligence platform — built with Streamlit and Plotly.**

Upload a messy CSV or Excel file, clean it up, explore it visually, train ML models, and generate executive-grade reports — all running on your machine. No cloud required.

---

## Features

### Phase 01 — Data Curation Engine
- Drag-and-drop CSV / Excel upload (up to 100 MB)
- Developer manual type casting (`int64`, `float64`, `str`, `category`, `datetime64`, `bool`)
- Automatic schema detection and high-confidence type coercion
- Missing-value imputation (median, mean, mode, forward/backward fill, constant)
- Outlier detection and handling (IQR clip, remove, or flag)
- Duplicate row removal
- Categorical encoding (one-hot or label)
- Non-destructive pipeline with undo history and live diff previews

### Phase 02 — Diagnostic Visualizations
- Real-time KPI telemetry (rows, nulls, duplicates, numeric/categorical split)
- Interactive cohort filtering with slice-and-dice controls
- Automated statistical insight cards
- Correlation heatmaps, scatter matrices, and 3D spatial projectors
- Custom chart builder with 12 chart types and 9 color palette schemes

### Phase 03 — ML Studio
- Auto task detection (Classification vs Regression)
- Multi-algorithm cross-validation leaderboard (Linear, Ridge, Lasso, Random Forest, Gradient Boosting, SVM, KNN, Decision Tree, Naive Bayes)
- Confusion matrices, residual histograms, and actual-vs-predicted plots
- Feature importance rankings
- Live What-If scenario simulation with custom input sliders
- Model export as `.pkl`

### Phase 04 — Executive Summary Report
- Consolidated summaries of data curation, statistical diagnostics, and ML results
- Auto-generated strategic recommendations & data purity scoring
- Custom chart builder with 7 chart types, custom axes, groupings, and 9 color palettes
- One-click downloads for curated dataset (CSV), ML leaderboard (CSV), and self-contained HTML reports

---

## Getting Started

### Prerequisites
- Python 3.10 or higher
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/VizML.git
cd VizML

# 2. Create and activate a virtual environment (recommended)
python -m venv env
source env/bin/activate      # macOS / Linux
env\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run App.py
```

The app opens at `http://localhost:8501` in your browser.

---

## Project Structure

```
VizML/
├── App.py                          # Landing page and navigation hub
├── pages/
│   ├── 1_Data_Curation.py         # Phase 01: upload, clean, profile
│   ├── 2_Diagnostic_Visualizations.py  # Phase 02: charts and analytics
│   ├── 3_ML_Studio.py             # Phase 03: ML training and simulation
│   └── 4_Executive_Reports.py     # Phase 04: dashboards and exports
├── src/
│   ├── cleaner.py                 # DataCleaner — all data cleaning logic
│   ├── ml_engine.py               # MLEngine — preprocessing, training, evaluation
│   └── styles.py                  # Claymorphic design system (CSS + Plotly themes)
├── .streamlit/
│   └── config.toml                # Theme and server configuration
├── requirements.txt
└── README.md
```

---

## Tech Stack

| Layer | Library |
|---|---|
| UI Framework | [Streamlit](https://streamlit.io) |
| Data Processing | [pandas](https://pandas.pydata.org), [NumPy](https://numpy.org) |
| Machine Learning | [scikit-learn](https://scikit-learn.org) |
| Visualization | [Plotly Express](https://plotly.com/python/plotly-express/) |
| Excel Support | [openpyxl](https://openpyxl.readthedocs.io) |
| Statistics | [statsmodels](https://www.statsmodels.org) |

---

## Notes

- All data stays on your machine — nothing is sent to external servers.
- Trained models are exported as `.pkl` files (excluded from git by default via `.gitignore`).
- The app uses session state to pass data between pages, so keep the browser tab open while working through the pipeline.

---

## License

MIT — free to use, modify, and distribute.
