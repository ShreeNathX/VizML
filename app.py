import streamlit as st
import pandas as pd
import os

st.set_page_config(
    page_title="VizML",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Premium CSS ────────────────────────────────────────────────────
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700;800&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Hide default Streamlit branding */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #0D0E14;
        border-right: 1px solid #1E2030;
    }
    section[data-testid="stSidebar"] .stMarkdown p {
        color: #8A8BA8;
        font-size: 13px;
    }

    /* Hero */
    .hero-wrap {
        background: linear-gradient(135deg, #0D0E14 0%, #13141F 100%);
        border: 1px solid #1E2030;
        border-radius: 16px;
        padding: 48px 40px;
        margin-bottom: 32px;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(108, 99, 255, 0.12);
        color: #6C63FF;
        font-family: 'Inter', sans-serif;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        padding: 5px 14px;
        border-radius: 50px;
        border: 1px solid rgba(108, 99, 255, 0.3);
        margin-bottom: 20px;
    }
    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 52px;
        font-weight: 800;
        background: linear-gradient(135deg, #EEEEF5 0%, #6C63FF 60%, #EC4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.15;
        margin-bottom: 16px;
    }
    .hero-sub {
        font-size: 17px;
        color: #8A8BA8;
        max-width: 600px;
        line-height: 1.7;
        margin-bottom: 32px;
    }

    /* Feature cards */
    .feature-card {
        background: #13141A;
        border: 1px solid #1E2030;
        border-radius: 12px;
        padding: 24px;
        height: 100%;
        transition: border-color 0.2s;
    }
    .feature-card:hover {
        border-color: #6C63FF;
    }
    .feature-card-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 16px;
        font-weight: 700;
        color: #EEEEF5;
        margin-bottom: 8px;
    }
    .feature-card-desc {
        font-size: 13px;
        color: #8A8BA8;
        line-height: 1.6;
    }
    .feature-card-num {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 11px;
        font-weight: 700;
        color: #6C63FF;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 12px;
    }

    /* Session status panel */
    .status-panel {
        background: #13141A;
        border: 1px solid #1E2030;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 24px;
    }
    .status-label {
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        color: #8A8BA8;
        margin-bottom: 12px;
    }
    .status-row {
        display: flex;
        justify-content: space-between;
        font-size: 13px;
        color: #EEEEF5;
        padding: 6px 0;
        border-bottom: 1px solid #1E2030;
    }
    .status-row:last-child { border-bottom: none; }
    .status-val { color: #6C63FF; font-weight: 600; font-family: 'Space Grotesk', sans-serif; }
    .status-val-none { color: #4B4C65; font-style: italic; }

    /* Metric override */
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 700;
        color: #6C63FF;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 13px;
        color: #8A8BA8;
        font-weight: 500;
    }
    </style>
""", unsafe_allow_html=True)


# ── Hero Section ───────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
    <div class="hero-badge">Machine Learning Toolkit</div>
    <div class="hero-title">VizML</div>
    <div class="hero-sub">
        An end-to-end data intelligence platform. Upload raw datasets, curate and clean them with
        intelligent tooling, explore multi-dimensional diagnostics, and prepare data for modeling —
        all within a unified, session-aware workspace.
    </div>
</div>
""", unsafe_allow_html=True)


# ── Feature Cards ──────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-card-num">Step 01</div>
        <div class="feature-card-title">Data Curation Engine</div>
        <div class="feature-card-desc">
            Upload CSV or Excel files, profile missing values, coerce types, handle outliers,
            remove duplicates, and encode categoricals — with live diffs and full undo history.
        </div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-card-num">Step 02</div>
        <div class="feature-card-title">Diagnostic Visualizations</div>
        <div class="feature-card-desc">
            Auto-generated insight dashboard, correlation matrices, scatter matrix, 3D scatter
            explorer, and a fully configurable custom chart builder with 11 chart types.
        </div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-card-num">Step 03</div>
        <div class="feature-card-title">Machine Learning</div>
        <div class="feature-card-desc">
            Train, evaluate, and compare machine learning models directly on your curated
            session dataset. Feature selection, cross-validation, and model export — coming soon.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ── Session Status ─────────────────────────────────────────────────
df = st.session_state.get("df", None)
original_df = st.session_state.get("original_df", None)
history = st.session_state.get("history", [])

if df is not None:
    rows, cols = df.shape
    missing = int(df.isna().sum().sum())
    dups = int(df.duplicated().sum())
    undo_steps = len(history)

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Rows", f"{rows:,}")
    col_m2.metric("Columns", cols)
    col_m3.metric("Missing Cells", f"{missing:,}")
    col_m4.metric("Undo Steps Available", undo_steps)

    file_name = st.session_state.get("current_file_name", "Unknown")
    st.markdown(f"""
    <div class="status-panel">
        <div class="status-label">Active Session</div>
        <div class="status-row">
            <span>Dataset</span><span class="status-val">{file_name}</span>
        </div>
        <div class="status-row">
            <span>Original Shape</span>
            <span class="status-val">{original_df.shape[0]} x {original_df.shape[1]}</span>
        </div>
        <div class="status-row">
            <span>Current Shape</span>
            <span class="status-val">{rows} x {cols}</span>
        </div>
        <div class="status-row">
            <span>Duplicate Rows</span>
            <span class="status-val">{dups}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.info("Dataset loaded. Navigate to **Data Curation** or **Diagnostic Visualizations** using the sidebar.")
else:
    st.markdown("""
    <div class="status-panel">
        <div class="status-label">Session Status</div>
        <div class="status-row">
            <span>Dataset</span><span class="status-val-none">No dataset loaded</span>
        </div>
        <div class="status-row">
            <span>Status</span><span class="status-val-none">Waiting for upload</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.info("No dataset loaded yet. Go to **Data Curation** in the sidebar to upload a CSV or Excel file.")


# ── Sidebar ────────────────────────────────────────────────────────
st.sidebar.markdown("### VizML")
st.sidebar.markdown("Select a page from the navigation above to begin.")

if df is not None:
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Active Dataset**")
    st.sidebar.markdown(f"Shape: `{df.shape[0]} x {df.shape[1]}`")
    st.sidebar.markdown(f"File: `{st.session_state.get('current_file_name', 'Unknown')}`")