import streamlit as st

st.set_page_config(
    page_title="VizML",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Design System
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    :root {
        --bg-main: #090A10;
        --bg-card: #121420;
        --bg-card-hover: #181B2B;
        --border-color: rgba(255, 255, 255, 0.08);
        --border-hover: rgba(99, 102, 241, 0.35);
        --accent-indigo: #6366F1;
        --accent-emerald: #10B981;
        --accent-amber: #F59E0B;
        --accent-rose: #F43F5E;
        --text-primary: #F1F3F9;
        --text-secondary: #94A3B8;
        --text-muted: #64748B;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: var(--text-primary);
    }

    /* Hide standard header & footer chrome */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }

    /* Main Container Background */
    .stApp {
        background-color: var(--bg-main);
        background-image: 
            radial-gradient(at 12% 15%, rgba(99, 102, 241, 0.07) 0px, transparent 40%),
            radial-gradient(at 88% 85%, rgba(16, 185, 129, 0.05) 0px, transparent 40%),
            radial-gradient(at 50% 50%, rgba(244, 63, 94, 0.03) 0px, transparent 60%);
        background-attachment: fixed;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: #0C0D16 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.06) !important;
    }

    /* Hero Wrapper */
    .hero-wrap {
        position: relative;
        background: linear-gradient(135deg, rgba(18, 20, 32, 0.95) 0%, rgba(12, 13, 22, 0.98) 100%);
        border: 1px solid var(--border-color);
        border-radius: 20px;
        padding: 44px 40px;
        margin-bottom: 32px;
        box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.08);
        overflow: hidden;
    }
    .hero-wrap::before {
        content: '';
        position: absolute;
        top: 0; right: 0; width: 350px; height: 100%;
        background: radial-gradient(circle at 80% 20%, rgba(99, 102, 241, 0.15), transparent 70%);
        pointer-events: none;
    }

    .hero-badge-container {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 18px;
    }
    .pulse-dot {
        width: 8px;
        height: 8px;
        background-color: var(--accent-emerald);
        border-radius: 50%;
        box-shadow: 0 0 10px var(--accent-emerald);
        animation: pulse-glow 2s infinite;
    }
    @keyframes pulse-glow {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.3; transform: scale(0.85); }
    }
    .hero-badge-text {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: #A5B4FC;
        background: rgba(99, 102, 241, 0.12);
        padding: 5px 14px;
        border-radius: 20px;
        border: 1px solid rgba(99, 102, 241, 0.25);
    }

    .hero-title {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 48px;
        font-weight: 800;
        letter-spacing: -1.2px;
        background: linear-gradient(135deg, #FFFFFF 0%, #C7D2FE 50%, #818CF8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.1;
        margin-bottom: 14px;
    }
    .hero-sub {
        font-size: 15px;
        color: var(--text-secondary);
        max-width: 680px;
        line-height: 1.65;
        margin-bottom: 0;
    }

    /* Workflow Feature Cards */
    .feature-card {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 16px;
        padding: 26px 24px;
        height: 100%;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        position: relative;
        overflow: hidden;
    }
    .feature-card:hover {
        border-color: var(--border-hover);
        transform: translateY(-4px);
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.35), 0 0 20px rgba(99, 102, 241, 0.1);
        background: var(--bg-card-hover);
    }
    .feature-step-pill {
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        font-weight: 700;
        color: var(--accent-indigo);
        letter-spacing: 1.5px;
        text-transform: uppercase;
        background: rgba(99, 102, 241, 0.1);
        padding: 3px 10px;
        border-radius: 6px;
        display: inline-block;
        margin-bottom: 14px;
    }
    .feature-title {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 17px;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 8px;
    }
    .feature-desc {
        font-size: 13px;
        color: var(--text-secondary);
        line-height: 1.6;
    }

    /* Status & Metrics Panel */
    .status-card {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 16px;
        padding: 24px;
        margin-top: 24px;
    }
    .status-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 16px;
        padding-bottom: 12px;
        border-bottom: 1px solid var(--border-color);
    }
    .status-title {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 14px;
        font-weight: 700;
        letter-spacing: 0.5px;
        color: var(--text-primary);
        text-transform: uppercase;
    }
    .status-row {
        display: flex;
        justify-content: space-between;
        font-size: 13px;
        padding: 8px 0;
        border-bottom: 1px dashed rgba(255, 255, 255, 0.05);
    }
    .status-row:last-child { border-bottom: none; }
    .status-key { color: var(--text-muted); }
    .status-val-active {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
        color: var(--accent-emerald);
    }
    .status-val-empty {
        font-style: italic;
        color: var(--text-muted);
    }

    /* Streamlit Widget Overrides */
    div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace;
        font-size: 26px;
        font-weight: 700;
        color: var(--accent-indigo);
    }
    div[data-testid="stMetricLabel"] {
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: var(--text-muted);
    }
    </style>
""", unsafe_allow_html=True)

# Bespoke Hero Section
st.markdown("""
<div class="hero-wrap">
    <div class="hero-badge-container">
        <div class="pulse-dot"></div>
        <div class="hero-badge-text">Interactive Machine Learning & Analytics Studio</div>
    </div>
    <div class="hero-title">VizML</div>
    <div class="hero-sub">
        An end-to-end data intelligence workspace. Upload raw tabular datasets, perform multi-stage data curation with full undo history, explore diagnostic visual analytics, and build predictive machine learning models seamlessly.
    </div>
</div>
""", unsafe_allow_html=True)

# Workflow Steps Grid
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-card">
        <span class="feature-step-pill">Phase 01</span>
        <div class="feature-title">Data Curation Engine</div>
        <div class="feature-desc">
            Profile dataset health, coerce unparseable types, impute missing values, handle outliers with custom thresholds, eliminate duplicate rows, and encode categoricals with full undo step control.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <span class="feature-step-pill">Phase 02</span>
        <div class="feature-title">Diagnostic Visualizations</div>
        <div class="feature-desc">
            Explore automated correlation matrices, scatter matrix projections, 3D spatial feature explorers, and a high-flexibility chart builder offering 10+ Plotly interactive chart types.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card">
        <span class="feature-step-pill">Phase 03</span>
        <div class="feature-title">Machine Learning Studio</div>
        <div class="feature-desc">
            Automated task inference (Classification or Regression), multi-model cross-validated benchmark leaderboards, interactive residual & confusion matrix diagnostics, and live What-If scenario prediction.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Session State Inspection
df = st.session_state.get("df", None)
original_df = st.session_state.get("original_df", None)
history = st.session_state.get("history", [])

if df is not None:
    rows, cols = df.shape
    missing = int(df.isna().sum().sum())
    dups = int(df.duplicated().sum())
    undo_steps = len(history)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Active Rows", f"{rows:,}")
    m2.metric("Active Columns", cols)
    m3.metric("Missing Values", f"{missing:,}")
    m4.metric("Undo History", f"{undo_steps} steps")

    file_name = st.session_state.get("current_file_name", "Unknown Dataset")
    st.markdown(f"""
    <div class="status-card">
        <div class="status-header">
            <span class="status-title">Active Workspace Session</span>
            <span style="color: #10B981; font-size: 11px; font-weight: 700; font-family: 'JetBrains Mono', monospace; letter-spacing: 1px; background: rgba(16, 185, 129, 0.12); padding: 4px 10px; border-radius: 6px; border: 1px solid rgba(16, 185, 129, 0.25);">SESSION READY</span>
        </div>
        <div class="status-row">
            <span class="status-key">Loaded File</span>
            <span class="status-val-active">{file_name}</span>
        </div>
        <div class="status-row">
            <span class="status-key">Original Dimensions</span>
            <span class="status-val-active">{original_df.shape[0]:,} rows x {original_df.shape[1]} cols</span>
        </div>
        <div class="status-row">
            <span class="status-key">Current Curated Shape</span>
            <span class="status-val-active">{rows:,} rows x {cols} cols</span>
        </div>
        <div class="status-row">
            <span class="status-key">Identified Duplicates</span>
            <span class="status-val-active">{dups} rows</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.success("Active dataset detected. Select Data Curation, Diagnostic Visualizations, or ML Studio from the sidebar to continue.")
else:
    st.markdown("""
    <div class="status-card">
        <div class="status-header">
            <span class="status-title">Session State</span>
            <span style="color: #F59E0B; font-size: 11px; font-weight: 700; font-family: 'JetBrains Mono', monospace; letter-spacing: 1px; background: rgba(245, 158, 11, 0.12); padding: 4px 10px; border-radius: 6px; border: 1px solid rgba(245, 158, 11, 0.25);">NO DATASET</span>
        </div>
        <div class="status-row">
            <span class="status-key">Current Dataset</span>
            <span class="status-val-empty">None loaded</span>
        </div>
        <div class="status-row">
            <span class="status-key">Status</span>
            <span class="status-val-empty">Awaiting CSV / Excel file upload</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.info("Getting started? Navigate to Data Curation in the sidebar navigation to upload your dataset.")

# Sidebar Branding & Info
st.sidebar.markdown("""
<div style="padding: 10px 0; border-bottom: 1px solid rgba(255,255,255,0.08); margin-bottom: 16px;">
    <div style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 20px; font-weight: 800; color: #F1F3F9;">
        VIZML
    </div>
    <div style="font-size: 11px; color: #64748B; margin-top: 2px; font-family: 'JetBrains Mono', monospace; letter-spacing: 1px;">
        DATA INTELLIGENCE STUDIO
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("Use the navigation menu above to switch between workspace modules.")

if df is not None:
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Active Session File**")
    st.sidebar.markdown(f"File: `{st.session_state.get('current_file_name', 'Unknown')}`")
    st.sidebar.markdown(f"Shape: `{df.shape[0]:,} rows x {df.shape[1]} cols`")