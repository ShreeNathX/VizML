import streamlit as st

from src.styles import inject_css

st.set_page_config(
    page_title="VizML: Data Intelligence Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

col_left, col_right = st.columns([7, 5])

with col_left:
    st.markdown(
        """
    <div class="glass-card">
        <span class="eyebrow">Data Intelligence Platform</span>
        <div class="page-title">VizML Studio</div>
        <div class="page-sub">
            End-to-end data curation, diagnostic visualization, and automated machine learning environment.
            Upload messy tabular files, explore high-dimensional geometry, and build predictive pipelines.
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col_right:
    df = st.session_state.get("df", None)
    if df is not None:
        file_name = st.session_state.get("current_file_name", "Active Dataset")
        rows, cols = df.shape
        missing = int(df.isna().sum().sum())
        dups = int(df.duplicated().sum())
        status_badge = '<span class="badge badge-emerald">ACTIVE</span>'
        rows_val = f"{rows:,} x {cols}"
        orig_missing = (
            int(st.session_state["original_df"].isna().sum().sum())
            if st.session_state.get("original_df") is not None
            else missing
        )
        if missing == 0 and orig_missing > 0:
            missing_val = f"0 cells (Fixed {orig_missing:,})"
        else:
            missing_val = f"{missing:,} cells"
        dups_val = str(dups)
    else:
        file_name = "None"
        status_badge = '<span class="badge badge-amber">NO DATASET</span>'
        rows_val = missing_val = dups_val = "—"

    st.markdown(
        f"""
    <div class="glass-card">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;padding-bottom:12px;border-bottom:1px solid rgba(99,102,241,0.10);">
            <span style="font-family:'Plus Jakarta Sans',sans-serif;font-size:14px;font-weight:700;color:#111827;">Workspace Session</span>
            {status_badge}
        </div>
        <div class="session-row"><span class="session-label">File</span><span class="session-val">{file_name}</span></div>
        <div class="session-row"><span class="session-label">Dimensions</span><span class="session-val">{rows_val}</span></div>
        <div class="session-row"><span class="session-label">Missing Cells</span><span class="session-val">{missing_val}</span></div>
        <div class="session-row"><span class="session-label">Duplicate Rows</span><span class="session-val">{dups_val}</span></div>
    </div>
    """,
        unsafe_allow_html=True,
    )

c1, c2, c3 = st.columns(3)
cards = [
    (
        "badge-indigo",
        "Phase 01",
        "Data Curation Engine",
        "Type coercion, missing value imputation, outlier handling, duplicate removal, and categorical encoding with full undo history.",
    ),
    (
        "badge-violet",
        "Phase 02",
        "Diagnostic Visualizations",
        "Automated insights, correlation redundancy detection, 3D feature explorers, and custom Plotly visual builders.",
    ),
    (
        "badge-emerald",
        "Phase 03",
        "Machine Learning Studio",
        "Auto task inference, multi-model CV leaderboards, confusion & residual diagnostics, and live What-If prediction.",
    ),
]
for col, (badge_cls, phase, title, desc) in zip([c1, c2, c3], cards):
    with col:
        st.markdown(
            f"""
        <div class="glass-card-sm">
            <span class="badge {badge_cls}">{phase}</span>
            <div style="font-family:'Plus Jakarta Sans',sans-serif;font-size:15px;font-weight:700;color:#111827;margin-top:12px;margin-bottom:7px;">{title}</div>
            <div style="font-size:13px;color:#4B5563;line-height:1.55;">{desc}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)
if df is not None:
    st.success("Dataset loaded. Navigate to any module from the sidebar.")
else:
    st.info("Select **Data Curation** from the sidebar to upload your dataset.")

st.sidebar.markdown(
    """
<div style="padding:10px 0 14px;border-bottom:1px solid rgba(99,102,241,0.12);margin-bottom:12px;">
    <div style="font-family:'Plus Jakarta Sans',sans-serif;font-size:18px;font-weight:800;color:#111827;">VizML Studio</div>
    <div style="font-size:10px;color:#4B5563;font-family:'JetBrains Mono',monospace;letter-spacing:1.5px;margin-top:2px;">DATA ENGINE</div>
</div>
""",
    unsafe_allow_html=True,
)

if df is not None:
    try:
        st.sidebar.caption(f"Shape: `{df.shape[0]:,} rows x {df.shape[1]} cols`")
    except Exception:
        pass
