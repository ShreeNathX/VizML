import streamlit as st
from src.styles import inject_css, render_html

st.set_page_config(
    page_title="VizML: Local Data Studio",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

# Topbar brand strip
render_html(
    """
    <div style="display:flex;align-items:center;justify-content:space-between;
                padding:16px 24px;border-radius:999px;margin-top:18px;margin-bottom:30px;
                background:#fff;
                box-shadow:10px 10px 22px rgba(163,160,201,0.55),-10px -10px 22px rgba(255,255,255,0.95);">
      <div style="display:flex;align-items:center;gap:14px;">
        <div class="brand-mark">
          <svg viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2"
               stroke-linecap="round" stroke-linejoin="round">
            <path d="M12 3c3 3.5 5 6.4 5 9.2a5 5 0 0 1-10 0C7 9.4 9 6.5 12 3z"/>
          </svg>
        </div>
        <div>
          <div style="font-family:'Baloo 2',sans-serif;font-size:22px;font-weight:700;
                      color:#211F36;letter-spacing:.2px;">VizML Studio</div>
          <div style="font-size:12px;color:#726E90;font-weight:600;margin-top:1px;">
            Data Intelligence Platform
          </div>
        </div>
      </div>
      <div style="display:flex;align-items:center;gap:14px;">
        <div class="status-chip">
          <span class="status-dot"></span>Running locally · Zero cloud
        </div>
      </div>
    </div>
    """
)

# Hero and workspace telemetry
col_left, col_right = st.columns([7, 5])

with col_left:
    render_html(
        """
        <div class="bp-card">
            <div class="bp-header" style="color:var(--indigo);">Stage 00 · Welcome</div>
            <div class="page-title">VizML Studio</div>
            <div class="page-sub">
                End-to-end data curation, diagnostic visualization, and automated machine learning.
                Upload a messy tabular file, trace it through cleaning, explore its geometry,
                and land on a benchmarked model, all on your local device.
            </div>
        </div>
        """
    )

with col_right:
    df = st.session_state.get("df", None)
    if df is not None:
        file_name = st.session_state.get("current_file_name", "Active Dataset")
        rows, cols = df.shape
        missing = int(df.isna().sum().sum())
        dups = int(df.duplicated().sum())
        status_badge = '<span class="badge badge-emerald">ACTIVE</span>'
        rows_val = f"{rows:,} × {cols}"
        orig_missing = (
            int(st.session_state["original_df"].isna().sum().sum())
            if st.session_state.get("original_df") is not None
            else missing
        )
        if missing == 0 and orig_missing > 0:
            missing_val = f"0 cells (Imputed {orig_missing:,})"
        else:
            missing_val = f"{missing:,} cells"
        dups_val = f"{dups:,}"
    else:
        file_name = "None"
        status_badge = '<span class="badge badge-gold">NO DATASET</span>'
        rows_val = missing_val = dups_val = "Not Loaded"

    render_html(
        f"""
        <div class="bp-card">
            <div style="display:flex;justify-content:space-between;align-items:center;
                        margin-bottom:14px;padding-bottom:10px;
                        border-bottom:1px solid rgba(163,160,201,0.25);">
                <span style="font-family:'Plus Jakarta Sans',sans-serif;font-size:12px;
                             font-weight:800;color:#726E90;letter-spacing:.06em;
                             text-transform:uppercase;">Workspace Session</span>
                {status_badge}
            </div>
            <div class="session-row"><span class="session-label">Source</span>
                <span class="session-val">{file_name}</span></div>
            <div class="session-row"><span class="session-label">Dimensions</span>
                <span class="session-val">{rows_val}</span></div>
            <div class="session-row"><span class="session-label">Missing Cells</span>
                <span class="session-val">{missing_val}</span></div>
            <div class="session-row"><span class="session-label">Duplicate Rows</span>
                <span class="session-val">{dups_val}</span></div>
        </div>
        """
    )

# Phase cards
render_html(
    """
    <div class="section-header" style="margin-top:8px;">
        Core Pipeline and Phases
    </div>
    """
)

PHASES = [
    {
        "color": "var(--coral)",
        "color_l": "var(--coral-l)",
        "color_bg": "var(--coral-bg)",
        "phase": "Phase 01",
        "title": "Data Curation Engine",
        "desc": (
            "Type coercion, missing value imputation, outlier handling, "
            "duplicate removal, and categorical encoding with full undo history."
        ),
        "page": "pages/1_Data_Curation.py",
        "action": "Launch Curation",
        "icon": (
            '<path d="M4 20l6-6M13 4l7 7-8.5 8.5a2 2 0 0 1-2.8 0l-4.2-4.2a2 2 0 0 1 0-2.8L13 4z"/>'
        ),
    },
    {
        "color": "var(--cyan)",
        "color_l": "var(--cyan-l)",
        "color_bg": "var(--cyan-bg)",
        "phase": "Phase 02",
        "title": "Diagnostic Visualizations",
        "desc": (
            "Automated insights, correlation redundancy detection, "
            "3D feature explorers, and custom Plotly builders."
        ),
        "page": "pages/2_Diagnostic_Visualizations.py",
        "action": "Launch Diagnostics",
        "icon": '<circle cx="7" cy="16" r="1.6"/><circle cx="12" cy="9" r="1.6"/><circle cx="17" cy="14" r="1.6"/><circle cx="16" cy="6" r="1.6"/>',
    },
    {
        "color": "var(--lime)",
        "color_l": "var(--lime-l)",
        "color_bg": "var(--lime-bg)",
        "phase": "Phase 03",
        "title": "Machine Learning Studio",
        "desc": (
            "Auto task inference, multi-model CV leaderboards, "
            "confusion and residual diagnostics, and live what-if simulation."
        ),
        "page": "pages/3_ML_Studio.py",
        "action": "Launch ML Studio",
        "icon": '<rect x="7" y="7" width="10" height="10" rx="2"/><path d="M12 2v3M12 19v3M2 12h3M19 12h3"/>',
    },
    {
        "color": "var(--orange)",
        "color_l": "var(--orange-l)",
        "color_bg": "var(--orange-bg)",
        "phase": "Phase 04",
        "title": "Executive Summary",
        "desc": (
            "Consolidated multi-stage telemetry summaries, strategic recommendations, "
            "customizable chart builder, and one-click data/report exports."
        ),
        "page": "pages/4_Executive_Reports.py",
        "action": "Open Executive Summary",
        "icon": '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/>',
    },
]

cols = st.columns(4)
for col, ph in zip(cols, PHASES):
    with col:
        render_html(
            f"""
            <div class="bp-card-sm" style="border-top:4px solid {ph['color']};min-height:190px;">
                <span class="badge" style="color:{ph['color']};background:{ph['color_bg']};
                                           margin-bottom:12px;display:inline-block;">
                    {ph['phase']}
                </span>
                <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
                  <div style="width:36px;height:36px;border-radius:12px;flex:none;
                              background:linear-gradient(145deg,{ph['color_l']},{ph['color']});
                              display:flex;align-items:center;justify-content:center;
                              box-shadow:4px 4px 10px rgba(163,160,201,0.5),-4px -4px 10px rgba(255,255,255,0.9);">
                    <svg viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.2"
                         stroke-linecap="round" stroke-linejoin="round"
                         style="width:16px;height:16px;">{ph['icon']}</svg>
                  </div>
                  <div style="font-family:'Baloo 2',sans-serif;font-size:15px;font-weight:700;
                              color:#211F36;">{ph['title']}</div>
                </div>
                <div style="font-size:12.5px;color:#726E90;line-height:1.5;font-weight:500;">
                    {ph['desc']}
                </div>
            </div>
            """
        )
        st.page_link(ph["page"], label=f"Launch {ph['title']}")

st.markdown("<br>", unsafe_allow_html=True)

# Status alert
if df is not None:
    render_html(
        f"""
        <div class="alert-box alert-success">
            <strong>Dataset Active:</strong> <code>{file_name}</code> loaded
            ({rows:,} rows × {cols} columns).
            Ready for curation, dashboard analytics, ML modeling, and executive reports.
        </div>
        """
    )
else:
    render_html(
        """
        <div class="alert-box alert-info">
            <strong>Awaiting Input:</strong> Navigate to
            <strong>Phase 01: Data Curation</strong> in the sidebar to upload
            a CSV or Excel dataset.
        </div>
        """
    )

# Sidebar
with st.sidebar:
    render_html(
        """
        <div style="padding:14px 4px 18px;margin-bottom:14px;
                    border-bottom:1px solid rgba(163,160,201,0.3);">
            <div style="display:flex;align-items:center;gap:10px;">
              <div class="brand-mark" style="width:36px;height:36px;border-radius:12px;">
                <svg viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.2"
                     style="width:18px;height:18px;">
                  <path d="M12 3c3 3.5 5 6.4 5 9.2a5 5 0 0 1-10 0C7 9.4 9 6.5 12 3z"/>
                </svg>
              </div>
              <div>
                <div style="font-family:'Baloo 2',sans-serif;font-size:18px;
                            font-weight:700;color:#211F36;">VizML Studio</div>
                <div style="font-size:10px;color:#726E90;font-weight:700;
                            letter-spacing:.06em;text-transform:uppercase;">
                  Data Platform
                </div>
              </div>
            </div>
        </div>
        """
    )
    if df is not None:
        try:
            st.caption(f"Active: {df.shape[0]:,} rows × {df.shape[1]} cols")
        except Exception:
            pass
