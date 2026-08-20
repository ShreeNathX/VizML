"""
Phase 04 — Executive Summary Report
Synthesises what was learned across Data Curation, Diagnostics, and ML Studio,
then lets the user build custom charts and download everything.
"""
import io
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.styles import (
    CLAY_COLOR_SEQUENCE,
    COLOR_PALETTES,
    CONTINUOUS_COLOR_SCALES,
    apply_clay_theme,
    inject_css,
    render_html,
    resolve_continuous_scale,
)

st.set_page_config(page_title="VizML: Executive Summary", layout="wide")
inject_css()

render_html(
    """
    <div class="bp-card">
        <div class="bp-header" style="color:var(--orange);">Stage 04 · Summarise</div>
        <div class="page-title">Executive Summary Report</div>
        <div class="page-sub">
            Everything learned across the three phases in one place —
            data quality, visual insights, ML results — plus a chart builder
            and one-click downloads.
        </div>
    </div>
    """
)

df = st.session_state.get("df", None)
if df is None:
    render_html(
        """
        <div class="alert-box alert-warning">
            <strong>No dataset loaded.</strong> Go to Phase 01 and upload a file first.
        </div>
        """
    )
    st.page_link("pages/1_Data_Curation.py", label="Go to Phase 01: Data Curation")
    st.stop()

file_name   = st.session_state.get("current_file_name", "dataset")
original_df = st.session_state.get("original_df", df)
ml_results  = st.session_state.get("ml_results", None)       # pd.DataFrame leaderboard
ml_best     = st.session_state.get("ml_best_name", None)
ml_task     = st.session_state.get("ml_task_result", None)
ml_target   = st.session_state.get("ml_target_result", None)
ml_fi       = st.session_state.get("ml_feature_importance", None)  # pd.DataFrame


# ── helpers ──────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _profile(df_json: str) -> dict:
    """Cache expensive per-column profiling (serialise df via JSON key)."""
    import pandas as pd, numpy as np
    df = pd.read_json(io.StringIO(df_json), orient="split")
    rows, cols_n = df.shape
    total_cells = df.size
    missing = int(df.isna().sum().sum())
    dups = int(df.duplicated().sum())
    mem = df.memory_usage(deep=True).sum() / (1024 * 1024)
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [c for c in df.columns if c not in numeric]

    # dtype / null profile (cheap)
    col_profile = []
    for c in df.columns:
        col_profile.append({
            "Column": c,
            "Type": str(df[c].dtype),
            "Nulls": int(df[c].isna().sum()),
            "Null %": f"{df[c].isna().mean()*100:.1f}%",
            "Unique": int(df[c].nunique()),
        })

    # correlation — sample for large frames to keep it fast
    high_corr = []
    if len(numeric) >= 2:
        sample = df[numeric]
        if len(sample) > 10_000:
            sample = sample.sample(10_000, random_state=42)
        try:
            cm = sample.corr()
            for i in range(len(numeric)):
                for j in range(i + 1, len(numeric)):
                    r = cm.iloc[i, j]
                    if abs(r) >= 0.75:
                        high_corr.append((numeric[i], numeric[j], round(float(r), 3)))
            high_corr.sort(key=lambda x: abs(x[2]), reverse=True)
        except Exception:
            pass

    # numeric describe — sample for large frames
    describe_dict = {}
    if numeric:
        sample_num = df[numeric]
        if len(sample_num) > 50_000:
            sample_num = sample_num.sample(50_000, random_state=42)
        try:
            describe_dict = sample_num.describe().round(4).to_dict()
        except Exception:
            pass

    # purity score
    missing_pct = (missing / max(1, total_cells)) * 100
    dups_pct    = (dups    / max(1, rows))        * 100
    purity = max(0.0, min(100.0, 100.0 - missing_pct * 1.5 - dups_pct * 2.0))

    return {
        "rows": rows, "cols": cols_n, "total_cells": total_cells,
        "missing": missing, "missing_pct": missing_pct,
        "dups": dups, "dups_pct": dups_pct,
        "mem_mb": mem,
        "numeric": numeric, "categorical": categorical,
        "col_profile": col_profile,
        "high_corr": high_corr,
        "describe": describe_dict,
        "purity": purity,
    }


@st.cache_data(show_spinner=False)
def _df_to_json(df: pd.DataFrame) -> str:
    return df.to_json(orient="split", date_format="iso")


def _grade(purity: float) -> tuple[str, str]:
    if purity >= 90: return "A — Optimal",  "#3B5F0B"
    if purity >= 75: return "B — Reliable", "#0B7EA6"
    if purity >= 60: return "C — Marginal", "#9A4A0E"
    return           "D — High Risk",       "#B32357"


def _recs(p: dict) -> list[str]:
    r = []
    if p["missing"]:
        r.append(f"{p['missing']:,} missing cells ({p['missing_pct']:.1f}%) — consider re-running imputation in Phase 01.")
    if p["dups"]:
        r.append(f"{p['dups']:,} duplicate rows ({p['dups_pct']:.1f}%) — remove them in Phase 01 to avoid bias.")
    if p["high_corr"]:
        r.append(f"{len(p['high_corr'])} highly-correlated feature pairs (|r| ≥ 0.75) — consider dropping redundant columns before ML.")
    if len(p["categorical"]) > len(p["numeric"]):
        r.append("Predominantly categorical dataset — ensure encoding is applied before modelling.")
    if not r:
        r.append("Dataset health is optimal. No critical issues found.")
    return r


# ── compute profile (cached) ──────────────────────────────────────────────────
with st.spinner("Profiling dataset…"):
    try:
        _json = _df_to_json(df)
        p = _profile(_json)
    except Exception as e:
        st.error(f"Could not profile dataset: {e}")
        st.stop()

numeric_cols     = p["numeric"]
categorical_cols = p["categorical"]
grade_label, grade_color = _grade(p["purity"])
recommendations  = _recs(p)

# ═════════════════════════════════════════════════════════════════════════════
# PHASE 01 SUMMARY
# ═════════════════════════════════════════════════════════════════════════════
render_html(
    """
    <div class="step-header">
        <span class="step-num coral">01</span>
        <span class="step-title">Data Curation — Health Summary</span>
    </div>
    """
)

kc = st.columns(5)
kc[0].metric("Rows",          f"{p['rows']:,}")
kc[1].metric("Columns",       f"{p['cols']}")
kc[2].metric("Missing Cells", f"{p['missing']:,}", delta=f"{p['missing_pct']:.1f}%", delta_color="inverse")
kc[3].metric("Duplicates",    f"{p['dups']:,}",    delta=f"{p['dups_pct']:.1f}%",    delta_color="inverse")
kc[4].metric("Memory",        f"{p['mem_mb']:.1f} MB")

st.markdown("<br>", unsafe_allow_html=True)
sc1, sc2 = st.columns([2, 3])

with sc1:
    purity_pct = p["purity"]
    render_html(
        f"""
        <div class="bp-card" style="text-align:center;padding:28px 20px;">
            <div style="font-size:12px;font-weight:700;color:var(--ink-soft);
                        letter-spacing:.08em;text-transform:uppercase;margin-bottom:8px;">
                Data Purity Index
            </div>
            <div style="font-family:'Baloo 2',sans-serif;font-size:52px;font-weight:800;
                        color:{grade_color};line-height:1.0;">
                {purity_pct:.0f}<span style="font-size:28px">%</span>
            </div>
            <div style="font-size:14px;font-weight:700;color:{grade_color};margin-top:4px;">
                {grade_label}
            </div>
        </div>
        """
    )

with sc2:
    render_html('<div class="section-header" style="margin-top:0;">Column Profile</div>')
    if p["col_profile"]:
        profile_df = pd.DataFrame(p["col_profile"])
        st.dataframe(profile_df, width='stretch', hide_index=True)
    else:
        st.info("No column data available.")

# Changes from original
st.markdown("<br>", unsafe_allow_html=True)
orig_missing = int(original_df.isna().sum().sum()) if original_df is not None else p["missing"]
orig_dups    = int(original_df.duplicated().sum()) if original_df is not None else p["dups"]

if orig_missing != p["missing"] or orig_dups != p["dups"]:
    render_html('<div class="section-header">Changes Applied During Curation</div>')
    dc1, dc2, dc3 = st.columns(3)
    dc1.metric("Cells Cleaned",      f"{orig_missing - p['missing']:,}", delta="imputed",  delta_color="normal")
    dc2.metric("Duplicates Removed", f"{orig_dups    - p['dups']:,}",    delta="scrubbed", delta_color="normal")
    dc3.metric("Rows After Curation", f"{p['rows']:,}")

# ═════════════════════════════════════════════════════════════════════════════
# PHASE 02 SUMMARY
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("---")
render_html(
    """
    <div class="step-header">
        <span class="step-num cyan">02</span>
        <span class="step-title">Diagnostic Visualizations — Key Insights</span>
    </div>
    """
)

ic1, ic2, ic3, ic4 = st.columns(4)
ic1.metric("Numeric Features",     len(numeric_cols))
ic2.metric("Categorical Features", len(categorical_cols))
ic3.metric("Correlated Pairs",     len(p["high_corr"]), help="Pairs with |r| ≥ 0.75")
if p["describe"] and numeric_cols:
    try:
        first = numeric_cols[0]
        mean_v = p["describe"].get(first, {}).get("mean", None)
        if mean_v is not None:
            ic4.metric(f"Mean ({first})", f"{mean_v:,.3g}")
    except Exception:
        pass

st.markdown("<br>", unsafe_allow_html=True)
vi1, vi2 = st.columns(2)

# Numeric distribution snapshot
with vi1:
    if numeric_cols and p["describe"]:
        try:
            desc_df = pd.DataFrame(p["describe"]).T.reset_index().rename(columns={"index": "Feature"})
            render_html('<div class="section-header" style="margin-top:0;">Numeric Statistics</div>')
            st.dataframe(desc_df, width='stretch', hide_index=True)
        except Exception:
            st.info("No numeric statistics available.")
    else:
        st.info("No numeric columns in the dataset.")

# Correlation summary
with vi2:
    render_html('<div class="section-header" style="margin-top:0;">Collinearity Findings</div>')
    if p["high_corr"]:
        corr_df = pd.DataFrame(p["high_corr"], columns=["Feature A", "Feature B", "r"])
        st.dataframe(corr_df, width='stretch', hide_index=True)
        render_html(
            f'<div class="alert-box alert-warning" style="margin-top:8px;">'
            f'<strong>{len(p["high_corr"])} highly-correlated pairs</strong> detected (|r| ≥ 0.75). '
            f'Consider removing redundant features before training models.</div>'
        )
    else:
        render_html(
            '<div class="alert-box alert-success">'
            'No high-correlation redundancy detected. Feature space looks healthy.</div>'
        )

# ═════════════════════════════════════════════════════════════════════════════
# PHASE 03 SUMMARY (Only displayed if ML Studio was executed)
# ═════════════════════════════════════════════════════════════════════════════
if ml_results is not None:
    st.markdown("---")
    render_html(
        """
        <div class="step-header">
            <span class="step-num lime">03</span>
            <span class="step-title">ML Studio — Model Results</span>
        </div>
        """
    )

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Task",          ml_task    or "—")
    mc2.metric("Target Column", ml_target  or "—")
    mc3.metric("Champion Model", ml_best   or "—")

    perf_col = next((c for c in ml_results.columns if c in ("Test Accuracy", "Test R²", "F1 Score (weighted)")), None)
    if perf_col and ml_best:
        row = ml_results[ml_results["Model"] == ml_best]
        if not row.empty:
            mc4.metric(perf_col, f"{row[perf_col].iloc[0]:.4f}")

    st.markdown("<br>", unsafe_allow_html=True)
    ml1, ml2 = st.columns(2)

    with ml1:
        render_html('<div class="section-header" style="margin-top:0;">Model Leaderboard</div>')
        display_cols = [c for c in ml_results.columns if not c.startswith("_")]
        st.dataframe(ml_results[display_cols], width='stretch', hide_index=True)

    with ml2:
        if ml_fi is not None and not ml_fi.empty:
            render_html('<div class="section-header" style="margin-top:0;">Top Feature Drivers</div>')
            top_fi = ml_fi.head(12).sort_values("Importance", ascending=True)
            fig_fi = px.bar(
                top_fi, x="Importance", y="Feature", orientation="h",
                color="Importance",
                color_continuous_scale=resolve_continuous_scale("Clay Violet"),
            )
            apply_clay_theme(fig_fi, title="Feature Importance")
            fig_fi.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig_fi, width='stretch', theme=None, key="summ_fi")
        elif perf_col:
            render_html('<div class="section-header" style="margin-top:0;">Performance Chart</div>')
            fig_lb = px.bar(
                ml_results[display_cols], x="Model", y=perf_col,
                color="Model", color_discrete_sequence=CLAY_COLOR_SEQUENCE,
            )
            fig_lb.update_layout(showlegend=False)
            apply_clay_theme(fig_lb, title=f"Benchmark — {perf_col}")
            st.plotly_chart(fig_lb, width='stretch', theme=None, key="summ_lb")

# ═════════════════════════════════════════════════════════════════════════════
# STRATEGIC RECOMMENDATIONS
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("---")
render_html(
    """
    <div class="step-header">
        <span class="step-num orange">04</span>
        <span class="step-title">Strategic Recommendations</span>
    </div>
    """
)

for i, rec in enumerate(recommendations, 1):
    render_html(
        f"""
        <div style="display:flex;align-items:flex-start;gap:12px;margin-bottom:10px;
                    padding:12px 16px;border-radius:12px;
                    background:var(--white);
                    box-shadow:4px 4px 10px var(--sh-dark),-4px -4px 10px var(--sh-light);">
            <span class="badge badge-indigo" style="flex-shrink:0;margin-top:2px;">{i:02d}</span>
            <span style="font-size:13px;color:var(--ink-soft);font-weight:500;line-height:1.6;">{rec}</span>
        </div>
        """
    )

# ═════════════════════════════════════════════════════════════════════════════
# CUSTOM CHART BUILDER
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("---")
render_html(
    """
    <div class="step-header">
        <span class="step-num indigo">05</span>
        <span class="step-title">Custom Chart Builder</span>
    </div>
    """
)
render_html(
    "<span style='font-size:13px;color:var(--ink-soft);font-weight:500;'>"
    "Build up to 4 custom charts using any columns from your dataset. "
    "Color palettes are fully customisable.</span>"
)
st.markdown("<br>", unsafe_allow_html=True)

# Cap displayed data for chart builder (performance)
CHART_SAMPLE = 20_000
df_chart = df.sample(CHART_SAMPLE, random_state=42) if len(df) > CHART_SAMPLE else df
if len(df) > CHART_SAMPLE:
    st.caption(f"Chart builder is using a {CHART_SAMPLE:,}-row sample for performance. Download still exports full data.")

all_cols  = df_chart.columns.tolist()
num_cols  = df_chart.select_dtypes(include=[np.number]).columns.tolist()
cat_cols  = [c for c in all_cols if c not in num_cols]

CHART_TYPES = ["Bar Chart", "Line Chart", "Scatter Plot", "Histogram", "Box Plot", "Violin Plot", "Donut Chart"]

num_widgets = st.slider("Number of charts", 1, 4, 2, key="cb_n")

chart_cfgs = []
for j in range(num_widgets):
    with st.expander(f"Chart {j + 1}", expanded=(j == 0)):
        ca, cb, cc = st.columns(3)
        with ca:
            c_title  = st.text_input("Title",    value=f"Chart {j+1}",       key=f"cb_t_{j}")
            c_type   = st.selectbox("Chart Type", CHART_TYPES, index=j % len(CHART_TYPES), key=f"cb_type_{j}")
        with cb:
            c_x = st.selectbox("X-Axis / Category", all_cols, index=0,          key=f"cb_x_{j}")
            y_options = ["(count)"] + num_cols
            c_y = st.selectbox("Y-Axis / Value",    y_options, index=min(1, len(y_options)-1), key=f"cb_y_{j}")
        with cc:
            hue_opts = ["(none)"] + cat_cols
            c_hue = st.selectbox("Color Grouping", hue_opts, index=0, key=f"cb_hue_{j}")
            c_pal = st.selectbox("Palette",         list(COLOR_PALETTES.keys()), index=j % len(COLOR_PALETTES), key=f"cb_pal_{j}")

        chart_cfgs.append({"title": c_title, "type": c_type, "x": c_x, "y": c_y, "hue": c_hue, "palette": c_pal})

st.markdown("<br>", unsafe_allow_html=True)
render_html('<div class="section-header">Chart Preview</div>')

# Render charts in 2-column grid
for i in range(0, len(chart_cfgs), 2):
    cols_grid = st.columns(2)
    for offset in range(2):
        idx = i + offset
        if idx >= len(chart_cfgs):
            break
        cfg = chart_cfgs[idx]
        with cols_grid[offset]:
            try:
                seq   = COLOR_PALETTES.get(cfg["palette"], CLAY_COLOR_SEQUENCE)
                hue   = None if cfg["hue"] == "(none)" else cfg["hue"]
                y_val = None if cfg["y"] == "(count)" else cfg["y"]
                fig   = None

                if cfg["type"] == "Bar Chart":
                    if y_val is None:
                        grp = df_chart[cfg["x"]].value_counts().reset_index()
                        grp.columns = [cfg["x"], "Count"]
                        fig = px.bar(grp, x=cfg["x"], y="Count", color_discrete_sequence=seq)
                    else:
                        fig = px.bar(df_chart, x=cfg["x"], y=y_val, color=hue, color_discrete_sequence=seq)

                elif cfg["type"] == "Line Chart":
                    if y_val is None:
                        grp = df_chart[cfg["x"]].value_counts().sort_index().reset_index()
                        grp.columns = [cfg["x"], "Count"]
                        fig = px.line(grp, x=cfg["x"], y="Count", color_discrete_sequence=seq)
                    else:
                        fig = px.line(df_chart.sort_values(cfg["x"]), x=cfg["x"], y=y_val, color=hue, color_discrete_sequence=seq)

                elif cfg["type"] == "Scatter Plot":
                    if y_val is None:
                        st.info("Scatter needs a numeric Y-axis. Choose a value column.")
                    else:
                        fig = px.scatter(df_chart, x=cfg["x"], y=y_val, color=hue, color_discrete_sequence=seq)
                        try:
                            fig.update_traces(opacity=0.7)
                        except Exception:
                            pass

                elif cfg["type"] == "Histogram":
                    fig = px.histogram(df_chart, x=cfg["x"], color=hue, color_discrete_sequence=seq, nbins=40)

                elif cfg["type"] == "Box Plot":
                    x_arg = cfg["x"] if cfg["x"] in cat_cols else None
                    y_arg = y_val if y_val else (cfg["x"] if cfg["x"] in num_cols else (num_cols[0] if num_cols else None))
                    if y_arg:
                        fig = px.box(df_chart, x=x_arg, y=y_arg, color=hue, color_discrete_sequence=seq)
                    else:
                        st.info("Box Plot needs at least one numeric column.")

                elif cfg["type"] == "Violin Plot":
                    x_arg = cfg["x"] if cfg["x"] in cat_cols else None
                    y_arg = y_val if y_val else (cfg["x"] if cfg["x"] in num_cols else (num_cols[0] if num_cols else None))
                    if y_arg:
                        fig = px.violin(df_chart, x=x_arg, y=y_arg, color=hue, box=True, color_discrete_sequence=seq)
                    else:
                        st.info("Violin Plot needs at least one numeric column.")

                elif cfg["type"] == "Donut Chart":
                    if y_val:
                        fig = px.pie(df_chart, names=cfg["x"], values=y_val, hole=0.45, color_discrete_sequence=seq)
                    else:
                        fig = px.pie(df_chart, names=cfg["x"], hole=0.45, color_discrete_sequence=seq)

                if fig:
                    apply_clay_theme(fig, title=cfg["title"])
                    st.plotly_chart(fig, width='stretch', theme=None, key=f"cb_chart_{idx}")

            except Exception as e:
                st.warning(f"Could not render Chart {idx + 1}: {e}")

# ═════════════════════════════════════════════════════════════════════════════
# DOWNLOADS
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("---")
render_html(
    """
    <div class="step-header">
        <span class="step-num emerald">06</span>
        <span class="step-title">Downloads</span>
    </div>
    """
)

if ml_results is not None:
    dc1, dc2, dc3 = st.columns(3)
else:
    dc1, dc3 = st.columns(2)
    dc2 = None

# 1. Curated dataset CSV
with dc1:
    render_html('<div class="section-header" style="margin-top:0;font-size:13px;">Curated Dataset</div>')
    try:
        buf_csv = io.StringIO()
        df.to_csv(buf_csv, index=False)
        st.download_button(
            "Download Dataset (.csv)",
            data=buf_csv.getvalue().encode(),
            file_name=f"vizml_curated_{file_name.replace('.', '_')}.csv",
            mime="text/csv",
            width='stretch',
            type="primary",
        )
        st.caption(f"{p['rows']:,} rows × {p['cols']} columns · full dataset")
    except Exception as e:
        st.error(f"CSV export failed: {e}")

# 2. ML leaderboard CSV (only if ML was run)
if dc2 is not None and ml_results is not None:
    with dc2:
        render_html('<div class="section-header" style="margin-top:0;font-size:13px;">ML Leaderboard</div>')
        try:
            display_cols = [c for c in ml_results.columns if not c.startswith("_")]
            buf_ml = io.StringIO()
            ml_results[display_cols].to_csv(buf_ml, index=False)
            st.download_button(
                "Download Leaderboard (.csv)",
                data=buf_ml.getvalue().encode(),
                file_name="vizml_ml_leaderboard.csv",
                mime="text/csv",
                width='stretch',
            )
            st.caption(f"{len(ml_results)} models · task: {ml_task}")
        except Exception as e:
            st.error(f"ML export failed: {e}")

# 3. Summary report HTML (dynamically adapts based on whether ML was executed)
with dc3:
    render_html('<div class="section-header" style="margin-top:0;font-size:13px;">Summary Report (HTML)</div>')
    try:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        rec_html = "".join(f"<li>{r}</li>" for r in recommendations)

        # Build Phase 03 HTML section only if ML was run
        if ml_results is not None:
            display_ml_cols = [c for c in ml_results.columns if not c.startswith("_")]
            ml_table = ml_results[display_ml_cols].to_html(border=0, index=False, classes="bp-table")
            ml_section_html = f"""
<h2>Phase 03 — Machine Learning Leaderboard</h2>
<p>Champion Model: <strong>{ml_best or 'N/A'}</strong> | Task: <strong>{ml_task or 'N/A'}</strong> | Target: <strong>{ml_target or 'N/A'}</strong></p>
{ml_table}
"""
        else:
            ml_section_html = ""

        corr_rows = "".join(
            f"<tr><td>{a}</td><td>{b}</td><td><strong>{r:+.3f}</strong></td></tr>"
            for a, b, r in p["high_corr"][:20]
        ) if p["high_corr"] else "<tr><td colspan='3'>No high-correlation pairs detected.</td></tr>"

        html_report = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>VizML Summary Report — {file_name}</title>
<link href="https://fonts.googleapis.com/css2?family=Baloo+2:wght@600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
body {{ font-family:'Plus Jakarta Sans',sans-serif; background:#F5F3FC; color:#211F36; margin:0; padding:32px; }}
h1 {{ font-family:'Baloo 2',sans-serif; font-size:28px; color:#211F36; margin-bottom:4px; }}
h2 {{ font-family:'Baloo 2',sans-serif; font-size:18px; color:#6C5CE7; border-bottom:2px solid #EDE9FE; padding-bottom:6px; margin-top:32px; }}
.meta {{ font-size:12px; color:#726E90; margin-bottom:32px; }}
.kpi-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:14px; margin:16px 0; }}
.kpi {{ background:#fff; border-radius:14px; padding:16px 20px; box-shadow:4px 4px 12px rgba(163,160,201,.4); text-align:center; }}
.kpi .val {{ font-family:'Baloo 2',sans-serif; font-size:24px; font-weight:800; color:#6C5CE7; }}
.kpi .lbl {{ font-size:11px; color:#726E90; font-weight:600; text-transform:uppercase; letter-spacing:.06em; margin-top:4px; }}
.bp-table {{ width:100%; border-collapse:collapse; font-size:12px; margin:12px 0; }}
.bp-table th {{ background:#EDE9FE; color:#6C5CE7; padding:8px 12px; text-align:left; font-weight:700; }}
.bp-table td {{ padding:7px 12px; border-bottom:1px solid #EDE9FE; }}
.grade {{ display:inline-block; padding:4px 14px; border-radius:999px; font-weight:700; font-size:14px; color:#fff; }}
ul.recs li {{ margin-bottom:8px; font-size:13px; }}
</style>
</head>
<body>
<h1>VizML Executive Summary Report</h1>
<p class="meta">
  Dataset: <strong>{file_name}</strong> &nbsp;|&nbsp;
  Generated: {ts} &nbsp;|&nbsp;
  Tool: VizML Studio
</p>

<h2>Phase 01 — Data Curation</h2>
<div class="kpi-grid">
  <div class="kpi"><div class="val">{p["rows"]:,}</div><div class="lbl">Rows</div></div>
  <div class="kpi"><div class="val">{p["cols"]}</div><div class="lbl">Columns</div></div>
  <div class="kpi"><div class="val">{p["missing"]:,}</div><div class="lbl">Missing Cells</div></div>
  <div class="kpi"><div class="val">{p["dups"]:,}</div><div class="lbl">Duplicates</div></div>
  <div class="kpi"><div class="val">{p["purity"]:.0f}%</div><div class="lbl">Purity Score</div></div>
  <div class="kpi"><div class="val" style="font-size:16px;color:{grade_color};">{grade_label}</div><div class="lbl">Grade</div></div>
</div>

<table class="bp-table">
<thead><tr><th>Column</th><th>Type</th><th>Nulls</th><th>Null %</th><th>Unique</th></tr></thead>
<tbody>{"".join(f"<tr><td>{r['Column']}</td><td>{r['Type']}</td><td>{r['Nulls']}</td><td>{r['Null %']}</td><td>{r['Unique']}</td></tr>" for r in p["col_profile"])}</tbody>
</table>

<h2>Phase 02 — Diagnostic Insights</h2>
<p><strong>Numeric features:</strong> {len(numeric_cols)} &nbsp;|&nbsp; <strong>Categorical features:</strong> {len(categorical_cols)}</p>
<h3 style="font-size:14px;color:#211F36;">High-Correlation Pairs (|r| ≥ 0.75)</h3>
<table class="bp-table">
<thead><tr><th>Feature A</th><th>Feature B</th><th>r</th></tr></thead>
<tbody>{corr_rows}</tbody>
</table>

{ml_section_html}

<h2>Strategic Recommendations</h2>
<ul class="recs">{rec_html}</ul>

</body>
</html>"""

        st.download_button(
            "Download Report (.html)",
            data=html_report.encode(),
            file_name=f"vizml_summary_{file_name.replace('.', '_')}.html",
            mime="text/html",
            width='stretch',
        )
        st.caption("Self-contained HTML — open in any browser")
    except Exception as e:
        st.error(f"Report generation failed: {e}")
