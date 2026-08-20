from typing import Any, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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

try:
    import statsmodels  # type: ignore

    has_statsmodels = True
except ImportError:
    has_statsmodels = False

st.set_page_config(page_title="VizML: Diagnostic Dashboard", layout="wide")
inject_css()

if "df" not in st.session_state or st.session_state["df"] is None:
    render_html(
        """
        <div class="bp-card">
            <div class="bp-header" style="color:var(--cyan);">Stage 03 · Explore</div>
            <div class="page-title">Diagnostic Visualizations</div>
            <div class="page-sub">Awaiting dataset input. Load a dataset in Data Curation to unlock real-time telemetry and multidimensional analytics.</div>
        </div>
        """
    )
    st.page_link("pages/1_Data_Curation.py", label="Launch Phase 01: Data Curation Engine")
    st.stop()

df_source = st.session_state["df"]
file_name = st.session_state.get("current_file_name", "Active Dataset")

# Sampling cap — keeps charts responsive for large datasets on Streamlit Cloud.
# All KPI metrics are computed on the full dataset; only chart data is sampled.
CHART_SAMPLE_LIMIT = 20_000
_is_large = len(df_source) > CHART_SAMPLE_LIMIT

render_html(
    """
    <div class="bp-card">
        <div class="bp-header" style="color:var(--cyan);">Stage 02 · Explore</div>
        <div class="page-title">Diagnostic Analytics Dashboard</div>
        <div class="page-sub">
            Real-time KPI telemetry, cohort filtering, correlation diagnostics,
            3D spatial explorers, and a custom chart builder with full color control.
        </div>
    </div>
    """
)

if _is_large:
    st.info(
        f"Large dataset detected ({len(df_source):,} rows). "
        f"Charts will use a representative {CHART_SAMPLE_LIMIT:,}-row sample for speed. "
        f"KPI metrics always reflect the full dataset."
    )

# -------------------------------------------------------------
# 1. Interactive Slice-and-Dice Filtering Studio
# -------------------------------------------------------------
numeric_all = df_source.select_dtypes(include=[np.number]).columns.tolist()
categorical_all = [c for c in df_source.columns if c not in numeric_all]

with st.expander("Interactive Slice-and-Dice Filtering Studio (Cohort Drilldown)", expanded=False):
    render_html("<span style='font-size:12.5px;color:var(--ink-soft);font-weight:500;'>Apply cross-column filters to isolate cohorts. All KPI metrics, distributions, and charts will reactively recalculate.</span>")
    f_cols = st.columns(min(4, max(1, len(df_source.columns))))
    filtered_df = df_source.copy()
    filter_active = False

    cat_filter_cols = [c for c in categorical_all if df_source[c].nunique() <= 50][:3]
    num_filter_cols = numeric_all[:2]

    for idx, c in enumerate(cat_filter_cols):
        with f_cols[idx % len(f_cols)]:
            opts = sorted(list(df_source[c].dropna().unique().astype(str)))
            selected_vals = st.multiselect(f"Filter `{c}`", opts, default=[], key=f"f_cat_{c}")
            if selected_vals:
                filtered_df = filtered_df[filtered_df[c].astype(str).isin(selected_vals)]
                filter_active = True

    for idx, c in enumerate(num_filter_cols):
        with f_cols[(len(cat_filter_cols) + idx) % len(f_cols)]:
            c_min = float(df_source[c].min()) if not df_source[c].dropna().empty else 0.0
            c_max = float(df_source[c].max()) if not df_source[c].dropna().empty else 1.0
            if c_min < c_max:
                sel_range = st.slider(f"Range `{c}`", c_min, c_max, (c_min, c_max), key=f"f_num_{c}")
                if sel_range != (c_min, c_max):
                    filtered_df = filtered_df[(filtered_df[c] >= sel_range[0]) & (filtered_df[c] <= sel_range[1])]
                    filter_active = True

    if filter_active:
        render_html(
            f'<div class="alert-box alert-warning">Filtered cohort active: <strong>{filtered_df.shape[0]:,}</strong> of {df_source.shape[0]:,} rows ({filtered_df.shape[0]/max(1, df_source.shape[0])*100:.1f}%)</div>'
        )

df_full = filtered_df  # full data for KPI metrics
if df_full.empty:
    st.warning("Active filters resulted in an empty dataset. Please adjust filter constraints.")
    st.stop()

# Chart-safe sample — avoids Streamlit Cloud memory/time limits
df = (
    df_full.sample(CHART_SAMPLE_LIMIT, random_state=42)
    if len(df_full) > CHART_SAMPLE_LIMIT
    else df_full
)

numeric_cols = df_full.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = [c for c in df_full.columns if c not in numeric_cols]

# -------------------------------------------------------------
# 2. Executive KPI Telemetry Deck
# -------------------------------------------------------------
render_html('<div class="step-header"><span class="step-num cyan">01</span><span class="step-title">Executive KPI Telemetry Deck</span></div>')

# KPIs computed on full (unsampled) data
total_cells = df_full.size
missing_cells = int(df_full.isna().sum().sum())
completeness_pct = ((total_cells - missing_cells) / max(1, total_cells)) * 100
dups_count = int(df_full.duplicated().sum())
dup_pct = (dups_count / max(1, len(df_full))) * 100
mem_mb = df_full.memory_usage(deep=True).sum() / (1024 * 1024)

purity_score = max(0.0, min(100.0, 100.0 - (missing_cells / max(1, total_cells) * 60) - (dup_pct * 40)))

k1, k2, k3, k4, k5 = st.columns(5)

with k1:
    render_html(
        f"""
        <div class="bp-display">
            <div class="bp-expr">Total Records</div>
            <div class="bp-value">{df.shape[0]:,}</div>
        </div>
        """
    )
with k2:
    render_html(
        f"""
        <div class="bp-display">
            <div class="bp-expr">Dimensions</div>
            <div class="bp-value">{df.shape[1]} <span style="font-size:13px;color:var(--ink-soft);">cols</span></div>
        </div>
        """
    )
with k3:
    badge_comp = "var(--lime)" if completeness_pct >= 95 else ("var(--orange)" if completeness_pct >= 80 else "var(--coral)")
    render_html(
        f"""
        <div class="bp-display">
            <div class="bp-expr">Completeness</div>
            <div class="bp-value" style="color:{badge_comp};">{completeness_pct:.1f}%</div>
        </div>
        """
    )
with k4:
    badge_purity = "var(--lime)" if purity_score >= 90 else ("var(--orange)" if purity_score >= 70 else "var(--coral)")
    render_html(
        f"""
        <div class="bp-display">
            <div class="bp-expr">Purity Index</div>
            <div class="bp-value" style="color:{badge_purity};">{purity_score:.1f}<span style="font-size:13px;color:var(--ink-soft);">/100</span></div>
        </div>
        """
    )
with k5:
    render_html(
        f"""
        <div class="bp-display">
            <div class="bp-expr">Footprint</div>
            <div class="bp-value">{mem_mb:.2f} <span style="font-size:13px;color:var(--ink-soft);">MB</span></div>
        </div>
        """
    )

# -------------------------------------------------------------
# 3. Automated Executive Strategic Insights Deck
# -------------------------------------------------------------
render_html('<div class="step-header"><span class="step-num indigo">02</span><span class="step-title">Automated Strategic Insights Deck</span></div>')

# Palette customization for automated deck
with st.expander("Deck Theme & Color Palette Controls", expanded=False):
    dc1, dc2 = st.columns(2)
    with dc1:
        auto_palette_name = st.selectbox("Categorical Palette", list(COLOR_PALETTES.keys()), index=0, key="auto_deck_palette")
        auto_palette = COLOR_PALETTES[auto_palette_name]
    with dc2:
        auto_scale_name = st.selectbox("Continuous Gradient Scale", CONTINUOUS_COLOR_SCALES, index=0, key="auto_deck_scale")
        auto_scale = resolve_continuous_scale(auto_scale_name)

auto_plots: list = []
auto_errors: list = []

# A. Categorical Proportion Donut
try:
    donut_cats = [c for c in categorical_cols if 2 <= df[c].nunique() <= 8]
    if donut_cats:
        fig_donut = px.pie(
            df,
            names=donut_cats[0],
            hole=0.45,
            color_discrete_sequence=auto_palette,
        )
        apply_clay_theme(fig_donut, title=f"Proportion Breakdown: {donut_cats[0]}")
        auto_plots.append(("donut", fig_donut))
except Exception as _e:
    auto_errors.append(f"Proportion chart: {_e}")

# B. Geospatial Distribution
try:
    lat_cols = [c for c in df.columns if "lat" in c.lower()]
    lon_cols = [c for c in df.columns if any(k in c.lower() for k in ["lon", "lng"])]
    country_cols = [c for c in df.columns if any(k in c.lower() for k in ["country", "nation", "state", "iso"])]
    if lat_cols and lon_cols:
        fig_geo = px.scatter_geo(
            df,
            lat=lat_cols[0],
            lon=lon_cols[0],
            color=numeric_cols[0] if numeric_cols else None,
            color_continuous_scale=auto_scale,
        )
        apply_clay_theme(fig_geo, title=f"Geospatial Coordinates: {lat_cols[0]}:{lon_cols[0]}")
        auto_plots.append(("geo", fig_geo))
    elif country_cols:
        gdf = df[country_cols[0]].dropna().value_counts().reset_index()
        gdf.columns = [country_cols[0], "Count"]
        lm = "country names" if "state" not in country_cols[0].lower() else "USA-states"
        fig_geo = px.choropleth(
            gdf,
            locations=country_cols[0],
            locationmode=lm,
            color="Count",
            color_continuous_scale=auto_scale,
        )
        apply_clay_theme(fig_geo, title=f"Geographical Density: {country_cols[0]}")
        auto_plots.append(("geo", fig_geo))
except Exception as _e:
    auto_errors.append(f"Geospatial chart: {_e}")

# C. Time-Series Trendline
try:
    dt_cols = df.select_dtypes(include=[np.datetime64]).columns.tolist()
    if not dt_cols:
        pot = [c for c in categorical_cols if any(k in c.lower() for k in ["date", "time", "year", "month"])]
        if pot:
            tmp = pd.to_datetime(df[pot[0]], errors="coerce", format="mixed")
            if tmp.notna().sum() > 0.5 * len(df):
                df_temp = df.copy()
                df_temp[pot[0]] = tmp
                dt_cols = [pot[0]]
                df = df_temp
    if dt_cols and numeric_cols:
        ts = df[[dt_cols[0], numeric_cols[0]]].dropna().sort_values(dt_cols[0])
        fig_ts = px.line(ts, x=dt_cols[0], y=numeric_cols[0], color_discrete_sequence=[auto_palette[0]])
        apply_clay_theme(fig_ts, title=f"Temporal Trendline: {numeric_cols[0]}")
        auto_plots.append(("ts", fig_ts))
except Exception as _e:
    auto_errors.append(f"Time-series chart: {_e}")

# D. Numeric Distribution Histogram
try:
    if len(auto_plots) < 4 and numeric_cols:
        fig_hist = px.histogram(
            df,
            x=numeric_cols[0],
            marginal="box",
            color_discrete_sequence=[auto_palette[0]],
        )
        try:
            fig_hist.update_traces(opacity=0.85)
        except Exception:
            pass
        apply_clay_theme(fig_hist, title=f"Primary Distribution: {numeric_cols[0]}")
        auto_plots.append(("hist", fig_hist))
except Exception as _e:
    auto_errors.append(f"Histogram: {_e}")

# E. Bivariate Relationship / Scatter
try:
    if len(auto_plots) < 4 and len(numeric_cols) >= 2:
        use_tl = has_statsmodels and df.shape[0] > 2 and df[numeric_cols[:2]].isna().sum().sum() == 0
        fig_scat = px.scatter(
            df,
            x=numeric_cols[0],
            y=numeric_cols[1],
            trendline="ols" if use_tl else None,
            color_discrete_sequence=[auto_palette[1 % len(auto_palette)]],
        )
        try:
            fig_scat.update_traces(opacity=0.75)
        except Exception:
            pass
        apply_clay_theme(fig_scat, title=f"Bivariate Correlation: {numeric_cols[0]} vs {numeric_cols[1]}")
        auto_plots.append(("scatter", fig_scat))
except Exception as _e:
    auto_errors.append(f"Bivariate scatter: {_e}")

if not auto_plots and not auto_errors:
    st.info("Insufficient features to generate automated strategic insights.")
else:
    for i in range(0, len(auto_plots), 2):
        cl, cr = st.columns(2)
        with cl:
            if i < len(auto_plots):
                try:
                    st.plotly_chart(auto_plots[i][1], theme=None, key=f"ap_{auto_plots[i][0]}_{i}")
                except Exception as e:
                    st.warning(f"Chart render warning: {e}")
        with cr:
            if i + 1 < len(auto_plots):
                try:
                    st.plotly_chart(auto_plots[i+1][1], theme=None, key=f"ap_{auto_plots[i+1][0]}_{i+1}")
                except Exception as e:
                    st.warning(f"Chart render warning: {e}")

# -------------------------------------------------------------
# 4. Multidimensional Diagnostic Suite
# -------------------------------------------------------------
render_html('<div class="step-header"><span class="step-num orange">03</span><span class="step-title">Multidimensional Diagnostic Suite</span></div>')

if len(numeric_cols) < 2:
    st.info("At least 2 numeric columns are required to run the Multidimensional Diagnostic Suite.")
else:
    tab_corr, tab_scatter, tab_3d = st.tabs([
        "Correlation and Redundancy Matrix",
        "High-Dimensional Scatter Matrix",
        "3D Spatial Projection Explorer",
    ])

    with tab_corr:
        cl, cr = st.columns([7, 5])
        with cl:
            sel_corr = st.multiselect("Active Correlation Columns:", numeric_cols, default=numeric_cols, key="corr_cols")
        with cr:
            c1, c2, c3 = st.columns(3)
            with c1:
                corr_method = st.radio(
                    "Method", ["pearson", "spearman", "kendall"], index=0, horizontal=True, key="corr_method"
                )
            with c2:
                threshold = st.slider("Redundancy Alert |r|", 0.50, 0.99, 0.80, 0.05, key="corr_thresh")
            with c3:
                corr_scale_choice = st.selectbox(
                    "Heatmap Gradient",
                    ["RdBu_r", "Clay Violet", "Viridis", "Plasma", "Warm Coral", "Cool Cyan", "Spectral"],
                    index=0,
                    key="corr_scale",
                )

        if len(sel_corr) < 2:
            st.warning("Select at least 2 columns to calculate correlation.")
        else:
            try:
                # Sample for large datasets to keep correlation fast
                _corr_src = df[sel_corr].dropna()
                if len(_corr_src) > 10_000:
                    _corr_src = _corr_src.sample(10_000, random_state=42)
                corr = _corr_src.corr(method=corr_method)
                pairs = [
                    (sel_corr[i], sel_corr[j], corr.loc[sel_corr[i], sel_corr[j]])
                    for i in range(len(sel_corr))
                    for j in range(i + 1, len(sel_corr))
                    if abs(corr.loc[sel_corr[i], sel_corr[j]]) >= threshold
                ]
                pairs.sort(key=lambda x: abs(x[2]), reverse=True)
                
                cp, ci = st.columns([7, 5])
                with cp:
                    fig_h = px.imshow(
                        corr,
                        text_auto=".2f" if len(sel_corr) <= 15 else False,
                        aspect="auto",
                        color_continuous_scale=resolve_continuous_scale(corr_scale_choice),
                        color_continuous_midpoint=0 if corr_scale_choice in ["RdBu_r", "Spectral"] else None,
                        range_color=[-1, 1] if corr_scale_choice in ["RdBu_r", "Spectral"] else None,
                    )
                    apply_clay_theme(fig_h, title=f"Correlation Matrix ({corr_method.capitalize()})")
                    st.plotly_chart(fig_h, theme=None)
                with ci:
                    render_html('<div class="section-header" style="margin-top:0;font-size:14px;"><span class="badge badge-violet">Collinearity</span> Redundancy Report</div>')
                    if pairs:
                        render_html(
                            f'<div class="alert-box alert-warning">Detected <strong>{len(pairs)}</strong> collinear pair(s) with |r| >= <strong>{threshold:.2f}</strong>. Collinear features can destabilize linear models and inflate variance.</div>'
                        )
                        for a, b, r in pairs:
                            st.markdown(f"- `{a}` <-> `{b}`: **{r:+.3f}**")
                    else:
                        render_html(
                            f'<div class="alert-box alert-success">Zero collinear redundancy detected above threshold |r| = {threshold:.2f}. Feature set is orthogonal.</div>'
                        )
            except Exception as err:
                st.error(f"Correlation calculation error: {err}")

    with tab_scatter:
        d1, d2, d3 = st.columns([5, 4, 3])
        with d1:
            dims = st.multiselect(
                "Matrix Dimensions (max 6):", numeric_cols, default=numeric_cols[:min(4, len(numeric_cols))], key="scatter_dims"
            )
        with d2:
            color_v = st.selectbox("Color Classification Hue:", ["None"] + df.columns.tolist(), key="scatter_color")
        with d3:
            sm_palette = st.selectbox("Color Palette", list(COLOR_PALETTES.keys()), index=0, key="sm_palette")

        if len(dims) < 2:
            st.info("Select at least 2 dimensions for the scatter matrix.")
        else:
            try:
                is_num_hue = color_v != "None" and color_v in numeric_cols
                _sm_src = df.dropna(subset=dims)
                if len(_sm_src) > 5_000:
                    _sm_src = _sm_src.sample(5_000, random_state=42)
                fig_m = px.scatter_matrix(
                    _sm_src,
                    dimensions=dims[:6],  # cap at 6 dims to avoid visual overload
                    color=None if color_v == "None" else color_v,
                    color_discrete_sequence=COLOR_PALETTES[sm_palette] if not is_num_hue else None,
                    color_continuous_scale=resolve_continuous_scale("Clay Violet") if is_num_hue else None,
                )
                try:
                    fig_m.update_traces(opacity=0.7)
                except Exception:
                    pass
                apply_clay_theme(fig_m, title="Multi-Attribute Scatter Matrix")
                st.plotly_chart(fig_m, theme=None)
            except Exception as err:
                st.error(f"Scatter matrix error: {err}")

    with tab_3d:
        ax, ex = st.columns([7, 5])
        with ax:
            xc, yc, zc = st.columns(3)
            with xc:
                x3 = st.selectbox("X Spatial Axis", numeric_cols, index=0, key="3dx")
            with yc:
                y3 = st.selectbox("Y Spatial Axis", numeric_cols, index=min(1, len(numeric_cols) - 1), key="3dy")
            with zc:
                z3 = st.selectbox("Z Spatial Axis", numeric_cols, index=min(2, len(numeric_cols) - 1), key="3dz")
        with ex:
            cc, sc, tc = st.columns(3)
            with cc:
                col3 = st.selectbox("Color Hue", ["None"] + df.columns.tolist(), key="3dcol")
            with sc:
                sz3 = st.selectbox("Marker Size", ["None"] + numeric_cols, key="3dsz")
            with tc:
                p3d_palette = st.selectbox("3D Palette / Scale", list(COLOR_PALETTES.keys()) + CONTINUOUS_COLOR_SCALES, index=0, key="3d_palette")

        try:
            pdf = df.copy()
            sp = None
            if sz3 != "None":
                sp = sz3
                pdf[sz3] = pdf[sz3].fillna(0)
                mn = pdf[sz3].min()
                if mn <= 0:
                    pdf[sz3] = pdf[sz3] - mn + 0.1

            is_3d_num = col3 != "None" and col3 in numeric_cols
            use_scale = resolve_continuous_scale(p3d_palette) if (p3d_palette in CONTINUOUS_COLOR_SCALES or is_3d_num) else None
            use_seq = COLOR_PALETTES.get(p3d_palette, CLAY_COLOR_SEQUENCE) if not is_3d_num else None

            fig_3d = px.scatter_3d(
                pdf,
                x=x3,
                y=y3,
                z=z3,
                color=None if col3 == "None" else col3,
                size=sp,
                color_discrete_sequence=use_seq,
                color_continuous_scale=use_scale,
            )
            try:
                fig_3d.update_traces(opacity=0.8)
            except Exception:
                pass
            apply_clay_theme(fig_3d, title=f"3D Spatial Projection: {x3} × {y3} × {z3}")
            st.plotly_chart(fig_3d, theme=None)
        except Exception as err:
            st.error(f"3D Projection error: {err}")

# -------------------------------------------------------------
# 5. Interactive Custom Visual Builder
# -------------------------------------------------------------
render_html('<div class="step-header"><span class="step-num lime">04</span><span class="step-title">Interactive Custom Chart Builder</span></div>')

ct1, ct2, ct3 = st.columns(3)
with ct1:
    chart_type = st.selectbox(
        "Chart Geometry Type",
        [
            "Scatter Plot",
            "Line Chart",
            "Bar Chart",
            "Histogram",
            "Box Plot",
            "Violin Plot",
            "Donut Chart",
            "Density Heatmap (2D)",
            "Area Chart",
            "World Map",
            "3D Scatter",
            "3D Line",
        ],
    )
with ct2:
    x_lbl = "Location Field" if chart_type == "World Map" else ("Category Field" if chart_type == "Donut Chart" else "X-Axis Variable")
    x_col = st.selectbox(x_lbl, df.columns.tolist())
with ct3:
    if chart_type in ["3D Scatter", "3D Line"]:
        y_col = st.selectbox("Y-Axis Variable", df.columns.tolist(), index=min(1, len(df.columns) - 1))
    elif chart_type == "Donut Chart":
        y_col = st.selectbox("Metric Value (optional)", ["None"] + numeric_cols)
    elif chart_type == "World Map":
        y_col = st.selectbox("Color Metric (optional)", ["None"] + df.columns.tolist())
    elif chart_type in ["Box Plot", "Violin Plot", "Histogram"]:
        y_opts = ["None"] + df.columns.tolist()
        y_col = st.selectbox("Y-Axis Variable (optional)", y_opts, index=0)
    elif chart_type in ["Bar Chart", "Density Heatmap (2D)", "Area Chart"]:
        y_opts = ["None (Count)"] + df.columns.tolist()
        y_col = st.selectbox("Y-Axis Variable", y_opts, index=min(1, len(y_opts)-1))
    else:
        y_opts = ["None"] + df.columns.tolist()
        default_y = 0
        if len(df.columns) > 1:
            others = [c for c in df.columns if c != x_col]
            if others:
                default_y = y_opts.index(others[0])
        y_col = st.selectbox("Y-Axis Variable", y_opts, index=default_y)

z_col = None
if chart_type in ["3D Scatter", "3D Line"]:
    cz, _ = st.columns([4, 8])
    with cz:
        z_col = st.selectbox("Z-Axis Variable", df.columns.tolist(), index=min(2, len(df.columns) - 1))

# Expandable aesthetic and COLOR options
with st.expander("Advanced Visualization, Color Selection & Aesthetic Controls", expanded=True):
    # Row 1: Hue & Color Selection Controls
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
    with col_ctrl1:
        color_col = st.selectbox("Color Grouping (Hue Variable)", ["None"] + df.columns.tolist(), key="cb_color")
    with col_ctrl2:
        is_numeric_hue = color_col != "None" and color_col in numeric_cols
        if is_numeric_hue or chart_type in ["Density Heatmap (2D)"]:
            selected_color_scale = st.selectbox("Continuous Gradient Scale", CONTINUOUS_COLOR_SCALES, index=0, key="cb_scale")
            selected_palette_name = "Clay Classic (Default)"
        else:
            selected_palette_name = st.selectbox("Color Palette Scheme", list(COLOR_PALETTES.keys()), index=0, key="cb_palette")
            selected_color_scale = "Clay Violet"
    with col_ctrl3:
        if color_col == "None":
            st.markdown("<span style='font-size:12.5px;color:var(--ink-soft);font-weight:700;'>Custom Primary Color</span>", unsafe_allow_html=True)
            chosen_single_color = st.color_picker(
                "Primary Color",
                value="#6C5CE7",
                key="cb_single_color",
                help="Applies a custom single color to the chart when Hue is set to None.",
                label_visibility="collapsed",
            )
        else:
            chosen_single_color = None
            st.markdown(
                f"<div style='font-size:12px;color:var(--ink-soft);margin-top:14px;'>Color mapped to <code>{color_col}</code></div>",
                unsafe_allow_html=True,
            )

    # Row 2: Subplots, Opacity, Title & Scale Toggles
    as1, as2, as3 = st.columns(3)
    with as1:
        facet_col = st.selectbox("Faceted Subplots", ["None"] + categorical_cols, key="cb_facet")
    with as2:
        opacity_val = st.slider("Element Opacity", 0.1, 1.0, 0.85, key="cb_opacity")
    with as3:
        custom_title = st.text_input("Custom Chart Title", "", key="cb_title")

    # Row 3: Advanced Geometry Controls
    ao1, ao2, ao3 = st.columns(3)
    with ao1:
        log_x = st.checkbox("Logarithmic X-Axis", key="cb_logx")
        log_y = st.checkbox("Logarithmic Y-Axis", key="cb_logy")
    with ao2:
        hist_bins = 30
        barmode_val = "group"
        box_points = "outliers"
        if chart_type == "Histogram":
            hist_bins = st.slider("Histogram Bins", 5, 100, 30, key="cb_bins")
        elif chart_type == "Bar Chart":
            barmode_val = st.radio("Bar Mode", ["group", "stack", "overlay"], index=0, horizontal=True, key="cb_barmode")
        elif chart_type in ["Box Plot", "Violin Plot"]:
            box_points = st.selectbox("Data Points Overlay", ["outliers", "all", "suspectedoutliers", False], index=0, key="cb_boxpts")
    with ao3:
        violin_box = True
        trendline_opt = "None"
        if chart_type == "Violin Plot":
            violin_box = st.checkbox("Show Mini Box Inside Violin", value=True, key="cb_vbox")
        elif chart_type == "Scatter Plot" and has_statsmodels:
            trendline_opt = st.selectbox("Trendline Fit", ["None", "ols", "lowess"], index=0, key="cb_trend")

chart_err = None
fig_custom = None

try:
    active_seq = [chosen_single_color] + COLOR_PALETTES[selected_palette_name] if chosen_single_color else COLOR_PALETTES[selected_palette_name]
    active_scale = resolve_continuous_scale(selected_color_scale)

    kwargs: dict[str, Any] = {"data_frame": df}

    # Safe sequence / scale assignment
    if is_numeric_hue or chart_type in ["Density Heatmap (2D)"]:
        kwargs["color_continuous_scale"] = active_scale
    else:
        kwargs["color_discrete_sequence"] = active_seq

    # Coordinate mapping per geometry type
    if chart_type in ["3D Scatter", "3D Line"]:
        if not z_col:
            chart_err = "Z-Axis variable is required for 3D visualizations."
        else:
            kwargs.update({"x": x_col, "y": y_col, "z": z_col})

    elif chart_type == "Donut Chart":
        kwargs["names"] = x_col
        if y_col not in [None, "None"]:
            kwargs["values"] = y_col

    elif chart_type == "World Map":
        if y_col not in [None, "None"]:
            kwargs.update({"locations": x_col, "color": y_col})
        else:
            mdf = df[x_col].dropna().value_counts().reset_index()
            mdf.columns = [x_col, "Count"]
            kwargs.update({"data_frame": mdf, "locations": x_col, "color": "Count"})
        kwargs["locationmode"] = (
            "USA-states" if "state" in x_col.lower() else ("ISO-3" if "iso" in x_col.lower() else "country names")
        )

    elif chart_type in ["Box Plot", "Violin Plot"]:
        # Flexible 1-variable or 2-variable setup
        if y_col in [None, "None"]:
            if x_col in numeric_cols:
                kwargs["y"] = x_col
            else:
                kwargs["x"] = x_col
        else:
            kwargs["x"] = x_col
            kwargs["y"] = y_col
        if chart_type == "Box Plot":
            kwargs["points"] = box_points
        elif chart_type == "Violin Plot":
            kwargs["box"] = violin_box
            kwargs["points"] = box_points

    elif chart_type == "Histogram":
        kwargs["x"] = x_col
        if y_col not in [None, "None"]:
            kwargs["y"] = y_col
        kwargs["nbins"] = hist_bins

    elif chart_type in ["Bar Chart", "Area Chart"]:
        if y_col in [None, "None", "None (Count)"]:
            bdf = df[x_col].value_counts().reset_index()
            bdf.columns = [x_col, "Count"]
            kwargs["data_frame"] = bdf
            kwargs["x"] = x_col
            kwargs["y"] = "Count"
        else:
            kwargs["x"] = x_col
            kwargs["y"] = y_col
        if chart_type == "Bar Chart":
            kwargs["barmode"] = barmode_val

    elif chart_type == "Density Heatmap (2D)":
        kwargs["x"] = x_col
        if y_col not in [None, "None", "None (Count)"]:
            kwargs["y"] = y_col
        else:
            num_others = [c for c in numeric_cols if c != x_col]
            kwargs["y"] = num_others[0] if num_others else x_col

    else:  # Scatter Plot, Line Chart
        kwargs["x"] = x_col
        if y_col not in [None, "None"]:
            kwargs["y"] = y_col
        else:
            chart_err = f"Y-Axis variable is required for {chart_type}. Please select a Y variable above."

    if chart_err is None:
        # Grouping (Hue) & Faceting
        if color_col != "None" and chart_type not in ["World Map", "Donut Chart"]:
            kwargs["color"] = color_col
        if facet_col != "None" and chart_type not in ["3D Scatter", "3D Line", "World Map", "Donut Chart"]:
            kwargs["facet_col"] = facet_col
            kwargs["facet_col_wrap"] = 2

        # Safe Log Transform Handling
        if chart_type not in ["Donut Chart", "World Map"]:
            if log_x:
                if x_col in numeric_cols and (df[x_col] > 0).all():
                    kwargs["log_x"] = True
                else:
                    st.caption("ℹ️ Logarithmic X omitted: X-axis contains non-positive values or is categorical.")
            if log_y and y_col not in [None, "None", "None (Count)"]:
                if y_col in numeric_cols and (df[y_col] > 0).all():
                    kwargs["log_y"] = True
                else:
                    st.caption("ℹ️ Logarithmic Y omitted: Y-axis contains non-positive values or is categorical.")

        # Trendline for Scatter
        if chart_type == "Scatter Plot" and trendline_opt != "None" and has_statsmodels:
            kwargs["trendline"] = trendline_opt

        # Chart builders map
        chart_map = {
            "Scatter Plot": px.scatter,
            "Line Chart": px.line,
            "Bar Chart": px.bar,
            "Histogram": px.histogram,
            "Box Plot": px.box,
            "Violin Plot": px.violin,
            "Donut Chart": lambda **kw: px.pie(hole=0.45, **kw),
            "Density Heatmap (2D)": px.density_heatmap,
            "Area Chart": px.area,
            "World Map": px.choropleth,
            "3D Scatter": px.scatter_3d,
            "3D Line": px.line_3d,
        }

        # Build figure
        if chart_type in chart_map:
            fig_custom = chart_map[chart_type](**kwargs)

            # Safely apply opacity via update_traces
            if fig_custom is not None:
                try:
                    fig_custom.update_traces(opacity=opacity_val)
                except Exception:
                    pass

                # Auto title
                y_disp = f" vs {y_col}" if y_col not in (None, "None", "None (Count)") and chart_type not in ["Donut Chart", "World Map"] else ""
                t_str = custom_title or f"{chart_type}: {x_col}{y_disp}"
                apply_clay_theme(fig_custom, title=t_str)
                st.plotly_chart(fig_custom, theme=None)

except Exception as e:
    chart_err = str(e)

if chart_err:
    render_html(
        f"""
        <div class="alert-box alert-danger">
            <strong>Chart Construction Advisory:</strong><br>
            <em>{chart_err}</em><br>
            <span style="font-size:12px;opacity:0.85;">Tip: Ensure appropriate variable types (numeric vs categorical) and try selecting a Y-axis variable or adjusting color grouping.</span>
        </div>
        """
    )

