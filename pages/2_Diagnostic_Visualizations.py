from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.styles import inject_css

try:
    import statsmodels  # type: ignore

    has_statsmodels = True
except ImportError:
    has_statsmodels = False

st.set_page_config(page_title="VizML: Diagnostic Visualizations", layout="wide")
inject_css()

if "df" not in st.session_state or st.session_state["df"] is None:
    st.markdown(
        """
    <div class="glass-card">
        <span class="eyebrow">Phase 02 | Exploratory Analytics</span>
        <div class="page-title">Diagnostic Visualizations</div>
        <div class="page-sub">Upload a dataset in Data Curation to unlock interactive diagnostics.</div>
    </div>
    """,
        unsafe_allow_html=True,
    )
    st.page_link("pages/1_Data_Curation.py", label="Go to Data Curation Engine")
    st.stop()

df = st.session_state["df"]
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = [c for c in df.columns if c not in numeric_cols]

st.markdown(
    """
<div class="glass-card">
    <span class="eyebrow">Phase 02 | Exploratory Analytics</span>
    <div class="page-title">Diagnostic Visualizations</div>
    <div class="page-sub">
        Multi-dimensional diagnostics, correlation redundancy detection, 3D feature explorers, and a custom Plotly builder.
    </div>
</div>
""",
    unsafe_allow_html=True,
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Rows", f"{df.shape[0]:,}")
m2.metric("Total Columns", df.shape[1])
m3.metric("Numeric Features", len(numeric_cols))
m4.metric("Categorical Features", len(categorical_cols))

st.markdown("<br>", unsafe_allow_html=True)

st.markdown(
    '<div class="section-header"><span class="badge badge-indigo">AUTO</span> Generated Insights Dashboard</div>',
    unsafe_allow_html=True,
)

auto_plots: list = []
auto_errors: list = []

try:
    donut_cats = [c for c in categorical_cols if 2 <= df[c].nunique() <= 6]
    if donut_cats:
        fig = px.pie(df, names=donut_cats[0], hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.update_layout(
            title=dict(text=f"Proportion: {donut_cats[0]}", font=dict(family="Plus Jakarta Sans", size=14, color="#111827")),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#111827"),
            margin=dict(t=40, b=10, l=10, r=10),
        )
        auto_plots.append(("donut", fig))
except Exception as _e:
    auto_errors.append(f"Proportion chart skipped: {_e}")

try:
    lat_cols = [c for c in df.columns if "lat" in c.lower()]
    lon_cols = [c for c in df.columns if any(k in c.lower() for k in ["lon", "lng"])]
    country_cols = [c for c in df.columns if any(k in c.lower() for k in ["country", "nation", "state", "iso"])]
    if lat_cols and lon_cols:
        fig = px.scatter_geo(
            df,
            lat=lat_cols[0],
            lon=lon_cols[0],
            color=numeric_cols[0] if numeric_cols else None,
            title=f"Geo: {lat_cols[0]}/{lon_cols[0]}",
            color_continuous_scale="Viridis",
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#111827"),
            margin=dict(t=40, b=10, l=10, r=10),
        )
        auto_plots.append(("geo", fig))
    elif country_cols:
        gdf = df[country_cols[0]].dropna().value_counts().reset_index()
        gdf.columns = [country_cols[0], "Count"]
        lm = "country names" if "state" not in country_cols[0].lower() else "USA-states"
        fig = px.choropleth(
            gdf,
            locations=country_cols[0],
            locationmode=lm,
            color="Count",
            title=f"Geo Distribution: {country_cols[0]}",
            color_continuous_scale="Viridis",
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#111827"),
            margin=dict(t=40, b=10, l=10, r=10),
        )
        auto_plots.append(("geo", fig))
except Exception as _e:
    auto_errors.append(f"Geo chart skipped: {_e}")

try:
    dt_cols = df.select_dtypes(include=[np.datetime64]).columns.tolist()
    if not dt_cols:
        pot = [c for c in categorical_cols if any(k in c.lower() for k in ["date", "time", "year", "month"])]
        if pot:
            tmp = pd.to_datetime(df[pot[0]], errors="coerce", format="mixed")
            if tmp.notna().sum() > 0.5 * len(df):
                df = df.copy()
                df[pot[0]] = tmp
                dt_cols = [pot[0]]
    if dt_cols and numeric_cols:
        ts = df[[dt_cols[0], numeric_cols[0]]].dropna().sort_values(dt_cols[0])
        fig = px.line(ts, x=dt_cols[0], y=numeric_cols[0], color_discrete_sequence=["#6366F1"])
        fig.update_layout(
            title=dict(text=f"Trend: {numeric_cols[0]}", font=dict(family="Plus Jakarta Sans", size=14, color="#111827")),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#111827"),
            margin=dict(t=40, b=10, l=10, r=10),
        )
        auto_plots.append(("ts", fig))
except Exception as _e:
    auto_errors.append(f"Time-series chart skipped: {_e}")

try:
    if len(auto_plots) < 4 and numeric_cols:
        fig = px.histogram(df, x=numeric_cols[0], marginal="box", color_discrete_sequence=["#6366F1"])
        fig.update_layout(
            title=dict(text=f"Distribution: {numeric_cols[0]}", font=dict(family="Plus Jakarta Sans", size=14, color="#111827")),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#111827"),
            margin=dict(t=40, b=10, l=10, r=10),
        )
        auto_plots.append(("hist", fig))
except Exception as _e:
    auto_errors.append(f"Histogram skipped: {_e}")

try:
    if len(auto_plots) < 4 and len(numeric_cols) >= 3:
        fig = px.scatter_3d(
            df,
            x=numeric_cols[0],
            y=numeric_cols[1],
            z=numeric_cols[2],
            color=categorical_cols[0] if categorical_cols else None,
            opacity=0.8,
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        fig.update_layout(
            title=dict(text="3D Feature Projection", font=dict(family="Plus Jakarta Sans", size=14, color="#111827")),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#111827"),
            margin=dict(t=40, b=10, l=10, r=10),
        )
        auto_plots.append(("3d", fig))
    elif len(auto_plots) < 4 and len(numeric_cols) >= 2:
        use_tl = has_statsmodels and df.shape[0] > 2 and df[numeric_cols[:2]].isna().sum().sum() == 0
        fig = px.scatter(
            df,
            x=numeric_cols[0],
            y=numeric_cols[1],
            trendline="ols" if use_tl else None,
            color_discrete_sequence=["#8B5CF6"],
            opacity=0.7,
        )
        fig.update_layout(
            title=dict(text=f"{numeric_cols[0]} vs {numeric_cols[1]}", font=dict(family="Plus Jakarta Sans", size=14, color="#111827")),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#111827"),
            margin=dict(t=40, b=10, l=10, r=10),
        )
        auto_plots.append(("scatter", fig))
except Exception as _e:
    auto_errors.append(f"Scatter/3D chart skipped: {_e}")

if not auto_plots and not auto_errors:
    st.info("Not enough columns to generate automatic insights.")
else:
    for i in range(0, len(auto_plots), 2):
        cl, cr = st.columns(2)
        with cl:
            if i < len(auto_plots):
                try:
                    st.plotly_chart(auto_plots[i][1], width="stretch", key=f"ap_{auto_plots[i][0]}")
                except Exception as e:
                    st.warning(f"Chart render error: {e}")
        with cr:
            if i + 1 < len(auto_plots):
                try:
                    st.plotly_chart(auto_plots[i+1][1], width="stretch", key=f"ap_{auto_plots[i+1][0]}")
                except Exception as e:
                    st.warning(f"Chart render error: {e}")
    if auto_errors:
        with st.expander(f"{len(auto_errors)} auto-chart(s) skipped", expanded=False):
            for msg in auto_errors:
                st.caption(msg)

st.markdown("<br>", unsafe_allow_html=True)

st.markdown(
    '<div class="section-header"><span class="badge badge-violet">SUITE</span> Multidimensional Diagnostic Suite</div>',
    unsafe_allow_html=True,
)

if len(numeric_cols) < 2:
    st.info("At least 2 numeric columns are required for the Diagnostic Suite.")
else:
    tab_corr, tab_scatter, tab_3d = st.tabs(["Correlation Matrix", "Scatter Matrix", "3D Explorer"])

    with tab_corr:
        cl, cr = st.columns([6, 6])
        with cl:
            sel_corr = st.multiselect("Columns:", numeric_cols, default=numeric_cols, key="corr_cols")
        with cr:
            c1, c2 = st.columns(2)
            with c1:
                corr_method = st.radio(
                    "Method", ["pearson", "spearman", "kendall"], index=0, horizontal=True, key="corr_method"
                )
            with c2:
                threshold = st.slider("Redundancy |r|", 0.50, 0.99, 0.80, 0.05, key="corr_thresh")

        if len(sel_corr) < 2:
            st.warning("Select at least 2 columns.")
        else:
            try:
                corr = df[sel_corr].corr(method=corr_method)
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
                        color_continuous_scale="RdBu",
                        color_continuous_midpoint=0,
                        range_color=[-1, 1],
                    )
                    fig_h.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#111827"),
                        margin=dict(t=10, b=10, l=10, r=10),
                    )
                    st.plotly_chart(fig_h, width="stretch")
                with ci:
                    st.markdown("#### Redundancy Inspector")
                    if pairs:
                        st.markdown(
                            f'<div class="alert-box alert-warning">Found <strong>{len(pairs)}</strong> pairs with |r| >= <strong>{threshold:.2f}</strong>.</div>',
                            unsafe_allow_html=True,
                        )
                        for a, b, r in pairs:
                            st.markdown(f"- `{a}` vs `{b}`: **{r:.3f}**")
                    else:
                        st.markdown(
                            f'<div class="alert-box alert-success">No redundant pairs at threshold {threshold:.2f}.</div>',
                            unsafe_allow_html=True,
                        )
            except Exception as err:
                st.error(str(err))

    with tab_scatter:
        d1, d2 = st.columns([8, 4])
        with d1:
            dims = st.multiselect(
                "Dimensions (max 6):", numeric_cols, default=numeric_cols[:min(4, len(numeric_cols))], key="scatter_dims"
            )
        with d2:
            color_v = st.selectbox("Color by:", ["None"] + df.columns.tolist(), key="scatter_color")
        if len(dims) < 2:
            st.info("Select at least 2 dimensions.")
        else:
            try:
                fig_m = px.scatter_matrix(
                    df,
                    dimensions=dims,
                    color=None if color_v == "None" else color_v,
                    opacity=0.7,
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                    color_continuous_scale="Viridis",
                )
                fig_m.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#111827"),
                    margin=dict(t=20, b=20, l=20, r=20),
                )
                st.plotly_chart(fig_m, width="stretch")
            except Exception as err:
                st.error(str(err))

    with tab_3d:
        ax, ex = st.columns([7, 5])
        with ax:
            xc, yc, zc = st.columns(3)
            with xc:
                x3 = st.selectbox("X", numeric_cols, index=0, key="3dx")
            with yc:
                y3 = st.selectbox("Y", numeric_cols, index=min(1, len(numeric_cols) - 1), key="3dy")
            with zc:
                z3 = st.selectbox("Z", numeric_cols, index=min(2, len(numeric_cols) - 1), key="3dz")
        with ex:
            cc, sc = st.columns(2)
            with cc:
                col3 = st.selectbox("Color", ["None"] + df.columns.tolist(), key="3dcol")
            with sc:
                sz3 = st.selectbox("Size", ["None"] + numeric_cols, key="3dsz")
        try:
            pdf = df.copy()
            sp = None
            if sz3 != "None":
                sp = sz3
                pdf[sz3] = pdf[sz3].fillna(0)
                mn = pdf[sz3].min()
                if mn <= 0:
                    pdf[sz3] = pdf[sz3] - mn + 0.1
            fig_3d = px.scatter_3d(
                pdf,
                x=x3,
                y=y3,
                z=z3,
                color=None if col3 == "None" else col3,
                size=sp,
                opacity=0.8,
                color_discrete_sequence=px.colors.qualitative.Pastel,
                color_continuous_scale="Viridis",
            )
            fig_3d.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#111827"),
                margin=dict(t=10, b=10, l=10, r=10),
            )
            st.plotly_chart(fig_3d, width="stretch")
        except Exception as err:
            st.error(str(err))

st.markdown("<br>", unsafe_allow_html=True)

st.markdown(
    '<div class="section-header"><span class="badge badge-emerald">BUILDER</span> Interactive Custom Chart</div>',
    unsafe_allow_html=True,
)

ct1, ct2, ct3 = st.columns(3)
with ct1:
    chart_type = st.selectbox(
        "Chart Type",
        [
            "Scatter Plot",
            "Line Chart",
            "Bar Chart",
            "Histogram",
            "Box Plot",
            "Violin Plot",
            "Donut Chart",
            "World Map",
            "3D Scatter",
            "3D Line",
        ],
    )
with ct2:
    x_lbl = "Locations" if chart_type == "World Map" else ("Names" if chart_type == "Donut Chart" else "X-Axis")
    x_col = st.selectbox(x_lbl, df.columns.tolist())
with ct3:
    if chart_type in ["3D Scatter", "3D Line"]:
        y_col = st.selectbox("Y-Axis", df.columns.tolist(), index=min(1, len(df.columns) - 1))
    elif chart_type == "Donut Chart":
        y_col = st.selectbox("Values (optional)", ["None"] + numeric_cols)
    elif chart_type == "World Map":
        y_col = st.selectbox("Color Value (optional)", ["None"] + df.columns.tolist())
    else:
        y_opts = ["None"] + df.columns.tolist()
        default_y = 0
        if chart_type in ["Scatter Plot", "Line Chart", "Box Plot", "Violin Plot"] and len(df.columns) > 1:
            others = [c for c in df.columns if c != x_col]
            if others:
                default_y = y_opts.index(others[0])
        y_col = st.selectbox("Y-Axis", y_opts, index=default_y)

z_col = None
if chart_type in ["3D Scatter", "3D Line"]:
    cz, _ = st.columns([4, 8])
    with cz:
        z_col = st.selectbox("Z-Axis", df.columns.tolist(), index=min(2, len(df.columns) - 1))

with st.expander("Advanced Settings", expanded=False):
    as1, as2, as3 = st.columns(3)
    with as1:
        color_col = st.selectbox("Color Grouping", ["None"] + df.columns.tolist(), key="cb_color")
    with as2:
        facet_col = st.selectbox("Facet Column", ["None"] + categorical_cols, key="cb_facet")
    with as3:
        opacity_val = st.slider("Opacity", 0.1, 1.0, 0.8, key="cb_opacity")
    ao1, ao2 = st.columns(2)
    with ao1:
        custom_title = st.text_input("Custom Title", "", key="cb_title")
        log_x = st.checkbox("Log X", key="cb_logx")
    with ao2:
        log_y = st.checkbox("Log Y", key="cb_logy")
        hist_bins = 30
        barmode_val = "group"
        if chart_type == "Histogram":
            hist_bins = st.slider("Bins", 5, 100, 30, key="cb_bins")
        elif chart_type == "Bar Chart":
            barmode_val = st.radio("Bar Mode", ["group", "stack", "overlay"], index=0, horizontal=True, key="cb_barmode")

chart_err = None
fig_custom = None
try:
    kwargs: dict[str, Any] = {"data_frame": df}
    if chart_type not in ["Donut Chart", "World Map"]:
        kwargs["opacity"] = opacity_val
    kwargs["color_discrete_sequence"] = px.colors.qualitative.Pastel

    if chart_type in ["3D Scatter", "3D Line"]:
        kwargs.update({"x": x_col, "y": y_col, "z": z_col})
    elif chart_type == "Donut Chart":
        kwargs["names"] = x_col
        if y_col != "None":
            kwargs["values"] = y_col
    elif chart_type == "World Map":
        if y_col != "None":
            kwargs.update({"locations": x_col, "color": y_col})
        else:
            mdf = df[x_col].dropna().value_counts().reset_index()
            mdf.columns = [x_col, "Count"]
            kwargs.update({"data_frame": mdf, "locations": x_col, "color": "Count"})
        kwargs["locationmode"] = (
            "USA-states" if "state" in x_col.lower() else ("ISO-3" if "iso" in x_col.lower() else "country names")
        )
    else:
        kwargs["x"] = x_col
        if y_col != "None":
            kwargs["y"] = y_col
        elif chart_type in ["Scatter Plot", "Line Chart", "Box Plot", "Violin Plot"]:
            chart_err = f"Y-Axis required for {chart_type}."

    if chart_err is None:
        if color_col != "None" and chart_type not in ["World Map", "Donut Chart"]:
            kwargs["color"] = color_col
        if facet_col != "None" and chart_type not in ["3D Scatter", "3D Line", "World Map", "Donut Chart"]:
            kwargs["facet_col"] = facet_col
            kwargs["facet_col_wrap"] = 2
        if chart_type not in ["Donut Chart", "World Map"]:
            kwargs.update({"log_x": log_x, "log_y": log_y})

        chart_map = {
            "Scatter Plot": px.scatter,
            "Line Chart": px.line,
            "Box Plot": px.box,
            "Violin Plot": px.violin,
            "Donut Chart": lambda **kw: px.pie(hole=0.4, **kw),
            "World Map": px.choropleth,
            "3D Scatter": px.scatter_3d,
            "3D Line": px.line_3d,
        }
        if chart_type == "Bar Chart":
            kwargs["barmode"] = barmode_val
            fig_custom = px.bar(**kwargs)
        elif chart_type == "Histogram":
            kwargs["nbins"] = hist_bins
            fig_custom = px.histogram(**kwargs)
        elif chart_type in chart_map:
            fig_custom = chart_map[chart_type](**kwargs)

        if fig_custom is not None:
            title = custom_title or f"{chart_type}: {x_col}" + (
                f" vs {y_col}" if y_col not in (None, "None") and chart_type not in ["Donut Chart", "World Map"] else ""
            )
            fig_custom.update_layout(
                title=dict(text=title, font=dict(family="Plus Jakarta Sans", size=16, color="#111827")),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#111827"),
            )
            if chart_type not in ["3D Scatter", "3D Line", "World Map"]:
                fig_custom.update_layout(plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_custom, width="stretch")
except Exception as e:
    chart_err = str(e)

if chart_err:
    st.markdown(
        f'<div class="alert-box alert-danger"><strong>Chart Error</strong><br><em>{chart_err}</em></div>',
        unsafe_allow_html=True,
    )
