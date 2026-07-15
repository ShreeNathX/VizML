import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Safe check for statsmodels (required for OLS trendlines in Plotly)
try:
    import statsmodels
    has_statsmodels = True
except ImportError:
    has_statsmodels = False

# Configure page settings
st.set_page_config(page_title="VizML - Diagnostic Visualizations", layout="wide")

# Inject Custom CSS for premium styling
st.markdown("""
    <style>
    /* Styling Metrics Cards */
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 700;
        color: #6C63FF !important;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 14px;
        color: #8A8BA8 !important;
        font-weight: 500;
    }
    
    /* Elegant Title and Subtitle styling */
    .app-title {
        font-family: 'Space Grotesk', 'Outfit', 'Inter', sans-serif;
        font-size: 38px;
        font-weight: 800;
        background: linear-gradient(135deg, #6C63FF 0%, #EC4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
    }
    .app-subtitle {
        font-size: 16px;
        color: #8A8BA8;
        margin-bottom: 25px;
    }

    /* Container cards style */
    .diag-card {
        border-radius: 12px;
        padding: 24px;
        background: #13141A;
        border: 1px solid #272836;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        margin-bottom: 25px;
    }

    .section-header {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 22px;
        font-weight: 700;
        color: #EEEEF5;
        margin-top: 10px;
        margin-bottom: 15px;
        background: linear-gradient(135deg, #EEEEF5 30%, #8A8BA8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Alert callouts */
    .alert-box {
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
        font-size: 14px;
    }
    .alert-warning {
        background-color: rgba(245, 166, 35, 0.1);
        border: 1px solid #F5A623;
        color: #F5A623;
    }
    .alert-danger {
        background-color: rgba(240, 92, 92, 0.1);
        border: 1px solid #F05C5C;
        color: #F05C5C;
    }
    .alert-info {
        background-color: rgba(108, 99, 255, 0.1);
        border: 1px solid #6C63FF;
        color: #EEEEF5;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------- DATASET CHECKS -----------------
if "df" not in st.session_state or st.session_state["df"] is None:
    st.markdown('<div class="app-title">Diagnostic Visualizations</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-subtitle">Spot correlations, redundant features, and distribution separations before modeling.</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="diag-card" style="text-align: center; padding: 40px; margin-top: 20px;">
        <h3 style="color: #F5A623; margin-bottom: 15px;">⚠️ No Active Dataset Found</h3>
        <p style="color: #8A8BA8; margin-bottom: 25px; font-size: 15px;">
            You need to upload and clean a dataset before exploring its visualizations. 
            All visualizations on this page run in-memory on your curated session state.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.page_link("pages/page1.py", label="Go to Data Curation Page", icon="🧹")

else:
    df = st.session_state["df"]
    
    # Identify numeric vs other columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]
    
    st.markdown('<div class="app-title">Diagnostic Visualizations</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-subtitle">Explore patterns, spot redundant features, and inspect data geometry. (Diagnostic Stage - Non-Mutating)</div>', unsafe_allow_html=True)

    # ----------------- DATASET METRICS GRID -----------------
    with st.container():
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("Total Rows", df.shape[0])
        with col_m2:
            st.metric("Total Columns", df.shape[1])
        with col_m3:
            st.metric("Numeric Features", len(numeric_cols))
        with col_m4:
            st.metric("Categorical Features", len(categorical_cols))

    st.markdown("---")

    # ----------------- STAGE 3.1: AUTO-GENERATED INSIGHTS DASHBOARD -----------------
    st.markdown('<div class="section-header">📊 Section 1: Auto-Generated Insights Dashboard</div>', unsafe_allow_html=True)
    st.write("VizML scans your dataset schemas and automatically compiles baseline visual diagnostics.")
    
    auto_plots = []
    
    # 1. Proportional Donut Chart (categorical columns with cardinality between 2 and 6)
    donut_cats = [c for c in categorical_cols if 2 <= df[c].nunique() <= 6]
    if donut_cats:
        prim_cat = donut_cats[0]
        fig_donut = px.pie(
            df,
            names=prim_cat,
            hole=0.4,
            template="plotly_dark",
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        fig_donut.update_layout(
            title=dict(text=f"Proportion Share: {prim_cat} (Donut)", font=dict(family="Space Grotesk", size=14)),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#EEEEF5"),
            margin=dict(t=40, b=10, l=10, r=10)
        )
        auto_plots.append(("donut_prop", fig_donut))
        
    # 2. Geographical Map (World/Country/States)
    # Search for latitude/longitude
    lat_cols = [c for c in df.columns if any(kw in c.lower() for kw in ["latitude", "lat_"]) or c.lower() == "lat"]
    lon_cols = [c for c in df.columns if any(kw in c.lower() for kw in ["longitude", "lon_"]) or c.lower() == "lon" or c.lower() == "lng"]
    
    # Search for country name columns
    country_cols = [c for c in df.columns if any(kw in c.lower() for kw in ["country", "nation", "state", "iso"])]
    
    if lat_cols and lon_cols:
        # Plot Scatter Geo Map
        fig_map = px.scatter_geo(
            df,
            lat=lat_cols[0],
            lon=lon_cols[0],
            color=numeric_cols[0] if numeric_cols else None,
            title=f"Geographical Coordinates: {lat_cols[0]} / {lon_cols[0]}",
            template="plotly_dark",
            color_continuous_scale="Plasma"
        )
        fig_map.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#EEEEF5"),
            margin=dict(t=40, b=10, l=10, r=10)
        )
        auto_plots.append(("geo_map", fig_map))
    elif country_cols:
        # Plot Choropleth Map of Country counts
        geo_col = country_cols[0]
        map_df = df[geo_col].dropna().value_counts().reset_index()
        map_df.columns = [geo_col, "Count"]
        
        # Decide location mode
        loc_mode = "country names"
        if "state" in geo_col.lower():
            loc_mode = "USA-states"
        elif "iso" in geo_col.lower():
            loc_mode = "ISO-3"
            
        fig_map = px.choropleth(
            map_df,
            locations=geo_col,
            locationmode=loc_mode,
            color="Count",
            title=f"Geographical Distribution: {geo_col}",
            template="plotly_dark",
            color_continuous_scale="Plasma"
        )
        fig_map.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#EEEEF5"),
            margin=dict(t=40, b=10, l=10, r=10)
        )
        auto_plots.append(("geo_map", fig_map))

    # 3. Temporal Trend / Line Graph
    datetime_cols = df.select_dtypes(include=[np.datetime64]).columns.tolist()
    if len(datetime_cols) == 0:
        potential_dates = [c for c in categorical_cols if any(kw in c.lower() for kw in ["date", "time", "year", "month"])]
        if potential_dates:
            try:
                temp_dates = pd.to_datetime(df[potential_dates[0]], errors="coerce")
                if temp_dates.notna().sum() > 0.5 * len(df):
                    df_temp = df.copy()
                    df_temp[potential_dates[0]] = temp_dates
                    datetime_cols = [potential_dates[0]]
                    df = df_temp
            except:
                pass

    if datetime_cols and numeric_cols:
        date_col = datetime_cols[0]
        num_target = numeric_cols[0]
        ts_df = df[[date_col, num_target]].dropna().sort_values(by=date_col)
        
        fig_ts = px.line(
            ts_df,
            x=date_col,
            y=num_target,
            template="plotly_dark",
            color_discrete_sequence=["#F5A623"]
        )
        fig_ts.update_layout(
            title=dict(text=f"Temporal Trend: {num_target} over time", font=dict(family="Space Grotesk", size=14)),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#EEEEF5"),
            margin=dict(t=40, b=10, l=10, r=10)
        )
        auto_plots.append(("time_series", fig_ts))

    # 4. Standard Numeric Distribution (if not enough specific charts, fill it up)
    if len(auto_plots) < 4 and len(numeric_cols) > 0:
        prim_num = numeric_cols[0]
        fig_num = px.histogram(
            df, 
            x=prim_num, 
            marginal="box",
            template="plotly_dark",
            color_discrete_sequence=["#6C63FF"]
        )
        fig_num.update_layout(
            title=dict(text=f"Distribution of {prim_num}", font=dict(family="Space Grotesk", size=14)),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#EEEEF5"),
            margin=dict(t=40, b=10, l=10, r=10)
        )
        auto_plots.append(("numeric_dist", fig_num))

    # 5. Numerical Interaction Scatter or 3D projection
    if len(auto_plots) < 4:
        if len(numeric_cols) >= 3:
            # 3D Feature Projection of top 3 features
            fig_3d_proj = px.scatter_3d(
                df,
                x=numeric_cols[0],
                y=numeric_cols[1],
                z=numeric_cols[2],
                color=categorical_cols[0] if categorical_cols else None,
                template="plotly_dark",
                opacity=0.8,
                color_discrete_sequence=px.colors.qualitative.Safe,
                color_continuous_scale="Plasma"
            )
            fig_3d_proj.update_layout(
                title=dict(text=f"3D Feature Projection: {numeric_cols[0]} / {numeric_cols[1]} / {numeric_cols[2]}", font=dict(family="Space Grotesk", size=14)),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#EEEEF5"),
                margin=dict(t=40, b=10, l=10, r=10)
            )
            auto_plots.append(("3d_projection", fig_3d_proj))
        elif len(numeric_cols) >= 2:
            fig_scatter = px.scatter(
                df,
                x=numeric_cols[0],
                y=numeric_cols[1],
                trendline="ols" if (has_statsmodels and df.shape[0] > 2 and df[numeric_cols[:2]].isna().sum().sum() == 0) else None,
                template="plotly_dark",
                color_discrete_sequence=["#2DD4A0"],
                opacity=0.7
            )
            fig_scatter.update_layout(
                title=dict(text=f"Numerical Interaction: {numeric_cols[0]} vs {numeric_cols[1]}", font=dict(family="Space Grotesk", size=14)),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#EEEEF5"),
                margin=dict(t=40, b=10, l=10, r=10)
            )
            auto_plots.append(("scatter_interaction", fig_scatter))

    # Render auto-dashboard in 2x2 grid
    if len(auto_plots) == 0:
        st.info("No columns available to compile automated insights.")
    else:
        st.markdown('<div class="diag-card">', unsafe_allow_html=True)
        cols_count = len(auto_plots)
        for i in range(0, cols_count, 2):
            col_left, col_right = st.columns(2)
            with col_left:
                if i < cols_count:
                    st.plotly_chart(auto_plots[i][1], use_container_width=True)
            with col_right:
                if i + 1 < cols_count:
                    st.plotly_chart(auto_plots[i+1][1], use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ----------------- STAGE 3.2: DIAGNOSTIC EXPLORATION SUITE -----------------
    st.markdown('<div class="section-header">🔍 Section 2: Multidimensional Diagnostic Suite</div>', unsafe_allow_html=True)
    st.write("Use specialized multi-variable diagnostic views to detect feature redundancy, multicollinearity, or high-dimensional clustering.")
    
    if len(numeric_cols) < 2:
        st.info("At least 2 numeric columns are required to unlock the Multidimensional Diagnostic Suite.")
    else:
        tab_corr, tab_scatter, tab_3d = st.tabs([
            "📊 Correlation Matrix & Redundancy Inspector", 
            "📈 2D Scatter Matrix", 
            "🌌 3D Scatter Explorer"
        ])
        
        # 1. Correlation Matrix Tab
        with tab_corr:
            col_setup_l, col_setup_r = st.columns([6, 6])
            with col_setup_l:
                selected_corr_cols = st.multiselect(
                    "Columns to include in correlation matrix:",
                    numeric_cols,
                    default=numeric_cols,
                    key="corr_multiselect_tier2"
                )
            with col_setup_r:
                col_sub1, col_sub2 = st.columns(2)
                with col_sub1:
                    corr_method = st.radio(
                        "Method", 
                        ["pearson", "spearman", "kendall"], 
                        index=0, 
                        horizontal=True,
                        key="corr_method_tier2"
                    )
                with col_sub2:
                    threshold = st.slider(
                        "Redundancy Threshold (|r|)", 
                        0.50, 0.99, 0.80, 0.05,
                        key="corr_threshold_tier2"
                    )
            
            if len(selected_corr_cols) < 2:
                st.warning("⚠️ Select at least 2 columns to calculate correlation.")
            else:
                corr_matrix = df[selected_corr_cols].corr(method=corr_method)
                redundant_pairs = []
                for i in range(len(selected_corr_cols)):
                    for j in range(i + 1, len(selected_corr_cols)):
                        col_a = selected_corr_cols[i]
                        col_b = selected_corr_cols[j]
                        r_val = corr_matrix.loc[col_a, col_b]
                        if abs(r_val) >= threshold:
                            redundant_pairs.append((col_a, col_b, r_val))
                redundant_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
                
                col_plot, col_inspect = st.columns([7, 5])
                with col_plot:
                    fig_heat = px.imshow(
                        corr_matrix,
                        text_auto=".2f" if len(selected_corr_cols) <= 15 else False,
                        aspect="auto",
                        color_continuous_scale="RdBu",
                        color_continuous_midpoint=0.0,
                        range_color=[-1.0, 1.0],
                        labels=dict(color="Correlation"),
                        template="plotly_dark"
                    )
                    fig_heat.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#EEEEF5"),
                        margin=dict(t=10, b=10, l=10, r=10)
                    )
                    st.plotly_chart(fig_heat, use_container_width=True)
                    
                with col_inspect:
                    st.markdown("#### 🔍 Feature Redundancy Inspector")
                    if redundant_pairs:
                        st.markdown(
                            f'<div class="alert-box alert-warning">'
                            f'Found <strong>{len(redundant_pairs)}</strong> pairs with absolute correlation '
                            f'$\geq$ <strong>{threshold:.2f}</strong>.'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                        st.write("Highly correlated features convey redundant information. Consider dropping one of the features in each pair:")
                        for col_a, col_b, r_val in redundant_pairs:
                            st.markdown(f"• `{col_a}` ↔️ `{col_b}` : **{r_val:.3f}** ({'positive' if r_val > 0 else 'negative'})")
                    else:
                        st.markdown(
                            f'<div class="alert-box alert-info" style="color: #2DD4A0; border-color: #2DD4A0; background-color: rgba(45, 212, 160, 0.1);">'
                            f'✅ No redundant feature pairs detected at threshold <strong>{threshold:.2f}</strong>.'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                        st.write("All selected numeric columns are linearly distinct under this threshold.")

        # 2. 2D Scatter Matrix Tab
        with tab_scatter:
            default_dims = numeric_cols[:min(4, len(numeric_cols))]
            col_sel1, col_sel2 = st.columns([8, 4])
            with col_sel1:
                selected_dims = st.multiselect(
                    "Select dimensions to plot (Max 6 recommended):",
                    numeric_cols,
                    default=default_dims,
                    key="scatter_matrix_dims_tier2"
                )
            with col_sel2:
                color_var = st.selectbox(
                    "Color Mapping Variable (Optional):",
                    ["None"] + df.columns.tolist(),
                    key="scatter_matrix_color_tier2"
                )
            
            if len(selected_dims) < 2:
                st.info("💡 Please select at least 2 dimensions to generate the scatter matrix.")
            else:
                if len(selected_dims) > 6:
                    st.markdown('<div class="alert-box alert-warning">⚠️ Selecting more than 6 dimensions can slow down rendering.</div>', unsafe_allow_html=True)
                
                with st.spinner("Generating Scatter Matrix..."):
                    fig_matrix = px.scatter_matrix(
                        df,
                        dimensions=selected_dims,
                        color=None if color_var == "None" else color_var,
                        opacity=0.7,
                        template="plotly_dark",
                        color_discrete_sequence=px.colors.qualitative.Safe,
                        color_continuous_scale="Plasma"
                    )
                    fig_matrix.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#EEEEF5"),
                        margin=dict(t=20, b=20, l=20, r=20)
                    )
                    fig_matrix.update_traces(diagonal_visible=True, showupperhalf=True)
                    st.plotly_chart(fig_matrix, use_container_width=True)

        # 3. 3D Scatter Explorer Tab
        with tab_3d:
            col_axes, col_extras = st.columns([7, 5])
            with col_axes:
                st.markdown("**Axis Mapping**")
                col_x, col_y, col_z = st.columns(3)
                with col_x:
                    x_var = st.selectbox("X-Axis (Numeric)", numeric_cols, index=0, key="3d_x_tier2")
                with col_y:
                    y_var = st.selectbox("Y-Axis (Numeric)", numeric_cols, index=min(1, len(numeric_cols)-1), key="3d_y_tier2")
                with col_z:
                    z_var = st.selectbox("Z-Axis (Numeric)", numeric_cols, index=min(2, len(numeric_cols)-1), key="3d_z_tier2")
            with col_extras:
                st.markdown("**Aesthetic Encodings**")
                col_col, col_sz = st.columns(2)
                with col_col:
                    color_var_3d = st.selectbox("Color mapping column", ["None"] + df.columns.tolist(), key="3d_color_val_tier2")
                with col_sz:
                    size_var_3d = st.selectbox("Size mapping column", ["None"] + numeric_cols, key="3d_size_val_tier2")
            
            with st.spinner("Generating 3D Scatter Explorer..."):
                plot_df = df.copy()
                size_param = None
                if size_var_3d != "None":
                    size_param = size_var_3d
                    plot_df[size_var_3d] = plot_df[size_var_3d].fillna(0.0)
                    min_val = plot_df[size_var_3d].min()
                    if min_val <= 0:
                        plot_df[size_var_3d] = plot_df[size_var_3d] - min_val + 0.1
                        
                fig_3d = px.scatter_3d(
                    plot_df,
                    x=x_var,
                    y=y_var,
                    z=z_var,
                    color=None if color_var_3d == "None" else color_var_3d,
                    size=size_param,
                    opacity=0.8,
                    template="plotly_dark",
                    color_discrete_sequence=px.colors.qualitative.Safe,
                    color_continuous_scale="Plasma"
                )
                fig_3d.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#EEEEF5"),
                    margin=dict(t=10, b=10, l=10, r=10),
                    scene=dict(
                        xaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="#272836", showbackground=True),
                        yaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="#272836", showbackground=True),
                        zaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="#272836", showbackground=True)
                    )
                )
                st.plotly_chart(fig_3d, use_container_width=True)

    st.markdown("---")

    # ----------------- SECTION 3: INTERACTIVE CUSTOM CHART BUILDER -----------------
    st.markdown('<div class="section-header">🎨 Section 3: Interactive Custom Chart Builder</div>', unsafe_allow_html=True)
    st.write("Configure and create custom 2D or 3D charts tailored to specific features of interest below.")
    
    st.markdown('<div class="diag-card">', unsafe_allow_html=True)
    
    col_ctrl_type, col_ctrl_x, col_ctrl_y = st.columns(3)
    
    with col_ctrl_type:
        chart_type = st.selectbox(
            "Select Chart Type",
            [
                "Scatter Plot", "Line Chart", "Bar Chart", "Histogram", 
                "Box Plot", "Violin Plot", "Heatmap (2D Density)", 
                "Donut Chart", "World Map (Choropleth)", 
                "3D Scatter Plot", "3D Line Plot"
            ]
        )
        
    with col_ctrl_x:
        # Dynamically display appropriate label based on chart type
        if chart_type == "World Map (Choropleth)":
            x_label = "Locations Column (Country Name/Code)"
        elif chart_type == "Donut Chart":
            x_label = "Slices/Names Column (Categorical)"
        else:
            x_label = "X-Axis Column"
            
        x_col = st.selectbox(x_label, df.columns.tolist())
        
    with col_ctrl_y:
        # Determine if Y/Z axes or values are shown
        if chart_type in ["3D Scatter Plot", "3D Line Plot"]:
            y_col = st.selectbox("Y-Axis Column", df.columns.tolist(), index=min(1, len(df.columns)-1))
        elif chart_type == "Donut Chart":
            y_col = st.selectbox("Values Column (Optional, Numeric)", ["None"] + numeric_cols)
        elif chart_type == "World Map (Choropleth)":
            y_col = st.selectbox("Color Value Column (Optional)", ["None"] + df.columns.tolist())
        else:
            y_opts = ["None"] + df.columns.tolist()
            default_y_index = 0
            if chart_type in ["Scatter Plot", "Line Chart", "Box Plot", "Violin Plot"] and len(df.columns) > 1:
                other_cols = [c for c in df.columns if c != x_col]
                if other_cols:
                    default_y_index = y_opts.index(other_cols[0])
            y_col = st.selectbox("Y-Axis Column (Optional/Required)", y_opts, index=default_y_index)

    # 3D specific selectors
    z_col = None
    if chart_type in ["3D Scatter Plot", "3D Line Plot"]:
        col_z_axis, col_z_space = st.columns([4, 8])
        with col_z_axis:
            z_col = st.selectbox("Z-Axis Column", df.columns.tolist(), index=min(2, len(df.columns)-1))

    # Secondary settings expander
    with st.expander("🛠️ Advanced Aesthetic & Subplot Settings", expanded=False):
        col_sub_a, col_sub_b, col_sub_c = st.columns(3)
        with col_sub_a:
            color_col = st.selectbox("Color Grouping (Hue)", ["None"] + df.columns.tolist(), key="custom_color_v3")
        with col_sub_b:
            facet_col = st.selectbox("Facet Subplots Column (2D Only)", ["None"] + categorical_cols, key="custom_facet_v3")
        with col_sub_c:
            opacity_val = st.slider("Opacity / Transparency", 0.1, 1.0, 0.8, key="custom_opacity_v3")
            
        col_opts1, col_opts2 = st.columns(2)
        with col_opts1:
            custom_title = st.text_input("Custom Chart Title", value="", key="custom_title_v3")
            log_x = st.checkbox("Log scale X-axis", key="custom_log_x_v3")
        with col_opts2:
            log_y = st.checkbox("Log scale Y-axis", key="custom_log_y_v3")
            # Extra context controls per chart type
            if chart_type == "Histogram":
                hist_bins = st.slider("Histogram Bins", 5, 100, 30, key="custom_hist_bins_v3")
            elif chart_type == "Bar Chart":
                barmode_val = st.radio("Bar Mode", ["group", "stack", "overlay"], index=0, horizontal=True, key="custom_barmode_v3")

    # Render custom chart
    with st.spinner("Rendering Interactive Chart..."):
        try:
            kwargs = {
                "data_frame": df,
                "opacity": opacity_val,
                "template": "plotly_dark",
                "color_discrete_sequence": px.colors.qualitative.Safe,
                "color_continuous_scale": "Plasma"
            }
            
            # Map specific layout arguments
            if chart_type in ["3D Scatter Plot", "3D Line Plot"]:
                kwargs["x"] = x_col
                kwargs["y"] = y_col
                kwargs["z"] = z_col
            elif chart_type == "Donut Chart":
                kwargs["names"] = x_col
                if y_col != "None":
                    kwargs["values"] = y_col
            elif chart_type == "World Map (Choropleth)":
                kwargs["locations"] = x_col
                if y_col != "None":
                    kwargs["color"] = y_col
                else:
                    # Fallback to counts if color column is omitted
                    map_df_custom = df[x_col].dropna().value_counts().reset_index()
                    map_df_custom.columns = [x_col, "Count"]
                    kwargs["data_frame"] = map_df_custom
                    kwargs["color"] = "Count"
                
                # Dynamic location mode check
                loc_mode_custom = "country names"
                if "state" in x_col.lower():
                    loc_mode_custom = "USA-states"
                elif "iso" in x_col.lower():
                    loc_mode_custom = "ISO-3"
                kwargs["locationmode"] = loc_mode_custom
            else:
                kwargs["x"] = x_col
                if y_col != "None":
                    kwargs["y"] = y_col
                elif chart_type in ["Scatter Plot", "Line Chart", "Box Plot", "Violin Plot"]:
                    st.warning(f"⚠️ Y-Axis Column is required for {chart_type}.")
                    st.stop()
            
            # Color coding (exclude if already mapped)
            if color_col != "None" and chart_type not in ["World Map (Choropleth)", "Donut Chart"]:
                kwargs["color"] = color_col
                
            # Facet check (only supported in 2D layouts)
            if facet_col != "None" and chart_type not in ["3D Scatter Plot", "3D Line Plot", "World Map (Choropleth)", "Donut Chart"]:
                kwargs["facet_col"] = facet_col
                kwargs["facet_col_wrap"] = 2
                
            # Log scales (applicable to continuous axes in 2D/3D)
            if chart_type not in ["Donut Chart", "World Map (Choropleth)"]:
                kwargs["log_x"] = log_x
                kwargs["log_y"] = log_y

            # Render
            if chart_type == "Scatter Plot":
                fig_custom = px.scatter(**kwargs)
            elif chart_type == "Line Chart":
                fig_custom = px.line(**kwargs)
            elif chart_type == "Bar Chart":
                kwargs["barmode"] = barmode_val
                fig_custom = px.bar(**kwargs)
            elif chart_type == "Histogram":
                kwargs["nbins"] = hist_bins
                fig_custom = px.histogram(**kwargs)
            elif chart_type == "Box Plot":
                fig_custom = px.box(**kwargs)
            elif chart_type == "Violin Plot":
                fig_custom = px.violin(**kwargs)
            elif chart_type == "Heatmap (2D Density)":
                fig_custom = px.density_heatmap(**kwargs)
            elif chart_type == "Donut Chart":
                kwargs["hole"] = 0.4
                fig_custom = px.pie(**kwargs)
            elif chart_type == "World Map (Choropleth)":
                fig_custom = px.choropleth(**kwargs)
            elif chart_type == "3D Scatter Plot":
                fig_custom = px.scatter_3d(**kwargs)
            elif chart_type == "3D Line Plot":
                fig_custom = px.line_3d(**kwargs)

            # Styling polish
            default_title = f"{chart_type}: {x_col}"
            if y_col != "None" and chart_type not in ["Donut Chart", "World Map (Choropleth)"]:
                default_title += f" vs {y_col}"
            if z_col:
                default_title += f" vs {z_col}"
                
            title_text = custom_title if custom_title else default_title
            
            fig_custom.update_layout(
                title=dict(text=title_text, font=dict(family="Space Grotesk", size=16)),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#EEEEF5")
            )
            
            if chart_type not in ["3D Scatter Plot", "3D Line Plot", "World Map (Choropleth)"]:
                fig_custom.update_layout(plot_bgcolor="rgba(0,0,0,0)")
                
            st.plotly_chart(fig_custom, use_container_width=True)
            
        except Exception as chart_err:
            st.markdown(f"""
            <div class="alert-box alert-danger">
                <strong>⚠️ Chart Execution Failure</strong><br>
                Plotly Express failed to render with this dataset configuration. Ensure variables are compatible (e.g. numeric variables for continuous axes or valid country names for map coordinates).<br><br>
                <em>Details: {str(chart_err)}</em>
            </div>
            """, unsafe_allow_html=True)
            
    st.markdown('</div>', unsafe_allow_html=True)
