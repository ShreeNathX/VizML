import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px

from sklearn.metrics import confusion_matrix, classification_report
from src.ml_engine import MLEngine

st.set_page_config(page_title="VizML — Machine Learning Studio", layout="wide")

# Senior Designer Custom CSS Design System
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

    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }

    .stApp {
        background-color: var(--bg-main);
        background-image: 
            radial-gradient(at 50% 10%, rgba(99, 102, 241, 0.08) 0px, transparent 45%),
            radial-gradient(at 80% 85%, rgba(16, 185, 129, 0.05) 0px, transparent 40%);
        background-attachment: fixed;
    }

    section[data-testid="stSidebar"] {
        background: #0C0D16 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.06) !important;
    }

    /* Header Container */
    .header-box {
        background: linear-gradient(135deg, rgba(18, 20, 32, 0.9) 0%, rgba(12, 13, 22, 0.95) 100%);
        border: 1px solid var(--border-color);
        border-radius: 18px;
        padding: 32px 36px;
        margin-bottom: 28px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
    }
    .header-eyebrow {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: #A5B4FC;
        margin-bottom: 8px;
    }
    .page-title {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 34px;
        font-weight: 800;
        letter-spacing: -0.8px;
        background: linear-gradient(135deg, #FFFFFF 0%, #C7D2FE 60%, #818CF8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
    }
    .page-sub {
        font-size: 14px;
        color: var(--text-secondary);
        max-width: 650px;
    }

    .section-header {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 18px;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: 0.3px;
        margin-top: 24px;
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 1px solid var(--border-color);
    }

    .badge-auto {
        display: inline-block;
        background: rgba(16, 185, 129, 0.12);
        color: #34D399;
        font-family: 'JetBrains Mono', monospace;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        padding: 4px 10px;
        border-radius: 20px;
        border: 1px solid rgba(16, 185, 129, 0.3);
        margin-left: 8px;
        vertical-align: middle;
    }
    .badge-manual {
        display: inline-block;
        background: rgba(99, 102, 241, 0.12);
        color: #A5B4FC;
        font-family: 'JetBrains Mono', monospace;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        padding: 4px 10px;
        border-radius: 20px;
        border: 1px solid rgba(99, 102, 241, 0.3);
        margin-left: 8px;
        vertical-align: middle;
    }

    .card-box {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
    }

    div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace;
        font-size: 26px;
        font-weight: 700;
        color: var(--accent-indigo) !important;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: var(--text-muted) !important;
    }
    </style>
""", unsafe_allow_html=True)

# Page Header
st.markdown("""
<div class="header-box">
    <div class="header-eyebrow">Phase 03 — Predictive Modeling</div>
    <div class="page-title">Machine Learning Studio</div>
    <div class="page-sub">
        Automated task inference, feature preprocessing pipelines, multi-algorithm cross-validation leaderboards, residual & confusion diagnostics, and live What-If scenario prediction.
    </div>
</div>
""", unsafe_allow_html=True)

df = st.session_state.get("df", None)
if df is None:
    st.markdown("""
    <div class="card-box" style="text-align: center; padding: 48px;">
        <h3 style="color: #F59E0B; margin-bottom: 14px; font-family: 'Plus Jakarta Sans', sans-serif;">
            No Active Dataset Session
        </h3>
        <p style="color: #94A3B8; margin-bottom: 20px; font-size: 14px; max-width: 480px; margin-left: auto; margin-right: auto;">
            You need to upload and clean a dataset in the Data Curation module before training machine learning models.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.page_link("pages/1_Data_Curation.py", label="Go to Data Curation Engine")
    st.stop()

all_cols = df.columns.tolist()
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = [c for c in all_cols if c not in numeric_cols]

st.info(
    f"Active Dataset Workspace — **{df.shape[0]:,} rows x {df.shape[1]} columns** "
    f"({len(numeric_cols)} numeric, {len(categorical_cols)} categorical/datetime)."
)

st.markdown('<div class="section-header">1 · Task & Target Setup</div>', unsafe_allow_html=True)

col_target, col_task = st.columns([2, 2])

with col_target:
    default_target_idx = len(all_cols) - 1
    target_col = st.selectbox(
        "Target Column (Y)",
        options=all_cols,
        index=default_target_idx,
        key="ml_target_col",
        help="Select the column you want the model to predict."
    )

auto_task, auto_reason = MLEngine.auto_detect_task(df, target_col)

with col_task:
    st.markdown(f"**Auto Recommendation**: `{auto_task}` <span style='font-size:12px; color:#94A3B8;'>({auto_reason})</span>", unsafe_allow_html=True)
    task_type = st.radio(
        "Task Type",
        options=["Classification", "Regression"],
        index=0 if auto_task == "Classification" else 1,
        horizontal=True,
        key="ml_task_type",
        help="Classification predicts discrete categories; Regression predicts continuous numeric values."
    )

st.markdown('<div class="section-header">2 · Feature Engineering & Selection</div>', unsafe_allow_html=True)

feat_mode = st.radio(
    "Feature Mode",
    options=["Auto (All features - Encoded automatically)", "Manual (Select specific features)"],
    horizontal=True,
    key="ml_feat_mode",
    help="Auto mode automatically cleans, imputes, encodes categoricals/datetimes, and drops uninformative IDs."
)

feature_pool = [c for c in all_cols if c != target_col]

if feat_mode.startswith("Auto"):
    selected_features = feature_pool
    st.markdown(
        f'<span class="badge-auto">Auto ML Engine</span> Utilizing all **{len(selected_features)}** remaining dataset columns. '
        f'Categorical & Datetime features will be encoded automatically.',
        unsafe_allow_html=True
    )
else:
    st.markdown('<span class="badge-manual">Manual Selection</span>', unsafe_allow_html=True)
    selected_features = st.multiselect(
        "Feature Columns (X)",
        options=feature_pool,
        default=feature_pool,
        key="ml_features",
        help="Pick features to include in training."
    )

if not selected_features:
    st.warning("Please select at least one feature column to continue.")
    st.stop()

st.markdown('<div class="section-header">3 · Model Selection & Benchmarking Setup</div>', unsafe_allow_html=True)

model_dict = MLEngine.get_model_dictionary(task_type)
model_options = list(model_dict.keys())

col_m1, col_m2 = st.columns([3, 1])

with col_m1:
    selected_models = st.multiselect(
        "Models to Benchmark",
        options=model_options,
        default=model_options[:min(4, len(model_options))],
        key="ml_selected_models",
        help="All selected models will be trained, evaluated, and ranked."
    )

with col_m2:
    test_size = st.slider("Test Split %", 10, 40, 20, 5, key="ml_test_size")
    cv_folds = st.slider("Cross-Val Folds", 2, 10, 5, 1, key="ml_cv")

if not selected_models:
    st.warning("Please select at least one model to train.")
    st.stop()

st.markdown("<br>", unsafe_allow_html=True)
train_btn = st.button("Train Models (Automated Pipeline)", type="primary", use_container_width=True, key="ml_train_btn")

if train_btn:
    prog_bar = st.progress(0, text="Initializing automated pipeline...")

    def update_progress(pct, text):
        prog_bar.progress(pct, text=text)

    try:
        output = MLEngine.train_models(
            df=df,
            target_col=target_col,
            selected_features=selected_features,
            task_type=task_type,
            selected_model_names=selected_models,
            test_size=test_size,
            cv_folds=cv_folds,
            progress_callback=update_progress
        )

        st.session_state["ml_engine_output"] = output
        st.success(f"Automated pipeline trained {len(output['results'])} model(s) successfully.")

        if output.get("cv_warning"):
            st.warning(f"Cross-Validation Notice: {output['cv_warning']}")

        if output.get("errors"):
            for err in output["errors"]:
                st.warning(f"Notice: {err}")

    except Exception as e:
        st.error(f"Training failed: {str(e)}")
        st.stop()

ml_output = st.session_state.get("ml_engine_output", None)

if ml_output:
    results = ml_output["results"]
    pipeline_meta = ml_output["pipeline"]
    task_done = ml_output["task_type"]
    target_done = ml_output["target_col"]

    st.markdown('<div class="section-header">4 · Model Leaderboard</div>', unsafe_allow_html=True)

    df_results = pd.DataFrame([
        {k: v for k, v in r.items() if not k.startswith("_")}
        for r in results
    ])

    best_idx = max(range(len(results)), key=lambda i: results[i]["_cv_mean"])
    best_model_name = results[best_idx]["Model"]
    best_model_obj = results[best_idx]["_model"]

    num_display = [c for c in df_results.columns if c != "Model"]
    try:
        styled = df_results.style.highlight_max(subset=num_display, color="#1e1f4a")
    except Exception:
        styled = df_results.style
    st.dataframe(styled, use_container_width=True, hide_index=True)
    st.caption(f"Top Performing Model: **{best_model_name}** (CV Score: {results[best_idx]['CV Score (mean)']})")

    primary_metric = "Test Accuracy" if task_done == "Classification" else "Test R²"
    if primary_metric in df_results.columns:
        fig_bar = px.bar(
            df_results, x="Model", y=primary_metric, color="Model",
            color_discrete_sequence=px.colors.qualitative.Vivid,
            title=f"{primary_metric} Benchmark Comparison Across Models",
            template="plotly_dark",
        )
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False, font_family="Inter", title_font_family="Plus Jakarta Sans",
            font=dict(color="#F1F3F9")
        )
        st.plotly_chart(fig_bar, use_container_width=True, key="leaderboard_bar")

    st.markdown('<div class="section-header">5 · Per-Model Diagnostic Detail</div>', unsafe_allow_html=True)
    tabs = st.tabs([r["Model"] for r in results])
    y_test_g = ml_output["y_test"]
    le_target = pipeline_meta.get("le_target", None)

    for tab, res in zip(tabs, results):
        with tab:
            model_name = res["Model"]
            if task_done == "Classification":
                y_pred = res["_y_pred"]
                label_ids = np.unique(np.concatenate([y_test_g, y_pred]))
                if le_target is not None:
                    labels = [str(le_target.inverse_transform([int(l)])[0]) for l in label_ids]
                else:
                    labels = [str(l) for l in label_ids]

                cm = confusion_matrix(y_test_g, y_pred, labels=label_ids)
                fig_cm = px.imshow(
                    cm, x=labels, y=labels, text_auto=True,
                    color_continuous_scale="Purples",
                    title="Confusion Matrix Diagnostics", template="plotly_dark",
                    labels=dict(x="Predicted", y="Actual"),
                )
                fig_cm.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_family="Inter", title_font_family="Plus Jakarta Sans",
                    font=dict(color="#F1F3F9")
                )
                st.plotly_chart(fig_cm, use_container_width=True, key=f"cm_{model_name}")

                report = classification_report(
                    y_test_g, y_pred, labels=label_ids, target_names=labels,
                    output_dict=True, zero_division=0
                )
                st.markdown("**Classification Report Breakdown:**")
                st.dataframe(
                    pd.DataFrame(report).transpose().round(4),
                    use_container_width=True,
                    key=f"report_{model_name}",
                )

            else:
                y_pred = res["_y_pred"]
                y_test_local = res["_y_test"]

                col_sc1, col_sc2 = st.columns(2)
                with col_sc1:
                    fig_scatter = px.scatter(
                        x=y_test_local, y=y_pred,
                        labels={"x": "Actual Values", "y": "Predicted Values"},
                        title="Actual vs Predicted Regression Alignment",
                        template="plotly_dark",
                        color_discrete_sequence=["#6366F1"],
                    )
                    mn = min(float(y_test_local.min()), float(y_pred.min()))
                    mx = max(float(y_test_local.max()), float(y_pred.max()))
                    fig_scatter.add_shape(
                        type="line", x0=mn, y0=mn, x1=mx, y1=mx,
                        line=dict(color="#10B981", width=2, dash="dash")
                    )
                    fig_scatter.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font_family="Inter", title_font_family="Plus Jakarta Sans",
                        font=dict(color="#F1F3F9")
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True, key=f"scatter_{model_name}")

                with col_sc2:
                    residuals = y_test_local - y_pred
                    fig_res = px.histogram(
                        x=residuals, nbins=30, title="Residual Error Distribution",
                        template="plotly_dark", color_discrete_sequence=["#F43F5E"],
                        labels={"x": "Residual Error"},
                    )
                    fig_res.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font_family="Inter", title_font_family="Plus Jakarta Sans",
                        font=dict(color="#F1F3F9")
                    )
                    st.plotly_chart(fig_res, use_container_width=True, key=f"residual_{model_name}")

    tree_results = [r for r in results if hasattr(r["_model"], "feature_importances_")]
    if tree_results:
        st.markdown('<div class="section-header">6 · Feature Importance Analysis</div>', unsafe_allow_html=True)
        feat_names = pipeline_meta["encoded_feature_names"]
        fi_tabs = st.tabs([r["Model"] for r in tree_results])

        for tab, res in zip(fi_tabs, tree_results):
            with tab:
                importances = res["_model"].feature_importances_
                n_feats = min(len(feat_names), len(importances))
                fi_df = pd.DataFrame({
                    "Feature": feat_names[:n_feats],
                    "Importance": importances[:n_feats]
                }).sort_values("Importance", ascending=False)

                fig_fi = px.bar(
                    fi_df.head(20), x="Importance", y="Feature", orientation="h",
                    color="Importance",
                    color_continuous_scale=["#6366F1", "#10B981"],
                    title=f"Top Feature Importance — {res['Model']}",
                    template="plotly_dark",
                )
                fig_fi.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_family="Inter", title_font_family="Plus Jakarta Sans",
                    yaxis={"categoryorder": "total ascending"},
                    coloraxis_showscale=False,
                    font=dict(color="#F1F3F9")
                )
                st.plotly_chart(fig_fi, use_container_width=True, key=f"fi_{res['Model']}")

    st.markdown('<div class="section-header">7 · Interactive Prediction Studio (What-If Tester) & Export</div>', unsafe_allow_html=True)

    col_play1, col_play2 = st.columns([3, 2])

    with col_play1:
        st.markdown("### Live Model Inference")
        selected_inference_model_name = st.selectbox(
            "Select Model for Inference",
            options=[r["Model"] for r in results],
            index=[r["Model"] for r in results].index(best_model_name),
            key="ml_inference_model_choice"
        )

        chosen_model_res = next(r for r in results if r["Model"] == selected_inference_model_name)
        chosen_model = chosen_model_res["_model"]

        raw_feats = pipeline_meta["raw_features"]
        st.markdown("Specify feature inputs to test What-If scenario predictions:")

        input_data = {}
        grid_cols = st.columns(min(3, max(1, len(raw_feats))))

        for idx, feature_name in enumerate(raw_feats):
            col_target_grid = grid_cols[idx % len(grid_cols)]
            with col_target_grid:
                col_series = df[feature_name].dropna()
                if pd.api.types.is_numeric_dtype(col_series):
                    mean_v = float(col_series.mean()) if len(col_series) > 0 else 0.0
                    val = st.number_input(
                        f"{feature_name}",
                        value=mean_v,
                        key=f"infer_input_{feature_name}"
                    )
                    input_data[feature_name] = val
                else:
                    unique_vals = col_series.unique().tolist()[:50]
                    if not unique_vals:
                        unique_vals = ["Missing"]
                    val = st.selectbox(
                        f"{feature_name}",
                        options=unique_vals,
                        key=f"infer_input_{feature_name}"
                    )
                    input_data[feature_name] = val

        predict_btn = st.button(
            "Predict Output", type="primary", use_container_width=True, key="ml_predict_btn"
        )

        if predict_btn:
            pred_res = MLEngine.predict_single(pipeline_meta, chosen_model, input_data)
            st.markdown(f"#### Predicted `{target_done}`: **{pred_res['prediction']}**")

            if pred_res.get("probabilities"):
                df_proba = pd.DataFrame([
                    {"Class": k, "Probability": v}
                    for k, v in pred_res["probabilities"].items()
                ])
                fig_prob = px.bar(
                    df_proba, x="Probability", y="Class", orientation="h",
                    color="Probability", color_continuous_scale="Purples",
                    title="Class Confidence Probabilities",
                    template="plotly_dark"
                )
                fig_prob.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_family="Inter", title_font_family="Plus Jakarta Sans",
                    font=dict(color="#F1F3F9")
                )
                st.plotly_chart(fig_prob, use_container_width=True, key="infer_proba_chart")

    with col_play2:
        st.markdown("### Model Bundle Export")
        st.write("Export the trained model pipeline bundle (`.pkl`) complete with preprocessing metadata.")

        export_bundle = {
            "model_name": best_model_name,
            "model": best_model_obj,
            "pipeline": pipeline_meta,
            "task_type": task_done,
            "target_col": target_done,
        }

        bundle_bytes = pickle.dumps(export_bundle)
        st.download_button(
            label=f"Export Best Model ({best_model_name})",
            data=bundle_bytes,
            file_name=f"vizml_{best_model_name.lower().replace(' ', '_')}_model.pkl",
            mime="application/octet-stream",
            use_container_width=True
        )


