import io
import pickle

import numpy as np
import pandas as pd
import plotly.express as px
import sklearn
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from src.ml_engine import MLEngine
from src.styles import (
    CLAY_COLOR_SEQUENCE,
    apply_clay_theme,
    inject_css,
    render_html,
)

st.set_page_config(page_title="VizML: ML Studio", layout="wide")
inject_css()

render_html(
    """
    <div class="bp-card">
        <div class="bp-header" style="color:var(--lime);">Stage 03 · Model</div>
        <div class="page-title">Machine Learning Studio</div>
        <div class="page-sub">
            Automated task inference, feature preprocessing pipelines, multi-algorithm cross-validation leaderboards,
            residual/confusion diagnostics, feature importance rankings, and live What-If scenario simulation.
        </div>
    </div>
    """
)

df = st.session_state.get("df", None)
if df is None or not isinstance(df, pd.DataFrame) or df.empty:
    render_html(
        """
        <div class="alert-box alert-warning">
            <strong>No active dataset found.</strong> Upload and curate a dataset in Phase 01 before configuring predictive models.
        </div>
        """
    )
    st.page_link("pages/1_Data_Curation.py", label="Launch Phase 01: Data Curation Engine")
    st.stop()

all_cols = df.columns.tolist()
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = [c for c in all_cols if c not in numeric_cols]

# -------------------------------------------------------------
# Defensive Session State Sanitization (Prevent NoneType Iteration Errors)
# -------------------------------------------------------------
if "ml_models" in st.session_state and (st.session_state["ml_models"] is None or not isinstance(st.session_state["ml_models"], (list, tuple))):
    del st.session_state["ml_models"]

if "ml_features" in st.session_state and (st.session_state["ml_features"] is None or not isinstance(st.session_state["ml_features"], (list, tuple))):
    del st.session_state["ml_features"]

if "ml_target" in st.session_state and (st.session_state["ml_target"] is None or not isinstance(st.session_state["ml_target"], str) or st.session_state["ml_target"] not in all_cols):
    del st.session_state["ml_target"]

if "ml_task" in st.session_state and st.session_state["ml_task"] not in ["Classification", "Regression"]:
    del st.session_state["ml_task"]

if "ml_test_size" in st.session_state and not isinstance(st.session_state["ml_test_size"], (int, float)):
    del st.session_state["ml_test_size"]

if "ml_cv" in st.session_state and not isinstance(st.session_state["ml_cv"], int):
    del st.session_state["ml_cv"]


render_html(
    f"""
    <div class="alert-box alert-info">
        <strong>Active Dataset:</strong> <code>{st.session_state.get('current_file_name', 'Active Dataset')}</code>
        ({df.shape[0]:,} rows × {df.shape[1]} cols | {len(numeric_cols)} numeric, {len(categorical_cols)} categorical).
    </div>
    """
)

# -------------------------------------------------------------
# 1. Task & Target Setup
# -------------------------------------------------------------
render_html('<div class="step-header"><span class="step-num indigo">01</span><span class="step-title">Task and Target Setup</span></div>')

target_default_idx = len(all_cols) - 1

ct, ck = st.columns([2, 2])
with ct:
    target_col = st.selectbox(
        "Predictive Target Variable (Y)",
        options=all_cols,
        index=target_default_idx,
        key="ml_target",
        help="The column the algorithm will learn to predict.",
    )

try:
    auto_task, auto_reason = MLEngine.auto_detect_task(df, target_col)
except Exception as err:
    auto_task, auto_reason = "Classification", f"Defaulted to classification ({err})"

with ck:
    render_html(
        f"**Heuristic Task Inference:** <span class='badge badge-gold'>{auto_task}</span> : <span style='color:var(--ink-soft);font-size:11px;'>{auto_reason}</span>"
    )
    task_type = st.radio(
        "Modeling Paradigm",
        ["Classification", "Regression"],
        index=0 if auto_task == "Classification" else 1,
        horizontal=True,
        key="ml_task",
    )

# -------------------------------------------------------------
# 2. Feature Selection
# -------------------------------------------------------------
render_html('<div class="step-header"><span class="step-num orange">02</span><span class="step-title">Feature Engineering and Selection</span></div>')

feat_mode = st.radio(
    "Feature Selection Mode",
    ["Auto (all features, automatic one-hot/label encoding)", "Manual (select specific feature set)"],
    horizontal=True,
    key="ml_feat_mode",
)
feature_pool = [c for c in all_cols if c != target_col]

# Clean selected features if columns changed
if "ml_features" in st.session_state:
    if not isinstance(st.session_state["ml_features"], (list, tuple)) or not all(f in feature_pool for f in st.session_state["ml_features"]):
        del st.session_state["ml_features"]

if feat_mode.startswith("Auto"):
    selected_features = feature_pool
    render_html(
        f'<span class="badge badge-emerald">SYS.AUTO</span>&nbsp; Utilizing all <strong>{len(selected_features)}</strong> available features. Categorical and datetime attributes will be encoded automatically.'
    )
else:
    render_html('<span class="badge badge-indigo">SYS.MANUAL</span>')
    selected_features = st.multiselect(
        "Selected Feature Predictors (X)", options=feature_pool, default=feature_pool, key="ml_features"
    )

if not selected_features:
    st.warning("Select at least one feature predictor column.")
    st.stop()

# -------------------------------------------------------------
# 3. Model Selection
# -------------------------------------------------------------
render_html('<div class="step-header"><span class="step-num cyan">03</span><span class="step-title">Model Selection and Benchmarking</span></div>')

try:
    model_dict = MLEngine.get_model_dictionary(task_type)
    model_options = list(model_dict.keys())
except Exception as err:
    st.error(f"Could not load algorithms: {err}")
    model_options = []

# Validate existing selection
if "ml_models" in st.session_state:
    if not isinstance(st.session_state["ml_models"], (list, tuple)) or not all(m in model_options for m in st.session_state["ml_models"]):
        del st.session_state["ml_models"]

default_models = model_options[:min(4, len(model_options))] if model_options else []

cm1, cm2 = st.columns([3, 1])
with cm1:
    selected_models = st.multiselect(
        "Candidate Algorithms to Train & Benchmark",
        options=model_options,
        default=default_models,
        key="ml_models",
    )
with cm2:
    test_size = st.slider("Holdout Test Split %", 10, 40, 20, 5, key="ml_test_size")
    cv_folds = st.slider("K-Fold Cross Validation", 2, 10, 5, 1, key="ml_cv")

if not selected_models:
    st.warning("Select at least one algorithm to train.")
    st.stop()

st.markdown("<br>", unsafe_allow_html=True)

if len(df) > 50_000:
    st.warning(
        f"Your dataset has {len(df):,} rows. Training may take a couple of minutes on Streamlit Cloud. "
        "Consider using fewer models if it times out."
    )

train_btn = st.button("Execute Automated Training Pipeline", type="primary", key="ml_train_btn")

if train_btn:
    prog = st.progress(0, text="Initializing model training pipeline...")
    try:
        output = MLEngine.train_models(
            df=df,
            target_col=target_col,
            selected_features=selected_features,
            task_type=task_type,
            selected_model_names=selected_models,
            test_size=test_size,
            cv_folds=cv_folds,
            progress_callback=lambda p, t: prog.progress(p, text=t),
        )
        st.session_state["ml_engine_output"] = output
        render_html(
            f'<div class="alert-box alert-success">Successfully trained <strong>{len(output["results"])}</strong> model(s). Leaderboard updated.</div>'
        )
        if output.get("cv_warning"):
            st.warning(output["cv_warning"])
        for e in output.get("errors", []):
            st.warning(e)
    except Exception as err:
        st.error(f"Training failed: {err}")
        st.stop()

ml_output = st.session_state.get("ml_engine_output", None)

if ml_output and isinstance(ml_output, dict) and "results" in ml_output and ml_output["results"]:
    results = ml_output["results"]
    pipeline_meta = ml_output.get("pipeline", {})
    task_done = ml_output.get("task_type", task_type)
    target_done = ml_output.get("target_col", target_col)

    # ---------------------------------------------------------
    # 4. Leaderboard
    # ---------------------------------------------------------
    try:
        render_html('<div class="step-header"><span class="step-num lime">04</span><span class="step-title">Multi-Model Performance Leaderboard</span></div>')
        df_res = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in results])
        best_idx = max(range(len(results)), key=lambda i: results[i].get("_cv_mean", 0))
        best_name = results[best_idx]["Model"]
        best_obj = results[best_idx]["_model"]

        # Cache in session state for Phase 04 reports
        st.session_state["ml_results"] = df_res
        st.session_state["ml_best_name"] = best_name
        st.session_state["ml_pipeline"] = pipeline_meta
        st.session_state["ml_task_result"] = task_done
        st.session_state["ml_target_result"] = target_done

        st.dataframe(df_res, hide_index=True, width='stretch')
        cv_score_val = results[best_idx].get('CV Score (mean)', 'N/A')
        st.caption(f"TELEMETRY: Champion Model: **{best_name}** (CV Score: {cv_score_val})")

        pm = "Test Accuracy" if task_done == "Classification" else "Test R²"
        if pm in df_res.columns:
            fig_bar = px.bar(
                df_res,
                x="Model",
                y=pm,
                color="Model",
                color_discrete_sequence=CLAY_COLOR_SEQUENCE,
            )
            apply_clay_theme(fig_bar, title=f"Holdout Benchmark Score ({pm})")
            fig_bar.update_layout(showlegend=False)
            st.plotly_chart(fig_bar, theme=None, key="lb_bar", width='stretch')
    except Exception as err:
        st.error(f"Error rendering leaderboard: {err}")

    # ---------------------------------------------------------
    # 5. Diagnostics
    # ---------------------------------------------------------
    try:
        render_html('<div class="step-header"><span class="step-num orange">05</span><span class="step-title">Diagnostic Confusion and Error Matrices</span></div>')
        tabs = st.tabs([r["Model"] for r in results])
        y_test_g = ml_output.get("y_test")
        le_target = pipeline_meta.get("le_target", None)

        for tab, res in zip(tabs, results):
            with tab:
                mn = res["Model"]
                if task_done == "Classification" and y_test_g is not None:
                    yp = res.get("_y_pred")
                    if yp is not None:
                        lids = np.unique(np.concatenate([y_test_g, yp]))
                        labels = (
                            [str(le_target.inverse_transform([int(l)])[0]) for l in lids]
                            if le_target and hasattr(le_target, "inverse_transform")
                            else [str(l) for l in lids]
                        )
                        cm_arr = confusion_matrix(y_test_g, yp, labels=lids)
                        fig_cm = px.imshow(
                            cm_arr,
                            x=labels,
                            y=labels,
                            text_auto=True,
                            color_continuous_scale=["#EDE9FE", "#8B7DF0", "#6C5CE7"],
                            labels=dict(x="Predicted Class", y="Actual Ground Truth"),
                        )
                        apply_clay_theme(fig_cm, title=f"Confusion Matrix: {mn}")
                        st.plotly_chart(fig_cm, theme=None, key=f"cm_{mn}", width='stretch')
                        rpt = classification_report(
                            y_test_g, yp, labels=lids, target_names=labels, output_dict=True, zero_division=0
                        )
                        st.dataframe(pd.DataFrame(rpt).transpose().round(4), key=f"rpt_{mn}", width='stretch')
                elif task_done == "Regression":
                    yp = res.get("_y_pred")
                    yt = res.get("_y_test", y_test_g)
                    if yp is not None and yt is not None:
                        s1, s2 = st.columns(2)
                        with s1:
                            fig_s = px.scatter(
                                x=yt,
                                y=yp,
                                labels={"x": "Actual Ground Truth", "y": "Predicted Response"},
                                color_discrete_sequence=["#6C5CE7"],
                                opacity=0.75,
                            )
                            mn_v = min(float(yt.min()), float(yp.min()))
                            mx_v = max(float(yt.max()), float(yp.max()))
                            fig_s.add_shape(
                                type="line",
                                x0=mn_v, y0=mn_v, x1=mx_v, y1=mx_v,
                                line=dict(color="#FF5D8F", width=2, dash="dash"),
                            )
                            apply_clay_theme(fig_s, title=f"Actual vs Predicted: {mn}")
                            st.plotly_chart(fig_s, theme=None, key=f"sc_{mn}", width='stretch')
                        with s2:
                            fig_r = px.histogram(
                                x=yt - yp,
                                nbins=30,
                                color_discrete_sequence=["#FF5D8F"],
                                labels={"x": "Residual Error (Actual - Predicted)"},
                            )
                            apply_clay_theme(fig_r, title=f"Residual Error Distribution: {mn}")
                            st.plotly_chart(fig_r, theme=None, key=f"res_{mn}", width='stretch')
    except Exception as err:
        st.error(f"Error rendering diagnostics: {err}")

    # ---------------------------------------------------------
    # 6. Feature Importance
    # ---------------------------------------------------------
    try:
        tree_res = [r for r in results if hasattr(r.get("_model"), "feature_importances_")]
        if tree_res:
            render_html('<div class="step-header"><span class="step-num indigo">06</span><span class="step-title">Predictive Feature Drivers</span></div>')
            feat_names = pipeline_meta.get("encoded_feature_names", [])
            fi_tabs = st.tabs([r["Model"] for r in tree_res])
            for tab, res in zip(fi_tabs, tree_res):
                with tab:
                    imps = res["_model"].feature_importances_
                    n = min(len(feat_names), len(imps))
                    fi_df = pd.DataFrame({"Feature": feat_names[:n], "Importance": imps[:n]}).sort_values(
                        "Importance", ascending=False
                    )
                    st.session_state["ml_feature_importance"] = fi_df
                    fig_fi = px.bar(
                        fi_df.head(15),
                        x="Importance",
                        y="Feature",
                        orientation="h",
                        color="Importance",
                        color_continuous_scale=["#EDE9FE", "#8B7DF0", "#6C5CE7"],
                    )
                    apply_clay_theme(fig_fi, title=f"Top Predictive Drivers: {res['Model']}")
                    fig_fi.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
                    st.plotly_chart(fig_fi, theme=None, key=f"fi_{res['Model']}", width='stretch')
    except Exception as err:
        st.error(f"Error rendering feature importance: {err}")

    # ---------------------------------------------------------
    # 7. Prediction Studio & Export
    # ---------------------------------------------------------
    render_html('<div class="step-header"><span class="step-num lime">07</span><span class="step-title">Live What-If Simulation and Model Export</span></div>')

    cp1, cp2 = st.columns([3, 2])

    with cp1:
        render_html('<div class="section-header" style="margin-top:0;font-size:14px;"><span class="badge badge-indigo">Simulation</span> Live What-If Prediction</div>')
        try:
            model_names_list = [r["Model"] for r in results]
            default_infer_idx = model_names_list.index(best_name) if best_name in model_names_list else 0

            infer_model_name = st.selectbox(
                "Inference Engine Algorithm",
                model_names_list,
                index=default_infer_idx,
                key="ml_infer_model",
            )
            chosen = next(r for r in results if r["Model"] == infer_model_name)
            raw_feats = pipeline_meta.get("raw_features", [])
            render_html("<span style='font-size:12px;color:var(--ink-soft);'>Configure scenario variables for instant inference:</span>")
            input_data = {}
            grid = st.columns(min(3, max(1, len(raw_feats))))
            for idx, feat in enumerate(raw_feats):
                with grid[idx % len(grid)]:
                    series = df[feat].dropna()
                    if pd.api.types.is_numeric_dtype(series):
                        val = st.number_input(
                            feat, value=float(series.mean()) if len(series) > 0 else 0.0, key=f"inf_{feat}"
                        )
                    else:
                        uniq = series.unique().tolist()[:50] or ["Missing"]
                        val = st.selectbox(feat, options=uniq, key=f"inf_{feat}")
                    input_data[feat] = val

            if st.button("Compute Prediction", type="primary", key="ml_predict"):
                try:
                    pred = MLEngine.predict_single(pipeline_meta, chosen["_model"], input_data)
                    render_html(
                        f"""
                        <div class="bp-display">
                            <div class="bp-expr">Predicted Response: {target_done}</div>
                            <div class="bp-value" style="color:var(--lime);">{pred['prediction']}</div>
                        </div>
                        """
                    )
                    if pred.get("probabilities"):
                        dfp = pd.DataFrame([{"Class": k, "Probability": v} for k, v in pred["probabilities"].items()])
                        fig_p = px.bar(
                            dfp,
                            x="Probability",
                            y="Class",
                            orientation="h",
                            color="Probability",
                            color_continuous_scale=["#EDE9FE", "#8B7DF0", "#6C5CE7"],
                        )
                        apply_clay_theme(fig_p, title="Class Confidence Probabilities")
                        st.plotly_chart(fig_p, theme=None, key="infer_proba", width='stretch')
                except Exception as err:
                    st.error(f"Prediction calculation error: {err}")
        except Exception as err:
            st.error(f"Inference setup error: {err}")

    with cp2:
        render_html('<div class="section-header" style="margin-top:0;font-size:14px;"><span class="badge badge-emerald">Export</span> Model Deployment Bundle</div>')
        st.write("Serialize the trained machine learning pipeline, scalers, encoders, and weights as a production `.pkl` bundle.")
        try:
            trained_sklearn_ver = pipeline_meta.get("sklearn_version", sklearn.__version__)
            current_sklearn_ver = sklearn.__version__
            if trained_sklearn_ver != current_sklearn_ver:
                st.warning(
                    f"Version mismatch: trained with scikit-learn {trained_sklearn_ver}, current environment is {current_sklearn_ver}."
                )
            bundle = {
                "model_name": best_name,
                "model": best_obj,
                "pipeline": pipeline_meta,
                "task_type": task_done,
                "target_col": target_done,
                "sklearn_version": current_sklearn_ver,
            }
            st.session_state["ml_bundle"] = bundle
            st.download_button(
                f"Export Champion Model Bundle ({best_name})",
                pickle.dumps(bundle),
                f"vizml_{best_name.lower().replace(' ', '_')}.pkl",
                "application/octet-stream",
                width='stretch',
            )
        except Exception as err:
            st.error(f"Model export packaging error: {err}")
