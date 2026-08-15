import io
import pickle
import sklearn

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from src.ml_engine import MLEngine
from src.styles import inject_css

st.set_page_config(page_title="VizML: Machine Learning Studio", layout="wide")
inject_css()

st.markdown(
    """
<div class="glass-card">
    <span class="eyebrow">Phase 03 | Predictive Modeling</span>
    <div class="page-title">Machine Learning Studio</div>
    <div class="page-sub">
        Automated task inference, preprocessing pipelines, multi-algorithm CV leaderboards,
        residual and confusion diagnostics, and live What-If scenario prediction.
    </div>
</div>
""",
    unsafe_allow_html=True,
)

df = st.session_state.get("df", None)
if df is None:
    st.markdown(
        """
    <div class="glass-card" style="text-align:center;padding:48px;">
        <div style="font-size:15px;font-weight:700;color:#D97706;margin-bottom:10px;font-family:'Plus Jakarta Sans',sans-serif;">No Active Dataset Session</div>
        <p style="color:#4B5563;font-size:14px;">Upload a dataset in Data Curation before training models.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )
    st.page_link("pages/1_Data_Curation.py", label="Go to Data Curation Engine")
    st.stop()

all_cols = df.columns.tolist()
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = [c for c in all_cols if c not in numeric_cols]

st.info(
    f"Active Dataset: **{df.shape[0]:,} rows x {df.shape[1]} columns** ({len(numeric_cols)} numeric, {len(categorical_cols)} categorical/datetime)."
)

# 1. Task & Target
st.markdown(
    '<div class="section-header"><span class="badge badge-indigo">01</span> Task & Target Setup</div>',
    unsafe_allow_html=True,
)

ct, ck = st.columns([2, 2])
with ct:
    target_col = st.selectbox(
        "Target Column (Y)",
        options=all_cols,
        index=len(all_cols) - 1,
        key="ml_target",
        help="The column the model will predict.",
    )

try:
    auto_task, auto_reason = MLEngine.auto_detect_task(df, target_col)
except Exception as err:
    auto_task, auto_reason = "Classification", str(err)

with ck:
    st.markdown(
        f"**Auto Recommendation:** `{auto_task}` — <span style='color:#4B5563;font-size:12px;'>{auto_reason}</span>",
        unsafe_allow_html=True,
    )
    task_type = st.radio(
        "Task Type",
        ["Classification", "Regression"],
        index=0 if auto_task == "Classification" else 1,
        horizontal=True,
        key="ml_task",
    )

# 2. Feature Selection
st.markdown(
    '<div class="section-header"><span class="badge badge-violet">02</span> Feature Engineering & Selection</div>',
    unsafe_allow_html=True,
)

feat_mode = st.radio(
    "Feature Mode",
    ["Auto (all features, encoded automatically)", "Manual (select specific features)"],
    horizontal=True,
    key="ml_feat_mode",
)
feature_pool = [c for c in all_cols if c != target_col]

if feat_mode.startswith("Auto"):
    selected_features = feature_pool
    st.markdown(
        f'<span class="badge badge-emerald">AUTO</span>&nbsp; Using all **{len(selected_features)}** columns. Categoricals and datetimes encoded automatically.',
        unsafe_allow_html=True,
    )
else:
    st.markdown('<span class="badge badge-indigo">MANUAL</span>', unsafe_allow_html=True)
    selected_features = st.multiselect(
        "Feature Columns (X)", options=feature_pool, default=feature_pool, key="ml_features"
    )

if not selected_features:
    st.warning("Select at least one feature column.")
    st.stop()

# 3. Model Selection
st.markdown(
    '<div class="section-header"><span class="badge badge-indigo">03</span> Model Selection & Benchmarking</div>',
    unsafe_allow_html=True,
)

try:
    model_dict = MLEngine.get_model_dictionary(task_type)
    model_options = list(model_dict.keys())
except Exception as err:
    st.error(str(err))
    model_options = []

cm1, cm2 = st.columns([3, 1])
with cm1:
    selected_models = st.multiselect(
        "Models to Benchmark",
        options=model_options,
        default=model_options[:min(4, len(model_options))],
        key="ml_models",
    )
with cm2:
    test_size = st.slider("Test Split %", 10, 40, 20, 5, key="ml_test_size")
    cv_folds = st.slider("CV Folds", 2, 10, 5, 1, key="ml_cv")

if not selected_models:
    st.warning("Select at least one model.")
    st.stop()

st.markdown("<br>", unsafe_allow_html=True)
train_btn = st.button("Train Models (Automated Pipeline)", type="primary", width="stretch", key="ml_train_btn")

if train_btn:
    prog = st.progress(0, text="Initializing...")
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
        st.success(f"Trained {len(output['results'])} model(s) successfully.")
        if output.get("cv_warning"):
            st.warning(output["cv_warning"])
        for e in output.get("errors", []):
            st.warning(e)
    except Exception as err:
        st.error(f"Training failed: {err}")
        st.stop()

ml_output = st.session_state.get("ml_engine_output", None)

if ml_output:
    results = ml_output["results"]
    pipeline_meta = ml_output["pipeline"]
    task_done = ml_output["task_type"]
    target_done = ml_output["target_col"]

    # 4. Leaderboard
    try:
        st.markdown(
            '<div class="section-header"><span class="badge badge-indigo">04</span> Model Leaderboard</div>',
            unsafe_allow_html=True,
        )
        df_res = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in results])
        best_idx = max(range(len(results)), key=lambda i: results[i]["_cv_mean"])
        best_name = results[best_idx]["Model"]
        best_obj = results[best_idx]["_model"]
        num_disp = [c for c in df_res.columns if c != "Model"]
        try:
            styled = df_res.style.highlight_max(subset=num_disp, color="rgba(99,102,241,0.15)")
        except Exception:
            styled = df_res.style
        st.dataframe(styled, width="stretch", hide_index=True)
        st.caption(f"Top Model: **{best_name}** (CV: {results[best_idx]['CV Score (mean)']})")
        pm = "Test Accuracy" if task_done == "Classification" else "Test R²"
        if pm in df_res.columns:
            fig_bar = px.bar(
                df_res,
                x="Model",
                y=pm,
                color="Model",
                color_discrete_sequence=px.colors.qualitative.Pastel,
                title=f"{pm} Benchmark",
            )
            fig_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#111827"),
                showlegend=False,
            )
            st.plotly_chart(fig_bar, width="stretch", key="lb_bar")
    except Exception as err:
        st.error(str(err))

    # 5. Diagnostics
    try:
        st.markdown(
            '<div class="section-header"><span class="badge badge-violet">05</span> Per-Model Diagnostics</div>',
            unsafe_allow_html=True,
        )
        tabs = st.tabs([r["Model"] for r in results])
        y_test_g = ml_output["y_test"]
        le_target = pipeline_meta.get("le_target", None)

        for tab, res in zip(tabs, results):
            with tab:
                mn = res["Model"]
                if task_done == "Classification":
                    yp = res["_y_pred"]
                    lids = np.unique(np.concatenate([y_test_g, yp]))
                    labels = (
                        [str(le_target.inverse_transform([int(l)])[0]) for l in lids]
                        if le_target
                        else [str(l) for l in lids]
                    )
                    cm_arr = confusion_matrix(y_test_g, yp, labels=lids)
                    fig_cm = px.imshow(
                        cm_arr,
                        x=labels,
                        y=labels,
                        text_auto=True,
                        color_continuous_scale="BuPu",
                        title="Confusion Matrix",
                        labels=dict(x="Predicted", y="Actual"),
                    )
                    fig_cm.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#111827"),
                    )
                    st.plotly_chart(fig_cm, width="stretch", key=f"cm_{mn}")
                    rpt = classification_report(
                        y_test_g, yp, labels=lids, target_names=labels, output_dict=True, zero_division=0
                    )
                    st.dataframe(pd.DataFrame(rpt).transpose().round(4), width="stretch", key=f"rpt_{mn}")
                else:
                    yp = res["_y_pred"]
                    yt = res["_y_test"]
                    s1, s2 = st.columns(2)
                    with s1:
                        fig_s = px.scatter(
                            x=yt,
                            y=yp,
                            labels={"x": "Actual", "y": "Predicted"},
                            title="Actual vs Predicted",
                            color_discrete_sequence=["#6366F1"],
                            opacity=0.7,
                        )
                        mn_v = min(float(yt.min()), float(yp.min()))
                        mx_v = max(float(yt.max()), float(yp.max()))
                        fig_s.add_shape(
                            type="line",
                            x0=mn_v,
                            y0=mn_v,
                            x1=mx_v,
                            y1=mx_v,
                            line=dict(color="#059669", width=2, dash="dash"),
                        )
                        fig_s.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="#111827"),
                        )
                        st.plotly_chart(fig_s, width="stretch", key=f"sc_{mn}")
                    with s2:
                        fig_r = px.histogram(
                            x=yt - yp,
                            nbins=30,
                            title="Residual Distribution",
                            color_discrete_sequence=["#8B5CF6"],
                            labels={"x": "Residual"},
                        )
                        fig_r.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="#111827"),
                        )
                        st.plotly_chart(fig_r, width="stretch", key=f"res_{mn}")
    except Exception as err:
        st.error(str(err))

    # 6. Feature Importance
    try:
        tree_res = [r for r in results if hasattr(r["_model"], "feature_importances_")]
        if tree_res:
            st.markdown(
                '<div class="section-header"><span class="badge badge-emerald">06</span> Feature Importance</div>',
                unsafe_allow_html=True,
            )
            feat_names = pipeline_meta["encoded_feature_names"]
            fi_tabs = st.tabs([r["Model"] for r in tree_res])
            for tab, res in zip(fi_tabs, tree_res):
                with tab:
                    imps = res["_model"].feature_importances_
                    n = min(len(feat_names), len(imps))
                    fi_df = pd.DataFrame({"Feature": feat_names[:n], "Importance": imps[:n]}).sort_values(
                        "Importance", ascending=False
                    )
                    fig_fi = px.bar(
                        fi_df.head(20),
                        x="Importance",
                        y="Feature",
                        orientation="h",
                        color="Importance",
                        color_continuous_scale=["#6366F1", "#8B5CF6"],
                        title=f"Top Features - {res['Model']}",
                    )
                    fig_fi.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#111827"),
                        yaxis={"categoryorder": "total ascending"},
                        coloraxis_showscale=False,
                    )
                    st.plotly_chart(fig_fi, width="stretch", key=f"fi_{res['Model']}")
    except Exception as err:
        st.error(str(err))

    # 7. Prediction Studio & Export
    st.markdown(
        '<div class="section-header"><span class="badge badge-indigo">07</span> Prediction Studio & Export</div>',
        unsafe_allow_html=True,
    )

    cp1, cp2 = st.columns([3, 2])

    with cp1:
        st.markdown("### Live Inference")
        try:
            infer_model_name = st.selectbox(
                "Model for inference",
                [r["Model"] for r in results],
                index=[r["Model"] for r in results].index(best_name),
                key="ml_infer_model",
            )
            chosen = next(r for r in results if r["Model"] == infer_model_name)
            raw_feats = pipeline_meta["raw_features"]
            st.markdown("Set feature values for What-If prediction:")
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

            if st.button("Predict Output", type="primary", width="stretch", key="ml_predict"):
                try:
                    pred = MLEngine.predict_single(pipeline_meta, chosen["_model"], input_data)
                    st.markdown(f"#### Predicted `{target_done}`: **{pred['prediction']}**")
                    if pred.get("probabilities"):
                        dfp = pd.DataFrame([{"Class": k, "Probability": v} for k, v in pred["probabilities"].items()])
                        fig_p = px.bar(
                            dfp,
                            x="Probability",
                            y="Class",
                            orientation="h",
                            color="Probability",
                            color_continuous_scale="BuPu",
                            title="Class Probabilities",
                        )
                        fig_p.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="#111827"),
                        )
                        st.plotly_chart(fig_p, width="stretch", key="infer_proba")
                except Exception as err:
                    st.error(str(err))
        except Exception as err:
            st.error(str(err))

    with cp2:
        st.markdown("### Export Model Bundle")
        st.write("Download the trained pipeline as a `.pkl` file with all preprocessing metadata.")
        try:
            trained_sklearn_ver = pipeline_meta.get("sklearn_version", "unknown")
            current_sklearn_ver = sklearn.__version__
            if trained_sklearn_ver != current_sklearn_ver:
                st.warning(
                    f"Version mismatch: model was trained with scikit-learn {trained_sklearn_ver}, "
                    f"but this environment has {current_sklearn_ver}. "
                    "Loading this bundle elsewhere may produce wrong predictions."
                )
            bundle = {
                "model_name": best_name,
                "model": best_obj,
                "pipeline": pipeline_meta,
                "task_type": task_done,
                "target_col": target_done,
                "sklearn_version": current_sklearn_ver,
            }
            st.download_button(
                f"Export Best Model ({best_name})",
                pickle.dumps(bundle),
                f"vizml_{best_name.lower().replace(' ', '_')}.pkl",
                "application/octet-stream",
                width="stretch",
            )
        except Exception as err:
            st.error(str(err))
