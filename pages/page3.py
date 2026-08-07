import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="VizML - ML Studio", layout="wide")

# ── CSS ────────────────────────────────────────────────────────────
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700;800&family=Inter:wght@400;500;600&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

    .page-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 36px;
        font-weight: 800;
        background: linear-gradient(135deg, #6C63FF 0%, #2DD4A0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
    }
    .page-sub {
        font-size: 14px;
        color: #8A8BA8;
        margin-bottom: 28px;
    }
    .section-header {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 15px;
        font-weight: 700;
        color: #EEEEF5;
        letter-spacing: 0.5px;
        margin-bottom: 12px;
        padding-bottom: 6px;
        border-bottom: 1px solid #1E2030;
    }
    .badge-auto {
        display: inline-block;
        background: rgba(45, 212, 160, 0.12);
        color: #2DD4A0;
        font-size: 10px;
        font-weight: 600;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        padding: 3px 10px;
        border-radius: 50px;
        border: 1px solid rgba(45, 212, 160, 0.3);
        margin-left: 8px;
        vertical-align: middle;
    }
    .badge-manual {
        display: inline-block;
        background: rgba(108, 99, 255, 0.12);
        color: #6C63FF;
        font-size: 10px;
        font-weight: 600;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        padding: 3px 10px;
        border-radius: 50px;
        border: 1px solid rgba(108, 99, 255, 0.3);
        margin-left: 8px;
        vertical-align: middle;
    }
    div[data-testid="stMetricValue"] {
        font-size: 26px;
        font-weight: 700;
        color: #6C63FF;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 12px;
        color: #8A8BA8;
        font-weight: 500;
    }
    </style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────
st.markdown('<div class="page-title">ML Studio</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="page-sub">Train and evaluate machine learning models on your session dataset — no code required.</div>',
    unsafe_allow_html=True
)

# ── Dataset guard ──────────────────────────────────────────────────
df = st.session_state.get("df", None)
if df is None:
    st.warning("⚠️ No dataset in session. Please upload and clean a dataset in the **Data Curation** page first.")
    st.stop()

numeric_cols = df.select_dtypes(include="number").columns.tolist()
all_cols = df.columns.tolist()

st.info(
    f"Dataset ready — **{df.shape[0]:,} rows × {df.shape[1]} columns**, "
    f"{len(numeric_cols)} numeric features available."
)

# ── Lazy imports ───────────────────────────────────────────────────
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, mean_squared_error, r2_score,
    confusion_matrix, classification_report
)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
import plotly.express as px

# ══════════════════════════════════════════════════════════════════
# SECTION 1 — Task & Target Configuration
# ══════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">1 · Task Configuration</div>', unsafe_allow_html=True)

col_task, col_target = st.columns([1, 2])
with col_task:
    task_type = st.radio(
        "Task Type",
        options=["Classification", "Regression"],
        horizontal=True,
        key="ml_task_type",
        help="Classification predicts categories; Regression predicts continuous values."
    )

with col_target:
    target_col = st.selectbox(
        "Target Column (Y)",
        options=all_cols,
        index=len(all_cols) - 1,
        key="ml_target_col",
        help="The column you want to predict."
    )

# ══════════════════════════════════════════════════════════════════
# SECTION 2 — Feature Selection
# ══════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">2 · Feature Selection</div>', unsafe_allow_html=True)

feat_mode = st.radio(
    "Feature Mode",
    options=["Auto (numeric only)", "Manual (pick features)"],
    horizontal=True,
    key="ml_feat_mode",
    help="Auto uses all numeric columns except the target. Manual lets you pick any columns."
)

available_features = [c for c in numeric_cols if c != target_col]

if feat_mode == "Auto (numeric only)":
    selected_features = available_features
    badge = '<span class="badge-auto">Auto</span>'
    if selected_features:
        st.markdown(
            f'{badge} Using **{len(selected_features)}** numeric features: '
            f'`{"`, `".join(selected_features[:8])}{"`, ..." if len(selected_features) > 8 else "`"}',
            unsafe_allow_html=True
        )
    else:
        st.error("No numeric feature columns available after excluding the target. Switch to Manual mode or choose a different target.")
        st.stop()
else:
    badge = '<span class="badge-manual">Manual</span>'
    st.markdown(badge, unsafe_allow_html=True)
    feature_pool = [c for c in all_cols if c != target_col]
    selected_features = st.multiselect(
        "Feature Columns (X)",
        options=feature_pool,
        default=available_features[:min(5, len(available_features))],
        key="ml_features",
        help="Select one or more columns to use as model inputs."
    )

if not selected_features:
    st.warning("Select at least one feature column to continue.")
    st.stop()

# ══════════════════════════════════════════════════════════════════
# SECTION 3 — Model Selection
# ══════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">3 · Model Selection</div>', unsafe_allow_html=True)

CLASSIFICATION_MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "SVM (RBF Kernel)": SVC(kernel="rbf", probability=True),
    "Naive Bayes": GaussianNB(),
}

REGRESSION_MODELS = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.1, max_iter=2000),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "K-Nearest Neighbors": KNeighborsRegressor(),
    "SVR (RBF Kernel)": SVR(kernel="rbf"),
}

model_options = (
    list(CLASSIFICATION_MODELS.keys())
    if task_type == "Classification"
    else list(REGRESSION_MODELS.keys())
)

col_m1, col_m2 = st.columns([3, 1])
with col_m1:
    selected_models = st.multiselect(
        "Models to Train",
        options=model_options,
        default=model_options[:3],
        key="ml_selected_models",
        help="All selected models will be trained and compared in a leaderboard."
    )
with col_m2:
    test_size = st.slider("Test Split %", 10, 40, 20, 5, key="ml_test_size")
    cv_folds = st.slider("Cross-Val Folds", 2, 10, 5, 1, key="ml_cv")

if not selected_models:
    st.warning("Select at least one model to train.")
    st.stop()

# ══════════════════════════════════════════════════════════════════
# SECTION 4 — Train
# ══════════════════════════════════════════════════════════════════
st.markdown("---")
train_btn = st.button("🚀 Train Models", type="primary", use_container_width=True, key="ml_train_btn")

if train_btn:
    with st.spinner("Preparing data and training models…"):
        try:
            # Build X, y
            X_raw = df[selected_features].copy()
            y_raw = df[target_col].copy()

            # Drop rows where target is null
            valid_mask = y_raw.notna()
            X_raw = X_raw[valid_mask]
            y_raw = y_raw[valid_mask]

            # Encode non-numeric features in X
            for col in X_raw.columns:
                if not pd.api.types.is_numeric_dtype(X_raw[col]):
                    le = LabelEncoder()
                    X_raw[col] = le.fit_transform(X_raw[col].astype(str))

            # Fill remaining NaN in X with column median
            X_raw = X_raw.apply(lambda c: c.fillna(c.median()) if c.notna().any() else c.fillna(0))

            # Encode target for classification if needed
            le_target = None
            if task_type == "Classification":
                if not pd.api.types.is_numeric_dtype(y_raw):
                    le_target = LabelEncoder()
                    y = le_target.fit_transform(y_raw.astype(str))
                else:
                    y = y_raw.values.astype(int)
            else:
                y = y_raw.values.astype(float)

            X = X_raw.values

            if task_type == "Classification" and len(np.unique(y)) < 2:
                st.error("Target column has only one unique class — cannot train a classifier.")
                st.stop()

            # Split
            strat = y if task_type == "Classification" else None
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size / 100, random_state=42, stratify=strat
            )

            # Scale
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            model_map = CLASSIFICATION_MODELS if task_type == "Classification" else REGRESSION_MODELS
            results = []
            prog = st.progress(0, text="Training…")

            for i, name in enumerate(selected_models):
                prog.progress((i) / len(selected_models), text=f"Training {name}…")
                model = model_map[name]
                cv_score = cross_val_score(
                    model, X_train_s, y_train,
                    cv=cv_folds,
                    scoring="accuracy" if task_type == "Classification" else "r2"
                )
                model.fit(X_train_s, y_train)
                y_pred = model.predict(X_test_s)

                if task_type == "Classification":
                    acc = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
                    results.append({
                        "Model": name,
                        "Test Accuracy": round(acc, 4),
                        "F1 Score (weighted)": round(f1, 4),
                        "CV Score (mean)": round(cv_score.mean(), 4),
                        "CV Score (std)": round(cv_score.std(), 4),
                        "_model": model,
                        "_y_pred": y_pred,
                        "_cv_mean": float(cv_score.mean()),
                    })
                else:
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = float(np.sqrt(mse))
                    r2 = r2_score(y_test, y_pred)
                    results.append({
                        "Model": name,
                        "Test R²": round(r2, 4),
                        "RMSE": round(rmse, 4),
                        "CV Score (mean)": round(cv_score.mean(), 4),
                        "CV Score (std)": round(cv_score.std(), 4),
                        "_model": model,
                        "_y_pred": y_pred,
                        "_y_test": y_test,
                        "_cv_mean": float(cv_score.mean()),
                    })
            prog.progress(1.0, text="Done!")

            st.session_state["ml_results"] = results
            st.session_state["ml_task_type_done"] = task_type
            st.session_state["ml_y_test"] = y_test
            st.session_state["ml_le_target"] = le_target
            st.session_state["ml_selected_features_done"] = selected_features
            st.success(f"✅ Trained {len(results)} model(s) successfully!")

        except Exception as e:
            st.error(f"Training failed: {e}")
            st.stop()

# ══════════════════════════════════════════════════════════════════
# SECTION 5 — Results
# ══════════════════════════════════════════════════════════════════
results = st.session_state.get("ml_results", None)
task_done = st.session_state.get("ml_task_type_done", None)

if results:
    st.markdown('<div class="section-header">4 · Leaderboard</div>', unsafe_allow_html=True)

    df_results = pd.DataFrame([
        {k: v for k, v in r.items() if not k.startswith("_")}
        for r in results
    ])
    best_idx = max(range(len(results)), key=lambda i: results[i]["_cv_mean"])

    # Highlight max on numeric display columns
    num_display = [c for c in df_results.columns if c not in ("Model",)]
    try:
        styled = df_results.style.highlight_max(subset=num_display, color="#1e1f4a")
    except Exception:
        styled = df_results.style
    st.dataframe(styled, use_container_width=True, hide_index=True)
    st.caption(f"🏆 Best by CV score: **{results[best_idx]['Model']}**")

    # Bar chart
    primary_metric = "Test Accuracy" if task_done == "Classification" else "Test R²"
    if primary_metric in df_results.columns:
        fig_bar = px.bar(
            df_results, x="Model", y=primary_metric, color="Model",
            color_discrete_sequence=px.colors.qualitative.Vivid,
            title=f"{primary_metric} Comparison",
            template="plotly_dark",
        )
        fig_bar.update_layout(
            paper_bgcolor="#0D0E14", plot_bgcolor="#13141A",
            showlegend=False, font_family="Inter", title_font_family="Space Grotesk"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Per-model details ─────────────────────────────────────
    st.markdown('<div class="section-header">5 · Per-Model Detail</div>', unsafe_allow_html=True)
    tabs = st.tabs([r["Model"] for r in results])
    y_test_g = st.session_state.get("ml_y_test", [])
    le_target = st.session_state.get("ml_le_target", None)

    for tab, res in zip(tabs, results):
        with tab:
            if task_done == "Classification":
                y_pred = res["_y_pred"]
                cm = confusion_matrix(y_test_g, y_pred)
                n = cm.shape[0]
                if le_target is not None:
                    labels = [str(c) for c in le_target.classes_[:n]]
                else:
                    labels = [str(c) for c in sorted(np.unique(y_test_g).tolist())[:n]]

                fig_cm = px.imshow(
                    cm, x=labels, y=labels, text_auto=True,
                    color_continuous_scale="Purples",
                    title="Confusion Matrix", template="plotly_dark",
                    labels=dict(x="Predicted", y="Actual"),
                )
                fig_cm.update_layout(
                    paper_bgcolor="#0D0E14", plot_bgcolor="#13141A",
                    font_family="Inter", title_font_family="Space Grotesk"
                )
                st.plotly_chart(fig_cm, use_container_width=True)

                report = classification_report(
                    y_test_g, y_pred, target_names=labels,
                    output_dict=True, zero_division=0
                )
                st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)

            else:
                y_pred = res["_y_pred"]
                y_test_local = res["_y_test"]
                fig_scatter = px.scatter(
                    x=y_test_local, y=y_pred,
                    labels={"x": "Actual", "y": "Predicted"},
                    title="Actual vs Predicted",
                    template="plotly_dark",
                    color_discrete_sequence=["#6C63FF"],
                )
                mn = min(float(y_test_local.min()), float(y_pred.min()))
                mx = max(float(y_test_local.max()), float(y_pred.max()))
                fig_scatter.add_shape(
                    type="line", x0=mn, y0=mn, x1=mx, y1=mx,
                    line=dict(color="#2DD4A0", width=2, dash="dash")
                )
                fig_scatter.update_layout(
                    paper_bgcolor="#0D0E14", plot_bgcolor="#13141A",
                    font_family="Inter", title_font_family="Space Grotesk"
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

                residuals = y_test_local - y_pred
                fig_res = px.histogram(
                    x=residuals, nbins=30, title="Residual Distribution",
                    template="plotly_dark", color_discrete_sequence=["#EC4899"],
                    labels={"x": "Residual"},
                )
                fig_res.update_layout(
                    paper_bgcolor="#0D0E14", plot_bgcolor="#13141A",
                    font_family="Inter", title_font_family="Space Grotesk"
                )
                st.plotly_chart(fig_res, use_container_width=True)

    # ── Feature importance ────────────────────────────────────
    tree_results = [r for r in results if hasattr(r["_model"], "feature_importances_")]
    if tree_results:
        st.markdown('<div class="section-header">6 · Feature Importance</div>', unsafe_allow_html=True)
        feat_names = st.session_state.get("ml_selected_features_done", [])
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
                    fi_df, x="Importance", y="Feature", orientation="h",
                    color="Importance",
                    color_continuous_scale=["#6C63FF", "#2DD4A0"],
                    title=f"Feature Importance — {res['Model']}",
                    template="plotly_dark",
                )
                fig_fi.update_layout(
                    paper_bgcolor="#0D0E14", plot_bgcolor="#13141A",
                    font_family="Inter", title_font_family="Space Grotesk",
                    yaxis={"categoryorder": "total ascending"},
                    coloraxis_showscale=False,
                )
                st.plotly_chart(fig_fi, use_container_width=True)
