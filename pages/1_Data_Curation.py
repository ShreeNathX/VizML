import io

import numpy as np
import pandas as pd
import streamlit as st

from src.cleaner import DataCleaner
from src.styles import inject_css, render_html

MAX_FILE_MB = 100
MAX_HISTORY = 10

st.set_page_config(page_title="VizML: Data Curation", layout="wide")
inject_css()

render_html(
    """
    <div class="bp-card">
        <div class="bp-header" style="color:var(--coral);">Stage 01 · Ingest &amp; Clean</div>
        <div class="page-title">Data Curation Engine</div>
        <div class="page-sub">
            Profile dataset integrity, coerce schema types, cast individual columns, impute missing values,
            handle statistical outliers, scrub duplicates, and inspect live diffs with non-destructive rollbacks.
            All curation tools operate independently in any order.
        </div>
    </div>
    """
)


def show_toast(msg: str) -> None:
    try:
        if hasattr(st, "toast"):
            st.toast(msg)
        else:
            st.sidebar.info(msg)
    except Exception:
        pass


def safe_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].apply(lambda x: str(x) if pd.notna(x) else None)
    return out


def reset_workspace_for_new_dataset(data_df: pd.DataFrame, file_name: str) -> None:
    """Sets active dataset and purges all old downstream ML, filters, and reports state."""
    st.session_state["df"] = data_df
    st.session_state["original_df"] = data_df.copy()
    st.session_state["history"] = []
    st.session_state["coercion_flags"] = {}
    st.session_state["current_file_name"] = file_name

    # Clear downstream ML state variables
    for ml_var in [
        "ml_engine_output", "ml_results", "ml_best_name", "ml_pipeline",
        "ml_task_result", "ml_target_result", "ml_feature_importance", "ml_bundle"
    ]:
        st.session_state[ml_var] = None

    # Delete widget keys (never set widget keys to None in Streamlit!)
    for widget_k in [
        "ml_target", "ml_task", "ml_feat_mode", "ml_features", "ml_models",
        "ml_test_size", "ml_cv", "ml_infer_model", "ml_predict"
    ]:
        if widget_k in st.session_state:
            del st.session_state[widget_k]

    # Clear downstream filter and chart widget keys
    for k in list(st.session_state.keys()):
        if k.startswith("f_cat_") or k.startswith("f_num_") or k.startswith("cb_") or k.startswith("inf_"):
            del st.session_state[k]


def unload_dataset() -> None:
    """Completely unloads the dataset and clears all workspace state."""
    st.session_state["df"] = None
    st.session_state["original_df"] = None
    st.session_state["history"] = []
    st.session_state["coercion_flags"] = {}
    st.session_state["current_file_name"] = None

    for ml_var in [
        "ml_engine_output", "ml_results", "ml_best_name", "ml_pipeline",
        "ml_task_result", "ml_target_result", "ml_feature_importance", "ml_bundle"
    ]:
        st.session_state[ml_var] = None

    for widget_k in [
        "ml_target", "ml_task", "ml_feat_mode", "ml_features", "ml_models",
        "ml_test_size", "ml_cv", "ml_infer_model", "ml_predict"
    ]:
        if widget_k in st.session_state:
            del st.session_state[widget_k]

    for k in list(st.session_state.keys()):
        if k.startswith("f_cat_") or k.startswith("f_num_") or k.startswith("cb_") or k.startswith("inf_"):
            del st.session_state[k]


# Initialize session state keys
for k, v in [
    ("df", None),
    ("original_df", None),
    ("history", []),
    ("coercion_flags", {}),
    ("current_file_name", None),
]:
    if k not in st.session_state:
        st.session_state[k] = v


def load_data(file_obj) -> None:
    try:
        size_mb = file_obj.size / (1024 * 1024)
        if size_mb > MAX_FILE_MB:
            st.error(
                f"File size ({size_mb:.1f} MB) exceeds maximum allowed limit of {MAX_FILE_MB} MB."
            )
            return

        if file_obj.name.endswith(".csv"):
            data_df = pd.read_csv(file_obj)
        elif file_obj.name.endswith((".xlsx", ".xls")):
            data_df = pd.read_excel(file_obj)
        else:
            st.error("Unsupported file format. Please upload CSV or Excel.")
            return

        if data_df.empty:
            st.error("The uploaded dataset contains no rows.")
            return

        reset_workspace_for_new_dataset(data_df, file_obj.name)
        show_toast(f"Loaded {file_obj.name} ({data_df.shape[0]:,} rows, {data_df.shape[1]} cols).")
    except Exception as err:
        st.error(f"Failed to load dataset: {err}")


def push_history(df: pd.DataFrame) -> None:
    history: list = st.session_state["history"]
    history.append(df.copy())
    if len(history) > MAX_HISTORY:
        history.pop(0)
    st.session_state["history"] = history


# -------------------------------------------------------------
# Active Dataset Header & Uploader Management
# -------------------------------------------------------------
has_active_df = st.session_state.get("df") is not None
active_file_name = st.session_state.get("current_file_name", "Active Dataset")

if has_active_df:
    active_df_obj = st.session_state["df"]
    cur_missing = int(active_df_obj.isna().sum().sum())
    cur_dups = int(active_df_obj.duplicated().sum())

    missing_badge = (
        '<span class="badge badge-emerald">0 Missing Cells</span>'
        if cur_missing == 0
        else f'<span class="badge badge-coral">{cur_missing:,} Missing</span>'
    )
    dup_badge = (
        '<span class="badge badge-emerald">0 Duplicates</span>'
        if cur_dups == 0
        else f'<span class="badge badge-gold">{cur_dups:,} Dups</span>'
    )

    render_html(
        f"""
        <div class="bp-card" style="padding:16px 22px;margin-bottom:16px;">
            <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px;">
                <div style="display:flex;align-items:center;gap:12px;">
                    <div class="brand-mark" style="width:38px;height:38px;border-radius:12px;background:linear-gradient(145deg,var(--coral-l),var(--coral));">
                        <svg viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.2" style="width:18px;height:18px;">
                            <path d="M4 20l6-6M13 4l7 7-8.5 8.5a2 2 0 0 1-2.8 0l-4.2-4.2a2 2 0 0 1 0-2.8L13 4z"/>
                        </svg>
                    </div>
                    <div>
                        <div style="font-family:'Baloo 2',sans-serif;font-size:17px;font-weight:700;color:var(--ink);">
                            Active Dataset: <code>{active_file_name}</code>
                        </div>
                        <div style="font-size:12px;color:var(--ink-soft);font-weight:600;">
                            {active_df_obj.shape[0]:,} rows × {active_df_obj.shape[1]} columns
                        </div>
                    </div>
                </div>
                <div style="display:flex;align-items:center;gap:8px;">
                    <span class="badge badge-indigo">ACTIVE IN WORKSPACE</span>
                    {missing_badge}
                    {dup_badge}
                </div>
            </div>
        </div>
        """
    )

    with st.expander("Replace / Upload Another Dataset", expanded=False):
        uploaded_file = st.file_uploader(
            f"Select a new CSV or Excel file to replace current dataset (max {MAX_FILE_MB} MB)",
            type=["csv", "xlsx", "xls"],
            key="curation_file_uploader_replace",
        )
        if uploaded_file is not None:
            if st.session_state.get("current_file_name") != uploaded_file.name:
                load_data(uploaded_file)
                st.rerun()

        if st.button("Unload Dataset (Clear Workspace)", type="secondary"):
            unload_dataset()
            show_toast("Dataset unloaded.")
            st.rerun()

else:
    uploaded_file = st.file_uploader(
        f"Upload Dataset (CSV or Excel, max {MAX_FILE_MB} MB)",
        type=["csv", "xlsx", "xls"],
        key="curation_file_uploader_fresh",
    )
    if uploaded_file is not None:
        if st.session_state.get("current_file_name") != uploaded_file.name:
            load_data(uploaded_file)
            st.rerun()


# -------------------------------------------------------------
# Curation Pipeline & Inspector
# -------------------------------------------------------------
if st.session_state.get("df") is not None:
    df_current: pd.DataFrame = st.session_state["df"]
    orig_df_obj = st.session_state.get("original_df", df_current)

    ctrl_c1, ctrl_c2, _ = st.columns([2, 2, 8])
    with ctrl_c1:
        undo_disabled = len(st.session_state["history"]) == 0
        if st.button("Undo Last Step", disabled=undo_disabled):
            try:
                st.session_state["df"] = st.session_state["history"].pop()
                show_toast("Undid last cleaning operation.")
                st.rerun()
            except Exception as err:
                st.error(f"Undo failed: {err}")
    with ctrl_c2:
        if st.button("Reset to Source"):
            try:
                st.session_state["df"] = st.session_state["original_df"].copy()
                st.session_state["history"] = []
                st.session_state["coercion_flags"] = {}
                show_toast("Reset to original dataset.")
                st.rerun()
            except Exception as err:
                st.error(f"Reset failed: {err}")

    st.markdown("<br>", unsafe_allow_html=True)

    col_tools, col_inspector = st.columns([5, 7])

    with col_tools:
        render_html('<div class="step-header"><span class="step-num coral">01</span><span class="step-title">Data Curation Toolkit</span></div>')

        try:
            cleaner = DataCleaner(df_current)
            cleaner.flagged_columns = st.session_state.get("coercion_flags", {})
        except Exception:
            cleaner = DataCleaner(df_current)

        proposed_dfs: dict = {}

        # ---------------------------------------------------------
        # 1. Missing-Value Imputation
        # ---------------------------------------------------------
        null_cols = [c for c in df_current.columns if df_current[c].isna().any()]
        total_curr_nulls = int(df_current.isna().sum().sum())
        orig_nulls = int(orig_df_obj.isna().sum().sum()) if orig_df_obj is not None else 0

        expander_missing_title = (
            f"Missing-Value Imputation [Status: {total_curr_nulls:,} missing cells]"
            if total_curr_nulls > 0
            else "Missing-Value Imputation [Status: Resolved]"
        )

        with st.expander(expander_missing_title, expanded=(total_curr_nulls > 0)):
            if null_cols:
                render_html(f'<div class="alert-box alert-warning">Found <strong>{total_curr_nulls:,} missing cells</strong> across {len(null_cols)} column(s). Select columns and fill strategy:</div>')
                num_strat = st.radio(
                    "Numeric Strategy",
                    ["median", "mean", "ffill", "bfill"],
                    index=0,
                    key="num_strat",
                    horizontal=True,
                )
                cat_strat = st.radio(
                    "Categorical Strategy",
                    ["mode", "constant", "ffill", "bfill"],
                    index=0,
                    key="cat_strat",
                    horizontal=True,
                )
                cat_const = st.text_input("Constant Fill Token", value="Unknown", key="cat_const")
                cols_to_impute = st.multiselect("Columns to impute", null_cols, default=null_cols)

                if cols_to_impute:
                    try:
                        df_prop_missing = cleaner.handle_missing(
                            numeric_strategy=num_strat,
                            categorical_strategy=cat_strat,
                            categorical_constant=cat_const,
                            columns=cols_to_impute,
                        )
                        proposed_dfs["missing"] = df_prop_missing

                        if st.button("Commit Imputation", key="btn_missing_commit", type="primary"):
                            with st.spinner("Applying imputation to missing values..."):
                                push_history(df_current)
                                st.session_state["df"] = df_prop_missing
                            show_toast("Missing values successfully imputed.")
                            st.rerun()
                    except Exception as err:
                        st.error(f"Imputation error: {err}")
                else:
                    st.info("Select at least one column to impute.")
            else:
                if orig_nulls > 0:
                    render_html(
                        f'<div class="alert-box alert-success">'
                        f'<strong>Problem Fixed:</strong> All {orig_nulls:,} missing cells have been successfully resolved.'
                        f'</div>'
                    )
                else:
                    render_html('<div class="alert-box alert-success"><strong>Optimal:</strong> No missing values detected in this dataset.</div>')

        # ---------------------------------------------------------
        # 2. Schema and Type Casting Toolkit
        # ---------------------------------------------------------
        with st.expander("Schema and Type Casting Toolkit", expanded=False):
            tab_manual_cast, tab_auto_coerce = st.tabs(["Manual Type Casting (Developer)", "Auto Schema Scanner"])

            with tab_manual_cast:
                render_html("<span style='font-size:12.5px;color:var(--ink-soft);font-weight:500;'>Directly cast any column to Integer, Float, String, Category, Datetime, or Boolean.</span>")
                all_df_cols = df_current.columns.tolist()

                if all_df_cols:
                    col_to_cast = st.selectbox(
                        "Select Column to Cast",
                        all_df_cols,
                        index=0,
                        key="manual_cast_col"
                    )

                    curr_col_dtype = str(df_current[col_to_cast].dtype)
                    render_html(f"Current Column Type: <code>{curr_col_dtype}</code> | Non-Null Values: <strong>{df_current[col_to_cast].notna().sum():,}</strong> / {len(df_current):,}")

                    # Auto-reset success state if user picks a different column
                    if st.session_state.get("_last_cast_col") != col_to_cast:
                        st.session_state.pop("_last_cast_col", None)
                        st.session_state.pop("_last_cast_type", None)

                    target_type_choice = st.selectbox(
                        "Target Data Type",
                        [
                            "int64 (Integer - whole numbers)",
                            "float64 (Float - decimals/numeric)",
                            "str (Text String / object)",
                            "category (Categorical / nominal)",
                            "datetime64 (Date / Timestamp)",
                            "bool (Boolean True/False)",
                        ],
                        index=0,
                        key="manual_cast_target_type"
                    )

                    target_type_key = target_type_choice.split()[0]
                    dt_fmt = None
                    if target_type_key == "datetime64":
                        dt_fmt = st.text_input(
                            "Datetime Format (Optional — leave blank for auto-parse)",
                            value="",
                            placeholder="e.g. %Y-%m-%d or %d/%m/%Y %H:%M:%S",
                            key="manual_cast_dt_fmt"
                        )

                    try:
                        df_prop_cast, cast_msg = cleaner.cast_column_type(
                            column=col_to_cast,
                            target_type=target_type_key,
                            datetime_format=dt_fmt
                        )
                        proposed_dfs["type_cast"] = df_prop_cast

                        # Show live preview
                        preview_df = pd.DataFrame({
                            f"Original ({curr_col_dtype})": df_current[col_to_cast].head(5).astype(str),
                            f"Target ({target_type_key})": df_prop_cast[col_to_cast].head(5).astype(str),
                        })
                        st.markdown("**Live Conversion Preview (First 5 records):**")
                        st.dataframe(preview_df, width='stretch')

                        # ── Data-loss guard ──────────────────────────────────
                        orig_nulls_col = int(df_current[col_to_cast].isna().sum())
                        prop_nulls_col = int(df_prop_cast[col_to_cast].isna().sum())
                        new_nulls = prop_nulls_col - orig_nulls_col   # values that become NaN after cast
                        total_rows = len(df_current)
                        loss_pct = (new_nulls / total_rows * 100) if total_rows > 0 else 0

                        _cast_confirmed = True   # default: no confirmation needed
                        if new_nulls > 0:
                            if loss_pct >= 50:
                                render_html(
                                    f'<div class="alert-box alert-error" style="margin-top:8px;font-size:12.5px;">'
                                    f'<strong>⚠ Destructive Cast Detected:</strong> Casting <code>{col_to_cast}</code> to '
                                    f'<code>{target_type_key}</code> will convert <strong>{new_nulls:,} of {total_rows:,} '
                                    f'values ({loss_pct:.1f}%)</strong> to <code>NaN</code> because they cannot be parsed as '
                                    f'{target_type_key} (e.g. text like &quot;male&quot;/&quot;female&quot; cannot become integers). '
                                    f'Use <strong>Undo Last Step</strong> above to reverse this at any time.'
                                    f'</div>'
                                )
                                _cast_confirmed = st.checkbox(
                                    f"I understand that {new_nulls:,} rows ({loss_pct:.1f}%) will become NaN — proceed anyway",
                                    value=False,
                                    key="cast_loss_confirm"
                                )
                            else:
                                render_html(
                                    f'<div class="alert-box alert-warning" style="margin-top:8px;font-size:12px;">'
                                    f'<strong>Notice:</strong> {new_nulls:,} value(s) ({loss_pct:.1f}%) will become '
                                    f'<code>NaN</code> after this cast. Use <strong>Undo Last Step</strong> to reverse.'
                                    f'</div>'
                                )

                        # Show success only if this column was just committed
                        last_cast = st.session_state.get("_last_cast_col")
                        last_cast_type = st.session_state.get("_last_cast_type")
                        if last_cast == col_to_cast and last_cast_type == target_type_key:
                            render_html(f'<div class="alert-box alert-success" style="margin-top:6px;font-size:12px;"><strong>Successfully cast</strong> — {cast_msg} &nbsp;|&nbsp; Use <strong>Undo Last Step</strong> above to reverse.</div>')
                        elif new_nulls == 0:
                            render_html(f'<div class="alert-box alert-info" style="margin-top:6px;font-size:12px;">Preview ready. Click <strong>Commit Type Cast</strong> to apply.</div>')

                        if st.button(
                            "Commit Type Cast",
                            key="btn_manual_cast_commit",
                            type="primary",
                            disabled=not _cast_confirmed
                        ):
                            with st.spinner(f"Casting '{col_to_cast}' to {target_type_key}..."):
                                push_history(df_current)
                                st.session_state["df"] = df_prop_cast
                                st.session_state["_last_cast_col"] = col_to_cast
                                st.session_state["_last_cast_type"] = target_type_key
                                # Reset confirmation checkbox for next action
                                st.session_state.pop("cast_loss_confirm", None)
                            show_toast(cast_msg)
                            st.rerun()
                    except Exception as err:
                        st.error(f"Type cast error: {err}")
                else:
                    st.info("No columns available to cast.")

            with tab_auto_coerce:
                try:
                    candidate_str_cols = [
                        c for c in df_current.columns
                        if pd.api.types.is_object_dtype(df_current[c]) or pd.api.types.is_string_dtype(df_current[c])
                    ]

                    if candidate_str_cols:
                        selected_coerce_cols = st.multiselect(
                            "Columns to evaluate for automated schema conversion",
                            candidate_str_cols,
                            default=candidate_str_cols,
                            key="sel_coerce_cols"
                        )

                        df_prop_coerce = cleaner.coerce_types(columns=selected_coerce_cols)
                        proposed_dfs["coercion"] = df_prop_coerce

                        coerced = [
                            (c, str(df_current[c].dtype), str(df_prop_coerce[c].dtype))
                            for c in selected_coerce_cols
                            if str(df_current[c].dtype) != str(df_prop_coerce[c].dtype)
                        ]

                        if coerced:
                            st.markdown("**Proposed Schema Type Conversions:**")
                            for c, old_t, new_t in coerced:
                                st.write(f"- `{c}`: `{old_t}` ➔ `{new_t}`")
                            if st.button("Commit Batch Schema Coercion", key="btn_coerce_commit", type="primary"):
                                with st.spinner("Running batch schema coercion..."):
                                    push_history(df_current)
                                    st.session_state["df"] = df_prop_coerce
                                    st.session_state["coercion_flags"] = cleaner.flagged_columns
                                show_toast("Batch type coercion applied.")
                                st.rerun()
                        else:
                            render_html('<div class="alert-box alert-success"><strong>Schema Verified:</strong> All evaluated columns match their ideal data types.</div>')

                        for c, flag in cleaner.flagged_columns.items():
                            if "cast" in flag.lower():
                                render_html(f"`{c}`: <span class='badge badge-gold'>PARSED</span>: {flag}")
                    else:
                        render_html('<div class="alert-box alert-success"><strong>Schema Optimal:</strong> No unparsed string columns detected.</div>')
                except Exception as err:
                    st.error(f"Type coercion error: {err}")

        # ---------------------------------------------------------
        # 3. Outlier Handling (IQR)
        # ---------------------------------------------------------
        num_cols_all = [
            c for c in df_current.columns
            if pd.api.types.is_numeric_dtype(df_current[c])
            and not pd.api.types.is_bool_dtype(df_current[c])
        ]

        if num_cols_all:
            mask_init = cleaner.detect_outliers(columns=num_cols_all)
            total_outliers_found = int(mask_init.sum().sum())
            outlier_expander_title = (
                f"Outlier Handling (IQR) [Status: {total_outliers_found:,} cells]"
                if total_outliers_found > 0
                else "Outlier Handling (IQR) [Status: Clean]"
            )
        else:
            total_outliers_found = 0
            outlier_expander_title = "Outlier Handling (IQR)"

        with st.expander(outlier_expander_title, expanded=False):
            if num_cols_all:
                outlier_cols = st.multiselect("Columns to evaluate", num_cols_all, default=num_cols_all)
                if not outlier_cols:
                    st.info("Select at least one numeric column to evaluate.")
                else:
                    mask = cleaner.detect_outliers(columns=outlier_cols)
                    total_outliers = int(mask.sum().sum())
                    if total_outliers == 0:
                        render_html('<div class="alert-box alert-success"><strong>Problem Fixed:</strong> No statistical outliers detected across selected columns (IQR 1.5x threshold).</div>')
                    else:
                        outlier_action = st.radio(
                            "Action",
                            ["clip", "remove", "flag"],
                            index=0,
                            key="outlier_action",
                            horizontal=True,
                        )
                        st.write(f"Outlier cells detected: **{total_outliers:,}**")
                        for c in outlier_cols:
                            n = int(mask[c].sum())
                            if n > 0:
                                st.write(f"- `{c}`: {n} outlier(s)")
                        try:
                            df_prop_outliers = cleaner.handle_outliers(
                                action=outlier_action, columns=outlier_cols
                            )
                            proposed_dfs["outliers"] = df_prop_outliers
                            if st.button(
                                f"Commit Outlier {outlier_action.capitalize()}",
                                key="btn_outlier_commit",
                                type="primary",
                            ):
                                with st.spinner(f"Applying outlier {outlier_action}..."):
                                    push_history(df_current)
                                    st.session_state["df"] = df_prop_outliers
                                show_toast(f"Outliers handled via '{outlier_action}'.")
                                st.rerun()
                        except Exception as err:
                            st.error(f"Outlier handling error: {err}")
            else:
                st.info("No numeric columns available for outlier detection.")

        # ---------------------------------------------------------
        # 4. Duplicate Scrubbing
        # ---------------------------------------------------------
        try:
            dup_count = cleaner.get_duplicate_count()
        except Exception:
            dup_count = 0

        dup_title = (
            f"Duplicate Scrubbing [Status: {dup_count:,} duplicate rows]"
            if dup_count > 0
            else "Duplicate Scrubbing [Status: Clean]"
        )

        with st.expander(dup_title, expanded=(dup_count > 0)):
            try:
                st.write(f"Duplicate rows detected: **{dup_count}**")
                if dup_count > 0:
                    df_prop_dups = cleaner.remove_duplicates()
                    proposed_dfs["duplicates"] = df_prop_dups
                    if st.button("Scrub Duplicates", key="btn_dups_commit", type="primary"):
                        with st.spinner("Scrubbing duplicate rows..."):
                            push_history(df_current)
                            st.session_state["df"] = df_prop_dups
                        show_toast(f"Removed {dup_count} duplicate rows.")
                        st.rerun()
                else:
                    render_html('<div class="alert-box alert-success"><strong>Problem Fixed:</strong> Zero duplicate rows detected in dataset.</div>')
            except Exception as err:
                st.error(f"Duplicate check error: {err}")

        # ---------------------------------------------------------
        # 5. Categorical Encoding
        # ---------------------------------------------------------
        with st.expander("Categorical Encoding", expanded=False):
            cat_enc_cols = [
                c for c in df_current.columns
                if (
                    pd.api.types.is_object_dtype(df_current[c])
                    or isinstance(df_current[c].dtype, pd.CategoricalDtype)
                    or pd.api.types.is_bool_dtype(df_current[c])
                )
            ]
            if cat_enc_cols:
                cols_to_encode = st.multiselect("Columns to encode", cat_enc_cols, default=cat_enc_cols)
                if cols_to_encode:
                    enc_method = st.radio(
                        "Method",
                        ["onehot", "label"],
                        index=0,
                        key="enc_method",
                        horizontal=True,
                    )
                    try:
                        df_prop_enc = cleaner.encode_categoricals(
                            method=enc_method, columns=cols_to_encode
                        )
                        proposed_dfs["encoding"] = df_prop_enc
                        if st.button("Commit Encoding", key="btn_encode_commit", type="primary"):
                            with st.spinner(f"Encoding columns via {enc_method}..."):
                                push_history(df_current)
                                st.session_state["df"] = df_prop_enc
                            show_toast(f"Encoded via '{enc_method}'.")
                            st.rerun()
                    except Exception as err:
                        st.error(f"Encoding error: {err}")
                else:
                    st.info("Select at least one column to encode.")
            else:
                render_html('<div class="alert-box alert-success"><strong>Encoded:</strong> All features are currently in numeric format.</div>')

    # ---------------------------------------------------------
    # Inspector Panel
    # ---------------------------------------------------------
    with col_inspector:
        render_html('<div class="step-header"><span class="step-num indigo">02</span><span class="step-title">Dataset and Diff Telemetry</span></div>')
        hist_count = len(st.session_state.get("history", []))
        if hist_count > 0:
            st.caption(f"TELEMETRY: Undo Stack: {hist_count}/{MAX_HISTORY} steps available")

        tab_data, tab_diff, tab_profile = st.tabs(["Active Dataset", "Live Diff Preview", "Health Profile"])

        with tab_data:
            try:
                st.markdown(f"**Dimensions:** `{df_current.shape[0]:,}` rows × `{df_current.shape[1]}` columns")
                st.dataframe(safe_dataframe(df_current), width='stretch')
                buf = io.StringIO()
                df_current.to_csv(buf, index=False)
                st.download_button(
                    "Export Curated Dataset (.CSV)",
                    buf.getvalue().encode(),
                    f"curated_{active_file_name.replace('.', '_')}.csv",
                    "text/csv",
                    type="primary",
                    width='stretch',
                )
            except Exception as err:
                st.error(f"Failed to display dataset: {err}")

        with tab_diff:
            if proposed_dfs:
                op_labels = {
                    "missing": "Missing-Value Imputation",
                    "type_cast": "Manual Type Casting",
                    "coercion": "Batch Schema Coercion",
                    "outliers": "Outlier Handling",
                    "duplicates": "Duplicate Removal",
                    "encoding": "Categorical Encoding",
                }
                try:
                    sel_op = st.selectbox(
                        "Preview operation diff:",
                        list(proposed_dfs.keys()),
                        format_func=lambda x: op_labels.get(x, x),
                    )
                    df_prop = proposed_dfs[sel_op]
                    dm1, dm2, dm3, dm4 = st.columns(4)
                    r_diff = df_prop.shape[0] - df_current.shape[0]
                    c_diff = df_prop.shape[1] - df_current.shape[1]
                    m_curr = int(df_current.isna().sum().sum())
                    m_prop = int(df_prop.isna().sum().sum())
                    d_curr = int(df_current.duplicated().sum())
                    d_prop = int(df_prop.duplicated().sum())
                    dm1.metric("Rows", f"{df_prop.shape[0]:,}", delta=f"{r_diff:+d}" if r_diff != 0 else "Unchanged")
                    dm2.metric("Columns", df_prop.shape[1], delta=f"{c_diff:+d}" if c_diff != 0 else "Unchanged")
                    dm3.metric("Missing Cells", f"{m_prop:,}", delta=f"{m_prop - m_curr:+d}" if m_prop != m_curr else "Unchanged", delta_color="inverse")
                    dm4.metric("Duplicates", d_prop, delta=f"{d_prop - d_curr:+d}" if d_prop != d_curr else "Unchanged", delta_color="inverse")
                    st.dataframe(safe_dataframe(df_prop.head(50)), width='stretch')
                    st.info("Click the corresponding Commit button in the left panel to apply changes.")
                except Exception as err:
                    st.error(f"Diff preview error: {err}")
            else:
                st.info("Configure a pipeline operation on the left to inspect live diffs.")

        with tab_profile:
            try:
                prof = cleaner.profile()
                pp1, pp2, pp3 = st.columns(3)
                pp1.metric("Total Data Cells", f"{df_current.size:,}")
                if df_current.size > 0:
                    pp2.metric(
                        "Null Rate",
                        f"{df_current.isna().sum().sum() / df_current.size * 100:.2f}%",
                    )
                pp3.metric("Duplicate Count", int(df_current.duplicated().sum()))
                profile_df = pd.DataFrame({
                    "Type": prof["dtypes"],
                    "Nulls": prof["null_counts"],
                    "Null %": {c: f"{v:.1f}%" for c, v in prof["null_percentages"].items()},
                    "Unique Values": prof["unique_counts"],
                })
                st.dataframe(safe_dataframe(profile_df), width='stretch')
                st.dataframe(safe_dataframe(pd.DataFrame(prof["describe"])), width='stretch')
            except Exception as err:
                st.error(f"Profile error: {err}")

else:
    render_html(
        """
        <div class="alert-box alert-info">
            <strong>No dataset loaded yet.</strong> Upload a CSV or Excel file above to begin.
            VizML supports tabular files up to 100 MB.
        </div>
        """
    )
