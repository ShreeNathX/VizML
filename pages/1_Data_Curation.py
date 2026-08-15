import io
import os

import numpy as np
import pandas as pd
import streamlit as st

from src.cleaner import DataCleaner
from src.styles import inject_css

MAX_FILE_MB = 100
MAX_HISTORY = 10

st.set_page_config(page_title="VizML: Data Curation Engine", layout="wide")
inject_css()

st.markdown(
    """
<div class="glass-card">
    <span class="eyebrow">Phase 01 | Pipeline Module</span>
    <div class="page-title">Data Curation Engine</div>
    <div class="page-sub">
        Profile dataset health, coerce types, impute missing values, handle outliers,
        remove duplicates, and inspect live diff previews with full undo history.
    </div>
</div>
""",
    unsafe_allow_html=True,
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
        if out[col].dtype == object or isinstance(out[col].dtype, pd.StringDtype):
            out[col] = out[col].fillna("").astype(str)
        elif pd.api.types.is_numeric_dtype(out[col]):
            continue
    return out


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

        st.session_state["df"] = data_df
        st.session_state["original_df"] = data_df.copy()
        st.session_state["history"] = []
        st.session_state["coercion_flags"] = {}
        show_toast(f"Loaded {file_obj.name} ({data_df.shape[0]:,} rows, {data_df.shape[1]} cols).")
    except Exception as err:
        st.error(f"Failed to load dataset: {err}")


def push_history(df: pd.DataFrame) -> None:
    history: list = st.session_state["history"]
    history.append(df.copy())
    if len(history) > MAX_HISTORY:
        history.pop(0)
    st.session_state["history"] = history


uploaded_file = st.file_uploader(
    f"Upload Dataset (CSV or Excel, max {MAX_FILE_MB} MB)",
    type=["csv", "xlsx", "xls"],
)

if uploaded_file is not None:
    if st.session_state.get("current_file_name") != uploaded_file.name:
        st.session_state["current_file_name"] = uploaded_file.name
        load_data(uploaded_file)
else:
    sample_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "Test Dataset", "messy_data.csv"
    )
    if st.session_state["df"] is None and os.path.exists(sample_path):
        try:
            df_sample = pd.read_csv(sample_path)
            st.session_state["df"] = df_sample
            st.session_state["original_df"] = df_sample.copy()
            st.session_state["history"] = []
            st.session_state["coercion_flags"] = {}
            st.session_state["current_file_name"] = "messy_data.csv (sample)"
        except Exception as err:
            st.warning(f"Could not load sample dataset: {err}")

if st.session_state["df"] is not None:
    df_current: pd.DataFrame = st.session_state["df"]

    ctrl_c1, ctrl_c2, _ = st.columns([2, 2, 8])
    with ctrl_c1:
        undo_disabled = len(st.session_state["history"]) == 0
        if st.button("Undo Last Action", disabled=undo_disabled, width="stretch"):
            try:
                st.session_state["df"] = st.session_state["history"].pop()
                show_toast("Undid last cleaning operation.")
                st.rerun()
            except Exception as err:
                st.error(f"Undo failed: {err}")
    with ctrl_c2:
        if st.button("Reset to Original", width="stretch"):
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
        st.markdown(
            '<div class="section-header"><span class="badge badge-indigo">TOOLKIT</span> Cleaning Operations</div>',
            unsafe_allow_html=True,
        )

        try:
            cleaner = DataCleaner(df_current)
            cleaner.flagged_columns = st.session_state["coercion_flags"]
        except Exception as err:
            cleaner = DataCleaner(df_current)

        proposed_dfs: dict = {}

        # 1. Missing values
        with st.expander("1. Missing-Value Imputation", expanded=False):
            null_cols = [c for c in df_current.columns if df_current[c].isna().any()]
            orig_nulls = (
                int(st.session_state["original_df"].isna().sum().sum())
                if st.session_state.get("original_df") is not None
                else 0
            )

            if null_cols:
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
                        if st.button("Apply Imputation", key="btn_missing", type="primary", width="stretch"):
                            push_history(df_current)
                            st.session_state["df"] = df_prop_missing
                            show_toast("Missing values imputed.")
                            st.rerun()
                    except Exception as err:
                        st.error(f"Imputation error: {err}")
                else:
                    st.info("Select at least one column to impute.")
            else:
                if orig_nulls > 0:
                    st.success(f"Missing values fixed. All {orig_nulls:,} missing cells resolved.")
                else:
                    st.success("No missing values detected in dataset.")

        # 2. Type coercion
        with st.expander("2. Smart Type Coercion", expanded=False):
            try:
                df_prop_coerce = cleaner.coerce_types()
                proposed_dfs["coercion"] = df_prop_coerce
                coerced = [
                    (c, str(df_current[c].dtype), str(df_prop_coerce[c].dtype))
                    for c in df_current.columns
                    if str(df_current[c].dtype) != str(df_prop_coerce[c].dtype)
                ]
                if coerced:
                    st.markdown("**Proposed type changes:**")
                    for c, old_t, new_t in coerced:
                        st.write(f"- `{c}`: `{old_t}` → `{new_t}`")
                else:
                    st.info("No coercible columns found.")
                for c, flag in cleaner.flagged_columns.items():
                    cls = "flag-warning" if "Partial" in flag else "flag-error"
                    label = "Partial Cast" if "Partial" in flag else "Unparseable"
                    st.markdown(f"`{c}`: <span class='{cls}'>{label}</span> — {flag}", unsafe_allow_html=True)
                if coerced or cleaner.flagged_columns:
                    if st.button("Apply Type Coercion", key="btn_coerce", type="primary", width="stretch"):
                        push_history(df_current)
                        st.session_state["df"] = df_prop_coerce
                        st.session_state["coercion_flags"] = cleaner.flagged_columns
                        show_toast("Type coercion applied.")
                        st.rerun()
            except Exception as err:
                st.error(f"Type coercion error: {err}")

        # 3. Outlier handling
        with st.expander("3. Outlier Handling (IQR)", expanded=False):
            num_cols_all = [
                c for c in df_current.columns
                if pd.api.types.is_numeric_dtype(df_current[c])
                and not pd.api.types.is_bool_dtype(df_current[c])
            ]
            if num_cols_all:
                outlier_cols = st.multiselect("Columns to evaluate", num_cols_all, default=num_cols_all)
                if not outlier_cols:
                    st.info("Select at least one numeric column to evaluate.")
                else:
                    mask = cleaner.detect_outliers(columns=outlier_cols)
                    total_outliers = int(mask.sum().sum())
                    if total_outliers == 0:
                        st.success("No outliers detected across selected columns (IQR 1.5x threshold).")
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
                                st.write(f"- `{c}`: {n}")
                        try:
                            df_prop_outliers = cleaner.handle_outliers(
                                action=outlier_action, columns=outlier_cols
                            )
                            proposed_dfs["outliers"] = df_prop_outliers
                            if st.button(
                                f"Apply Outlier {outlier_action.capitalize()}",
                                key="btn_outlier",
                                type="primary",
                                width="stretch",
                            ):
                                push_history(df_current)
                                st.session_state["df"] = df_prop_outliers
                                show_toast(f"Outliers handled via '{outlier_action}'.")
                                st.rerun()
                        except Exception as err:
                            st.error(f"Outlier handling error: {err}")
            else:
                st.info("No numeric columns available for outlier detection.")

        # 4. Duplicate removal
        with st.expander("4. Duplicate Removal", expanded=False):
            try:
                dup_count = cleaner.get_duplicate_count()
                st.write(f"Duplicate rows found: **{dup_count}**")
                if dup_count > 0:
                    df_prop_dups = cleaner.remove_duplicates()
                    proposed_dfs["duplicates"] = df_prop_dups
                    if st.button("Remove Duplicates", key="btn_dups", type="primary", width="stretch"):
                        push_history(df_current)
                        st.session_state["df"] = df_prop_dups
                        show_toast(f"Removed {dup_count} duplicate rows.")
                        st.rerun()
                else:
                    st.success("No duplicate rows found.")
            except Exception as err:
                st.error(f"Duplicate check error: {err}")

        # 5. Categorical encoding
        with st.expander("5. Categorical Encoding", expanded=False):
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
                        if st.button("Apply Encoding", key="btn_encode", type="primary", width="stretch"):
                            push_history(df_current)
                            st.session_state["df"] = df_prop_enc
                            show_toast(f"Encoded via '{enc_method}'.")
                            st.rerun()
                    except Exception as err:
                        st.error(f"Encoding error: {err}")
                else:
                    st.info("Select at least one column to encode.")
            else:
                st.info("No categorical columns detected.")

    with col_inspector:
        st.markdown(
            '<div class="section-header"><span class="badge badge-indigo">INSPECTOR</span> Dataset Preview & Diff Tracker</div>',
            unsafe_allow_html=True,
        )
        hist_count = len(st.session_state["history"])
        if hist_count > 0:
            st.caption(f"Undo stack: {hist_count}/{MAX_HISTORY} operations")

        tab_data, tab_diff, tab_profile = st.tabs(["Active Dataset", "Live Diff Finder", "Health Profile"])

        with tab_data:
            try:
                st.markdown(f"**Shape:** `{df_current.shape[0]:,}` rows x `{df_current.shape[1]}` columns")
                st.dataframe(safe_dataframe(df_current), width="stretch")
                buf = io.StringIO()
                df_current.to_csv(buf, index=False)
                st.download_button(
                    "Export Curated CSV",
                    buf.getvalue().encode(),
                    "curated_dataset.csv",
                    "text/csv",
                    width="stretch",
                )
            except Exception as err:
                st.error(f"Failed to display dataset: {err}")

        with tab_diff:
            if proposed_dfs:
                op_labels = {
                    "missing": "Missing-Value Imputation",
                    "coercion": "Type Coercion",
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
                    dm3.metric("Missing", f"{m_prop:,}", delta=f"{m_prop - m_curr:+d}" if m_prop != m_curr else "Unchanged", delta_color="inverse")
                    dm4.metric("Duplicates", d_prop, delta=f"{d_prop - d_curr:+d}" if d_prop != d_curr else "Unchanged", delta_color="inverse")
                    st.dataframe(safe_dataframe(df_prop.head(50)), width="stretch")
                    st.info("Click Apply in the left panel to commit this change.")
                except Exception as err:
                    st.error(f"Diff preview error: {err}")
            else:
                st.info("Configure a cleaning operation on the left to see a live diff.")

        with tab_profile:
            try:
                prof = cleaner.profile()
                pp1, pp2, pp3 = st.columns(3)
                pp1.metric("Total Cells", f"{df_current.size:,}")
                if df_current.size > 0:
                    pp2.metric(
                        "Missing Rate",
                        f"{df_current.isna().sum().sum() / df_current.size * 100:.2f}%",
                    )
                pp3.metric("Duplicate Rows", int(df_current.duplicated().sum()))
                profile_df = pd.DataFrame({
                    "Dtype": prof["dtypes"],
                    "Missing": prof["null_counts"],
                    "Missing %": {c: f"{v:.1f}%" for c, v in prof["null_percentages"].items()},
                    "Unique": prof["unique_counts"],
                })
                st.dataframe(safe_dataframe(profile_df), width="stretch")
                st.dataframe(safe_dataframe(pd.DataFrame(prof["describe"])), width="stretch")
            except Exception as err:
                st.error(f"Profile error: {err}")

else:
    st.info("Upload a CSV or Excel file above to begin, or wait for the sample dataset to load.")
