import streamlit as st
import pandas as pd
import numpy as np
import io
import importlib
import src.cleaner
importlib.reload(src.cleaner)
from src.cleaner import DataCleaner

# Configure page settings
st.set_page_config(page_title="VizML - Data Upload & Clean", layout="wide")

def show_toast(message, icon=None):
    if hasattr(st, "toast"):
        st.toast(message, icon=icon)
    else:
        icon_str = f"{icon} " if icon else ""
        st.sidebar.info(f"{icon_str}{message}")


# Modern CSS Injection for Premium Look
st.markdown("""
    <style>
    /* Styling Metrics Cards */
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 700;
        color: #6366F1;
    }
    div[data-testid="stMetricDelta"] {
        font-size: 14px;
    }
    
    /* Elegant Title and Subtitle styling */
    .app-title {
        font-family: 'Outfit', 'Inter', sans-serif;
        font-size: 38px;
        font-weight: 800;
        background: linear-gradient(135deg, #6366F1 0%, #EC4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
    }
    .app-subtitle {
        font-size: 16px;
        color: #6B7280;
        margin-bottom: 25px;
    }

    /* Step containers style */
    .step-card {
        border-radius: 12px;
        padding: 20px;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        margin-bottom: 15px;
    }
    
    /* Tag styles */
    .flag-warning {
        background-color: #FEF3C7;
        color: #D97706;
        padding: 4px 8px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 600;
        border: 1px solid #FDE68A;
    }
    .flag-error {
        background-color: #FEE2E2;
        color: #DC2626;
        padding: 4px 8px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 600;
        border: 1px solid #FCA5A5;
    }
    .flag-success {
        background-color: #D1FAE5;
        color: #059669;
        padding: 4px 8px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 600;
        border: 1px solid #6EE7B7;
    }

    /* Force expander header text to be visible regardless of theme/extensions */
    div[data-testid="stExpander"] summary,
    div[data-testid="stExpander"] summary p,
    div[data-testid="stExpander"] summary span,
    .streamlit-expanderHeader,
    .streamlit-expanderHeader p {
        color: #FFFFFF !important;
        opacity: 1 !important;
        font-weight: 600 !important;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="app-title">VizML Curation Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">Upload, profile, and clean your data with real-time change tracking and undo capability.</div>', unsafe_allow_html=True)

# ----------------- SESSION STATE SETUP -----------------
if "df" not in st.session_state:
    st.session_state["df"] = None
if "original_df" not in st.session_state:
    st.session_state["original_df"] = None
if "history" not in st.session_state:
    st.session_state["history"] = []
if "coercion_flags" not in st.session_state:
    st.session_state["coercion_flags"] = {}

def load_data(uploaded_file):
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.session_state["df"] = df
        st.session_state["original_df"] = df.copy()
        st.session_state["history"] = []
        st.session_state["coercion_flags"] = {}
        show_toast("Dataset uploaded and initialized successfully.")
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")

# ----------------- UPLOAD WIDGET -----------------
uploaded_file = st.file_uploader("Drop your dataset here (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Check if a new file is uploaded
    if st.session_state["original_df"] is None or st.session_state.get("current_file_name") != uploaded_file.name:
        st.session_state["current_file_name"] = uploaded_file.name
        load_data(uploaded_file)
else:
    import os
    sample_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "messy_data.csv")
    if st.session_state["df"] is None and os.path.exists(sample_path):
        df_fallback = pd.read_csv(sample_path)
        st.session_state["df"] = df_fallback
        st.session_state["original_df"] = df_fallback.copy()
        st.session_state["history"] = []
        st.session_state["coercion_flags"] = {}

# If dataset is loaded, proceed
if st.session_state["df"] is not None:
    df_current = st.session_state["df"]
    
    # ----------------- HISTORY & UNDO ACTIONS -----------------
    col_hist1, col_hist2, col_hist3 = st.columns([1, 1, 6])
    with col_hist1:
        undo_disabled = len(st.session_state["history"]) == 0
        if st.button("Undo Last Action", disabled=undo_disabled, use_container_width=True):
            st.session_state["df"] = st.session_state["history"].pop()
            show_toast("Undid last cleaning operation.")
            st.rerun()
            
    with col_hist2:
        if st.button("Reset to Original", use_container_width=True):
            st.session_state["df"] = st.session_state["original_df"].copy()
            st.session_state["history"] = []
            st.session_state["coercion_flags"] = {}
            show_toast("Reset dataset to original state.")
            st.rerun()
            
    st.markdown("---")

    # Create two primary columns: Controls (left) and Visual Diff + Data (right)
    col_left, col_right = st.columns([5, 7])

    with col_left:
        st.subheader("Cleaning Operations")
        
        # Instantiate a cleaner with the current state of df
        cleaner = DataCleaner(df_current)
        # Apply any coercion flags already stored in state
        cleaner.flagged_columns = st.session_state["coercion_flags"]
        
        # We will define a dict of proposed dataframes per cleaning tab
        proposed_dfs = {}
        operation_applied = None

        # Operation 1: Missing values
        with st.expander(r"1\. Missing-Value Imputation", expanded=False):
            st.markdown("Impute missing values based on columns types.")
            
            # Numeric columns config
            st.markdown("**Numeric Columns**")
            num_strat = st.radio("Numeric Imputation Strategy", ["median", "mean", "ffill", "bfill"], index=0, key="missing_num_strat", horizontal=True)
            
            # Categorical columns config
            st.markdown("**Categorical Columns**")
            cat_strat = st.radio("Categorical Imputation Strategy", ["mode", "constant", "ffill", "bfill"], index=0, key="missing_cat_strat", horizontal=True)
            cat_const = st.text_input("Constant Fill Token", value="Unknown", key="missing_cat_const")
            
            # Identify columns with nulls
            null_cols = [c for c in df_current.columns if df_current[c].isna().any()]
            if null_cols:
                cols_to_impute = st.multiselect("Select columns to impute (Default: All)", null_cols, default=null_cols)
                
                df_proposed_missing = cleaner.handle_missing(
                    numeric_strategy=num_strat,
                    categorical_strategy=cat_strat,
                    categorical_constant=cat_const,
                    columns=cols_to_impute
                )
                proposed_dfs["missing"] = df_proposed_missing
                
                if st.button("Apply Imputation", key="btn_apply_missing", type="primary"):
                    st.session_state["history"].append(df_current.copy())
                    st.session_state["df"] = df_proposed_missing
                    show_toast("Missing values imputed.")
                    st.rerun()
            else:
                st.success("No missing values detected in the current dataset!")

        # Operation 2: Type Coercion
        with st.expander(r"2\. Smart Type Coercion", expanded=False):
            st.markdown("""
                Automatically convert columns of text/object types that represent numeric or datetime values to their actual types.
                *Unparseable columns are flagged rather than silently dropped.*
            """)
            
            df_proposed_coercion = cleaner.coerce_types()
            proposed_dfs["coercion"] = df_proposed_coercion
            
            # Show proposed casts and warnings
            coerced_cols = []
            flagged_cols = []
            
            for col in df_current.columns:
                old_dtype = str(df_current[col].dtype)
                new_dtype = str(df_proposed_coercion[col].dtype)
                if old_dtype != new_dtype:
                    coerced_cols.append((col, old_dtype, new_dtype))
                if col in cleaner.flagged_columns:
                    flagged_cols.append((col, cleaner.flagged_columns[col]))
                    
            if coerced_cols:
                st.markdown("**Proposed Type Changes:**")
                for col, old_t, new_t in coerced_cols:
                    st.write(f"• `{col}`: `{old_t}` -> `{new_t}`")
            else:
                st.info("No columns suitable for type coercion detected.")
                
            if flagged_cols:
                st.markdown("**Parser Status & Flags:**")
                for col, flag in flagged_cols:
                    if "Partial" in flag:
                        st.markdown(f"`{col}`: <span class='flag-warning'>Partial Cast</span> - {flag}", unsafe_allow_html=True)
                    else:
                        st.markdown(f"`{col}`: <span class='flag-error'>Unparseable</span> - {flag}", unsafe_allow_html=True)
                        
            if coerced_cols or flagged_cols:
                if st.button("Apply Type Coercion", key="btn_apply_coercion", type="primary"):
                    st.session_state["history"].append(df_current.copy())
                    st.session_state["df"] = df_proposed_coercion
                    st.session_state["coercion_flags"] = cleaner.flagged_columns
                    show_toast("Type coercion completed.")
                    st.rerun()

        # Operation 3: Outlier Handling
        with st.expander(r"3\. Outlier Handling (IQR Method)", expanded=False):
            st.markdown("Detect outliers using the IQR range ($Q1 - 1.5 \\times IQR$, $Q3 + 1.5 \\times IQR$).")
            
            num_cols = [c for c in df_current.columns if pd.api.types.is_numeric_dtype(df_current[c]) and not pd.api.types.is_bool_dtype(df_current[c])]
            if num_cols:
                outlier_cols = st.multiselect("Select columns to handle outliers (Default: All)", num_cols, default=num_cols)
                outlier_action = st.radio("Outlier Action", ["clip", "remove", "flag"], index=0, key="outlier_action_choice", horizontal=True)
                
                df_proposed_outliers = cleaner.handle_outliers(action=outlier_action, columns=outlier_cols)
                proposed_dfs["outliers"] = df_proposed_outliers
                
                # Show outlier summary
                outlier_mask = cleaner.detect_outliers(columns=outlier_cols)
                total_outliers = outlier_mask.sum().sum()
                
                st.write(f"Total outlier cells detected: **{total_outliers}**")
                for col in outlier_cols:
                    col_outlier_count = outlier_mask[col].sum()
                    if col_outlier_count > 0:
                        st.write(f"• `{col}`: {col_outlier_count} outliers")
                        
                if st.button("Apply Outlier Action", key="btn_apply_outliers", type="primary"):
                    st.session_state["history"].append(df_current.copy())
                    st.session_state["df"] = df_proposed_outliers
                    show_toast(f"Outliers handled using {outlier_action}.")
                    st.rerun()
            else:
                st.info("No numeric columns available for outlier detection.")

        # Operation 4: Duplicate Removal
        with st.expander(r"4\. Duplicate Removal", expanded=False):
            dup_count = cleaner.get_duplicate_count()
            st.write(f"Exact duplicate rows detected: **{dup_count}**")
            
            if dup_count > 0:
                df_proposed_dups = cleaner.remove_duplicates()
                proposed_dfs["duplicates"] = df_proposed_dups
                
                if st.button("Remove Duplicates", key="btn_apply_dups", type="primary"):
                    st.session_state["history"].append(df_current.copy())
                    st.session_state["df"] = df_proposed_dups
                    show_toast(f"Removed {dup_count} duplicate rows.")
                    st.rerun()
            else:
                st.success("No duplicate rows found!")

        # Operation 5: Categorical Encoding
        with st.expander(r"5\. Categorical Encoding (Prep Step)", expanded=False):
            st.markdown("""
                Convert categorical columns into numerical representation. 
                *Note: You can defer this until modeling if you want to keep categories human-readable for visualization.*
            """)
            
            cat_cols = []
            for col in df_current.columns:
                if (pd.api.types.is_object_dtype(df_current[col]) or 
                    isinstance(df_current[col].dtype, pd.CategoricalDtype) or 
                    pd.api.types.is_string_dtype(df_current[col]) or
                    pd.api.types.is_bool_dtype(df_current[col])):
                    cat_cols.append(col)
                    
            if cat_cols:
                cols_to_encode = st.multiselect("Select columns to encode (Default: All)", cat_cols, default=cat_cols)
                encoding_method = st.radio("Encoding Method", ["onehot", "label"], index=0, key="encoding_method_choice", horizontal=True)
                
                df_proposed_encode = cleaner.encode_categoricals(method=encoding_method, columns=cols_to_encode)
                proposed_dfs["encoding"] = df_proposed_encode
                
                if st.button("Apply Categorical Encoding", key="btn_apply_encoding", type="primary"):
                    st.session_state["history"].append(df_current.copy())
                    st.session_state["df"] = df_proposed_encode
                    show_toast(f"Encoded categories using {encoding_method}.")
                    st.rerun()
            else:
                st.info("No categorical columns detected in the current dataset.")

    # ----------------- DISPLAY PREVIEW & LIVE DIFF -----------------
    with col_right:
        st.subheader("Data Preview & Change Tracker")
        
        # Figure out if any expander is focused or we have active configurations
        # We can dynamically show a diff if we have a proposed dataframe
        active_prop_key = None
        # Check which expander controls are active (we can guess based on session state changes or simply provide a selector)
        available_props = list(proposed_dfs.keys())
        
        tab_data, tab_diff, tab_profile = st.tabs(["Active Dataset", "Live Diff Finder", "Data Profile"])
        
        with tab_data:
            st.markdown(f"**Shape:** `{df_current.shape[0]}` rows, `{df_current.shape[1]}` columns")
            st.dataframe(df_current, use_container_width=True)
            
            # Export functionality
            csv_buffer = io.StringIO()
            df_current.to_csv(csv_buffer, index=False)
            csv_bytes = csv_buffer.getvalue().encode('utf-8')
            
            st.download_button(
                label="Export Cleaned Dataset",
                data=csv_bytes,
                file_name="cleaned_dataset.csv",
                mime="text/csv",
                use_container_width=True
            )
            
        with tab_diff:
            if available_props:
                selected_diff_op = st.selectbox(
                    "Select cleaning operation to preview:",
                    available_props,
                    format_func=lambda x: {
                        "missing": "Missing-Value Imputation",
                        "coercion": "Smart Type Coercion",
                        "outliers": "Outlier Handling",
                        "duplicates": "Duplicate Removal",
                        "encoding": "Categorical Encoding"
                    }.get(x, x)
                )
                
                df_proposed = proposed_dfs[selected_diff_op]
                
                # Show changes metrics
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                
                # Row comparison
                r_old, r_new = df_current.shape[0], df_proposed.shape[0]
                r_diff = r_new - r_old
                col_m1.metric("Row Count", r_new, delta=f"{r_diff} rows" if r_diff != 0 else "Unchanged")
                
                # Column comparison
                c_old, c_new = df_current.shape[1], df_proposed.shape[1]
                c_diff = c_new - c_old
                col_m2.metric("Column Count", c_new, delta=f"{c_diff} cols" if c_diff != 0 else "Unchanged")
                
                # Missing value comparison
                m_old = df_current.isna().sum().sum()
                m_new = df_proposed.isna().sum().sum()
                m_diff = m_new - m_old
                col_m3.metric("Missing Cells", m_new, delta=f"{m_diff} cells" if m_diff != 0 else "Unchanged", delta_color="inverse")
                
                # Duplicate rows comparison
                d_old = df_current.duplicated().sum()
                d_new = df_proposed.duplicated().sum()
                d_diff = d_new - d_old
                col_m4.metric("Duplicates", d_new, delta=f"{d_diff} rows" if d_diff != 0 else "Unchanged", delta_color="inverse")
                
                st.write("### Proposed DataFrame Preview:")
                st.dataframe(df_proposed.head(50), use_container_width=True)
                
                st.info("Review the changes above. If they look correct, click the **Apply** button in the corresponding cleaning panel on the left to commit them.")
            else:
                st.info("Configure a cleaning operation on the left to view a live diff.")

        with tab_profile:
            prof = cleaner.profile()
            
            col_p1, col_p2, col_p3 = st.columns(3)
            col_p1.write(f"**Total Cells:** {df_current.size}")
            col_p2.write(f"**Missing Values:** {df_current.isna().sum().sum()} ({df_current.isna().sum().sum() / df_current.size * 100:.2f}%)" if df_current.size > 0 else "")
            col_p3.write(f"**Duplicates:** {df_current.duplicated().sum()}")
            
            # Show summary profile table
            profile_df = pd.DataFrame({
                "Dtype": prof["dtypes"],
                "Missing Count": prof["null_counts"],
                "Missing %": {col: f"{v:.2f}%" for col, v in prof["null_percentages"].items()},
                "Unique Values": prof["unique_counts"]
            })
            st.dataframe(profile_df, use_container_width=True)
            
            # Show description summary
            st.write("#### Detailed Feature Statistics")
            st.dataframe(pd.DataFrame(prof["describe"]), use_container_width=True)
else:
    st.info("Please upload a CSV or Excel file to begin profiling and cleaning.")