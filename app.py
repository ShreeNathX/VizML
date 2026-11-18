import streamlit as st

from src.styles import inject_css

st.set_page_config(
    page_title="VizML: Local Data Studio",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

pages = [
    st.Page("pages/0_Home.py", title="Home", url_path="home", default=True),
    st.Page("pages/1_Data_Curation.py", title="Data Curation", url_path="Data_Curation"),
    st.Page(
        "pages/2_Diagnostic_Visualizations.py",
        title="Diagnostic Visualizations",
        url_path="Diagnostic_Visualizations",
    ),
    st.Page("pages/3_ML_Studio.py", title="ML Studio", url_path="ML_Studio"),
    st.Page("pages/4_Executive_Reports.py", title="Executive Summary", url_path="Executive_Reports"),
]

st.navigation(pages).run()
