import streamlit as st

_CLAY_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

/* Canvas & Global Reset */
html, body,
[data-testid="stApp"],
[data-testid="stAppViewContainer"],
[data-testid="stMainBlockContainer"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background-color: #ECEEF8 !important;
    background-image: radial-gradient(ellipse at 15% 10%, rgba(139,92,246,0.06) 0%, transparent 60%),
                      radial-gradient(ellipse at 85% 85%, rgba(99,102,241,0.06) 0%, transparent 60%) !important;
    background-attachment: fixed !important;
    color: #111827 !important;
}

[data-testid="stHeader"], [data-testid="stToolbar"] {
    background-color: #ECEEF8 !important;
    border-bottom: none !important;
    box-shadow: 0 2px 12px rgba(99,102,241,0.06) !important;
}

#MainMenu, footer, header { visibility: hidden; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #E8EBF6 !important;
    border-right: 1.5px solid rgba(99,102,241,0.12) !important;
    box-shadow: 4px 0 24px rgba(99,102,241,0.08) !important;
}

section[data-testid="stSidebar"] * {
    color: #111827 !important;
}

section[data-testid="stSidebar"] a {
    color: #111827 !important;
    font-weight: 600 !important;
}

section[data-testid="stSidebar"] a:hover {
    color: #4338CA !important;
}

/* Global Typography - Pure Black & Deep Slate */
h1, h2, h3, h4, h5, h6 {
    color: #111827 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 700 !important;
}

p, span, li, label, dt, dd {
    color: #1F2937 !important;
}

[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span {
    color: #1F2937 !important;
}

div[data-testid="stCaptionContainer"] p {
    color: #4B5563 !important;
    font-size: 12px !important;
    font-weight: 500 !important;
}

/* Clay Cards */
.glass-card {
    background: #FFFFFF;
    border-radius: 26px;
    border: 2px solid rgba(255, 255, 255, 0.95);
    box-shadow:
        0 1px 0 rgba(255,255,255,0.9) inset,
        0 8px 20px rgba(99,102,241,0.08),
        0 20px 48px rgba(99,102,241,0.11),
        0 2px 4px rgba(0,0,0,0.03);
    padding: 28px 32px;
    margin-bottom: 24px;
    transition: box-shadow 0.22s ease, transform 0.22s ease;
}

.glass-card:hover {
    transform: translateY(-2px);
    box-shadow:
        0 1px 0 rgba(255,255,255,0.9) inset,
        0 12px 28px rgba(99,102,241,0.12),
        0 28px 64px rgba(99,102,241,0.14);
}

.glass-card-sm {
    background: #FFFFFF;
    border-radius: 20px;
    border: 2px solid rgba(255, 255, 255, 0.95);
    box-shadow:
        0 1px 0 rgba(255,255,255,0.9) inset,
        0 6px 16px rgba(99,102,241,0.07),
        0 16px 36px rgba(99,102,241,0.09);
    padding: 22px 24px;
    margin-bottom: 16px;
    transition: box-shadow 0.22s ease, transform 0.22s ease;
}

.glass-card-sm:hover {
    transform: translateY(-2px);
    box-shadow:
        0 1px 0 rgba(255,255,255,0.9) inset,
        0 10px 24px rgba(99,102,241,0.11),
        0 22px 48px rgba(99,102,241,0.12);
}

/* Headings & Badges */
.eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #4338CA;
    background: rgba(99,102,241,0.12);
    padding: 5px 14px;
    border-radius: 99px;
    border: 1.5px solid rgba(99,102,241,0.25);
    display: inline-block;
    margin-bottom: 14px;
}

.page-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 34px;
    font-weight: 800;
    letter-spacing: -1px;
    color: #111827;
    margin-bottom: 10px;
    line-height: 1.15;
}

.page-sub {
    font-size: 14px;
    color: #4B5563;
    line-height: 1.65;
    max-width: 620px;
}

.section-header {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 16px;
    font-weight: 700;
    color: #111827;
    margin-top: 28px;
    margin-bottom: 14px;
    padding-bottom: 10px;
    border-bottom: 2px solid rgba(99,102,241,0.15);
    display: flex;
    align-items: center;
    gap: 10px;
}

.badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    padding: 5px 12px;
    border-radius: 99px;
    vertical-align: middle;
    display: inline-block;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}

.badge-indigo  { color:#3730A3; background:#EEF2FF; border:1.5px solid rgba(99,102,241,0.30); }
.badge-emerald { color:#065F46; background:#ECFDF5; border:1.5px solid rgba(5,150,105,0.30); }
.badge-violet  { color:#5B21B6; background:#F5F3FF; border:1.5px solid rgba(124,58,237,0.30); }
.badge-amber   { color:#92400E; background:#FFFBEB; border:1.5px solid rgba(217,119,6,0.30); }
.badge-rose    { color:#9F1239; background:#FFF1F2; border:1.5px solid rgba(225,29,72,0.30); }

/* File Uploader - Pure White Clay Styling */
[data-testid="stFileUploader"] {
    background-color: #FFFFFF !important;
    border-radius: 18px !important;
    padding: 12px !important;
    box-shadow: 0 4px 18px rgba(99,102,241,0.08) !important;
    border: 1.5px solid rgba(99,102,241,0.15) !important;
}

[data-testid="stFileUploader"] section {
    background-color: #FFFFFF !important;
}

[data-testid="stFileUploader"] * {
    color: #111827 !important;
}

[data-testid="stFileUploader"] button {
    background-color: #FFFFFF !important;
    color: #111827 !important;
    border: 1.5px solid rgba(99,102,241,0.25) !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}

/* Native Widgets, Selectboxes, Inputs */
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label,
[data-testid="stWidgetLabel"] span {
    color: #111827 !important;
    font-weight: 600 !important;
    font-size: 13px !important;
}

div[data-testid="stRadio"] label, div[data-testid="stRadio"] span, div[data-testid="stRadio"] p,
div[data-testid="stCheckbox"] label, div[data-testid="stCheckbox"] span, div[data-testid="stCheckbox"] p {
    color: #111827 !important;
    font-weight: 500 !important;
}

div[data-testid="stTextInput"] input,
div[data-testid="stNumberInput"] input {
    color: #111827 !important;
    background-color: #FFFFFF !important;
    border: 1.5px solid rgba(99,102,241,0.22) !important;
    border-radius: 12px !important;
}

div[data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
    border: 1.5px solid rgba(99,102,241,0.22) !important;
    border-radius: 12px !important;
    color: #111827 !important;
}

div[data-baseweb="select"] span, div[data-baseweb="select"] div {
    color: #111827 !important;
}

span[data-baseweb="tag"] {
    background-color: #EEF2FF !important;
    border: 1px solid rgba(99,102,241,0.30) !important;
    border-radius: 8px !important;
}

span[data-baseweb="tag"] * {
    color: #3730A3 !important;
    font-weight: 600 !important;
}

div[data-baseweb="popover"], div[data-baseweb="menu"], ul[data-baseweb="menu"] {
    background-color: #FFFFFF !important;
    border-radius: 14px !important;
    box-shadow: 0 10px 30px rgba(99,102,241,0.18) !important;
    border: 1px solid rgba(99,102,241,0.15) !important;
}

ul[data-baseweb="menu"] li, ul[data-baseweb="menu"] div, ul[data-baseweb="menu"] span {
    color: #111827 !important;
}

ul[data-baseweb="menu"] li:hover {
    background-color: #EEF2FF !important;
}

/* Expanders */
div[data-testid="stExpander"] {
    background: #FFFFFF !important;
    border: 1.5px solid rgba(99,102,241,0.18) !important;
    border-radius: 18px !important;
    margin-bottom: 12px !important;
    box-shadow: 0 4px 18px rgba(99,102,241,0.08) !important;
    overflow: hidden !important;
}

div[data-testid="stExpander"] details summary,
div[data-testid="stExpander"] details summary p,
div[data-testid="stExpander"] details summary span {
    color: #111827 !important;
    font-weight: 700 !important;
    font-size: 14px !important;
}

div[data-testid="stExpander"] details summary svg {
    fill: #111827 !important;
}

/* Tabs */
div[data-testid="stTabs"] button {
    color: #4B5563 !important;
    font-weight: 600 !important;
    font-size: 14px !important;
}

div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #4338CA !important;
    font-weight: 700 !important;
}

/* Buttons */
button[kind="primary"] {
    background-color: #6366F1 !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 14px rgba(99,102,241,0.28) !important;
}

button[kind="primary"] * {
    color: #FFFFFF !important;
}

button[kind="secondary"] {
    background-color: #FFFFFF !important;
    color: #111827 !important;
    border: 1.5px solid rgba(99,102,241,0.25) !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
}

button[kind="secondary"] * {
    color: #111827 !important;
}

/* Metrics */
div[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 26px !important;
    font-weight: 700 !important;
    color: #4338CA !important;
}

div[data-testid="stMetricLabel"] {
    font-size: 11px !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: #4B5563 !important;
}

/* Session Rows */
.session-row {
    display: flex;
    justify-content: space-between;
    font-size: 13px;
    padding: 8px 0;
    border-bottom: 1px solid rgba(99,102,241,0.08);
}
.session-row:last-child { border-bottom: none; }
.session-label { color: #4B5563; font-size: 13px; font-weight: 500; }
.session-val {
    font-family: 'JetBrains Mono', monospace;
    color: #4338CA;
    font-weight: 700;
    font-size: 13px;
}
</style>
"""


def inject_css():
    st.markdown(_CLAY_CSS, unsafe_allow_html=True)
