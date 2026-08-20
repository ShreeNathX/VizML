import textwrap
from typing import Any, Optional
import streamlit as st

_CLAY_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Baloo+2:wght@500;600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');

/* ===== CLAY DESIGN TOKENS ===== */
:root {
    --white:     #FFFFFF;
    --bg:        #F5F3FC;
    --bg-deep:   #EFEBFA;
    --ink:       #211F36;
    --ink-soft:  #726E90;
    --ink-faint: #A7A3C2;

    --indigo:    #6C5CE7;
    --indigo-l:  #8B7DF0;
    --indigo-bg: #EDE9FE;

    --coral:     #FF5D8F;
    --coral-l:   #FF83AA;
    --coral-bg:  #FFE4EC;

    --cyan:      #22C0F5;
    --cyan-l:    #5CD2FF;
    --cyan-bg:   #DEF6FF;

    --lime:      #A6E22E;
    --lime-l:    #C0F156;
    --lime-bg:   #EEFAD1;

    --orange:    #FF8A3D;
    --orange-l:  #FFA666;
    --orange-bg: #FFE9D6;

    --r-lg:   28px;
    --r-md:   20px;
    --r-sm:   14px;
    --r-pill: 999px;

    --sh-light: rgba(255,255,255,0.95);
    --sh-dark:  rgba(163,160,201,0.55);
}

/* ===== PRESERVE STREAMLIT MATERIAL ICONS & SYMBOLS ===== */
[data-testid="stIconMaterial"],
[data-testid="stIcon"],
.material-symbols-rounded,
.material-symbols-outlined,
.material-icons,
[class*="material-symbols"],
[class*="material-icons"],
[data-testid="stSidebarCollapseButton"] span,
[data-testid="stSidebarCollapseButton"] i,
[data-testid="stSidebarCollapseButton"] svg,
[data-testid="stFileUploader"] span[data-testid="stIconMaterial"],
[data-testid="stFileUploader"] .material-symbols-rounded,
[data-testid="stFileUploader"] .material-symbols-outlined,
[data-testid="stFileUploader"] span[data-testid="stIcon"],
button span[data-testid="stIconMaterial"],
button .material-symbols-rounded,
button .material-symbols-outlined,
button .material-icons,
button [data-testid="stIcon"] {
    font-family: 'Material Symbols Rounded', 'Material Symbols Outlined', 'Material Icons' !important;
    font-style: normal !important;
    font-weight: normal !important;
    line-height: 1 !important;
    letter-spacing: normal !important;
    text-transform: none !important;
    display: inline-block !important;
    white-space: nowrap !important;
    word-wrap: normal !important;
    direction: ltr !important;
    -webkit-font-feature-settings: 'liga' !important;
    font-feature-settings: 'liga' !important;
    -webkit-font-smoothing: antialiased !important;
}

/* ===== GLOBAL RESET ===== */
*, *::before, *::after { box-sizing: border-box; }

html, body,
[data-testid="stApp"],
[data-testid="stAppViewContainer"] {
    background:
      radial-gradient(1.5px 1.5px at 24px 24px, rgba(108,92,231,0.08) 1.5px, transparent 0) 0 0/48px 48px,
      var(--bg) !important;
    font-family: 'Plus Jakarta Sans', sans-serif;
    color: var(--ink) !important;
    -webkit-font-smoothing: antialiased;
}

/* Remove old dark grid overlay */
[data-testid="stAppViewContainer"] {
    background-image: none !important;
}

.main .block-container,
[data-testid="stMainBlockContainer"],
[data-testid="block-container"] {
    padding-top: 6.5rem !important;
    padding-bottom: 4rem !important;
    max-width: 95% !important;
    background-color: transparent !important;
}

#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }

header[data-testid="stHeader"] {
    background-color: transparent !important;
    border-bottom: none !important;
    box-shadow: none !important;
}

/* ===== SIDEBAR ===== */
section[data-testid="stSidebar"] {
    background-color: var(--white) !important;
    background-image: none !important;
    border-right: none !important;
    box-shadow: 6px 0 20px rgba(163,160,201,0.22) !important;
    font-family: 'Plus Jakarta Sans', sans-serif;
}
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span:not([data-testid="stIconMaterial"]):not([class*="material-symbols"]):not([class*="material-icons"]),
section[data-testid="stSidebar"] label {
    color: var(--ink-soft) !important;
    font-family: 'Plus Jakarta Sans', sans-serif;
}
section[data-testid="stSidebar"] a {
    color: var(--indigo) !important;
    font-weight: 600 !important;
    border-radius: var(--r-sm) !important;
    transition: background 0.15s ease;
    font-family: 'Plus Jakarta Sans', sans-serif;
}
section[data-testid="stSidebar"] a:hover {
    background: var(--indigo-bg) !important;
    color: var(--indigo) !important;
}

/* ===== TYPOGRAPHY ===== */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Baloo 2', sans-serif !important;
    color: var(--ink) !important;
    font-weight: 700 !important;
    margin: 0 !important;
}

p, li, dt, dd {
    color: var(--ink) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {
    color: var(--ink) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

div[data-testid="stCaptionContainer"] p {
    color: var(--ink-soft) !important;
    font-size: 12px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

/* ===== CLAY CARD PRIMITIVES ===== */
.clay-raised {
    background: var(--white);
    border-radius: var(--r-md);
    box-shadow: 10px 10px 22px var(--sh-dark), -10px -10px 22px var(--sh-light);
}

.clay-pressed {
    background: var(--bg-deep);
    border-radius: var(--r-md);
    box-shadow: inset 7px 7px 14px var(--sh-dark), inset -7px -7px 14px var(--sh-light);
}

.clay-btn {
    border: none;
    cursor: pointer;
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-weight: 700;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.clay-btn:hover  { transform: translateY(-3px); }
.clay-btn:active {
    transform: translateY(1px);
    box-shadow: inset 5px 5px 10px var(--sh-dark), inset -5px -5px 10px var(--sh-light) !important;
}

/* ===== SHARED CARDS ===== */
.bp-card, .glass-card {
    background: var(--white);
    border-radius: var(--r-md);
    padding: 26px 28px;
    margin-bottom: 20px;
    box-shadow: 10px 10px 22px var(--sh-dark), -10px -10px 22px var(--sh-light);
    position: relative;
}

.bp-card-sm, .glass-card-sm {
    background: var(--white);
    border-radius: var(--r-sm);
    padding: 18px 20px;
    margin-bottom: 14px;
    box-shadow: 6px 6px 14px var(--sh-dark), -6px -6px 14px var(--sh-light);
    position: relative;
}

/* ===== HEADERS & EYEBROWS ===== */
.bp-header {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 11.5px;
    font-weight: 800;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    color: var(--indigo);
    margin-bottom: 8px;
}

.eyebrow {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 12.5px;
    font-weight: 800;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--indigo);
    display: inline-block;
    margin-bottom: 6px;
}

.page-title {
    font-family: 'Baloo 2', sans-serif;
    font-size: 30px;
    font-weight: 800;
    color: var(--ink);
    margin-bottom: 10px;
    line-height: 1.15;
}

.page-sub {
    font-size: 14.5px;
    color: var(--ink-soft);
    line-height: 1.55;
    max-width: 680px;
    font-weight: 500;
}

/* Section header with a bottom rule for visual rhythm */
.section-header {
    font-family: 'Baloo 2', sans-serif;
    font-size: 17px;
    font-weight: 700;
    color: var(--ink);
    margin-top: 28px;
    margin-bottom: 16px;
    padding-bottom: 8px;
    border-bottom: 2px solid var(--bg-deep);
    display: flex;
    align-items: center;
    gap: 12px;
}

/* ===== STEP HEADERS & BADGES ===== */
.step-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-top: 26px;
    margin-bottom: 16px;
    padding-bottom: 8px;
    border-bottom: 2px solid var(--bg-deep);
}
.step-num {
    width: 32px;
    height: 32px;
    border-radius: 10px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    font-weight: 800;
    color: white;
    background: linear-gradient(145deg, var(--indigo-l), var(--indigo));
    box-shadow: 3px 3px 8px var(--sh-dark), -2px -2px 6px var(--sh-light);
    flex-shrink: 0;
}
.step-num.coral  { background: linear-gradient(145deg, var(--coral-l), var(--coral)); }
.step-num.cyan   { background: linear-gradient(145deg, var(--cyan-l), var(--cyan)); color: #07475E; }
.step-num.lime   { background: linear-gradient(145deg, var(--lime-l), var(--lime)); color: #2B4806; }
.step-num.orange { background: linear-gradient(145deg, var(--orange-l), var(--orange)); }
.step-title {
    font-family: 'Baloo 2', sans-serif;
    font-size: 17.5px;
    font-weight: 700;
    color: var(--ink);
}

/* ===== BADGES ===== */
.badge {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 10.5px;
    font-weight: 800;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: var(--r-pill);
    display: inline-block;
}
.badge-indigo, .badge-blueprint { color: var(--indigo);  background: var(--indigo-bg); }
.badge-gold,   .badge-amber     { color: #9A4A0E;         background: var(--orange-bg); }
.badge-emerald,.badge-success, .badge-lime { color: #3B5F0B; background: var(--lime-bg);   }
.badge-rose,   .badge-danger,  .badge-coral { color: #B32357; background: var(--coral-bg);  }
.badge-violet, .badge-cyan      { color: #0B7EA6;         background: var(--cyan-bg);   }

/* ===== KPI DISPLAY BOX ===== */
.bp-display {
    background: var(--bg-deep);
    border-radius: var(--r-sm);
    padding: 12px 14px;
    margin-bottom: 14px;
    box-shadow: inset 4px 4px 9px var(--sh-dark), inset -4px -4px 9px var(--sh-light);
    overflow: hidden;
    min-width: 0;
}
.bp-expr {
    font-size: 10.5px;
    color: var(--ink-faint);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.bp-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: clamp(14px, 1.2vw, 20px);
    font-weight: 700;
    color: var(--ink);
    white-space: normal;
    word-break: break-word;
    overflow-wrap: break-word;
    line-height: 1.25;
}

/* ===== STAT TILES ===== */
.stat-tile {
    background: var(--white);
    border-radius: var(--r-md);
    padding: 22px;
    position: relative;
    overflow: hidden;
    box-shadow: 8px 8px 18px var(--sh-dark), -8px -8px 18px var(--sh-light);
}
.stat-tile .tile-icon {
    position: absolute; top: 16px; right: 16px;
    width: 36px; height: 36px; border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
}
.stat-tile .num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 28px; font-weight: 700; color: var(--ink);
}
.stat-tile .lab {
    font-size: 12.5px; color: var(--ink-soft); font-weight: 700; margin-top: 4px;
}

/* ===== CHIP ===== */
.chip {
    padding: 9px 16px; border-radius: var(--r-pill);
    font-size: 12.5px; font-weight: 700; color: var(--ink-soft);
    display: inline-flex; align-items: center; gap: 8px;
    background: var(--white);
    box-shadow: 4px 4px 10px var(--sh-dark), -4px -4px 10px var(--sh-light);
    cursor: pointer; user-select: none; transition: all 0.15s ease;
}
.chip.on, .chip:hover {
    background: var(--indigo-bg); color: var(--indigo);
    box-shadow: inset 3px 3px 7px rgba(108,92,231,.2), inset -3px -3px 7px var(--sh-light);
}
.chip .dot { width: 7px; height: 7px; border-radius: 50%; background: var(--indigo); }

/* ===== PILL GROUP ===== */
.pill-group { display: flex; gap: 10px; flex-wrap: wrap; }
.pill-group input { display: none; }
.pill-group label {
    padding: 11px 18px; border-radius: var(--r-pill);
    background: var(--white); font-size: 13px; font-weight: 700; color: var(--ink-soft);
    cursor: pointer;
    box-shadow: 4px 4px 10px var(--sh-dark), -4px -4px 10px var(--sh-light);
    transition: all 0.15s ease;
}
.pill-group input:checked + label {
    background: linear-gradient(145deg, var(--indigo-l), var(--indigo));
    color: white;
    box-shadow: inset 3px 3px 7px rgba(0,0,0,.15), -3px -3px 8px var(--sh-light);
}

/* ===== ALERT / CALLOUT BOXES ===== */
.alert-box {
    border-radius: var(--r-sm);
    padding: 14px 18px; margin-bottom: 16px;
    font-size: 13.5px; font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 500;
    line-height: 1.5;
}
.alert-box strong { font-weight: 800; }
.alert-info    { background: var(--indigo-bg); color: var(--indigo);
    box-shadow: inset 4px 4px 9px rgba(108,92,231,.15), inset -4px -4px 9px var(--sh-light); }
.alert-warning { background: var(--orange-bg); color: #9A4A0E;
    box-shadow: inset 4px 4px 9px rgba(255,138,61,.18), inset -4px -4px 9px var(--sh-light); }
.alert-success { background: var(--lime-bg); color: #3B5F0B;
    box-shadow: inset 4px 4px 9px rgba(166,226,46,.22), inset -4px -4px 9px var(--sh-light); }
.alert-danger  { background: var(--coral-bg); color: #B32357;
    box-shadow: inset 4px 4px 9px rgba(255,93,143,.18), inset -4px -4px 9px var(--sh-light); }

/* ===== BANNERS ===== */
.dup-banner {
    display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 14px;
    padding: 18px 22px; background: var(--coral-bg); border-radius: var(--r-md);
    box-shadow: inset 4px 4px 9px rgba(255,93,143,.18), inset -4px -4px 9px rgba(255,255,255,.7);
}
.dup-banner .msg { font-size: 14px; font-weight: 700; color: #B32357; }

.task-banner {
    display: flex; align-items: center; gap: 14px; padding: 18px 22px; margin-bottom: 22px;
    background: var(--lime-bg); border-radius: var(--r-md);
    box-shadow: inset 4px 4px 9px rgba(166,226,46,.22), inset -4px -4px 9px rgba(255,255,255,.75);
}
.task-banner .t1 { font-weight: 800; font-size: 14px; color: #3B5F0B; }
.task-banner .t2 { font-size: 12.5px; color: #5C7A2E; font-weight: 600; }

/* ===== SESSION ROWS ===== */
.session-row {
    display: flex; justify-content: space-between;
    font-size: 13px; padding: 8px 0;
    border-bottom: 1px solid rgba(163,160,201,0.25);
    font-family: 'Plus Jakarta Sans', sans-serif;
}
.session-row:last-child { border-bottom: none; }
.session-label { color: var(--ink-soft); font-weight: 600; }
.session-val   { color: var(--indigo); font-weight: 700; }

/* ===== BUTTONS (HTML) ===== */
.big-btn {
    padding: 14px 26px; border-radius: var(--r-pill); color: white;
    font-size: 14.5px; font-weight: 700;
    display: inline-flex; align-items: center; gap: 10px;
    box-shadow: 8px 8px 16px var(--sh-dark), -6px -6px 14px var(--sh-light);
    background: linear-gradient(145deg, var(--indigo-l), var(--indigo));
    border: none; cursor: pointer; transition: transform 0.15s ease;
}
.big-btn:hover { transform: translateY(-3px); }

.small-btn {
    padding: 10px 18px; border-radius: var(--r-pill); color: white; font-size: 12.5px;
    background: linear-gradient(145deg, var(--coral-l), var(--coral));
    box-shadow: 5px 5px 10px var(--sh-dark), -4px -4px 10px var(--sh-light);
    border: none; cursor: pointer; font-weight: 700;
}

/* ===== BRAND ELEMENTS ===== */
.brand-mark {
    width: 46px; height: 46px; border-radius: 16px;
    background: linear-gradient(145deg, var(--indigo-l), var(--indigo));
    box-shadow: 6px 6px 14px var(--sh-dark), -6px -6px 14px var(--sh-light);
    display: flex; align-items: center; justify-content: center;
}
.status-chip {
    display: inline-flex; align-items: center; gap: 8px;
    padding: 9px 16px; border-radius: var(--r-pill);
    font-size: 13px; font-weight: 700; color: #3B8F1F; background: var(--lime-bg);
    box-shadow: inset 3px 3px 7px rgba(166,226,46,.25), inset -3px -3px 7px rgba(255,255,255,.7);
}
.status-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: #3B8F1F; box-shadow: 0 0 0 3px rgba(59,143,31,.18);
}

/* ===== DROPZONE ===== */
.dropzone {
    padding: 44px 30px; text-align: center;
    border: 3px dashed var(--indigo-l); border-radius: var(--r-md); cursor: pointer;
    background: var(--bg-deep);
    box-shadow: inset 7px 7px 14px var(--sh-dark), inset -7px -7px 14px var(--sh-light);
}
.dropzone .icon-circle {
    width: 74px; height: 74px; margin: 0 auto 18px; border-radius: 50%;
    background: var(--indigo-bg); display: flex; align-items: center; justify-content: center;
    color: var(--indigo);
}

/* ===== MODEL TABLE (HTML) ===== */
.model-row {
    display: grid; grid-template-columns: 1.4fr 2fr 70px 70px;
    align-items: center; gap: 18px; padding: 16px 20px;
    background: var(--white); border-radius: 16px;
    box-shadow: 5px 5px 12px var(--sh-dark), -5px -5px 12px var(--sh-light);
    margin-bottom: 12px;
}
.model-row .mname { display: flex; align-items: center; gap: 10px; font-weight: 700; font-size: 14px; }
.best-badge {
    font-size: 10px; font-weight: 800; color: white;
    background: linear-gradient(145deg, var(--lime-l), var(--lime));
    padding: 3px 9px; border-radius: var(--r-pill);
}
.bar-track {
    height: 12px; border-radius: 8px; background: var(--bg-deep);
    box-shadow: inset 2px 2px 5px var(--sh-dark), inset -2px -2px 5px var(--sh-light); overflow: hidden;
}
.bar-fill { height: 100%; border-radius: 8px; background: linear-gradient(90deg, var(--indigo-l), var(--indigo)); }
.bar-fill.best { background: linear-gradient(90deg, var(--lime-l), var(--lime)); }
.mono-val { font-family: 'JetBrains Mono', monospace; font-size: 13px; font-weight: 700; text-align: right; color: var(--ink-soft); }

/* ===== EXPORT CARDS (HTML) ===== */
.export-card {
    padding: 28px; text-align: center; cursor: pointer;
    background: var(--white); border-radius: var(--r-md);
    box-shadow: 10px 10px 22px var(--sh-dark), -10px -10px 22px var(--sh-light);
    transition: transform 0.15s ease;
}
.export-card:hover { transform: translateY(-4px); }
.export-card .ic { width: 60px; height: 60px; margin: 0 auto 16px; border-radius: 18px; display: flex; align-items: center; justify-content: center; }
.export-card.csv .ic { background: linear-gradient(145deg, var(--indigo-l), var(--indigo)); }
.export-card.pkl .ic { background: linear-gradient(145deg, var(--orange-l), var(--orange)); }
.export-card h4  { font-family: 'Baloo 2', sans-serif; font-size: 16px; color: var(--ink); }
.export-card .size { font-family: 'JetBrains Mono', monospace; font-size: 12px; color: var(--ink-soft); margin-top: 6px; }

.local-banner {
    display: flex; align-items: center; gap: 14px; padding: 18px 22px;
    background: var(--orange-bg); border-radius: var(--r-md); border: 2px solid var(--orange-l);
}
.local-banner .t1 { font-weight: 800; font-size: 13.5px; color: #9A4A0E; }
.local-banner .t2 { font-size: 12px; color: #B15F1F; font-weight: 600; }

/* ============================================================
   STREAMLIT NATIVE WIDGET OVERRIDES
   ============================================================ */

/* --- File Uploader --- */
[data-testid="stFileUploader"] {
    background-color: var(--bg-deep) !important;
    border: 2px dashed var(--indigo-l) !important;
    border-radius: var(--r-md) !important;
    box-shadow: inset 4px 4px 10px var(--sh-dark), inset -4px -4px 10px var(--sh-light) !important;
    padding: 22px !important;
}
[data-testid="stFileUploader"] section { background-color: transparent !important; }
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] label {
    color: var(--ink-soft) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}
[data-testid="stFileUploader"] button,
[data-testid="stFileUploader"] button[data-testid="stBaseButton-secondary"],
[data-testid="stFileUploader"] button[kind="secondary"] {
    background: linear-gradient(145deg, var(--indigo-l), var(--indigo)) !important;
    background-color: var(--indigo) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: var(--r-pill) !important;
    font-weight: 700 !important;
    padding: 10px 24px !important;
    box-shadow: 4px 4px 10px var(--sh-dark), -2px -2px 6px var(--sh-light) !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 8px !important;
    cursor: pointer !important;
}
[data-testid="stFileUploader"] button *,
[data-testid="stFileUploader"] button[data-testid="stBaseButton-secondary"] *,
[data-testid="stFileUploader"] button span,
[data-testid="stFileUploader"] button p {
    color: #ffffff !important;
    font-weight: 700 !important;
    fill: #ffffff !important;
    font-size: 13.5px !important;
    display: inline-block !important;
    visibility: visible !important;
    opacity: 1 !important;
}
[data-testid="stFileUploader"] button svg,
[data-testid="stFileUploader"] button [data-testid="stIconMaterial"] {
    color: #ffffff !important;
    fill: #ffffff !important;
}
[data-testid="stFileUploadDropzone"] {
    background: transparent !important;
    border: none !important;
}

/* --- Widget labels --- */
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label,
[data-testid="stWidgetLabel"] span {
    color: var(--ink-soft) !important;
    font-weight: 700 !important;
    font-size: 12.5px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

/* --- Text / Number / Textarea Inputs --- */
div[data-testid="stTextInput"] input,
div[data-testid="stNumberInput"] input,
div[data-testid="stTextArea"] textarea {
    color: var(--ink) !important;
    background-color: var(--white) !important;
    border: none !important;
    border-radius: var(--r-sm) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    box-shadow: inset 4px 4px 8px var(--sh-dark), inset -4px -4px 8px var(--sh-light) !important;
}
div[data-testid="stTextInput"] input:focus,
div[data-testid="stNumberInput"] input:focus,
div[data-testid="stTextArea"] textarea:focus {
    outline: none !important;
    box-shadow: inset 4px 4px 8px var(--sh-dark), inset -4px -4px 8px var(--sh-light),
                0 0 0 2px var(--indigo) !important;
}

/* --- Selectboxes / Dropdowns --- */
div[data-baseweb="select"] > div {
    background-color: var(--white) !important;
    border: none !important;
    border-radius: var(--r-sm) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    box-shadow: inset 4px 4px 8px var(--sh-dark), inset -4px -4px 8px var(--sh-light) !important;
}
div[data-baseweb="select"] span,
div[data-baseweb="select"] div { color: var(--ink) !important; }
div[data-baseweb="select"] input { color: var(--ink) !important; background-color: transparent !important; }

span[data-baseweb="tag"] {
    background-color: var(--indigo-bg) !important;
    border: none !important;
    border-radius: var(--r-pill) !important;
}
span[data-baseweb="tag"] * { color: var(--indigo) !important; font-weight: 700 !important; }

div[data-baseweb="popover"],
div[data-baseweb="menu"],
ul[data-baseweb="menu"] {
    background-color: var(--white) !important;
    border: none !important;
    border-radius: var(--r-md) !important;
    box-shadow: 12px 12px 28px var(--sh-dark), -6px -6px 16px var(--sh-light) !important;
}
ul[data-baseweb="menu"] li,
ul[data-baseweb="menu"] div,
ul[data-baseweb="menu"] span {
    color: var(--ink) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}
ul[data-baseweb="menu"] li:hover,
ul[data-baseweb="menu"] li[aria-selected="true"] {
    background-color: var(--indigo-bg) !important;
    color: var(--indigo) !important;
}

/* --- EXPANDERS (FIX: only color, never break layout) ---
   The arrow icon is a sibling SVG inside <summary>.
   We must NOT touch display/position on summary children
   or the arrow will overlap the label text. */
div[data-testid="stExpander"] {
    background: var(--white) !important;
    border: none !important;
    border-radius: var(--r-md) !important;
    box-shadow: 7px 7px 16px var(--sh-dark), -7px -7px 16px var(--sh-light) !important;
    margin-bottom: 14px !important;
    overflow: hidden;
}

/* Only style the expander's label text element - NOT the whole summary */
div[data-testid="stExpander"] details summary {
    border-radius: var(--r-md) !important;
    padding: 14px 18px !important;
    background: var(--white) !important;
    list-style: none !important;
    cursor: pointer;
}

/* The label text (p tag) inside the summary */
div[data-testid="stExpander"] details > summary > div > p,
div[data-testid="stExpander"] details > summary p {
    color: var(--ink) !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    margin: 0 !important;
}

/* The expand/collapse arrow SVG: keep it positioned correctly */
div[data-testid="stExpander"] details > summary svg {
    color: var(--indigo) !important;
    flex-shrink: 0 !important;
}

/* Content area inside expander */
div[data-testid="stExpander"] [data-testid="stExpanderDetails"] {
    padding: 4px 18px 16px !important;
}

/* --- Tabs --- */
div[data-testid="stTabs"] button {
    color: var(--ink-soft) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 13.5px !important;
    border-radius: 0 !important;
    border-bottom: 2px solid transparent !important;
    background: none !important;
    transition: color 0.15s ease !important;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--indigo) !important;
    border-bottom: 2px solid var(--indigo) !important;
    font-weight: 800 !important;
}
div[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: none !important;
    border-bottom: 2px solid var(--bg-deep) !important;
    gap: 4px;
}

/* --- Primary Streamlit Buttons & Primary Download Buttons --- */
button[kind="primary"],
button[data-testid="stBaseButton-primary"],
.stButton > button[kind="primary"],
div[data-testid="stDownloadButton"] button[kind="primary"],
div[data-testid="stDownloadButton"] button[data-testid="stBaseButton-primary"],
.stDownloadButton button[kind="primary"],
.stDownloadButton button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(145deg, var(--indigo-l), var(--indigo)) !important;
    background-color: var(--indigo) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: var(--r-pill) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 700 !important;
    box-shadow: 6px 6px 14px var(--sh-dark), -4px -4px 10px var(--sh-light) !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
    letter-spacing: 0.02em;
}
button[kind="primary"]:hover,
button[data-testid="stBaseButton-primary"]:hover,
div[data-testid="stDownloadButton"] button[kind="primary"]:hover,
.stDownloadButton button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 8px 8px 18px var(--sh-dark), -6px -6px 14px var(--sh-light) !important;
}
button[kind="primary"] *,
button[data-testid="stBaseButton-primary"] *,
div[data-testid="stDownloadButton"] button[kind="primary"] *,
div[data-testid="stDownloadButton"] button[data-testid="stBaseButton-primary"] *,
.stDownloadButton button[kind="primary"] *,
.stDownloadButton button[data-testid="stBaseButton-primary"] * {
    color: #ffffff !important;
    font-weight: 700 !important;
}

/* Code tags inside primary buttons - ensure crisp semi-transparent pill */
button[kind="primary"] code,
button[data-testid="stBaseButton-primary"] code,
.stButton > button[kind="primary"] code,
div[data-testid="stDownloadButton"] button[kind="primary"] code {
    background-color: rgba(255, 255, 255, 0.25) !important;
    color: #ffffff !important;
    border: 1px solid rgba(255, 255, 255, 0.4) !important;
    padding: 2px 6px !important;
    border-radius: 6px !important;
    font-weight: 700 !important;
}

/* General inline code tags across the platform */
code {
    background-color: var(--indigo-bg) !important;
    color: var(--indigo) !important;
    border: 1px solid rgba(108, 92, 231, 0.2) !important;
    padding: 2px 6px !important;
    border-radius: 6px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.9em !important;
    font-weight: 600 !important;
}

/* --- Secondary Streamlit Buttons & Secondary Download Buttons --- */
button[kind="secondary"],
button[data-testid="stBaseButton-secondary"],
.stButton > button:not([kind="primary"]):not([data-testid="stBaseButton-primary"]),
div[data-testid="stDownloadButton"] button:not([kind="primary"]):not([data-testid="stBaseButton-primary"]),
.stDownloadButton button:not([kind="primary"]):not([data-testid="stBaseButton-primary"]) {
    background: var(--white) !important;
    background-color: var(--white) !important;
    color: var(--indigo) !important;
    border: 2px solid rgba(108, 92, 231, 0.3) !important;
    border-radius: var(--r-pill) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 700 !important;
    box-shadow: 5px 5px 12px var(--sh-dark), -5px -5px 12px var(--sh-light) !important;
    transition: transform 0.15s ease, border-color 0.15s ease, background 0.15s ease !important;
}
button[kind="secondary"]:hover,
button[data-testid="stBaseButton-secondary"]:hover,
.stButton > button:not([kind="primary"]):not([data-testid="stBaseButton-primary"]):hover,
div[data-testid="stDownloadButton"] button:not([kind="primary"]):not([data-testid="stBaseButton-primary"]):hover,
.stDownloadButton button:not([kind="primary"]):not([data-testid="stBaseButton-primary"]):hover {
    transform: translateY(-2px) !important;
    border-color: var(--indigo) !important;
    background: var(--indigo-bg) !important;
    background-color: var(--indigo-bg) !important;
    color: var(--indigo) !important;
}
button[kind="secondary"] *,
button[data-testid="stBaseButton-secondary"] *,
.stButton > button:not([kind="primary"]):not([data-testid="stBaseButton-primary"]) *,
div[data-testid="stDownloadButton"] button:not([kind="primary"]):not([data-testid="stBaseButton-primary"]) *,
.stDownloadButton button:not([kind="primary"]):not([data-testid="stBaseButton-primary"]) * {
    color: var(--indigo) !important;
    font-weight: 700 !important;
}

/* --- Metrics --- */
[data-testid="metric-container"] {
    background: var(--white) !important;
    border-radius: var(--r-sm) !important;
    padding: 16px !important;
    box-shadow: 6px 6px 14px var(--sh-dark), -6px -6px 14px var(--sh-light) !important;
}
div[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 26px !important;
    font-weight: 700 !important;
    color: var(--ink) !important;
}
div[data-testid="stMetricLabel"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 11.5px !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--ink-soft) !important;
}
[data-testid="stMetricDelta"] { font-family: 'JetBrains Mono', monospace !important; }

/* --- Dataframe / Table wrapper --- */
[data-testid="stDataFrame"],
[data-testid="stTable"] {
    border-radius: var(--r-md) !important;
    overflow: hidden;
    box-shadow: 6px 6px 16px var(--sh-dark), -6px -6px 16px var(--sh-light) !important;
    border: none !important;
}

/* --- Radio Buttons --- */
[data-testid="stRadio"] label,
[data-testid="stRadio"] label p {
    color: var(--ink) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 600 !important;
}

/* --- Checkboxes --- */
[data-testid="stCheckbox"] label,
[data-testid="stCheckbox"] label p {
    color: var(--ink) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 500 !important;
}

/* --- Sliders --- */
[data-testid="stSlider"] [role="slider"] { background: var(--indigo) !important; }

/* --- Progress bar --- */
[data-testid="stProgressBar"] > div { background: var(--indigo) !important; border-radius: 8px !important; }
[data-testid="stProgressBar"] { background: var(--bg-deep) !important; border-radius: 8px !important; }

/* --- Toast notifications --- */
[data-testid="stToast"] {
    background: var(--white) !important;
    border-radius: var(--r-md) !important;
    box-shadow: 10px 10px 24px var(--sh-dark), -6px -6px 14px var(--sh-light) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

/* --- Page links --- */
[data-testid="stPageLink"] a {
    color: var(--indigo) !important;
    font-weight: 700 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 14px !important;
}
[data-testid="stPageLink"] a:hover { text-decoration: underline !important; }

/* --- st.info / st.warning / st.success / st.error native boxes --- */
[data-testid="stAlert"] {
    border-radius: var(--r-sm) !important;
    border: none !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

/* --- Code blocks --- */
code {
    background: var(--indigo-bg) !important;
    color: var(--indigo) !important;
    border-radius: 6px !important;
    padding: 1px 6px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.9em !important;
}

/* --- Plotly chart containers and Modebar Controls --- */
[data-testid="stPlotlyChart"] {
    border-radius: var(--r-md) !important;
    overflow: visible !important;
    box-shadow: 6px 6px 16px var(--sh-dark), -6px -6px 16px var(--sh-light) !important;
    background: var(--white) !important;
    padding: 10px !important;
}
[data-testid="stPlotlyChart"] > div { background: transparent !important; }

/* Plotly Modebar styling: visible on chart hover for clean unobtrusive view */
.js-plotly-plot .plotly .modebar-container {
    top: 6px !important;
    right: 10px !important;
    z-index: 100 !important;
}

.js-plotly-plot .plotly .modebar {
    opacity: 0 !important;
    visibility: hidden !important;
    pointer-events: none !important;
    transition: opacity 0.22s ease, visibility 0.22s ease !important;
    background: rgba(255, 255, 255, 0.94) !important;
    backdrop-filter: blur(8px) !important;
    border-radius: var(--r-pill) !important;
    padding: 3px 6px !important;
    box-shadow: 3px 3px 10px var(--sh-dark), -2px -2px 8px var(--sh-light) !important;
    border: 1px solid rgba(163, 160, 201, 0.35) !important;
    display: inline-flex !important;
    align-items: center !important;
    gap: 2px !important;
}

.js-plotly-plot:hover .plotly .modebar,
.js-plotly-plot .plotly .modebar:hover {
    opacity: 1 !important;
    visibility: visible !important;
    pointer-events: auto !important;
}

.js-plotly-plot .plotly .modebar-group {
    background: transparent !important;
    padding: 0 1px !important;
    display: inline-flex !important;
    align-items: center !important;
    border: none !important;
}

.js-plotly-plot .plotly .modebar-btn,
.js-plotly-plot .plotly .modebar-btn:not([kind="primary"]) {
    width: 22px !important;
    height: 22px !important;
    min-width: 22px !important;
    min-height: 22px !important;
    padding: 2px !important;
    margin: 0 1px !important;
    border-radius: 6px !important;
    border: 1px solid rgba(108, 92, 231, 0.12) !important;
    background: var(--white) !important;
    background-color: var(--white) !important;
    box-shadow: 1px 1px 4px var(--sh-dark) !important;
    cursor: pointer !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    transition: all 0.15s ease !important;
    opacity: 0.9 !important;
}

.js-plotly-plot .plotly .modebar-btn svg {
    width: 14px !important;
    height: 14px !important;
    display: block !important;
}

.js-plotly-plot .plotly .modebar-btn svg path {
    fill: #6C5CE7 !important;
    transition: fill 0.15s ease !important;
}

.js-plotly-plot .plotly .modebar-btn:hover {
    background: var(--indigo-bg) !important;
    background-color: var(--indigo-bg) !important;
    border-color: var(--indigo) !important;
    transform: scale(1.12) !important;
    opacity: 1.0 !important;
}

.js-plotly-plot .plotly .modebar-btn:hover svg path,
.js-plotly-plot .plotly .modebar-btn.active svg path {
    fill: #211F36 !important;
}

.js-plotly-plot .plotly .modebar-btn.active {
    background: var(--indigo-bg) !important;
    border-color: var(--indigo) !important;
}

/* --- Column gaps --- */
[data-testid="stColumns"] { gap: 1.2rem !important; }

/* --- Captions --- */
[data-testid="stCaptionContainer"] p {
    color: var(--ink-soft) !important;
    font-size: 12px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 500 !important;
}

/* --- Horizontal rule --- */
hr { border: none !important; border-top: 2px solid var(--bg-deep) !important; }

@media (prefers-reduced-motion: reduce) {
    * { animation: none !important; transition: none !important; }
}
</style>
"""


def render_html(html_content: str) -> None:
    """Renders raw HTML safely directly in Streamlit's container with full CSS support."""
    cleaned = textwrap.dedent(html_content).strip()
    st.markdown(cleaned, unsafe_allow_html=True)


def inject_css() -> None:
    """Injects the Claymorphic design system CSS into Streamlit."""
    render_html(_CLAY_CSS)


# Curated discrete color palettes for multi-category and single-series styling
COLOR_PALETTES = {
    "Clay Classic (Default)": [
        "#6C5CE7", "#FF5D8F", "#22C0F5", "#A6E22E", "#FF8A3D", "#8B7DF0", "#FF83AA"
    ],
    "Electric Neon": [
        "#00F5D4", "#7B2CBF", "#F72585", "#4CC9F0", "#FFE600", "#7209B7", "#4361EE"
    ],
    "Ocean Sunset": [
        "#2E384D", "#00B4D8", "#F77F00", "#FCBF49", "#E63946", "#457B9D", "#D62828"
    ],
    "Emerald Forest": [
        "#2D6A4F", "#52B788", "#74C69D", "#95D5B2", "#D8F3DC", "#1B4332", "#40916C"
    ],
    "Berry Velvet": [
        "#7209B7", "#B5179E", "#C77DFF", "#E0AAFF", "#560BAD", "#F72585", "#3A0CA3"
    ],
    "Obsidian & Gold": [
        "#D4AF37", "#2B2D42", "#8D99AE", "#EF233C", "#D90429", "#F4A261", "#E76F51"
    ],
    "Pastel Dream": [
        "#B3C5FF", "#FFB3C6", "#C7F9CC", "#FFEAA7", "#DFCCF1", "#A8DADC", "#FFCAD4"
    ],
    "Vibrant Warmth": [
        "#E63946", "#F4A261", "#E76F51", "#2A9D8F", "#264653", "#E9C46A", "#F77F00"
    ],
    "Monochrome Indigo": [
        "#3F37C9", "#4361EE", "#4895EF", "#4CC9F0", "#6C5CE7", "#8B7DF0", "#A594F9"
    ],
}

# Continuous color scales for heatmaps, numeric hue gradients, and density contours
CONTINUOUS_COLOR_SCALES = [
    "Clay Violet",
    "RdBu_r",
    "Viridis",
    "Plasma",
    "Inferno",
    "Turbo",
    "Warm Coral",
    "Cool Cyan",
    "Emerald Glow",
    "Amber Sunset",
    "Spectral",
    "Blues",
    "Purples",
    "Reds",
    "Greens",
]


def resolve_continuous_scale(scale_name: str) -> Any:
    """Resolves custom or named continuous color scales for Plotly."""
    custom_scales = {
        "Clay Violet": ["#EDE9FE", "#8B7DF0", "#6C5CE7", "#3B28B0"],
        "Warm Coral": ["#FFE4EC", "#FF83AA", "#FF5D8F", "#B32357"],
        "Cool Cyan": ["#DEF6FF", "#5CD2FF", "#22C0F5", "#0B7EA6"],
        "Emerald Glow": ["#EEFAD1", "#C0F156", "#A6E22E", "#2D6A4F"],
        "Amber Sunset": ["#FFE9D6", "#FFA666", "#FF8A3D", "#9A4A0E"],
    }
    return custom_scales.get(scale_name, scale_name)


def apply_clay_theme(fig: Any, title: Optional[str] = None) -> Any:
    """Standardizes a Plotly figure with the Clay soft-light styling, generous margins, automargins, and visible legends."""
    if fig is None:
        return fig

    clay_layout = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(245,243,252,0.5)",
        font=dict(family="Plus Jakarta Sans, sans-serif", color="#211F36", size=12),
        xaxis=dict(
            gridcolor="rgba(163,160,201,0.2)",
            linecolor="rgba(163,160,201,0.35)",
            tickcolor="rgba(163,160,201,0.35)",
            tickfont=dict(family="Plus Jakarta Sans, sans-serif", color="#726E90", size=11),
            title=dict(font=dict(family="Baloo 2, sans-serif", color="#211F36", size=13), standoff=8),
            zerolinecolor="rgba(163,160,201,0.3)",
            automargin=True,
        ),
        yaxis=dict(
            gridcolor="rgba(163,160,201,0.2)",
            linecolor="rgba(163,160,201,0.35)",
            tickcolor="rgba(163,160,201,0.35)",
            tickfont=dict(family="Plus Jakarta Sans, sans-serif", color="#726E90", size=11),
            title=dict(font=dict(family="Baloo 2, sans-serif", color="#211F36", size=13), standoff=8),
            zerolinecolor="rgba(163,160,201,0.3)",
            automargin=True,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(163,160,201,0.25)",
            borderwidth=1,
            font=dict(family="Plus Jakarta Sans, sans-serif", color="#211F36", size=11),
        ),
        margin=dict(t=58 if title else 28, b=52, l=62, r=32),
        modebar=dict(
            bgcolor="rgba(245,243,252,0.85)",
            color="#6C5CE7",
            activecolor="#211F36",
        ),
    )

    if title:
        clay_layout["title"] = dict(
            text=title,
            font=dict(family="Baloo 2, sans-serif", size=15, color="#211F36"),
            x=0.02,
            y=0.97,
            xanchor="left",
            yanchor="top",
        )

    fig.update_layout(**clay_layout)
    return fig


CLAY_COLOR_SEQUENCE = COLOR_PALETTES["Clay Classic (Default)"]
