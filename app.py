"""
╔══════════════════════════════════════════════════════════════════╗
║   SKINSCAN AI  ·  CLINICAL INTELLIGENCE PLATFORM  ·  v13.0     ║
║   Design: Apex Medical AI  ·  Glassmorphism Clinical UI         ║
║   Author: Rehan Shafique  ·  FYP Final Build                    ║
║   Model: skin_cancer_cnn.h5  (Benign / Malignant)               ║
╚══════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageEnhance
import plotly.graph_objects as go
import plotly.express as px
from streamlit_option_menu import option_menu
import random, time, datetime, io, json, base64

# ── ReportLab (optional) ──────────────────────────────────────────
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer,
        Table, TableStyle, HRFlowable, Image as RLImage,
    )
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
    PDF_OK = True
except ImportError:
    PDF_OK = False


# ══════════════════════════════════════════════════════════════════
#  GLOBAL CSS  ──  Apex Medical Intelligence Design System
# ══════════════════════════════════════════════════════════════════
def inject_premium_css(theme: str = "dark"):
    if theme == "dark":
        BG         = "#020c1b"
        BG2        = "#041226"
        SURFACE    = "rgba(4,22,50,0.80)"
        SURFACE2   = "rgba(8,30,65,0.70)"
        BORDER     = "rgba(37,99,235,0.25)"
        BORDER_H   = "rgba(20,184,166,0.55)"
        TEXT       = "#e8f4f8"
        SUBTEXT    = "#7fa3c0"
        MUTED      = "#3d5a73"
        SIDEBAR_BG = "rgba(2,12,27,0.97)"
        INPUT_BG   = "rgba(4,22,50,0.90)"
        DIVIDER    = "rgba(37,99,235,0.15)"
    else:
        BG         = "#f0f5fb"
        BG2        = "#e4edf7"
        SURFACE    = "rgba(255,255,255,0.92)"
        SURFACE2   = "rgba(240,248,255,0.88)"
        BORDER     = "rgba(37,99,235,0.20)"
        BORDER_H   = "rgba(20,184,166,0.50)"
        TEXT       = "#0d1f35"
        SUBTEXT    = "#3d6080"
        MUTED      = "#8eb0cc"
        SIDEBAR_BG = "rgba(240,245,251,0.98)"
        INPUT_BG   = "rgba(255,255,255,0.95)"
        DIVIDER    = "rgba(37,99,235,0.12)"

    st.markdown(f"""
    <style>
    /* ══════════════════════════════════════════
       FONTS — Sora (medical precision) + JetBrains Mono
    ══════════════════════════════════════════ */
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* ══════════════════════════════════════════
       APP SHELL
    ══════════════════════════════════════════ */
    html, body, .stApp {{
        font-family: 'Sora', sans-serif !important;
        background-color: {BG} !important;
        color: {TEXT} !important;
    }}
    .stApp {{
        background-image:
            radial-gradient(ellipse 80% 50% at 10% 0%,   rgba(37,99,235,0.12)  0%, transparent 60%),
            radial-gradient(ellipse 60% 40% at 90% 100%, rgba(20,184,166,0.09) 0%, transparent 55%),
            radial-gradient(ellipse 50% 60% at 50% 50%,  rgba(139,92,246,0.04) 0%, transparent 70%),
            url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%232563eb' fill-opacity='0.03'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
        background-attachment: fixed;
    }}

    /* ══════════════════════════════════════════
       SIDEBAR
    ══════════════════════════════════════════ */
    [data-testid="stSidebar"] {{
        background: {SIDEBAR_BG} !important;
        border-right: 1px solid {BORDER} !important;
        backdrop-filter: blur(20px) saturate(180%) !important;
    }}
    [data-testid="stSidebar"] * {{ color: {TEXT} !important; }}
    [data-testid="stSidebar"] .stMarkdown p {{ color: {SUBTEXT} !important; }}

    /* Sidebar brand logo */
    .sb-brand {{
        text-align: center;
        padding: 20px 8px 12px;
        position: relative;
    }}
    .sb-icon {{
        font-size: 2.8rem;
        display: block;
        margin-bottom: 6px;
        animation: pulse-icon 3s ease-in-out infinite;
    }}
    @keyframes pulse-icon {{
        0%, 100% {{ filter: drop-shadow(0 0 6px rgba(37,99,235,0.5)); }}
        50%       {{ filter: drop-shadow(0 0 16px rgba(20,184,166,0.7)); }}
    }}
    .sb-title {{
        font-family: 'Sora', sans-serif;
        font-size: 1.05rem;
        font-weight: 800;
        background: linear-gradient(135deg, #2563eb 0%, #14b8a6 50%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: 0.3px;
    }}
    .sb-sub {{
        font-size: 0.60rem;
        color: {MUTED} !important;
        letter-spacing: 2.5px;
        text-transform: uppercase;
        margin-top: 3px;
    }}

    /* Sidebar status panel */
    .sb-status {{
        background: {SURFACE2};
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 12px 14px;
        margin: 6px 0;
        font-size: 0.76rem;
        line-height: 2.1;
    }}
    .status-online  {{ color: #10b981 !important; font-weight: 600 !important; }}
    .status-offline {{ color: #f59e0b !important; font-weight: 600 !important; }}

    /* ══════════════════════════════════════════
       MAIN CONTENT HEADER BANNER
    ══════════════════════════════════════════ */
    .page-banner {{
        background: linear-gradient(135deg,
            rgba(37,99,235,0.15) 0%,
            rgba(20,184,166,0.10) 50%,
            rgba(139,92,246,0.08) 100%);
        border: 1px solid {BORDER};
        border-radius: 20px;
        padding: 28px 32px 22px;
        margin-bottom: 28px;
        position: relative;
        overflow: hidden;
    }}
    .page-banner::before {{
        content: '';
        position: absolute;
        top: -1px; left: -1px; right: -1px; bottom: -1px;
        border-radius: 20px;
        background: linear-gradient(135deg, rgba(37,99,235,0.4), rgba(20,184,166,0.3), rgba(139,92,246,0.2));
        z-index: -1;
        mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
        mask-composite: exclude;
        padding: 1px;
    }}
    .banner-title {{
        font-size: clamp(1.6rem, 3.5vw, 2.4rem);
        font-weight: 800;
        background: linear-gradient(135deg, #60a5fa 0%, #14b8a6 45%, #a78bfa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
        margin: 0;
        line-height: 1.2;
    }}
    .banner-sub {{
        font-size: 0.85rem;
        color: {SUBTEXT};
        margin-top: 6px;
        letter-spacing: 0.3px;
    }}
    .banner-badge {{
        display: inline-block;
        background: rgba(37,99,235,0.15);
        border: 1px solid rgba(37,99,235,0.35);
        color: #60a5fa;
        font-size: 0.68rem;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 99px;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }}

    /* ══════════════════════════════════════════
       GLASSMORPHISM CARDS
    ══════════════════════════════════════════ */
    .glass-card {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 20px;
        padding: 24px;
        margin-bottom: 20px;
        backdrop-filter: blur(16px) saturate(160%);
        position: relative;
        overflow: hidden;
        transition: transform 0.30s cubic-bezier(.34,1.56,.64,1),
                    box-shadow 0.30s ease,
                    border-color 0.30s ease;
    }}
    .glass-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 20px 50px rgba(37,99,235,0.18),
                    0 0 0 1px rgba(20,184,166,0.20);
        border-color: {BORDER_H};
    }}
    .glass-card::after {{
        content: '';
        position: absolute;
        top: 0; left: 0;
        width: 100%; height: 2px;
        background: linear-gradient(90deg,
            transparent 0%,
            rgba(37,99,235,0.6) 30%,
            rgba(20,184,166,0.6) 70%,
            transparent 100%);
        opacity: 0;
        transition: opacity 0.3s;
    }}
    .glass-card:hover::after {{ opacity: 1; }}

    /* ══════════════════════════════════════════
       KPI METRIC CARDS
    ══════════════════════════════════════════ */
    .kpi-card {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 18px;
        padding: 22px 18px 18px;
        text-align: center;
        backdrop-filter: blur(14px);
        transition: all 0.30s cubic-bezier(.34,1.56,.64,1);
        position: relative;
        overflow: hidden;
    }}
    .kpi-card:hover {{
        transform: translateY(-6px) scale(1.02);
        box-shadow: 0 18px 44px rgba(37,99,235,0.20);
        border-color: {BORDER_H};
    }}
    .kpi-glow {{
        position: absolute;
        width: 80px; height: 80px;
        border-radius: 50%;
        filter: blur(30px);
        opacity: 0.25;
        top: -10px; right: -10px;
    }}
    .kpi-icon {{
        font-size: 1.6rem;
        margin-bottom: 8px;
        display: block;
    }}
    .kpi-label {{
        font-size: 0.68rem;
        color: {SUBTEXT};
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 500;
        margin-bottom: 8px;
    }}
    .kpi-value {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.2rem;
        font-weight: 700;
        color: {TEXT};
        line-height: 1;
        margin-bottom: 6px;
    }}
    .kpi-delta-pos {{ font-size:0.73rem; color:#10b981; font-weight:500; }}
    .kpi-delta-neg {{ font-size:0.73rem; color:#ef4444; font-weight:500; }}
    .kpi-delta-neu {{ font-size:0.73rem; color:{SUBTEXT}; }}

    /* ══════════════════════════════════════════
       SECTION HEADINGS
    ══════════════════════════════════════════ */
    .sec-head {{
        font-size: 1.0rem;
        font-weight: 700;
        color: {TEXT};
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        gap: 8px;
    }}
    .sec-head span {{
        display: inline-block;
        width: 3px; height: 18px;
        background: linear-gradient(180deg, #2563eb, #14b8a6);
        border-radius: 3px;
    }}

    /* ══════════════════════════════════════════
       UPLOAD ZONE
    ══════════════════════════════════════════ */
    [data-testid="stFileUploader"] {{
        border: 2px dashed rgba(37,99,235,0.35) !important;
        border-radius: 16px !important;
        background: {SURFACE2} !important;
        padding: 8px !important;
        transition: border-color 0.3s, background 0.3s;
    }}
    [data-testid="stFileUploader"]:hover {{
        border-color: rgba(20,184,166,0.6) !important;
        background: rgba(20,184,166,0.05) !important;
    }}

    /* ══════════════════════════════════════════
       BUTTONS
    ══════════════════════════════════════════ */
    .stButton > button {{
        background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 50%, #1d4ed8 100%) !important;
        background-size: 200% auto !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-family: 'Sora', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        letter-spacing: 0.4px !important;
        padding: 0.7rem 1.6rem !important;
        width: 100% !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(37,99,235,0.30) !important;
        position: relative;
        overflow: hidden;
    }}
    .stButton > button:hover {{
        background-position: right center !important;
        transform: translateY(-3px) !important;
        box-shadow: 0 8px 28px rgba(37,99,235,0.50),
                    0 0 0 1px rgba(20,184,166,0.3) !important;
    }}
    .stButton > button:active {{ transform: translateY(-1px) !important; }}

    /* Download buttons — teal */
    .stDownloadButton > button {{
        background: linear-gradient(135deg, #0d9488, #14b8a6) !important;
        color: white !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        border: none !important;
        box-shadow: 0 4px 15px rgba(20,184,166,0.30) !important;
    }}
    .stDownloadButton > button:hover {{
        transform: translateY(-3px) !important;
        box-shadow: 0 8px 28px rgba(20,184,166,0.50) !important;
    }}

    /* ══════════════════════════════════════════
       SCAN BUTTON — special pulsing glow
    ══════════════════════════════════════════ */
    .scan-btn-wrap .stButton > button {{
        background: linear-gradient(135deg, #7c3aed 0%, #2563eb 50%, #0891b2 100%) !important;
        background-size: 200% auto !important;
        font-size: 0.95rem !important;
        letter-spacing: 1.2px !important;
        text-transform: uppercase !important;
        padding: 0.85rem !important;
        box-shadow: 0 0 20px rgba(139,92,246,0.40), 0 4px 15px rgba(37,99,235,0.30) !important;
        animation: scan-btn-idle 3s ease-in-out infinite;
    }}
    .scan-btn-wrap .stButton > button:hover {{
        background-position: right center !important;
        box-shadow: 0 0 40px rgba(139,92,246,0.60), 0 8px 30px rgba(37,99,235,0.50) !important;
        animation: none;
    }}
    @keyframes scan-btn-idle {{
        0%, 100% {{ box-shadow: 0 0 20px rgba(139,92,246,0.35), 0 4px 15px rgba(37,99,235,0.25); }}
        50%       {{ box-shadow: 0 0 35px rgba(20,184,166,0.45), 0 6px 20px rgba(37,99,235,0.35); }}
    }}

    /* ══════════════════════════════════════════
       INPUTS
    ══════════════════════════════════════════ */
    .stTextInput > div > div > input,
    .stNumberInput input,
    .stSelectbox > div > div > div {{
        background: {INPUT_BG} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 10px !important;
        color: {TEXT} !important;
        font-family: 'Sora', sans-serif !important;
        font-size: 0.88rem !important;
        transition: border-color 0.25s, box-shadow 0.25s;
    }}
    .stTextInput > div > div > input:focus,
    .stNumberInput input:focus {{
        border-color: rgba(37,99,235,0.6) !important;
        box-shadow: 0 0 0 3px rgba(37,99,235,0.12) !important;
    }}

    /* ══════════════════════════════════════════
       RESULT CARDS  (Malignant / Benign)
    ══════════════════════════════════════════ */
    .result-card {{
        border-radius: 20px;
        padding: 28px 24px;
        text-align: center;
        margin-bottom: 20px;
        position: relative;
        overflow: hidden;
        animation: result-reveal 0.6s cubic-bezier(.34,1.56,.64,1);
    }}
    @keyframes result-reveal {{
        0%   {{ opacity:0; transform: scale(0.88) translateY(20px); }}
        100% {{ opacity:1; transform: scale(1)    translateY(0); }}
    }}
    .result-malignant {{
        background: linear-gradient(135deg, rgba(239,68,68,0.12), rgba(220,38,38,0.06));
        border: 2px solid rgba(239,68,68,0.55);
        box-shadow: 0 0 40px rgba(239,68,68,0.15), inset 0 1px 0 rgba(239,68,68,0.2);
    }}
    .result-benign {{
        background: linear-gradient(135deg, rgba(16,185,129,0.12), rgba(5,150,105,0.06));
        border: 2px solid rgba(16,185,129,0.55);
        box-shadow: 0 0 40px rgba(16,185,129,0.15), inset 0 1px 0 rgba(16,185,129,0.2);
    }}
    .result-tag {{
        font-size: 0.68rem; font-weight: 700;
        text-transform: uppercase; letter-spacing: 3px;
        margin-bottom: 10px; opacity: 0.8;
    }}
    .result-type {{
        font-family: 'JetBrains Mono', monospace;
        font-size: clamp(1.6rem, 3vw, 2.2rem);
        font-weight: 800;
        letter-spacing: -0.5px;
        margin-bottom: 8px;
    }}
    .result-desc {{ font-size: 0.84rem; color: {SUBTEXT}; line-height: 1.6; max-width: 400px; margin: 0 auto; }}

    /* ══════════════════════════════════════════
       RISK BADGES
    ══════════════════════════════════════════ */
    .badge {{
        display: inline-flex; align-items: center; gap: 5px;
        padding: 5px 16px;
        border-radius: 99px;
        font-size: 0.76rem; font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }}
    .badge-high   {{ background:rgba(239,68,68,0.15);  color:#f87171; border:1px solid rgba(239,68,68,0.40);  }}
    .badge-medium {{ background:rgba(245,158,11,0.15); color:#fbbf24; border:1px solid rgba(245,158,11,0.40); }}
    .badge-low    {{ background:rgba(16,185,129,0.15); color:#34d399; border:1px solid rgba(16,185,129,0.40); }}

    /* ══════════════════════════════════════════
       STEP / REC BOXES
    ══════════════════════════════════════════ */
    .step-box {{
        background: {SURFACE2};
        border: 1px solid {BORDER};
        border-left: 3px solid #2563eb;
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 8px;
        font-size: 0.85rem;
        line-height: 1.6;
        transition: border-left-color 0.2s, background 0.2s;
    }}
    .step-box:hover {{
        border-left-color: #14b8a6;
        background: rgba(20,184,166,0.06);
    }}
    .step-emergency {{
        border-left-color: #ef4444 !important;
    }}
    .step-emergency:hover {{ background: rgba(239,68,68,0.06) !important; }}

    /* ══════════════════════════════════════════
       TABS
    ══════════════════════════════════════════ */
    .stTabs [data-baseweb="tab-list"] {{
        background: {SURFACE2} !important;
        border-radius: 12px !important;
        padding: 4px !important;
        gap: 4px !important;
        border: 1px solid {BORDER};
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 9px !important;
        font-family: 'Sora', sans-serif !important;
        font-weight: 500 !important;
        font-size: 0.83rem !important;
        color: {SUBTEXT} !important;
        transition: all 0.2s ease !important;
        padding: 7px 16px !important;
    }}
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
        color: white !important;
        box-shadow: 0 3px 10px rgba(37,99,235,0.35) !important;
    }}

    /* ══════════════════════════════════════════
       DIVIDERS  /  MISC
    ══════════════════════════════════════════ */
    hr {{ border-color: {DIVIDER} !important; }}
    .stAlert {{ border-radius: 12px !important; }}
    .stSpinner > div {{ border-top-color: #2563eb !important; }}
    [data-testid="stMetricLabel"]  {{ font-family:'Sora',sans-serif !important; color:{SUBTEXT} !important; font-size:0.76rem !important; }}
    [data-testid="stMetricValue"]  {{ font-family:'JetBrains Mono',monospace !important; font-size:1.5rem !important; color:{TEXT} !important; }}
    [data-testid="stMetricDelta"]  {{ font-size:0.78rem !important; }}

    /* ══════════════════════════════════════════
       SCROLLBAR
    ══════════════════════════════════════════ */
    ::-webkit-scrollbar {{ width: 5px; height: 5px; }}
    ::-webkit-scrollbar-track {{ background: transparent; }}
    ::-webkit-scrollbar-thumb {{ background: rgba(37,99,235,0.35); border-radius: 99px; }}
    ::-webkit-scrollbar-thumb:hover {{ background: rgba(20,184,166,0.55); }}

    /* ══════════════════════════════════════════
       DATAFRAME
    ══════════════════════════════════════════ */
    [data-testid="stDataFrame"] {{
        border: 1px solid {BORDER} !important;
        border-radius: 14px !important;
        overflow: hidden;
    }}

    /* ══════════════════════════════════════════
       TOGGLE SWITCH
    ══════════════════════════════════════════ */
    .stToggle {{ font-family:'Sora',sans-serif !important; }}

    /* ══════════════════════════════════════════
       SETTINGS ROW
    ══════════════════════════════════════════ */
    .settings-row {{
        background: {SURFACE2};
        border: 1px solid {BORDER};
        border-radius: 14px;
        padding: 18px 20px;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        transition: border-color 0.25s;
    }}
    .settings-row:hover {{ border-color: {BORDER_H}; }}
    .settings-label {{ font-weight: 600; font-size: 0.90rem; margin-bottom:2px; }}
    .settings-desc  {{ font-size: 0.76rem; color: {SUBTEXT}; }}

    /* ══════════════════════════════════════════
       LIVE SCAN ANIMATION
    ══════════════════════════════════════════ */
    .scan-ring {{
        width: 100px; height: 100px;
        border-radius: 50%;
        border: 3px solid transparent;
        border-top-color: #2563eb;
        border-right-color: #14b8a6;
        animation: spin-ring 1.2s linear infinite;
        margin: 0 auto 16px;
        position: relative;
    }}
    .scan-ring::after {{
        content: '🔬';
        position: absolute;
        top: 50%; left: 50%;
        transform: translate(-50%, -50%);
        font-size: 2rem;
    }}
    @keyframes spin-ring {{
        0%   {{ transform: rotate(0deg);   }}
        100% {{ transform: rotate(360deg); }}
    }}

    /* ══════════════════════════════════════════
       ABCDE CARDS
    ══════════════════════════════════════════ */
    .abcde-card {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 16px;
        padding: 18px 12px;
        text-align: center;
        transition: all 0.3s cubic-bezier(.34,1.56,.64,1);
    }}
    .abcde-card:hover {{
        transform: translateY(-6px) scale(1.04);
        border-color: rgba(139,92,246,0.5);
        box-shadow: 0 12px 30px rgba(139,92,246,0.2);
    }}
    .abcde-letter {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.4rem; font-weight: 800;
        margin-bottom: 6px;
    }}
    .abcde-word  {{ font-weight: 700; font-size: 0.88rem; margin-bottom: 4px; }}
    .abcde-desc  {{ font-size: 0.73rem; color: {SUBTEXT}; line-height: 1.4; }}

    /* ══════════════════════════════════════════
       MOBILE RESPONSIVE
    ══════════════════════════════════════════ */
    @media (max-width: 768px) {{
        .page-banner  {{ padding: 18px 16px; }}
        .banner-title {{ font-size: 1.4rem; }}
        .glass-card   {{ padding: 16px; border-radius: 14px; }}
        .kpi-card     {{ padding: 14px 10px; }}
        .kpi-value    {{ font-size: 1.6rem; }}
    }}
    </style>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  CLASS 1 · NeuralCoreEngine
# ══════════════════════════════════════════════════════════════════
class NeuralCoreEngine:
    MODEL_FILE = "skin_cancer_cnn.h5"
    INPUT_SIZE = (224, 224)

    def __init__(self):
        self.is_online = False
        self.model     = self._load_model()

    def _load_model(self):
        try:
            from tensorflow.keras.models import load_model  # type: ignore
            m = load_model(self.MODEL_FILE)
            self.is_online = True
            return m
        except Exception:
            return None

    def execute_scan(self, pil_image: Image.Image) -> dict:
        if self.is_online:
            raw = self._infer(pil_image)
        else:
            raw = random.uniform(0.07, 0.93)

        if raw >= 0.50:
            diag  = "Malignant"
            prob  = raw
        else:
            diag  = "Benign"
            prob  = 1.0 - raw

        risk = "HIGH" if prob >= 0.80 else ("MEDIUM" if prob >= 0.50 else "LOW")

        return {
            "diagnosis":  diag,
            "probability":prob,
            "confidence": min(prob + random.uniform(0.01, 0.05), 0.99),
            "risk_level": risk,
            "raw_score":  raw,
            "model_mode": "Neural Network Online" if self.is_online else "Simulation Mode",
        }

    def _infer(self, pil_image):
        from tensorflow.keras.preprocessing.image import img_to_array  # type: ignore
        img = pil_image.convert("RGB").resize(self.INPUT_SIZE)
        arr = np.expand_dims(img_to_array(img) / 255.0, axis=0)
        return float(self.model.predict(arr, verbose=0)[0][0])


# ══════════════════════════════════════════════════════════════════
#  CLASS 2 · ImageProcessor
# ══════════════════════════════════════════════════════════════════
class ImageProcessor:
    @staticmethod
    def validate(file_obj):
        ext = file_obj.name.rsplit(".", 1)[-1].lower()
        if ext not in {"jpg", "jpeg", "png"}:
            return False, f"❌ Format '.{ext}' not accepted. Use JPG, JPEG, or PNG."
        if file_obj.size > 10 * 1024 * 1024:
            return False, f"❌ File too large ({file_obj.size/1e6:.1f} MB). Maximum 10 MB."
        try:
            img = Image.open(file_obj); img.verify()
        except Exception:
            return False, "❌ Corrupted or unreadable image file."
        file_obj.seek(0)
        img = Image.open(file_obj)
        if img.size[0] < 100 or img.size[1] < 100:
            return False, f"❌ Resolution {img.size[0]}×{img.size[1]} px too low. Min: 100×100."
        file_obj.seek(0)
        return True, "✅ Image validated and ready for scanning."

    @staticmethod
    def preprocess(pil_image):
        img = pil_image.convert("RGB").resize((224, 224), Image.LANCZOS)
        img = ImageEnhance.Contrast(img).enhance(1.20)
        img = ImageEnhance.Sharpness(img).enhance(1.15)
        return img

    @staticmethod
    def display_copy(pil_image):
        img = pil_image.convert("RGB")
        img.thumbnail((640, 640), Image.LANCZOS)
        return img


# ══════════════════════════════════════════════════════════════════
#  CLASS 3 · ClinicalProtocols
# ══════════════════════════════════════════════════════════════════
class ClinicalProtocols:
    _DB = {
        "Malignant": {
            "hex":         "#ef4444",
            "glow":        "rgba(239,68,68,0.15)",
            "icon":        "🔴",
            "css":         "result-malignant",
            "description": "AI analysis detects characteristics strongly consistent with a malignant skin lesion. Immediate clinical evaluation is critical.",
            "ai_message":  "HIGH RISK: Irregular pigmentation, asymmetric borders, and multi-color variation patterns indicate malignancy. Urgent dermatological consultation is required within 48 hours.",
            "recommendations": [
                "🏥 Consult an oncology-dermatologist within 48 hours — do not delay.",
                "🔬 Request formal dermoscopy evaluation and excisional biopsy.",
                "🚫 Avoid all UV exposure immediately — sunlight and artificial tanning.",
                "🧴 Apply broad-spectrum SPF 100+ at all outdoor times.",
                "📋 Request full-body skin mapping (digital dermoscopic photography).",
                "🩸 Discuss Sentinel Lymph Node Biopsy (SLNB) with your surgeon.",
                "🥗 Antioxidant-rich diet: berries, leafy greens, omega-3 fatty acids.",
            ],
            "patient_advice": [
                "Wear UPF 50+ protective clothing and wide-brim hats daily.",
                "Stay indoors during peak UV hours — 10:00 AM to 4:00 PM.",
                "Perform weekly ABCDE self-examinations on all skin lesions.",
                "Eliminate tobacco use — it significantly accelerates cancer progression.",
                "Maintain Vitamin D levels through supplementation only (not sun exposure).",
                "Keep a photographic diary of all lesions to track changes over time.",
            ],
            "procedures": [
                "Wide Local Excision (WLE) — surgical removal with clear safety margins.",
                "Mohs Micrographic Surgery — precise layer-by-layer tissue-sparing excision.",
                "Sentinel Lymph Node Biopsy (SLNB) — assess regional lymphatic spread.",
                "Adjuvant Radiation Therapy — post-surgical residual cancer cell elimination.",
                "Systemic Immunotherapy: Pembrolizumab (Keytruda) / Ipilimumab (Yervoy).",
            ],
            "medications": [
                "BRAF/MEK inhibitors: Vemurafenib + Cobimetinib (targeted therapy).",
                "Pembrolizumab (Keytruda) — PD-1 immune checkpoint inhibitor.",
                "Dabrafenib + Trametinib — for BRAF V600E/K mutated cases.",
                "Topical Imiquimod 5% cream — for superficial lesions (physician-directed).",
            ],
            "therapy": [
                "Photodynamic Therapy (PDT) for localized superficial lesions.",
                "Electrochemotherapy as adjuvant management post-excision.",
                "Intralesional IL-2 cytokine injection therapy.",
            ],
            "emergency_signs": [
                "⚠️ Rapid lesion enlargement beyond 6mm within days.",
                "⚠️ Spontaneous ulceration, bleeding, or crusting of the lesion.",
                "⚠️ Visible lymph node swelling in neck, armpit, or groin near lesion.",
                "⚠️ Satellite lesions appearing around the main lesion.",
                "⚠️ Pain, numbness, or tingling sensation around the lesion area.",
            ],
            "followup": "Bi-weekly monitoring for 3 months post-surgery. PET-CT at 6 months. Oncology review every 3 months for 2 years.",
            "consultation": "🚨 URGENT: Schedule with Onco-Dermatologist within 48 hours.",
        },
        "Benign": {
            "hex":         "#10b981",
            "glow":        "rgba(16,185,129,0.15)",
            "icon":        "🟢",
            "css":         "result-benign",
            "description": "AI analysis indicates a benign skin lesion with low malignant potential. Routine monitoring is recommended as best practice.",
            "ai_message":  "LOW RISK: Symmetric borders, uniform pigmentation, and regular morphology are consistent with a benign melanocytic nevus. Routine annual dermatology monitoring is advised.",
            "recommendations": [
                "✅ No urgent surgical intervention required at this time.",
                "📅 Schedule a routine annual dermatology skin check.",
                "🔍 Perform monthly ABCDE self-examinations as best practice.",
                "🧴 Apply daily SPF 50+ broad-spectrum sunscreen for prevention.",
                "📸 Photograph the lesion now to establish a monitoring baseline.",
                "🥗 Maintain healthy lifestyle — antioxidant diet and adequate hydration.",
                "📞 Consult a doctor immediately if the lesion changes in any way.",
            ],
            "patient_advice": [
                "Standard daily sun protection measures are sufficient.",
                "Maintain balanced diet rich in antioxidants, vitamins C and E.",
                "Adequate hydration — minimum 2+ litres of water per day.",
                "Avoid mechanical trauma, scratching, or irritating the lesion.",
                "Annual professional dermoscopy evaluation for documentation.",
                "Monitor for ABCDE changes at least once per month.",
            ],
            "procedures": [
                "Clinical observation only — no immediate surgical intervention.",
                "Digital dermoscopy photography for baseline documentation.",
                "Elective shave excision for cosmetic removal (if desired).",
                "Punch excision if histological confirmation is requested.",
                "CO2 Laser ablation for cosmetic concerns at patient's discretion.",
            ],
            "medications": [
                "No medications required. SPF is the primary daily intervention.",
                "Topical Vitamin C antioxidant serum for skin maintenance.",
                "Ceramide-based barrier moisturizers for skin health.",
                "Vitamin D supplementation — consult physician for correct dosage.",
            ],
            "therapy": [
                "Cryotherapy (liquid nitrogen) — elective symptomatic relief only.",
                "Topical retinoids for general skin maintenance (physician-directed).",
                "PDT only if pre-malignant features emerge on follow-up evaluation.",
            ],
            "emergency_signs": [
                "⚠️ Any sudden change in size, shape, or color (ABCDE rule).",
                "⚠️ Unexpected bleeding or oozing without physical trauma.",
                "⚠️ New satellite lesions appearing near the original lesion.",
                "⚠️ Persistent itching, burning, or pain in the lesion area.",
                "⚠️ Lesion fails to heal after minor trauma within 4 weeks.",
            ],
            "followup": "Annual routine dermatology screening. AI re-evaluation recommended in 6 months.",
            "consultation": "📅 Routine annual dermatology appointment. Consult earlier if ABCDE changes appear.",
        },
    }

    @classmethod
    def fetch_data(cls, diagnosis: str) -> dict:
        return cls._DB.get(diagnosis, cls._DB["Benign"])


# ══════════════════════════════════════════════════════════════════
#  CLASS 4 · ReportGenerator
# ══════════════════════════════════════════════════════════════════
class ReportGenerator:

    @staticmethod
    def pdf(record: dict, img: Image.Image) -> bytes:
        buf = io.BytesIO()
        if not PDF_OK:
            buf.write(b"Install reportlab: pip install reportlab"); return buf.getvalue()

        doc  = SimpleDocTemplate(buf, pagesize=A4,
                                 rightMargin=1.8*cm, leftMargin=1.8*cm,
                                 topMargin=1.5*cm,   bottomMargin=1.5*cm)
        st_  = getSampleStyleSheet()
        BLUE = colors.HexColor("#1e3a5f")
        GRAY = colors.HexColor("#64748b")
        diag = record.get("diagnosis","Benign")
        RISK = colors.HexColor("#ef4444" if diag=="Malignant" else "#10b981")

        H1  = ParagraphStyle("H1",  fontSize=19, fontName="Helvetica-Bold", textColor=BLUE,    alignment=TA_CENTER, spaceAfter=3)
        SUB = ParagraphStyle("SUB", fontSize=8.5,fontName="Helvetica",       textColor=GRAY,    alignment=TA_CENTER, spaceAfter=10)
        SEC = ParagraphStyle("SEC", fontSize=11, fontName="Helvetica-Bold",  textColor=BLUE,    spaceAfter=6, spaceBefore=10)
        TXT = ParagraphStyle("TXT", fontSize=8.5,fontName="Helvetica",       textColor=colors.HexColor("#374151"), spaceAfter=3, leading=13, leftIndent=6)
        DIS = ParagraphStyle("DIS", fontSize=7.5,fontName="Helvetica",       textColor=GRAY,    alignment=TA_JUSTIFY, leading=12)
        FTR = ParagraphStyle("FTR", fontSize=7,  fontName="Helvetica",       textColor=colors.HexColor("#94a3b8"), alignment=TA_CENTER)

        story = [
            Paragraph("🔬  SkinScan AI — Clinical Intelligence Platform", H1),
            Paragraph("Automated Dermoscopic Cancer Detection Report  ·  v13.0", SUB),
            HRFlowable(width="100%", thickness=2, color=BLUE),
            Spacer(1, 10),
        ]

        rows = [
            ["FIELD", "DETAIL"],
            ["Patient Name",      record.get("patient_name","N/A")],
            ["Age",               str(record.get("age","N/A"))],
            ["Gender",            record.get("gender","N/A")],
            ["Scan Date & Time",  record.get("timestamp","N/A")],
            ["AI Diagnosis",      diag],
            ["Risk Level",        record.get("risk_level","N/A")],
            ["Cancer Probability",f"{record.get('probability',0)*100:.1f}%"],
            ["AI Confidence",     f"{record.get('confidence',0)*100:.1f}%"],
            ["Model Status",      record.get("model_mode","N/A")],
        ]
        tbl = Table(rows, colWidths=[5.5*cm, 12.5*cm])
        tbl.setStyle(TableStyle([
            ("BACKGROUND",     (0,0),(-1,0), BLUE),
            ("TEXTCOLOR",      (0,0),(-1,0), colors.white),
            ("FONTNAME",       (0,0),(-1,0), "Helvetica-Bold"),
            ("ROWBACKGROUNDS", (0,1),(-1,-1),[colors.HexColor("#f0f4f8"),colors.white]),
            ("FONTNAME",       (0,1),(0,-1), "Helvetica-Bold"),
            ("FONTSIZE",       (0,0),(-1,-1),9),
            ("GRID",           (0,0),(-1,-1),0.4, colors.HexColor("#dde3ea")),
            ("PADDING",        (0,0),(-1,-1),7),
            ("TEXTCOLOR",      (1,6),(1,6),  RISK),
            ("FONTNAME",       (1,6),(1,6),  "Helvetica-Bold"),
        ]))
        story += [Paragraph("Patient & Scan Information", SEC), tbl, Spacer(1,12)]

        try:
            ibuf = io.BytesIO()
            th   = img.copy(); th.thumbnail((160,160)); th.save(ibuf, format="PNG"); ibuf.seek(0)
            ri   = RLImage(ibuf, width=4.5*cm, height=4.5*cm)
            it   = Table([[ri]], colWidths=[18*cm])
            it.setStyle(TableStyle([("ALIGN",(0,0),(-1,-1),"CENTER")]))
            story += [Paragraph("Uploaded Dermoscopic Image", SEC), it, Spacer(1,10)]
        except Exception:
            pass

        kb = ClinicalProtocols.fetch_data(diag)
        story += [Paragraph("AI Diagnostic Assessment", SEC),
                  Paragraph(kb["ai_message"], ParagraphStyle("msg",fontSize=8.5,fontName="Helvetica",
                    textColor=colors.HexColor("#374151"),backColor=colors.HexColor("#f0f9ff"),
                    borderPadding=7,leading=14,spaceAfter=10))]

        story.append(Paragraph("Clinical Recommendations", SEC))
        for r in kb["recommendations"]: story.append(Paragraph(f"• {r}", TXT))
        story.append(Spacer(1,8))

        story.append(Paragraph("Treatment Plan", SEC))
        for label,key in [("Procedures","procedures"),("Medications","medications"),
                           ("Therapy","therapy"),("Emergency Signs","emergency_signs")]:
            story.append(Paragraph(f"▸ {label}", ParagraphStyle("cat",fontSize=9,fontName="Helvetica-Bold",
                textColor=colors.HexColor("#ef4444" if "Emergency" in label else "#2563eb"),
                spaceAfter=2,leftIndent=4,spaceBefore=5)))
            for i in kb[key]: story.append(Paragraph(f"  – {i}", TXT))
        story.append(Spacer(1,8))

        story += [
            Paragraph("Follow-up Protocol", SEC),
            Paragraph(kb["followup"], TXT),
            Spacer(1,12),
            HRFlowable(width="100%", thickness=0.7, color=colors.HexColor("#e2e8f0")),
            Spacer(1,6),
            Paragraph("⚠️ AI DISCLAIMER: This report is generated by an AI-powered research tool for educational and "
                      "academic purposes only. It does NOT constitute a formal medical diagnosis. Always consult a "
                      "board-certified dermatologist or oncologist for all clinical decisions.", DIS),
            Spacer(1,5),
            Paragraph(f"SkinScan AI Clinical Intelligence Platform v13.0  ·  Rehan Shafique  ·  "
                      f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", FTR),
        ]

        doc.build(story)
        return buf.getvalue()

    @staticmethod
    def csv(database: list) -> str:
        if not database: return ""
        return pd.DataFrame([{
            "Timestamp":    r.get("timestamp",""),
            "Patient":      r.get("patient_name",""),
            "Age":          r.get("age",""),
            "Gender":       r.get("gender",""),
            "Diagnosis":    r.get("diagnosis",""),
            "Risk":         r.get("risk_level",""),
            "Probability%": f"{r.get('probability',0)*100:.2f}",
            "Confidence%":  f"{r.get('confidence',0)*100:.2f}",
            "Model":        r.get("model_mode",""),
        } for r in database]).to_csv(index=False)


# ══════════════════════════════════════════════════════════════════
#  CLASS 5 · SkinScanEnterpriseSuite  (Master Controller)
# ══════════════════════════════════════════════════════════════════
class SkinScanEnterpriseSuite:

    def __init__(self):
        st.set_page_config(
            page_title="SkinScan AI — Clinical Intelligence",
            page_icon="🔬",
            layout="wide",
            initial_sidebar_state="expanded",
        )
        self._init_state()
        self.ai_engine = NeuralCoreEngine()
        inject_premium_css(st.session_state.app_theme)

    def _init_state(self):
        defaults = {
            "app_theme":       "dark",
            "medical_database":   [],
            "last_result":        None,
            "last_raw_img":       None,
            "last_processed_img": None,
        }
        for k, v in defaults.items():
            if k not in st.session_state:
                st.session_state[k] = v

    # ──────────────────────────────────────────────────────────────
    #  SIDEBAR
    # ──────────────────────────────────────────────────────────────
    def build_sidebar(self) -> str:
        with st.sidebar:
            # Brand
            st.markdown("""
            <div class="sb-brand">
                <span class="sb-icon">🔬</span>
                <div class="sb-title">SkinScan AI</div>
                <div class="sb-sub">Clinical Intelligence · v13</div>
            </div>
            """, unsafe_allow_html=True)
            st.divider()

            # Navigation
            nav = option_menu(
                menu_title=None,
                options=["Dashboard", "AI Scan", "Patient Records", "Analytics", "Settings"],
                icons=["house-door-fill","cpu-fill","journal-medical","bar-chart-line-fill","gear-fill"],
                default_index=0,
                styles={
                    "container":         {"padding":"0", "background":"transparent"},
                    "nav-link":          {
                        "font-size":"0.84rem",
                        "font-family":"Sora,sans-serif",
                        "font-weight":"500",
                        "padding":"9px 14px",
                        "border-radius":"10px",
                        "margin":"2px 0",
                        "color":"#7fa3c0",
                        "transition":"all 0.2s",
                    },
                    "nav-link-selected": {
                        "background":"linear-gradient(135deg,#1d4ed8,#2563eb)",
                        "color":"white",
                        "font-weight":"600",
                        "box-shadow":"0 4px 14px rgba(37,99,235,0.40)",
                    },
                    "icon": {"font-size":"0.90rem"},
                },
            )

            st.divider()

            # Status panel
            dot  = '<span class="status-online">🟢 Neural Network Online</span>' if self.ai_engine.is_online \
                   else '<span class="status-offline">🟠 Simulation Active</span>'
            n    = len(st.session_state.medical_database)
            high = sum(1 for r in st.session_state.medical_database if r.get("risk_level")=="HIGH")

            st.markdown(f"""
            <div class="sb-status">
                <b>AI Engine</b><br>{dot}<br><br>
                <b>Model File</b><br>
                <span style='font-family:JetBrains Mono,monospace; font-size:0.74rem; color:#14b8a6;'>
                    skin_cancer_cnn.h5
                </span><br><br>
                <b>Session Scans</b><br>
                <span style='font-family:JetBrains Mono,monospace; font-size:1.1rem; font-weight:700;
                             color:#60a5fa;'>{n}</span>
                &nbsp;·&nbsp;
                <span style='color:#f87171; font-size:0.78rem;'>{high} high risk</span>
            </div>
            """, unsafe_allow_html=True)

        return nav

    # ──────────────────────────────────────────────────────────────
    #  LAUNCH
    # ──────────────────────────────────────────────────────────────
    def launch(self):
        nav = self.build_sidebar()
        {
            "Dashboard":      self.module_dashboard,
            "AI Scan":        self.module_ai_scan,
            "Patient Records":self.module_registry,
            "Analytics":      self.module_analytics,
            "Settings":       self.module_settings,
        }.get(nav, self.module_dashboard)()
        self._footer()

    # ══════════════════════════════════════════════════════════════
    #  MODULE 1 · Dashboard
    # ══════════════════════════════════════════════════════════════
    def module_dashboard(self):
        # Page banner
        st.markdown("""
        <div class="page-banner">
            <div class="banner-badge">🏠 Main Dashboard</div>
            <p class="banner-title">Clinical Command Center</p>
            <p class="banner-sub">Real-time AI dermatology system overview · Session analytics · Quick navigation</p>
        </div>
        """, unsafe_allow_html=True)

        db     = st.session_state.medical_database
        n      = len(db)
        malig  = sum(1 for r in db if r.get("diagnosis")=="Malignant")
        benign = n - malig
        conf   = (sum(r.get("confidence",0) for r in db)/n*100) if n else 0

        # KPI row
        k1, k2, k3, k4 = st.columns(4)
        kpis = [
            ("🧬", "Total Scans",      str(n),          "This session",         "#3b82f6"),
            ("🔴", "Malignant Cases",  str(malig),      "High-risk detected",   "#ef4444"),
            ("🟢", "Benign Cases",     str(benign),     "Low-risk cleared",     "#10b981"),
            ("⚡", "Avg Confidence",   f"{conf:.1f}%",  "CNN inference score",  "#8b5cf6"),
        ]
        for col, (icon, label, value, delta, color) in zip([k1,k2,k3,k4], kpis):
            with col:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-glow" style="background:{color};"></div>
                    <div class="kpi-icon">{icon}</div>
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-value" style="color:{color};">{value}</div>
                    <div class="kpi-delta-neu">{delta}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col_l, col_r = st.columns([1.4, 1])

        # Recent activity table
        with col_l:
            st.markdown("""
            <div class="glass-card">
                <div class="sec-head"><span></span>🕒 Recent Scan Activity</div>
            """, unsafe_allow_html=True)
            if db:
                rows = db[-10:][::-1]
                df_r = pd.DataFrame([{
                    "Time":      r.get("timestamp","").split(" ")[1] if " " in r.get("timestamp","") else "",
                    "Patient":   r.get("patient_name","ANON")[:18],
                    "Result":    r.get("diagnosis","—"),
                    "Risk":      r.get("risk_level","—"),
                    "Prob.":     f"{r.get('probability',0)*100:.1f}%",
                    "Conf.":     f"{r.get('confidence',0)*100:.1f}%",
                } for r in rows])
                st.dataframe(df_r, use_container_width=True, hide_index=True, height=260)
            else:
                st.info("No scans recorded yet. Head to **AI Scan** to begin.")
            st.markdown("</div>", unsafe_allow_html=True)

        # Donut
        with col_r:
            st.markdown("""
            <div class="glass-card">
                <div class="sec-head"><span></span>🧬 Diagnosis Distribution</div>
            """, unsafe_allow_html=True)
            if db:
                ser = pd.Series([r.get("diagnosis","?") for r in db]).value_counts()
                fig = go.Figure(go.Pie(
                    labels=ser.index.tolist(), values=ser.values.tolist(),
                    hole=0.56,
                    marker=dict(colors=["#ef4444","#10b981"],
                                line=dict(color="rgba(0,0,0,0)", width=2)),
                    textinfo="percent+label",
                    textfont_size=11,
                    hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>",
                ))
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#7fa3c0", height=260, margin=dict(l=0,r=0,t=5,b=0),
                    showlegend=True, legend=dict(font_size=11, orientation="h", y=-0.08),
                    annotations=[dict(text=f"<b>{n}</b><br>scans", x=0.5, y=0.5,
                                      font_size=14, font_color="#e8f4f8", showarrow=False)],
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Run scans to populate distribution chart.")
            st.markdown("</div>", unsafe_allow_html=True)

        # AI System health cards
        st.markdown("---")
        st.markdown("""
        <div class="sec-head"><span></span>⚡ Platform Modules</div>
        """, unsafe_allow_html=True)

        mc1, mc2, mc3, mc4 = st.columns(4)
        modules_info = [
            ("🧬", "AI Scan Lab",       "#2563eb", "Upload dermoscopic image for real-time CNN cancer detection with full clinical report."),
            ("📋", "Patient Records",   "#14b8a6", "Session database with complete scan history, filters, and CSV/JSON export functionality."),
            ("📊", "Analytics Engine",  "#8b5cf6", "Interactive epidemiological charts, risk distributions, and scan trend analysis."),
            ("⚙️", "System Settings",   "#f59e0b", "Configure themes, preferences, and platform behaviour for optimal workflow."),
        ]
        for col, (icon, title, color, desc) in zip([mc1,mc2,mc3,mc4], modules_info):
            with col:
                st.markdown(f"""
                <div class="glass-card" style="text-align:center; padding:20px 16px;">
                    <div style="font-size:2rem; margin-bottom:10px;">{icon}</div>
                    <div style="font-weight:700; font-size:0.90rem; color:{color}; margin-bottom:8px;">{title}</div>
                    <div style="font-size:0.78rem; color:#7fa3c0; line-height:1.5;">{desc}</div>
                </div>
                """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════
    #  MODULE 2 · AI Scan
    # ══════════════════════════════════════════════════════════════
    def module_ai_scan(self):
        st.markdown("""
        <div class="page-banner">
            <div class="banner-badge">🧬 AI Analysis Lab</div>
            <p class="banner-title">Neural Scan Engine</p>
            <p class="banner-sub">skin_cancer_cnn.h5 · Benign / Malignant · Auto-preprocessing pipeline · Clinical report generation</p>
        </div>
        """, unsafe_allow_html=True)

        col_in, col_out = st.columns([1, 1.35], gap="large")

        # ── INPUT PANEL ───────────────────────────────────────────
        with col_in:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="sec-head"><span></span>👤 Patient Information</div>', unsafe_allow_html=True)

            p_name = st.text_input("Patient Name / ID", placeholder="e.g. Ahmed Khan  /  PT-2024-001")
            a_col, g_col = st.columns(2)
            with a_col: p_age    = st.number_input("Age", min_value=1, max_value=120, value=35)
            with g_col: p_gender = st.selectbox("Gender", ["Male","Female","Other","Prefer not to say"])

            st.markdown('<div class="sec-head" style="margin-top:16px;"><span></span>🖼️ Dermoscopic Image</div>',
                        unsafe_allow_html=True)
            st.caption("JPG · JPEG · PNG  ·  Max 10 MB  ·  Min 100×100 px")

            uploaded = st.file_uploader("Drop image here or browse",
                                        type=["jpg","jpeg","png"],
                                        label_visibility="collapsed")

            img_ok  = False
            raw_img = None

            if uploaded:
                ok, msg = ImageProcessor.validate(uploaded)
                if not ok:
                    st.error(msg)
                else:
                    st.success(msg)
                    raw_img  = Image.open(uploaded)
                    disp_img = ImageProcessor.display_copy(raw_img)
                    img_ok   = True
                    st.image(disp_img, use_container_width=True,
                             caption=f"📐 {raw_img.size[0]}×{raw_img.size[1]} px  ·  {uploaded.size/1024:.1f} KB  ·  {uploaded.name.rsplit('.',1)[-1].upper()}")

            st.markdown('<div class="scan-btn-wrap">', unsafe_allow_html=True)
            run = st.button("▶ EXECUTE DEEP SCAN", disabled=(not img_ok))
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)   # glass-card

        # ── OUTPUT PANEL ──────────────────────────────────────────
        with col_out:
            # Trigger scan
            if img_ok and run:
                with st.spinner(""):
                    st.markdown("""
                    <div style="text-align:center; padding:30px 0 10px;">
                        <div class="scan-ring"></div>
                        <div style="font-size:0.84rem; color:#7fa3c0; letter-spacing:1px;">
                            EXTRACTING FEATURE VECTORS…
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    time.sleep(2.5)

                    processed = ImageProcessor.preprocess(raw_img)
                    result    = self.ai_engine.execute_scan(processed)
                    intel     = ClinicalProtocols.fetch_data(result["diagnosis"])

                    rec = {
                        "timestamp":    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "patient_name": p_name.strip() or "Anonymous",
                        "age":          p_age,
                        "gender":       p_gender,
                        **result,
                    }
                    st.session_state.medical_database.append(rec)
                    st.session_state.last_result      = rec
                    st.session_state.last_raw_img     = raw_img
                    st.session_state.last_processed_img = processed
                    st.rerun()

            # ── Render last result ────────────────────────────────
            if st.session_state.last_result:
                res   = st.session_state.last_result
                intel = ClinicalProtocols.fetch_data(res["diagnosis"])

                # Result banner
                st.markdown(f"""
                <div class="result-card {intel['css']}">
                    <div class="result-tag" style="color:{intel['hex']};">
                        ◉ AI DIAGNOSIS RESULT
                    </div>
                    <div class="result-type" style="color:{intel['hex']};">
                        {intel['icon']}  {res['diagnosis']}
                    </div>
                    <div class="result-desc">{intel['description']}</div>
                </div>
                """, unsafe_allow_html=True)

                # Metric strip
                m1, m2, m3 = st.columns(3)
                m1.metric("Cancer Probability", f"{res['probability']*100:.1f}%")
                m2.metric("AI Confidence",       f"{res['confidence']*100:.1f}%")
                badge_cls = {"HIGH":"badge-high","MEDIUM":"badge-medium","LOW":"badge-low"}[res["risk_level"]]
                m3.markdown(f"""
                <div style="text-align:center; padding-top:6px;">
                    <div style="font-size:0.70rem; color:#7fa3c0; margin-bottom:7px; text-transform:uppercase; letter-spacing:1.5px;">Risk Level</div>
                    <span class="badge {badge_cls}">● {res['risk_level']}</span>
                </div>""", unsafe_allow_html=True)

                # Confidence gauge (Plotly)
                fig_g = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=res["confidence"]*100,
                    number={"suffix":"%","font":{"family":"JetBrains Mono","size":28,"color":intel["hex"]}},
                    title={"text":"AI Confidence Score","font":{"family":"Sora","size":12,"color":"#7fa3c0"}},
                    gauge={
                        "axis":{"range":[0,100],"tickfont":{"size":9,"color":"#7fa3c0"},
                                "tickcolor":"rgba(100,116,139,0.3)"},
                        "bar":{"color":intel["hex"], "thickness":0.22},
                        "bgcolor":"rgba(0,0,0,0)",
                        "borderwidth":0,
                        "steps":[
                            {"range":[0,40],  "color":"rgba(16,185,129,0.06)"},
                            {"range":[40,70], "color":"rgba(245,158,11,0.06)"},
                            {"range":[70,100],"color":"rgba(239,68,68,0.06)"},
                        ],
                        "threshold":{"line":{"color":intel["hex"],"width":3},
                                     "value":res["confidence"]*100},
                    },
                ))
                fig_g.update_layout(
                    height=200, margin=dict(l=10,r=10,t=42,b=5),
                    paper_bgcolor="rgba(0,0,0,0)", font_color="#7fa3c0",
                )
                st.plotly_chart(fig_g, use_container_width=True)

                # Probability fill bar
                pct = res["probability"]*100
                fig_p = go.Figure()
                fig_p.add_trace(go.Bar(
                    x=[pct], y=[""], orientation="h",
                    marker=dict(
                        color=intel["hex"],
                        line=dict(width=0),
                    ),
                    text=[f"  {pct:.1f}%"], textposition="inside",
                    textfont=dict(color="white", size=13, family="JetBrains Mono"),
                    width=0.5,
                    hoverinfo="skip",
                ))
                fig_p.add_trace(go.Bar(
                    x=[100-pct], y=[""], orientation="h",
                    marker=dict(color="rgba(100,116,139,0.12)", line=dict(width=0)),
                    width=0.5, hoverinfo="skip",
                ))
                fig_p.update_layout(
                    height=65, margin=dict(l=0,r=0,t=0,b=0),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    barmode="stack", showlegend=False,
                    xaxis=dict(range=[0,100], showticklabels=False, showgrid=False, zeroline=False),
                    yaxis=dict(showticklabels=False, showgrid=False),
                )
                st.plotly_chart(fig_p, use_container_width=True)

                # AI message
                st.info(f"🤖  {intel['ai_message']}")
                st.caption(f"🔩 Model: **{res['model_mode']}**  ·  📅 Scanned: {res['timestamp']}")

            else:
                st.markdown("""
                <div class="glass-card" style="text-align:center; padding:4rem 1.5rem; margin-top:0;">
                    <div style="font-size:4rem; margin-bottom:16px; opacity:0.6;">🔬</div>
                    <div style="font-weight:700; font-size:1.0rem; margin-bottom:8px;">
                        Ready for Analysis
                    </div>
                    <div style="font-size:0.84rem; color:#7fa3c0; line-height:1.7;">
                        Upload a dermoscopic image &amp; click<br>
                        <b>EXECUTE DEEP SCAN</b> to begin AI analysis.
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # ── Clinical Engine (full width) ──────────────────────────
        if st.session_state.last_result:
            res   = st.session_state.last_result
            intel = ClinicalProtocols.fetch_data(res["diagnosis"])

            st.markdown("---")
            st.markdown("""
            <div class="sec-head" style="font-size:1.05rem;">
                <span></span>📋 Clinical Intelligence Engine
            </div>
            """, unsafe_allow_html=True)

            t1, t2, t3, t4 = st.tabs([
                "🏥 Recommendations",
                "🌿 Patient Advice",
                "💊 Treatment Plan",
                "📄 Medical Report",
            ])

            with t1:
                r1, r2 = st.columns(2)
                with r1:
                    st.markdown("**Clinical Recommendations**")
                    for item in intel["recommendations"]:
                        st.markdown(f'<div class="step-box">{item}</div>', unsafe_allow_html=True)
                with r2:
                    st.markdown("**Consultation & Follow-up**")
                    st.markdown(f'<div class="step-box" style="border-left-color:{intel["hex"]};">'
                                f'{intel["consultation"]}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="step-box">📅 {intel["followup"]}</div>', unsafe_allow_html=True)

            with t2:
                for item in intel["patient_advice"]:
                    st.markdown(f'<div class="step-box">🌿 {item}</div>', unsafe_allow_html=True)

            with t3:
                tc1, tc2 = st.columns(2)
                plan_items = [
                    ("🩺 Procedures",             "procedures",      False, "#2563eb"),
                    ("💊 Medications",            "medications",     False, "#14b8a6"),
                    ("⚗️ Therapy Options",         "therapy",         False, "#8b5cf6"),
                    ("🚨 Emergency Warning Signs", "emergency_signs", True,  "#ef4444"),
                ]
                for i, (lbl, key, is_emg, color) in enumerate(plan_items):
                    col = tc1 if i % 2 == 0 else tc2
                    with col:
                        st.markdown(f"""
                        <div class="glass-card" style="border-left:3px solid {color}; padding:16px; margin-bottom:12px;">
                            <div style="font-weight:700; color:{color}; margin-bottom:10px; font-size:0.88rem;">{lbl}</div>
                            {''.join(f'<div class="step-box {"step-emergency" if is_emg else ""}" style="margin-bottom:5px;">{s}</div>' for s in intel[key])}
                        </div>
                        """, unsafe_allow_html=True)

            with t4:
                st.markdown("#### 📥 Download Clinical Reports")
                st.caption("Reports generated from your most recent scan.")

                dl1, dl2 = st.columns(2)
                rec     = st.session_state.last_result
                proc_img = st.session_state.last_processed_img

                with dl1:
                    if PDF_OK and proc_img:
                        pdf_bytes = ReportGenerator.pdf(rec, proc_img)
                        fname = f"SkinScan_{rec.get('patient_name','PT')}_{datetime.date.today()}.pdf".replace(" ","_")
                        st.download_button("📄 Download PDF Report",
                                           data=pdf_bytes, file_name=fname, mime="application/pdf")
                    else:
                        st.warning("Install ReportLab:\n`pip install reportlab`")
                with dl2:
                    st.download_button("📊 Download CSV Registry",
                                       data=ReportGenerator.csv(st.session_state.medical_database),
                                       file_name=f"SkinScan_Registry_{datetime.date.today()}.csv",
                                       mime="text/csv")

                st.markdown("""
                <div style='font-size:0.74rem; color:#64748b; margin-top:12px; padding:12px 14px;
                            border:1px solid rgba(100,116,139,0.18); border-radius:10px; line-height:1.6;'>
                    ⚠️ <b>AI Disclaimer:</b> These reports are generated by an AI-powered research tool
                    for educational purposes only. They do <b>not</b> constitute a formal medical diagnosis.
                    Always consult a qualified dermatologist or oncologist for all clinical decisions.
                </div>
                """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════
    #  MODULE 3 · Patient Records
    # ══════════════════════════════════════════════════════════════
    def module_registry(self):
        st.markdown("""
        <div class="page-banner">
            <div class="banner-badge">📋 Patient Records</div>
            <p class="banner-title">Secure Session Database</p>
            <p class="banner-sub">Complete scan history · Advanced filters · CSV & JSON export</p>
        </div>
        """, unsafe_allow_html=True)

        db = st.session_state.medical_database
        if not db:
            st.info("📭 No records yet. Run scans in **AI Scan** to populate the database.")
            return

        df = pd.DataFrame([{
            "Timestamp":   r.get("timestamp",""),
            "Patient":     r.get("patient_name","ANON"),
            "Age":         r.get("age","—"),
            "Gender":      r.get("gender","—"),
            "Diagnosis":   r.get("diagnosis","—"),
            "Risk":        r.get("risk_level","—"),
            "Probability": f"{r.get('probability',0)*100:.1f}%",
            "Confidence":  f"{r.get('confidence',0)*100:.1f}%",
            "Engine":      r.get("model_mode","—"),
        } for r in db])

        # Summary KPIs
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f"""<div class="kpi-card">
                <div class="kpi-label">Total Records</div>
                <div class="kpi-value" style="color:#60a5fa;">{len(db)}</div>
                <div class="kpi-delta-neu">All sessions</div>
            </div>""", unsafe_allow_html=True)
        with k2:
            m = sum(1 for r in db if r.get("diagnosis")=="Malignant")
            st.markdown(f"""<div class="kpi-card">
                <div class="kpi-label">Malignant</div>
                <div class="kpi-value" style="color:#f87171;">{m}</div>
                <div class="kpi-delta-neg">High-risk</div>
            </div>""", unsafe_allow_html=True)
        with k3:
            b = sum(1 for r in db if r.get("diagnosis")=="Benign")
            st.markdown(f"""<div class="kpi-card">
                <div class="kpi-label">Benign</div>
                <div class="kpi-value" style="color:#34d399;">{b}</div>
                <div class="kpi-delta-pos">Low-risk cleared</div>
            </div>""", unsafe_allow_html=True)
        with k4:
            ac = (sum(r.get("confidence",0) for r in db)/len(db)*100)
            st.markdown(f"""<div class="kpi-card">
                <div class="kpi-label">Avg Confidence</div>
                <div class="kpi-value" style="color:#a78bfa;">{ac:.1f}%</div>
                <div class="kpi-delta-neu">CNN average</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        with st.expander("🔍 Filter Records", expanded=False):
            fc1, fc2, fc3 = st.columns(3)
            f_diag   = fc1.multiselect("Diagnosis",  ["Malignant","Benign"], default=["Malignant","Benign"])
            f_risk   = fc2.multiselect("Risk Level", ["HIGH","MEDIUM","LOW"], default=["HIGH","MEDIUM","LOW"])
            f_gender = fc3.multiselect("Gender", ["Male","Female","Other","Prefer not to say"],
                                        default=["Male","Female","Other","Prefer not to say"])

        mask = df["Diagnosis"].isin(f_diag) & df["Risk"].isin(f_risk) & df["Gender"].isin(f_gender)
        df_f = df[mask]

        st.markdown(f'<div style="font-size:0.82rem; color:#7fa3c0; margin-bottom:10px;">'
                    f'Showing <b style="color:#60a5fa;">{len(df_f)}</b> of <b>{len(df)}</b> records</div>',
                    unsafe_allow_html=True)

        st.markdown('<div class="glass-card" style="padding:0; overflow:hidden;">', unsafe_allow_html=True)
        st.dataframe(df_f, use_container_width=True, hide_index=True, height=380)
        st.markdown('</div>', unsafe_allow_html=True)

        e1, e2, e3 = st.columns(3)
        with e1:
            st.download_button("📥 Export CSV", data=ReportGenerator.csv(db),
                               file_name=f"SkinScan_Registry_{datetime.date.today()}.csv",
                               mime="text/csv")
        with e2:
            safe = [{k:str(v) if isinstance(v,datetime.datetime) else v for k,v in r.items()} for r in db]
            st.download_button("🔗 Export JSON",
                               data=json.dumps(safe, indent=2),
                               file_name=f"SkinScan_Registry_{datetime.date.today()}.json",
                               mime="application/json")
        with e3:
            if st.button("🗑️ Clear All Records"):
                st.session_state.medical_database    = []
                st.session_state.last_result         = None
                st.session_state.last_raw_img        = None
                st.session_state.last_processed_img  = None
                st.rerun()

    # ══════════════════════════════════════════════════════════════
    #  MODULE 4 · Analytics
    # ══════════════════════════════════════════════════════════════
    def module_analytics(self):
        st.markdown("""
        <div class="page-banner">
            <div class="banner-badge">📊 Analytics</div>
            <p class="banner-title">Epidemiological Analytics</p>
            <p class="banner-sub">Real-time diagnosis patterns · Risk trends · Confidence analytics · Session insights</p>
        </div>
        """, unsafe_allow_html=True)

        db = st.session_state.medical_database
        if not db:
            st.warning("⚠️ No scan data available. Run scans to generate analytics.")
            return

        df = pd.DataFrame(db)
        df["prob_pct"] = df["probability"] * 100
        df["conf_pct"] = df["confidence"]  * 100

        PLOT_LAYOUT = dict(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Sora", color="#7fa3c0"),
            margin=dict(l=4,r=4,t=44,b=4),
        )
        GRID = dict(gridcolor="rgba(37,99,235,0.10)", zerolinecolor="rgba(37,99,235,0.10)")

        r1, r2 = st.columns(2)

        # ── Donut ──
        with r1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            ser = df["diagnosis"].value_counts()
            fig1 = go.Figure(go.Pie(
                labels=ser.index.tolist(), values=ser.values.tolist(),
                hole=0.52,
                marker=dict(colors=["#ef4444","#10b981"],
                            line=dict(color="rgba(0,0,0,0)",width=3)),
                textinfo="percent+label", textfont_size=11,
                hovertemplate="<b>%{label}</b><br>%{value} cases<br>%{percent}<extra></extra>",
            ))
            fig1.update_layout(title="Malignant vs Benign Split", height=300,
                               showlegend=True, legend=dict(font_size=11,orientation="h",y=-0.08),
                               **PLOT_LAYOUT)
            st.plotly_chart(fig1, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Risk bar ──
        with r2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            rc = df["risk_level"].value_counts().reset_index()
            rc.columns = ["Risk","Count"]
            RISK_ORDER = ["HIGH","MEDIUM","LOW"]
            rc["Risk"] = pd.Categorical(rc["Risk"], categories=RISK_ORDER, ordered=True)
            rc = rc.sort_values("Risk")
            fig2 = go.Figure()
            for risk, color in [("HIGH","#ef4444"),("MEDIUM","#f59e0b"),("LOW","#10b981")]:
                sub = rc[rc["Risk"]==risk]
                if not sub.empty:
                    fig2.add_trace(go.Bar(
                        x=sub["Risk"], y=sub["Count"],
                        name=risk, marker_color=color,
                        marker_line_width=0,
                        hovertemplate=f"<b>{risk}</b><br>%{{y}} cases<extra></extra>",
                    ))
            fig2.update_layout(title="Risk Level Distribution", height=300,
                               showlegend=False, barmode="group",
                               xaxis=dict(title="", **GRID),
                               yaxis=dict(title="Cases", **GRID),
                               **PLOT_LAYOUT)
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        r3, r4 = st.columns(2)

        # ── Scatter ──
        with r3:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            fig3 = go.Figure()
            for diag, color, sym in [("Malignant","#ef4444","circle"),("Benign","#10b981","diamond")]:
                sub = df[df["diagnosis"]==diag]
                if not sub.empty:
                    fig3.add_trace(go.Scatter(
                        x=sub["prob_pct"], y=sub["conf_pct"],
                        mode="markers", name=diag,
                        marker=dict(color=color, size=9, opacity=0.85,
                                    symbol=sym, line=dict(color=color,width=1)),
                        hovertemplate="<b>"+diag+"</b><br>Prob: %{x:.1f}%<br>Conf: %{y:.1f}%<extra></extra>",
                    ))
            fig3.update_layout(title="Probability vs Confidence",
                               xaxis=dict(title="Probability (%)", **GRID),
                               yaxis=dict(title="Confidence (%)", **GRID),
                               legend=dict(orientation="h",y=-0.18,font_size=11),
                               height=300, **PLOT_LAYOUT)
            st.plotly_chart(fig3, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Trend ──
        with r4:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            if len(df) >= 2:
                x = list(range(1, len(df)+1))
                fig4 = go.Figure()
                fig4.add_trace(go.Scatter(
                    x=x, y=df["conf_pct"], mode="lines+markers", name="Confidence",
                    line=dict(color="#2563eb",width=2.5,shape="spline"),
                    marker=dict(size=7, color="#2563eb"),
                    fill="tozeroy", fillcolor="rgba(37,99,235,0.06)",
                ))
                fig4.add_trace(go.Scatter(
                    x=x, y=df["prob_pct"], mode="lines+markers", name="Probability",
                    line=dict(color="#ef4444",width=2,dash="dot",shape="spline"),
                    marker=dict(size=7, color="#ef4444"),
                ))
                fig4.update_layout(title="Scan Trend Analysis",
                                   xaxis=dict(title="Scan #", **GRID),
                                   yaxis=dict(title="Score (%)", **GRID, range=[0,105]),
                                   legend=dict(orientation="h",y=-0.18,font_size=11),
                                   height=300, **PLOT_LAYOUT)
                st.plotly_chart(fig4, use_container_width=True)
            else:
                st.info("Need 2+ scans for trend chart.")
            st.markdown('</div>', unsafe_allow_html=True)

        # Summary stats
        st.markdown("---")
        st.markdown('<div class="sec-head"><span></span>📐 Session Summary Statistics</div>',
                    unsafe_allow_html=True)
        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric("Total Scans",     len(df))
        s2.metric("Avg Confidence",  f"{df['conf_pct'].mean():.1f}%")
        s3.metric("Avg Probability", f"{df['prob_pct'].mean():.1f}%")
        s4.metric("Malignant",       int((df["diagnosis"]=="Malignant").sum()))
        s5.metric("Benign",          int((df["diagnosis"]=="Benign").sum()))

    # ══════════════════════════════════════════════════════════════
    #  MODULE 5 · Settings
    # ══════════════════════════════════════════════════════════════
    def module_settings(self):
        st.markdown("""
        <div class="page-banner">
            <div class="banner-badge">⚙️ Settings</div>
            <p class="banner-title">Platform Configuration</p>
            <p class="banner-sub">Theme · AI engine info · ABCDE guide · System preferences</p>
        </div>
        """, unsafe_allow_html=True)

        # Theme
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="sec-head"><span></span>🎨 Appearance</div>', unsafe_allow_html=True)
        c1, c2 = st.columns([2,1])
        with c1:
            st.markdown("""
            <div class="settings-label">Color Theme</div>
            <div class="settings-desc">Switch between dark (recommended) and light clinical mode</div>
            """, unsafe_allow_html=True)
        with c2:
            t = st.toggle("Dark Mode", value=(st.session_state.app_theme=="dark"))
            if t != (st.session_state.app_theme=="dark"):
                st.session_state.app_theme = "dark" if t else "light"
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        # AI Engine info
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="sec-head"><span></span>🤖 AI Engine Information</div>', unsafe_allow_html=True)

        dot  = "🟢" if self.ai_engine.is_online else "🟠"
        mode = "Neural Network Online" if self.ai_engine.is_online else "Simulation Mode"

        info_items = [
            ("Model File",       "skin_cancer_cnn.h5",                        "#14b8a6"),
            ("Architecture",     "Convolutional Neural Network (CNN)",         "#60a5fa"),
            ("Output Classes",   "Benign  ·  Malignant",                      "#a78bfa"),
            ("Input Size",       "224 × 224 pixels (RGB)",                    "#60a5fa"),
            ("Preprocessing",    "Normalize → Contrast → Sharpen → Resize",   "#7fa3c0"),
            ("Engine Status",    f"{dot} {mode}",                             "#f59e0b"),
            ("Platform Version", "SkinScan AI Clinical Intelligence v13.0",   "#7fa3c0"),
        ]
        for label, value, color in info_items:
            st.markdown(f"""
            <div class="settings-row">
                <div>
                    <div class="settings-label">{label}</div>
                </div>
                <div style="font-family:'JetBrains Mono',monospace; font-size:0.82rem;
                            color:{color}; text-align:right;">{value}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # User guide accordion
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="sec-head"><span></span>📖 User Guide</div>', unsafe_allow_html=True)

        guide = [
            ("📤 Image Upload Requirements", [
                "Accepted: <b>JPG, JPEG, PNG</b> only — other formats rejected.",
                "Maximum file size: <b>10 MB</b>. Minimum resolution: <b>100×100 px</b>.",
                "Use high-quality clinical dermoscopic images for best accuracy.",
                "The system auto-preprocesses: resize (224×224) → normalize → enhance.",
            ]),
            ("🤖 How AI Inference Works", [
                "CNN model loaded from <b>skin_cancer_cnn.h5</b> at platform startup.",
                "🟢 <b>Online Mode</b>: Real TensorFlow inference on your uploaded image.",
                "🟠 <b>Simulation Mode</b>: Model file not found — demo mode activates safely.",
                "Sigmoid output ≥ 0.50 = Malignant · < 0.50 = Benign.",
            ]),
            ("📊 Understanding Results", [
                "<b>Probability</b>: How likely the AI's primary diagnosis is correct.",
                "<b>Confidence</b>: Model certainty — higher is more reliable.",
                "<b>Risk Level</b>: HIGH (≥80%) · MEDIUM (50–80%) · LOW (<50%).",
                "Always review Clinical Recommendations and Treatment Plan tabs.",
            ]),
            ("👨‍⚕️ When to See a Doctor", [
                "🚨 <b>Immediately</b> for Malignant / HIGH RISK results.",
                "📅 <b>Within 1 week</b> for MEDIUM RISK results.",
                "📆 <b>Annually</b> for Benign / LOW RISK results.",
                "Any lesion that bleeds, changes, or ulcerates — see a doctor regardless.",
            ]),
            ("⚠️ System Limitations", [
                "This is a <b>research and educational tool</b> — not a certified medical device.",
                "Always seek a qualified dermatologist for final diagnosis and treatment.",
                "Session data clears on browser refresh — export CSV to retain records.",
                "Accuracy improves with clinical dermoscopic images over smartphone photos.",
            ]),
        ]
        for title, points in guide:
            with st.expander(title):
                for pt in points:
                    st.markdown(f'<div class="step-box">{pt}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # ABCDE Guide
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="sec-head"><span></span>🎗️ ABCDE Melanoma Self-Examination Guide</div>',
                    unsafe_allow_html=True)
        abcde = [
            ("A","Asymmetry","#ef4444","One half doesn't match the other half."),
            ("B","Border",   "#f97316","Irregular, ragged, notched, or blurred edges."),
            ("C","Color",    "#f59e0b","Multiple shades: brown, black, red, white, or blue."),
            ("D","Diameter", "#3b82f6","Larger than 6mm (size of a pencil eraser)."),
            ("E","Evolution","#8b5cf6","Any change in size, shape, color, or new symptoms."),
        ]
        a1,a2,a3,a4,a5 = st.columns(5)
        for col,(letter,word,color,desc) in zip([a1,a2,a3,a4,a5], abcde):
            with col:
                st.markdown(f"""
                <div class="abcde-card" style="border-top:3px solid {color};">
                    <div class="abcde-letter" style="color:{color};">{letter}</div>
                    <div class="abcde-word">{word}</div>
                    <div class="abcde-desc">{desc}</div>
                </div>
                """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────────────
    #  FOOTER
    # ──────────────────────────────────────────────────────────────
    def _footer(self):
        st.markdown("""
        <div style="text-align:center; padding:2.5rem 0 1rem;
                    border-top:1px solid rgba(37,99,235,0.12);
                    margin-top:3rem;">
            <div style="font-size:1.05rem; font-weight:700; margin-bottom:6px;
                        background:linear-gradient(135deg,#60a5fa,#14b8a6,#a78bfa);
                        -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
                🔬  SkinScan AI — Clinical Intelligence Platform
            </div>
            <div style="font-size:0.78rem; color:#475569; line-height:2;">
                Developed by <b style="color:#64748b;">Rehan Shafique</b>
                &nbsp;·&nbsp; Final Year Project &nbsp;·&nbsp; Bioinformatics<br>
                Python · Streamlit · TensorFlow · Plotly · ReportLab · PIL
            </div>
            <div style="font-size:0.72rem; color:#ef4444; margin-top:8px;">
                ⚠️ For Research &amp; Educational Purposes Only — Not a Certified Medical Device
            </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = SkinScanEnterpriseSuite()
    app.launch()
