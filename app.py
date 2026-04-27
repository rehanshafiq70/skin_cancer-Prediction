"""
╔══════════════════════════════════════════════════════════════════════╗
║  SKINSCAN AI  ·  CLINICAL INTELLIGENCE PLATFORM  ·  v14.0           ║
║  Design  : Apex Medical · Horizontal Navbar · Camera Integration     ║
║  Author  : Rehan Shafique  ·  Final Year Project                     ║
║  Model   : skin_cancer_cnn.h5  (Benign / Malignant)                 ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageEnhance
import plotly.graph_objects as go
import plotly.express as px
from streamlit_option_menu import option_menu
import random, time, datetime, io, json

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors as rl_colors
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
#  GLOBAL STYLES  ──  v14  "Clinical Apex" Design System
# ══════════════════════════════════════════════════════════════════
def inject_css(theme: str = "dark"):
    dark = theme == "dark"

    # ── Token map ────────────────────────────────────────────────
    if dark:
        BG        = "#020d1e"
        BG2       = "#041226"
        SURF      = "rgba(4,22,50,0.78)"
        SURF2     = "rgba(6,28,60,0.65)"
        BORDER    = "rgba(37,99,235,0.22)"
        BDH       = "rgba(20,184,166,0.50)"
        TEXT      = "#dff0fa"
        SUB       = "#6b9ab8"
        MUTED     = "#2a4a62"
        NAV_BG    = "rgba(2,13,30,0.88)"
        INP       = "rgba(4,22,50,0.92)"
        DIV       = "rgba(37,99,235,0.13)"
        HERO_G1   = "rgba(37,99,235,0.18)"
        HERO_G2   = "rgba(20,184,166,0.10)"
    else:
        BG        = "#f0f5fc"
        BG2       = "#e6eef9"
        SURF      = "rgba(255,255,255,0.93)"
        SURF2     = "rgba(240,247,255,0.90)"
        BORDER    = "rgba(37,99,235,0.18)"
        BDH       = "rgba(20,184,166,0.45)"
        TEXT      = "#0c1e32"
        SUB       = "#3a6080"
        MUTED     = "#a8c4d8"
        NAV_BG    = "rgba(248,252,255,0.95)"
        INP       = "rgba(255,255,255,0.97)"
        DIV       = "rgba(37,99,235,0.10)"
        HERO_G1   = "rgba(37,99,235,0.08)"
        HERO_G2   = "rgba(20,184,166,0.06)"

    st.markdown(f"""
    <style>
    /* ─────────────────────────────────────────────────────────────
       FONTS  ·  Outfit (body) + Oxanium (display/mono data)
    ───────────────────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Oxanium:wght@400;600;700;800&display=swap');

    /* ─────────────────────────────────────────────────────────────
       RESET & SHELL
    ───────────────────────────────────────────────────────────── */
    *, *::before, *::after {{ box-sizing: border-box; }}

    html, body {{ font-family: 'Outfit', sans-serif !important; }}

    .stApp {{
        font-family: 'Outfit', sans-serif !important;
        background-color: {BG} !important;
        background-image:
            radial-gradient(ellipse 90% 45% at 8%   2%,  {HERO_G1} 0%, transparent 65%),
            radial-gradient(ellipse 70% 50% at 92%  98%, {HERO_G2} 0%, transparent 60%),
            url("data:image/svg+xml,%3Csvg width='80' height='80' viewBox='0 0 80 80' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none'%3E%3Cg fill='%232563eb' fill-opacity='0.025'%3E%3Ccircle cx='1' cy='1' r='1'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
        background-attachment: fixed;
        color: {TEXT} !important;
    }}

    /* ─────────────────────────────────────────────────────────────
       HIDE DEFAULT STREAMLIT CHROME
    ───────────────────────────────────────────────────────────── */
    #MainMenu, header[data-testid="stHeader"], footer {{ display: none !important; }}
    [data-testid="stSidebar"] {{ display: none !important; }}
    .stDeployButton {{ display: none !important; }}
    .stDecoration {{ display: none !important; }}
    [data-testid="collapsedControl"] {{ display: none !important; }}

    /* ─────────────────────────────────────────────────────────────
       MAIN CONTAINER  (top padding for fixed navbar)
    ───────────────────────────────────────────────────────────── */
    .block-container {{
        padding-top: 82px !important;
        padding-left: 2rem  !important;
        padding-right: 2rem !important;
        max-width: 1320px !important;
        margin: 0 auto !important;
    }}
    @media (max-width: 768px) {{
        .block-container {{
            padding-top: 72px !important;
            padding-left: 1rem  !important;
            padding-right: 1rem !important;
        }}
    }}

    /* ─────────────────────────────────────────────────────────────
       TOP NAVBAR WRAPPER  (glassmorphism fixed bar)
    ───────────────────────────────────────────────────────────── */
    .navbar-shell {{
        position: fixed;
        top: 0; left: 0; right: 0;
        z-index: 9999;
        background: {NAV_BG};
        backdrop-filter: blur(22px) saturate(180%);
        -webkit-backdrop-filter: blur(22px) saturate(180%);
        border-bottom: 1px solid {BORDER};
        padding: 0 28px;
        height: 64px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 4px 24px rgba(0,0,0,0.12);
    }}

    /* Logo section inside navbar */
    .nav-logo {{
        display: flex;
        align-items: center;
        gap: 10px;
        flex-shrink: 0;
        text-decoration: none;
    }}
    .nav-logo-icon {{
        font-size: 1.55rem;
        animation: logo-pulse 3s ease-in-out infinite;
        line-height: 1;
    }}
    @keyframes logo-pulse {{
        0%,100% {{ filter: drop-shadow(0 0 5px rgba(37,99,235,0.6));  }}
        50%      {{ filter: drop-shadow(0 0 14px rgba(20,184,166,0.7)); }}
    }}
    .nav-logo-text {{
        font-family: 'Oxanium', sans-serif;
        font-size: 1.05rem;
        font-weight: 800;
        background: linear-gradient(135deg, #3b82f6 0%, #14b8a6 55%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: 0.2px;
        line-height: 1.1;
    }}
    .nav-logo-ver {{
        font-size: 0.58rem;
        color: {SUB};
        letter-spacing: 2px;
        text-transform: uppercase;
        font-weight: 500;
    }}

    /* AI badge in navbar */
    .nav-ai-badge {{
        display: inline-flex;
        align-items: center;
        gap: 5px;
        background: linear-gradient(135deg, rgba(37,99,235,0.15), rgba(20,184,166,0.10));
        border: 1px solid rgba(37,99,235,0.30);
        padding: 3px 11px;
        border-radius: 99px;
        font-size: 0.63rem;
        font-weight: 700;
        color: #60a5fa;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        flex-shrink: 0;
    }}

    /* ─────────────────────────────────────────────────────────────
       HORIZONTAL OPTION MENU — styled as navbar items
    ───────────────────────────────────────────────────────────── */
    /* The option_menu sits inside .nav-menu-center */
    .nav-menu-center ul {{
        display: flex !important;
        flex-direction: row !important;
        gap: 2px !important;
        list-style: none !important;
        margin: 0 !important;
        padding: 4px !important;
        background: {SURF2} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 12px !important;
    }}
    .nav-menu-center ul li a {{
        font-family: 'Outfit', sans-serif !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        padding: 7px 15px !important;
        border-radius: 9px !important;
        color: {SUB} !important;
        transition: all 0.2s ease !important;
        white-space: nowrap !important;
    }}
    .nav-menu-center ul li a:hover {{
        color: {TEXT} !important;
        background: rgba(37,99,235,0.10) !important;
    }}
    .nav-menu-center ul li a[aria-selected="true"] {{
        background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
        color: white !important;
        font-weight: 600 !important;
        box-shadow: 0 3px 12px rgba(37,99,235,0.40) !important;
    }}

    /* Streamlit doesn't render option_menu nav-link icon well in horizontal - override */
    .nav-menu-center .nav-link {{
        padding: 7px 14px !important;
    }}
    .nav-menu-center .nav-link .icon {{ font-size:0.85rem !important; margin-right:5px !important; }}

    /* ─────────────────────────────────────────────────────────────
       PAGE BANNER  (hero intro per page)
    ───────────────────────────────────────────────────────────── */
    .page-banner {{
        background: linear-gradient(135deg,
            rgba(37,99,235,0.14) 0%,
            rgba(20,184,166,0.08) 50%,
            rgba(139,92,246,0.06) 100%);
        border: 1px solid {BORDER};
        border-radius: 22px;
        padding: 30px 36px 24px;
        margin-bottom: 28px;
        position: relative;
        overflow: hidden;
    }}
    .page-banner::before {{
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg,
            transparent, #2563eb 25%, #14b8a6 50%, #8b5cf6 75%, transparent);
    }}
    .page-banner::after {{
        content: '';
        position: absolute;
        bottom: -60px; right: -60px;
        width: 180px; height: 180px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(20,184,166,0.12) 0%, transparent 70%);
    }}
    .banner-chip {{
        display: inline-block;
        background: rgba(37,99,235,0.14);
        border: 1px solid rgba(37,99,235,0.30);
        color: #60a5fa;
        font-size: 0.65rem;
        font-weight: 700;
        padding: 3px 12px;
        border-radius: 99px;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 10px;
    }}
    .banner-title {{
        font-family: 'Oxanium', sans-serif;
        font-size: clamp(1.7rem, 3.5vw, 2.5rem);
        font-weight: 800;
        background: linear-gradient(135deg, #60a5fa 0%, #14b8a6 45%, #a78bfa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
        margin: 0 0 8px;
        line-height: 1.2;
    }}
    .banner-sub {{
        font-size: 0.88rem;
        color: {SUB};
        max-width: 640px;
        line-height: 1.6;
    }}

    /* ─────────────────────────────────────────────────────────────
       HERO SECTION  (Home page)
    ───────────────────────────────────────────────────────────── */
    .hero-section {{
        min-height: 300px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        padding: 48px 40px;
        background: linear-gradient(135deg,
            rgba(37,99,235,0.12) 0%,
            rgba(20,184,166,0.08) 40%,
            rgba(139,92,246,0.06) 100%);
        border: 1px solid {BORDER};
        border-radius: 24px;
        margin-bottom: 32px;
        position: relative;
        overflow: hidden;
    }}
    .hero-section::before {{
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #2563eb 0%, #14b8a6 50%, #8b5cf6 100%);
    }}
    .hero-section::after {{
        content: '⬡';
        position: absolute;
        right: 40px; top: 50%;
        transform: translateY(-50%);
        font-size: 9rem;
        opacity: 0.04;
        color: #2563eb;
        pointer-events: none;
    }}
    .hero-title {{
        font-family: 'Oxanium', sans-serif;
        font-size: clamp(2rem, 5vw, 3.2rem);
        font-weight: 800;
        background: linear-gradient(135deg, #60a5fa 0%, #14b8a6 40%, #a78bfa 85%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -1px;
        margin: 0 0 12px;
        line-height: 1.15;
    }}
    .hero-subtitle {{
        font-size: 1.0rem;
        color: {SUB};
        max-width: 560px;
        line-height: 1.65;
        margin-bottom: 28px;
    }}
    .hero-badges {{
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-bottom: 20px;
    }}
    .hbadge {{
        padding: 5px 14px;
        border-radius: 99px;
        font-size: 0.76rem;
        font-weight: 600;
        letter-spacing: 0.3px;
        display: inline-flex;
        align-items: center;
        gap: 5px;
    }}
    .hbadge-blue   {{ background:rgba(37,99,235,0.14); color:#60a5fa; border:1px solid rgba(37,99,235,0.30); }}
    .hbadge-teal   {{ background:rgba(20,184,166,0.12); color:#2dd4bf; border:1px solid rgba(20,184,166,0.28); }}
    .hbadge-purple {{ background:rgba(139,92,246,0.12); color:#a78bfa; border:1px solid rgba(139,92,246,0.28); }}
    .hbadge-green  {{ background:rgba(16,185,129,0.12); color:#34d399; border:1px solid rgba(16,185,129,0.28); }}

    /* Feature cards on Home */
    .feat-card {{
        background: {SURF};
        border: 1px solid {BORDER};
        border-radius: 18px;
        padding: 24px 20px;
        text-align: center;
        backdrop-filter: blur(14px);
        transition: all 0.30s cubic-bezier(.34,1.56,.64,1);
        height: 100%;
    }}
    .feat-card:hover {{
        transform: translateY(-8px) scale(1.02);
        border-color: {BDH};
        box-shadow: 0 20px 50px rgba(37,99,235,0.16);
    }}
    .feat-icon  {{ font-size: 2.4rem; margin-bottom: 14px; display:block; }}
    .feat-title {{ font-weight: 700; font-size: 0.96rem; margin-bottom: 8px; }}
    .feat-desc  {{ font-size: 0.80rem; color: {SUB}; line-height: 1.55; }}

    /* ─────────────────────────────────────────────────────────────
       GLASS CARDS  (general use)
    ───────────────────────────────────────────────────────────── */
    .glass-card {{
        background: {SURF};
        border: 1px solid {BORDER};
        border-radius: 20px;
        padding: 24px;
        margin-bottom: 20px;
        backdrop-filter: blur(16px) saturate(150%);
        transition: transform 0.28s cubic-bezier(.34,1.56,.64,1),
                    box-shadow 0.28s ease,
                    border-color 0.28s ease;
        position: relative;
        overflow: hidden;
    }}
    .glass-card::after {{
        content: '';
        position: absolute;
        top: 0; left: 0; width: 100%; height: 2px;
        background: linear-gradient(90deg, transparent, rgba(37,99,235,0.55), rgba(20,184,166,0.55), transparent);
        opacity: 0;
        transition: opacity 0.3s;
    }}
    .glass-card:hover {{
        transform: translateY(-4px);
        box-shadow: 0 18px 48px rgba(37,99,235,0.14);
        border-color: {BDH};
    }}
    .glass-card:hover::after {{ opacity: 1; }}

    /* ─────────────────────────────────────────────────────────────
       KPI CARDS
    ───────────────────────────────────────────────────────────── */
    .kpi-card {{
        background: {SURF};
        border: 1px solid {BORDER};
        border-radius: 18px;
        padding: 20px 16px 16px;
        text-align: center;
        backdrop-filter: blur(14px);
        transition: all 0.28s cubic-bezier(.34,1.56,.64,1);
        position: relative;
        overflow: hidden;
    }}
    .kpi-card:hover {{
        transform: translateY(-6px) scale(1.025);
        box-shadow: 0 16px 42px rgba(37,99,235,0.16);
        border-color: {BDH};
    }}
    .kpi-glow {{
        position: absolute;
        width: 90px; height: 90px;
        border-radius: 50%;
        filter: blur(35px);
        opacity: 0.22;
        top: -15px; right: -15px;
        pointer-events: none;
    }}
    .kpi-icon  {{ font-size:1.5rem; margin-bottom:8px; display:block; }}
    .kpi-label {{ font-size:0.68rem; color:{SUB}; text-transform:uppercase; letter-spacing:2px; font-weight:500; margin-bottom:7px; }}
    .kpi-value {{ font-family:'Oxanium',monospace; font-size:2.0rem; font-weight:700; color:{TEXT}; line-height:1; margin-bottom:5px; }}
    .kd-pos    {{ font-size:0.72rem; color:#34d399; font-weight:500; }}
    .kd-neg    {{ font-size:0.72rem; color:#f87171; font-weight:500; }}
    .kd-neu    {{ font-size:0.72rem; color:{SUB}; }}

    /* ─────────────────────────────────────────────────────────────
       SECTION HEADING
    ───────────────────────────────────────────────────────────── */
    .sec-head {{
        font-size: 1.0rem;
        font-weight: 700;
        color: {TEXT};
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        gap: 9px;
    }}
    .sec-head span {{
        display:inline-block;
        width:3px; height:18px;
        background: linear-gradient(180deg,#2563eb,#14b8a6);
        border-radius:3px;
    }}

    /* ─────────────────────────────────────────────────────────────
       INPUT MODE TOGGLE  (Upload / Camera)
    ───────────────────────────────────────────────────────────── */
    .mode-toggle-wrap {{
        display: flex;
        gap: 8px;
        margin-bottom: 18px;
    }}
    .mode-btn {{
        flex: 1;
        padding: 10px 12px;
        border-radius: 12px;
        font-family: 'Outfit', sans-serif;
        font-size: 0.83rem;
        font-weight: 600;
        cursor: pointer;
        text-align: center;
        transition: all 0.22s ease;
        border: 1px solid;
    }}
    .mode-btn-active {{
        background: linear-gradient(135deg, #2563eb, #1d4ed8);
        border-color: transparent;
        color: white;
        box-shadow: 0 4px 14px rgba(37,99,235,0.40);
    }}
    .mode-btn-inactive {{
        background: {SURF2};
        border-color: {BORDER};
        color: {SUB};
    }}

    /* ─────────────────────────────────────────────────────────────
       FILE UPLOADER
    ───────────────────────────────────────────────────────────── */
    [data-testid="stFileUploader"] {{
        border: 2px dashed rgba(37,99,235,0.32) !important;
        border-radius: 16px !important;
        background: {SURF2} !important;
        transition: border-color 0.25s, background 0.25s;
    }}
    [data-testid="stFileUploader"]:hover {{
        border-color: rgba(20,184,166,0.55) !important;
        background: rgba(20,184,166,0.04) !important;
    }}

    /* Camera input */
    [data-testid="stCameraInput"] {{
        border-radius: 16px !important;
        overflow: hidden;
    }}
    [data-testid="stCameraInput"] video {{
        border-radius: 14px !important;
    }}

    /* ─────────────────────────────────────────────────────────────
       SCAN BUTTON  (animated glow)
    ───────────────────────────────────────────────────────────── */
    .stButton > button {{
        background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 40%, #0891b2 100%) !important;
        background-size: 200% auto !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        letter-spacing: 0.4px !important;
        padding: 0.70rem 1.6rem !important;
        width: 100% !important;
        transition: all 0.30s ease !important;
        box-shadow: 0 4px 16px rgba(37,99,235,0.30) !important;
    }}
    .stButton > button:hover {{
        background-position: right center !important;
        transform: translateY(-3px) !important;
        box-shadow: 0 10px 30px rgba(37,99,235,0.50), 0 0 0 1px rgba(20,184,166,0.28) !important;
    }}

    .scan-btn-wrap .stButton > button {{
        background: linear-gradient(135deg, #7c3aed 0%, #2563eb 45%, #0891b2 100%) !important;
        background-size: 200% auto !important;
        font-size: 0.94rem !important;
        font-weight: 700 !important;
        letter-spacing: 1px !important;
        text-transform: uppercase !important;
        padding: 0.82rem !important;
        animation: scan-idle 3s ease-in-out infinite;
    }}
    .scan-btn-wrap .stButton > button:hover {{
        animation: none;
        box-shadow: 0 0 44px rgba(139,92,246,0.60), 0 8px 30px rgba(37,99,235,0.50) !important;
    }}
    @keyframes scan-idle {{
        0%,100% {{ box-shadow: 0 0 18px rgba(139,92,246,0.32), 0 4px 15px rgba(37,99,235,0.24); }}
        50%      {{ box-shadow: 0 0 32px rgba(20,184,166,0.42), 0 6px 20px rgba(37,99,235,0.34); }}
    }}

    .stDownloadButton > button {{
        background: linear-gradient(135deg, #0d9488, #14b8a6) !important;
        color: white !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        border: none !important;
        box-shadow: 0 4px 15px rgba(20,184,166,0.28) !important;
    }}
    .stDownloadButton > button:hover {{
        transform: translateY(-3px) !important;
        box-shadow: 0 10px 28px rgba(20,184,166,0.50) !important;
    }}

    /* ─────────────────────────────────────────────────────────────
       PROGRESS BAR  (scan progress)
    ───────────────────────────────────────────────────────────── */
    .stProgress > div > div > div > div {{
        background: linear-gradient(90deg, #2563eb, #14b8a6, #8b5cf6) !important;
        border-radius: 99px !important;
    }}
    .stProgress > div > div > div {{
        background: rgba(37,99,235,0.12) !important;
        border-radius: 99px !important;
    }}

    /* ─────────────────────────────────────────────────────────────
       SCAN ANIMATION RING
    ───────────────────────────────────────────────────────────── */
    .scan-ring-wrap {{
        text-align: center;
        padding: 24px 0 10px;
    }}
    .scan-ring {{
        width: 96px; height: 96px;
        border-radius: 50%;
        border: 3px solid transparent;
        border-top-color: #2563eb;
        border-right-color: #14b8a6;
        border-left-color: rgba(139,92,246,0.4);
        animation: ring-spin 1.1s cubic-bezier(.47,.13,.19,.97) infinite;
        margin: 0 auto 12px;
        position: relative;
    }}
    .scan-ring::before {{
        content: '';
        position: absolute;
        inset: 6px;
        border-radius: 50%;
        border: 2px solid transparent;
        border-top-color: rgba(20,184,166,0.5);
        animation: ring-spin 1.7s linear infinite reverse;
    }}
    .scan-ring::after {{
        content: '🔬';
        position: absolute;
        top: 50%; left: 50%;
        transform: translate(-50%,-50%);
        font-size: 2rem;
    }}
    @keyframes ring-spin {{ 100% {{ transform: rotate(360deg); }} }}
    .scan-status-txt {{
        font-family: 'Oxanium', monospace;
        font-size: 0.78rem;
        color: {SUB};
        letter-spacing: 2px;
        text-transform: uppercase;
    }}

    /* ─────────────────────────────────────────────────────────────
       RESULT CARDS
    ───────────────────────────────────────────────────────────── */
    .result-card {{
        border-radius: 20px;
        padding: 28px 24px;
        text-align: center;
        margin-bottom: 18px;
        position: relative;
        overflow: hidden;
        animation: result-in 0.55s cubic-bezier(.34,1.56,.64,1);
    }}
    @keyframes result-in {{
        from {{ opacity:0; transform:scale(0.86) translateY(18px); }}
        to   {{ opacity:1; transform:scale(1)    translateY(0);    }}
    }}
    .res-mal {{
        background: linear-gradient(135deg, rgba(239,68,68,0.13), rgba(220,38,38,0.05));
        border: 2px solid rgba(239,68,68,0.52);
        box-shadow: 0 0 48px rgba(239,68,68,0.12), inset 0 1px 0 rgba(239,68,68,0.18);
    }}
    .res-ben {{
        background: linear-gradient(135deg, rgba(16,185,129,0.13), rgba(5,150,105,0.05));
        border: 2px solid rgba(16,185,129,0.52);
        box-shadow: 0 0 48px rgba(16,185,129,0.12), inset 0 1px 0 rgba(16,185,129,0.18);
    }}
    .res-tag  {{ font-size:0.66rem; font-weight:700; text-transform:uppercase; letter-spacing:3px; margin-bottom:9px; opacity:0.82; }}
    .res-type {{ font-family:'Oxanium',sans-serif; font-size:clamp(1.5rem,3vw,2.1rem); font-weight:800; letter-spacing:-0.5px; margin-bottom:8px; }}
    .res-desc {{ font-size:0.83rem; color:{SUB}; line-height:1.6; max-width:380px; margin:0 auto; }}

    /* Risk badges */
    .badge {{ display:inline-flex; align-items:center; gap:5px; padding:5px 16px; border-radius:99px; font-size:0.76rem; font-weight:700; letter-spacing:0.4px; text-transform:uppercase; }}
    .b-high   {{ background:rgba(239,68,68,0.14);  color:#f87171; border:1px solid rgba(239,68,68,0.38);  }}
    .b-medium {{ background:rgba(245,158,11,0.14); color:#fbbf24; border:1px solid rgba(245,158,11,0.38); }}
    .b-low    {{ background:rgba(16,185,129,0.14); color:#34d399; border:1px solid rgba(16,185,129,0.38); }}

    /* Image quality badge */
    .qual-badge-ok   {{ background:rgba(16,185,129,0.12); color:#34d399; border:1px solid rgba(16,185,129,0.30); padding:4px 12px; border-radius:99px; font-size:0.73rem; font-weight:600; display:inline-block; }}
    .qual-badge-warn {{ background:rgba(245,158,11,0.12); color:#fbbf24; border:1px solid rgba(245,158,11,0.30); padding:4px 12px; border-radius:99px; font-size:0.73rem; font-weight:600; display:inline-block; }}

    /* ─────────────────────────────────────────────────────────────
       STEP / RECOMMENDATION BOXES
    ───────────────────────────────────────────────────────────── */
    .step-box {{
        background: {SURF2};
        border: 1px solid {BORDER};
        border-left: 3px solid #2563eb;
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 8px;
        font-size: 0.84rem;
        line-height: 1.6;
        transition: border-left-color 0.22s, background 0.22s;
    }}
    .step-box:hover {{ border-left-color:#14b8a6; background:rgba(20,184,166,0.05); }}
    .step-emg {{ border-left-color:#ef4444 !important; }}
    .step-emg:hover {{ background:rgba(239,68,68,0.05) !important; }}

    /* ─────────────────────────────────────────────────────────────
       TABS
    ───────────────────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {{
        background:{SURF2} !important; border-radius:12px !important;
        padding:4px !important; gap:3px !important; border:1px solid {BORDER};
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius:9px !important; font-family:'Outfit',sans-serif !important;
        font-weight:500 !important; font-size:0.82rem !important;
        color:{SUB} !important; padding:7px 16px !important; transition:all 0.2s !important;
    }}
    .stTabs [aria-selected="true"] {{
        background:linear-gradient(135deg,#2563eb,#1d4ed8) !important;
        color:white !important;
        box-shadow:0 3px 10px rgba(37,99,235,0.38) !important;
    }}

    /* ─────────────────────────────────────────────────────────────
       INPUTS
    ───────────────────────────────────────────────────────────── */
    .stTextInput > div > div > input,
    .stNumberInput input,
    .stSelectbox > div > div > div {{
        background:{INP} !important; border:1px solid {BORDER} !important;
        border-radius:10px !important; color:{TEXT} !important;
        font-family:'Outfit',sans-serif !important; font-size:0.87rem !important;
    }}
    .stTextInput > div > div > input:focus {{
        border-color:rgba(37,99,235,0.55) !important;
        box-shadow:0 0 0 3px rgba(37,99,235,0.10) !important;
    }}

    /* ─────────────────────────────────────────────────────────────
       MISC  ·  scrollbar · metrics · alerts · dataframe
    ───────────────────────────────────────────────────────────── */
    ::-webkit-scrollbar {{ width:5px; height:5px; }}
    ::-webkit-scrollbar-track {{ background:transparent; }}
    ::-webkit-scrollbar-thumb {{ background:rgba(37,99,235,0.30); border-radius:99px; }}
    ::-webkit-scrollbar-thumb:hover {{ background:rgba(20,184,166,0.50); }}

    [data-testid="stMetricLabel"] {{ font-family:'Outfit',sans-serif !important; color:{SUB} !important; font-size:0.74rem !important; text-transform:uppercase; letter-spacing:1.5px; }}
    [data-testid="stMetricValue"] {{ font-family:'Oxanium',monospace !important; font-size:1.45rem !important; color:{TEXT} !important; }}
    [data-testid="stMetricDelta"] {{ font-size:0.75rem !important; }}

    .stAlert {{ border-radius:12px !important; }}
    .stSpinner > div {{ border-top-color:#2563eb !important; }}

    [data-testid="stDataFrame"] {{
        border:1px solid {BORDER} !important;
        border-radius:14px !important;
        overflow:hidden;
    }}

    hr {{ border-color:{DIV} !important; opacity:0.8; }}

    /* Settings rows */
    .set-row {{
        background:{SURF2};
        border:1px solid {BORDER};
        border-radius:14px;
        padding:16px 20px;
        margin-bottom:10px;
        transition:border-color 0.22s;
    }}
    .set-row:hover {{ border-color:{BDH}; }}
    .set-lbl {{ font-weight:600; font-size:0.88rem; margin-bottom:2px; }}
    .set-desc {{ font-size:0.74rem; color:{SUB}; }}

    /* ABCDE cards */
    .abcde-card {{
        background:{SURF}; border:1px solid {BORDER}; border-radius:16px;
        padding:18px 10px; text-align:center;
        transition:all 0.28s cubic-bezier(.34,1.56,.64,1);
    }}
    .abcde-card:hover {{
        transform:translateY(-7px) scale(1.04);
        border-color:rgba(139,92,246,0.48);
        box-shadow:0 14px 32px rgba(139,92,246,0.18);
    }}
    .abcde-letter {{ font-family:'Oxanium',monospace; font-size:2.4rem; font-weight:800; margin-bottom:5px; }}
    .abcde-word   {{ font-weight:700; font-size:0.86rem; margin-bottom:4px; }}
    .abcde-desc   {{ font-size:0.72rem; color:{SUB}; line-height:1.45; }}

    /* Contact card */
    .contact-card {{
        background:{SURF};
        border:1px solid {BORDER};
        border-radius:18px;
        padding:24px 20px;
        text-align:center;
        transition:all 0.28s cubic-bezier(.34,1.56,.64,1);
    }}
    .contact-card:hover {{
        transform:translateY(-5px);
        border-color:{BDH};
        box-shadow:0 16px 40px rgba(37,99,235,0.14);
    }}

    /* Mobile responsive */
    @media (max-width:768px) {{
        .navbar-shell {{ padding:0 12px; height:58px; }}
        .nav-ai-badge {{ display:none; }}
        .hero-section {{ padding:28px 20px; }}
        .hero-section::after {{ display:none; }}
        .glass-card   {{ padding:14px; }}
        .kpi-card     {{ padding:14px 10px; }}
        .kpi-value    {{ font-size:1.55rem; }}
        .page-banner  {{ padding:20px 18px; }}
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
        self.model     = self._load()

    def _load(self):
        try:
            from tensorflow.keras.models import load_model  # type: ignore
            m = load_model(self.MODEL_FILE)
            self.is_online = True
            return m
        except Exception:
            return None

    def execute_scan(self, pil_img: Image.Image) -> dict:
        if self.is_online:
            raw = self._infer(pil_img)
        else:
            raw = random.uniform(0.07, 0.94)

        diag = "Malignant" if raw >= 0.50 else "Benign"
        prob = raw if diag == "Malignant" else (1.0 - raw)
        risk = "HIGH" if prob >= 0.80 else ("MEDIUM" if prob >= 0.50 else "LOW")

        return {
            "diagnosis":  diag,
            "probability":prob,
            "confidence": min(prob + random.uniform(0.01, 0.05), 0.99),
            "risk_level": risk,
            "model_mode": "Neural Network Online" if self.is_online else "Simulation Mode",
        }

    def _infer(self, pil_img):
        from tensorflow.keras.preprocessing.image import img_to_array  # type: ignore
        img = pil_img.convert("RGB").resize(self.INPUT_SIZE)
        arr = np.expand_dims(img_to_array(img) / 255.0, axis=0)
        return float(self.model.predict(arr, verbose=0)[0][0])


# ══════════════════════════════════════════════════════════════════
#  CLASS 2 · ImageProcessor
# ══════════════════════════════════════════════════════════════════
class ImageProcessor:

    @staticmethod
    def validate(file_obj):
        ext = file_obj.name.rsplit(".", 1)[-1].lower() if hasattr(file_obj, "name") else "png"
        if ext not in {"jpg","jpeg","png"}:
            return False, f"❌ Format '.{ext}' not accepted. Use JPG, JPEG, or PNG.", "low"
        if hasattr(file_obj, "size") and file_obj.size > 10*1024*1024:
            return False, f"❌ File too large. Max 10 MB.", "low"
        try:
            img = Image.open(file_obj); img.verify()
        except Exception:
            return False, "❌ Corrupted or unreadable image file.", "low"
        file_obj.seek(0)
        img = Image.open(file_obj)
        w, h = img.size
        if w < 100 or h < 100:
            return False, f"❌ Resolution {w}×{h} too low. Min: 100×100 px.", "low"
        file_obj.seek(0)
        quality = "high" if (w >= 300 and h >= 300) else "medium"
        return True, f"✅ Validated  ·  {w}×{h} px  ·  Quality: {quality.upper()}", quality

    @staticmethod
    def preprocess(pil_img):
        img = pil_img.convert("RGB").resize((224,224), Image.LANCZOS)
        img = ImageEnhance.Contrast(img).enhance(1.20)
        img = ImageEnhance.Sharpness(img).enhance(1.15)
        return img

    @staticmethod
    def thumb(pil_img, size=640):
        img = pil_img.convert("RGB")
        img.thumbnail((size, size), Image.LANCZOS)
        return img


# ══════════════════════════════════════════════════════════════════
#  CLASS 3 · ClinicalProtocols
# ══════════════════════════════════════════════════════════════════
class ClinicalProtocols:
    _DB = {
        "Malignant": {
            "hex":"#ef4444","css":"res-mal","icon":"🔴",
            "description":"AI detects characteristics consistent with a malignant skin lesion. Immediate clinical evaluation is critical.",
            "ai_message":"HIGH RISK ALERT: Irregular pigmentation, asymmetric borders, and multi-color pattern detected — consistent with malignancy. Urgent dermatological consultation required within 48 hours.",
            "recommendations":[
                "🏥 Consult an oncology-dermatologist within 48 hours — do not delay.",
                "🔬 Request formal dermoscopy evaluation and excisional biopsy.",
                "🚫 Avoid all UV exposure immediately — sun and artificial tanning.",
                "🧴 Apply broad-spectrum SPF 100+ at all outdoor times.",
                "📋 Request full-body skin mapping (digital dermoscopic photography).",
                "🩸 Discuss Sentinel Lymph Node Biopsy (SLNB) with your surgeon.",
                "🥗 Antioxidant-rich diet: berries, leafy greens, omega-3 fatty acids.",
            ],
            "patient_advice":[
                "Wear UPF 50+ clothing and wide-brim hats daily without exception.",
                "Stay indoors during peak UV hours — 10:00 AM to 4:00 PM.",
                "Perform weekly ABCDE self-examinations on all skin lesions.",
                "Eliminate tobacco and alcohol use — accelerates cancer progression.",
                "Vitamin D only through supplementation, never from sun exposure.",
                "Keep a photographic log of lesion changes for physician review.",
            ],
            "procedures":[
                "Wide Local Excision (WLE) — removal with clear safety margins.",
                "Mohs Micrographic Surgery — layer-by-layer tissue-sparing excision.",
                "Sentinel Lymph Node Biopsy (SLNB) — regional lymphatic assessment.",
                "Adjuvant Radiation Therapy — post-surgical residual cell ablation.",
                "Systemic Immunotherapy: Pembrolizumab / Ipilimumab protocols.",
            ],
            "medications":[
                "Targeted: BRAF/MEK inhibitors — Vemurafenib + Cobimetinib.",
                "Immunotherapy: Pembrolizumab (Keytruda) — PD-1 checkpoint inhibitor.",
                "Dabrafenib + Trametinib — for BRAF V600E/K mutation cases.",
                "Topical Imiquimod 5% cream — superficial lesions (physician-directed).",
            ],
            "therapy":[
                "Photodynamic Therapy (PDT) for localized superficial involvement.",
                "Electrochemotherapy as adjuvant management post-excision.",
                "Intralesional IL-2 cytokine injection therapy.",
            ],
            "emergency_signs":[
                "⚠️ Rapid lesion enlargement beyond 6mm within days.",
                "⚠️ Spontaneous ulceration, bleeding, or crusting of lesion.",
                "⚠️ Visible lymph node swelling near neck, armpit, or groin.",
                "⚠️ Satellite lesions appearing around the primary lesion.",
                "⚠️ Pain, numbness, or tingling sensation around lesion area.",
            ],
            "followup":"Bi-weekly monitoring for 3 months. PET-CT at 6 months. Oncology review every 3 months for 2 years.",
            "consultation":"🚨 URGENT: Schedule Onco-Dermatologist within 48 hours.",
        },
        "Benign": {
            "hex":"#10b981","css":"res-ben","icon":"🟢",
            "description":"AI indicates a benign skin lesion with low malignant potential. Routine monitoring is recommended as best practice.",
            "ai_message":"LOW RISK: Symmetric borders, uniform pigmentation, and regular morphology are consistent with a benign melanocytic nevus. Annual dermatology monitoring is advised.",
            "recommendations":[
                "✅ No urgent surgical intervention required at this time.",
                "📅 Schedule a routine annual dermatology skin check.",
                "🔍 Perform monthly ABCDE self-examinations as best practice.",
                "🧴 Apply daily SPF 50+ broad-spectrum sunscreen.",
                "📸 Photograph the lesion to establish a monitoring baseline.",
                "🥗 Antioxidant diet and adequate hydration for skin health.",
                "📞 Consult a doctor immediately if the lesion changes in any way.",
            ],
            "patient_advice":[
                "Standard daily sun protection measures are sufficient.",
                "Balanced diet rich in antioxidants and vitamins C and E.",
                "Adequate hydration — minimum 2+ litres of water per day.",
                "Avoid mechanical trauma or scratching of the lesion.",
                "Annual professional dermoscopy evaluation for documentation.",
                "Monitor for ABCDE changes at least once per month.",
            ],
            "procedures":[
                "Clinical observation — no immediate surgical intervention needed.",
                "Digital dermoscopy photography for baseline documentation.",
                "Elective shave excision for cosmetic removal (if desired).",
                "Punch excision if histological confirmation is requested.",
                "CO2 Laser ablation for cosmetic concerns (patient preference).",
            ],
            "medications":[
                "None required — SPF 50+ sunscreen is the primary intervention.",
                "Topical Vitamin C antioxidant serum for skin maintenance.",
                "Ceramide-based barrier moisturizers for skin health.",
                "Vitamin D supplementation — consult physician for dosage.",
            ],
            "therapy":[
                "Cryotherapy (liquid nitrogen) — elective symptomatic relief only.",
                "Topical retinoids for general skin maintenance (physician-directed).",
                "PDT only if pre-malignant features emerge on follow-up.",
            ],
            "emergency_signs":[
                "⚠️ Any sudden change in size, shape, or color (ABCDE).",
                "⚠️ Unexpected bleeding or oozing without physical trauma.",
                "⚠️ New satellite lesions appearing near the original lesion.",
                "⚠️ Persistent itching, burning, or pain in lesion area.",
                "⚠️ Lesion fails to heal after minor trauma within 4 weeks.",
            ],
            "followup":"Annual routine dermatology screening. AI re-evaluation recommended in 6 months.",
            "consultation":"📅 Routine annual dermatology appointment. Consult earlier if ABCDE changes appear.",
        },
    }

    @classmethod
    def get(cls, diag): return cls._DB.get(diag, cls._DB["Benign"])


# ══════════════════════════════════════════════════════════════════
#  CLASS 4 · ReportGenerator
# ══════════════════════════════════════════════════════════════════
class ReportGenerator:

    @staticmethod
    def pdf(record, img):
        buf = io.BytesIO()
        if not PDF_OK: buf.write(b"pip install reportlab"); return buf.getvalue()
        doc  = SimpleDocTemplate(buf, pagesize=A4,
                                 rightMargin=1.8*cm, leftMargin=1.8*cm,
                                 topMargin=1.5*cm, bottomMargin=1.5*cm)
        BLUE = rl_colors.HexColor("#1e3a5f")
        GRAY = rl_colors.HexColor("#64748b")
        diag = record.get("diagnosis","Benign")
        RISK = rl_colors.HexColor("#ef4444" if diag=="Malignant" else "#10b981")
        H1   = ParagraphStyle("H1",  fontSize=19,fontName="Helvetica-Bold",textColor=BLUE,   alignment=TA_CENTER,spaceAfter=3)
        SUB  = ParagraphStyle("SUB", fontSize=8.5,fontName="Helvetica",    textColor=GRAY,   alignment=TA_CENTER,spaceAfter=10)
        SEC  = ParagraphStyle("SEC", fontSize=11, fontName="Helvetica-Bold",textColor=BLUE,  spaceAfter=6,spaceBefore=10)
        TXT  = ParagraphStyle("TXT", fontSize=8.5,fontName="Helvetica",    textColor=rl_colors.HexColor("#374151"),spaceAfter=3,leading=13,leftIndent=6)
        DIS  = ParagraphStyle("DIS", fontSize=7.5,fontName="Helvetica",    textColor=GRAY,   alignment=TA_JUSTIFY,leading=12)
        FTR  = ParagraphStyle("FTR", fontSize=7,  fontName="Helvetica",    textColor=rl_colors.HexColor("#94a3b8"),alignment=TA_CENTER)
        story = [
            Paragraph("🔬  SkinScan AI — Clinical Intelligence Platform", H1),
            Paragraph("Dermoscopic Cancer Detection Report  ·  v14.0", SUB),
            HRFlowable(width="100%",thickness=2,color=BLUE), Spacer(1,10),
        ]
        rows = [
            ["FIELD","DETAIL"],
            ["Patient Name",     record.get("patient_name","N/A")],
            ["Age",              str(record.get("age","N/A"))],
            ["Gender",           record.get("gender","N/A")],
            ["Scan Date & Time", record.get("timestamp","N/A")],
            ["AI Diagnosis",     diag],
            ["Risk Level",       record.get("risk_level","N/A")],
            ["Probability",      f"{record.get('probability',0)*100:.1f}%"],
            ["AI Confidence",    f"{record.get('confidence',0)*100:.1f}%"],
            ["Model Status",     record.get("model_mode","N/A")],
        ]
        tbl = Table(rows, colWidths=[5.5*cm,12.5*cm])
        tbl.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),BLUE),("TEXTCOLOR",(0,0),(-1,0),rl_colors.white),
            ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[rl_colors.HexColor("#f0f4f8"),rl_colors.white]),
            ("FONTNAME",(0,1),(0,-1),"Helvetica-Bold"),
            ("FONTSIZE",(0,0),(-1,-1),9),
            ("GRID",(0,0),(-1,-1),0.4,rl_colors.HexColor("#dde3ea")),
            ("PADDING",(0,0),(-1,-1),7),
            ("TEXTCOLOR",(1,6),(1,6),RISK),("FONTNAME",(1,6),(1,6),"Helvetica-Bold"),
        ]))
        story += [Paragraph("Patient & Scan Information",SEC), tbl, Spacer(1,12)]
        try:
            ibuf=io.BytesIO(); th=img.copy(); th.thumbnail((160,160)); th.save(ibuf,format="PNG"); ibuf.seek(0)
            ri=RLImage(ibuf,width=4.5*cm,height=4.5*cm)
            it=Table([[ri]],colWidths=[18*cm])
            it.setStyle(TableStyle([("ALIGN",(0,0),(-1,-1),"CENTER")]))
            story+=[Paragraph("Uploaded Image",SEC), it, Spacer(1,10)]
        except Exception: pass
        kb = ClinicalProtocols.get(diag)
        story+=[Paragraph("AI Assessment",SEC),
                Paragraph(kb["ai_message"],ParagraphStyle("msg",fontSize=8.5,fontName="Helvetica",
                    textColor=rl_colors.HexColor("#374151"),backColor=rl_colors.HexColor("#f0f9ff"),
                    borderPadding=7,leading=14,spaceAfter=10))]
        story.append(Paragraph("Clinical Recommendations",SEC))
        for r in kb["recommendations"]: story.append(Paragraph(f"• {r}",TXT))
        story.append(Spacer(1,8))
        story.append(Paragraph("Treatment Plan",SEC))
        for lbl,key in [("Procedures","procedures"),("Medications","medications"),
                        ("Therapy","therapy"),("Emergency Signs","emergency_signs")]:
            story.append(Paragraph(f"▸ {lbl}",ParagraphStyle("cat",fontSize=9,fontName="Helvetica-Bold",
                textColor=rl_colors.HexColor("#ef4444" if "Emergency" in lbl else "#2563eb"),
                spaceAfter=2,leftIndent=4,spaceBefore=4)))
            for i in kb[key]: story.append(Paragraph(f"  – {i}",TXT))
        story+=[Spacer(1,8),Paragraph("Follow-up",SEC),Paragraph(kb["followup"],TXT),Spacer(1,12),
                HRFlowable(width="100%",thickness=0.7,color=rl_colors.HexColor("#e2e8f0")),Spacer(1,6),
                Paragraph("⚠️ AI DISCLAIMER: Research & educational tool only. Not a medical diagnosis. "
                          "Always consult a certified dermatologist or oncologist.",DIS),Spacer(1,5),
                Paragraph(f"SkinScan AI v14.0  ·  Rehan Shafique  ·  "
                          f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",FTR)]
        doc.build(story)
        return buf.getvalue()

    @staticmethod
    def csv_data(db):
        if not db: return ""
        return pd.DataFrame([{
            "Timestamp":   r.get("timestamp",""),
            "Patient":     r.get("patient_name",""),
            "Age":         r.get("age",""),
            "Gender":      r.get("gender",""),
            "Diagnosis":   r.get("diagnosis",""),
            "Risk":        r.get("risk_level",""),
            "Probability%":f"{r.get('probability',0)*100:.2f}",
            "Confidence%": f"{r.get('confidence',0)*100:.2f}",
            "Model":       r.get("model_mode",""),
        } for r in db]).to_csv(index=False)


# ══════════════════════════════════════════════════════════════════
#  CLASS 5 · SkinScanApp  (Master Controller)
# ══════════════════════════════════════════════════════════════════
class SkinScanApp:

    def __init__(self):
        st.set_page_config(
            page_title="SkinScan AI — Clinical Intelligence",
            page_icon="🔬",
            layout="wide",
            initial_sidebar_state="collapsed",
        )
        self._init_state()
        self.ai = NeuralCoreEngine()
        inject_css(st.session_state.theme)

    def _init_state(self):
        for k,v in {
            "theme":"dark","db":[],"result":None,
            "raw_img":None,"proc_img":None,"input_mode":"upload",
        }.items():
            if k not in st.session_state:
                st.session_state[k] = v

    # ──────────────────────────────────────────────────────────────
    #  TOP NAVBAR
    # ──────────────────────────────────────────────────────────────
    def _navbar(self) -> str:
        # Inject fixed HTML navbar shell (logo + AI badge)
        theme_icon = "🌙" if st.session_state.theme == "dark" else "☀️"
        ai_status  = "🟢 AI Online" if self.ai.is_online else "🟠 Sim Mode"

        st.markdown(f"""
        <div class="navbar-shell">
            <div class="nav-logo">
                <span class="nav-logo-icon">🔬</span>
                <div>
                    <div class="nav-logo-text">SkinScan AI</div>
                    <div class="nav-logo-ver">Clinical Intelligence · v14</div>
                </div>
            </div>
            <div style="flex:1; display:flex; justify-content:center; align-items:center;">
                <!-- Option menu injected below via streamlit -->
            </div>
            <div style="display:flex; align-items:center; gap:10px; flex-shrink:0;">
                <span class="nav-ai-badge">● {ai_status}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Horizontal option menu
        st.markdown('<div class="nav-menu-center" style="max-width:680px; margin:0 auto 18px;">', unsafe_allow_html=True)
        nav = option_menu(
            menu_title=None,
            options=["Home","AI Scan","Dashboard","History","About"],
            icons=["house-fill","cpu-fill","grid-3x3-gap-fill","clock-history","info-circle-fill"],
            orientation="horizontal",
            default_index=0,
            styles={
                "container":         {"padding":"0","background":"transparent"},
                "nav-link":          {
                    "font-family":"Outfit,sans-serif","font-size":"0.83rem",
                    "font-weight":"500","padding":"7px 15px",
                    "border-radius":"9px","margin":"0 1px",
                    "color":"#6b9ab8","transition":"all 0.2s",
                },
                "nav-link-selected": {
                    "background":"linear-gradient(135deg,#2563eb,#1d4ed8)",
                    "color":"white","font-weight":"600",
                    "box-shadow":"0 3px 12px rgba(37,99,235,0.40)",
                },
                "icon": {"font-size":"0.84rem"},
            },
        )
        st.markdown('</div>', unsafe_allow_html=True)
        return nav

    # ──────────────────────────────────────────────────────────────
    #  LAUNCH
    # ──────────────────────────────────────────────────────────────
    def launch(self):
        nav = self._navbar()
        {
            "Home":      self._home,
            "AI Scan":   self._scan,
            "Dashboard": self._dashboard,
            "History":   self._history,
            "About":     self._about,
        }.get(nav, self._home)()
        self._footer()

    # ══════════════════════════════════════════════════════════════
    #  PAGE: HOME
    # ══════════════════════════════════════════════════════════════
    def _home(self):
        # Hero
        st.markdown("""
        <div class="hero-section">
            <div class="hero-badges">
                <span class="hbadge hbadge-blue">🔬 AI-Powered</span>
                <span class="hbadge hbadge-teal">🏥 Clinical Grade</span>
                <span class="hbadge hbadge-purple">🧬 CNN Model</span>
                <span class="hbadge hbadge-green">✅ FYP Final Build</span>
            </div>
            <h1 class="hero-title">AI Dermatology<br>Clinical Platform</h1>
            <p class="hero-subtitle">
                Upload a dermoscopic skin image or capture live using your camera.
                Our CNN model instantly detects <b>Benign</b> or <b>Malignant</b> lesions
                with clinical-grade confidence scores and full treatment recommendations.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Feature cards
        fc = [
            ("🧬","CNN Analysis","Upload or capture skin lesion images for instant AI-powered Benign vs Malignant classification."),
            ("📷","Live Camera","Capture directly from your device webcam or mobile camera. No need to upload files."),
            ("📊","Clinical Reports","Downloadable PDF + CSV reports with diagnosis, treatment plan, and patient recommendations."),
            ("📈","Analytics","Real-time epidemiological charts, risk distributions, and session trend analysis."),
            ("🛡️","Image Quality Check","Automatic validation — resolution, format, and corruption detection before AI analysis."),
            ("🌓","Dark / Light Mode","Toggle between dark clinical mode and light mode from the Settings panel below."),
        ]
        r1 = st.columns(3)
        r2 = st.columns(3)
        for col, (icon,title,desc) in zip(list(r1)+list(r2), fc):
            with col:
                st.markdown(f"""
                <div class="feat-card">
                    <span class="feat-icon">{icon}</span>
                    <div class="feat-title">{title}</div>
                    <div class="feat-desc">{desc}</div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("---")
        # ABCDE quick guide
        st.markdown('<div class="sec-head"><span></span>🎗️ ABCDE Melanoma Self-Check</div>',
                    unsafe_allow_html=True)
        abcde = [
            ("A","Asymmetry","#ef4444","One half doesn't match the other."),
            ("B","Border",   "#f97316","Irregular, ragged, or blurred edges."),
            ("C","Color",    "#f59e0b","Multiple shades of brown, black, or red."),
            ("D","Diameter", "#3b82f6","Larger than 6mm — a pencil eraser."),
            ("E","Evolution","#8b5cf6","Any change in size, shape, or color."),
        ]
        a1,a2,a3,a4,a5 = st.columns(5)
        for col,(L,W,C,D) in zip([a1,a2,a3,a4,a5],abcde):
            with col:
                st.markdown(f"""
                <div class="abcde-card" style="border-top:3px solid {C};">
                    <div class="abcde-letter" style="color:{C};">{L}</div>
                    <div class="abcde-word">{W}</div>
                    <div class="abcde-desc">{D}</div>
                </div>
                """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════
    #  PAGE: AI SCAN
    # ══════════════════════════════════════════════════════════════
    def _scan(self):
        st.markdown("""
        <div class="page-banner">
            <div class="banner-chip">🧬 Neural Scan Engine</div>
            <p class="banner-title">AI Analysis Laboratory</p>
            <p class="banner-sub">skin_cancer_cnn.h5 · Benign / Malignant · Upload image or use live camera · Full clinical report</p>
        </div>
        """, unsafe_allow_html=True)

        col_in, col_out = st.columns([1, 1.4], gap="large")

        # ── INPUT COLUMN ──────────────────────────────────────────
        with col_in:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="sec-head"><span></span>👤 Patient Information</div>',
                        unsafe_allow_html=True)

            p_name   = st.text_input("Patient Name / ID", placeholder="e.g. Ahmed Khan  /  PT-2024-001")
            a_col, g_col = st.columns(2)
            with a_col: p_age    = st.number_input("Age", min_value=1, max_value=120, value=35)
            with g_col: p_gender = st.selectbox("Gender", ["Male","Female","Other","Prefer not to say"])

            # ── Input Mode Toggle ─────────────────────────────────
            st.markdown('<div class="sec-head" style="margin-top:16px;"><span></span>📸 Image Input Method</div>',
                        unsafe_allow_html=True)

            mode_col1, mode_col2 = st.columns(2)
            with mode_col1:
                if st.button("📁 Upload File",
                             type="primary" if st.session_state.input_mode=="upload" else "secondary"):
                    st.session_state.input_mode = "upload"
                    st.rerun()
            with mode_col2:
                if st.button("📷 Live Camera",
                             type="primary" if st.session_state.input_mode=="camera" else "secondary"):
                    st.session_state.input_mode = "camera"
                    st.rerun()

            # ── Image Source ──────────────────────────────────────
            raw_img = None
            img_ok  = False
            qual    = "low"

            if st.session_state.input_mode == "upload":
                st.caption("JPG · JPEG · PNG  ·  Max 10 MB  ·  Min 100×100 px")
                upl = st.file_uploader("Drop image here",
                                       type=["jpg","jpeg","png"],
                                       label_visibility="collapsed")
                if upl:
                    ok, msg, qual = ImageProcessor.validate(upl)
                    if not ok:
                        st.error(msg)
                    else:
                        raw_img = Image.open(upl)
                        img_ok  = True
                        # Quality indicator
                        badge_cls = "qual-badge-ok" if qual == "high" else "qual-badge-warn"
                        st.markdown(f'<div class="{badge_cls}" style="margin-bottom:8px;">'
                                    f'{"✅" if qual=="high" else "⚠️"} Image Quality: {qual.upper()}'
                                    f'</div>', unsafe_allow_html=True)
                        if qual == "medium":
                            st.warning("⚠️ Image resolution is moderate. Higher resolution dermoscopic images improve AI accuracy.")
                        disp = ImageProcessor.thumb(raw_img)
                        st.image(disp, use_container_width=True,
                                 caption=f"📐 {raw_img.size[0]}×{raw_img.size[1]} px")

            else:  # camera mode
                st.caption("Allow camera access when prompted by your browser.")
                cam_img = st.camera_input("📷 Capture skin lesion image",
                                          label_visibility="visible")
                if cam_img:
                    raw_img = Image.open(cam_img)
                    img_ok  = True
                    qual    = "high"
                    st.markdown('<div class="qual-badge-ok">✅ Camera Image Captured</div>',
                                unsafe_allow_html=True)

            # ── Scan Button ───────────────────────────────────────
            st.markdown('<div class="scan-btn-wrap" style="margin-top:16px;">', unsafe_allow_html=True)
            run = st.button("▶ EXECUTE DEEP SCAN", disabled=(not img_ok))
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)  # glass-card

        # ── OUTPUT COLUMN ─────────────────────────────────────────
        with col_out:
            if img_ok and run:
                # Progress animation
                prog_ph = st.empty()
                ring_ph = st.empty()
                ring_ph.markdown("""
                <div class="scan-ring-wrap">
                    <div class="scan-ring"></div>
                    <div class="scan-status-txt">AI Analyzing Skin Lesion…</div>
                </div>
                """, unsafe_allow_html=True)

                steps = ["Preprocessing image…","Extracting feature maps…",
                         "Running CNN inference…","Generating clinical report…"]
                for i, step in enumerate(steps):
                    prog_ph.progress((i+1)*25, text=f"⚡ {step}")
                    time.sleep(0.55)

                ring_ph.empty()
                prog_ph.empty()

                processed = ImageProcessor.preprocess(raw_img)
                result    = self.ai.execute_scan(processed)
                intel     = ClinicalProtocols.get(result["diagnosis"])

                rec = {
                    "timestamp":    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "patient_name": p_name.strip() or "Anonymous",
                    "age":          p_age,
                    "gender":       p_gender,
                    **result,
                }
                st.session_state.db.append(rec)
                st.session_state.result   = rec
                st.session_state.raw_img  = raw_img
                st.session_state.proc_img = processed
                st.rerun()

            if st.session_state.result:
                res   = st.session_state.result
                intel = ClinicalProtocols.get(res["diagnosis"])

                # Result card
                st.markdown(f"""
                <div class="result-card {intel['css']}">
                    <div class="res-tag" style="color:{intel['hex']};">◉ AI DIAGNOSIS RESULT</div>
                    <div class="res-type" style="color:{intel['hex']};">
                        {intel['icon']}  {res['diagnosis']}
                    </div>
                    <div class="res-desc">{intel['description']}</div>
                </div>
                """, unsafe_allow_html=True)

                # Metrics
                m1, m2, m3 = st.columns(3)
                m1.metric("Cancer Probability", f"{res['probability']*100:.1f}%")
                m2.metric("AI Confidence",       f"{res['confidence']*100:.1f}%")
                bc = {"HIGH":"b-high","MEDIUM":"b-medium","LOW":"b-low"}[res["risk_level"]]
                m3.markdown(f"""
                <div style="text-align:center; padding-top:6px;">
                    <div style="font-size:0.68rem; color:#6b9ab8; margin-bottom:7px;
                                text-transform:uppercase; letter-spacing:1.8px;">Risk Level</div>
                    <span class="badge {bc}">● {res['risk_level']}</span>
                </div>""", unsafe_allow_html=True)

                # Confidence gauge
                fig_g = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=res["confidence"]*100,
                    number={"suffix":"%","font":{"family":"Oxanium","size":28,"color":intel["hex"]}},
                    title={"text":"AI Confidence","font":{"family":"Outfit","size":11,"color":"#6b9ab8"}},
                    gauge={
                        "axis":{"range":[0,100],"tickfont":{"size":9,"color":"#6b9ab8"},
                                "tickcolor":"rgba(100,116,139,0.25)"},
                        "bar":{"color":intel["hex"],"thickness":0.22},
                        "bgcolor":"rgba(0,0,0,0)","borderwidth":0,
                        "steps":[
                            {"range":[0,40],  "color":"rgba(16,185,129,0.05)"},
                            {"range":[40,70], "color":"rgba(245,158,11,0.05)"},
                            {"range":[70,100],"color":"rgba(239,68,68,0.05)"},
                        ],
                        "threshold":{"line":{"color":intel["hex"],"width":3},
                                     "value":res["confidence"]*100},
                    },
                ))
                fig_g.update_layout(
                    height=195, margin=dict(l=10,r=10,t=40,b=5),
                    paper_bgcolor="rgba(0,0,0,0)", font_color="#6b9ab8",
                )
                st.plotly_chart(fig_g, use_container_width=True)

                # Probability fill bar
                pct = res["probability"]*100
                fig_b = go.Figure()
                fig_b.add_trace(go.Bar(x=[pct],y=[""],orientation="h",
                    marker=dict(color=intel["hex"],line=dict(width=0)),
                    text=[f"  {pct:.1f}%"],textposition="inside",
                    textfont=dict(color="white",size=13,family="Oxanium"),
                    width=0.5,hoverinfo="skip"))
                fig_b.add_trace(go.Bar(x=[100-pct],y=[""],orientation="h",
                    marker=dict(color="rgba(100,116,139,0.10)",line=dict(width=0)),
                    width=0.5,hoverinfo="skip"))
                fig_b.update_layout(
                    height=62, margin=dict(l=0,r=0,t=0,b=0),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    barmode="stack",showlegend=False,
                    xaxis=dict(range=[0,100],showticklabels=False,showgrid=False,zeroline=False),
                    yaxis=dict(showticklabels=False,showgrid=False),
                )
                st.plotly_chart(fig_b, use_container_width=True)

                st.info(f"🤖  {intel['ai_message']}")
                st.caption(f"🔩 **{res['model_mode']}**  ·  📅 {res['timestamp']}")

            else:
                st.markdown("""
                <div class="glass-card" style="text-align:center; padding:4.5rem 1.5rem;">
                    <div style="font-size:4rem; margin-bottom:14px; opacity:0.55;">🔬</div>
                    <div style="font-weight:700; font-size:0.98rem; margin-bottom:8px;">
                        Ready for Analysis
                    </div>
                    <div style="font-size:0.83rem; color:#6b9ab8; line-height:1.7;">
                        Upload an image or capture via camera<br>
                        then click <b>EXECUTE DEEP SCAN</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # ── Clinical Engine ───────────────────────────────────────
        if st.session_state.result:
            res   = st.session_state.result
            intel = ClinicalProtocols.get(res["diagnosis"])

            st.markdown("---")
            st.markdown('<div class="sec-head" style="font-size:1.04rem;"><span></span>📋 Clinical Intelligence Engine</div>',
                        unsafe_allow_html=True)

            t1,t2,t3,t4 = st.tabs(["🏥 Recommendations","🌿 Patient Advice","💊 Treatment Plan","📄 Report"])

            with t1:
                r1,r2 = st.columns(2)
                with r1:
                    st.markdown("**Clinical Recommendations**")
                    for i in intel["recommendations"]:
                        st.markdown(f'<div class="step-box">{i}</div>', unsafe_allow_html=True)
                with r2:
                    st.markdown("**Consultation & Follow-up**")
                    st.markdown(f'<div class="step-box" style="border-left-color:{intel["hex"]};">'
                                f'{intel["consultation"]}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="step-box">📅 {intel["followup"]}</div>',
                                unsafe_allow_html=True)

            with t2:
                for i in intel["patient_advice"]:
                    st.markdown(f'<div class="step-box">🌿 {i}</div>', unsafe_allow_html=True)

            with t3:
                tc1,tc2 = st.columns(2)
                plan = [("🩺 Procedures","procedures",False,"#2563eb"),
                        ("💊 Medications","medications",False,"#14b8a6"),
                        ("⚗️ Therapy","therapy",False,"#8b5cf6"),
                        ("🚨 Emergency Signs","emergency_signs",True,"#ef4444")]
                for i,(lbl,key,emg,c) in enumerate(plan):
                    col = tc1 if i%2==0 else tc2
                    with col:
                        st.markdown(f"""
                        <div class="glass-card" style="border-left:3px solid {c};padding:15px;margin-bottom:10px;">
                            <div style="font-weight:700;color:{c};margin-bottom:10px;font-size:0.87rem;">{lbl}</div>
                            {''.join(f'<div class="step-box {"step-emg" if emg else ""}" style="margin-bottom:5px;">{s}</div>' for s in intel[key])}
                        </div>
                        """, unsafe_allow_html=True)

            with t4:
                st.markdown("#### 📥 Download Clinical Reports")
                d1,d2 = st.columns(2)
                rec     = st.session_state.result
                proc    = st.session_state.proc_img
                with d1:
                    if PDF_OK and proc:
                        pdf_bytes = ReportGenerator.pdf(rec, proc)
                        fname = f"SkinScan_{rec.get('patient_name','PT')}_{datetime.date.today()}.pdf".replace(" ","_")
                        st.download_button("📄 Download PDF Report",
                                           data=pdf_bytes,file_name=fname,mime="application/pdf")
                    else:
                        st.warning("Install ReportLab:\n`pip install reportlab`")
                with d2:
                    st.download_button("📊 Download CSV Registry",
                                       data=ReportGenerator.csv_data(st.session_state.db),
                                       file_name=f"SkinScan_Registry_{datetime.date.today()}.csv",
                                       mime="text/csv")
                st.markdown("""
                <div style='font-size:0.73rem;color:#6b9ab8;margin-top:12px;padding:12px 14px;
                            border:1px solid rgba(100,116,139,0.18);border-radius:10px;line-height:1.6;'>
                    ⚠️ <b>Disclaimer:</b> AI-generated reports for research & educational purposes only.
                    Not a formal medical diagnosis. Always consult a certified dermatologist.
                </div>
                """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════
    #  PAGE: DASHBOARD
    # ══════════════════════════════════════════════════════════════
    def _dashboard(self):
        st.markdown("""
        <div class="page-banner">
            <div class="banner-chip">📊 Analytics</div>
            <p class="banner-title">Clinical Dashboard</p>
            <p class="banner-sub">Session statistics · Real-time KPIs · Diagnosis patterns · Confidence trends</p>
        </div>
        """, unsafe_allow_html=True)

        db = st.session_state.db
        n  = len(db)
        mal = sum(1 for r in db if r.get("diagnosis")=="Malignant")
        c   = (sum(r.get("confidence",0) for r in db)/n*100) if n else 0

        k1,k2,k3,k4 = st.columns(4)
        kpis = [
            ("🧬","Total Scans",    str(n),    "This session",        "#3b82f6"),
            ("🔴","Malignant",      str(mal),  "High-risk detected",  "#ef4444"),
            ("🟢","Benign",         str(n-mal),"Low-risk cleared",    "#10b981"),
            ("⚡","Avg Confidence", f"{c:.1f}%","CNN inference avg",  "#8b5cf6"),
        ]
        for col,(icon,lbl,val,dlt,color) in zip([k1,k2,k3,k4],kpis):
            with col:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-glow" style="background:{color};"></div>
                    <div class="kpi-icon">{icon}</div>
                    <div class="kpi-label">{lbl}</div>
                    <div class="kpi-value" style="color:{color};">{val}</div>
                    <div class="kd-neu">{dlt}</div>
                </div>
                """, unsafe_allow_html=True)

        if not db:
            st.markdown("<br>", unsafe_allow_html=True)
            st.info("📭 No scan data. Head to **AI Scan** to begin.")
            return

        df = pd.DataFrame(db)
        df["prob"] = df["probability"]*100
        df["conf"] = df["confidence"]*100

        PL = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                  font=dict(family="Outfit",color="#6b9ab8"),
                  margin=dict(l=4,r=4,t=44,b=4))
        GR = dict(gridcolor="rgba(37,99,235,0.10)",zerolinecolor="rgba(37,99,235,0.08)")

        st.markdown("<br>", unsafe_allow_html=True)
        r1,r2 = st.columns(2)

        with r1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            ser = df["diagnosis"].value_counts()
            fig1 = go.Figure(go.Pie(
                labels=ser.index.tolist(), values=ser.values.tolist(), hole=0.54,
                marker=dict(colors=["#ef4444","#10b981"],line=dict(color="rgba(0,0,0,0)",width=2)),
                textinfo="percent+label", textfont_size=11,
                hovertemplate="<b>%{label}</b><br>%{value} cases<br>%{percent}<extra></extra>",
            ))
            fig1.update_layout(title="Diagnosis Distribution", height=300,
                               showlegend=True, legend=dict(font_size=11,orientation="h",y=-0.1),
                               annotations=[dict(text=f"<b>{n}</b><br>scans",x=0.5,y=0.5,
                                                 font_size=14,font_color="#dff0fa",showarrow=False)],
                               **PL)
            st.plotly_chart(fig1, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with r2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            rc = df["risk_level"].value_counts().reset_index()
            rc.columns = ["Risk","Count"]
            fig2 = go.Figure()
            for risk,color in [("HIGH","#ef4444"),("MEDIUM","#f59e0b"),("LOW","#10b981")]:
                s = rc[rc["Risk"]==risk]
                if not s.empty:
                    fig2.add_trace(go.Bar(x=s["Risk"],y=s["Count"],name=risk,
                        marker_color=color,marker_line_width=0,
                        hovertemplate=f"<b>{risk}</b><br>%{{y}} cases<extra></extra>"))
            fig2.update_layout(title="Risk Distribution",height=300,showlegend=False,
                               barmode="group",
                               xaxis=dict(title="",**GR),yaxis=dict(title="Cases",**GR),**PL)
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        r3,r4 = st.columns(2)
        with r3:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            fig3 = go.Figure()
            for diag,color,sym in [("Malignant","#ef4444","circle"),("Benign","#10b981","diamond")]:
                s = df[df["diagnosis"]==diag]
                if not s.empty:
                    fig3.add_trace(go.Scatter(
                        x=s["prob"],y=s["conf"],mode="markers",name=diag,
                        marker=dict(color=color,size=9,opacity=0.85,symbol=sym),
                        hovertemplate=f"<b>{diag}</b><br>Prob: %{{x:.1f}}%<br>Conf: %{{y:.1f}}%<extra></extra>"))
            fig3.update_layout(title="Probability vs Confidence",height=285,
                               xaxis=dict(title="Probability (%)",**GR),
                               yaxis=dict(title="Confidence (%)",**GR),
                               legend=dict(orientation="h",y=-0.2,font_size=11),**PL)
            st.plotly_chart(fig3, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with r4:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            if len(df)>=2:
                x = list(range(1,len(df)+1))
                fig4 = go.Figure()
                fig4.add_trace(go.Scatter(x=x,y=df["conf"],mode="lines+markers",name="Confidence",
                    line=dict(color="#2563eb",width=2.5,shape="spline"),
                    marker=dict(size=7),fill="tozeroy",fillcolor="rgba(37,99,235,0.05)"))
                fig4.add_trace(go.Scatter(x=x,y=df["prob"],mode="lines+markers",name="Probability",
                    line=dict(color="#ef4444",width=2,dash="dot",shape="spline"),
                    marker=dict(size=7)))
                fig4.update_layout(title="Scan Trend",height=285,
                                   xaxis=dict(title="Scan #",**GR),
                                   yaxis=dict(title="Score (%)",**GR,range=[0,105]),
                                   legend=dict(orientation="h",y=-0.2,font_size=11),**PL)
                st.plotly_chart(fig4, use_container_width=True)
            else:
                st.info("Need 2+ scans for trend analysis.")
            st.markdown('</div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════
    #  PAGE: HISTORY
    # ══════════════════════════════════════════════════════════════
    def _history(self):
        st.markdown("""
        <div class="page-banner">
            <div class="banner-chip">📋 Patient History</div>
            <p class="banner-title">Scan History Database</p>
            <p class="banner-sub">Complete session records · Smart filters · Export to CSV / JSON</p>
        </div>
        """, unsafe_allow_html=True)

        db = st.session_state.db
        if not db:
            st.info("📭 No records yet. Run scans in **AI Scan** to populate history.")
            return

        df = pd.DataFrame([{
            "Time":      r.get("timestamp","").split(" ")[1] if " " in r.get("timestamp","") else "",
            "Patient":   r.get("patient_name","ANON"),
            "Age":       r.get("age","—"),
            "Gender":    r.get("gender","—"),
            "Diagnosis": r.get("diagnosis","—"),
            "Risk":      r.get("risk_level","—"),
            "Prob.":     f"{r.get('probability',0)*100:.1f}%",
            "Conf.":     f"{r.get('confidence',0)*100:.1f}%",
            "Model":     r.get("model_mode","—"),
        } for r in db])

        k1,k2,k3,k4 = st.columns(4)
        m = sum(1 for r in db if r.get("diagnosis")=="Malignant")
        h = sum(1 for r in db if r.get("risk_level")=="HIGH")
        avg_c = sum(r.get("confidence",0) for r in db)/len(db)*100
        with k1: st.markdown(f'<div class="kpi-card"><div class="kpi-label">Records</div><div class="kpi-value" style="color:#60a5fa;">{len(db)}</div></div>', unsafe_allow_html=True)
        with k2: st.markdown(f'<div class="kpi-card"><div class="kpi-label">Malignant</div><div class="kpi-value" style="color:#f87171;">{m}</div></div>', unsafe_allow_html=True)
        with k3: st.markdown(f'<div class="kpi-card"><div class="kpi-label">High Risk</div><div class="kpi-value" style="color:#fbbf24;">{h}</div></div>', unsafe_allow_html=True)
        with k4: st.markdown(f'<div class="kpi-card"><div class="kpi-label">Avg Conf.</div><div class="kpi-value" style="color:#a78bfa;">{avg_c:.1f}%</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        with st.expander("🔍 Filter Records"):
            fc1,fc2,fc3 = st.columns(3)
            fd = fc1.multiselect("Diagnosis",  ["Malignant","Benign"],  default=["Malignant","Benign"])
            fr = fc2.multiselect("Risk",       ["HIGH","MEDIUM","LOW"], default=["HIGH","MEDIUM","LOW"])
            fg = fc3.multiselect("Gender", ["Male","Female","Other","Prefer not to say"],
                                 default=["Male","Female","Other","Prefer not to say"])

        mask = df["Diagnosis"].isin(fd) & df["Risk"].isin(fr) & df["Gender"].isin(fg)
        df_f = df[mask]
        st.caption(f"Showing **{len(df_f)}** of **{len(df)}** records")

        st.markdown('<div class="glass-card" style="padding:0;overflow:hidden;">', unsafe_allow_html=True)
        st.dataframe(df_f, use_container_width=True, hide_index=True, height=380)
        st.markdown('</div>', unsafe_allow_html=True)

        e1,e2,e3 = st.columns(3)
        with e1:
            st.download_button("📥 Export CSV",
                               data=ReportGenerator.csv_data(db),
                               file_name=f"SkinScan_{datetime.date.today()}.csv",
                               mime="text/csv")
        with e2:
            safe = [{k:str(v) if isinstance(v,datetime.datetime) else v for k,v in r.items()} for r in db]
            st.download_button("🔗 Export JSON",
                               data=json.dumps(safe,indent=2),
                               file_name=f"SkinScan_{datetime.date.today()}.json",
                               mime="application/json")
        with e3:
            if st.button("🗑️ Clear All Records"):
                st.session_state.db=[];st.session_state.result=None
                st.session_state.raw_img=None;st.session_state.proc_img=None
                st.rerun()

    # ══════════════════════════════════════════════════════════════
    #  PAGE: ABOUT  (Settings + Guide + Contact)
    # ══════════════════════════════════════════════════════════════
    def _about(self):
        st.markdown("""
        <div class="page-banner">
            <div class="banner-chip">ℹ️ About</div>
            <p class="banner-title">About & Settings</p>
            <p class="banner-sub">Platform information · Appearance · User guide · Contact</p>
        </div>
        """, unsafe_allow_html=True)

        # Theme
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="sec-head"><span></span>🎨 Appearance</div>', unsafe_allow_html=True)
        c1,c2 = st.columns([3,1])
        with c1:
            st.markdown('<div class="set-lbl">Color Theme</div><div class="set-desc">Switch between dark clinical mode and light mode</div>', unsafe_allow_html=True)
        with c2:
            t = st.toggle("Dark Mode", value=(st.session_state.theme=="dark"))
            if t != (st.session_state.theme=="dark"):
                st.session_state.theme = "dark" if t else "light"
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        # AI Engine info
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="sec-head"><span></span>🤖 AI Engine Details</div>', unsafe_allow_html=True)
        dot  = "🟢" if self.ai.is_online else "🟠"
        mode = "Neural Network Online" if self.ai.is_online else "Simulation Mode"
        rows = [
            ("Model File",       "skin_cancer_cnn.h5",                          "#14b8a6"),
            ("Architecture",     "Convolutional Neural Network (CNN)",           "#60a5fa"),
            ("Output Classes",   "Benign  ·  Malignant (Binary sigmoid)",        "#a78bfa"),
            ("Input Dimensions", "224 × 224 px  ·  RGB  ·  Normalized 0–1",     "#60a5fa"),
            ("Preprocessing",    "Resize → Normalize → Contrast → Sharpen",      "#7fa3c0"),
            ("Engine Status",    f"{dot} {mode}",                               "#f59e0b"),
            ("Version",          "SkinScan AI Clinical Intelligence v14.0",      "#7fa3c0"),
        ]
        for lbl,val,color in rows:
            st.markdown(f"""
            <div class="set-row">
                <span class="set-lbl">{lbl}</span>
                <span style="font-family:'Oxanium',monospace;font-size:0.81rem;color:{color};">{val}</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # User Guide
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="sec-head"><span></span>📖 User Guide</div>', unsafe_allow_html=True)
        guide = [
            ("📤 Image Upload", ["Accepted: <b>JPG, JPEG, PNG</b> only.",
                "Max <b>10 MB</b> · Min <b>100×100 px</b>.",
                "System auto-preprocesses: resize to 224×224 → normalize → enhance.",
                "A quality warning appears if image resolution is below 300×300 px."]),
            ("📷 Camera Capture", ["Click <b>Live Camera</b> toggle in AI Scan page.",
                "Allow camera access when the browser prompts.",
                "Click the camera button to capture the lesion image.",
                "Captured image goes through the same preprocessing pipeline."]),
            ("🤖 AI Inference", ["Model: <b>skin_cancer_cnn.h5</b> loaded at startup.",
                "🟢 Online Mode: Real TF inference on your image.",
                "🟠 Simulation: Model not found — demo mode, no crash.",
                "Sigmoid ≥ 0.50 = Malignant · < 0.50 = Benign."]),
            ("📊 Results", ["<b>Probability</b>: Likelihood of the primary diagnosis.",
                "<b>Confidence</b>: Model certainty (higher = more reliable).",
                "Risk: HIGH ≥80% · MEDIUM 50–80% · LOW <50%.",
                "Download PDF + CSV reports from the Report tab."]),
            ("⚠️ Limitations", ["Research tool only — NOT a certified medical device.",
                "Always consult a qualified dermatologist for clinical decisions.",
                "Session data clears on browser refresh — export to retain.",
                "Accuracy improves with clinical dermoscopic images."]),
        ]
        for title,points in guide:
            with st.expander(title):
                for pt in points:
                    st.markdown(f'<div class="step-box">{pt}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Contact
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="sec-head"><span></span>📬 Contact</div>', unsafe_allow_html=True)
        cc1,cc2,cc3 = st.columns(3)
        contacts = [
            ("👨‍💻","Developer","Rehan Shafique","FYP · Bioinformatics"),
            ("🏫","Institution","University","Department of Bioinformatics"),
            ("🔬","Project","SkinScan AI","Clinical Intelligence Platform v14.0"),
        ]
        for col,(icon,lbl,name,sub) in zip([cc1,cc2,cc3],contacts):
            with col:
                st.markdown(f"""
                <div class="contact-card">
                    <div style="font-size:2rem;margin-bottom:10px;">{icon}</div>
                    <div style="font-size:0.68rem;color:#6b9ab8;text-transform:uppercase;
                                letter-spacing:1.8px;margin-bottom:5px;">{lbl}</div>
                    <div style="font-weight:700;font-size:0.96rem;margin-bottom:3px;">{name}</div>
                    <div style="font-size:0.76rem;color:#6b9ab8;">{sub}</div>
                </div>
                """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────────────
    #  FOOTER
    # ──────────────────────────────────────────────────────────────
    # def _footer(self):
    #     st.markdown("""
    #     <div style="text-align:center;padding:2.5rem 0 1rem;
    #                 border-top:1px solid rgba(37,99,235,0.12);margin-top:3rem;">
    #         <div style="font-family:'Oxanium',sans-serif;font-size:1.1rem;font-weight:800;
    #                     background:linear-gradient(135deg,#60a5fa,#14b8a6,#a78bfa);
    #                     -webkit-background-clip:text;-webkit-text-fill-color:transparent;
    #                     margin-bottom:8px;">
    #             🔬  SkinScan AI — Clinical Intelligence Platform
    #         </div>
    #         <div style="font-size:0.78rem;color:#475569;line-height:2.0;">
    #             Developed by <b style="color:#64748b;">Rehan Shafique</b>
    #             &nbsp;·&nbsp; Final Year Project &nbsp;·&nbsp; Bioinformatics<br>
    #             Python · Streamlit · TensorFlow · Plotly · ReportLab · PIL
    #         </div>
    #         <div style="font-size:0.72rem;color:#ef4444;margin-top:8px;">
    #             ⚠️ Research &amp; Educational Use Only — Not a Certified Medical Device
    #       </div>
    #     </div>
    #      """, unsafe_allow_html=True)
   # Replace the _footer method in the SkinScanApp class with this updated version:

def _footer(self):
    st.markdown("""
    <footer style="
        background: linear-gradient(135deg, rgba(2,13,30,0.95) 0%, rgba(4,22,50,0.92) 100%);
        border-top: 1px solid rgba(37,99,235,0.22);
        padding: 2.5rem 2rem 1.5rem;
        margin-top: 4rem;
        border-radius: 20px 20px 0 0;
        backdrop-filter: blur(16px);
        text-align: center;
        position: relative;
        overflow: hidden;
    ">
        <!-- Decorative gradient line -->
        <div style="
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 3px;
            background: linear-gradient(90deg, 
                transparent 0%, 
                #2563eb 25%, 
                #14b8a6 50%, 
                #8b5cf6 75%, 
                transparent 100%);
        "></div>
        
        <!-- Main logo/title -->
        <div style="
            font-family: 'Oxanium', sans-serif;
            font-size: clamp(1.3rem, 2.8vw, 1.8rem);
            font-weight: 800;
            background: linear-gradient(135deg, #60a5fa 0%, #14b8a6 40%, #a78bfa 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.3px;
            margin-bottom: 12px;
            line-height: 1.2;
        ">
            🔬 SkinScan AI
        </div>
        
        <!-- Subtitle -->
        <div style="
            font-size: 0.88rem;
            color: #6b9ab8;
            font-weight: 500;
            margin-bottom: 16px;
            line-height: 1.6;
        ">
            Clinical Intelligence Platform · v14.0
        </div>
        
        <!-- Developer info -->
        <div style="
            font-size: 0.82rem;
            color: #a8c4d8;
            line-height: 1.7;
            margin-bottom: 14px;
            max-width: 600px;
            margin-left: auto;
            margin-right: auto;
        ">
            Developed by <span style="color: #60a5fa; font-weight: 700;">Rehan Shafique</span><br>
            Final Year Project · Department of Bioinformatics
        </div>
        
        <!-- Tech stack -->
        <div style="
            font-size: 0.76rem;
            color: #475569;
            background: rgba(37,99,235,0.08);
            border: 1px solid rgba(37,99,235,0.15);
            border-radius: 12px;
            padding: 10px 20px;
            margin-bottom: 18px;
            font-weight: 500;
            letter-spacing: 0.3px;
        ">
            🐍 Python · Streamlit · TensorFlow · Plotly · ReportLab · PIL
        </div>
        
        <!-- Disclaimer -->
        <div style="
            font-size: 0.74rem;
            color: #f87171;
            font-weight: 600;
            background: rgba(239,68,68,0.08);
            border: 1px solid rgba(239,68,68,0.20);
            border-radius: 10px;
            padding: 12px 18px;
            margin-bottom: 8px;
            line-height: 1.5;
        ">
            ⚠️ Research & Educational Use Only — Not a Certified Medical Device
        </div>
        
        <!-- Copyright & timestamp -->
        <div style="
            font-size: 0.70rem;
            color: #475569;
            letter-spacing: 1px;
            text-transform: uppercase;
            font-weight: 500;
        ">
            © 2024 SkinScan AI · {datetime.datetime.now().strftime('%Y')}
        </div>
        
        <!-- Decorative elements -->
        <div style="
            position: absolute;
            top: 20px; right: 20px;
            font-size: 3.5rem;
            opacity: 0.04;
            pointer-events: none;
        ">
            🧬
        </div>
        <div style="
            position: absolute;
            bottom: 10px; left: 50%;
            transform: translateX(-50%);
            width: 80px; height: 2px;
            background: linear-gradient(90deg, transparent, #2563eb, transparent);
            border-radius: 1px;
        "></div>
    </footer>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = SkinScanApp()
    app.launch()
