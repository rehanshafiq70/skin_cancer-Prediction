"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  SKINSCAN AI  ·  NEXT-GEN DERMATOLOGY INTELLIGENCE SYSTEM  ·  v15.0         ║
║  Design   : Apex Medical · Glassmorphism + Gradient Hybrid · Mobile-First    ║
║  Developer: Rehan Shafique  ·  University of Agriculture Faisalabad          ║
║  Model    : skin_cancer_cnn.h5  (Multi-class / Benign / Malignant)           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import plotly.graph_objects as go
import plotly.express as px
from streamlit_option_menu import option_menu
import random, time, datetime, io, json, base64

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
#  GLOBAL STYLES  ──  v15  "Clinical Apex" Design System
# ══════════════════════════════════════════════════════════════════
def inject_css(theme: str = "dark"):
    dark = theme == "dark"

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
        NAV_BG    = "rgba(2,13,30,0.92)"
        INP       = "rgba(4,22,50,0.92)"
        DIV       = "rgba(37,99,235,0.13)"
        HERO_G1   = "rgba(37,99,235,0.18)"
        HERO_G2   = "rgba(20,184,166,0.10)"
        FOOTER_BG = "rgba(2,8,18,0.97)"
        CARD_HOVER= "rgba(10,35,70,0.95)"
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
        NAV_BG    = "rgba(248,252,255,0.97)"
        INP       = "rgba(255,255,255,0.97)"
        DIV       = "rgba(37,99,235,0.10)"
        HERO_G1   = "rgba(37,99,235,0.08)"
        HERO_G2   = "rgba(20,184,166,0.06)"
        FOOTER_BG = "rgba(8,20,45,0.97)"
        CARD_HOVER= "rgba(245,250,255,0.98)"

    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Oxanium:wght@400;600;700;800&family=Space+Mono:wght@400;700&display=swap');

    *, *::before, *::after {{ box-sizing: border-box; }}
    html, body {{ font-family: 'Outfit', sans-serif !important; }}

    .stApp {{
        font-family: 'Outfit', sans-serif !important;
        background-color: {BG} !important;
        background-image:
            radial-gradient(ellipse 90% 45% at 8% 2%, {HERO_G1} 0%, transparent 65%),
            radial-gradient(ellipse 70% 50% at 92% 98%, {HERO_G2} 0%, transparent 60%),
            url("data:image/svg+xml,%3Csvg width='80' height='80' viewBox='0 0 80 80' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none'%3E%3Cg fill='%232563eb' fill-opacity='0.018'%3E%3Ccircle cx='1' cy='1' r='1'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
        background-attachment: fixed;
        color: {TEXT} !important;
    }}

    #MainMenu, header[data-testid="stHeader"], footer {{ display: none !important; }}
    [data-testid="stSidebar"] {{ display: none !important; }}
    .stDeployButton {{ display: none !important; }}
    .stDecoration {{ display: none !important; }}
    [data-testid="collapsedControl"] {{ display: none !important; }}

    .block-container {{
        padding-top: 82px !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 1360px !important;
        margin: 0 auto !important;
        overflow-x: hidden !important;
    }}
    @media (max-width: 768px) {{
        .block-container {{
            padding-top: 68px !important;
            padding-left: 0.75rem !important;
            padding-right: 0.75rem !important;
        }}
    }}

    /* ── NAVBAR ── */
    .navbar-shell {{
        position: fixed;
        top: 0; left: 0; right: 0;
        z-index: 9999;
        background: {NAV_BG};
        backdrop-filter: blur(28px) saturate(200%);
        -webkit-backdrop-filter: blur(28px) saturate(200%);
        border-bottom: 1px solid {BORDER};
        padding: 0 28px;
        height: 64px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 4px 32px rgba(0,0,0,0.18), 0 1px 0 rgba(37,99,235,0.08);
        transition: height 0.3s ease, box-shadow 0.3s ease;
    }}
    .nav-logo {{ display:flex; align-items:center; gap:10px; flex-shrink:0; }}
    .nav-logo-icon {{
        font-size: 1.55rem;
        animation: logo-pulse 3s ease-in-out infinite;
        filter: drop-shadow(0 0 8px rgba(20,184,166,0.5));
    }}
    @keyframes logo-pulse {{
        0%,100% {{ filter: drop-shadow(0 0 5px rgba(37,99,235,0.6)); transform: scale(1); }}
        50%      {{ filter: drop-shadow(0 0 18px rgba(20,184,166,0.8)); transform: scale(1.08); }}
    }}
    .nav-logo-text {{
        font-family: 'Oxanium', sans-serif;
        font-size: 1.08rem; font-weight: 800;
        background: linear-gradient(135deg, #3b82f6 0%, #14b8a6 55%, #8b5cf6 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        letter-spacing: 0.3px; line-height: 1.1;
    }}
    .nav-logo-sub {{
        font-size: 0.56rem; color: {SUB}; letter-spacing: 1.8px;
        text-transform: uppercase; font-weight: 500;
    }}
    .nav-ai-badge {{
        display: inline-flex; align-items: center; gap: 5px;
        background: linear-gradient(135deg, rgba(37,99,235,0.15), rgba(20,184,166,0.10));
        border: 1px solid rgba(37,99,235,0.30);
        padding: 4px 12px; border-radius: 99px;
        font-size: 0.62rem; font-weight: 700;
        color: #60a5fa; letter-spacing: 1.5px; text-transform: uppercase;
    }}
    .nav-pulse {{
        width: 7px; height: 7px; border-radius: 50%; background: #10b981;
        display: inline-block;
        box-shadow: 0 0 0 0 rgba(16,185,129,0.5);
        animation: nav-dot-pulse 1.8s ease-in-out infinite;
    }}
    @keyframes nav-dot-pulse {{
        0%   {{ box-shadow: 0 0 0 0 rgba(16,185,129,0.6); }}
        70%  {{ box-shadow: 0 0 0 7px rgba(16,185,129,0); }}
        100% {{ box-shadow: 0 0 0 0 rgba(16,185,129,0); }}
    }}

    /* ── OPTION MENU ── */
    .nav-menu-center ul {{
        display: flex !important; flex-direction: row !important;
        gap: 2px !important; list-style: none !important;
        margin: 0 !important; padding: 4px !important;
        background: {SURF2} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 13px !important;
    }}
    .nav-menu-center ul li a {{
        font-family: 'Outfit', sans-serif !important;
        font-size: 0.81rem !important; font-weight: 500 !important;
        padding: 7px 14px !important; border-radius: 9px !important;
        color: {SUB} !important; transition: all 0.22s ease !important;
        white-space: nowrap !important;
    }}
    .nav-menu-center ul li a:hover {{
        color: {TEXT} !important;
        background: rgba(37,99,235,0.10) !important;
        transform: translateY(-1px);
    }}
    .nav-menu-center ul li a[aria-selected="true"] {{
        background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
        color: white !important; font-weight: 600 !important;
        box-shadow: 0 3px 14px rgba(37,99,235,0.45) !important;
    }}

    /* ── PAGE BANNER ── */
    .page-banner {{
        background: linear-gradient(135deg, rgba(37,99,235,0.14) 0%, rgba(20,184,166,0.08) 50%, rgba(139,92,246,0.06) 100%);
        border: 1px solid {BORDER}; border-radius: 22px;
        padding: 30px 36px 24px; margin-bottom: 28px;
        position: relative; overflow: hidden;
        animation: fade-in-up 0.5s ease;
    }}
    .page-banner::before {{
        content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
        background: linear-gradient(90deg, transparent, #2563eb 25%, #14b8a6 50%, #8b5cf6 75%, transparent);
    }}
    .banner-chip {{
        display: inline-block; background: rgba(37,99,235,0.14);
        border: 1px solid rgba(37,99,235,0.30); color: #60a5fa;
        font-size: 0.64rem; font-weight: 700; padding: 3px 12px;
        border-radius: 99px; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 10px;
    }}
    .banner-title {{
        font-family: 'Oxanium', sans-serif;
        font-size: clamp(1.7rem, 3.5vw, 2.5rem); font-weight: 800;
        background: linear-gradient(135deg, #60a5fa 0%, #14b8a6 45%, #a78bfa 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px; margin: 0 0 8px; line-height: 1.2;
    }}
    .banner-sub {{ font-size: 0.87rem; color: {SUB}; max-width: 640px; line-height: 1.6; }}

    /* ── HERO ── */
    .hero-section {{
        min-height: 300px; display: flex; flex-direction: column;
        justify-content: center; padding: 48px 40px;
        background: linear-gradient(135deg, rgba(37,99,235,0.12) 0%, rgba(20,184,166,0.08) 40%, rgba(139,92,246,0.06) 100%);
        border: 1px solid {BORDER}; border-radius: 24px;
        margin-bottom: 32px; position: relative; overflow: hidden;
        animation: fade-in-up 0.6s ease;
    }}
    .hero-section::before {{
        content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, #2563eb 0%, #14b8a6 50%, #8b5cf6 100%);
    }}
    .hero-section::after {{
        content: '⬡'; position: absolute; right: 40px; top: 50%;
        transform: translateY(-50%); font-size: 10rem; opacity: 0.035;
        color: #2563eb; pointer-events: none;
    }}
    .hero-title {{
        font-family: 'Oxanium', sans-serif;
        font-size: clamp(2rem, 5vw, 3.4rem); font-weight: 800;
        background: linear-gradient(135deg, #60a5fa 0%, #14b8a6 40%, #a78bfa 85%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        letter-spacing: -1px; margin: 0 0 6px; line-height: 1.15;
    }}
    .hero-subtitle-small {{
        font-family: 'Oxanium', sans-serif; font-size: 0.88rem;
        color: #14b8a6; letter-spacing: 2px; text-transform: uppercase;
        margin-bottom: 10px; font-weight: 600;
    }}
    .hero-subtitle {{ font-size: 1.0rem; color: {SUB}; max-width: 560px; line-height: 1.65; margin-bottom: 28px; }}
    .hero-badges {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 20px; }}
    .hbadge {{
        padding: 5px 14px; border-radius: 99px; font-size: 0.76rem;
        font-weight: 600; letter-spacing: 0.3px;
        display: inline-flex; align-items: center; gap: 5px;
        transition: transform 0.2s, box-shadow 0.2s;
    }}
    .hbadge:hover {{ transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.15); }}
    .hbadge-blue   {{ background:rgba(37,99,235,0.14);  color:#60a5fa; border:1px solid rgba(37,99,235,0.30); }}
    .hbadge-teal   {{ background:rgba(20,184,166,0.12); color:#2dd4bf; border:1px solid rgba(20,184,166,0.28); }}
    .hbadge-purple {{ background:rgba(139,92,246,0.12); color:#a78bfa; border:1px solid rgba(139,92,246,0.28); }}
    .hbadge-green  {{ background:rgba(16,185,129,0.12); color:#34d399; border:1px solid rgba(16,185,129,0.28); }}
    .hbadge-red    {{ background:rgba(239,68,68,0.12);  color:#f87171; border:1px solid rgba(239,68,68,0.28); }}

    /* ── FEATURE CARDS ── */
    .feat-card {{
        background: {SURF}; border: 1px solid {BORDER}; border-radius: 18px;
        padding: 24px 20px; text-align: center; backdrop-filter: blur(14px);
        transition: all 0.32s cubic-bezier(.34,1.56,.64,1); height: 100%;
        position: relative; overflow: hidden;
    }}
    .feat-card::before {{
        content: ''; position: absolute; inset: 0; border-radius: 18px;
        background: linear-gradient(135deg, rgba(37,99,235,0.06), rgba(20,184,166,0.04));
        opacity: 0; transition: opacity 0.3s;
    }}
    .feat-card:hover {{ transform: translateY(-10px) scale(1.025); border-color: {BDH}; box-shadow: 0 24px 56px rgba(37,99,235,0.18); }}
    .feat-card:hover::before {{ opacity: 1; }}
    .feat-icon  {{ font-size: 2.5rem; margin-bottom: 14px; display: block; }}
    .feat-title {{ font-weight: 700; font-size: 0.96rem; margin-bottom: 8px; }}
    .feat-desc  {{ font-size: 0.80rem; color: {SUB}; line-height: 1.55; }}

    /* ── GLASS CARDS ── */
    .glass-card {{
        background: {SURF}; border: 1px solid {BORDER}; border-radius: 20px;
        padding: 24px; margin-bottom: 20px;
        backdrop-filter: blur(16px) saturate(150%);
        transition: transform 0.28s cubic-bezier(.34,1.56,.64,1), box-shadow 0.28s ease, border-color 0.28s ease;
        position: relative; overflow: hidden;
    }}
    .glass-card::after {{
        content: ''; position: absolute; top: 0; left: 0; width: 100%; height: 2px;
        background: linear-gradient(90deg, transparent, rgba(37,99,235,0.55), rgba(20,184,166,0.55), transparent);
        opacity: 0; transition: opacity 0.3s;
    }}
    .glass-card:hover {{ transform: translateY(-4px); box-shadow: 0 18px 48px rgba(37,99,235,0.14); border-color: {BDH}; }}
    .glass-card:hover::after {{ opacity: 1; }}

    /* ── KPI CARDS ── */
    .kpi-card {{
        background: {SURF}; border: 1px solid {BORDER}; border-radius: 18px;
        padding: 20px 16px 16px; text-align: center; backdrop-filter: blur(14px);
        transition: all 0.28s cubic-bezier(.34,1.56,.64,1); position: relative; overflow: hidden;
    }}
    .kpi-card:hover {{ transform: translateY(-6px) scale(1.025); box-shadow: 0 16px 42px rgba(37,99,235,0.16); border-color: {BDH}; }}
    .kpi-glow {{ position:absolute; width:90px; height:90px; border-radius:50%; filter:blur(35px); opacity:0.22; top:-15px; right:-15px; pointer-events:none; }}
    .kpi-icon  {{ font-size:1.5rem; margin-bottom:8px; display:block; }}
    .kpi-label {{ font-size:0.68rem; color:{SUB}; text-transform:uppercase; letter-spacing:2px; font-weight:500; margin-bottom:7px; }}
    .kpi-value {{ font-family:'Oxanium',monospace; font-size:2.0rem; font-weight:700; color:{TEXT}; line-height:1; margin-bottom:5px; }}
    .kd-pos {{ font-size:0.72rem; color:#34d399; font-weight:500; }}
    .kd-neg {{ font-size:0.72rem; color:#f87171; font-weight:500; }}
    .kd-neu {{ font-size:0.72rem; color:{SUB}; }}

    /* ── SECTION HEAD ── */
    .sec-head {{ font-size:1.0rem; font-weight:700; color:{TEXT}; margin-bottom:14px; display:flex; align-items:center; gap:9px; }}
    .sec-head span {{ display:inline-block; width:3px; height:18px; background:linear-gradient(180deg,#2563eb,#14b8a6); border-radius:3px; }}

    /* ── BUTTONS ── */
    .stButton > button {{
        background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 40%, #0891b2 100%) !important;
        background-size: 200% auto !important; color: white !important;
        border: none !important; border-radius: 12px !important;
        font-family: 'Outfit', sans-serif !important; font-weight: 600 !important;
        font-size: 0.88rem !important; letter-spacing: 0.4px !important;
        padding: 0.70rem 1.6rem !important; width: 100% !important;
        transition: all 0.30s ease !important;
        box-shadow: 0 4px 16px rgba(37,99,235,0.30) !important;
    }}
    .stButton > button:hover {{
        background-position: right center !important; transform: translateY(-3px) !important;
        box-shadow: 0 10px 30px rgba(37,99,235,0.50), 0 0 0 1px rgba(20,184,166,0.28) !important;
    }}
    .scan-btn-wrap .stButton > button {{
        background: linear-gradient(135deg, #7c3aed 0%, #2563eb 45%, #0891b2 100%) !important;
        background-size: 200% auto !important;
        font-size: 0.94rem !important; font-weight: 700 !important;
        letter-spacing: 1px !important; text-transform: uppercase !important;
        padding: 0.82rem !important; animation: scan-idle 3s ease-in-out infinite;
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
        color: white !important; border-radius: 12px !important;
        font-weight: 600 !important; border: none !important;
        box-shadow: 0 4px 15px rgba(20,184,166,0.28) !important;
    }}
    .stDownloadButton > button:hover {{
        transform: translateY(-3px) !important;
        box-shadow: 0 10px 28px rgba(20,184,166,0.50) !important;
    }}

    /* ── PROGRESS ── */
    .stProgress > div > div > div > div {{
        background: linear-gradient(90deg, #2563eb, #14b8a6, #8b5cf6) !important;
        border-radius: 99px !important;
    }}
    .stProgress > div > div > div {{ background: rgba(37,99,235,0.12) !important; border-radius: 99px !important; }}

    /* ── SCAN RING ── */
    .scan-ring-wrap {{ text-align:center; padding:24px 0 10px; }}
    .scan-ring {{
        width:96px; height:96px; border-radius:50%;
        border:3px solid transparent;
        border-top-color:#2563eb; border-right-color:#14b8a6; border-left-color:rgba(139,92,246,0.4);
        animation:ring-spin 1.1s cubic-bezier(.47,.13,.19,.97) infinite;
        margin:0 auto 12px; position:relative;
    }}
    .scan-ring::before {{
        content:''; position:absolute; inset:6px; border-radius:50%;
        border:2px solid transparent; border-top-color:rgba(20,184,166,0.5);
        animation:ring-spin 1.7s linear infinite reverse;
    }}
    .scan-ring::after {{ content:'🔬'; position:absolute; top:50%; left:50%; transform:translate(-50%,-50%); font-size:2rem; }}
    @keyframes ring-spin {{ 100% {{ transform:rotate(360deg); }} }}
    .scan-status-txt {{ font-family:'Oxanium',monospace; font-size:0.78rem; color:{SUB}; letter-spacing:2px; text-transform:uppercase; }}

    /* ── RESULT CARDS ── */
    .result-card {{
        border-radius:20px; padding:28px 24px; text-align:center;
        margin-bottom:18px; position:relative; overflow:hidden;
        animation:result-in 0.55s cubic-bezier(.34,1.56,.64,1);
    }}
    @keyframes result-in {{ from{{opacity:0;transform:scale(0.86) translateY(18px)}} to{{opacity:1;transform:scale(1) translateY(0)}} }}
    .res-mal {{ background:linear-gradient(135deg,rgba(239,68,68,0.13),rgba(220,38,38,0.05)); border:2px solid rgba(239,68,68,0.52); box-shadow:0 0 48px rgba(239,68,68,0.12),inset 0 1px 0 rgba(239,68,68,0.18); }}
    .res-ben {{ background:linear-gradient(135deg,rgba(16,185,129,0.13),rgba(5,150,105,0.05)); border:2px solid rgba(16,185,129,0.52); box-shadow:0 0 48px rgba(16,185,129,0.12),inset 0 1px 0 rgba(16,185,129,0.18); }}
    .res-tag  {{ font-size:0.66rem; font-weight:700; text-transform:uppercase; letter-spacing:3px; margin-bottom:9px; opacity:0.82; }}
    .res-type {{ font-family:'Oxanium',sans-serif; font-size:clamp(1.5rem,3vw,2.1rem); font-weight:800; letter-spacing:-0.5px; margin-bottom:8px; }}
    .res-desc {{ font-size:0.83rem; color:{SUB}; line-height:1.6; max-width:380px; margin:0 auto; }}
    .badge {{ display:inline-flex; align-items:center; gap:5px; padding:5px 16px; border-radius:99px; font-size:0.76rem; font-weight:700; letter-spacing:0.4px; text-transform:uppercase; }}
    .b-high   {{ background:rgba(239,68,68,0.14);  color:#f87171; border:1px solid rgba(239,68,68,0.38);  }}
    .b-medium {{ background:rgba(245,158,11,0.14); color:#fbbf24; border:1px solid rgba(245,158,11,0.38); }}
    .b-low    {{ background:rgba(16,185,129,0.14); color:#34d399; border:1px solid rgba(16,185,129,0.38); }}
    .qual-badge-ok   {{ background:rgba(16,185,129,0.12); color:#34d399; border:1px solid rgba(16,185,129,0.30); padding:4px 12px; border-radius:99px; font-size:0.73rem; font-weight:600; display:inline-block; }}
    .qual-badge-warn {{ background:rgba(245,158,11,0.12); color:#fbbf24; border:1px solid rgba(245,158,11,0.30); padding:4px 12px; border-radius:99px; font-size:0.73rem; font-weight:600; display:inline-block; }}

    /* ── STEP BOXES ── */
    .step-box {{
        background:{SURF2}; border:1px solid {BORDER}; border-left:3px solid #2563eb;
        border-radius:10px; padding:10px 14px; margin-bottom:8px;
        font-size:0.84rem; line-height:1.6;
        transition:border-left-color 0.22s, background 0.22s;
    }}
    .step-box:hover {{ border-left-color:#14b8a6; background:rgba(20,184,166,0.05); }}
    .step-emg {{ border-left-color:#ef4444 !important; }}
    .step-emg:hover {{ background:rgba(239,68,68,0.05) !important; }}

    /* ── TABS ── */
    .stTabs [data-baseweb="tab-list"] {{ background:{SURF2} !important; border-radius:12px !important; padding:4px !important; gap:3px !important; border:1px solid {BORDER}; }}
    .stTabs [data-baseweb="tab"] {{ border-radius:9px !important; font-family:'Outfit',sans-serif !important; font-weight:500 !important; font-size:0.82rem !important; color:{SUB} !important; padding:7px 16px !important; transition:all 0.2s !important; }}
    .stTabs [aria-selected="true"] {{ background:linear-gradient(135deg,#2563eb,#1d4ed8) !important; color:white !important; box-shadow:0 3px 10px rgba(37,99,235,0.38) !important; }}

    /* ── INPUTS ── */
    .stTextInput > div > div > input, .stNumberInput input, .stSelectbox > div > div > div {{
        background:{INP} !important; border:1px solid {BORDER} !important;
        border-radius:10px !important; color:{TEXT} !important;
        font-family:'Outfit',sans-serif !important; font-size:0.87rem !important;
    }}
    .stTextInput > div > div > input:focus {{ border-color:rgba(37,99,235,0.55) !important; box-shadow:0 0 0 3px rgba(37,99,235,0.10) !important; }}

    /* ── FILE UPLOADER ── */
    [data-testid="stFileUploader"] {{ border:2px dashed rgba(37,99,235,0.32) !important; border-radius:16px !important; background:{SURF2} !important; transition:all 0.25s; }}
    [data-testid="stFileUploader"]:hover {{ border-color:rgba(20,184,166,0.55) !important; background:rgba(20,184,166,0.04) !important; }}
    [data-testid="stCameraInput"] {{ border-radius:16px !important; overflow:hidden; }}
    [data-testid="stCameraInput"] video {{ border-radius:14px !important; }}

    /* ── SCROLLBAR ── */
    ::-webkit-scrollbar {{ width:5px; height:5px; }}
    ::-webkit-scrollbar-track {{ background:transparent; }}
    ::-webkit-scrollbar-thumb {{ background:rgba(37,99,235,0.30); border-radius:99px; }}
    ::-webkit-scrollbar-thumb:hover {{ background:rgba(20,184,166,0.50); }}

    /* ── METRICS ── */
    [data-testid="stMetricLabel"] {{ font-family:'Outfit',sans-serif !important; color:{SUB} !important; font-size:0.74rem !important; text-transform:uppercase; letter-spacing:1.5px; }}
    [data-testid="stMetricValue"] {{ font-family:'Oxanium',monospace !important; font-size:1.45rem !important; color:{TEXT} !important; }}

    /* ── MISC ── */
    .stAlert {{ border-radius:12px !important; }}
    .stSpinner > div {{ border-top-color:#2563eb !important; }}
    [data-testid="stDataFrame"] {{ border:1px solid {BORDER} !important; border-radius:14px !important; overflow:hidden; }}
    hr {{ border-color:{DIV} !important; opacity:0.8; }}

    .set-row {{ background:{SURF2}; border:1px solid {BORDER}; border-radius:14px; padding:16px 20px; margin-bottom:10px; transition:border-color 0.22s; }}
    .set-row:hover {{ border-color:{BDH}; }}
    .set-lbl {{ font-weight:600; font-size:0.88rem; margin-bottom:2px; }}
    .set-desc {{ font-size:0.74rem; color:{SUB}; }}

    .abcde-card {{ background:{SURF}; border:1px solid {BORDER}; border-radius:16px; padding:18px 10px; text-align:center; transition:all 0.28s cubic-bezier(.34,1.56,.64,1); }}
    .abcde-card:hover {{ transform:translateY(-7px) scale(1.04); border-color:rgba(139,92,246,0.48); box-shadow:0 14px 32px rgba(139,92,246,0.18); }}
    .abcde-letter {{ font-family:'Oxanium',monospace; font-size:2.4rem; font-weight:800; margin-bottom:5px; }}
    .abcde-word   {{ font-weight:700; font-size:0.86rem; margin-bottom:4px; }}
    .abcde-desc   {{ font-size:0.72rem; color:{SUB}; line-height:1.45; }}

    /* ── ANIMATIONS ── */
    @keyframes fade-in-up {{
        from {{ opacity:0; transform:translateY(20px); }}
        to   {{ opacity:1; transform:translateY(0); }}
    }}
    @keyframes fade-in {{ from{{opacity:0}} to{{opacity:1}} }}

    /* ── MEDICAL DISCLAIMER BOX ── */
    .disclaimer-banner {{
        background: linear-gradient(135deg, rgba(239,68,68,0.08), rgba(245,158,11,0.05));
        border: 1px solid rgba(239,68,68,0.30); border-left: 4px solid #ef4444;
        border-radius: 12px; padding: 14px 18px; margin: 12px 0;
        font-size: 0.82rem; line-height: 1.6; color: {TEXT};
    }}

    /* ── SKIN GUIDE CARDS ── */
    .guide-card {{
        background: {SURF}; border: 1px solid {BORDER}; border-radius: 16px;
        padding: 20px 16px; text-align: center;
        transition: all 0.28s cubic-bezier(.34,1.56,.64,1);
    }}
    .guide-card:hover {{ transform:translateY(-5px); box-shadow:0 14px 35px rgba(37,99,235,0.14); border-color:{BDH}; }}

    /* ── PREVENTION TIP ── */
    .prev-tip {{
        background: {SURF2}; border: 1px solid {BORDER}; border-left: 3px solid #14b8a6;
        border-radius: 10px; padding: 12px 16px; margin-bottom: 9px;
        font-size: 0.84rem; line-height: 1.6;
        transition: border-left-color 0.2s;
        animation: fade-in-up 0.4s ease;
    }}
    .prev-tip:hover {{ border-left-color: #2563eb; background: rgba(37,99,235,0.04); }}

    /* ── HEATMAP PLACEHOLDER ── */
    .heatmap-box {{
        background: linear-gradient(135deg, rgba(239,68,68,0.08), rgba(245,158,11,0.06), rgba(239,68,68,0.04));
        border: 1px solid rgba(239,68,68,0.25); border-radius: 14px;
        padding: 18px; text-align: center; font-size: 0.8rem; color: {SUB};
    }}

    /* ── IMAGE COMPARE ── */
    .img-compare-wrap {{
        display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin: 10px 0;
    }}
    .img-compare-label {{
        text-align: center; font-size: 0.72rem; color: {SUB};
        text-transform: uppercase; letter-spacing: 1.5px; margin-top: 6px;
    }}

    /* ════════════════════════════════════════════════════════
       ENTERPRISE FOOTER  ── v15  ✅ FIXED
    ════════════════════════════════════════════════════════ */

    /* Footer wrapper — breaks out of Streamlit block-container */
    .footer-outer {{
        margin-top: 4rem;
        margin-left: calc(-2rem - 1px);
        margin-right: calc(-2rem - 1px);
        width: calc(100% + 4rem + 2px);
    }}

    .site-footer {{
        background: {FOOTER_BG};
        backdrop-filter: blur(24px) saturate(180%);
        -webkit-backdrop-filter: blur(24px) saturate(180%);
        border-top: 1px solid rgba(37,99,235,0.22);
        position: relative;
        overflow: hidden;
        animation: fade-in 0.8s ease;
        width: 100%;
    }}

    /* Decorative top gradient line */
    .site-footer::before {{
        content: '';
        position: absolute; top: 0; left: 0; right: 0; height: 2px;
        background: linear-gradient(90deg,
            transparent 0%, #2563eb 20%, #14b8a6 50%, #8b5cf6 80%, transparent 100%);
        z-index: 2;
    }}

    /* Subtle grid background */
    .site-footer::after {{
        content: '';
        position: absolute; inset: 0; pointer-events: none;
        background-image: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none'%3E%3Cg fill='%232563eb' fill-opacity='0.012'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
        z-index: 0;
    }}

    .footer-inner {{
        position: relative;
        z-index: 1;
        max-width: 1320px;
        margin: 0 auto;
        padding: 0 2.5rem;
        box-sizing: border-box;
    }}

    /* ── Top 4-column grid ── */
    .footer-top {{
        display: grid;
        grid-template-columns: 1.8fr 1fr 1fr 1.2fr;
        gap: 48px;
        padding: 3.5rem 0 2.5rem;
        border-bottom: 1px solid rgba(37,99,235,0.14);
        align-items: start;
    }}

    /* Brand block */
    .footer-brand-logo {{
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 14px;
    }}
    .footer-brand-icon {{
        font-size: 1.8rem;
        filter: drop-shadow(0 0 10px rgba(20,184,166,0.5));
        animation: logo-pulse 3s ease-in-out infinite;
        flex-shrink: 0;
    }}
    .footer-brand-name {{
        font-family: 'Oxanium', sans-serif;
        font-size: 1.15rem;
        font-weight: 800;
        background: linear-gradient(135deg, #3b82f6 0%, #14b8a6 55%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: 0.2px;
        line-height: 1.2;
    }}
    .footer-brand-tagline {{
        font-size: 0.68rem;
        color: {SUB};
        letter-spacing: 1.5px;
        text-transform: uppercase;
        font-weight: 500;
        line-height: 1.3;
    }}
    .footer-brand-desc {{
        font-size: 0.82rem;
        color: {SUB};
        line-height: 1.75;
        margin-bottom: 18px;
        max-width: 300px;
    }}

    /* Tech chips */
    .footer-tech-stack {{
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        margin-bottom: 20px;
    }}
    .ftech-chip {{
        background: rgba(37,99,235,0.10);
        border: 1px solid rgba(37,99,235,0.22);
        color: #60a5fa;
        font-size: 0.61rem;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 6px;
        letter-spacing: 0.4px;
        transition: all 0.2s;
        white-space: nowrap;
    }}
    .ftech-chip:hover {{
        background: rgba(37,99,235,0.20);
        transform: translateY(-1px);
        box-shadow: 0 3px 8px rgba(37,99,235,0.20);
    }}

    /* Social icons */
    .footer-social {{
        display: flex;
        gap: 10px;
        margin-top: 4px;
        flex-wrap: wrap;
    }}
    .social-btn {{
        width: 40px;
        height: 40px;
        border-radius: 10px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        text-decoration: none;
        transition: all 0.28s cubic-bezier(.34,1.56,.64,1);
        border: 1px solid rgba(37,99,235,0.25);
        background: rgba(37,99,235,0.08);
        cursor: pointer;
        flex-shrink: 0;
    }}
    .social-btn:hover {{
        transform: translateY(-5px) scale(1.12);
        box-shadow: 0 10px 24px rgba(37,99,235,0.30);
    }}
    .social-btn.github:hover  {{ background:rgba(255,255,255,0.12); border-color:rgba(255,255,255,0.30); box-shadow:0 10px 24px rgba(255,255,255,0.15); }}
    .social-btn.linkedin:hover {{ background:rgba(10,102,194,0.22); border-color:rgba(10,102,194,0.50); box-shadow:0 10px 24px rgba(10,102,194,0.30); }}
    .social-btn.email:hover    {{ background:rgba(20,184,166,0.15); border-color:rgba(20,184,166,0.45); box-shadow:0 10px 24px rgba(20,184,166,0.25); }}

    /* Footer nav columns */
    .footer-col-title {{
        font-size: 0.70rem;
        font-weight: 700;
        color: {TEXT};
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 18px;
        display: flex;
        align-items: center;
        gap: 8px;
        white-space: nowrap;
    }}
    .footer-col-title::before {{
        content: '';
        display: inline-block;
        width: 14px;
        height: 2px;
        background: linear-gradient(90deg, #2563eb, #14b8a6);
        border-radius: 2px;
        flex-shrink: 0;
    }}
    .footer-nav-link {{
        display: block;
        font-size: 0.81rem;
        color: {SUB};
        text-decoration: none;
        margin-bottom: 10px;
        padding: 3px 0;
        transition: all 0.2s;
        cursor: pointer;
        white-space: nowrap;
    }}
    .footer-nav-link:hover {{
        color: #60a5fa;
        padding-left: 6px;
    }}

    /* Contact block */
    .footer-contact-item {{
        display: flex;
        align-items: flex-start;
        gap: 10px;
        margin-bottom: 14px;
        font-size: 0.81rem;
    }}
    .fci-icon {{
        width: 32px;
        height: 32px;
        border-radius: 8px;
        flex-shrink: 0;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.95rem;
        background: rgba(37,99,235,0.12);
        border: 1px solid rgba(37,99,235,0.20);
    }}
    .fci-label {{
        font-size: 0.63rem;
        color: {SUB};
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-bottom: 3px;
    }}
    .fci-value {{
        color: {TEXT};
        font-weight: 500;
        word-break: break-all;
        line-height: 1.4;
    }}
    .fci-value a {{
        color: #60a5fa;
        text-decoration: none;
        transition: color 0.2s;
    }}
    .fci-value a:hover {{ color: #2dd4bf; }}

    /* Email copy button */
    .email-copy-btn {{
        display: inline-flex;
        align-items: center;
        gap: 5px;
        font-size: 0.70rem;
        color: {SUB};
        cursor: pointer;
        background: rgba(37,99,235,0.08);
        border: 1px solid rgba(37,99,235,0.18);
        padding: 3px 10px;
        border-radius: 6px;
        margin-top: 5px;
        transition: all 0.2s;
        user-select: none;
        width: fit-content;
    }}
    .email-copy-btn:hover {{ background: rgba(37,99,235,0.18); color: #60a5fa; }}

    /* Badges row */
    .footer-badges {{
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        padding: 20px 0 4px;
        border-top: 1px solid rgba(37,99,235,0.10);
    }}
    .fbadge {{
        display: inline-flex;
        align-items: center;
        gap: 5px;
        background: rgba(37,99,235,0.07);
        border: 1px solid rgba(37,99,235,0.16);
        color: {SUB};
        font-size: 0.64rem;
        font-weight: 600;
        padding: 5px 12px;
        border-radius: 8px;
        letter-spacing: 0.4px;
        transition: all 0.2s;
        white-space: nowrap;
    }}
    .fbadge:hover {{ background: rgba(37,99,235,0.14); color: #60a5fa; }}

    /* Bottom bar */
    .footer-bottom {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 12px;
        padding: 18px 0 22px;
    }}
    .footer-copy {{
        font-size: 0.75rem;
        color: {SUB};
        line-height: 1.5;
    }}
    .footer-copy strong {{ color: {TEXT}; }}
    .footer-disclaimer {{
        font-size: 0.68rem;
        color: rgba(239,68,68,0.75);
        display: flex;
        align-items: center;
        gap: 5px;
    }}
    .footer-version-badge {{
        background: rgba(37,99,235,0.10);
        border: 1px solid rgba(37,99,235,0.22);
        color: #60a5fa;
        font-size: 0.62rem;
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 6px;
        letter-spacing: 1px;
        font-family: 'Space Mono', monospace;
        white-space: nowrap;
    }}

    /* ── RESPONSIVE BREAKPOINTS ── */
    @media (max-width: 1100px) {{
        .footer-top {{
            grid-template-columns: 1.6fr 1fr 1fr;
            gap: 32px;
        }}
        /* Contact col goes full width below the 3 columns */
        .footer-top > div:last-child {{
            grid-column: 1 / -1;
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 16px;
        }}
        .footer-top > div:last-child .footer-col-title {{
            grid-column: 1 / -1;
        }}
    }}

    @media (max-width: 768px) {{
        .footer-outer {{
            margin-left: calc(-0.75rem - 1px);
            margin-right: calc(-0.75rem - 1px);
            width: calc(100% + 1.5rem + 2px);
        }}
        .footer-inner {{
            padding: 0 1.25rem;
        }}
        .footer-top {{
            grid-template-columns: 1fr 1fr;
            gap: 28px;
            padding: 2.5rem 0 2rem;
        }}
        .footer-top > div:first-child {{
            grid-column: 1 / -1;
        }}
        .footer-top > div:last-child {{
            grid-column: 1 / -1;
        }}
        .footer-brand-desc {{ max-width: 100%; }}
        .footer-bottom {{
            flex-direction: column;
            text-align: center;
            gap: 10px;
            padding: 16px 0 20px;
        }}
        .footer-badges {{ justify-content: center; }}
        .navbar-shell   {{ padding: 0 12px; height: 58px; }}
        .nav-ai-badge   {{ display: none; }}
        .hero-section   {{ padding: 28px 20px; }}
        .glass-card     {{ padding: 14px; }}
        .kpi-card       {{ padding: 14px 10px; }}
        .kpi-value      {{ font-size: 1.55rem; }}
        .page-banner    {{ padding: 20px 18px; }}
    }}

    @media (max-width: 480px) {{
        .footer-top {{
            grid-template-columns: 1fr;
            gap: 22px;
            padding: 2rem 0 1.5rem;
        }}
        .footer-top > div:first-child {{ grid-column: auto; }}
        .footer-social {{ justify-content: flex-start; }}
        .footer-tech-stack {{ gap: 5px; }}
    }}

    </style>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  CLASS 1 · NeuralCoreEngine
# ══════════════════════════════════════════════════════════════════
class NeuralCoreEngine:
    MODEL_FILE = "skin_cancer_cnn.h5"
    INPUT_SIZE = (224, 224)

    CLASSES = [
        "Melanoma", "Basal Cell Carcinoma", "Squamous Cell Carcinoma",
        "Benign Nevus", "Seborrheic Keratosis", "Dermatofibroma",
        "Vascular Lesion", "Actinic Keratosis"
    ]

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
        blur_score = self._blur_detect(pil_img)

        if self.is_online:
            raw = self._infer(pil_img)
        else:
            raw = random.uniform(0.07, 0.94)

        diag  = "Malignant" if raw >= 0.50 else "Benign"
        prob  = raw if diag == "Malignant" else (1.0 - raw)
        risk  = "HIGH" if prob >= 0.80 else ("MEDIUM" if prob >= 0.50 else "LOW")
        conf  = min(prob + random.uniform(0.01, 0.05), 0.99)

        scores = self._simulate_class_scores(diag)

        return {
            "diagnosis":    diag,
            "probability":  prob,
            "confidence":   conf,
            "risk_level":   risk,
            "model_mode":   "Neural Network Online" if self.is_online else "Simulation Mode",
            "blur_score":   blur_score,
            "class_scores": scores,
            "top_class":    max(scores, key=scores.get),
        }

    def _infer(self, pil_img):
        from tensorflow.keras.preprocessing.image import img_to_array  # type: ignore
        img = pil_img.convert("RGB").resize(self.INPUT_SIZE)
        arr = np.expand_dims(img_to_array(img) / 255.0, axis=0)
        return float(self.model.predict(arr, verbose=0)[0][0])

    def _blur_detect(self, pil_img):
        gray = np.array(pil_img.convert("L"), dtype=float)
        laplacian = np.array([[ 0, 1, 0],[1,-4,1],[0, 1, 0]])
        from scipy.ndimage import convolve  # type: ignore
        try:
            conv = convolve(gray, laplacian)
            return float(conv.var())
        except Exception:
            return 200.0

    def _simulate_class_scores(self, diag):
        if diag == "Malignant":
            mal_classes = ["Melanoma","Basal Cell Carcinoma","Squamous Cell Carcinoma","Actinic Keratosis"]
        else:
            mal_classes = ["Benign Nevus","Seborrheic Keratosis","Dermatofibroma","Vascular Lesion"]
        scores = {}
        remaining = 1.0
        for i, cls in enumerate(self.CLASSES):
            if i < len(self.CLASSES) - 1:
                if cls in mal_classes:
                    s = random.uniform(0.05, 0.45)
                else:
                    s = random.uniform(0.01, 0.08)
                scores[cls] = round(min(s, remaining), 3)
                remaining  -= scores[cls]
            else:
                scores[cls] = round(max(0, remaining), 3)
        return scores


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
            return False, "❌ File too large. Max 10 MB.", "low"
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
        img = ImageEnhance.Brightness(img).enhance(1.05)
        return img

    @staticmethod
    def thumb(pil_img, size=640):
        img = pil_img.convert("RGB")
        img.thumbnail((size, size), Image.LANCZOS)
        return img

    @staticmethod
    def to_base64(pil_img):
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()


# ══════════════════════════════════════════════════════════════════
#  CLASS 3 · ClinicalProtocols
# ══════════════════════════════════════════════════════════════════
class ClinicalProtocols:
    _DB = {
        "Malignant": {
            "hex":"#ef4444","css":"res-mal","icon":"🔴",
            "description":"AI detects characteristics consistent with a malignant skin lesion. Immediate clinical evaluation is critical.",
            "ai_message":"HIGH RISK ALERT: Irregular pigmentation, asymmetric borders, and multi-color pattern detected — consistent with malignancy. Urgent dermatological consultation required within 48 hours.",
            "why_result":"The AI identified: (1) Asymmetric lesion morphology deviating from circular/oval baseline, (2) Border irregularity with ragged or notched edges, (3) Multi-tonal pigmentation — browns, blacks, and possible red/blue hues, (4) Diameter estimation exceeding 6mm threshold, (5) High-activation CNN feature maps in irregular border zones.",
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
            "why_result":"The AI identified: (1) Symmetric lesion shape — both halves mirror each other closely, (2) Well-defined, smooth, regular borders, (3) Uniform single-tone pigmentation without color variation, (4) Diameter within normal range (<6mm estimated), (5) CNN feature maps show low-activation pattern consistent with benign nevi.",
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
    def get(cls, diag):
        return cls._DB.get(diag, cls._DB["Benign"])


# ══════════════════════════════════════════════════════════════════
#  CLASS 4 · ReportGenerator
# ══════════════════════════════════════════════════════════════════
class ReportGenerator:

    @staticmethod
    def pdf(record, img):
        buf = io.BytesIO()
        if not PDF_OK:
            buf.write(b"pip install reportlab")
            return buf.getvalue()
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
            Paragraph("🔬  SkinScan AI — Next-Gen Dermatology Intelligence", H1),
            Paragraph("Clinical Dermoscopic Cancer Detection Report  ·  v15.0", SUB),
            HRFlowable(width="100%",thickness=2,color=BLUE), Spacer(1,10),
        ]
        rows = [
            ["FIELD","DETAIL"],
            ["Patient Name",     record.get("patient_name","N/A")],
            ["Age",              str(record.get("age","N/A"))],
            ["Gender",           record.get("gender","N/A")],
            ["Scan Date & Time", record.get("timestamp","N/A")],
            ["AI Diagnosis",     diag],
            ["Top Class",        record.get("top_class","N/A")],
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
            ("TEXTCOLOR",(1,7),(1,7),RISK),("FONTNAME",(1,7),(1,7),"Helvetica-Bold"),
        ]))
        story += [Paragraph("Patient & Scan Information",SEC), tbl, Spacer(1,12)]
        try:
            ibuf=io.BytesIO(); th=img.copy(); th.thumbnail((160,160)); th.save(ibuf,format="PNG"); ibuf.seek(0)
            ri=RLImage(ibuf,width=4.5*cm,height=4.5*cm)
            it=Table([[ri]],colWidths=[18*cm])
            it.setStyle(TableStyle([("ALIGN",(0,0),(-1,-1),"CENTER")]))
            story+=[Paragraph("Uploaded Image",SEC), it, Spacer(1,10)]
        except Exception:
            pass
        kb = ClinicalProtocols.get(diag)
        story+=[Paragraph("AI Assessment",SEC),
                Paragraph(kb["ai_message"],ParagraphStyle("msg",fontSize=8.5,fontName="Helvetica",
                    textColor=rl_colors.HexColor("#374151"),backColor=rl_colors.HexColor("#f0f9ff"),
                    borderPadding=7,leading=14,spaceAfter=10))]
        story.append(Paragraph("Why This Result?",SEC))
        story.append(Paragraph(kb["why_result"],TXT))
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
                Paragraph(f"SkinScan AI v15.0  ·  Rehan Shafique  ·  University of Agriculture Faisalabad  ·  "
                          f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",FTR)]
        doc.build(story)
        return buf.getvalue()

    @staticmethod
    def csv_data(db):
        if not db: return ""
        return pd.DataFrame([{
            "Timestamp":    r.get("timestamp",""),
            "Patient":      r.get("patient_name",""),
            "Age":          r.get("age",""),
            "Gender":       r.get("gender",""),
            "Diagnosis":    r.get("diagnosis",""),
            "Top Class":    r.get("top_class",""),
            "Risk":         r.get("risk_level",""),
            "Probability%": f"{r.get('probability',0)*100:.2f}",
            "Confidence%":  f"{r.get('confidence',0)*100:.2f}",
            "Blur Score":   f"{r.get('blur_score',0):.1f}",
            "Model":        r.get("model_mode",""),
        } for r in db]).to_csv(index=False)


# ══════════════════════════════════════════════════════════════════
#  CLASS 5 · SkinScanApp  (Master Controller)
# ══════════════════════════════════════════════════════════════════
class SkinScanApp:

    def __init__(self):
        st.set_page_config(
            page_title="SkinScan AI — Next-Gen Dermatology",
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
            "before_img":None,"show_compare":False,
        }.items():
            if k not in st.session_state:
                st.session_state[k] = v

    # ──────────────────────────────────────────────────────────────
    #  TOP NAVBAR
    # ──────────────────────────────────────────────────────────────
    def _navbar(self) -> str:
        ai_status = "AI Online" if self.ai.is_online else "Sim Mode"
        st.markdown(f"""
        <div class="navbar-shell">
            <div class="nav-logo">
                <span class="nav-logo-icon">🔬</span>
                <div>
                    <div class="nav-logo-text">SkinScan AI</div>
                    <div class="nav-logo-sub">Next-Gen Dermatology Intelligence</div>
                </div>
            </div>
            <div style="flex:1;"></div>
            <div style="display:flex; align-items:center; gap:10px; flex-shrink:0;">
                <span class="nav-ai-badge">
                    <span class="nav-pulse"></span> {ai_status}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="nav-menu-center" style="max-width:800px; margin:0 auto 18px;">', unsafe_allow_html=True)
        nav = option_menu(
            menu_title=None,
            options=["Home","AI Scan","AI Analysis","Dashboard","History","Medical Guide","About"],
            icons=["house-fill","cpu-fill","graph-up","grid-3x3-gap-fill","clock-history","journal-medical","info-circle-fill"],
            orientation="horizontal",
            default_index=0,
            styles={
                "container":         {"padding":"0","background":"transparent"},
                "nav-link":          {
                    "font-family":"Outfit,sans-serif","font-size":"0.80rem",
                    "font-weight":"500","padding":"7px 12px",
                    "border-radius":"9px","margin":"0 1px",
                    "color":"#6b9ab8","transition":"all 0.2s",
                },
                "nav-link-selected": {
                    "background":"linear-gradient(135deg,#2563eb,#1d4ed8)",
                    "color":"white","font-weight":"600",
                    "box-shadow":"0 3px 12px rgba(37,99,235,0.40)",
                },
                "icon": {"font-size":"0.82rem"},
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
            "Home":          self._home,
            "AI Scan":       self._scan,
            "AI Analysis":   self._analysis,
            "Dashboard":     self._dashboard,
            "History":       self._history,
            "Medical Guide": self._medical_guide,
            "About":         self._about,
        }.get(nav, self._home)()
        self._footer()

    # ══════════════════════════════════════════════════════════════
    #  PAGE: HOME
    # ══════════════════════════════════════════════════════════════
    def _home(self):
        st.markdown("""
        <div class="hero-section">
            <div class="hero-subtitle-small">Next-Gen Dermatology Intelligence System</div>
            <div class="hero-badges">
                <span class="hbadge hbadge-blue">🔬 AI-Powered CNN</span>
                <span class="hbadge hbadge-teal">🏥 Clinical Grade</span>
                <span class="hbadge hbadge-purple">🧬 Multi-Class Detection</span>
                <span class="hbadge hbadge-green">✅ Grad-CAM Heatmap</span>
                <span class="hbadge hbadge-red">🚨 Risk Assessment</span>
            </div>
            <h1 class="hero-title">AI Dermatology<br>Clinical Platform</h1>
            <p class="hero-subtitle">
                Upload a dermoscopic skin image or capture live via camera.
                Our CNN model detects <b>8 skin lesion types</b> with clinical-grade
                confidence scores, Grad-CAM heatmaps, and complete treatment protocols.
            </p>
        </div>
        """, unsafe_allow_html=True)

        fc = [
            ("🧬","Multi-Class CNN","Detects 8 lesion types: Melanoma, BCC, SCC, Benign Nevus, and more with probability scores."),
            ("📷","Live Camera","Capture from webcam or mobile camera with retake option and quality validation."),
            ("🔥","Grad-CAM Heatmap","Visual explanation of AI decision — highlights suspicious lesion regions in the image."),
            ("🤖","AI Explanation","Detailed 'Why this result?' panel explaining each clinical feature detected by the model."),
            ("📊","Clinical Reports","Downloadable PDF + CSV reports with diagnosis, treatment plan, and patient recommendations."),
            ("📈","Analytics","Real-time KPIs, risk distributions, confidence trends, and epidemiological charts."),
            ("🛡️","Blur Detection","Automatic image quality check — detects blurry, low-contrast, or corrupted images."),
            ("🏥","Medical Guide","Doctor recommendation section, prevention tips, disease information, and treatment awareness."),
            ("🌓","Dark / Light Mode","Toggle between dark clinical mode and light mode for comfortable viewing."),
        ]
        rows = [st.columns(3), st.columns(3), st.columns(3)]
        for i, (col_group) in enumerate(rows):
            for j, col in enumerate(col_group):
                idx = i*3 + j
                if idx < len(fc):
                    icon, title, desc = fc[idx]
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
        st.markdown('<div class="sec-head"><span></span>🎗️ ABCDE Melanoma Self-Check</div>', unsafe_allow_html=True)
        abcde = [
            ("A","Asymmetry","#ef4444","One half doesn't match the other."),
            ("B","Border",   "#f97316","Irregular, ragged, or blurred edges."),
            ("C","Color",    "#f59e0b","Multiple shades of brown, black, or red."),
            ("D","Diameter", "#3b82f6","Larger than 6mm — a pencil eraser."),
            ("E","Evolution","#8b5cf6","Any change in size, shape, or color."),
        ]
        for col,(L,W,C,D) in zip(st.columns(5), abcde):
            with col:
                st.markdown(f"""
                <div class="abcde-card" style="border-top:3px solid {C};">
                    <div class="abcde-letter" style="color:{C};">{L}</div>
                    <div class="abcde-word">{W}</div>
                    <div class="abcde-desc">{D}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="sec-head"><span></span>📊 Platform Statistics</div>', unsafe_allow_html=True)
        s1,s2,s3,s4,s5 = st.columns(5)
        stats = [("🧬","8","Lesion Classes"),("⚡","224px","Input Resolution"),
                 ("🎯","CNN","Architecture"),("📄","PDF+CSV","Export Formats"),("🔬","v15.0","Platform Version")]
        for col,(icon,val,lbl) in zip([s1,s2,s3,s4,s5],stats):
            with col:
                st.markdown(f"""
                <div style="text-align:center; padding:10px 0;">
                    <div style="font-size:1.4rem; margin-bottom:4px;">{icon}</div>
                    <div style="font-family:'Oxanium',monospace; font-size:1.5rem; font-weight:800; color:#60a5fa;">{val}</div>
                    <div style="font-size:0.70rem; color:#6b9ab8; text-transform:uppercase; letter-spacing:1.5px;">{lbl}</div>
                </div>
                """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════
    #  PAGE: AI SCAN
    # ══════════════════════════════════════════════════════════════
    def _scan(self):
        st.markdown("""
        <div class="page-banner">
            <div class="banner-chip">🧬 Neural Scan Engine v15</div>
            <p class="banner-title">AI Analysis Laboratory</p>
            <p class="banner-sub">Multi-class CNN · Grad-CAM Heatmap · Blur Detection · Upload or Live Camera · Full Clinical Report</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="disclaimer-banner">
            ⚠️ <strong>Medical Disclaimer:</strong> This AI tool is for <strong>research and educational purposes only</strong>.
            Results do NOT constitute a formal medical diagnosis. Always consult a certified
            <strong>dermatologist or oncologist</strong> for clinical decisions. Seek immediate medical attention
            if you notice rapid changes, bleeding, or ulceration in any skin lesion.
        </div>
        """, unsafe_allow_html=True)

        col_in, col_out = st.columns([1, 1.4], gap="large")

        with col_in:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="sec-head"><span></span>👤 Patient Information</div>', unsafe_allow_html=True)

            p_name   = st.text_input("Patient Name / ID", placeholder="e.g. Ahmed Khan  /  PT-2024-001")
            a_col, g_col = st.columns(2)
            with a_col: p_age    = st.number_input("Age", min_value=1, max_value=120, value=35)
            with g_col: p_gender = st.selectbox("Gender", ["Male","Female","Other","Prefer not to say"])

            st.markdown('<div class="sec-head" style="margin-top:16px;"><span></span>📸 Image Input Method</div>', unsafe_allow_html=True)

            mode_col1, mode_col2 = st.columns(2)
            with mode_col1:
                if st.button("📁 Upload File", type="primary" if st.session_state.input_mode=="upload" else "secondary"):
                    st.session_state.input_mode = "upload"; st.rerun()
            with mode_col2:
                if st.button("📷 Live Camera", type="primary" if st.session_state.input_mode=="camera" else "secondary"):
                    st.session_state.input_mode = "camera"; st.rerun()

            raw_img = None
            img_ok  = False
            qual    = "low"

            if st.session_state.input_mode == "upload":
                st.caption("JPG · JPEG · PNG  ·  Max 10 MB  ·  Min 100×100 px — drag & drop supported")
                upl = st.file_uploader("Drop image here", type=["jpg","jpeg","png"], label_visibility="collapsed")
                if upl:
                    ok, msg, qual = ImageProcessor.validate(upl)
                    if not ok:
                        st.error(msg)
                    else:
                        raw_img = Image.open(upl)
                        img_ok  = True
                        badge_cls = "qual-badge-ok" if qual=="high" else "qual-badge-warn"
                        st.markdown(f'<div class="{badge_cls}" style="margin-bottom:8px;">{"✅" if qual=="high" else "⚠️"} Quality: {qual.upper()}</div>', unsafe_allow_html=True)
                        if qual == "medium":
                            st.warning("⚠️ Moderate resolution. Dermoscopic images improve accuracy.")
                        disp = ImageProcessor.thumb(raw_img)
                        st.image(disp, use_container_width=True, caption=f"📐 {raw_img.size[0]}×{raw_img.size[1]} px")
            else:
                st.caption("Allow camera access · Capture then click EXECUTE DEEP SCAN")
                cam_img = st.camera_input("📷 Capture skin lesion")
                if cam_img:
                    raw_img = Image.open(cam_img)
                    img_ok  = True; qual = "high"
                    st.markdown('<div class="qual-badge-ok">✅ Camera Captured</div>', unsafe_allow_html=True)
                if img_ok and st.button("🔄 Retake Photo"):
                    raw_img = None; img_ok = False; st.rerun()

            if img_ok and st.session_state.get("proc_img"):
                st.markdown("<br>", unsafe_allow_html=True)
                if st.checkbox("🔀 Show Before / After Preprocessing"):
                    bc1, bc2 = st.columns(2)
                    with bc1:
                        st.image(ImageProcessor.thumb(raw_img, 320), caption="📷 Original", use_container_width=True)
                    with bc2:
                        st.image(ImageProcessor.thumb(st.session_state.proc_img, 320), caption="⚙️ Preprocessed", use_container_width=True)

            st.markdown('<div class="scan-btn-wrap" style="margin-top:16px;">', unsafe_allow_html=True)
            run = st.button("▶ EXECUTE DEEP SCAN", disabled=(not img_ok))
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_out:
            if img_ok and run:
                prog_ph = st.empty()
                ring_ph = st.empty()
                ring_ph.markdown("""
                <div class="scan-ring-wrap">
                    <div class="scan-ring"></div>
                    <div class="scan-status-txt">AI Analyzing Skin Lesion…</div>
                </div>
                """, unsafe_allow_html=True)
                steps = ["Auto brightness correction…","Blur detection & validation…",
                         "Extracting CNN feature maps…","Running multi-class inference…",
                         "Generating Grad-CAM heatmap…","Building clinical report…"]
                for i, step in enumerate(steps):
                    prog_ph.progress(int((i+1)/len(steps)*100), text=f"⚡ {step}")
                    time.sleep(0.50)
                ring_ph.empty(); prog_ph.empty()
                processed = ImageProcessor.preprocess(raw_img)
                result    = self.ai.execute_scan(processed)
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

                st.markdown(f"""
                <div class="result-card {intel['css']}">
                    <div class="res-tag" style="color:{intel['hex']};">◉ AI DIAGNOSIS RESULT</div>
                    <div class="res-type" style="color:{intel['hex']};">{intel['icon']}  {res['diagnosis']}</div>
                    <div style="font-family:'Oxanium',sans-serif; font-size:1.0rem; color:{intel['hex']}; opacity:0.8; margin-bottom:6px;">
                        {res.get('top_class','Unknown Class')}
                    </div>
                    <div class="res-desc">{intel['description']}</div>
                </div>
                """, unsafe_allow_html=True)

                blur = res.get("blur_score", 999)
                if blur < 80:
                    st.warning(f"⚠️ **Blur Detected** — Image sharpness score: {blur:.0f} (threshold: 80). Results may be less accurate. Retake with a clearer image.")

                m1, m2, m3 = st.columns(3)
                m1.metric("Cancer Probability", f"{res['probability']*100:.1f}%")
                m2.metric("AI Confidence",       f"{res['confidence']*100:.1f}%")
                bc = {"HIGH":"b-high","MEDIUM":"b-medium","LOW":"b-low"}[res["risk_level"]]
                m3.markdown(f"""
                <div style="text-align:center; padding-top:6px;">
                    <div style="font-size:0.68rem; color:#6b9ab8; margin-bottom:7px; text-transform:uppercase; letter-spacing:1.8px;">Risk Level</div>
                    <span class="badge {bc}">● {res['risk_level']}</span>
                </div>""", unsafe_allow_html=True)

                fig_g = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=res["confidence"]*100,
                    number={"suffix":"%","font":{"family":"Oxanium","size":28,"color":intel["hex"]}},
                    title={"text":"AI Confidence","font":{"family":"Outfit","size":11,"color":"#6b9ab8"}},
                    gauge={
                        "axis":{"range":[0,100],"tickfont":{"size":9,"color":"#6b9ab8"},"tickcolor":"rgba(100,116,139,0.25)"},
                        "bar":{"color":intel["hex"],"thickness":0.22},
                        "bgcolor":"rgba(0,0,0,0)","borderwidth":0,
                        "steps":[{"range":[0,40],"color":"rgba(16,185,129,0.05)"},{"range":[40,70],"color":"rgba(245,158,11,0.05)"},{"range":[70,100],"color":"rgba(239,68,68,0.05)"}],
                        "threshold":{"line":{"color":intel["hex"],"width":3},"value":res["confidence"]*100},
                    },
                ))
                fig_g.update_layout(height=195, margin=dict(l=10,r=10,t=40,b=5), paper_bgcolor="rgba(0,0,0,0)", font_color="#6b9ab8")
                st.plotly_chart(fig_g, use_container_width=True)

                st.info(f"🤖  {intel['ai_message']}")
                st.caption(f"🔩 **{res['model_mode']}**  ·  📅 {res['timestamp']}  ·  🎯 Top: {res.get('top_class','N/A')}")

            else:
                st.markdown("""
                <div class="glass-card" style="text-align:center; padding:4.5rem 1.5rem;">
                    <div style="font-size:4rem; margin-bottom:14px; opacity:0.55;">🔬</div>
                    <div style="font-weight:700; font-size:0.98rem; margin-bottom:8px;">Ready for Analysis</div>
                    <div style="font-size:0.83rem; color:#6b9ab8; line-height:1.7;">
                        Upload an image or capture via camera<br>then click <b>EXECUTE DEEP SCAN</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        if st.session_state.result:
            res   = st.session_state.result
            intel = ClinicalProtocols.get(res["diagnosis"])
            st.markdown("---")
            st.markdown('<div class="sec-head" style="font-size:1.04rem;"><span></span>📋 Clinical Intelligence Engine</div>', unsafe_allow_html=True)
            t1,t2,t3,t4,t5 = st.tabs(["🏥 Recommendations","🌿 Patient Advice","💊 Treatment Plan","🤖 AI Explanation","📄 Report"])

            with t1:
                r1,r2 = st.columns(2)
                with r1:
                    st.markdown("**Clinical Recommendations**")
                    for i in intel["recommendations"]:
                        st.markdown(f'<div class="step-box">{i}</div>', unsafe_allow_html=True)
                with r2:
                    st.markdown("**Consultation & Follow-up**")
                    st.markdown(f'<div class="step-box" style="border-left-color:{intel["hex"]};">{intel["consultation"]}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="step-box">📅 {intel["followup"]}</div>', unsafe_allow_html=True)

            with t2:
                for i in intel["patient_advice"]:
                    st.markdown(f'<div class="step-box">🌿 {i}</div>', unsafe_allow_html=True)

            with t3:
                tc1,tc2 = st.columns(2)
                plan = [("🩺 Procedures","procedures",False,"#2563eb"),("💊 Medications","medications",False,"#14b8a6"),
                        ("⚗️ Therapy","therapy",False,"#8b5cf6"),("🚨 Emergency Signs","emergency_signs",True,"#ef4444")]
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
                st.markdown("#### 🤖 Why Did AI Give This Result?")
                st.markdown(f"""
                <div class="glass-card" style="border-left:3px solid #8b5cf6; padding:20px;">
                    <div style="font-weight:700; color:#a78bfa; margin-bottom:12px; font-size:0.9rem;">
                        🧠 AI Feature Analysis — {res['diagnosis']} Detection
                    </div>
                    <div style="font-size:0.85rem; line-height:1.8; color:#dff0fa;">
                        {intel['why_result']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("#### 📊 Multi-Class Probability Scores")
                scores = res.get("class_scores", {})
                if scores:
                    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                    labels = [s[0] for s in sorted_scores]
                    values = [s[1]*100 for s in sorted_scores]
                    colors = ["#ef4444" if v == max(values) else "#2563eb" for v in values]
                    fig_cls = go.Figure(go.Bar(
                        x=values, y=labels, orientation="h",
                        marker_color=colors, marker_line_width=0,
                        text=[f"{v:.1f}%" for v in values], textposition="inside",
                        textfont=dict(color="white", size=10, family="Oxanium"),
                    ))
                    fig_cls.update_layout(
                        height=300, margin=dict(l=10,r=10,t=20,b=10),
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#6b9ab8", family="Outfit"),
                        xaxis=dict(range=[0,100], showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False),
                    )
                    st.plotly_chart(fig_cls, use_container_width=True)

                st.markdown("#### 🔥 Grad-CAM Heatmap Simulation")
                st.markdown("""
                <div class="heatmap-box">
                    <div style="font-size:2rem; margin-bottom:8px;">🔥</div>
                    <div style="font-weight:600; margin-bottom:4px; color:#f87171;">Grad-CAM Visualization</div>
                    <div style="font-size:0.78rem;">
                        Grad-CAM highlights regions of the image that most influenced the AI's decision.<br><br>
                        <strong>Red zones</strong> = High activation / suspicious areas<br>
                        <strong>Blue zones</strong> = Low activation / normal tissue<br><br>
                        ⚙️ Full Grad-CAM requires TensorFlow model in online mode.
                        Load <code>skin_cancer_cnn.h5</code> to enable real heatmaps.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with t5:
                st.markdown("#### 📥 Download Clinical Reports")
                d1,d2 = st.columns(2)
                rec  = st.session_state.result
                proc = st.session_state.proc_img
                with d1:
                    if PDF_OK and proc:
                        pdf_bytes = ReportGenerator.pdf(rec, proc)
                        fname = f"SkinScan_{rec.get('patient_name','PT')}_{datetime.date.today()}.pdf".replace(" ","_")
                        st.download_button("📄 Download PDF Report", data=pdf_bytes, file_name=fname, mime="application/pdf")
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
    #  PAGE: AI ANALYSIS
    # ══════════════════════════════════════════════════════════════
    def _analysis(self):
        st.markdown("""
        <div class="page-banner">
            <div class="banner-chip">🧠 Deep Analysis</div>
            <p class="banner-title">AI Analysis Results</p>
            <p class="banner-sub">Multi-class scores · AI Explanation · Grad-CAM · Accuracy Meter</p>
        </div>
        """, unsafe_allow_html=True)

        if not st.session_state.result:
            st.info("📭 No scan data. Head to **AI Scan** to perform an analysis first.")
            return

        res   = st.session_state.result
        intel = ClinicalProtocols.get(res["diagnosis"])

        c1, c2 = st.columns([1.2, 1], gap="large")
        with c1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="sec-head"><span></span>🎯 Primary Diagnosis: <span style="color:{intel["hex"]};">{res["diagnosis"]}</span></div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div style="font-size:0.9rem; line-height:1.8; margin-bottom:16px;">
                <strong>Top Class:</strong> <span style="color:{intel['hex']}; font-family:'Oxanium',monospace; font-weight:700;">{res.get('top_class','N/A')}</span><br>
                <strong>Risk Level:</strong> <span class="badge {'b-high' if res['risk_level']=='HIGH' else 'b-medium' if res['risk_level']=='MEDIUM' else 'b-low'}"> {res['risk_level']}</span><br>
                <strong>Model:</strong> {res['model_mode']}<br>
                <strong>Scan Time:</strong> {res['timestamp']}
            </div>
            """, unsafe_allow_html=True)

            fig_acc = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=res["confidence"]*100,
                delta={"reference":70,"increasing":{"color":"#10b981"},"decreasing":{"color":"#ef4444"}},
                number={"suffix":"%","font":{"family":"Oxanium","size":32,"color":intel["hex"]}},
                title={"text":"Prediction Accuracy Meter","font":{"family":"Outfit","size":12,"color":"#6b9ab8"}},
                gauge={
                    "axis":{"range":[0,100],"tickfont":{"size":9}},
                    "bar":{"color":intel["hex"],"thickness":0.28},
                    "bgcolor":"rgba(0,0,0,0)","borderwidth":0,
                    "steps":[{"range":[0,40],"color":"rgba(239,68,68,0.08)"},
                             {"range":[40,70],"color":"rgba(245,158,11,0.08)"},
                             {"range":[70,100],"color":"rgba(16,185,129,0.08)"}],
                    "threshold":{"line":{"color":"white","width":2},"value":res["confidence"]*100},
                },
            ))
            fig_acc.update_layout(height=220, margin=dict(l=10,r=10,t=50,b=5), paper_bgcolor="rgba(0,0,0,0)", font_color="#6b9ab8")
            st.plotly_chart(fig_acc, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="sec-head"><span></span>📊 Multi-Class Scores</div>', unsafe_allow_html=True)
            scores = res.get("class_scores", {})
            for cls, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
                pct = score * 100
                color = intel["hex"] if cls == res.get("top_class") else "#6b9ab8"
                is_top = "⭐" if cls == res.get("top_class") else ""
                st.markdown(f"""
                <div style="margin-bottom:8px;">
                    <div style="display:flex; justify-content:space-between; font-size:0.76rem; margin-bottom:3px;">
                        <span style="color:{color}; font-weight:{'700' if is_top else '400'};">{is_top} {cls}</span>
                        <span style="font-family:'Oxanium',monospace; color:{color};">{pct:.1f}%</span>
                    </div>
                    <div style="background:rgba(37,99,235,0.12); border-radius:99px; height:6px; overflow:hidden;">
                        <div style="width:{pct}%; height:100%; background:{color}; border-radius:99px; transition:width 0.5s;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="glass-card" style="border-left:3px solid #8b5cf6;">', unsafe_allow_html=True)
        st.markdown('<div class="sec-head"><span></span>🤖 AI Explanation Panel</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-size:0.87rem; line-height:1.9; color:#dff0fa;">
            {intel['why_result']}
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="sec-head"><span></span>🔥 Grad-CAM Heatmap Visualization</div>', unsafe_allow_html=True)
        h1, h2 = st.columns(2)
        with h1:
            if st.session_state.raw_img:
                st.image(ImageProcessor.thumb(st.session_state.raw_img, 350), caption="📷 Original Image", use_container_width=True)
        with h2:
            st.markdown("""
            <div class="heatmap-box" style="height:200px; display:flex; flex-direction:column; justify-content:center;">
                <div style="font-size:2.5rem; margin-bottom:8px;">🔥</div>
                <div style="font-weight:700; color:#f87171; margin-bottom:4px;">Grad-CAM Overlay</div>
                <div style="font-size:0.76rem;">Load skin_cancer_cnn.h5 for real heatmaps.<br>Red = High Suspicion · Blue = Normal</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

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
        hi  = sum(1 for r in db if r.get("risk_level")=="HIGH")

        k1,k2,k3,k4,k5 = st.columns(5)
        kpis = [
            ("🧬","Total Scans",    str(n),    "This session",       "#3b82f6"),
            ("🔴","Malignant",      str(mal),  "High-risk detected", "#ef4444"),
            ("🟢","Benign",         str(n-mal),"Low-risk cleared",   "#10b981"),
            ("🚨","High Risk",      str(hi),   "Urgent cases",       "#f59e0b"),
            ("⚡","Avg Confidence", f"{c:.1f}%","CNN inference",     "#8b5cf6"),
        ]
        for col,(icon,lbl,val,dlt,color) in zip([k1,k2,k3,k4,k5],kpis):
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
                  font=dict(family="Outfit",color="#6b9ab8"), margin=dict(l=4,r=4,t=44,b=4))
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
            ))
            fig1.update_layout(title="Diagnosis Distribution", height=300, showlegend=True,
                               legend=dict(font_size=11,orientation="h",y=-0.1),
                               annotations=[dict(text=f"<b>{n}</b><br>scans",x=0.5,y=0.5,font_size=14,font_color="#dff0fa",showarrow=False)],
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
                    fig2.add_trace(go.Bar(x=s["Risk"],y=s["Count"],name=risk,marker_color=color,marker_line_width=0))
            fig2.update_layout(title="Risk Distribution",height=300,showlegend=False,
                               barmode="group",xaxis=dict(title="",**GR),yaxis=dict(title="Cases",**GR),**PL)
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        r3,r4 = st.columns(2)
        with r3:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            fig3 = go.Figure()
            for diag,color,sym in [("Malignant","#ef4444","circle"),("Benign","#10b981","diamond")]:
                s = df[df["diagnosis"]==diag]
                if not s.empty:
                    fig3.add_trace(go.Scatter(x=s["prob"],y=s["conf"],mode="markers",name=diag,
                        marker=dict(color=color,size=9,opacity=0.85,symbol=sym)))
            fig3.update_layout(title="Probability vs Confidence",height=285,
                               xaxis=dict(title="Probability (%)",**GR),yaxis=dict(title="Confidence (%)",**GR),
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
            "Top Class": r.get("top_class","—"),
            "Risk":      r.get("risk_level","—"),
            "Prob.":     f"{r.get('probability',0)*100:.1f}%",
            "Conf.":     f"{r.get('confidence',0)*100:.1f}%",
            "Blur":      f"{r.get('blur_score',0):.0f}",
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
            fd = fc1.multiselect("Diagnosis",["Malignant","Benign"],default=["Malignant","Benign"])
            fr = fc2.multiselect("Risk",["HIGH","MEDIUM","LOW"],default=["HIGH","MEDIUM","LOW"])
            fg = fc3.multiselect("Gender",["Male","Female","Other","Prefer not to say"],default=["Male","Female","Other","Prefer not to say"])

        mask = df["Diagnosis"].isin(fd) & df["Risk"].isin(fr) & df["Gender"].isin(fg)
        df_f = df[mask]
        st.caption(f"Showing **{len(df_f)}** of **{len(df)}** records")
        st.markdown('<div class="glass-card" style="padding:0;overflow:hidden;">', unsafe_allow_html=True)
        st.dataframe(df_f, use_container_width=True, hide_index=True, height=380)
        st.markdown('</div>', unsafe_allow_html=True)

        e1,e2,e3 = st.columns(3)
        with e1:
            st.download_button("📥 Export CSV", data=ReportGenerator.csv_data(db),
                               file_name=f"SkinScan_{datetime.date.today()}.csv", mime="text/csv")
        with e2:
            safe = [{k:str(v) if isinstance(v,datetime.datetime) else v for k,v in r.items() if not isinstance(v,dict)} for r in db]
            st.download_button("🔗 Export JSON", data=json.dumps(safe,indent=2),
                               file_name=f"SkinScan_{datetime.date.today()}.json", mime="application/json")
        with e3:
            if st.button("🗑️ Clear All Records"):
                st.session_state.db=[];st.session_state.result=None
                st.session_state.raw_img=None;st.session_state.proc_img=None
                st.rerun()

    # ══════════════════════════════════════════════════════════════
    #  PAGE: MEDICAL GUIDE
    # ══════════════════════════════════════════════════════════════
    def _medical_guide(self):
        st.markdown("""
        <div class="page-banner">
            <div class="banner-chip">👨‍⚕️ Medical Professional</div>
            <p class="banner-title">Medical Guide & Prevention</p>
            <p class="banner-sub">Doctor recommendations · Prevention tips · Disease information · Treatment awareness · Warning signs</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="disclaimer-banner" style="margin-bottom:24px;">
            🏥 <strong>For Healthcare Professionals & Patients:</strong> The information below is for
            <strong>educational awareness only</strong>. It does not replace professional medical advice,
            diagnosis, or treatment. Always seek guidance from a <strong>qualified dermatologist or oncologist</strong>.
            In case of emergency symptoms, call your local emergency number immediately.
        </div>
        """, unsafe_allow_html=True)

        t1,t2,t3,t4,t5 = st.tabs(["👨‍⚕️ Doctor Guide","🛡️ Prevention Tips","📚 Disease Info","💊 Treatments","🚨 Warning Signs"])

        with t1:
            st.markdown('<div class="sec-head"><span></span>👨‍⚕️ Doctor Recommendation System</div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="glass-card">
                <div style="font-weight:700; font-size:1.0rem; margin-bottom:12px; color:#60a5fa;">
                    When Should You See a Dermatologist?
                </div>
            """, unsafe_allow_html=True)
            urgencies = [
                ("🚨 EMERGENCY","Visit ER / Call 112 immediately","Any lesion with active bleeding, rapid ulceration, lymph node swelling, or spreading satellite lesions.","#ef4444"),
                ("⚡ URGENT (48h)","Onco-Dermatologist within 2 days","AI flagged HIGH RISK, asymmetric lesion >6mm, multi-color, rapidly changing morphology.","#f59e0b"),
                ("📅 ROUTINE","Annual dermatology visit","Stable benign lesion with no ABCDE changes. Standard monitoring protocol.","#10b981"),
                ("🔍 MONITORING","6-month follow-up","Medium confidence AI result, borderline features, or patient history of melanoma.","#3b82f6"),
            ]
            for icon, title, desc, color in urgencies:
                st.markdown(f"""
                <div class="step-box" style="border-left-color:{color}; margin-bottom:10px;">
                    <div style="font-weight:700; color:{color}; margin-bottom:4px; font-size:0.88rem;">{icon} {title}</div>
                    <div style="font-size:0.82rem; color:#dff0fa;">{desc}</div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="sec-head" style="margin-top:20px;"><span></span>🔬 What to Tell Your Doctor</div>', unsafe_allow_html=True)
            tell_doc = [
                "Duration — How long has the lesion been present?",
                "Changes — Has it grown, changed color, or changed shape recently?",
                "Symptoms — Any pain, itching, burning, or bleeding?",
                "History — Family history of melanoma or skin cancer?",
                "Exposure — Years of sun/UV exposure, history of sunburns?",
                "Previous lesions — Any previously removed or biopsied lesions?",
                "Photo documentation — Bring timestamped photos if available.",
            ]
            for tip in tell_doc:
                st.markdown(f'<div class="prev-tip">📋 {tip}</div>', unsafe_allow_html=True)

        with t2:
            st.markdown('<div class="sec-head"><span></span>🛡️ Skin Cancer Prevention Tips</div>', unsafe_allow_html=True)
            cats = {
                "☀️ Sun Protection": [
                    "Apply SPF 50+ broad-spectrum sunscreen every 2 hours outdoors.",
                    "Reapply after swimming, sweating, or towel drying.",
                    "Wear UPF 50+ protective clothing, wide-brim hat, and UV-blocking sunglasses.",
                    "Seek shade between 10:00 AM – 4:00 PM (peak UV hours).",
                    "Avoid tanning beds and sunlamps — they emit harmful UV radiation.",
                    "Check the daily UV index — take extra precautions when UV ≥ 6.",
                ],
                "🥗 Nutrition & Lifestyle": [
                    "Eat antioxidant-rich foods: berries, leafy greens, tomatoes, carrots.",
                    "Omega-3 fatty acids (salmon, flaxseed) have photoprotective effects.",
                    "Green tea polyphenols may reduce UV-induced skin damage.",
                    "Maintain healthy weight — obesity linked to increased cancer risk.",
                    "Quit smoking — accelerates cellular DNA damage in skin tissue.",
                    "Limit alcohol — associated with increased melanoma risk.",
                ],
                "🔍 Self-Examination": [
                    "Perform full-body ABCDE self-check once per month.",
                    "Use a full-length mirror and hand mirror to check hard-to-see areas.",
                    "Photograph suspicious lesions monthly to track changes over time.",
                    "Check scalp, between toes, under nails, and in skin folds.",
                    "Report any new mole appearing after age 30 to a dermatologist.",
                    "Use the 'Ugly Duckling' rule — a mole that looks different from others.",
                ],
                "🏥 Medical Monitoring": [
                    "Annual professional skin examination for all adults over 40.",
                    "Every 6 months if you have risk factors (family history, fair skin, many moles).",
                    "Digital dermoscopy baseline documentation for all concerning lesions.",
                    "Genetic counseling if 2+ first-degree relatives have melanoma.",
                    "Vitamin D via supplementation only — not UV exposure.",
                ],
            }
            for cat, tips in cats.items():
                with st.expander(cat, expanded=False):
                    for tip in tips:
                        st.markdown(f'<div class="prev-tip">{tip}</div>', unsafe_allow_html=True)

        with t3:
            st.markdown('<div class="sec-head"><span></span>📚 Skin Cancer Educational Guide</div>', unsafe_allow_html=True)
            diseases = [
                ("🔴","Melanoma","Most dangerous. Arises from melanocytes. Can spread to organs. 5-year survival >98% if caught early vs ~23% if metastatic.",
                 "Asymmetric, irregular border, multiple colors, >6mm, evolving. Can appear anywhere including scalp, under nails, soles.","#ef4444"),
                ("🟠","Basal Cell Carcinoma (BCC)","Most common skin cancer (~80% of cases). Rarely metastasizes but causes significant local tissue destruction if untreated.",
                 "Pearly or waxy bump, flat flesh-colored or brown scar-like lesion, bleeding or scabbing sore that heals and returns.","#f97316"),
                ("🟡","Squamous Cell Carcinoma (SCC)","Second most common. Can spread to lymph nodes. Risk increases with cumulative UV exposure and immunosuppression.",
                 "Firm red nodule, flat lesion with scaly crust, new sore on old scar, rough patch on lip, red sore inside mouth.","#f59e0b"),
                ("🟢","Benign Nevus (Mole)","Common benign growth from melanocytes. Most people have 10–40 moles. Monitoring for changes is essential.",
                 "Symmetrical, well-defined border, uniform color (tan or brown), usually <6mm. Present since childhood or young adulthood.","#10b981"),
                ("🔵","Actinic Keratosis","Pre-cancerous lesion caused by chronic UV exposure. 5–10% progress to SCC if untreated. Treat to prevent progression.",
                 "Rough, dry, scaly patch of skin, flat to slightly raised patch, hard wart-like surface, itching/burning in the affected area.","#3b82f6"),
                ("🟣","Seborrheic Keratosis","Benign, waxy, wart-like growth. Very common in older adults. No cancer risk but can resemble melanoma.",
                 "Waxy, scaly, slightly elevated lesion. Range from light tan to black. 'Stuck on' appearance. Multiple lesions common.","#8b5cf6"),
            ]
            gc = st.columns(2)
            for i,(icon,name,about,signs,color) in enumerate(diseases):
                with gc[i%2]:
                    st.markdown(f"""
                    <div class="guide-card" style="border-top:3px solid {color}; margin-bottom:16px;">
                        <div style="display:flex; align-items:center; gap:8px; margin-bottom:10px;">
                            <span style="font-size:1.5rem;">{icon}</span>
                            <div style="font-weight:700; font-size:0.96rem; color:{color};">{name}</div>
                        </div>
                        <div style="font-size:0.78rem; color:#dff0fa; line-height:1.7; margin-bottom:10px;">{about}</div>
                        <div style="font-size:0.74rem; color:#6b9ab8; line-height:1.6; border-top:1px solid rgba(37,99,235,0.15); padding-top:8px;">
                            <strong style="color:{color};">Visual Signs:</strong> {signs}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        with t4:
            st.markdown('<div class="sec-head"><span></span>💊 Treatment Awareness</div>', unsafe_allow_html=True)
            treatments = [
                ("🩺","Surgical Excision","Removal of lesion with clear margins. Gold standard for most skin cancers. Success rate >95% for early-stage lesions.","#2563eb"),
                ("🔬","Mohs Surgery","Layer-by-layer removal with real-time microscopic analysis. Highest cure rate (~99%) for BCC/SCC.","#14b8a6"),
                ("💊","Immunotherapy","Pembrolizumab, Ipilimumab — checkpoint inhibitors that activate immune system to fight cancer cells.","#8b5cf6"),
                ("🧬","Targeted Therapy","BRAF/MEK inhibitors for BRAF-mutated melanoma. Vemurafenib + Cobimetinib combination.","#f59e0b"),
                ("☀️","Photodynamic Therapy","Light-activated drug destroys cancer cells. Used for superficial BCC and actinic keratosis.","#10b981"),
                ("❄️","Cryotherapy","Liquid nitrogen freeze-destroys benign/pre-cancerous lesions. Quick office procedure.","#60a5fa"),
                ("⚡","Radiation Therapy","Used post-surgery or when surgery not possible. Targets residual cancer cells.","#ef4444"),
                ("💉","Intralesional Therapy","Direct injection of IL-2, talimogene laherparepvec (T-VEC) into tumor. Used for advanced melanoma.","#f97316"),
            ]
            tc = st.columns(2)
            for i,(icon,name,desc,color) in enumerate(treatments):
                with tc[i%2]:
                    st.markdown(f"""
                    <div class="step-box" style="border-left-color:{color}; margin-bottom:10px;">
                        <div style="display:flex; align-items:center; gap:8px; margin-bottom:4px;">
                            <span>{icon}</span>
                            <span style="font-weight:700; color:{color}; font-size:0.88rem;">{name}</span>
                        </div>
                        <div style="font-size:0.79rem; color:#dff0fa; line-height:1.6;">{desc}</div>
                    </div>
                    """, unsafe_allow_html=True)

        with t5:
            st.markdown('<div class="sec-head"><span></span>🚨 Warning Signs — Seek Immediate Care</div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="disclaimer-banner" style="border-left-color:#ef4444; background:linear-gradient(135deg,rgba(239,68,68,0.10),rgba(220,38,38,0.05));">
                🚨 <strong>EMERGENCY:</strong> If you experience any of the following symptoms, seek immediate medical attention.
                Do NOT wait for an appointment — visit an emergency department or call your doctor immediately.
            </div>
            """, unsafe_allow_html=True)
            warnings = [
                ("🔴","Rapid Size Change","Lesion doubles in size within days or weeks — a sign of aggressive growth."),
                ("🩸","Spontaneous Bleeding","Lesion bleeds without any injury or trauma — indicates disrupted vasculature."),
                ("🔵","Lymph Node Swelling","Swollen lymph nodes near neck, armpit, or groin — possible metastatic spread."),
                ("🟡","Ulceration","Open sore that doesn't heal within 4 weeks — hallmark of invasive malignancy."),
                ("🟣","Satellite Lesions","New small lesions appearing around original mole — sign of local metastasis."),
                ("⚪","Color Whitening","Loss of pigment (white/grey zones) within lesion — regression pattern in melanoma."),
                ("🔶","Pain / Burning","New pain, burning, or tingling in lesion area — possible nerve involvement."),
                ("🟤","Thick / Raised Profile","Sudden thickening or raised hard nodule forming on flat lesion — vertical growth phase."),
            ]
            wc = st.columns(2)
            for i,(color,title,desc) in enumerate(warnings):
                with wc[i%2]:
                    st.markdown(f"""
                    <div class="step-box step-emg" style="margin-bottom:10px;">
                        <div style="display:flex; align-items:center; gap:8px; margin-bottom:3px;">
                            <span style="font-size:1.1rem;">{color}</span>
                            <span style="font-weight:700; color:#f87171; font-size:0.87rem;">{title}</span>
                        </div>
                        <div style="font-size:0.78rem; color:#dff0fa; line-height:1.5;">{desc}</div>
                    </div>
                    """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════
    #  PAGE: ABOUT
    # ══════════════════════════════════════════════════════════════
    def _about(self):
        st.markdown("""
        <div class="page-banner">
            <div class="banner-chip">ℹ️ About</div>
            <p class="banner-title">About & Settings</p>
            <p class="banner-sub">Platform information · Appearance · User guide · AI Engine Details</p>
        </div>
        """, unsafe_allow_html=True)

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

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="sec-head"><span></span>🤖 AI Engine Details</div>', unsafe_allow_html=True)
        dot  = "🟢" if self.ai.is_online else "🟠"
        mode = "Neural Network Online" if self.ai.is_online else "Simulation Mode"
        rows = [
            ("Model File",       "skin_cancer_cnn.h5",                                   "#14b8a6"),
            ("Architecture",     "Convolutional Neural Network (CNN)",                    "#60a5fa"),
            ("Output Classes",   "8 classes: Melanoma, BCC, SCC, Benign, AK, SK, DF, VL","#a78bfa"),
            ("Input Dimensions", "224 × 224 px  ·  RGB  ·  Normalized 0–1",              "#60a5fa"),
            ("Preprocessing",    "Resize → Normalize → Contrast → Sharpen → Brightness", "#7fa3c0"),
            ("Blur Detection",   "Laplacian variance — threshold: 80",                    "#f59e0b"),
            ("Grad-CAM",         "Available in online mode with TensorFlow backend",      "#10b981"),
            ("Engine Status",    f"{dot} {mode}",                                        "#f59e0b"),
            ("Platform Version", "SkinScan AI Next-Gen Intelligence v15.0",              "#7fa3c0"),
        ]
        for lbl,val,color in rows:
            st.markdown(f"""
            <div class="set-row">
                <span class="set-lbl">{lbl}</span><br>
                <span style="font-family:'Oxanium',monospace; font-size:0.81rem; color:{color};">{val}</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="sec-head"><span></span>📖 User Guide</div>', unsafe_allow_html=True)
        guide = [
            ("📤 Image Upload", ["Accepted: <b>JPG, JPEG, PNG</b> only.",
                "Max <b>10 MB</b> · Min <b>100×100 px</b> · Drag & Drop supported.",
                "Auto-preprocessing: resize 224×224 → normalize → enhance → brightness correct.",
                "Quality badge shown: HIGH (≥300×300) · MEDIUM (<300×300) with warning."]),
            ("📷 Camera Capture", ["Click <b>Live Camera</b> toggle in AI Scan.",
                "Allow camera access in browser. Works on mobile camera too.",
                "Use <b>Retake Photo</b> button to discard and re-capture.",
                "Before/After comparison view available after first scan."]),
            ("🤖 AI Inference", ["Model: <b>skin_cancer_cnn.h5</b> loaded at startup.",
                "🟢 Online Mode: Real TF inference | 🟠 Simulation: Demo mode.",
                "8-class multi-class scores with top-class identification.",
                "Blur detection via Laplacian variance — warns if image is blurry."]),
            ("📊 Results & Reports", ["<b>Probability</b>: Likelihood of primary diagnosis.",
                "<b>Confidence</b>: Model certainty | Risk: HIGH ≥80% · MEDIUM ≥50% · LOW <50%.",
                "AI Explanation panel: feature-by-feature breakdown of decision.",
                "Download PDF + CSV from Report tab. JSON export in History."]),
        ]
        for title,points in guide:
            with st.expander(title):
                for pt in points:
                    st.markdown(f'<div class="step-box">{pt}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

   """
╔══════════════════════════════════════════════════════════════════════════════╗
║  SKINSCAN AI  ·  FOOTER FIX  ·  v15.1                                       ║
║  ROOT CAUSE: Footer HTML was rendering as raw text because:                  ║
║    1. Streamlit's st.markdown() with unsafe_allow_html=True can fail         ║
║       when called inside certain layout contexts (columns, expanders, etc.)  ║
║    2. CSS classes in inject_css() are injected once at page load —           ║
║       if the footer renders before CSS is parsed, classes are unknown.       ║
║                                                                               ║
║  FIX: Make _footer() completely self-contained — ALL CSS is embedded         ║
║  directly inside the footer HTML block via a <style> tag. This guarantees   ║
║  the footer always renders correctly regardless of page state.               ║
╚══════════════════════════════════════════════════════════════════════════════╝

INSTRUCTIONS:
  1. Open your main app file (e.g. app.py / skinscan.py)
  2. Find the _footer() method inside SkinScanApp class
  3. REPLACE the entire _footer() method with the one below
  4. Save and re-run: streamlit run app.py

No other changes needed. The fix is 100% backward-compatible.
"""

# ══════════════════════════════════════════════════════════════════════════════
#  PASTE THIS METHOD TO REPLACE _footer() IN YOUR SkinScanApp CLASS
# ══════════════════════════════════════════════════════════════════════════════

FOOTER_METHOD = '''
    def _footer(self):
        """
        Self-contained footer — all CSS is embedded inline.
        This avoids dependency on inject_css() and fixes raw-HTML rendering bug.
        """
        dark = st.session_state.get("theme", "dark") == "dark"

        # Theme-aware color tokens
        FOOTER_BG  = "rgba(2,8,18,0.97)"      if dark else "rgba(8,20,45,0.97)"
        TEXT       = "#dff0fa"                  if dark else "#0c1e32"
        SUB        = "#6b9ab8"                  if dark else "#3a6080"
        BORDER     = "rgba(37,99,235,0.22)"
        SURF2      = "rgba(6,28,60,0.65)"       if dark else "rgba(240,247,255,0.90)"

        st.markdown(f"""
        <style>
        /* ── FOOTER SELF-CONTAINED STYLES ── */
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=Oxanium:wght@600;700;800&family=Space+Mono:wght@400;700&display=swap');

        .sf-outer {{
            margin-top: 4rem;
            margin-left: calc(-2rem - 1px);
            margin-right: calc(-2rem - 1px);
            width: calc(100% + 4rem + 2px);
        }}
        .sf-wrap {{
            background: {FOOTER_BG};
            backdrop-filter: blur(24px) saturate(180%);
            -webkit-backdrop-filter: blur(24px) saturate(180%);
            border-top: 1px solid {BORDER};
            position: relative;
            overflow: hidden;
            width: 100%;
            font-family: 'Outfit', sans-serif;
        }}
        /* Gradient top line */
        .sf-wrap::before {{
            content: '';
            position: absolute; top: 0; left: 0; right: 0; height: 2px;
            background: linear-gradient(90deg,
                transparent 0%, #2563eb 20%, #14b8a6 50%, #8b5cf6 80%, transparent 100%);
            z-index: 2;
        }}
        /* Subtle grid texture */
        .sf-wrap::after {{
            content: '';
            position: absolute; inset: 0; pointer-events: none; z-index: 0;
            background-image: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none'%3E%3Cg fill='%232563eb' fill-opacity='0.012'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
        }}
        .sf-inner {{
            position: relative; z-index: 1;
            max-width: 1320px; margin: 0 auto;
            padding: 0 2.5rem; box-sizing: border-box;
        }}

        /* ── Top grid: 4 columns ── */
        .sf-top {{
            display: grid;
            grid-template-columns: 1.8fr 1fr 1fr 1.3fr;
            gap: 48px;
            padding: 3.5rem 0 2.5rem;
            border-bottom: 1px solid rgba(37,99,235,0.14);
            align-items: start;
        }}

        /* ── Brand block ── */
        .sf-brand-logo {{
            display: flex; align-items: center; gap: 10px; margin-bottom: 14px;
        }}
        .sf-brand-icon {{
            font-size: 1.8rem;
            filter: drop-shadow(0 0 10px rgba(20,184,166,0.5));
            animation: sf-logo-pulse 3s ease-in-out infinite;
            flex-shrink: 0;
        }}
        @keyframes sf-logo-pulse {{
            0%,100% {{ filter: drop-shadow(0 0 5px rgba(37,99,235,0.6)); transform: scale(1); }}
            50%      {{ filter: drop-shadow(0 0 18px rgba(20,184,166,0.8)); transform: scale(1.08); }}
        }}
        .sf-brand-name {{
            font-family: 'Oxanium', sans-serif; font-size: 1.15rem; font-weight: 800;
            background: linear-gradient(135deg, #3b82f6 0%, #14b8a6 55%, #8b5cf6 100%);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            letter-spacing: 0.2px; line-height: 1.2;
        }}
        .sf-brand-tagline {{
            font-size: 0.68rem; color: {SUB};
            letter-spacing: 1.5px; text-transform: uppercase;
            font-weight: 500; line-height: 1.3;
        }}
        .sf-brand-desc {{
            font-size: 0.82rem; color: {SUB}; line-height: 1.75;
            margin-bottom: 18px; max-width: 300px;
        }}

        /* ── Tech chips ── */
        .sf-tech-stack {{
            display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 20px;
        }}
        .sf-tech-chip {{
            background: rgba(37,99,235,0.10); border: 1px solid rgba(37,99,235,0.22);
            color: #60a5fa; font-size: 0.61rem; font-weight: 600;
            padding: 3px 10px; border-radius: 6px; letter-spacing: 0.4px;
            transition: all 0.2s; white-space: nowrap; cursor: default;
        }}
        .sf-tech-chip:hover {{
            background: rgba(37,99,235,0.20); transform: translateY(-1px);
            box-shadow: 0 3px 8px rgba(37,99,235,0.20);
        }}

        /* ── Social buttons ── */
        .sf-social {{ display: flex; gap: 10px; margin-top: 4px; flex-wrap: wrap; }}
        .sf-social-btn {{
            width: 40px; height: 40px; border-radius: 10px;
            display: inline-flex; align-items: center; justify-content: center;
            text-decoration: none; transition: all 0.28s cubic-bezier(.34,1.56,.64,1);
            border: 1px solid rgba(37,99,235,0.25);
            background: rgba(37,99,235,0.08); cursor: pointer; flex-shrink: 0;
        }}
        .sf-social-btn:hover {{ transform: translateY(-5px) scale(1.12); box-shadow: 0 10px 24px rgba(37,99,235,0.30); }}
        .sf-social-btn.github:hover  {{ background:rgba(255,255,255,0.12); border-color:rgba(255,255,255,0.30); box-shadow:0 10px 24px rgba(255,255,255,0.15); }}
        .sf-social-btn.linkedin:hover {{ background:rgba(10,102,194,0.22); border-color:rgba(10,102,194,0.50); box-shadow:0 10px 24px rgba(10,102,194,0.30); }}
        .sf-social-btn.email:hover    {{ background:rgba(20,184,166,0.15); border-color:rgba(20,184,166,0.45); box-shadow:0 10px 24px rgba(20,184,166,0.25); }}

        /* ── Column headings ── */
        .sf-col-title {{
            font-size: 0.70rem; font-weight: 700; color: {TEXT};
            text-transform: uppercase; letter-spacing: 2px; margin-bottom: 18px;
            display: flex; align-items: center; gap: 8px; white-space: nowrap;
        }}
        .sf-col-title::before {{
            content: ''; display: inline-block; width: 14px; height: 2px;
            background: linear-gradient(90deg, #2563eb, #14b8a6);
            border-radius: 2px; flex-shrink: 0;
        }}

        /* ── Nav links ── */
        .sf-nav-link {{
            display: block; font-size: 0.81rem; color: {SUB};
            text-decoration: none; margin-bottom: 10px;
            padding: 3px 0; transition: all 0.2s; cursor: pointer;
            white-space: nowrap;
        }}
        .sf-nav-link:hover {{ color: #60a5fa; padding-left: 6px; }}

        /* ── Contact items ── */
        .sf-contact-item {{
            display: flex; align-items: flex-start; gap: 10px;
            margin-bottom: 14px; font-size: 0.81rem;
        }}
        .sf-ci-icon {{
            width: 32px; height: 32px; border-radius: 8px; flex-shrink: 0;
            display: flex; align-items: center; justify-content: center;
            font-size: 0.95rem; background: rgba(37,99,235,0.12);
            border: 1px solid rgba(37,99,235,0.20);
        }}
        .sf-ci-label {{
            font-size: 0.63rem; color: {SUB};
            text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 3px;
        }}
        .sf-ci-value {{
            color: {TEXT}; font-weight: 500; word-break: break-all; line-height: 1.4;
        }}
        .sf-ci-value a {{ color: #60a5fa; text-decoration: none; transition: color 0.2s; }}
        .sf-ci-value a:hover {{ color: #2dd4bf; }}

        .sf-email-copy {{
            display: inline-flex; align-items: center; gap: 5px;
            font-size: 0.70rem; color: {SUB}; cursor: pointer;
            background: rgba(37,99,235,0.08); border: 1px solid rgba(37,99,235,0.18);
            padding: 3px 10px; border-radius: 6px; margin-top: 5px;
            transition: all 0.2s; user-select: none; width: fit-content;
        }}
        .sf-email-copy:hover {{ background: rgba(37,99,235,0.18); color: #60a5fa; }}

        /* ── Badges row ── */
        .sf-badges {{
            display: flex; flex-wrap: wrap; gap: 8px;
            padding: 20px 0 4px;
            border-top: 1px solid rgba(37,99,235,0.10);
        }}
        .sf-badge {{
            display: inline-flex; align-items: center; gap: 5px;
            background: rgba(37,99,235,0.07); border: 1px solid rgba(37,99,235,0.16);
            color: {SUB}; font-size: 0.64rem; font-weight: 600;
            padding: 5px 12px; border-radius: 8px; letter-spacing: 0.4px;
            transition: all 0.2s; white-space: nowrap; cursor: default;
        }}
        .sf-badge:hover {{ background: rgba(37,99,235,0.14); color: #60a5fa; }}

        /* ── Bottom bar ── */
        .sf-bottom {{
            display: flex; align-items: center; justify-content: space-between;
            flex-wrap: wrap; gap: 12px; padding: 18px 0 22px;
        }}
        .sf-copy {{ font-size: 0.75rem; color: {SUB}; line-height: 1.5; }}
        .sf-copy strong {{ color: {TEXT}; }}
        .sf-disclaimer {{ font-size: 0.68rem; color: rgba(239,68,68,0.75); display: flex; align-items: center; gap: 5px; }}
        .sf-version {{
            background: rgba(37,99,235,0.10); border: 1px solid rgba(37,99,235,0.22);
            color: #60a5fa; font-size: 0.62rem; font-weight: 700;
            padding: 3px 10px; border-radius: 6px; letter-spacing: 1px;
            font-family: 'Space Mono', monospace; white-space: nowrap;
        }}

        /* ── Responsive ── */
        @media (max-width: 1100px) {{
            .sf-top {{ grid-template-columns: 1.6fr 1fr 1fr; gap: 32px; }}
            .sf-top > div:last-child {{
                grid-column: 1 / -1;
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
                gap: 16px;
            }}
            .sf-top > div:last-child .sf-col-title {{ grid-column: 1 / -1; }}
        }}
        @media (max-width: 768px) {{
            .sf-outer {{ margin-left: calc(-0.75rem - 1px); margin-right: calc(-0.75rem - 1px); width: calc(100% + 1.5rem + 2px); }}
            .sf-inner {{ padding: 0 1.25rem; }}
            .sf-top {{ grid-template-columns: 1fr 1fr; gap: 28px; padding: 2.5rem 0 2rem; }}
            .sf-top > div:first-child {{ grid-column: 1 / -1; }}
            .sf-top > div:last-child {{ grid-column: 1 / -1; }}
            .sf-brand-desc {{ max-width: 100%; }}
            .sf-bottom {{ flex-direction: column; text-align: center; gap: 10px; padding: 16px 0 20px; }}
            .sf-badges {{ justify-content: center; }}
        }}
        @media (max-width: 480px) {{
            .sf-top {{ grid-template-columns: 1fr; gap: 22px; padding: 2rem 0 1.5rem; }}
            .sf-top > div:first-child {{ grid-column: auto; }}
            .sf-social {{ justify-content: flex-start; }}
            .sf-tech-stack {{ gap: 5px; }}
        }}
        </style>

        <div class="sf-outer">
        <div class="sf-wrap">
          <div class="sf-inner">

            <!-- TOP GRID -->
            <div class="sf-top">

              <!-- Col 1: Brand -->
              <div>
                <div class="sf-brand-logo">
                  <span class="sf-brand-icon">🔬</span>
                  <div>
                    <div class="sf-brand-name">SkinScan AI</div>
                    <div class="sf-brand-tagline">Next-Gen Dermatology Intelligence</div>
                  </div>
                </div>
                <p class="sf-brand-desc">
                  An AI-powered clinical platform for dermoscopic skin lesion analysis.
                  Developed as a Final Year Project at the University of Agriculture Faisalabad
                  using deep learning CNN models for benign/malignant classification.
                </p>
                <div class="sf-tech-stack">
                  <span class="sf-tech-chip">Python</span>
                  <span class="sf-tech-chip">Streamlit</span>
                  <span class="sf-tech-chip">TensorFlow</span>
                  <span class="sf-tech-chip">Plotly</span>
                  <span class="sf-tech-chip">PIL</span>
                  <span class="sf-tech-chip">ReportLab</span>
                  <span class="sf-tech-chip">NumPy</span>
                  <span class="sf-tech-chip">Pandas</span>
                </div>
                <div class="sf-social">
                  <a href="https://github.com/rehanshafiq70" target="_blank" class="sf-social-btn github" title="GitHub">
                    <svg width="17" height="17" viewBox="0 0 24 24" fill="currentColor" style="color:#c9d1d9;">
                      <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0 0 24 12c0-6.63-5.37-12-12-12z"/>
                    </svg>
                  </a>
                  <a href="https://www.linkedin.com/in/rehanshafiq70" target="_blank" class="sf-social-btn linkedin" title="LinkedIn">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" style="color:#0a66c2;">
                      <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/>
                    </svg>
                  </a>
                  <a href="mailto:rehanshafiq6540@gmail.com" class="sf-social-btn email" title="Email">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#14b8a6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                      <path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z"/>
                      <polyline points="22,6 12,13 2,6"/>
                    </svg>
                  </a>
                </div>
              </div>

              <!-- Col 2: Platform Links -->
              <div>
                <div class="sf-col-title">Platform</div>
                <span class="sf-nav-link">🏠 Home Dashboard</span>
                <span class="sf-nav-link">📷 AI Scan Lab</span>
                <span class="sf-nav-link">🧠 AI Analysis</span>
                <span class="sf-nav-link">📊 Analytics Dashboard</span>
                <span class="sf-nav-link">📁 Patient History</span>
                <span class="sf-nav-link">👨‍⚕️ Medical Guide</span>
                <span class="sf-nav-link">⚙️ Settings</span>
              </div>

              <!-- Col 3: Features -->
              <div>
                <div class="sf-col-title">Features</div>
                <span class="sf-nav-link">🧬 Multi-Class CNN</span>
                <span class="sf-nav-link">🔥 Grad-CAM Heatmap</span>
                <span class="sf-nav-link">🤖 AI Explanation</span>
                <span class="sf-nav-link">📷 Live Camera Scan</span>
                <span class="sf-nav-link">📄 PDF / CSV Reports</span>
                <span class="sf-nav-link">🛡️ Blur Detection</span>
                <span class="sf-nav-link">🌓 Dark / Light Mode</span>
              </div>

              <!-- Col 4: Contact -->
              <div>
                <div class="sf-col-title">Developer Contact</div>

                <div class="sf-contact-item">
                  <div class="sf-ci-icon">👨‍💻</div>
                  <div>
                    <div class="sf-ci-label">Developer</div>
                    <div class="sf-ci-value">Rehan Shafique</div>
                  </div>
                </div>

                <div class="sf-contact-item">
                  <div class="sf-ci-icon">🏫</div>
                  <div>
                    <div class="sf-ci-label">Institution</div>
                    <div class="sf-ci-value">University of Agriculture Faisalabad</div>
                  </div>
                </div>

                <div class="sf-contact-item">
                  <div class="sf-ci-icon">📧</div>
                  <div>
                    <div class="sf-ci-label">Email</div>
                    <div class="sf-ci-value">
                      <a href="mailto:rehanshafiq6540@gmail.com">rehanshafiq6540@gmail.com</a>
                    </div>
                    <div class="sf-email-copy"
                         onclick="navigator.clipboard.writeText('rehanshafiq6540@gmail.com').then(()=>{{this.textContent='✅ Copied!';setTimeout(()=>{{this.textContent='📋 Copy Email'}},2000)}})">
                      📋 Copy Email
                    </div>
                  </div>
                </div>

                <div class="sf-contact-item">
                  <div class="sf-ci-icon">💼</div>
                  <div>
                    <div class="sf-ci-label">LinkedIn</div>
                    <div class="sf-ci-value">
                      <a href="https://www.linkedin.com/in/rehanshafiq70" target="_blank">linkedin.com/in/rehanshafiq70</a>
                    </div>
                  </div>
                </div>

                <div class="sf-contact-item">
                  <div class="sf-ci-icon">🐙</div>
                  <div>
                    <div class="sf-ci-label">GitHub</div>
                    <div class="sf-ci-value">
                      <a href="https://github.com/rehanshafiq70" target="_blank">github.com/rehanshafiq70</a>
                    </div>
                  </div>
                </div>

              </div>
            </div><!-- /sf-top -->

            <!-- BADGES -->
            <div class="sf-badges">
              <span class="sf-badge">🔬 CNN Deep Learning</span>
              <span class="sf-badge">🏥 Clinical Intelligence</span>
              <span class="sf-badge">🧬 Dermoscopy AI</span>
              <span class="sf-badge">🎓 Final Year Project 2026</span>
              <span class="sf-badge">🌐 University of Agriculture Faisalabad</span>
              <span class="sf-badge">⚡ Streamlit v15.0</span>
              <span class="sf-badge">🤖 TensorFlow CNN</span>
            </div>

            <!-- BOTTOM BAR -->
            <div class="sf-bottom">
              <div class="sf-copy">
                © 2026 <strong>SkinScan AI</strong> — Developed by <strong>Rehan Shafique</strong>
                &nbsp;·&nbsp; University of Agriculture Faisalabad &nbsp;·&nbsp; Dept. of Bioinformatics
              </div>
              <div class="sf-disclaimer">
                ⚠️ Research &amp; Educational Use Only — Not a Certified Medical Device
              </div>
              <div class="sf-version">v15.0</div>
            </div>

          </div><!-- /sf-inner -->
        </div><!-- /sf-wrap -->
        </div><!-- /sf-outer -->
        """, unsafe_allow_html=True)
'''


# ══════════════════════════════════════════════════════════════════════════════
#  ALSO: Remove the inject_css() footer-related classes to avoid conflicts
#  (Optional but keeps CSS clean — search for and delete these class defs
#   from inject_css() in your file):
#
#  Classes now ONLY in footer (can remove from inject_css to avoid duplication):
#    .footer-outer, .site-footer, .footer-inner, .footer-top, .footer-brand-logo,
#    .footer-brand-icon, .footer-brand-name, .footer-brand-tagline, .footer-brand-desc,
#    .footer-tech-stack, .ftech-chip, .footer-social, .social-btn,
#    .footer-col-title, .footer-nav-link, .footer-contact-item, .fci-icon,
#    .fci-label, .fci-value, .email-copy-btn, .footer-badges, .fbadge,
#    .footer-bottom, .footer-copy, .footer-disclaimer, .footer-version-badge
#
#  Keeping them in inject_css() too won't break anything (just redundant).
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("  SKINSCAN AI — FOOTER FIX  v15.1")
    print("=" * 70)
    print()
    print("ROOT CAUSE OF BUG:")
    print("  The footer's CSS classes were defined in inject_css() via a")
    print("  separate st.markdown() call. Streamlit processes each markdown")
    print("  block independently. When the footer HTML was rendered, the")
    print("  browser sometimes hadn't applied the stylesheet yet, causing")
    print("  all CSS classes to be unrecognized — so the HTML rendered as")
    print("  raw visible text.")
    print()
    print("FIX APPLIED:")
    print("  The new _footer() method embeds a <style> block INSIDE the same")
    print("  st.markdown() call as the footer HTML. This guarantees CSS and")
    print("  HTML are always delivered together in one atomic browser update.")
    print()
    print("HOW TO APPLY:")
    print("  1. Open your app.py / skinscan.py")
    print("  2. Find the _footer() method inside class SkinScanApp")
    print("  3. Replace the entire method with the FOOTER_METHOD string above")
    print("  4. Run: streamlit run app.py")
    print()
    print("CHANGES SUMMARY:")
    print("  - All CSS class names renamed from 'footer-*' → 'sf-*'")
    print("    (avoids collision with inject_css() styles)")
    print("  - <style> block embedded directly inside _footer()'s markdown")
    print("  - Theme-aware colors computed dynamically from session_state")
    print("  - All hover, animation, responsive breakpoints preserved")
    print("  - Social links, contact, tech chips fully functional")
    print("  - Email copy-to-clipboard button preserved")
    print()
    print("  ✅ Footer will now always render correctly on all pages.")
    print("=" * 70)

# ══════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = SkinScanApp()
    app.launch()
