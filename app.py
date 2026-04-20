"""
=============================================================
SkinCancerPrediction AI — Medical Diagnostic System v12.0
University of Agricultural Faisalabad | Rehan Shafique
Architecture: OOP + Streamlit | Real Deep Learning Inference
=============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import time
import datetime
import plotly.express as px
import plotly.graph_objects as go
import io
import os
import cv2
import base64
import uuid

# ─────────────────────────────────────────────
# PAGE CONFIG (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SkinCancerPrediction AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# 1. MODEL LOADER
# ─────────────────────────────────────────────
class ModelLoader:
    """
    Loads the real CNN model from disk.
    If not found, attempts to download via gdown from Google Drive.
    No random / simulation fallback — real inference only.
    """
    MODEL_FILENAME = "skin_cancer_model.h5"
    # Replace with your actual Google Drive file ID
    GDRIVE_FILE_ID = "YOUR_GOOGLE_DRIVE_FILE_ID_HERE"

    def __init__(self):
        self.model = None
        self.is_online = False
        self.status_message = ""
        self._load()

    def _download_model(self):
        try:
            import gdown
            url = f"https://drive.google.com/uc?id={self.GDRIVE_FILE_ID}"
            gdown.download(url, self.MODEL_FILENAME, quiet=False)
            return os.path.exists(self.MODEL_FILENAME)
        except Exception as e:
            self.status_message = f"Download failed: {e}"
            return False

    def _load(self):
        from tensorflow.keras.models import load_model
        if not os.path.exists(self.MODEL_FILENAME):
            self.status_message = "Model not found locally. Attempting download..."
            if not self._download_model():
                self.status_message = (
                    "❌ Model file not found. Place `skin_cancer_model.h5` in the app directory "
                    "or set a valid GDRIVE_FILE_ID."
                )
                return
        try:
            self.model = load_model(self.MODEL_FILENAME)
            self.is_online = True
            self.status_message = "✅ CNN Model loaded successfully"
        except Exception as e:
            self.status_message = f"❌ Model load error: {e}"

    def predict(self, pil_image: Image.Image):
        """Real inference — no random output."""
        if not self.is_online or self.model is None:
            raise RuntimeError("Model is not loaded. Cannot perform inference.")

        from tensorflow.keras.preprocessing.image import img_to_array
        img = pil_image.convert("RGB").resize((224, 224))
        arr = img_to_array(img) / 255.0
        arr = np.expand_dims(arr, axis=0)
        raw_score = float(self.model.predict(arr, verbose=0)[0][0])

        if raw_score >= 0.5:
            diagnosis = "Malignant"
            confidence = raw_score
        else:
            diagnosis = "Benign"
            confidence = 1.0 - raw_score

        return diagnosis, confidence, raw_score


# ─────────────────────────────────────────────
# 2. IMAGE VALIDATOR
# ─────────────────────────────────────────────
class ImageValidator:
    """Validates uploaded images before feeding to the model."""

    MIN_SIZE = 128
    BLUR_THRESHOLD = 80.0

    @classmethod
    def validate(cls, pil_image: Image.Image):
        errors = []
        # RGB check
        if pil_image.mode not in ("RGB", "RGBA"):
            errors.append("Image must be RGB.")
        # Size check
        w, h = pil_image.size
        if w < cls.MIN_SIZE or h < cls.MIN_SIZE:
            errors.append(f"Image too small ({w}×{h}). Minimum: {cls.MIN_SIZE}×{cls.MIN_SIZE}.")
        # Blur check
        img_rgb = pil_image.convert("RGB")
        cv_img = np.array(img_rgb)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if lap_var < cls.BLUR_THRESHOLD:
            errors.append(f"Image appears blurry (sharpness score: {lap_var:.1f}). Upload a clearer photo.")
        return errors, lap_var


# ─────────────────────────────────────────────
# 3. CLINICAL KNOWLEDGE BASE
# ─────────────────────────────────────────────
class ClinicalKnowledgeBase:
    """Evidence-based medical recommendations per diagnosis."""

    DATA = {
        "Malignant": {
            "color": "#ef4444",
            "badge_bg": "rgba(239,68,68,0.15)",
            "risk_label": "CRITICAL RISK",
            "short_desc": (
                "High-probability malignancy detected. Immediate clinical evaluation is strongly advised. "
                "This result is for screening purposes only and must be confirmed by a board-certified dermatologist."
            ),
            "treatments": [
                "Wide Local Excision (WLE) — primary surgical removal",
                "Mohs Micrographic Surgery — for margin-sensitive regions",
                "Sentinel Lymph Node Biopsy (SLNB) — staging evaluation",
                "Adjuvant immunotherapy (e.g., PD-1 inhibitors) if indicated",
                "Radiation therapy mapping if surgical margins are unclear",
                "PET/CT scan if systemic metastasis is suspected",
            ],
            "patient_care": [
                "Seek immediate consultation with a board-certified dermatologist",
                "Urgent referral to Oncology/Onco-Dermatology department",
                "Avoid UV exposure — wear UPF 50+ protective clothing",
                "Apply broad-spectrum SPF 50+ daily to all exposed skin",
                "Monitor the lesion for rapid changes, ulceration, or bleeding",
                "Maintain a photographic record of the lesion for comparison",
            ],
            "monitoring": [
                "Full-body dermoscopy every 3 months post-treatment",
                "Excisional biopsy for Breslow depth determination",
                "Monthly ABCDE self-examinations",
                "Immediate ER presentation if ulceration becomes severe",
                "Family members screened for hereditary melanoma syndromes",
            ],
        },
        "Benign": {
            "color": "#10b981",
            "badge_bg": "rgba(16,185,129,0.15)",
            "risk_label": "LOW RISK",
            "short_desc": (
                "No high-risk features detected in this lesion. Continue routine monitoring "
                "and consult a dermatologist if morphological changes occur."
            ),
            "treatments": [
                "No immediate surgical intervention required",
                "Elective cosmetic laser ablation if aesthetically desired",
                "Targeted cryotherapy for symptomatic benign lesions",
                "Diagnostic shave biopsy available upon patient request",
                "Digital dermoscopic baseline photography for future comparison",
            ],
            "patient_care": [
                "Maintain daily SPF 50+ sunscreen application",
                "Use ceramide-based moisturizers for barrier repair",
                "Perform monthly ABCDE self-skin examinations",
                "Avoid mechanical trauma or irritation to the lesion",
                "Dietary antioxidant intake (Vitamins C, E, selenium)",
            ],
            "monitoring": [
                "Annual routine dermatology screening",
                "AI re-evaluation in 6 months",
                "Consult immediately if lesion changes shape, color, or size",
                "Rule out atypical nevi syndrome if multiple lesions present",
                "Monitor for development of satellite lesions nearby",
            ],
        },
    }

    @classmethod
    def get(cls, diagnosis: str) -> dict:
        return cls.DATA.get(diagnosis, cls.DATA["Benign"])

    @staticmethod
    def risk_level(confidence: float) -> tuple:
        if confidence < 0.50:
            return "LOW", "#10b981"
        elif confidence < 0.75:
            return "MEDIUM", "#f59e0b"
        else:
            return "CRITICAL", "#ef4444"


# ─────────────────────────────────────────────
# 4. REPORT GENERATOR
# ─────────────────────────────────────────────
class ReportGenerator:
    """Generates downloadable plain-text diagnostic reports."""

    @staticmethod
    def build(record: dict, info: dict) -> str:
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        risk, _ = ClinicalKnowledgeBase.risk_level(float(record["Confidence_Raw"]))
        sep = "=" * 65
        lines = [
            sep,
            "     SKINCANCERPREDICTION AI — DIAGNOSTIC REPORT v12.0",
            "     University of Agricultural Faisalabad",
            sep,
            f"  Report Generated : {now}",
            f"  Scan ID          : {record['Scan_ID']}",
            f"  Patient ID       : {record['Patient_ID']}",
            f"  Patient Name     : {record['Patient_Name']}",
            sep,
            "  AI DIAGNOSIS SUMMARY",
            sep,
            f"  Result           : {record['AI_Diagnosis']}",
            f"  Confidence Score : {record['Confidence_Score']}",
            f"  Risk Level       : {risk}",
            "",
            f"  {info['short_desc']}",
            sep,
            "  RECOMMENDED TREATMENTS",
            sep,
        ]
        for i, t in enumerate(info["treatments"], 1):
            lines.append(f"  {i}. {t}")
        lines += [
            sep,
            "  PATIENT CARE INSTRUCTIONS",
            sep,
        ]
        for i, c in enumerate(info["patient_care"], 1):
            lines.append(f"  {i}. {c}")
        lines += [
            sep,
            "  MONITORING PROTOCOL",
            sep,
        ]
        for i, m in enumerate(info["monitoring"], 1):
            lines.append(f"  {i}. {m}")
        lines += [
            sep,
            "  DISCLAIMER",
            sep,
            "  This AI system is a screening aid only. It does NOT replace",
            "  professional medical diagnosis. Always consult a licensed",
            "  dermatologist or oncologist for clinical decisions.",
            sep,
            "  SkinCancerPrediction AI — Medical Diagnostic System v12.0",
            "  Developer : Rehan Shafique | rehanshafiq6540@gmail.com",
            sep,
        ]
        return "\n".join(lines)


# ─────────────────────────────────────────────
# 5. GLOBAL CSS — DARK MEDICAL / NEON DESIGN
# ─────────────────────────────────────────────
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@400;500&display=swap');

    :root {
        --bg:        #050a0e;
        --surface:   #0b1520;
        --surface2:  #0f1e2e;
        --border:    rgba(56,189,248,0.12);
        --border-hi: rgba(56,189,248,0.35);
        --text:      #e2e8f0;
        --muted:     #64748b;
        --accent:    #38bdf8;
        --green:     #10b981;
        --red:       #ef4444;
        --amber:     #f59e0b;
        --purple:    #a78bfa;
        --font-head: 'Syne', sans-serif;
        --font-mono: 'DM Mono', monospace;
    }

    html, body, [class*="css"] {
        font-family: var(--font-head);
        background-color: var(--bg) !important;
        color: var(--text);
    }

    /* ── Remove default Streamlit chrome ── */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding: 1.5rem 2.5rem 4rem; max-width: 1400px; margin: 0 auto; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: var(--surface) !important;
        border-right: 1px solid var(--border);
    }
    [data-testid="stSidebar"] * { color: var(--text) !important; }

    /* ── Glass cards ── */
    .glass {
        background: linear-gradient(135deg, rgba(15,30,46,0.85), rgba(11,21,32,0.9));
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.6rem 1.8rem;
        backdrop-filter: blur(14px);
        margin-bottom: 1.2rem;
        transition: border-color 0.25s;
    }
    .glass:hover { border-color: var(--border-hi); }

    /* ── Neon headings ── */
    .neon-title {
        font-family: var(--font-head);
        font-weight: 800;
        font-size: 2.8rem;
        background: linear-gradient(120deg, #38bdf8 0%, #818cf8 50%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
        line-height: 1.1;
    }
    .section-title {
        font-family: var(--font-head);
        font-weight: 700;
        font-size: 1.5rem;
        color: var(--accent);
        letter-spacing: 0.3px;
        margin-bottom: 0.3rem;
    }
    .mono { font-family: var(--font-mono); color: var(--accent); font-size: 0.82rem; }

    /* ── Metric tiles ── */
    .metric-tile {
        background: var(--surface2);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        text-align: center;
        transition: transform 0.2s, border-color 0.2s;
    }
    .metric-tile:hover { transform: translateY(-3px); border-color: var(--border-hi); }
    .metric-val { font-size: 2rem; font-weight: 800; color: var(--accent); }
    .metric-lbl { font-size: 0.78rem; color: var(--muted); margin-top: 0.2rem; font-family: var(--font-mono); }
    .metric-sub { font-size: 0.75rem; color: var(--green); margin-top: 0.15rem; }

    /* ── Steps ── */
    .step-item {
        display: flex; align-items: center; gap: 0.9rem;
        padding: 0.7rem 1rem; border-left: 3px solid var(--accent);
        background: rgba(56,189,248,0.04); border-radius: 0 10px 10px 0;
        margin-bottom: 0.5rem;
    }
    .step-num {
        background: var(--accent); color: #050a0e; border-radius: 50%;
        width: 26px; height: 26px; font-weight: 800; font-size: 0.8rem;
        display: flex; align-items: center; justify-content: center; flex-shrink: 0;
    }

    /* ── Feature cards ── */
    .feat-card {
        background: var(--surface2);
        border: 1px solid var(--border);
        border-radius: 14px; padding: 1.2rem; text-align: center;
    }
    .feat-icon { font-size: 2rem; margin-bottom: 0.5rem; }
    .feat-name { font-weight: 700; font-size: 1rem; color: var(--accent); }
    .feat-desc { font-size: 0.8rem; color: var(--muted); margin-top: 0.3rem; }

    /* ── Diagnosis badge ── */
    .diag-badge {
        display: inline-block;
        border-radius: 8px;
        padding: 0.3rem 1rem;
        font-weight: 700;
        font-size: 1.1rem;
        font-family: var(--font-mono);
        letter-spacing: 1.5px;
    }
    .risk-badge {
        display: inline-block;
        border-radius: 6px;
        padding: 0.2rem 0.8rem;
        font-weight: 600;
        font-size: 0.8rem;
        font-family: var(--font-mono);
        letter-spacing: 1px;
    }

    /* ── Recommendation lists ── */
    .rec-item {
        display: flex; align-items: flex-start; gap: 0.6rem;
        padding: 0.55rem 0; border-bottom: 1px solid rgba(56,189,248,0.06);
        font-size: 0.88rem;
    }
    .rec-num { color: var(--accent); font-weight: 700; min-width: 1.3rem; font-family: var(--font-mono); }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #0369a1, #1d4ed8) !important;
        color: white !important; border: none !important;
        border-radius: 10px !important; font-weight: 700 !important;
        letter-spacing: 0.8px !important; padding: 0.7rem 1.2rem !important;
        text-transform: uppercase !important; transition: all 0.2s !important;
        font-family: var(--font-head) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 0 18px rgba(56,189,248,0.4) !important;
    }
    .stDownloadButton > button {
        background: linear-gradient(135deg, #065f46, #047857) !important;
        color: white !important; border: none !important;
        border-radius: 10px !important; font-weight: 700 !important;
        font-family: var(--font-head) !important;
    }

    /* ── File uploader ── */
    [data-testid="stFileUploader"] {
        background: var(--surface2) !important;
        border: 2px dashed var(--border-hi) !important;
        border-radius: 14px !important;
        padding: 1rem !important;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab"] {
        font-family: var(--font-head) !important;
        font-weight: 600 !important; color: var(--muted) !important;
    }
    .stTabs [aria-selected="true"] {
        color: var(--accent) !important;
        border-bottom: 2px solid var(--accent) !important;
    }

    /* ── Inputs ── */
    .stTextInput input, .stNumberInput input {
        background: var(--surface2) !important;
        border: 1px solid var(--border-hi) !important;
        border-radius: 8px !important;
        color: var(--text) !important;
        font-family: var(--font-mono) !important;
    }

    /* ── Spinner ── */
    .stSpinner { color: var(--accent) !important; }

    /* ── Dataframe ── */
    [data-testid="stDataFrame"] { border-radius: 12px !important; overflow: hidden; }

    /* ── Divider ── */
    hr { border-color: var(--border) !important; }

    /* ── Sidebar nav ── */
    .sidebar-nav-item {
        padding: 0.6rem 0.9rem; border-radius: 10px;
        cursor: pointer; transition: background 0.2s;
        font-weight: 600; font-size: 0.92rem;
        display: flex; align-items: center; gap: 0.6rem;
        margin-bottom: 0.3rem;
    }
    .sidebar-nav-item:hover { background: rgba(56,189,248,0.08); }
    .sidebar-nav-active {
        background: rgba(56,189,248,0.14) !important;
        border-left: 3px solid var(--accent);
        color: var(--accent) !important;
    }
    
    /* ── Model status bar ── */
    .status-bar {
        border-radius: 10px; padding: 0.5rem 1rem;
        font-size: 0.8rem; font-family: var(--font-mono);
        display: flex; align-items: center; gap: 0.5rem;
    }
    .status-online  { background: rgba(16,185,129,0.12); border: 1px solid rgba(16,185,129,0.3); color: #10b981; }
    .status-offline { background: rgba(239,68,68,0.12);  border: 1px solid rgba(239,68,68,0.3);  color: #ef4444; }

    /* ── Pulse dot ── */
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.3} }
    .dot-pulse { width:9px; height:9px; border-radius:50%; display:inline-block; animation:pulse 1.4s infinite; }
    .dot-green { background:#10b981; }
    .dot-red   { background:#ef4444; }
    </style>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 6. SESSION STATE INIT
# ─────────────────────────────────────────────
def init_state():
    if "registry" not in st.session_state:
        st.session_state.registry = []
    if "page" not in st.session_state:
        st.session_state.page = "Home"
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "model_loader" not in st.session_state:
        with st.spinner("Initializing CNN model..."):
            st.session_state.model_loader = ModelLoader()


# ─────────────────────────────────────────────
# 7. SIDEBAR
# ─────────────────────────────────────────────
def render_sidebar():
    ml: ModelLoader = st.session_state.model_loader

    with st.sidebar:
        st.markdown("""
        <div style='padding: 0.5rem 0 1.2rem;'>
            <div style='font-size:1.5rem; font-weight:800; color:#38bdf8; font-family:Syne,sans-serif;'>
                🧬 SkinCancerPrediction
            </div>
            <div style='font-size:0.72rem; color:#64748b; font-family:"DM Mono",monospace; margin-top:3px;'>
                MEDICAL DIAGNOSTIC SYSTEM v12.0
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        pages = [
            ("🏠", "Home"),
            ("🔬", "AI Scanner"),
            ("📊", "Analytics"),
            ("🗂️", "Registry"),
        ]
        for icon, name in pages:
            active = st.session_state.page == name
            cls = "sidebar-nav-item sidebar-nav-active" if active else "sidebar-nav-item"
            if st.button(f"{icon}  {name}", key=f"nav_{name}", use_container_width=True):
                st.session_state.page = name
                st.rerun()

        st.divider()

        # Model status
        if ml.is_online:
            st.markdown("""
            <div class='status-bar status-online'>
                <span class='dot-pulse dot-green'></span>
                CNN Model Active
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='status-bar status-offline'>
                <span class='dot-pulse dot-red'></span>
                Model Not Loaded
            </div>""", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:0.72rem;color:#64748b;margin-top:0.4rem;'>{ml.status_message}</div>",
                        unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        scans = len(st.session_state.registry)
        benign = sum(1 for r in st.session_state.registry if r["AI_Diagnosis"] == "Benign")
        malignant = scans - benign
        st.markdown(f"""
        <div style='font-size:0.78rem; color:#64748b; font-family:"DM Mono",monospace; line-height:2;'>
            Total Scans: <b style='color:#e2e8f0;'>{scans}</b><br>
            Malignant:   <b style='color:#ef4444;'>{malignant}</b><br>
            Benign:      <b style='color:#10b981;'>{benign}</b>
        </div>
        """, unsafe_allow_html=True)

        st.divider()
        st.markdown("""
        <div style='font-size:0.7rem; color:#334155; text-align:center; font-family:"DM Mono",monospace; line-height:1.8;'>
            Rehan Shafique<br>
            Univ. of Agricultural Faisalabad<br>
            rehanshafiq6540@gmail.com
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 8. PAGE: HOME
# ─────────────────────────────────────────────
def page_home():
    # ── Hero ──────────────────────────────────
    st.markdown("""
    <div class='glass' style='padding:3rem 2.5rem; text-align:center; margin-bottom:2rem;
         background:linear-gradient(135deg,rgba(3,105,161,0.15),rgba(99,102,241,0.12),rgba(244,114,182,0.08));
         border-color:rgba(56,189,248,0.25);'>
        <div style='font-size:0.82rem; font-family:"DM Mono",monospace; color:#38bdf8;
             letter-spacing:3px; text-transform:uppercase; margin-bottom:0.8rem;'>
            AI-POWERED CLINICAL SCREENING
        </div>
        <div class='neon-title'>SkinCancerPrediction AI</div>
        <div style='color:#94a3b8; font-size:1.1rem; margin:1rem 0 1.8rem; max-width:600px; margin-left:auto; margin-right:auto;'>
            Hospital-grade deep learning diagnosis for skin lesion classification —
            <em>Benign</em> vs <em>Malignant</em> — powered by a real CNN trained on melanoma data.
        </div>
        <div style='font-size:0.75rem; color:#475569; font-family:"DM Mono",monospace;'>
            ⚠ Screening aid only — always confirm with a licensed dermatologist
        </div>
    </div>
    """, unsafe_allow_html=True)

    cta_col = st.columns([1, 0.4, 1])
    with cta_col[1]:
        if st.button("▶ START DIAGNOSIS", key="hero_cta"):
            st.session_state.page = "AI Scanner"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Step guide ────────────────────────────
    st.markdown("<div class='section-title'>📋 How It Works</div>", unsafe_allow_html=True)
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    steps = [
        ("Upload a dermoscopic skin lesion image (JPG / PNG)"),
        ("Enter optional patient ID and name for the registry"),
        ("Click EXECUTE SCAN to start AI inference"),
        ("Wait 2–4 seconds for deep learning processing"),
        ("View diagnosis, confidence score, and risk level"),
        ("Download medical report or export registry as CSV"),
    ]
    for i, s in enumerate(steps, 1):
        st.markdown(f"""
        <div class='step-item'>
            <div class='step-num'>{i}</div>
            <div style='font-size:0.9rem;'>{s}</div>
        </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Feature cards ─────────────────────────
    st.markdown("<div class='section-title'>⚡ Platform Features</div>", unsafe_allow_html=True)
    features = [
        ("🧠", "Real AI Inference", "CNN trained on Kaggle melanoma dataset — no random outputs"),
        ("🔍", "Image Validation", "Blur detection, size check & RGB validation before scanning"),
        ("📑", "Medical Reports", "One-click downloadable diagnostic report per patient"),
        ("📊", "Analytics Dashboard", "Live charts: scan distribution, confidence trends"),
        ("🗂️", "Patient Registry", "Searchable table with CSV export functionality"),
        ("⚡", "Fast Inference", "224×224 preprocessing + optimized Keras prediction"),
    ]
    cols = st.columns(3)
    for i, (icon, name, desc) in enumerate(features):
        with cols[i % 3]:
            st.markdown(f"""
            <div class='feat-card'>
                <div class='feat-icon'>{icon}</div>
                <div class='feat-name'>{name}</div>
                <div class='feat-desc'>{desc}</div>
            </div><br>""", unsafe_allow_html=True)

    # ── Dataset stats ─────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>📊 Dataset Reference (ISIC Melanoma — Kaggle)</div>", unsafe_allow_html=True)
    d1, d2, d3, d4 = st.columns(4)
    tiles = [
        ("33,126", "Total Training Images", "ISIC Archive"),
        ("26,501", "Benign Samples", "≈ 80%"),
        ("6,625",  "Malignant Samples", "≈ 20%"),
        ("224×224", "Input Resolution", "RGB Normalized"),
    ]
    for col, (val, lbl, sub) in zip([d1, d2, d3, d4], tiles):
        with col:
            st.markdown(f"""
            <div class='metric-tile'>
                <div class='metric-val'>{val}</div>
                <div class='metric-lbl'>{lbl}</div>
                <div class='metric-sub'>{sub}</div>
            </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 9. PAGE: AI SCANNER
# ─────────────────────────────────────────────
def page_scanner():
    ml: ModelLoader = st.session_state.model_loader

    st.markdown("<div class='neon-title' style='font-size:2rem; margin-bottom:1.2rem;'>🔬 AI Diagnostic Scanner</div>",
                unsafe_allow_html=True)

    if not ml.is_online:
        st.error(f"**Model not available.** {ml.status_message}\n\n"
                 "Place `skin_cancer_model.h5` in the app directory or configure a valid Google Drive file ID.")
        return

    left, right = st.columns([1, 1.15])

    # ── LEFT: upload + patient info ──────────
    with left:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Patient Information</div>", unsafe_allow_html=True)
        patient_id   = st.text_input("Patient ID",   placeholder="e.g. PT-2025-001")
        patient_name = st.text_input("Patient Name", placeholder="e.g. John Doe")

        st.markdown("<div class='section-title' style='margin-top:1rem;'>Upload Skin Lesion Image</div>",
                    unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Accepted formats: JPG, JPEG, PNG",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )

        if uploaded:
            pil_img = Image.open(uploaded)
            st.image(pil_img, use_container_width=True, caption=f"📷 {uploaded.name}")
            w, h = pil_img.size
            st.markdown(f"<div class='mono'>Resolution: {w}×{h} px  |  Mode: {pil_img.mode}</div>",
                        unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        scan_btn = st.button("▶  EXECUTE SCAN", disabled=(not uploaded))
        st.markdown("</div>", unsafe_allow_html=True)

    # ── RIGHT: results ───────────────────────
    with right:
        if uploaded and scan_btn:
            pil_img = Image.open(uploaded)

            # Validation
            errors, sharpness = ImageValidator.validate(pil_img)
            if errors:
                st.error("❌ **Invalid or unclear image.** Please upload a proper dermoscopic skin image.\n\n"
                         + "\n".join(f"• {e}" for e in errors))
                return

            with st.spinner("Extracting feature vectors — running inference..."):
                time.sleep(1.2)  # UX delay for spinner visibility
                diagnosis, confidence, raw_score = ml.predict(pil_img)

            info = ClinicalKnowledgeBase.get(diagnosis)
            risk_lbl, risk_color = ClinicalKnowledgeBase.risk_level(confidence)
            scan_id = f"SKN-{uuid.uuid4().hex[:8].upper()}"
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Save to registry
            record = {
                "Scan_ID":          scan_id,
                "Patient_ID":       patient_id  if patient_id  else "—",
                "Patient_Name":     patient_name if patient_name else "—",
                "AI_Diagnosis":     diagnosis,
                "Confidence_Score": f"{confidence*100:.2f}%",
                "Confidence_Raw":   confidence,
                "Risk_Level":       risk_lbl,
                "Timestamp":        ts,
            }
            st.session_state.registry.append(record)
            st.session_state.last_result = record

            # ── Result card ──────────────────
            st.markdown(f"""
            <div class='glass' style='border-color:{info["color"]}55;'>
                <div style='display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.5rem;'>
                    <div>
                        <div style='font-size:0.75rem; color:#64748b; font-family:"DM Mono",monospace; letter-spacing:2px;'>
                            AI DIAGNOSIS
                        </div>
                        <div style='font-size:2rem; font-weight:800; color:{info["color"]}; font-family:Syne,sans-serif;'>
                            {diagnosis.upper()}
                        </div>
                    </div>
                    <div style='text-align:right;'>
                        <span class='diag-badge' style='background:{info["badge_bg"]}; color:{info["color"]}; border:1px solid {info["color"]}44;'>
                            {info["risk_label"]}
                        </span><br><br>
                        <span class='risk-badge' style='background:rgba({",".join(str(int(risk_color.lstrip("#")[i:i+2],16)) for i in (0,2,4))},0.15);
                              color:{risk_color}; border:1px solid {risk_color}44;'>
                            ⚠ RISK: {risk_lbl}
                        </span>
                    </div>
                </div>
                <div style='color:#94a3b8; font-size:0.85rem; margin-top:0.8rem;'>
                    {info["short_desc"]}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Gauge + probability bars ─────
            c1, c2 = st.columns(2)
            with c1:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=round(confidence * 100, 1),
                    title={"text": "Confidence", "font": {"color": "#94a3b8", "size": 13}},
                    number={"suffix": "%", "font": {"color": info["color"], "size": 28}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "#475569"},
                        "bar":  {"color": info["color"]},
                        "bgcolor": "#0b1520",
                        "bordercolor": "#1e293b",
                        "steps": [
                            {"range": [0, 50],  "color": "rgba(16,185,129,0.12)"},
                            {"range": [50, 75], "color": "rgba(245,158,11,0.12)"},
                            {"range": [75, 100],"color": "rgba(239,68,68,0.12)"},
                        ],
                    },
                ))
                fig_gauge.update_layout(
                    height=210, margin=dict(l=10, r=10, t=30, b=0),
                    paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0",
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

            with c2:
                ben_pct = (1 - raw_score) * 100
                mal_pct = raw_score * 100
                fig_bar = go.Figure()
                fig_bar.add_trace(go.Bar(
                    x=[ben_pct], y=["Benign"],  orientation="h",
                    marker_color="#10b981", name="Benign",
                    text=[f"{ben_pct:.1f}%"], textposition="auto",
                ))
                fig_bar.add_trace(go.Bar(
                    x=[mal_pct], y=["Malignant"], orientation="h",
                    marker_color="#ef4444", name="Malignant",
                    text=[f"{mal_pct:.1f}%"], textposition="auto",
                ))
                fig_bar.update_layout(
                    title={"text": "Probability Breakdown", "font": {"color": "#94a3b8", "size": 13}},
                    height=210, margin=dict(l=0, r=10, t=30, b=0),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#e2e8f0", showlegend=False, barmode="group",
                    xaxis={"range": [0, 100], "gridcolor": "#1e293b"},
                    yaxis={"gridcolor": "#1e293b"},
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            # ── Clinical recommendations ─────
            st.markdown("<div class='section-title'>📋 Clinical Recommendations</div>", unsafe_allow_html=True)
            tab1, tab2, tab3 = st.tabs(["🩺 Treatments", "🛡️ Patient Care", "🔭 Monitoring"])
            for tab, key in zip([tab1, tab2, tab3], ["treatments", "patient_care", "monitoring"]):
                with tab:
                    for i, item in enumerate(info[key], 1):
                        st.markdown(f"""
                        <div class='rec-item'>
                            <span class='rec-num'>{i:02d}</span>
                            <span>{item}</span>
                        </div>""", unsafe_allow_html=True)

            # ── Action buttons ───────────────
            st.markdown("<br>", unsafe_allow_html=True)
            b1, b2, b3 = st.columns(3)
            report_text = ReportGenerator.build(record, info)

            with b1:
                st.download_button(
                    "📄 Download Report",
                    data=report_text,
                    file_name=f"SkinCancerAI_{scan_id}.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
            with b2:
                if st.button("📊 View Analytics", use_container_width=True):
                    st.session_state.page = "Analytics"
                    st.rerun()
            with b3:
                if st.button("🗂️ Patient Registry", use_container_width=True):
                    st.session_state.page = "Registry"
                    st.rerun()

        elif not uploaded:
            st.markdown("""
            <div class='glass' style='text-align:center; padding:3.5rem; color:#475569;'>
                <div style='font-size:3.5rem; margin-bottom:0.5rem;'>🔬</div>
                <div style='font-weight:600; font-size:1rem; color:#64748b;'>
                    Upload a dermoscopic image to begin
                </div>
                <div style='font-size:0.8rem; margin-top:0.5rem;'>
                    Real CNN inference — Benign / Malignant
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Medical disclaimer ───────────────────
    st.markdown("""
    <div style='margin-top:2rem; padding:0.8rem 1.2rem; border-radius:10px;
         background:rgba(245,158,11,0.07); border:1px solid rgba(245,158,11,0.2);
         font-size:0.78rem; color:#92400e; font-family:"DM Mono",monospace; color:#fbbf24;'>
        ⚠ DISCLAIMER — This AI system is a <b>screening aid only</b>. It does not constitute medical advice
        and must not replace the judgment of a licensed dermatologist or oncologist.
        All results require clinical confirmation.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 10. PAGE: ANALYTICS
# ─────────────────────────────────────────────
def page_analytics():
    st.markdown("<div class='neon-title' style='font-size:2rem; margin-bottom:1.2rem;'>📊 Analytics Dashboard</div>",
                unsafe_allow_html=True)

    reg = st.session_state.registry
    if not reg:
        st.info("No scan data yet. Run at least one diagnosis to see analytics.")
        return

    df = pd.DataFrame(reg)
    df["Confidence_Pct"] = df["Confidence_Raw"].astype(float) * 100
    total   = len(df)
    benign  = (df["AI_Diagnosis"] == "Benign").sum()
    malign  = (df["AI_Diagnosis"] == "Malignant").sum()
    avg_conf = df["Confidence_Pct"].mean()
    critical = (df["Risk_Level"] == "CRITICAL").sum()

    # ── KPI row ──────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    tiles = [
        (total,             "Total Scans",       "#38bdf8"),
        (benign,            "Benign",             "#10b981"),
        (malign,            "Malignant",          "#ef4444"),
        (f"{avg_conf:.1f}%","Avg Confidence",     "#a78bfa"),
        (critical,          "Critical Alerts",    "#f59e0b"),
    ]
    for col, (val, lbl, color) in zip([k1, k2, k3, k4, k5], tiles):
        with col:
            st.markdown(f"""
            <div class='metric-tile'>
                <div class='metric-val' style='color:{color};'>{val}</div>
                <div class='metric-lbl'>{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts ───────────────────────────────
    row1_l, row1_r = st.columns(2)

    with row1_l:
        fig_pie = px.pie(
            df, names="AI_Diagnosis",
            title="Diagnosis Distribution",
            hole=0.45,
            color="AI_Diagnosis",
            color_discrete_map={"Benign": "#10b981", "Malignant": "#ef4444"},
        )
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0", title_font_color="#38bdf8",
            legend=dict(bgcolor="rgba(0,0,0,0)"),
        )
        st.markdown("<div class='glass' style='padding:1rem;'>", unsafe_allow_html=True)
        st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with row1_r:
        fig_hist = px.histogram(
            df, x="Confidence_Pct", nbins=20,
            title="Confidence Score Distribution",
            color_discrete_sequence=["#38bdf8"],
        )
        fig_hist.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0", title_font_color="#38bdf8",
            xaxis=dict(gridcolor="#1e293b", title="Confidence (%)"),
            yaxis=dict(gridcolor="#1e293b", title="Count"),
        )
        st.markdown("<div class='glass' style='padding:1rem;'>", unsafe_allow_html=True)
        st.plotly_chart(fig_hist, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Confidence over time ──────────────────
    if total > 1:
        fig_line = px.line(
            df.reset_index(), x="index", y="Confidence_Pct",
            color="AI_Diagnosis",
            title="Confidence Trend Over Scans",
            color_discrete_map={"Benign": "#10b981", "Malignant": "#ef4444"},
            markers=True,
        )
        fig_line.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0", title_font_color="#38bdf8",
            xaxis=dict(gridcolor="#1e293b", title="Scan #"),
            yaxis=dict(gridcolor="#1e293b", title="Confidence (%)"),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
        )
        st.markdown("<div class='glass' style='padding:1rem;'>", unsafe_allow_html=True)
        st.plotly_chart(fig_line, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Risk breakdown bar ────────────────────
    risk_counts = df["Risk_Level"].value_counts().reset_index()
    risk_counts.columns = ["Risk Level", "Count"]
    color_map = {"LOW": "#10b981", "MEDIUM": "#f59e0b", "CRITICAL": "#ef4444"}
    fig_risk = px.bar(
        risk_counts, x="Risk Level", y="Count",
        title="Risk Level Breakdown",
        color="Risk Level", color_discrete_map=color_map,
    )
    fig_risk.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0", title_font_color="#38bdf8",
        xaxis=dict(gridcolor="#1e293b"), yaxis=dict(gridcolor="#1e293b"),
        showlegend=False,
    )
    st.markdown("<div class='glass' style='padding:1rem;'>", unsafe_allow_html=True)
    st.plotly_chart(fig_risk, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 11. PAGE: REGISTRY
# ─────────────────────────────────────────────
def page_registry():
    st.markdown("<div class='neon-title' style='font-size:2rem; margin-bottom:1.2rem;'>🗂️ Patient Registry</div>",
                unsafe_allow_html=True)

    reg = st.session_state.registry
    if not reg:
        st.info("Registry is empty. Complete a scan to populate records.")
        return

    df = pd.DataFrame(reg).drop(columns=["Confidence_Raw"], errors="ignore")

    # Search filter
    search = st.text_input("🔍 Search by Patient ID or Name", placeholder="Type to filter...")
    if search:
        mask = (
            df["Patient_ID"].str.contains(search, case=False, na=False) |
            df["Patient_Name"].str.contains(search, case=False, na=False)
        )
        df_view = df[mask]
    else:
        df_view = df

    st.markdown("<div class='glass' style='padding:0.5rem;'>", unsafe_allow_html=True)
    st.dataframe(
        df_view.style.applymap(
            lambda v: "color: #ef4444; font-weight:700" if v == "Malignant" else
                      "color: #10b981; font-weight:700" if v == "Benign" else "",
            subset=["AI_Diagnosis"],
        ),
        use_container_width=True,
        hide_index=True,
        height=min(600, 40 + 37 * len(df_view)),
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    csv = df.to_csv(index=False).encode("utf-8")
    c1, c2, _ = st.columns([0.3, 0.3, 0.4])
    with c1:
        st.download_button(
            "📥 Export Registry CSV",
            data=csv,
            file_name=f"SkinCancerAI_Registry_{datetime.date.today()}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with c2:
        if st.button("🗑️ Clear Registry", use_container_width=True):
            st.session_state.registry = []
            st.rerun()


# ─────────────────────────────────────────────
# 12. FOOTER
# ─────────────────────────────────────────────
def render_footer():
    st.markdown("""
    <div style='text-align:center; margin-top:4rem; padding:2rem 0 1rem;
         border-top:1px solid rgba(56,189,248,0.1);
         font-family:"DM Mono",monospace; font-size:0.75rem; color:#334155; line-height:2;'>
        <b style='color:#475569;'>SkinCancerPrediction AI — Medical Diagnostic System v12.0</b><br>
        University of Agricultural Faisalabad &nbsp;|&nbsp; Rehan Shafique
        &nbsp;|&nbsp; rehanshafiq6540@gmail.com<br>
        <span style='color:#1e293b;'>
            ⚠ For clinical screening purposes only. Not a substitute for professional medical diagnosis.
        </span>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 13. MAIN ENTRYPOINT
# ─────────────────────────────────────────────
def main():
    inject_css()
    init_state()
    render_sidebar()

    page = st.session_state.page
    if page == "Home":
        page_home()
    elif page == "AI Scanner":
        page_scanner()
    elif page == "Analytics":
        page_analytics()
    elif page == "Registry":
        page_registry()

    render_footer()


if __name__ == "__main__":
    main()
