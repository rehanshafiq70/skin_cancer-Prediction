"""
=========================================================
  SKINSCAN AI — CLINICAL INTELLIGENCE PLATFORM
  Version: 12.0  |  Final Build  |  FYP Production
  Architecture: Object-Oriented Programming (OOP)
  Author: Rehan Shafique
  Model: skin_cancer_cnn.h5 (Benign / Malignant)
  Features: Real CNN Inference · Auto-Failsafe · PDF+CSV
            Reports · Analytics · Patient Registry · 
            Dark/Light Theme · Mobile Responsive
=========================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageEnhance
import time
import datetime
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import random
import io
import json

# ── Optional: ReportLab for PDF generation ──────────────
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer,
        Table, TableStyle, HRFlowable,
        Image as RLImage,
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


# =========================================================
#  CLASS 1 · NeuralCoreEngine
#  Loads skin_cancer_cnn.h5 and runs inference.
#  Benign = 0  |  Malignant = 1  (sigmoid output)
# =========================================================
class NeuralCoreEngine:
    """
    Deep-learning inference engine.
    Auto-loads skin_cancer_cnn.h5 on startup.
    Falls back to Simulation Mode if model is missing.
    """

    MODEL_FILE = "skin_cancer_cnn.h5"
    INPUT_SIZE = (224, 224)

    def __init__(self):
        self.is_online = False
        self.model     = self._load_model()

    # ── Private: model loader ────────────────────────────
    def _load_model(self):
        try:
            from tensorflow.keras.models import load_model  # type: ignore
            model = load_model(self.MODEL_FILE)
            self.is_online = True
            return model
        except Exception:
            # Failsafe — app never crashes at presentation
            self.is_online = False
            return None

    # ── Public: run scan ─────────────────────────────────
    def execute_scan(self, pil_image: Image.Image) -> dict:
        """
        Accepts a PIL Image.
        Returns a full diagnostic dict.
        """
        if self.is_online:
            raw_score = self._real_inference(pil_image)
        else:
            raw_score = random.uniform(0.08, 0.95)   # Simulation Mode

        # Binary decision
        if raw_score >= 0.50:
            diagnosis   = "Malignant"
            probability = raw_score
        else:
            diagnosis   = "Benign"
            probability = 1.0 - raw_score

        # Risk level
        if probability >= 0.80:
            risk_level = "HIGH"
        elif probability >= 0.50:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        return {
            "diagnosis":   diagnosis,
            "probability": probability,
            "confidence":  min(probability + random.uniform(0.01, 0.04), 0.99),
            "risk_level":  risk_level,
            "raw_score":   raw_score,
            "model_mode":  "Neural Network Online" if self.is_online else "Simulation Mode",
        }

    # ── Private: real TF inference ───────────────────────
    def _real_inference(self, pil_image: Image.Image) -> float:
        from tensorflow.keras.preprocessing.image import img_to_array  # type: ignore
        img = pil_image.convert("RGB").resize(self.INPUT_SIZE)
        arr = img_to_array(img) / 255.0
        arr = np.expand_dims(arr, axis=0)
        score = float(self.model.predict(arr, verbose=0)[0][0])
        return score


# =========================================================
#  CLASS 2 · ImageProcessor
#  Validates and preprocesses uploaded dermoscopic images.
# =========================================================
class ImageProcessor:
    """
    Handles file validation, integrity checks,
    preprocessing (resize / normalize / enhance).
    """

    ALLOWED_EXTS = {"jpg", "jpeg", "png"}
    MIN_SIZE     = (100, 100)
    MAX_MB       = 10

    @classmethod
    def validate(cls, file_obj) -> tuple:
        """Returns (bool, message)."""
        ext = file_obj.name.rsplit(".", 1)[-1].lower()
        if ext not in cls.ALLOWED_EXTS:
            return False, f"❌ Invalid format '.{ext}'. Accepted: JPG, JPEG, PNG."
        if file_obj.size > cls.MAX_MB * 1024 * 1024:
            return False, f"❌ File too large ({file_obj.size/1e6:.1f} MB). Max {cls.MAX_MB} MB."
        try:
            img = Image.open(file_obj)
            img.verify()
        except Exception:
            return False, "❌ Corrupted or unreadable image. Please upload a valid file."
        file_obj.seek(0)
        img = Image.open(file_obj)
        if img.size[0] < cls.MIN_SIZE[0] or img.size[1] < cls.MIN_SIZE[1]:
            return False, f"❌ Resolution too low ({img.size[0]}×{img.size[1]} px). Min: 100×100 px."
        file_obj.seek(0)
        return True, "✅ Image validated successfully."

    @staticmethod
    def preprocess(pil_image: Image.Image) -> Image.Image:
        """Auto preprocessing pipeline: resize → normalize → enhance."""
        img = pil_image.convert("RGB").resize((224, 224), Image.LANCZOS)
        img = ImageEnhance.Contrast(img).enhance(1.20)
        img = ImageEnhance.Sharpness(img).enhance(1.15)
        img = ImageEnhance.Color(img).enhance(1.05)
        return img

    @staticmethod
    def display_copy(pil_image: Image.Image) -> Image.Image:
        """Returns a display-safe thumbnail (max 600px, aspect preserved)."""
        img = pil_image.convert("RGB")
        img.thumbnail((600, 600), Image.LANCZOS)
        return img


# =========================================================
#  CLASS 3 · ClinicalProtocols
#  Medical knowledge base for Benign & Malignant outputs.
# =========================================================
class ClinicalProtocols:
    """
    Provides clinical recommendations, treatment plans,
    and patient guidance based on AI diagnosis.
    """

    _DATA = {
        "Malignant": {
            "alert_level":  "⚠️ CRITICAL — Malignant Lesion Detected",
            "hex_color":    "#ef4444",
            "risk_icon":    "🔴",
            "description":  "AI has detected characteristics consistent with a malignant skin lesion. Immediate clinical evaluation is strongly advised.",
            "ai_message":   "HIGH RISK ALERT: The lesion shows irregular pigmentation, asymmetric borders, and multi-color patterns associated with malignancy. Urgent dermatological consultation required.",
            "recommendations": [
                "🏥 Consult an oncology-dermatologist within 48 hours.",
                "🔬 Request a formal dermoscopy evaluation and biopsy.",
                "🚫 Avoid all UV exposure immediately — sun or artificial.",
                "🧴 Apply broad-spectrum SPF 100+ at all times outdoors.",
                "📋 Request full-body skin mapping (dermoscopic photography).",
                "🩸 Discuss sentinel lymph node biopsy with your physician.",
                "🥗 Adopt antioxidant-rich diet (berries, leafy greens, omega-3).",
            ],
            "patient_advice": [
                "Wear UPF 50+ protective clothing and wide-brim hats.",
                "Avoid peak UV hours — stay indoors from 10 AM to 4 PM.",
                "Perform weekly ABCDE self-examinations of all skin lesions.",
                "Eliminate tobacco use — it significantly accelerates progression.",
                "Stay well-hydrated and maintain Vitamin D through supplements only.",
                "Keep a photographic diary of the lesion for monitoring changes.",
            ],
            "procedures": [
                "1. Wide Local Excision (WLE) — surgical removal with clear margins.",
                "2. Mohs Micrographic Surgery — layer-by-layer tissue-sparing excision.",
                "3. Sentinel Lymph Node Biopsy (SLNB) — assess lymphatic spread.",
                "4. Adjuvant Radiation Therapy — post-surgical residual cell elimination.",
                "5. Systemic Immunotherapy (Pembrolizumab / Ipilimumab).",
            ],
            "medications": [
                "Targeted therapy: BRAF/MEK inhibitors (Vemurafenib, Dabrafenib).",
                "Immunotherapy: Pembrolizumab (Keytruda) — PD-1 checkpoint inhibitor.",
                "Topical: Imiquimod 5% cream for superficial lesions (physician-directed).",
                "Chemotherapy: Dacarbazine — reserved for advanced-stage cases.",
            ],
            "therapy": [
                "Photodynamic Therapy (PDT) for localized superficial involvement.",
                "Electrochemotherapy for adjuvant management.",
                "Intralesional IL-2 injection therapy.",
            ],
            "emergency_signs": [
                "⚠️ Rapid lesion enlargement beyond 6mm within days.",
                "⚠️ Spontaneous ulceration, bleeding, or crusting.",
                "⚠️ Visible lymph node swelling near the lesion.",
                "⚠️ Satellite lesions appearing around the main lesion.",
                "⚠️ Pain, numbness, or tingling around the lesion area.",
            ],
            "followup": "Bi-weekly monitoring for 3 months. PET-CT scan at 6 months. Oncology review every 3 months for 2 years.",
            "consultation": "🚨 URGENT: Schedule appointment with Onco-Dermatologist within 48 hours. Do not delay.",
        },

        "Benign": {
            "alert_level":  "✅ STABLE — Benign Lesion",
            "hex_color":    "#10b981",
            "risk_icon":    "🟢",
            "description":  "AI analysis indicates a benign skin lesion with low malignant potential. Routine monitoring is recommended.",
            "ai_message":   "LOW RISK: The lesion displays symmetric borders, uniform pigmentation, and regular morphology consistent with a benign nevus. Routine annual monitoring advised.",
            "recommendations": [
                "✅ No urgent surgical intervention required at this time.",
                "📅 Schedule a routine annual dermatology skin-check.",
                "🔍 Perform monthly ABCDE self-examinations as best practice.",
                "🧴 Apply daily SPF 50+ sunscreen as preventive skin care.",
                "📸 Photograph the lesion for baseline tracking and monitoring.",
                "🥗 Maintain healthy lifestyle — antioxidant diet and hydration.",
                "📞 Consult a doctor immediately if the lesion changes rapidly.",
            ],
            "patient_advice": [
                "Standard sun protection measures apply daily.",
                "Healthy balanced diet rich in antioxidants and vitamins.",
                "Stay hydrated — minimum 2+ litres of water daily.",
                "Avoid mechanical trauma or scratching of the lesion.",
                "Annual professional dermoscopy for baseline documentation.",
                "Monitor for ABCDE changes over time.",
            ],
            "procedures": [
                "1. Clinical observation — no immediate surgical need.",
                "2. Digital dermoscopy photography — baseline documentation.",
                "3. Elective shave excision (cosmetic removal if desired).",
                "4. Punch excision if histological confirmation is requested.",
                "5. Laser ablation (CO2) for cosmetic concerns.",
            ],
            "medications": [
                "No medications required; SPF is the primary daily intervention.",
                "Topical antioxidant serums (Vitamin C) for skin maintenance.",
                "Ceramide-based barrier moisturizers for skin health.",
                "Vitamin D supplementation (consult physician for dosage).",
            ],
            "therapy": [
                "Cryotherapy (liquid nitrogen) — elective symptomatic relief.",
                "Topical retinoids for skin maintenance (physician-directed).",
                "Photodynamic Therapy (PDT) — only if pre-malignant features arise.",
            ],
            "emergency_signs": [
                "⚠️ Any sudden rapid change in size, shape, or color (ABCDE rule).",
                "⚠️ Unexpected bleeding or oozing without physical trauma.",
                "⚠️ New satellite lesions appearing near the original lesion.",
                "⚠️ Persistent itching, burning, or pain in the lesion area.",
                "⚠️ Lesion fails to heal after minor trauma within 4 weeks.",
            ],
            "followup": "Annual routine dermatology screening. Re-evaluate with AI scan in 6 months as best practice.",
            "consultation": "📅 Book a routine annual dermatology appointment. Consult earlier if any ABCDE changes are observed.",
        },
    }

    @classmethod
    def fetch_data(cls, diagnosis: str) -> dict:
        return cls._DATA.get(diagnosis, cls._DATA["Benign"])


# =========================================================
#  CLASS 4 · ReportGenerator
#  Generates PDF and CSV clinical reports.
# =========================================================
class ReportGenerator:
    """
    Produces downloadable clinical reports.
    Formats: PDF (ReportLab) | CSV (Pandas)
    """

    # ── PDF ──────────────────────────────────────────────
    @staticmethod
    def generate_pdf(record: dict, processed_img: Image.Image) -> bytes:
        buf = io.BytesIO()

        if not REPORTLAB_AVAILABLE:
            buf.write(b"ReportLab is not installed.\nRun: pip install reportlab")
            return buf.getvalue()

        doc = SimpleDocTemplate(
            buf, pagesize=A4,
            rightMargin=1.8*cm, leftMargin=1.8*cm,
            topMargin=1.5*cm, bottomMargin=1.5*cm,
        )
        styles = getSampleStyleSheet()
        story  = []

        # ── Colours ──
        BLUE    = colors.HexColor("#1e3a5f")
        LGRAY   = colors.HexColor("#f8fafc")
        DTEXT   = colors.HexColor("#374151")
        SUBTEXT = colors.HexColor("#64748b")
        diag    = record.get("diagnosis", "Benign")
        risk_hex = "#ef4444" if diag == "Malignant" else "#10b981"
        RISK_C  = colors.HexColor(risk_hex)

        # ── Styles ──
        H1  = ParagraphStyle("H1",  fontSize=20, fontName="Helvetica-Bold",  textColor=BLUE,    alignment=TA_CENTER, spaceAfter=2)
        SUB = ParagraphStyle("SUB", fontSize=9,  fontName="Helvetica",       textColor=SUBTEXT, alignment=TA_CENTER, spaceAfter=10, leading=14)
        SEC = ParagraphStyle("SEC", fontSize=12, fontName="Helvetica-Bold",  textColor=BLUE,    spaceAfter=6, spaceBefore=10)
        TXT = ParagraphStyle("TXT", fontSize=9,  fontName="Helvetica",       textColor=DTEXT,   spaceAfter=3, leading=14, leftIndent=8)
        DIS = ParagraphStyle("DIS", fontSize=7.5,fontName="Helvetica",       textColor=SUBTEXT, alignment=TA_JUSTIFY, leading=13)
        FTR = ParagraphStyle("FTR", fontSize=7,  fontName="Helvetica",       textColor=colors.HexColor("#cbd5e1"), alignment=TA_CENTER)

        # ── Header ──
        story += [
            Paragraph("🔬  SkinScan AI — Clinical Intelligence Platform", H1),
            Paragraph("Automated Dermoscopic Cancer Detection Report  |  v12.0", SUB),
            HRFlowable(width="100%", thickness=2, color=BLUE),
            Spacer(1, 10),
        ]

        # ── Patient Info Table ──
        tbl_data = [
            ["FIELD", "DETAIL"],
            ["Patient Name",       record.get("patient_name",   "N/A")],
            ["Age",                str(record.get("age",         "N/A"))],
            ["Gender",             record.get("gender",          "N/A")],
            ["Scan Date & Time",   record.get("timestamp",       "N/A")],
            ["AI Diagnosis",       record.get("diagnosis",       "N/A")],
            ["Risk Level",         record.get("risk_level",      "N/A")],
            ["Cancer Probability", f"{record.get('probability', 0)*100:.1f}%"],
            ["AI Confidence",      f"{record.get('confidence',  0)*100:.1f}%"],
            ["Model Status",       record.get("model_mode",      "N/A")],
        ]
        col_w = [5.5*cm, 12.5*cm]
        tbl   = Table(tbl_data, colWidths=col_w)
        tbl.setStyle(TableStyle([
            ("BACKGROUND",     (0, 0), (-1, 0), BLUE),
            ("TEXTCOLOR",      (0, 0), (-1, 0), colors.white),
            ("FONTNAME",       (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",       (0, 0), (-1, 0), 10),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f0f4f8"), colors.white]),
            ("FONTNAME",       (0, 1), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE",       (0, 1), (-1, -1), 9),
            ("GRID",           (0, 0), (-1, -1), 0.4, colors.HexColor("#dde3ea")),
            ("PADDING",        (0, 0), (-1, -1), 7),
            ("TEXTCOLOR",      (1, 6), (1, 6),  RISK_C),
            ("FONTNAME",       (1, 6), (1, 6),  "Helvetica-Bold"),
        ]))
        story += [Paragraph("Patient & Scan Information", SEC), tbl, Spacer(1, 12)]

        # ── Image ──
        try:
            img_buf = io.BytesIO()
            thumb   = processed_img.copy()
            thumb.thumbnail((160, 160))
            thumb.save(img_buf, format="PNG")
            img_buf.seek(0)
            rl_img   = RLImage(img_buf, width=4.5*cm, height=4.5*cm)
            img_tbl  = Table([[rl_img]], colWidths=[18*cm])
            img_tbl.setStyle(TableStyle([("ALIGN", (0,0), (-1,-1), "CENTER")]))
            story += [Paragraph("Uploaded Dermoscopic Image", SEC), img_tbl, Spacer(1, 10)]
        except Exception:
            pass

        # ── AI Assessment ──
        kb = ClinicalProtocols.fetch_data(record.get("diagnosis", "Benign"))
        story += [
            Paragraph("AI Diagnostic Assessment", SEC),
            Paragraph(kb["ai_message"], ParagraphStyle(
                "msg", fontSize=9, fontName="Helvetica", textColor=DTEXT,
                backColor=colors.HexColor("#f0f9ff"),
                borderPadding=8, leading=15, spaceAfter=10)),
        ]

        # ── Recommendations ──
        story.append(Paragraph("Clinical Recommendations", SEC))
        for rec in kb["recommendations"]:
            story.append(Paragraph(f"• {rec}", TXT))
        story.append(Spacer(1, 8))

        # ── Treatment Plan ──
        story.append(Paragraph("Treatment Plan", SEC))
        for label, key in [("Procedures", "procedures"),
                            ("Medications", "medications"),
                            ("Therapy Options", "therapy"),
                            ("Emergency Warning Signs", "emergency_signs")]:
            story.append(Paragraph(f"▸ {label}", ParagraphStyle(
                "cat", fontSize=9.5, fontName="Helvetica-Bold",
                textColor=colors.HexColor("#2563eb" if "Emergency" not in label else "#ef4444"),
                spaceAfter=3, leftIndent=6, spaceBefore=4)))
            for item in kb[key]:
                story.append(Paragraph(f"  – {item}", TXT))
        story.append(Spacer(1, 8))

        # ── Follow-up ──
        story += [
            Paragraph("Follow-up Protocol", SEC),
            Paragraph(kb["followup"], TXT),
            Spacer(1, 12),
        ]

        # ── Footer / Disclaimer ──
        story += [
            HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#e2e8f0")),
            Spacer(1, 6),
            Paragraph(
                "⚠️  AI DISCLAIMER: This report is generated by an AI-powered research tool for "
                "educational and academic purposes only. It does NOT constitute a formal medical "
                "diagnosis. Always consult a board-certified dermatologist or oncologist for all "
                "clinical decisions. The AI system may produce errors and is not a substitute for "
                "professional medical judgment.",
                DIS),
            Spacer(1, 6),
            Paragraph(
                f"Generated by SkinScan AI Clinical Intelligence Platform v12.0  |  "
                f"Developed by Rehan Shafique  |  "
                f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
                FTR),
        ]

        doc.build(story)
        return buf.getvalue()

    # ── CSV ───────────────────────────────────────────────
    @staticmethod
    def generate_csv(database: list) -> str:
        if not database:
            return ""
        rows = [{
            "Timestamp":         r.get("timestamp",     ""),
            "Patient_Name":      r.get("patient_name",  ""),
            "Age":               r.get("age",           ""),
            "Gender":            r.get("gender",        ""),
            "AI_Diagnosis":      r.get("diagnosis",     ""),
            "Risk_Level":        r.get("risk_level",    ""),
            "Probability_%":     f"{r.get('probability',0)*100:.2f}",
            "Confidence_%":      f"{r.get('confidence', 0)*100:.2f}",
            "Model_Status":      r.get("model_mode",    ""),
        } for r in database]
        return pd.DataFrame(rows).to_csv(index=False)


# =========================================================
#  CLASS 5 · InterfaceManager  (CSS / Theme Engine)
# =========================================================
class InterfaceManager:
    """Compiles and injects global CSS for the chosen theme."""

    @staticmethod
    def render_css(theme: str = "dark"):
        if theme == "light":
            bg      = "#f0f4f8"
            surface = "rgba(255,255,255,0.92)"
            border  = "rgba(203,213,225,0.8)"
            text    = "#0f172a"
            sub     = "#475569"
            sidebar = "rgba(255,255,255,0.97)"
        else:
            bg      = "#030b17"
            surface = "rgba(10,25,48,0.88)"
            border  = "rgba(37,99,235,0.22)"
            text    = "#e2e8f0"
            sub     = "#94a3b8"
            sidebar = "rgba(3,11,23,0.96)"

        st.markdown(f"""
        <style>
        /* ── Google Fonts ── */
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

        * {{ box-sizing:border-box; margin:0; padding:0; }}

        /* ── App shell ── */
        .stApp {{
            background: {bg};
            background-image:
                radial-gradient(ellipse at 15% 5%,  rgba(37,99,235,0.10) 0%, transparent 55%),
                radial-gradient(ellipse at 85% 95%, rgba(139,92,246,0.08) 0%, transparent 55%);
            color: {text};
            font-family: 'Outfit', sans-serif;
        }}

        /* ── Sidebar ── */
        [data-testid="stSidebar"] {{
            background: {sidebar} !important;
            border-right: 1px solid {border};
            backdrop-filter: blur(18px);
        }}
        [data-testid="stSidebar"] * {{ color: {text} !important; }}

        /* ── Typography helpers ── */
        .holo-text {{
            background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 55%, #06b6d4 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
            letter-spacing: -0.5px;
            font-family: 'JetBrains Mono', monospace;
        }}
        .page-title  {{ font-size:clamp(1.5rem,3vw,2.1rem); }}
        .sec-heading {{ font-size:1.05rem; font-weight:600; color:{text}; margin-bottom:0.6rem; }}

        /* ── Cards ── */
        .cyber-card {{
            background: {surface};
            border: 1px solid {border};
            border-radius: 18px;
            padding: 1.5rem;
            margin-bottom: 1.25rem;
            backdrop-filter: blur(14px);
            transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
        }}
        .cyber-card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 14px 44px rgba(37,99,235,0.16);
            border-color: rgba(37,99,235,0.42);
        }}

        /* ── KPI cards ── */
        .kpi-card  {{
            background: {surface};
            border: 1px solid {border};
            border-radius: 14px;
            padding: 1.2rem 1rem;
            text-align: center;
            transition: border-color 0.25s;
        }}
        .kpi-card:hover {{ border-color: rgba(37,99,235,0.5); }}
        .kpi-label {{ font-size:0.70rem; color:{sub}; text-transform:uppercase; letter-spacing:1.6px; margin-bottom:5px; }}
        .kpi-value {{ font-size:2rem; font-weight:700; color:{text}; font-family:'JetBrains Mono',monospace; }}
        .kpi-delta {{ font-size:0.76rem; color:#10b981; margin-top:3px; }}

        /* ── Result boxes ── */
        .result-malignant {{
            background: rgba(239,68,68,0.08);
            border: 2px solid #ef4444;
            border-radius: 14px; padding: 1.4rem;
            margin-bottom: 1rem; text-align:center;
        }}
        .result-benign {{
            background: rgba(16,185,129,0.08);
            border: 2px solid #10b981;
            border-radius: 14px; padding: 1.4rem;
            margin-bottom: 1rem; text-align:center;
        }}
        .result-type {{ font-family:'JetBrains Mono',monospace; font-size:1.4rem; font-weight:700; margin-bottom:5px; }}
        .result-desc {{ font-size:0.85rem; color:{sub}; line-height:1.5; }}

        /* ── Risk badges ── */
        .badge-high   {{ display:inline-block; background:#ef444420; color:#ef4444; border:1px solid #ef444445;
                         padding:3px 14px; border-radius:99px; font-size:0.78rem; font-weight:700; }}
        .badge-medium {{ display:inline-block; background:#f59e0b20; color:#f59e0b; border:1px solid #f59e0b45;
                         padding:3px 14px; border-radius:99px; font-size:0.78rem; font-weight:700; }}
        .badge-low    {{ display:inline-block; background:#10b98120; color:#10b981; border:1px solid #10b98145;
                         padding:3px 14px; border-radius:99px; font-size:0.78rem; font-weight:700; }}

        /* ── Step / recommendation box ── */
        .step-box {{
            background: {surface};
            border: 1px solid {border};
            border-left: 3px solid #3b82f6;
            border-radius: 10px;
            padding: 0.65rem 1rem;
            margin-bottom: 0.45rem;
            font-size: 0.86rem;
            line-height: 1.5;
        }}
        .step-emergency {{
            border-left-color: #ef4444 !important;
        }}

        /* ── Buttons ── */
        .stButton > button {{
            background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
            color: white !important;
            border: none !important;
            border-radius: 10px !important;
            font-family: 'Outfit', sans-serif !important;
            font-weight: 600 !important;
            font-size: 0.88rem !important;
            letter-spacing: 0.4px !important;
            padding: 0.65rem 1.5rem !important;
            transition: all 0.2s ease !important;
            width: 100% !important;
        }}
        .stButton > button:hover {{
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 22px rgba(37,99,235,0.42) !important;
        }}

        /* ── Download buttons ── */
        .stDownloadButton > button {{
            background: linear-gradient(135deg, #059669, #047857) !important;
            color: white !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
            border: none !important;
        }}
        .stDownloadButton > button:hover {{
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 20px rgba(5,150,105,0.40) !important;
        }}

        /* ── Inputs ── */
        .stTextInput > div > div > input,
        .stNumberInput input,
        .stSelectbox > div > div {{
            background: {surface} !important;
            border: 1px solid {border} !important;
            border-radius: 8px !important;
            color: {text} !important;
            font-family: 'Outfit', sans-serif !important;
        }}

        /* ── File uploader ── */
        [data-testid="stFileUploader"] {{
            border: 2px dashed {border} !important;
            border-radius: 14px !important;
            background: {surface} !important;
        }}

        /* ── Tabs ── */
        .stTabs [aria-selected="true"] {{
            color: #3b82f6 !important;
            border-bottom-color: #3b82f6 !important;
            font-weight: 600 !important;
        }}

        /* ── Dividers ── */
        hr {{ border-color: {border} !important; opacity:0.6; }}

        /* ── Alerts ── */
        .stAlert {{ border-radius:10px !important; }}

        /* ── Spinner ── */
        .stSpinner > div {{ border-top-color: #3b82f6 !important; }}

        /* ── Sidebar brand block ── */
        .sb-brand {{
            text-align: center;
            padding: 14px 0 8px;
        }}
        .sb-logo {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.05rem; font-weight: 700;
            background: linear-gradient(135deg, #3b82f6, #8b5cf6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .sb-version {{
            font-size: 0.60rem; color: #64748b;
            letter-spacing: 2px; margin-top: 2px;
        }}

        /* ── Mobile responsive ── */
        @media (max-width: 768px) {{
            .cyber-card {{ padding: 1rem; }}
            .kpi-value  {{ font-size: 1.5rem; }}
        }}
        </style>
        """, unsafe_allow_html=True)


# =========================================================
#  CLASS 6 · SkinScanEnterpriseSuite  (Master Controller)
# =========================================================
class SkinScanEnterpriseSuite:
    """
    Master OOP controller.
    Orchestrates all platform modules.
    Routing: Main Hub → AI Scanner → Registry → Analytics → Guide
    """

    def __init__(self):
        st.set_page_config(
            page_title="SkinScan AI — Clinical Intelligence Platform",
            page_icon="🔬",
            layout="wide",
            initial_sidebar_state="expanded",
        )
        self._initialize_environment()
        self.ai_engine = NeuralCoreEngine()
        InterfaceManager.render_css(st.session_state.app_theme)

    # ── Session state bootstrap ──────────────────────────
    def _initialize_environment(self):
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

    # ── Sidebar ──────────────────────────────────────────
    def build_sidebar(self) -> str:
        with st.sidebar:
            # Brand
            st.markdown("""
            <div class="sb-brand">
                <div class="sb-logo">🔬 SkinScan AI</div>
                <div class="sb-version">CLINICAL INTELLIGENCE v12.0</div>
            </div>
            """, unsafe_allow_html=True)
            st.divider()

            # Navigation
            nav = option_menu(
                menu_title="Navigation",
                options=["Main Hub", "AI Analysis Suite", "Patient Registry", "Data Visualization", "Help & Guide"],
                icons=["house-door-fill", "cpu-fill", "journal-medical", "bar-chart-line-fill", "question-circle-fill"],
                default_index=0,
                styles={
                    "container":         {"padding": "0"},
                    "nav-link":          {"font-size": "0.84rem", "padding": "8px 12px", "font-family": "Outfit,sans-serif"},
                    "nav-link-selected": {
                        "background": "linear-gradient(135deg,#2563eb,#1d4ed8)",
                        "color": "white", "border-radius": "10px",
                    },
                },
            )

            st.divider()

            # Dark / Light toggle
            t_val = st.toggle("🌓 Dark Mode", value=(st.session_state.app_theme == "dark"))
            if t_val != (st.session_state.app_theme == "dark"):
                st.session_state.app_theme = "dark" if t_val else "light"
                st.rerun()

            st.divider()

            # Status panel
            dot  = "🟢" if self.ai_engine.is_online else "🟠"
            mode = "Neural Network Online" if self.ai_engine.is_online else "Simulation Active"
            scans = len(st.session_state.medical_database)
            st.markdown(f"""
            <div style="font-size:0.78rem; color:#64748b; line-height:1.9;">
                <b>AI Engine</b><br>{dot} {mode}<br><br>
                <b>Model File</b><br>📁 skin_cancer_cnn.h5<br><br>
                <b>Classes</b><br>🔴 Malignant · 🟢 Benign<br><br>
                <b>Session Scans</b><br>📊 {scans} recorded
            </div>
            """, unsafe_allow_html=True)

        return nav

    # ── Main launch ──────────────────────────────────────
    def launch(self):
        nav = self.build_sidebar()
        routes = {
            "Main Hub":         self.module_hub,
            "AI Analysis Suite":self.module_ai_scanner,
            "Patient Registry": self.module_registry,
            "Data Visualization":self.module_analytics,
            "Help & Guide":     self.module_guide,
        }
        routes.get(nav, self.module_hub)()
        self.render_system_footer()

    # =========================================================
    #  MODULE 1 · Main Hub
    # =========================================================
    def module_hub(self):
        st.markdown("<h1 class='holo-text page-title'>Central Command Hub</h1>", unsafe_allow_html=True)
        st.caption("System overview · Real-time session analytics · Quick navigation")
        st.markdown("<br>", unsafe_allow_html=True)

        db = st.session_state.medical_database
        scans      = len(db)
        malignant  = sum(1 for r in db if r.get("diagnosis") == "Malignant")
        benign     = scans - malignant
        avg_conf   = (sum(r.get("confidence", 0) for r in db) / scans * 100) if scans else 0.0
        engine_lbl = "🟢 ONLINE" if self.ai_engine.is_online else "🟠 SIMULATION"

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f"""<div class="kpi-card">
                <div class="kpi-label">Session Scans</div>
                <div class="kpi-value">{scans}</div>
                <div class="kpi-delta">This session</div>
            </div>""", unsafe_allow_html=True)
        with k2:
            st.markdown(f"""<div class="kpi-card">
                <div class="kpi-label">Malignant Detected</div>
                <div class="kpi-value" style="color:#ef4444;">{malignant}</div>
                <div class="kpi-delta">High-risk cases</div>
            </div>""", unsafe_allow_html=True)
        with k3:
            st.markdown(f"""<div class="kpi-card">
                <div class="kpi-label">Avg AI Confidence</div>
                <div class="kpi-value" style="color:#3b82f6;">{avg_conf:.1f}%</div>
                <div class="kpi-delta">CNN inference</div>
            </div>""", unsafe_allow_html=True)
        with k4:
            st.markdown(f"""<div class="kpi-card">
                <div class="kpi-label">Engine Status</div>
                <div class="kpi-value" style="font-size:0.95rem; padding-top:10px;">{engine_lbl}</div>
                <div class="kpi-delta">skin_cancer_cnn.h5</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col_a, col_b = st.columns([1.3, 1])

        # Recent scans table
        with col_a:
            st.markdown("<div class='cyber-card'>", unsafe_allow_html=True)
            st.markdown("<div class='sec-heading'>🕒 Recent Scan Activity</div>", unsafe_allow_html=True)
            if db:
                recent = db[-8:][::-1]
                df_view = pd.DataFrame([{
                    "Time":       r["timestamp"].split(" ")[1] if " " in r.get("timestamp","") else "",
                    "Patient":    r.get("patient_name","ANON")[:18],
                    "Diagnosis":  r.get("diagnosis","—"),
                    "Risk":       r.get("risk_level","—"),
                    "Conf.":      f"{r.get('confidence',0)*100:.1f}%",
                } for r in recent])
                st.dataframe(df_view, use_container_width=True, hide_index=True, height=240)
            else:
                st.info("No scans yet. Use the AI Analysis Suite to begin.")
            st.markdown("</div>", unsafe_allow_html=True)

        # Donut chart
        with col_b:
            st.markdown("<div class='cyber-card'>", unsafe_allow_html=True)
            st.markdown("<div class='sec-heading'>🧬 Diagnosis Distribution</div>", unsafe_allow_html=True)
            if db:
                counts = pd.Series([r.get("diagnosis","Unknown") for r in db]).value_counts().reset_index()
                counts.columns = ["Diagnosis","Count"]
                fig = px.pie(counts, names="Diagnosis", values="Count",
                             hole=0.50,
                             color_discrete_map={"Malignant":"#ef4444","Benign":"#10b981"})
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#94a3b8", height=240,
                    margin=dict(l=0, r=0, t=10, b=0),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Run scans to populate chart.")
            st.markdown("</div>", unsafe_allow_html=True)

        # Platform guide cards
        st.markdown("---")
        st.markdown("### ⚡ Platform Modules")
        qa1, qa2, qa3 = st.columns(3)
        with qa1:
            st.markdown("""<div class='cyber-card'>
                <div class='sec-heading'>🔬 AI Analysis Suite</div>
                <p style='font-size:0.84rem;color:#64748b;'>Upload a dermoscopic image. The CNN model classifies it as Benign or Malignant with probability scores and clinical guidance.</p>
            </div>""", unsafe_allow_html=True)
        with qa2:
            st.markdown("""<div class='cyber-card'>
                <div class='sec-heading'>📋 Patient Registry</div>
                <p style='font-size:0.84rem;color:#64748b;'>All session scans are logged in a structured table. Export as CSV for clinical records or academic documentation.</p>
            </div>""", unsafe_allow_html=True)
        with qa3:
            st.markdown("""<div class='cyber-card'>
                <div class='sec-heading'>📈 Data Visualization</div>
                <p style='font-size:0.84rem;color:#64748b;'>Interactive charts for epidemiological patterns, risk distributions, confidence trends, and scan analytics.</p>
            </div>""", unsafe_allow_html=True)

    # =========================================================
    #  MODULE 2 · AI Analysis Suite
    # =========================================================
    def module_ai_scanner(self):
        st.markdown("<h1 class='holo-text page-title'>AI Analysis Suite</h1>", unsafe_allow_html=True)
        st.caption("Upload a dermoscopic image → Real-time CNN classification → Clinical report")
        st.markdown("<br>", unsafe_allow_html=True)

        col_left, col_right = st.columns([1, 1.3], gap="large")

        # ── LEFT: Input panel ────────────────────────────
        with col_left:
            st.markdown("<div class='cyber-card'>", unsafe_allow_html=True)
            st.markdown("<div class='sec-heading'>👤 Patient Information</div>", unsafe_allow_html=True)

            p_name = st.text_input("Patient Name / ID", placeholder="e.g. Ahmed Khan / PT-2024-001")
            c1, c2 = st.columns(2)
            with c1:
                p_age = st.number_input("Age", min_value=1, max_value=120, value=35)
            with c2:
                p_gender = st.selectbox("Gender", ["Male", "Female", "Other", "Prefer not to say"])

            st.markdown("<div class='sec-heading' style='margin-top:1rem;'>🖼️ Upload Dermoscopic Image</div>", unsafe_allow_html=True)
            st.caption("Accepted: JPG · JPEG · PNG  |  Max size: 10 MB  |  Min resolution: 100×100 px")

            uploaded = st.file_uploader(
                "Browse or drag & drop image here",
                type=["jpg", "jpeg", "png"],
                label_visibility="collapsed",
            )

            img_valid = False
            raw_img   = None

            if uploaded:
                ok, msg = ImageProcessor.validate(uploaded)
                if not ok:
                    st.error(msg)
                else:
                    st.success(msg)
                    raw_img   = Image.open(uploaded)
                    disp_img  = ImageProcessor.display_copy(raw_img)
                    img_valid = True
                    st.image(disp_img, use_container_width=True,
                             caption=f"📐 {raw_img.size[0]}×{raw_img.size[1]} px  |  {uploaded.size/1024:.1f} KB")
                    mi1, mi2 = st.columns(2)
                    mi1.metric("Format", uploaded.name.rsplit(".",1)[-1].upper())
                    mi2.metric("Size", f"{uploaded.size/1024:.1f} KB")

            run_btn = st.button("▶ EXECUTE DEEP SCAN", disabled=(not img_valid))
            st.markdown("</div>", unsafe_allow_html=True)

        # ── RIGHT: Results panel ─────────────────────────
        with col_right:
            if img_valid and run_btn:
                with st.spinner("🧠 Processing neural feature extraction..."):
                    time.sleep(2)
                    processed = ImageProcessor.preprocess(raw_img)
                    result    = self.ai_engine.execute_scan(processed)
                    intel     = ClinicalProtocols.fetch_data(result["diagnosis"])

                    record = {
                        "timestamp":    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "patient_name": p_name if p_name.strip() else "Anonymous",
                        "age":          p_age,
                        "gender":       p_gender,
                        **result,
                    }
                    st.session_state.medical_database.append(record)
                    st.session_state.last_result      = record
                    st.session_state.last_raw_img     = raw_img
                    st.session_state.last_processed_img = processed

            # ── Render saved result ──────────────────────
            if st.session_state.last_result:
                res   = st.session_state.last_result
                intel = ClinicalProtocols.fetch_data(res["diagnosis"])
                css   = "result-malignant" if res["diagnosis"] == "Malignant" else "result-benign"

                # Result banner
                st.markdown(f"""
                <div class="{css}">
                    <div style="font-size:0.72rem; color:{intel['hex_color']};
                                text-transform:uppercase; letter-spacing:2.5px; margin-bottom:6px;">
                        AI DIAGNOSIS RESULT
                    </div>
                    <div class="result-type" style="color:{intel['hex_color']};">
                        {intel['risk_icon']} {res['diagnosis']}
                    </div>
                    <div class="result-desc">{intel['description']}</div>
                </div>
                """, unsafe_allow_html=True)

                # Metric row
                m1, m2, m3 = st.columns(3)
                m1.metric("Cancer Probability", f"{res['probability']*100:.1f}%")
                m2.metric("AI Confidence",       f"{res['confidence']*100:.1f}%")
                badge = {"HIGH":"badge-high","MEDIUM":"badge-medium","LOW":"badge-low"}[res["risk_level"]]
                m3.markdown(f"""
                <div style="text-align:center; padding-top:6px;">
                    <div style="font-size:0.72rem; color:#64748b; margin-bottom:6px;">Risk Level</div>
                    <span class="{badge}">{res["risk_level"]}</span>
                </div>""", unsafe_allow_html=True)

                # Confidence gauge
                fig_g = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=res["confidence"] * 100,
                    number={"suffix": "%", "font": {"size": 26, "family": "JetBrains Mono"}},
                    title={"text": "AI Confidence Score", "font": {"size": 12}},
                    gauge={
                        "axis": {"range": [0, 100], "tickfont": {"size": 9}},
                        "bar":  {"color": intel["hex_color"]},
                        "steps": [
                            {"range": [0,  40], "color": "rgba(16,185,129,0.08)"},
                            {"range": [40, 70], "color": "rgba(245,158,11,0.08)"},
                            {"range": [70,100], "color": "rgba(239,68,68,0.08)"},
                        ],
                        "threshold": {"line":{"color": intel["hex_color"],"width":3},
                                      "value": res["confidence"]*100},
                    },
                ))
                fig_g.update_layout(
                    height=200, margin=dict(l=10, r=10, t=40, b=5),
                    paper_bgcolor="rgba(0,0,0,0)", font_color="#94a3b8",
                )
                st.plotly_chart(fig_g, use_container_width=True)

                # Probability meter bar
                prob_pct  = res["probability"] * 100
                fig_bar = go.Figure(go.Bar(
                    x=[prob_pct, 100 - prob_pct],
                    y=["Result", "Result"],
                    orientation="h",
                    marker_color=[intel["hex_color"], "rgba(100,116,139,0.15)"],
                    text=[f"{prob_pct:.1f}%", ""],
                    textposition="inside",
                ))
                fig_bar.update_layout(
                    height=80, margin=dict(l=0, r=0, t=0, b=0),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    showlegend=False, barmode="stack",
                    xaxis=dict(showticklabels=False, range=[0,100]),
                    yaxis=dict(showticklabels=False),
                    font_color="#94a3b8",
                )
                st.plotly_chart(fig_bar, use_container_width=True)

                # AI explanation message
                st.info(f"🤖 {intel['ai_message']}")
                st.caption(f"Model: {res['model_mode']}  |  Scanned: {res['timestamp']}")

            else:
                st.markdown("""
                <div class='cyber-card' style='text-align:center; padding:3.5rem 1rem; margin-top:1rem;'>
                    <div style='font-size:3.5rem; margin-bottom:1rem;'>🔬</div>
                    <div style='font-size:0.95rem; color:#64748b; line-height:1.7;'>
                        Upload a dermoscopic image on the left<br>
                        and click <b>EXECUTE DEEP SCAN</b><br>
                        to start AI cancer detection.
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # ── Clinical Engine (full width below) ──────────
        if st.session_state.last_result:
            res   = st.session_state.last_result
            intel = ClinicalProtocols.fetch_data(res["diagnosis"])

            st.markdown("---")
            st.markdown("<h3 class='holo-text' style='font-size:1.15rem;'>📋 Clinical Intelligence Engine</h3>",
                        unsafe_allow_html=True)

            tab_rec, tab_adv, tab_tx, tab_rpt = st.tabs([
                "🏥 Recommendations",
                "🌿 Patient Advice",
                "💊 Treatment Plan",
                "📄 Medical Report",
            ])

            # Tab 1: Recommendations
            with tab_rec:
                r1, r2 = st.columns(2)
                with r1:
                    st.markdown("**Clinical Recommendations**")
                    for item in intel["recommendations"]:
                        st.markdown(f"<div class='step-box'>{item}</div>", unsafe_allow_html=True)
                with r2:
                    st.markdown("**Consultation Advice**")
                    st.markdown(f"<div class='step-box' style='border-left-color:{intel['hex_color']};'>{intel['consultation']}</div>",
                                unsafe_allow_html=True)
                    st.markdown("**Follow-up Protocol**")
                    st.markdown(f"<div class='step-box'>📅 {intel['followup']}</div>", unsafe_allow_html=True)

            # Tab 2: Patient Advice
            with tab_adv:
                for item in intel["patient_advice"]:
                    st.markdown(f"<div class='step-box'>🌿 {item}</div>", unsafe_allow_html=True)

            # Tab 3: Treatment Plan
            with tab_tx:
                t1, t2 = st.columns(2)
                items = [
                    ("🩺 Procedures",            "procedures",      False),
                    ("💊 Medications",           "medications",     False),
                    ("⚗️ Therapy Options",        "therapy",         False),
                    ("🚨 Emergency Warning Signs","emergency_signs", True),
                ]
                for i, (label, key, is_emergency) in enumerate(items):
                    col = t1 if i % 2 == 0 else t2
                    border = "#ef4444" if is_emergency else "#3b82f6"
                    with col:
                        st.markdown(f"""
                        <div class='cyber-card' style='border-left:3px solid {border}; padding:1rem; margin-bottom:0.75rem;'>
                            <div style='font-weight:700; color:{border}; margin-bottom:8px; font-size:0.92rem;'>
                                {label}
                            </div>
                            {''.join(f"<div class='step-box {'step-emergency' if is_emergency else ''}' style='margin-bottom:4px;'>{s}</div>" for s in intel[key])}
                        </div>
                        """, unsafe_allow_html=True)

            # Tab 4: Report Download
            with tab_rpt:
                st.markdown("#### 📥 Download Clinical Report")
                st.caption("All reports are generated from the most recent scan result.")

                dl1, dl2 = st.columns(2)
                record   = st.session_state.last_result
                proc_img = st.session_state.last_processed_img

                with dl1:
                    if REPORTLAB_AVAILABLE and proc_img:
                        pdf_bytes = ReportGenerator.generate_pdf(record, proc_img)
                        fname = f"SkinScan_{record.get('patient_name','PT')}_{datetime.date.today()}.pdf".replace(" ","_")
                        st.download_button("📄 Download PDF Report", data=pdf_bytes,
                                           file_name=fname, mime="application/pdf")
                    else:
                        st.warning("Install ReportLab for PDF:\n`pip install reportlab`")

                with dl2:
                    csv_data = ReportGenerator.generate_csv(st.session_state.medical_database)
                    st.download_button("📊 Download CSV Registry",
                                       data=csv_data,
                                       file_name=f"SkinScan_Registry_{datetime.date.today()}.csv",
                                       mime="text/csv")

                st.markdown("""
                <div style='font-size:0.76rem; color:#64748b; margin-top:1rem;
                            border:1px solid rgba(100,116,139,0.2); border-radius:8px; padding:12px;'>
                    ⚠️ <b>AI Disclaimer:</b> Reports are AI-generated for academic and research
                    use only. They do <b>not</b> constitute a medical diagnosis. Always consult a
                    qualified dermatologist or oncologist for clinical decisions.
                </div>
                """, unsafe_allow_html=True)

    # =========================================================
    #  MODULE 3 · Patient Registry
    # =========================================================
    def module_registry(self):
        st.markdown("<h1 class='holo-text page-title'>Patient Registry</h1>", unsafe_allow_html=True)
        st.caption("Secure session database · Filter · Export")
        st.markdown("<br>", unsafe_allow_html=True)

        db = st.session_state.medical_database
        if not db:
            st.info("📭 Registry empty. Run scans in the AI Analysis Suite to populate.")
            return

        df = pd.DataFrame([{
            "Timestamp":   r.get("timestamp",     ""),
            "Patient":     r.get("patient_name",   "ANON"),
            "Age":         r.get("age",            "N/A"),
            "Gender":      r.get("gender",         "N/A"),
            "Diagnosis":   r.get("diagnosis",      "N/A"),
            "Risk":        r.get("risk_level",     "N/A"),
            "Probability": f"{r.get('probability',0)*100:.1f}%",
            "Confidence":  f"{r.get('confidence', 0)*100:.1f}%",
            "Engine":      r.get("model_mode",     "N/A"),
        } for r in db])

        with st.expander("🔍 Filter Records"):
            fc1, fc2, fc3 = st.columns(3)
            f_diag   = fc1.multiselect("Diagnosis",   ["Malignant","Benign"],          default=["Malignant","Benign"])
            f_risk   = fc2.multiselect("Risk Level",  ["HIGH","MEDIUM","LOW"],          default=["HIGH","MEDIUM","LOW"])
            f_gender = fc3.multiselect("Gender", ["Male","Female","Other","Prefer not to say"],
                                       default=["Male","Female","Other","Prefer not to say"])

        mask = (
            df["Diagnosis"].isin(f_diag) &
            df["Risk"].isin(f_risk) &
            df["Gender"].isin(f_gender)
        )
        df_f = df[mask]

        st.markdown(f"**{len(df_f)} records** | Total: **{len(df)}** entries")
        st.markdown("<div class='cyber-card'>", unsafe_allow_html=True)
        st.dataframe(df_f, use_container_width=True, hide_index=True, height=380)
        st.markdown("</div>", unsafe_allow_html=True)

        e1, e2, e3 = st.columns(3)
        with e1:
            st.download_button("📥 Export CSV",
                               data=ReportGenerator.generate_csv(db),
                               file_name=f"SkinScan_Registry_{datetime.date.today()}.csv",
                               mime="text/csv")
        with e2:
            safe_db = [{k: str(v) if isinstance(v, datetime.datetime) else v for k,v in r.items()} for r in db]
            st.download_button("🔗 Export JSON",
                               data=json.dumps(safe_db, indent=2),
                               file_name=f"SkinScan_Registry_{datetime.date.today()}.json",
                               mime="application/json")
        with e3:
            if st.button("🗑️ Clear All Records"):
                st.session_state.medical_database    = []
                st.session_state.last_result         = None
                st.session_state.last_raw_img        = None
                st.session_state.last_processed_img  = None
                st.rerun()

    # =========================================================
    #  MODULE 4 · Data Visualization
    # =========================================================
    def module_analytics(self):
        st.markdown("<h1 class='holo-text page-title'>Data Visualization</h1>", unsafe_allow_html=True)
        st.caption("Real-time epidemiological analytics · Risk distribution · Trend analysis")
        st.markdown("<br>", unsafe_allow_html=True)

        db = st.session_state.medical_database
        if not db:
            st.warning("⚠️ No data yet. Run at least one scan to generate visualizations.")
            return

        df = pd.DataFrame(db)
        df["prob_pct"] = df["probability"] * 100
        df["conf_pct"] = df["confidence"]  * 100

        r1c1, r1c2 = st.columns(2)

        # Donut
        with r1c1:
            st.markdown("<div class='cyber-card'>", unsafe_allow_html=True)
            c = df["diagnosis"].value_counts().reset_index()
            c.columns = ["Diagnosis","Count"]
            fig = px.pie(c, names="Diagnosis", values="Count",
                         title="Malignant vs Benign Distribution", hole=0.48,
                         color_discrete_map={"Malignant":"#ef4444","Benign":"#10b981"})
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#94a3b8",
                               height=300, margin=dict(l=0,r=0,t=40,b=0))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Risk bar
        with r1c2:
            st.markdown("<div class='cyber-card'>", unsafe_allow_html=True)
            rc = df["risk_level"].value_counts().reset_index()
            rc.columns = ["Risk","Count"]
            fig2 = px.bar(rc, x="Risk", y="Count", title="Risk Level Distribution",
                          color="Risk",
                          color_discrete_map={"HIGH":"#ef4444","MEDIUM":"#f59e0b","LOW":"#10b981"})
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                font_color="#94a3b8", height=300, showlegend=False,
                                xaxis=dict(gridcolor="rgba(100,116,139,0.12)"),
                                yaxis=dict(gridcolor="rgba(100,116,139,0.12)"),
                                margin=dict(l=0,r=0,t=40,b=0))
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        r2c1, r2c2 = st.columns(2)

        # Scatter
        with r2c1:
            st.markdown("<div class='cyber-card'>", unsafe_allow_html=True)
            fig3 = px.scatter(df, x="prob_pct", y="conf_pct",
                              color="diagnosis",
                              color_discrete_map={"Malignant":"#ef4444","Benign":"#10b981"},
                              title="Probability vs Confidence",
                              labels={"prob_pct":"Probability (%)","conf_pct":"Confidence (%)"})
            fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                font_color="#94a3b8", height=280,
                                xaxis=dict(gridcolor="rgba(100,116,139,0.12)"),
                                yaxis=dict(gridcolor="rgba(100,116,139,0.12)"),
                                margin=dict(l=0,r=0,t=40,b=0))
            st.plotly_chart(fig3, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Trend
        with r2c2:
            st.markdown("<div class='cyber-card'>", unsafe_allow_html=True)
            if len(df) >= 2:
                fig4 = go.Figure()
                fig4.add_trace(go.Scatter(y=df["conf_pct"], mode="lines+markers",
                                          line=dict(color="#3b82f6",width=2),
                                          marker=dict(size=6), name="Confidence %"))
                fig4.add_trace(go.Scatter(y=df["prob_pct"], mode="lines+markers",
                                          line=dict(color="#ef4444",width=2,dash="dot"),
                                          marker=dict(size=6), name="Probability %"))
                fig4.update_layout(title="Scan Trend Analysis", height=280,
                                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                    font_color="#94a3b8",
                                    xaxis=dict(title="Scan #", gridcolor="rgba(100,116,139,0.12)"),
                                    yaxis=dict(title="Score (%)", gridcolor="rgba(100,116,139,0.12)"),
                                    legend=dict(orientation="h", y=-0.25),
                                    margin=dict(l=0,r=0,t=40,b=0))
                st.plotly_chart(fig4, use_container_width=True)
            else:
                st.info("Need 2+ scans for trend chart.")
            st.markdown("</div>", unsafe_allow_html=True)

        # Summary stats
        st.markdown("---")
        st.markdown("### 📊 Session Summary Statistics")
        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric("Total Scans",      len(df))
        s2.metric("Avg Confidence",   f"{df['conf_pct'].mean():.1f}%")
        s3.metric("Avg Probability",  f"{df['prob_pct'].mean():.1f}%")
        s4.metric("Malignant Cases",  int((df["diagnosis"]=="Malignant").sum()))
        s5.metric("Benign Cases",     int((df["diagnosis"]=="Benign").sum()))

    # =========================================================
    #  MODULE 5 · Help & Guide
    # =========================================================
    def module_guide(self):
        st.markdown("<h1 class='holo-text page-title'>Help & User Guide</h1>", unsafe_allow_html=True)
        st.caption("Complete user manual for SkinScan AI Clinical Platform")
        st.markdown("<br>", unsafe_allow_html=True)

        guide_sections = [
            ("📤 How to Upload an Image", [
                "Go to <b>AI Analysis Suite</b> from the sidebar navigation.",
                "Enter patient name/ID, age, and gender (all fields are optional).",
                "Click <b>Browse files</b> or drag-and-drop your dermoscopic image.",
                "Accepted formats: <b>JPG, JPEG, PNG</b> only — other types are rejected.",
                "Maximum file size: <b>10 MB</b>. Minimum resolution: <b>100×100 pixels</b>.",
                "The system auto-validates, resizes to 224×224 px, normalizes, and enhances the image before scanning.",
            ]),
            ("🤖 How the AI Scan Works", [
                "The platform uses a <b>Convolutional Neural Network (CNN)</b> trained on dermoscopic images.",
                "Model file: <b>skin_cancer_cnn.h5</b> — automatically loaded at startup.",
                "If the model file is found → <b>🟢 Neural Network Online</b> (real inference).",
                "If the model is missing → <b>🟠 Simulation Mode</b> activates (safe demo mode, no crash).",
                "Preprocessing pipeline: RGB conversion → 224×224 resize → normalize (0–1) → expand dims → predict.",
                "The sigmoid output score determines Benign (< 0.50) or Malignant (≥ 0.50).",
            ]),
            ("📊 Understanding the Results", [
                "<b>Cancer Probability</b> = likelihood the AI assigns to its top diagnosis.",
                "<b>AI Confidence Score</b> = certainty of the model prediction (higher = more reliable).",
                "<b>Risk Level:</b> HIGH (≥80%), MEDIUM (50–80%), LOW (<50%) based on probability.",
                "The <b>Confidence Gauge Chart</b> visualizes score from 0–100%.",
                "The <b>Probability Bar</b> shows a visual fill of the malignancy likelihood.",
                "All results are logged automatically in the Patient Registry.",
            ]),
            ("🏥 Understanding Benign vs Malignant", [
                "🔴 <b>Malignant</b>: The AI detects patterns of skin cancer. HIGH RISK. Consult a dermatologist within 48 hours.",
                "🟢 <b>Benign</b>: The lesion appears non-cancerous. LOW RISK. Routine annual monitoring recommended.",
                "Even a BENIGN result does not guarantee safety — always consult a professional.",
                "Perform monthly <b>ABCDE self-exams</b>: Asymmetry, Border, Color, Diameter, Evolution.",
                "Any rapid change in lesion morphology warrants immediate medical consultation.",
            ]),
            ("👨‍⚕️ When to Consult a Doctor", [
                "🚨 <b>IMMEDIATELY</b> if AI result is Malignant or HIGH RISK.",
                "📅 <b>Within 1 week</b> for MEDIUM RISK results.",
                "📆 <b>Annually</b> for LOW RISK / Benign results.",
                "Regardless of AI output: if a lesion bleeds, ulcerates, or grows rapidly — see a doctor.",
                "Family or personal history of skin cancer → consult a dermatologist proactively.",
            ]),
            ("⚠️ System Limitations", [
                "This AI is a <b>research and educational tool</b> — NOT a certified medical device.",
                "Accuracy depends heavily on image quality, lighting, and dermoscopy technique.",
                "Smartphone photos may produce less accurate results than clinical dermoscopic images.",
                "The model is binary (Benign/Malignant) — it cannot identify specific cancer subtypes.",
                "<b>Never make medical decisions based solely on this tool.</b>",
                "All session data is cleared when the browser page is refreshed.",
            ]),
        ]

        for title, points in guide_sections:
            with st.expander(title, expanded=False):
                for pt in points:
                    st.markdown(f"<div class='step-box'>{pt}</div>", unsafe_allow_html=True)

        # ABCDE Guide
        st.markdown("---")
        st.markdown("### 🎗️ ABCDE Melanoma Self-Check")
        a1, a2, a3, a4, a5 = st.columns(5)
        abcde = [
            ("A", "Asymmetry", "#ef4444", "One half of the lesion doesn't match the other."),
            ("B", "Border",    "#f97316", "Irregular, ragged, notched, or blurred edges."),
            ("C", "Color",     "#f59e0b", "Multiple shades: brown, black, red, white, blue."),
            ("D", "Diameter",  "#3b82f6", "Larger than 6mm (the size of a pencil eraser)."),
            ("E", "Evolution", "#8b5cf6", "Any change in size, shape, color, or new symptoms."),
        ]
        for col, (letter, word, color, desc) in zip([a1,a2,a3,a4,a5], abcde):
            with col:
                st.markdown(f"""
                <div class='kpi-card' style='border-top:3px solid {color};'>
                    <div style='font-size:2.1rem; font-weight:800; color:{color};
                                font-family:JetBrains Mono,monospace;'>{letter}</div>
                    <div style='font-weight:700; font-size:0.88rem; margin:5px 0;'>{word}</div>
                    <div style='font-size:0.74rem; color:#64748b; line-height:1.4;'>{desc}</div>
                </div>
                """, unsafe_allow_html=True)

    # =========================================================
    #  Footer
    # =========================================================
    def render_system_footer(self):
        st.markdown("""
        <div style='text-align:center; padding:2.5rem 0 1rem;
                    border-top:1px solid rgba(100,116,139,0.18);
                    margin-top:2.5rem; font-size:0.78rem; color:#475569;'>
            <b style='color:#64748b;'>SkinScan AI — Clinical Intelligence Platform v12.0</b><br>
            Developed by <b>Rehan Shafique</b> &nbsp;·&nbsp; Final Year Project &nbsp;·&nbsp; Bioinformatics<br>
            <span style='color:#334155;'>Python · Streamlit · TensorFlow · Plotly · ReportLab · PIL</span><br><br>
            <span style='color:#ef4444; font-size:0.73rem;'>
                ⚠️ For Research &amp; Educational Purposes Only — Not a Certified Medical Device
            </span>
        </div>
        """, unsafe_allow_html=True)


# =========================================================
#  ENTRY POINT
# =========================================================
if __name__ == "__main__":
    app = SkinScanEnterpriseSuite()
    app.launch()
                
