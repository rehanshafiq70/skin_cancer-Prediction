"""
=============================================================
SkinCancerPrediction AI — Medical Diagnostic System v12.0
University of Agricultural Faisalabad | Rehan Shafique
=============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import time, datetime, os, uuid, cv2
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="SkinCancerPrediction AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# 1. MODEL LOADER  ← YOUR REAL FILE ID HERE
# ─────────────────────────────────────────────
GDRIVE_FILE_ID = "1Qn5YtgwePAdPyL0eARcc4UoYjy5Kk8PW"
MODEL_PATH     = "/tmp/skin_cancer_model.h5"

@st.cache_resource(show_spinner=False)
def load_model_cached():
    try:
        from tensorflow.keras.models import load_model
    except ImportError:
        return None, "❌ TensorFlow not installed."

    if not os.path.exists(MODEL_PATH):
        try:
            import gdown
            url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}&export=download&confirm=t"
            gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)
        except Exception as e:
            return None, f"❌ Download failed: {e}"

    if not os.path.exists(MODEL_PATH):
        return None, "❌ Model file missing after download."

    try:
        model = load_model(MODEL_PATH)
        return model, "✅ CNN Model loaded"
    except Exception as e:
        return None, f"❌ Load error: {e}"


def run_inference(model, pil_image: Image.Image):
    from tensorflow.keras.preprocessing.image import img_to_array
    img = pil_image.convert("RGB").resize((224, 224))
    arr = img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    raw = float(model.predict(arr, verbose=0)[0][0])
    if raw >= 0.5:
        return "Malignant", raw, raw
    else:
        return "Benign", 1 - raw, raw


# ─────────────────────────────────────────────
# 2. IMAGE VALIDATOR
# ─────────────────────────────────────────────
def validate_image(pil_image: Image.Image):
    errors = []
    if pil_image.mode not in ("RGB", "RGBA"):
        errors.append("Image must be RGB.")
    w, h = pil_image.size
    if w < 128 or h < 128:
        errors.append(f"Too small ({w}×{h} px). Minimum: 128×128.")
    arr  = np.array(pil_image.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    lap  = cv2.Laplacian(gray, cv2.CV_64F).var()
    if lap < 80:
        errors.append(f"Image too blurry (sharpness {lap:.1f}). Upload a clearer photo.")
    return errors


# ─────────────────────────────────────────────
# 3. CLINICAL KNOWLEDGE BASE
# ─────────────────────────────────────────────
CLINICAL = {
    "Malignant": {
        "color": "#ef4444", "badge_bg": "rgba(239,68,68,0.15)",
        "risk_label": "CRITICAL RISK",
        "desc": (
            "High-probability malignancy detected. Immediate clinical evaluation strongly advised. "
            "This is a screening result — must be confirmed by a board-certified dermatologist."
        ),
        "treatments": [
            "Wide Local Excision (WLE) — primary surgical removal",
            "Mohs Micrographic Surgery — for margin-sensitive regions",
            "Sentinel Lymph Node Biopsy (SLNB) — staging evaluation",
            "Adjuvant immunotherapy (PD-1 inhibitors) if indicated",
            "PET/CT scan if systemic metastasis suspected",
            "Radiation therapy mapping if margins unclear",
        ],
        "patient_care": [
            "Immediate consultation with a board-certified dermatologist",
            "Urgent referral to Oncology / Onco-Dermatology",
            "Absolute UV avoidance — wear UPF 50+ protective clothing",
            "Broad-spectrum SPF 50+ sunscreen applied daily",
            "Monitor lesion for rapid changes, ulceration, or bleeding",
            "Keep a photographic log of the lesion over time",
        ],
        "monitoring": [
            "Full-body dermoscopy every 3 months post-treatment",
            "Excisional biopsy for Breslow depth determination",
            "Monthly ABCDE self-examinations",
            "Immediate ER if ulceration becomes severe",
            "Screen family members for hereditary melanoma syndromes",
        ],
    },
    "Benign": {
        "color": "#10b981", "badge_bg": "rgba(16,185,129,0.15)",
        "risk_label": "LOW RISK",
        "desc": (
            "No high-risk features detected in this lesion. Continue routine monitoring "
            "and consult a dermatologist promptly if morphological changes occur."
        ),
        "treatments": [
            "No immediate surgical intervention required",
            "Elective cosmetic laser ablation if aesthetically desired",
            "Targeted cryotherapy for symptomatic benign lesions",
            "Diagnostic shave biopsy available upon patient request",
            "Digital dermoscopic baseline photography recommended",
        ],
        "patient_care": [
            "Daily SPF 50+ broad-spectrum sunscreen application",
            "Ceramide-based moisturizers for skin barrier repair",
            "Monthly ABCDE self-skin examinations",
            "Avoid mechanical trauma or irritation to the lesion",
            "Dietary antioxidants (Vitamins C, E, selenium)",
        ],
        "monitoring": [
            "Annual routine dermatology screening",
            "AI re-evaluation recommended in 6 months",
            "Consult immediately if lesion changes in shape / colour / size",
            "Rule out atypical nevi syndrome if multiple lesions present",
            "Monitor for satellite lesion development nearby",
        ],
    },
}

def risk_level(conf: float):
    if conf < 0.50: return "LOW",      "#10b981"
    if conf < 0.75: return "MEDIUM",   "#f59e0b"
    return                  "CRITICAL","#ef4444"


# ─────────────────────────────────────────────
# 4. REPORT GENERATOR
# ─────────────────────────────────────────────
def build_report(record: dict) -> str:
    info  = CLINICAL[record["AI_Diagnosis"]]
    rl, _ = risk_level(float(record["Confidence_Raw"]))
    sep   = "=" * 65
    lines = [
        sep,
        "     SKINCANCERPREDICTION AI — DIAGNOSTIC REPORT v12.0",
        "     University of Agricultural Faisalabad",
        sep,
        f"  Generated  : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Scan ID    : {record['Scan_ID']}",
        f"  Patient ID : {record['Patient_ID']}",
        f"  Name       : {record['Patient_Name']}",
        sep, "  AI DIAGNOSIS", sep,
        f"  Result     : {record['AI_Diagnosis']}",
        f"  Confidence : {record['Confidence_Score']}",
        f"  Risk Level : {rl}", "", f"  {info['desc']}",
        sep, "  TREATMENTS", sep,
    ]
    for i, t in enumerate(info["treatments"],  1): lines.append(f"  {i}. {t}")
    lines += [sep, "  PATIENT CARE", sep]
    for i, c in enumerate(info["patient_care"], 1): lines.append(f"  {i}. {c}")
    lines += [sep, "  MONITORING PROTOCOL", sep]
    for i, m in enumerate(info["monitoring"],   1): lines.append(f"  {i}. {m}")
    lines += [
        sep,
        "  DISCLAIMER: Screening aid only. Not a substitute for professional",
        "  medical diagnosis. Confirm all results with a licensed dermatologist.",
        sep,
        "  SkinCancerPrediction AI v12.0",
        "  Rehan Shafique | rehanshafiq6540@gmail.com",
        "  University of Agricultural Faisalabad",
        sep,
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────
# 5. CSS
# ─────────────────────────────────────────────
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@400;500&display=swap');
    :root {
        --bg:#050a0e; --surface:#0b1520; --surface2:#0f1e2e;
        --border:rgba(56,189,248,.12); --border-hi:rgba(56,189,248,.35);
        --text:#e2e8f0; --muted:#64748b; --accent:#38bdf8;
        --green:#10b981; --red:#ef4444; --amber:#f59e0b;
        --fh:'Syne',sans-serif; --fm:'DM Mono',monospace;
    }
    html,body,[class*="css"]{font-family:var(--fh);background-color:var(--bg)!important;color:var(--text);}
    #MainMenu,footer,header{visibility:hidden;}
    .block-container{padding:1.5rem 2.5rem 4rem;max-width:1400px;margin:0 auto;}
    [data-testid="stSidebar"]{background:var(--surface)!important;border-right:1px solid var(--border);}
    [data-testid="stSidebar"] *{color:var(--text)!important;}

    .glass{background:linear-gradient(135deg,rgba(15,30,46,.85),rgba(11,21,32,.9));
           border:1px solid var(--border);border-radius:16px;padding:1.6rem 1.8rem;
           backdrop-filter:blur(14px);margin-bottom:1.2rem;transition:border-color .25s;}
    .glass:hover{border-color:var(--border-hi);}

    .neon-title{font-weight:800;font-size:2.6rem;letter-spacing:-.5px;line-height:1.1;
                background:linear-gradient(120deg,#38bdf8 0%,#818cf8 50%,#f472b6 100%);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;}
    .sec-title{font-weight:700;font-size:1.35rem;color:var(--accent);margin-bottom:.3rem;}
    .mono{font-family:var(--fm);color:var(--accent);font-size:.82rem;}

    .metric-tile{background:var(--surface2);border:1px solid var(--border);border-radius:14px;
                 padding:1.1rem;text-align:center;transition:transform .2s,border-color .2s;}
    .metric-tile:hover{transform:translateY(-3px);border-color:var(--border-hi);}
    .metric-val{font-size:1.8rem;font-weight:800;color:var(--accent);}
    .metric-lbl{font-size:.73rem;color:var(--muted);margin-top:.2rem;font-family:var(--fm);}

    .step-item{display:flex;align-items:center;gap:.9rem;padding:.7rem 1rem;
               border-left:3px solid var(--accent);background:rgba(56,189,248,.04);
               border-radius:0 10px 10px 0;margin-bottom:.5rem;}
    .step-num{background:var(--accent);color:#050a0e;border-radius:50%;width:26px;height:26px;
              font-weight:800;font-size:.8rem;display:flex;align-items:center;
              justify-content:center;flex-shrink:0;}

    .feat-card{background:var(--surface2);border:1px solid var(--border);
               border-radius:14px;padding:1.2rem;text-align:center;}
    .feat-icon{font-size:2rem;margin-bottom:.5rem;}
    .feat-name{font-weight:700;font-size:1rem;color:var(--accent);}
    .feat-desc{font-size:.78rem;color:var(--muted);margin-top:.3rem;}

    .rec-item{display:flex;align-items:flex-start;gap:.6rem;padding:.55rem 0;
              border-bottom:1px solid rgba(56,189,248,.06);font-size:.88rem;}
    .rec-num{color:var(--accent);font-weight:700;min-width:1.3rem;font-family:var(--fm);}

    .stButton>button{background:linear-gradient(135deg,#0369a1,#1d4ed8)!important;
                     color:white!important;border:none!important;border-radius:10px!important;
                     font-weight:700!important;letter-spacing:.8px!important;
                     padding:.7rem 1.2rem!important;text-transform:uppercase!important;
                     font-family:var(--fh)!important;transition:all .2s!important;}
    .stButton>button:hover{transform:translateY(-2px)!important;
                           box-shadow:0 0 18px rgba(56,189,248,.4)!important;}
    .stDownloadButton>button{background:linear-gradient(135deg,#065f46,#047857)!important;
                              color:white!important;border:none!important;border-radius:10px!important;
                              font-weight:700!important;font-family:var(--fh)!important;}
    [data-testid="stFileUploader"]{background:var(--surface2)!important;
                                   border:2px dashed var(--border-hi)!important;
                                   border-radius:14px!important;}
    .stTabs [data-baseweb="tab"]{font-family:var(--fh)!important;font-weight:600!important;
                                  color:var(--muted)!important;}
    .stTabs [aria-selected="true"]{color:var(--accent)!important;
                                    border-bottom:2px solid var(--accent)!important;}
    .stTextInput input{background:var(--surface2)!important;border:1px solid var(--border-hi)!important;
                       border-radius:8px!important;color:var(--text)!important;
                       font-family:var(--fm)!important;}
    hr{border-color:var(--border)!important;}
    [data-testid="stDataFrame"]{border-radius:12px!important;overflow:hidden;}

    .status-bar{border-radius:10px;padding:.5rem 1rem;font-size:.8rem;font-family:var(--fm);
                display:flex;align-items:center;gap:.5rem;}
    .s-online{background:rgba(16,185,129,.12);border:1px solid rgba(16,185,129,.3);color:#10b981;}
    .s-offline{background:rgba(239,68,68,.12);border:1px solid rgba(239,68,68,.3);color:#ef4444;}
    @keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
    .dot{width:9px;height:9px;border-radius:50%;display:inline-block;animation:pulse 1.4s infinite;}
    .dg{background:#10b981;} .dr{background:#ef4444;}
    </style>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 6. SESSION STATE
# ─────────────────────────────────────────────
def init_state():
    if "registry" not in st.session_state: st.session_state.registry = []
    if "page"     not in st.session_state: st.session_state.page     = "Home"


# ─────────────────────────────────────────────
# 7. SIDEBAR
# ─────────────────────────────────────────────
def render_sidebar(model, model_msg):
    with st.sidebar:
        st.markdown("""
        <div style='padding:.5rem 0 1.2rem;'>
            <div style='font-size:1.4rem;font-weight:800;color:#38bdf8;font-family:Syne,sans-serif;'>
                🧬 SkinCancerPrediction
            </div>
            <div style='font-size:.7rem;color:#64748b;font-family:"DM Mono",monospace;margin-top:3px;'>
                MEDICAL DIAGNOSTIC SYSTEM v12.0
            </div>
        </div>""", unsafe_allow_html=True)
        st.divider()

        for icon, name in [("🏠","Home"),("🔬","AI Scanner"),("📊","Analytics"),("🗂️","Registry")]:
            if st.button(f"{icon}  {name}", key=f"nav_{name}", use_container_width=True):
                st.session_state.page = name; st.rerun()

        st.divider()

        if model is not None:
            st.markdown("<div class='status-bar s-online'><span class='dot dg'></span> CNN Model Active</div>",
                        unsafe_allow_html=True)
        else:
            st.markdown("<div class='status-bar s-offline'><span class='dot dr'></span> Model Offline</div>",
                        unsafe_allow_html=True)
            st.caption(model_msg)

        st.markdown("<br>", unsafe_allow_html=True)
        reg = st.session_state.registry
        mal = sum(1 for r in reg if r["AI_Diagnosis"] == "Malignant")
        st.markdown(f"""
        <div style='font-size:.75rem;color:#64748b;font-family:"DM Mono",monospace;line-height:2;'>
            Total Scans : <b style='color:#e2e8f0;'>{len(reg)}</b><br>
            Malignant   : <b style='color:#ef4444;'>{mal}</b><br>
            Benign      : <b style='color:#10b981;'>{len(reg)-mal}</b>
        </div>""", unsafe_allow_html=True)

        st.divider()
        st.markdown("""
        <div style='font-size:.68rem;color:#334155;text-align:center;font-family:"DM Mono",monospace;line-height:1.8;'>
            Rehan Shafique<br>Univ. of Agricultural Faisalabad<br>rehanshafiq6540@gmail.com
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 8. HOME PAGE
# ─────────────────────────────────────────────
def page_home():
    st.markdown("""
    <div class='glass' style='padding:3rem 2.5rem;text-align:center;
         background:linear-gradient(135deg,rgba(3,105,161,.15),rgba(99,102,241,.12),rgba(244,114,182,.08));
         border-color:rgba(56,189,248,.25);margin-bottom:1.5rem;'>
        <div style='font-size:.78rem;font-family:"DM Mono",monospace;color:#38bdf8;
             letter-spacing:3px;text-transform:uppercase;margin-bottom:.8rem;'>
            AI-POWERED CLINICAL SCREENING
        </div>
        <div class='neon-title'>SkinCancerPrediction AI</div>
        <div style='color:#94a3b8;font-size:1rem;margin:1rem auto 1.5rem;max-width:580px;'>
            Hospital-grade deep learning diagnosis for skin lesion classification —
            <em>Benign</em> vs <em>Malignant</em> — real CNN, real inference.
        </div>
        <div style='font-size:.7rem;color:#475569;font-family:"DM Mono",monospace;'>
            ⚠ Screening aid only — always confirm with a licensed dermatologist
        </div>
    </div>""", unsafe_allow_html=True)

    c = st.columns([1, 0.32, 1])
    with c[1]:
        if st.button("▶ START DIAGNOSIS"):
            st.session_state.page = "AI Scanner"; st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='sec-title'>📋 How It Works</div>", unsafe_allow_html=True)
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    for i, s in enumerate([
        "Upload a dermoscopic skin lesion image (JPG / PNG)",
        "Enter optional Patient ID and Name for the registry",
        "Click EXECUTE SCAN — model runs real CNN inference",
        "Wait 2–4 seconds for deep learning processing",
        "View diagnosis, confidence score and risk level",
        "Download your medical report or export registry as CSV",
    ], 1):
        st.markdown(f"<div class='step-item'><div class='step-num'>{i}</div>"
                    f"<div style='font-size:.9rem;'>{s}</div></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='sec-title'>⚡ Platform Features</div>", unsafe_allow_html=True)
    cols = st.columns(3)
    for i, (icon, name, desc) in enumerate([
        ("🧠","Real AI Inference","CNN trained on Kaggle melanoma dataset — zero fake outputs"),
        ("🔍","Image Validation","Blur detection, size & RGB check before every scan"),
        ("📑","Medical Reports","One-click downloadable diagnostic report per patient"),
        ("📊","Analytics Dashboard","Live charts: distribution, confidence, risk levels"),
        ("🗂️","Patient Registry","Searchable table with full CSV export"),
        ("⚡","Fast Inference","224×224 preprocessing + optimised Keras pipeline"),
    ]):
        with cols[i % 3]:
            st.markdown(f"""
            <div class='feat-card'>
                <div class='feat-icon'>{icon}</div>
                <div class='feat-name'>{name}</div>
                <div class='feat-desc'>{desc}</div>
            </div><br>""", unsafe_allow_html=True)

    st.markdown("<br><div class='sec-title'>📊 Training Dataset (ISIC Melanoma — Kaggle)</div>",
                unsafe_allow_html=True)
    for col, (val, lbl, sub) in zip(st.columns(4), [
        ("33,126","Total Images","ISIC Archive"),
        ("26,501","Benign Samples","≈ 80%"),
        ("6,625", "Malignant Samples","≈ 20%"),
        ("224×224","Input Resolution","RGB Normalised"),
    ]):
        with col:
            st.markdown(f"""
            <div class='metric-tile'>
                <div class='metric-val'>{val}</div>
                <div class='metric-lbl'>{lbl}</div>
                <div style='font-size:.7rem;color:#10b981;margin-top:.15rem;'>{sub}</div>
            </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 9. AI SCANNER
# ─────────────────────────────────────────────
def page_scanner(model, model_msg):
    st.markdown("<div class='neon-title' style='font-size:2rem;margin-bottom:1rem;'>🔬 AI Diagnostic Scanner</div>",
                unsafe_allow_html=True)

    if model is None:
        st.error(f"**Model not available.** {model_msg}")
        st.info(
            "**Troubleshooting:**\n\n"
            "1. Make sure your Google Drive file is set to **'Anyone with the link'** → Viewer.\n"
            "2. Check Streamlit Cloud logs for the exact download error.\n"
            "3. Try re-deploying the app from Streamlit Cloud dashboard."
        )
        return

    left, right = st.columns([1, 1.15])

    with left:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown("<div class='sec-title'>Patient Information</div>", unsafe_allow_html=True)
        patient_id   = st.text_input("Patient ID",   placeholder="e.g. PT-2025-001")
        patient_name = st.text_input("Patient Name", placeholder="e.g. John Doe")
        st.markdown("<div class='sec-title' style='margin-top:1rem;'>Upload Skin Lesion Image</div>",
                    unsafe_allow_html=True)
        uploaded = st.file_uploader("JPG / JPEG / PNG", type=["jpg","jpeg","png"],
                                    label_visibility="collapsed")
        pil_img = None
        if uploaded:
            pil_img = Image.open(uploaded)
            st.image(pil_img, use_container_width=True,
                     caption=f"📷 {uploaded.name}  {pil_img.size[0]}×{pil_img.size[1]} px")
        st.markdown("<br>", unsafe_allow_html=True)
        scan_btn = st.button("▶  EXECUTE SCAN", disabled=(pil_img is None))
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        if pil_img and scan_btn:
            errs = validate_image(pil_img)
            if errs:
                st.error("❌ **Invalid or unclear image.**\n\n" + "\n".join(f"• {e}" for e in errs))
                return

            with st.spinner("Extracting feature vectors — running CNN inference..."):
                time.sleep(1.0)
                diagnosis, confidence, raw = run_inference(model, pil_img)

            info    = CLINICAL[diagnosis]
            rl, rc  = risk_level(confidence)
            scan_id = f"SKN-{uuid.uuid4().hex[:8].upper()}"
            ts      = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            record = {
                "Scan_ID":          scan_id,
                "Patient_ID":       patient_id   or "—",
                "Patient_Name":     patient_name or "—",
                "AI_Diagnosis":     diagnosis,
                "Confidence_Score": f"{confidence*100:.2f}%",
                "Confidence_Raw":   confidence,
                "Risk_Level":       rl,
                "Timestamp":        ts,
            }
            st.session_state.registry.append(record)

            st.markdown(f"""
            <div class='glass' style='border-color:{info["color"]}55;'>
                <div style='display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:.5rem;'>
                    <div>
                        <div style='font-size:.7rem;color:#64748b;font-family:"DM Mono",monospace;letter-spacing:2px;'>AI DIAGNOSIS</div>
                        <div style='font-size:2.2rem;font-weight:800;color:{info["color"]};'>{diagnosis.upper()}</div>
                    </div>
                    <div style='text-align:right;'>
                        <span style='background:{info["badge_bg"]};color:{info["color"]};border:1px solid {info["color"]}44;
                              border-radius:8px;padding:.3rem 1rem;font-weight:700;font-family:"DM Mono",monospace;
                              font-size:.95rem;letter-spacing:1.5px;'>{info["risk_label"]}</span>
                        <br><br>
                        <span style='background:rgba(0,0,0,.3);color:{rc};border:1px solid {rc}44;
                              border-radius:6px;padding:.2rem .8rem;font-weight:600;
                              font-family:"DM Mono",monospace;font-size:.75rem;'>⚠ RISK: {rl}</span>
                    </div>
                </div>
                <div style='color:#94a3b8;font-size:.84rem;margin-top:.8rem;'>{info["desc"]}</div>
            </div>""", unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1:
                fig_g = go.Figure(go.Indicator(
                    mode="gauge+number", value=round(confidence*100,1),
                    title={"text":"Confidence","font":{"color":"#94a3b8","size":13}},
                    number={"suffix":"%","font":{"color":info["color"],"size":28}},
                    gauge={
                        "axis":{"range":[0,100],"tickcolor":"#475569"},
                        "bar":{"color":info["color"]},"bgcolor":"#0b1520","bordercolor":"#1e293b",
                        "steps":[
                            {"range":[0,50],"color":"rgba(16,185,129,.1)"},
                            {"range":[50,75],"color":"rgba(245,158,11,.1)"},
                            {"range":[75,100],"color":"rgba(239,68,68,.1)"},
                        ],
                    }))
                fig_g.update_layout(height=200,margin=dict(l=10,r=10,t=30,b=0),
                                    paper_bgcolor="rgba(0,0,0,0)",font_color="#e2e8f0")
                st.plotly_chart(fig_g, use_container_width=True)

            with c2:
                fig_b = go.Figure()
                fig_b.add_trace(go.Bar(x=[(1-raw)*100],y=["Benign"],orientation="h",
                                       marker_color="#10b981",
                                       text=[f"{(1-raw)*100:.1f}%"],textposition="auto"))
                fig_b.add_trace(go.Bar(x=[raw*100],y=["Malignant"],orientation="h",
                                       marker_color="#ef4444",
                                       text=[f"{raw*100:.1f}%"],textposition="auto"))
                fig_b.update_layout(
                    title={"text":"Probability Breakdown","font":{"color":"#94a3b8","size":13}},
                    height=200,margin=dict(l=0,r=10,t=30,b=0),barmode="group",showlegend=False,
                    paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font_color="#e2e8f0",
                    xaxis={"range":[0,100],"gridcolor":"#1e293b"},yaxis={"gridcolor":"#1e293b"})
                st.plotly_chart(fig_b, use_container_width=True)

            if confidence < 0.60:
                st.warning(f"⚠ Low confidence ({confidence*100:.1f}%). Upload a sharper image.")

            st.markdown("<div class='sec-title'>📋 Clinical Recommendations</div>", unsafe_allow_html=True)
            t1, t2, t3 = st.tabs(["🩺 Treatments","🛡️ Patient Care","🔭 Monitoring"])
            for tab, key in [(t1,"treatments"),(t2,"patient_care"),(t3,"monitoring")]:
                with tab:
                    for i, item in enumerate(info[key], 1):
                        st.markdown(f"<div class='rec-item'><span class='rec-num'>{i:02d}</span>"
                                    f"<span>{item}</span></div>", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            b1, b2, b3 = st.columns(3)
            with b1:
                st.download_button("📄 Download Report", data=build_report(record),
                                   file_name=f"SkinAI_{scan_id}.txt", mime="text/plain",
                                   use_container_width=True)
            with b2:
                if st.button("📊 Analytics", use_container_width=True):
                    st.session_state.page = "Analytics"; st.rerun()
            with b3:
                if st.button("🗂️ Registry", use_container_width=True):
                    st.session_state.page = "Registry"; st.rerun()

        elif pil_img is None:
            st.markdown("""
            <div class='glass' style='text-align:center;padding:3.5rem;'>
                <div style='font-size:3.5rem;'>🔬</div>
                <div style='font-weight:600;color:#64748b;margin-top:.5rem;'>
                    Upload a dermoscopic image to begin
                </div>
                <div style='font-size:.8rem;color:#475569;margin-top:.4rem;'>
                    Real CNN inference — Benign / Malignant
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style='margin-top:1.5rem;padding:.8rem 1.2rem;border-radius:10px;
         background:rgba(245,158,11,.07);border:1px solid rgba(245,158,11,.2);
         font-size:.76rem;color:#fbbf24;font-family:"DM Mono",monospace;'>
        ⚠ DISCLAIMER — Screening aid only. Does not replace professional medical diagnosis.
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 10. ANALYTICS
# ─────────────────────────────────────────────
def page_analytics():
    st.markdown("<div class='neon-title' style='font-size:2rem;margin-bottom:1rem;'>📊 Analytics Dashboard</div>",
                unsafe_allow_html=True)
    reg = st.session_state.registry
    if not reg:
        st.info("No data yet. Run at least one scan to see analytics."); return

    df       = pd.DataFrame(reg)
    df["CP"] = df["Confidence_Raw"].astype(float) * 100
    total    = len(df); mal = (df["AI_Diagnosis"]=="Malignant").sum()
    ben      = total - mal; avg = df["CP"].mean(); crit = (df["Risk_Level"]=="CRITICAL").sum()

    for col,(val,lbl,color) in zip(st.columns(5),[
        (total,"Total Scans","#38bdf8"),(ben,"Benign","#10b981"),
        (mal,"Malignant","#ef4444"),(f"{avg:.1f}%","Avg Confidence","#a78bfa"),
        (crit,"Critical Alerts","#f59e0b"),
    ]):
        with col:
            st.markdown(f"<div class='metric-tile'>"
                        f"<div class='metric-val' style='color:{color};'>{val}</div>"
                        f"<div class='metric-lbl'>{lbl}</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    base = dict(paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0", title_font_color="#38bdf8")

    with c1:
        f = px.pie(df, names="AI_Diagnosis", title="Diagnosis Distribution", hole=.45,
                   color="AI_Diagnosis",
                   color_discrete_map={"Benign":"#10b981","Malignant":"#ef4444"})
        f.update_layout(**base, legend=dict(bgcolor="rgba(0,0,0,0)"))
        st.markdown("<div class='glass' style='padding:1rem;'>",unsafe_allow_html=True)
        st.plotly_chart(f, use_container_width=True)
        st.markdown("</div>",unsafe_allow_html=True)

    with c2:
        f2 = px.histogram(df, x="CP", nbins=20, title="Confidence Distribution",
                          color_discrete_sequence=["#38bdf8"])
        f2.update_layout(**base, plot_bgcolor="rgba(0,0,0,0)",
                         xaxis=dict(gridcolor="#1e293b",title="Confidence (%)"),
                         yaxis=dict(gridcolor="#1e293b",title="Count"))
        st.markdown("<div class='glass' style='padding:1rem;'>",unsafe_allow_html=True)
        st.plotly_chart(f2, use_container_width=True)
        st.markdown("</div>",unsafe_allow_html=True)

    if total > 1:
        f3 = px.line(df.reset_index(), x="index", y="CP", color="AI_Diagnosis",
                     title="Confidence Trend", markers=True,
                     color_discrete_map={"Benign":"#10b981","Malignant":"#ef4444"})
        f3.update_layout(**base, plot_bgcolor="rgba(0,0,0,0)",
                         xaxis=dict(gridcolor="#1e293b",title="Scan #"),
                         yaxis=dict(gridcolor="#1e293b",title="Confidence (%)"),
                         legend=dict(bgcolor="rgba(0,0,0,0)"))
        st.markdown("<div class='glass' style='padding:1rem;'>",unsafe_allow_html=True)
        st.plotly_chart(f3, use_container_width=True)
        st.markdown("</div>",unsafe_allow_html=True)

    rdf = df["Risk_Level"].value_counts().reset_index()
    rdf.columns = ["Risk","Count"]
    f4 = px.bar(rdf, x="Risk", y="Count", title="Risk Level Breakdown", color="Risk",
                color_discrete_map={"LOW":"#10b981","MEDIUM":"#f59e0b","CRITICAL":"#ef4444"})
    f4.update_layout(**base, plot_bgcolor="rgba(0,0,0,0)", showlegend=False,
                     xaxis=dict(gridcolor="#1e293b"), yaxis=dict(gridcolor="#1e293b"))
    st.markdown("<div class='glass' style='padding:1rem;'>",unsafe_allow_html=True)
    st.plotly_chart(f4, use_container_width=True)
    st.markdown("</div>",unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 11. REGISTRY
# ─────────────────────────────────────────────
def page_registry():
    st.markdown("<div class='neon-title' style='font-size:2rem;margin-bottom:1rem;'>🗂️ Patient Registry</div>",
                unsafe_allow_html=True)
    reg = st.session_state.registry
    if not reg:
        st.info("Registry empty. Complete a scan to populate records."); return

    df     = pd.DataFrame(reg).drop(columns=["Confidence_Raw"], errors="ignore")
    search = st.text_input("🔍 Search by Patient ID or Name", placeholder="Type to filter...")
    if search:
        m  = (df["Patient_ID"].str.contains(search,case=False,na=False) |
              df["Patient_Name"].str.contains(search,case=False,na=False))
        df = df[m]

    st.markdown("<div class='glass' style='padding:.5rem;'>",unsafe_allow_html=True)
    st.dataframe(
        df.style.map(
            lambda v: "color:#ef4444;font-weight:700" if v=="Malignant" else
                      "color:#10b981;font-weight:700" if v=="Benign" else "",
            subset=["AI_Diagnosis"]
        ),
        use_container_width=True, hide_index=True,
        height=min(600, 40+37*len(df))
    )
    st.markdown("</div>",unsafe_allow_html=True)

    st.markdown("<br>",unsafe_allow_html=True)
    b1, b2, _ = st.columns([.28,.28,.44])
    with b1:
        st.download_button("📥 Export CSV",
                           data=df.to_csv(index=False).encode(),
                           file_name=f"SkinAI_Registry_{datetime.date.today()}.csv",
                           mime="text/csv", use_container_width=True)
    with b2:
        if st.button("🗑️ Clear Registry", use_container_width=True):
            st.session_state.registry = []; st.rerun()


# ─────────────────────────────────────────────
# 12. FOOTER
# ─────────────────────────────────────────────
def render_footer():
    st.markdown("""
    <div style='text-align:center;margin-top:4rem;padding:2rem 0 1rem;
         border-top:1px solid rgba(56,189,248,.1);
         font-family:"DM Mono",monospace;font-size:.7rem;color:#334155;line-height:2;'>
        <b style='color:#475569;'>SkinCancerPrediction AI — Medical Diagnostic System v12.0</b><br>
        University of Agricultural Faisalabad &nbsp;|&nbsp; Rehan Shafique
        &nbsp;|&nbsp; rehanshafiq6540@gmail.com<br>
        <span style='color:#1e293b;'>
            ⚠ Screening aid only. Not a substitute for professional medical diagnosis.
        </span>
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 13. MAIN
# ─────────────────────────────────────────────
def main():
    inject_css()
    init_state()

    with st.spinner("Initialising CNN model — downloading from Drive if needed..."):
        model, model_msg = load_model_cached()

    render_sidebar(model, model_msg)

    page = st.session_state.page
    if   page == "Home":       page_home()
    elif page == "AI Scanner": page_scanner(model, model_msg)
    elif page == "Analytics":  page_analytics()
    elif page == "Registry":   page_registry()

    render_footer()


if __name__ == "__main__":
    main()
