import streamlit as st
import pandas as pd
from PIL import Image

from model_loader import load_model_safe
from utils import (
    preprocess_image,
    validate_image,
    predict_skin_cancer,
    risk_level
)
from report import generate_report
from style import load_global_style

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="SkinScan AI",
    layout="wide",
    page_icon="🧬"
)

st.markdown(load_global_style(), unsafe_allow_html=True)

# =========================
# SESSION DATABASE
# =========================
if "history" not in st.session_state:
    st.session_state.history = []

# =========================
# SIDEBAR
# =========================
st.sidebar.title("🧬 SkinScan AI")

page = st.sidebar.radio(
    "Navigation",
    ["Home","Scanner","Reports","Analytics"]
)

# =========================
# HOME PAGE
# =========================
def home():

    st.title("AI Skin Cancer Detection")

    col1,col2,col3 = st.columns(3)

    col1.metric("Total Scans",
                len(st.session_state.history))

    col2.metric("Model Accuracy","92%")

    last = (
        st.session_state.history[-1]["label"]
        if st.session_state.history else "None"
    )

    col3.metric("Last Result", last)

    st.info("""
Step 1: Upload image  
Step 2: Click Analyze  
Step 3: View diagnosis  
Step 4: Download report
""")

# =========================
# SCANNER PAGE
# =========================
def scanner():

    st.header("🔬 Skin Scanner")

    file = st.file_uploader("Upload skin image")

    if not file:
        return

    image = Image.open(file)
    st.image(image,width=350)

    if st.button("Analyze"):

        if not validate_image(image):
            st.error("Invalid or unclear image")
            return

        with st.spinner("Running AI analysis..."):

            model = load_model_safe()
            data = preprocess_image(image)

            label,conf,prob = \
                predict_skin_cancer(model,data)

            risk = risk_level(conf)

            st.success(f"Diagnosis: {label}")
            st.metric("Confidence",f"{conf:.2f}%")

            report = generate_report(label,conf,risk)

            st.download_button(
                "Download Report",
                report,
                "report.txt"
            )

            st.session_state.history.append({
                "label":label,
                "confidence":conf
            })

# =========================
# REPORT PAGE
# =========================
def reports():

    st.header("Patient Reports")

    if not st.session_state.history:
        st.warning("No records yet.")
        return

    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df)

    st.download_button(
        "Export CSV",
        df.to_csv(index=False),
        "records.csv"
    )

# =========================
# ANALYTICS PAGE
# =========================
def analytics():

    import plotly.express as px

    if not st.session_state.history:
        st.info("No data available")
        return

    df = pd.DataFrame(st.session_state.history)

    st.plotly_chart(
        px.pie(df,names="label",
               title="Cancer Ratio")
    )

    st.plotly_chart(
        px.histogram(df,x="confidence",
                     title="Confidence Distribution")
    )

# =========================
# ROUTER
# =========================
if page=="Home":
    home()
elif page=="Scanner":
    scanner()
elif page=="Reports":
    reports()
elif page=="Analytics":
    analytics()

# =========================
# FOOTER
# =========================
st.markdown("""
---
University of Agriculture Faisalabad  
Rehan Shafique  
rehanshafiq6540@gmail.com
""")
