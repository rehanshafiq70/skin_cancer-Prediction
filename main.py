import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
from model_loader import load_ai_model
from utils import preprocess_image, predict
from report_generator import generate_report

st.set_page_config(page_title="SkinScan AI", layout="wide")

# ---------------- SIDEBAR ----------------
st.sidebar.title("🧬 SkinScan AI")

page = st.sidebar.radio(
    "Navigation",
    ["Home", "Scan", "Reports", "Analytics"]
)

if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- HOME ----------------
if page == "Home":

    st.title("🧬 Skin Cancer Detection Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Scans", len(st.session_state.history))
    col2.metric("Accuracy", "92%")
    col3.metric(
        "Last Result",
        st.session_state.history[-1]["result"]
        if st.session_state.history else "None"
    )

    st.markdown("### How to Use")
    st.write("""
    Step 1: Upload clear skin image  
    Step 2: Click Analyze  
    Step 3: View results  
    Step 4: Download report
    """)

    st.button("Start Scan")

# ---------------- SCANNER ----------------
elif page == "Scan":

    st.header("🔬 Skin Scanner")

    uploaded = st.file_uploader("Upload Skin Image")

    if uploaded:
        image = Image.open(uploaded)
        st.image(image, width=300)

        if st.button("Analyze"):

            with st.spinner("Analyzing Skin Lesion..."):

                model = load_ai_model()
                img = preprocess_image(image)

                if img is None:
                    st.error("Invalid or low-quality image")
                else:

                    result, confidence, prob = predict(model, img)

                    color = "red" if result == "Malignant" else "green"

                    st.markdown(
                        f"## Diagnosis: :{color}[{result}]"
                    )

                    st.metric("Confidence", f"{confidence:.2f}%")

                    df = pd.DataFrame({
                        "Class": ["Benign", "Malignant"],
                        "Probability": [1-prob, prob]
                    })

                    fig = px.bar(df, x="Class", y="Probability")
                    st.plotly_chart(fig)

                    risk = (
                        "High" if confidence > 80
                        else "Medium" if confidence > 60
                        else "Low"
                    )

                    st.success(f"Risk Level: {risk}")

                    st.info("""
                    ✔ Keep area clean  
                    ✔ Avoid sun exposure  
                    ✔ Consult dermatologist if suspicious
                    """)

                    report = generate_report(result, confidence)

                    st.download_button(
                        "Download Report",
                        report,
                        "report.txt"
                    )

                    st.session_state.history.append({
                        "result": result,
                        "confidence": confidence
                    })

# ---------------- REPORTS ----------------
elif page == "Reports":

    st.header("📄 Previous Scans")

    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df)

        csv = df.to_csv(index=False)
        st.download_button("Export CSV", csv)

# ---------------- ANALYTICS ----------------
elif page == "Analytics":

    st.header("📊 Analytics Dashboard")

    if st.session_state.history:

        df = pd.DataFrame(st.session_state.history)

        fig1 = px.pie(df, names="result",
                      title="Benign vs Malignant")
        st.plotly_chart(fig1)

        fig2 = px.histogram(
            df,
            x="confidence",
            title="Confidence Distribution"
        )
        st.plotly_chart(fig2)

# ---------------- FOOTER ----------------
st.markdown("""
---
University of Agricultural Faisalabad  
Rehan Shafique  
rehanshafiq6540@gmail.com
""")
