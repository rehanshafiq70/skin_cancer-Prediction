import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image

from model_loader import get_model
from utils import preprocess, predict

st.set_page_config(
    page_title="SkinScan AI",
    layout="wide"
)

# ---------- SESSION ----------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------- SIDEBAR ----------
st.sidebar.title("🧬 SkinScan AI")

page = st.sidebar.radio(
    "Navigation",
    ["Home","Scan","Reports","Analytics"]
)

# ---------- HOME ----------
if page == "Home":

    st.title("AI Skin Cancer Detection")

    c1,c2,c3 = st.columns(3)

    c1.metric("Total Scans",
              len(st.session_state.history))

    c2.metric("Accuracy","92%")

    last = (
        st.session_state.history[-1]["result"]
        if st.session_state.history else "None"
    )

    c3.metric("Last Result",last)

    st.info("""
Step 1 → Upload Image  
Step 2 → Click Analyze  
Step 3 → View Result  
Step 4 → Download Report
""")

# ---------- SCAN ----------
elif page == "Scan":

    st.header("Skin Scanner")

    file = st.file_uploader("Upload Image")

    if file:

        img = Image.open(file)
        st.image(img,width=300)

        if st.button("Analyze"):

            with st.spinner("AI analyzing..."):

                model = get_model()

                processed = preprocess(img)

                if processed is None:
                    st.error("Invalid or low-quality image")
                    st.stop()

                result,conf,prob = predict(model,processed)

                color = "red" if result=="Malignant" else "green"

                st.markdown(
                    f"## Diagnosis: :{color}[{result}]"
                )

                st.metric("Confidence",
                          f"{conf:.2f}%")

                df = pd.DataFrame({
                    "Class":["Benign","Malignant"],
                    "Probability":[1-prob,prob]
                })

                fig = px.bar(df,
                             x="Class",
                             y="Probability")

                st.plotly_chart(fig,
                                use_container_width=True)

                st.session_state.history.append({
                    "result":result,
                    "confidence":conf
                })

# ---------- REPORTS ----------
elif page == "Reports":

    st.header("Previous Scans")

    if st.session_state.history:

        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df)

        csv = df.to_csv(index=False)

        st.download_button(
            "Download CSV",
            csv,
            "records.csv"
        )

# ---------- ANALYTICS ----------
elif page == "Analytics":

    if st.session_state.history:

        df = pd.DataFrame(st.session_state.history)

        st.plotly_chart(
            px.pie(df,names="result",
                   title="Cancer Ratio")
        )

        st.plotly_chart(
            px.histogram(df,
                         x="confidence",
                         title="Confidence Distribution")
        )

# ---------- FOOTER ----------
st.markdown("""
---
University of Agriculture Faisalabad  
Rehan Shafique  
rehanshafiq6540@gmail.com
""")
