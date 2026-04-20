import streamlit as st
from ui.home_dashboard import show_home
from ui.scanner_ui import show_scanner
from ui.analytics_ui import show_analytics
from ui.doctor_panel import show_doctor

st.sidebar.title("🧬 SkinScan AI")

page=st.sidebar.radio(
    "Navigation",
    ["Home","Scan","Analytics","Doctor Panel"]
)

if page=="Home":
    show_home()

elif page=="Scan":
    show_scanner()

elif page=="Analytics":
    show_analytics()

elif page=="Doctor Panel":
    show_doctor()
