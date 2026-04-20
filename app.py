import streamlit as st

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="SkinScan AI",
    page_icon="🧬",
    layout="wide"
)

# ==============================
# IMPORT MODULES
# ==============================
from assets.theme import load_theme
from ui.home_dashboard import show_home
from ui.scanner_ui import show_scanner
from ui.analytics_ui import show_analytics
from ui.results_ui import show_results
from ui.doctor_panel import show_doctor

# ==============================
# LOAD MEDICAL THEME
# ==============================
st.markdown(load_theme(), unsafe_allow_html=True)

# ==============================
# SESSION STATE INIT
# ==============================
if "records" not in st.session_state:
    st.session_state.records = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "current_page" not in st.session_state:
    st.session_state.current_page = "Home"

# ==============================
# SIDEBAR NAVIGATION
# ==============================
with st.sidebar:

    st.title("🧬 SkinScan AI")

    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "Home",
            "Scan",
            "Results",
            "Analytics",
            "Doctor Panel"
        ],
        index=[
            "Home",
            "Scan",
            "Results",
            "Analytics",
            "Doctor Panel"
        ].index(st.session_state.current_page)
    )

    st.session_state.current_page = page

    st.markdown("---")

    st.info(
        """
        **AI Skin Cancer Detection**

        Upload dermoscopy image  
        Run deep learning model  
        View medical report
        """
    )

# ==============================
# ROUTING CONTROLLER
# ==============================
try:

    if page == "Home":
        show_home()

    elif page == "Scan":
        show_scanner()

    elif page == "Results":
        show_results()

    elif page == "Analytics":
        show_analytics()

    elif page == "Doctor Panel":
        show_doctor()

# ==============================
# GLOBAL ERROR HANDLER
# ==============================
except Exception as e:

    st.error("System Error Occurred")

    with st.expander("Technical Details"):
        st.exception(e)

# ==============================
# FOOTER
# ==============================
st.markdown(
    """
---
**SkinScan AI — Clinical Decision Support System**

University of Agriculture Faisalabad  
Rehan Shafique  
rehanshafiq6540@gmail.com
""",
    unsafe_allow_html=True
)
