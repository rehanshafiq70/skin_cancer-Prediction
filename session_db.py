import streamlit as st

def save(record):

    if "records" not in st.session_state:
        st.session_state.records=[]

    st.session_state.records.append(record)
