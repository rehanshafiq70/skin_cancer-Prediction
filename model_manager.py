import os
import gdown
import streamlit as st
from tensorflow.keras.models import load_model

MODEL="skin_cancer_model.h5"
FILE_ID="18VE_D81425cZVYwAXjOn0gWti8_lZSML"

@st.cache_resource
def get_model():

    if not os.path.exists(MODEL):
        url=f"https://drive.google.com/uc?export=download&id={FILE_ID}"
        gdown.download(url,MODEL)

    return load_model(MODEL)
