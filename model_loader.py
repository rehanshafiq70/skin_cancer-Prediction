import os
import gdown
from tensorflow.keras.models import load_model

MODEL_PATH = "skin_cancer_model.h5"
FILE_ID = "18VE_D81425cZVYwAXjOn0gWti8_lZSML"

def load_ai_model():

    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?export=download&id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)

    model = load_model(MODEL_PATH)
    return model
