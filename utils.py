import numpy as np

def preprocess_image(img):
    img = img.resize((224,224))
    arr = np.array(img)/255.0
    return np.expand_dims(arr,0)

def validate_image(img):
    arr=np.array(img)
    return arr.std()>5

def predict_skin_cancer(model,data):

    prob=model.predict(data)[0][0]

    label="Malignant" if prob>0.5 else "Benign"
    conf=prob*100 if label=="Malignant" else (1-prob)*100

    return label,conf,prob

def risk_level(conf):

    if conf>80:
        return "High"
    elif conf>60:
        return "Medium"
    return "Low"
