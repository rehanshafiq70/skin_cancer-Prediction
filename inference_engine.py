def run_prediction(model,img):

    prob=model.predict(img)[0][0]

    label="Malignant" if prob>0.5 else "Benign"
    confidence=prob*100 if label=="Malignant" else (1-prob)*100

    return label,confidence,prob
