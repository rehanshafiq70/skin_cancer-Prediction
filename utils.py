import numpy as np

def preprocess(img):

    img = img.resize((224,224))
    img = np.array(img)/255.0

    if img.std() < 0.02:
        return None

    img = np.expand_dims(img,0)
    return img


def predict(model,img):

    p = model.predict(img)[0][0]

    if p > 0.5:
        return "Malignant", p*100, p
    else:
        return "Benign", (1-p)*100, p
