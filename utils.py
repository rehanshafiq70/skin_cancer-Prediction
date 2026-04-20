import numpy as np
from PIL import Image

IMG_SIZE = 224

def preprocess_image(image):

    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image) / 255.0

    if image.std() < 0.02:
        return None

    image = np.expand_dims(image, axis=0)
    return image


def predict(model, img):

    pred = model.predict(img)[0][0]

    if pred > 0.5:
        label = "Malignant"
        confidence = pred * 100
    else:
        label = "Benign"
        confidence = (1 - pred) * 100

    return label, confidence, pred
