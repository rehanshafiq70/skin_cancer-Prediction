import numpy as np

def prepare(image):

    image=image.resize((224,224))
    arr=np.array(image)/255.0

    if arr.std()<0.02:
        return None

    return np.expand_dims(arr,0)
