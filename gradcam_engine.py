import cv2
import numpy as np

def generate_heatmap(img):
    heatmap=cv2.applyColorMap(
        np.uint8(img*255),
        cv2.COLORMAP_JET
    )
    return heatmap
