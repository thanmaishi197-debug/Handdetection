import cv2
import numpy as np

def apply_filter(image, current_filter):
    if current_filter == "original":
        return image

    if current_filter == "grey":
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if current_filter == "blur":
        return cv2.GaussianBlur(image, (25, 25), 0)

    if current_filter == "sepia":
        kernel = np.array([
            [0.272, 0.534, 0.131],
            [0.349, 0.686, 0.168],
            [0.393, 0.769, 0.189]
        ])
        return cv2.transform(image, kernel)

    if current_filter == "edges":
        return cv2.Canny(image, 50, 150)

    return image
