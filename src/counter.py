import cv2 as cv
import numpy as np

def count_objects(binary_image : np.ndarray)-> int:
    # count separate objects in binary image

    contours, _ = cv.findContours(
        binary_image,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE,

    )

    return len(contours)