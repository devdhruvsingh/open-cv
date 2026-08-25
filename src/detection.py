import cv2 as cv
import numpy as np

def find_contours(binary_image : np.ndarray) -> list:
    # find object countours in binary way

    contours, _ = cv.findContours(
        binary_image,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE
    )

    return contours


def draw_contours(
        image : np.ndarray,
        contours : list,
) -> np.ndarray:
    output = image.copy()

    cv.drawContours(
        output,
        contours,
        -1, (0,255,0),
        2,
    )

    return output