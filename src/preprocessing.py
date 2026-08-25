import cv2 as cv
import numpy as np


def load_image(image_path: str) -> np.ndarray:
    # loading the image

    image = cv.imread(image_path)

    if image is None:
        raise FileNotFoundError(
            f"Image not loaded: {image_path}"
        )

    return image


def preprocessing_image(image: np.ndarray) -> np.ndarray:
    # image to binary

    # Convert to grayscale
    gray = cv.cvtColor(
        image,
        cv.COLOR_BGR2GRAY
    )

    # Reduce noise
    blurred = cv.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # Adaptive threshold
    binary = cv.adaptiveThreshold(
        blurred,
        255,
        cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv.THRESH_BINARY_INV,
        21,
        5,
    )

    return binary


def extract_grain(
        image : np.ndarray,
        binary : np.ndarray
,) -> np.ndarray | None:
    # extracting the largest image from the binary
    contours, _ = cv.findContours(
        binary, 
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE,


    )

    if not contours:
        return None

    largest_contours = max(
        contours,
        key = cv.contourArea,
    )

    # get rectangle
    x, y, width , height = cv.boundingRect(
        largest_contours

    )

    # crop the original iamge
    grain = image[
        y:y + height,
        x:x + width
    ]

    return grain
