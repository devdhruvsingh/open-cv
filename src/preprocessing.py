import cv2 as cv
import numpy as np


def load_image(image_path: str) -> np.ndarray:
    """Load an image from disk."""

    image = cv.imread(image_path)

    if image is None:
        raise FileNotFoundError(
            f"Image not loaded: {image_path}"
        )

    return image


def preprocessing_image(image: np.ndarray) -> np.ndarray:
    """Convert the image into a binary image."""

    # Convert the image into grayscale
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Reduce small amounts of image noise
    blurred = cv.GaussianBlur(gray, (5, 5), 0)

    # Convert the image into binary
    _, binary = cv.threshold(
        blurred,
        0,
        255,
        cv.THRESH_BINARY_INV + cv.THRESH_OTSU
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

    mask = np.zeros_like(binary)

    cv.drawContours(
        mask,
        [largest_contours],
        -1,255, thickness=cv.FILLED
    )


    # extracting the grain
    grain = cv.bitwise_and(
        image, image, mask = mask,
    )

    return grain 