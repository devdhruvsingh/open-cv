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