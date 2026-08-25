import cv2 as cv
import numpy as np


def detect_objects(binary_image: np.ndarray) -> list:
    """Detect objects in a binary image."""

    contours, _ = cv.findContours(
        binary_image,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE,
    )

    return contours


def draw_detected_objects(
    image: np.ndarray,
    contours: list,
) -> np.ndarray:

    output = image.copy()

    for index, contour in enumerate(contours):

        area = cv.contourArea(contour)

        print(
            f"Contour {index + 1}: "
            f"area = {area:.2f}"
        )

        cv.drawContours(
            output,
            [contour],
            -1,
            (0, 255, 0),
            2,
        )

        moments = cv.moments(contour)

        if moments["m00"] != 0:

            center_x = int(
                moments["m10"] / moments["m00"]
            )

            center_y = int(
                moments["m01"] / moments["m00"]
            )

            cv.putText(
                output,
                str(index + 1),
                (center_x, center_y),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

    return output