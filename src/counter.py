import cv2 as cv
import numpy as np


MIN_AREA = 1000
MAX_SINGLE_GRAIN_AREA = 11000
AVERAGE_GRAIN_AREA = 6500


def detect_objects(
    binary: np.ndarray,
) -> list[np.ndarray]:

    contours, _ = cv.findContours(
        binary,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE,
    )

    detected = []

    for contour in contours:

        area = cv.contourArea(contour)

        if area < MIN_AREA:
            continue

        if area <= MAX_SINGLE_GRAIN_AREA:
            detected.append(contour)

        else:

            estimated_count = max(
                1,
                round(
                    area / AVERAGE_GRAIN_AREA
                ),
            )

            detected.extend(
                [contour] * estimated_count
            )

    detected.sort(
        key=lambda contour: (
            cv.boundingRect(contour)[1],
            cv.boundingRect(contour)[0],
        )
    )

    return detected


def draw_detected_objects(
    image: np.ndarray,
    contours: list[np.ndarray],
) -> np.ndarray:

    result = image.copy()

    for index, contour in enumerate(
        contours,
        start=1,
    ):

        x, y, width, height = cv.boundingRect(
            contour
        )

        cv.rectangle(
            result,
            (x, y),
            (x + width, y + height),
            (0, 255, 0),
            2,
        )

        cv.putText(
            result,
            str(index),
            (
                x,
                max(
                    y - 8,
                    20,
                ),
            ),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv.LINE_AA,
        )

    cv.putText(
        result,
        f"Count: {len(contours)}",
        (20, 40),
        cv.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2,
        cv.LINE_AA,
    )

    return result


def count_grains(
    image: np.ndarray,
    binary: np.ndarray,
) -> int:

    contours = detect_objects(
        binary
    )

    return len(contours)


def get_detected_grains(
    image: np.ndarray,
    binary: np.ndarray,
) -> tuple[
    int,
    list[np.ndarray],
    np.ndarray,
]:

    contours = detect_objects(
        binary
    )

    result = draw_detected_objects(
        image,
        contours,
    )

    return (
        len(contours),
        contours,
        result,
    )