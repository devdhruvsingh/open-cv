import cv2 as cv
import numpy as np


def detect_objects(binary_image: np.ndarray) -> list:
    contours, _ = cv.findContours(
        binary_image,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE,
    )
    return contours


def analyze_contours(
    contours: list,
    area_threshold: float = 10000,
) -> tuple[list, list]:
    normal_contours = []
    suspicious_contours = []

    for contour in contours:
        area = cv.contourArea(contour)

        if area > area_threshold:
            suspicious_contours.append(contour)
        else:
            normal_contours.append(contour)

    return normal_contours, suspicious_contours


def calculate_average_grain_area(
    normal_contours: list,
) -> float:
    if not normal_contours:
        return 0.0

    areas = [
        cv.contourArea(contour)
        for contour in normal_contours
    ]

    return float(np.mean(areas))


def estimate_grains_from_area(
    contour: np.ndarray,
    average_grain_area: float,
) -> int:
    if average_grain_area <= 0:
        return 1

    area = cv.contourArea(contour)

    estimated_count = round(
        area / average_grain_area
    )

    return max(estimated_count, 1)


def count_grains(
    image: np.ndarray,
    binary_image: np.ndarray,
    area_threshold: float = 10000,
) -> int:

    contours = detect_objects(binary_image)

    normal_contours, suspicious_contours = (
        analyze_contours(
            contours,
            area_threshold,
        )
    )

    normal_count = len(normal_contours)

    average_grain_area = (
        calculate_average_grain_area(
            normal_contours
        )
    )

    suspicious_count = sum(
        estimate_grains_from_area(
            contour,
            average_grain_area,
        )
        for contour in suspicious_contours
    )

    return normal_count + suspicious_count


def draw_detected_objects(
    image: np.ndarray,
    contours: list,
) -> np.ndarray:

    output = image.copy()

    for index, contour in enumerate(contours):

        cv.drawContours(
            output,
            [contour],
            -1,
            (0, 255, 0),
            2,
        )

        moments = cv.moments(contour)

        if moments["m00"] == 0:
            continue

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