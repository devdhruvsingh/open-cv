import cv2 as cv

from src.preprocessing import (
    load_image,
    preprocessing_image,
)

from src.counter import (
    detect_objects,
    draw_detected_objects,
)


IMAGE_PATH = "data/generated/rice_20_1.png"


def main():

    image = load_image(IMAGE_PATH)

    binary = preprocessing_image(image)

    contours = detect_objects(binary)

    result = draw_detected_objects(
        image,
        contours,
    )

    print(f"Detected contours: {len(contours)}")

    cv.imshow("Original", image)
    cv.imshow("Binary", binary)
    cv.imshow("Detected Objects", result)

    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()