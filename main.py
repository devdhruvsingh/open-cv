import cv2 as cv

from src.preprocessing import (
    load_image,
    preprocessing_image,
)

from src.counter import count_grains


IMAGE_PATH = "data/generated/rice_20_1.png"


def main():

    image = load_image(
        IMAGE_PATH
    )

    binary = preprocessing_image(
        image
    )

    count = count_grains(
        image,
        binary,
    )

    print(
        f"Detected grains: {count}"
    )

    cv.imshow(
        "Original",
        image,
    )

    cv.imshow(
        "Binary",
        binary,
    )

    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()