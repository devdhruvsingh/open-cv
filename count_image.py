import argparse

import cv2 as cv

from src.preprocessing import (
    load_image,
    preprocessing_image,
)

from src.counter import (
    get_detected_grains,
)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        required=True,
    )

    args = parser.parse_args()

    image = load_image(
        args.image
    )

    binary = preprocessing_image(
        image
    )

    count, contours, result = get_detected_grains(
        image,
        binary,
    )

    print()
    print("Rice Grain Counter")
    print("-" * 30)
    print(f"Image: {args.image}")
    print(f"Detected grains: {count}")
    print()

    cv.imshow(
        "Original",
        image,
    )

    cv.imshow(
        "Binary",
        binary,
    )

    cv.imshow(
        "Detected Grains",
        result,
    )

    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()