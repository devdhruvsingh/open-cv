import cv2 as cv

from src.preprocessing import (
    load_image,
    preprocessing_image,
)

from src.counter import count_objects


IMAGE_PATH = "data/generated/rice_20_1.png"


def main():

    # Load generated image
    image = load_image(IMAGE_PATH)

    # Preprocess image
    binary = preprocessing_image(image)

    # Count objects
    count = count_objects(binary)

    print(f"Detected grains: {count}")

    # Display images
    cv.imshow("Image", image)
    cv.imshow("Binary", binary)

    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()