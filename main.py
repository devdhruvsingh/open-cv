import cv2 as cv

from src.preprocessing import (
    load_image,
    preprocessing_image,
    extract_grain,
)


IMAGE_PATH = (
    "/Users/dhruvsingh/Desktop/open-cv/"
    "Rice_Image_Dataset/Arborio/Arborio (1).jpg"
)


def main():

    # Load image
    image = load_image(IMAGE_PATH)

    # Preprocess
    binary = preprocessing_image(image)

    # Extract grain
    grain = extract_grain(
        image,
        binary,
    )

    if grain is None:
        print("No grain detected.")
        return

    # Display results
    cv.imshow("Original", image)
    cv.imshow("Binary", binary)
    cv.imshow("Extracted Grain", grain)

    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()