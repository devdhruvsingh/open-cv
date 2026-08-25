import cv2 as cv
from src.preprocessing import load_image, preprocessing_image


IMAGE_PATH = "/Users/dhruvsingh/Desktop/open-cv/Rice_Image_Dataset/Arborio/Arborio (1).jpg"

def main():

    image = load_image(IMAGE_PATH)
    binary = preprocessing_image(image)

    cv.imshow("Original", image)
    cv.imshow("Binary", binary)

    print(f"Image preprocessed successfully")
    print(f"Original shape : {image.shape}")
    print(f"Binary shape : {binary.shape}")

    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()