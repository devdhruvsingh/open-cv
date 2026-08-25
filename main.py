import cv2 as cv
from src.preprocessing import load_image, preprocessing_image
from src.detection import find_contours, draw_contours


IMAGE_PATH = "/Users/dhruvsingh/Desktop/open-cv/Rice_Image_Dataset/Arborio/Arborio (1).jpg"

def main():

    image = load_image(IMAGE_PATH)
    binary = preprocessing_image(image)

    contours = find_contours(binary)

    result = draw_contours(image, contours)

    cv.imshow("Original", image)
    cv.imshow("Binary", binary)

    print(f"Contours detected : {len(contours)}")

    print(f"Image preprocessed successfully")
    
    cv.imshow("Detected contours ", result )

    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()