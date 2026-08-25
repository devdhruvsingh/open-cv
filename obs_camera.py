import cv2 as cv

from src.preprocessing import preprocessing_image
from src.counter import detect_objects


def main():
    camera = cv.VideoCapture(1, cv.CAP_AVFOUNDATION)

    if not camera.isOpened():
        raise RuntimeError("Could not open OBS virtual camera")

    while True:
        success, frame = camera.read()

        if not success:
            print("Failed to read frame")
            break

        binary = preprocessing_image(frame)
        contours = detect_objects(binary)

        display = frame.copy()

        cv.putText(
            display,
            f"Rice Grains: {len(contours)}",
            (30, 50),
            cv.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3,
        )

        cv.imshow("Live Rice Counter", display)
        cv.imshow("Binary", binary)

        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()