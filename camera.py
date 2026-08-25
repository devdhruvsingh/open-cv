import cv2 as cv

def main():
    camera = cv.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError("Could not open camera")

    while True:
        success, frame = camera.read()

        if not success:
            break

        cv.imshow("Camera", frame)

        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
