import cv2 as cv
import numpy as np

def main():
    camera = cv.VideoCapture(1, cv.CAP_AVFOUNDATION)

    if not camera.isOpened():
        raise RuntimeError("Could not open camera")

    while True:
        success, frame = camera.read()

        if not success:
            break

        frame = cv.resize(frame, (960, 540))

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(gray, (7, 7), 0)

        _, binary = cv.threshold(
            blur,
            0,
            255,
            cv.THRESH_BINARY_INV + cv.THRESH_OTSU
        )

        kernel = np.ones((3, 3), np.uint8)

        binary = cv.morphologyEx(
            binary,
            cv.MORPH_OPEN,
            kernel,
            iterations=1
        )

        distance = cv.distanceTransform(
            binary,
            cv.DIST_L2,
            5
        )

        _, sure_foreground = cv.threshold(
            distance,
            0.35 * distance.max(),
            255,
            cv.THRESH_BINARY
        )

        sure_foreground = np.uint8(
            sure_foreground
        )

        background = cv.dilate(
            binary,
            kernel,
            iterations=2
        )

        unknown = cv.subtract(
            background,
            sure_foreground
        )

        _, markers = cv.connectedComponents(
            sure_foreground
        )

        markers = markers + 1
        markers[unknown == 255] = 0

        watershed_image = frame.copy()

        markers = cv.watershed(
            watershed_image,
            markers
        )

        result = frame.copy()
        count = 0

        for label in np.unique(markers):

            if label <= 1:
                continue

            mask = np.uint8(
                markers == label
            ) * 255

            area = cv.countNonZero(mask)

            if area < 500:
                continue

            count += 1

            contours, _ = cv.findContours(
                mask,
                cv.RETR_EXTERNAL,
                cv.CHAIN_APPROX_SIMPLE
            )

            if contours:
                cv.drawContours(
                    result,
                    contours,
                    -1,
                    (0, 255, 0),
                    2
                )

        cv.putText(
            result,
            f"Rice Grains: {count}",
            (25, 45),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        binary_display = cv.cvtColor(
            binary,
            cv.COLOR_GRAY2BGR
        )

        foreground_display = cv.cvtColor(
            sure_foreground,
            cv.COLOR_GRAY2BGR
        )

        top = np.hstack(
            (result, binary_display)
        )

        bottom = np.hstack(
            (foreground_display, foreground_display)
        )

        combined = np.vstack(
            (top, bottom)
        )

        cv.imshow(
            "Live Rice Detection",
            combined
        )

        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()