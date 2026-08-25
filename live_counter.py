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

        lab = cv.cvtColor(frame, cv.COLOR_BGR2LAB)

        l, a, b = cv.split(lab)

        clahe = cv.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        l = clahe.apply(l)

        enhanced = cv.cvtColor(
            cv.merge((l, a, b)),
            cv.COLOR_LAB2BGR
        )

        gray = cv.cvtColor(
            enhanced,
            cv.COLOR_BGR2GRAY
        )

        blur = cv.GaussianBlur(
            gray,
            (7, 7),
            0
        )

        binary = cv.adaptiveThreshold(
            blur,
            255,
            cv.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv.THRESH_BINARY_INV,
            31,
            7
        )

        kernel = np.ones(
            (3, 3),
            np.uint8
        )

        binary = cv.morphologyEx(
            binary,
            cv.MORPH_OPEN,
            kernel,
            iterations=1
        )

        binary = cv.morphologyEx(
            binary,
            cv.MORPH_CLOSE,
            kernel,
            iterations=2
        )

        distance = cv.distanceTransform(
            binary,
            cv.DIST_L2,
            5
        )

        _, foreground = cv.threshold(
            distance,
            0.28 * distance.max(),
            255,
            cv.THRESH_BINARY
        )

        foreground = np.uint8(
            foreground
        )

        background = cv.dilate(
            binary,
            kernel,
            iterations=3
        )

        unknown = cv.subtract(
            background,
            foreground
        )

        _, markers = cv.connectedComponents(
            foreground
        )

        markers = markers + 1
        markers[unknown == 255] = 0

        watershed_image = enhanced.copy()

        markers = cv.watershed(
            watershed_image,
            markers
        )

        result = enhanced.copy()

        count = 0

        for label in np.unique(markers):

            if label <= 1:
                continue

            mask = np.uint8(
                markers == label
            ) * 255

            area = cv.countNonZero(mask)

            if area < 700:
                continue

            contours, _ = cv.findContours(
                mask,
                cv.RETR_EXTERNAL,
                cv.CHAIN_APPROX_SIMPLE
            )

            if not contours:
                continue

            contour = max(
                contours,
                key=cv.contourArea
            )

            area = cv.contourArea(contour)

            if area < 700:
                continue

            x, y, w, h = cv.boundingRect(
                contour
            )

            if w < 20 or h < 20:
                continue

            count += 1

            cv.rectangle(
                result,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            cv.putText(
                result,
                str(count),
                (x, max(25, y - 8)),
                cv.FONT_HERSHEY_SIMPLEX,
                0.7,
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

        top = np.hstack(
            (result, binary_display)
        )

        cv.imshow(
            "Live Rice Counter",
            top
        )

        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()