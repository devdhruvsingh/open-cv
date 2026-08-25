import cv2 as cv

from src.preprocessing import (
    load_image,
    preprocessing_image,
)

from src.counter import count_grains


TEST_CASES = [
    ("data/generated/rice_5_1.png", 5),
    ("data/generated/rice_10_1.png", 10),
    ("data/generated/rice_20_1.png", 20),
    ("data/generated/rice_30_1.png", 30),
    ("data/generated/rice_50_1.png", 50),
]


def main():

    print()
    print("Rice Grain Counter Test")
    print("-" * 50)

    correct = 0

    for image_path, expected in TEST_CASES:

        image = load_image(
            image_path
        )

        binary = preprocessing_image(
            image
        )

        detected = count_grains(
            image,
            binary,
        )

        status = (
            "PASS"
            if detected == expected
            else "FAIL"
        )

        if detected == expected:
            correct += 1

        print(
            f"Expected: {expected:2d} | "
            f"Detected: {detected:2d} | "
            f"{status} | "
            f"{image_path}"
        )

    print("-" * 50)

    accuracy = (
        correct / len(TEST_CASES)
    ) * 100

    print(
        f"Test accuracy: {accuracy:.1f}%"
    )


if __name__ == "__main__":
    main()
