from pathlib import Path

from src.preprocessing import (
    load_image,
    preprocessing_image,
)

from src.counter import count_grains


def main():

    image_paths = sorted(
        Path("data/generated").glob("*.png")
    )

    total = 0
    correct = 0

    print()
    print("Rice Grain Counter Test")
    print("-" * 70)

    for image_path in image_paths:

        parts = image_path.stem.split("_")

        expected = int(parts[1])

        image = load_image(
            str(image_path)
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

        total += 1

        print(
            f"Expected: {expected:2d} | "
            f"Detected: {detected:2d} | "
            f"{status} | "
            f"{image_path}"
        )

    accuracy = (
        correct / total * 100
        if total
        else 0
    )

    print("-" * 70)
    print(
        f"Passed: {correct}/{total}"
    )
    print(
        f"Test accuracy: {accuracy:.1f}%"
    )


if __name__ == "__main__":
    main()