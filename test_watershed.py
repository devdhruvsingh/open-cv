from src.preprocessing import load_image
from src.counter import watershed_count


NORMAL_IMAGE = "data/generated/rice_20_1.png"
TOUCHING_IMAGE = "data/generated/touching_20_1.png"


def main():

    normal_image = load_image(
        NORMAL_IMAGE
    )

    touching_image = load_image(
        TOUCHING_IMAGE
    )

    thresholds = [
        0.630,
        0.635,
        0.640,
        0.645,
        0.650,
        0.655,
        0.660,
    ]

    print()
    print("Watershed threshold test")
    print("-" * 60)
    print(
        "Threshold | Normal | Touching"
    )
    print("-" * 60)

    for threshold in thresholds:

        normal_count, _ = watershed_count(
            normal_image,
            foreground_threshold=threshold,
        )

        touching_count, _ = watershed_count(
            touching_image,
            foreground_threshold=threshold,
        )

        print(
            f"{threshold:9.3f} |"
            f"{normal_count:8d} |"
            f"{touching_count:9d}"
        )


if __name__ == "__main__":
    main()