from pathlib import Path
import random

import cv2 as cv
import numpy as np

from src.preprocessing import (
    load_image,
    preprocessing_image,
    extract_grain,
)


# Project directories
DATASET_DIR = Path("Rice_Image_Dataset")
OUTPUT_DIR = Path("data/generated")


def prepare_grain(image_path: Path) -> np.ndarray | None:

    image = load_image(str(image_path))

    binary = preprocessing_image(image)

    grain = extract_grain(
        image,
        binary,
    )

    return grain


def create_canvas(
    width: int = 1000,
    height: int = 1000,
) -> np.ndarray:

    return np.ones(
        (height, width, 3),
        dtype=np.uint8,
    ) * 255


def resize_grain(
    grain: np.ndarray,
    max_size: int = 100,
) -> np.ndarray:

    height, width = grain.shape[:2]

    scale = max_size / max(
        height,
        width,
    )

    new_width = max(
        1,
        int(width * scale),
    )

    new_height = max(
        1,
        int(height * scale),
    )

    return cv.resize(
        grain,
        (new_width, new_height),
        interpolation=cv.INTER_AREA,
    )


def generate_image(
    grain_count: int,
    variety: str = "Arborio",
    image_number: int = 1,
) -> Path:
   

    # Dataset variety directory
    variety_dir = DATASET_DIR / variety

    if not variety_dir.exists():
        raise FileNotFoundError(
            f"Rice variety not found: {variety_dir}"
        )

    # Get dataset images
    image_paths = list(
        variety_dir.glob("*.jpg")
    )

    if len(image_paths) < grain_count:
        raise ValueError(
            f"Not enough images available. "
            f"Required: {grain_count}, "
            f"Available: {len(image_paths)}"
        )

    # Randomly select individual rice images
    selected_images = random.sample(
        image_paths,
        grain_count,
    )

    # Create canvas
    canvas = create_canvas()

    canvas_height, canvas_width = canvas.shape[:2]

    # Grid configuration
    columns = 5
    rows = int(np.ceil(grain_count / columns))

    cell_width = canvas_width // columns
    cell_height = canvas_height // rows

    # Place each grain
    for index, image_path in enumerate(
        selected_images
    ):

        # Extract grain
        grain = prepare_grain(
            image_path
        )

        if grain is None:
            print(
                f"Warning: Could not extract "
                f"grain from {image_path}"
            )
            continue

        # Resize grain
        grain = resize_grain(
            grain
        )

        grain_height, grain_width = (
            grain.shape[:2]
        )

        # Determine grid position
        column = index % columns
        row = index // columns

        # Calculate available space
        cell_x = column * cell_width
        cell_y = row * cell_height

        # Center grain inside grid cell
        x = cell_x + (
            cell_width - grain_width
        ) // 2

        y = cell_y + (
            cell_height - grain_height
        ) // 2

        # Safety check
        if (
            x < 0
            or y < 0
            or x + grain_width > canvas_width
            or y + grain_height > canvas_height
        ):
            print(
                f"Warning: Grain {index + 1} "
                f"does not fit on canvas."
            )
            continue

        # Place grain
        canvas[
            y:y + grain_height,
            x:x + grain_width,
        ] = grain

    # Create output directory
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Output path
    output_path = (
        OUTPUT_DIR
        / f"rice_{grain_count}_{image_number}.png"
    )

    # Save image
    success = cv.imwrite(
        str(output_path),
        canvas,
    )

    if not success:
        raise IOError(
            f"Failed to save image: "
            f"{output_path}"
        )

    return output_path


if __name__ == "__main__":

    output = generate_image(
        grain_count=20,
        variety="Arborio",
        image_number=1,
    )

    print(
        f"Generated image: {output}"
    )