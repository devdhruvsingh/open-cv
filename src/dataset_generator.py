from pathlib import Path
import random

import cv2
import numpy as np


DATASET_DIR = Path("Rice_Image_Dataset")
OUTPUT_DIR = Path("data/generated")


def load_grain(image_path: Path) -> np.ndarray | None:
    """Load a single rice grain image."""
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Warning: Could not load {image_path}")
        return None

    return image


def create_canvas(width: int = 1000, height: int = 1000) -> np.ndarray:
    """Create a white background canvas."""
    return np.ones((height, width, 3), dtype=np.uint8) * 255


def resize_grain(
    grain: np.ndarray,
    max_size: int = 100,
) -> np.ndarray:

    height, width = grain.shape[:2]

    scale = max_size / max(height, width)

    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))

    return cv2.resize(
        grain,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )


def generate_image(
    grain_count: int,
    variety: str = "Arborio",
    image_number: int = 1,
) -> Path:

    variety_dir = DATASET_DIR / variety

    image_paths = list(variety_dir.glob("*.jpg"))

    if len(image_paths) < grain_count:
        raise ValueError(
            f"Not enough images available for {variety}. "
            f"Required: {grain_count}, available: {len(image_paths)}"
        )

    selected_images = random.sample(image_paths, grain_count)

    canvas = create_canvas()

    canvas_height, canvas_width = canvas.shape[:2]

    for image_path in selected_images:

        # Load the actual image.
        grain = load_grain(image_path)

        if grain is None:
            continue

        # Resize the image.
        grain = resize_grain(grain)

        grain_height, grain_width = grain.shape[:2]

        max_x = canvas_width - grain_width
        max_y = canvas_height - grain_height

        if max_x <= 0 or max_y <= 0:
            continue

        x = random.randint(0, max_x)
        y = random.randint(0, max_y)

        canvas[
            y:y + grain_height,
            x:x + grain_width
        ] = grain

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / f"rice_{grain_count}_{image_number}.png"

    cv2.imwrite(str(output_path), canvas)

    return output_path


if __name__ == "__main__":

    output = generate_image(
        grain_count=20,
        variety="Arborio",
        image_number=1,
    )

    print(f"Generated image: {output}")