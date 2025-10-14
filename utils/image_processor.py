from typing import Tuple, List
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim

ImageUnit = Tuple[Tuple[int, int], np.ndarray]

def divide_image_into_units(
    image_path: str, unit_size: Tuple[int, int] = (512, 512)
) -> List[ImageUnit]:
    """
    Divides an image into units of a specified size.

    Args:
        image_path (str): The path to the input image.
        unit_size (tuple): The (width, height) of the units.

    Returns:
        list: A list of tuples, where each tuple contains the coordinates
              (row, col) and the image unit as a numpy array.
    """
    try:
        image = Image.open(image_path)
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return []

    image = image.convert("L")
    width, height = image.size
    unit_width, unit_height = unit_size
    image_units = []

    for i in range(height // unit_height):
        for j in range(width // unit_width):
            left = j * unit_width
            top = i * unit_height
            right = left + unit_width
            bottom = top + unit_height
            box = (left, top, right, bottom)
            unit = image.crop(box)
            image_units.append(((i, j), np.array(unit)))

    return image_units


def compare_units(unit1: np.ndarray, unit2: np.ndarray) -> float:
    """
    Compares two image units using Structural Similarity Index (SSIM).

    Args:
        unit1 (np.array): The first image unit.
        unit2 (np.array): The second image unit.

    Returns:
        float: The SSIM score.
    """
    # For uint8 images, the data range is 255. Using a fixed data_range
    # helps stabilize SSIM calculation, especially for low-variance units.
    return ssim(unit1, unit2, data_range=255.0)
