import argparse
import itertools
from typing import Tuple

from utils.image_processor import compare_units, divide_image_into_units


def find_different_units(
    image_path: str, threshold: float = 0.9, unit_size: Tuple[int, int] = (512, 512)
) -> None:
    """
    Finds and reports image units that are different based on SSIM.

    Args:
        image_path (str): Path to the image.
        threshold (float): SSIM threshold. Units with SSIM below this are
                         considered different.
        unit_size (tuple): The size of the image units to compare.
    """
    image_units = divide_image_into_units(image_path, unit_size)

    if not image_units:
        return

    print(f"Divided image into {len(image_units)} units.")

    different_pairs = []

    for (pos1, unit1), (pos2, unit2) in itertools.combinations(image_units, 2):
        ssim_score = compare_units(unit1, unit2)
        if ssim_score < threshold:
            pair_info = (pos1, pos2, ssim_score)
            different_pairs.append(pair_info)
            print(
                f"Difference found between unit at {pos1} and unit at {pos2}. "
                f"SSIM: {ssim_score:.4f}"
            )

    if not different_pairs:
        print("No significant differences found between any units.")


def main() -> None:
    """
    Main function to parse arguments and run the image comparison.
    """
    parser = argparse.ArgumentParser(
        description="Compare 512x512 units of an image using SSIM."
    )
    parser.add_argument("image", help="Path to the input image.")
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.9,
        help="SSIM threshold for detecting differences (0.0 to 1.0). Default is 0.9.",
    )
    args = parser.parse_args()

    if not (0.0 <= args.threshold <= 1.0):
        print("Error: Threshold must be between 0.0 and 1.0.")
        return

    find_different_units(args.image, args.threshold)


if __name__ == "__main__":
    main()
