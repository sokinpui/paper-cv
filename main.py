import argparse
import itertools
import time
from typing import Tuple

import numpy as np
import torch
from pytorch_msssim import SSIM

from utils.image_processor import divide_image_into_units


def find_different_units(
    image_path: str, threshold: float = 0.9, unit_size: Tuple[int, int] = (512, 512)
) -> None:
    """
    Finds and reports image units that are different based on SSIM using GPU.

    Args:
        image_path (str): Path to the image.
        threshold (float): SSIM threshold. Units with SSIM below this are
                         considered different.
        unit_size (tuple): The size of the image units to compare.
    """
    image_units = divide_image_into_units(image_path, unit_size)

    if not image_units:
        return

    device = torch.device("mps")
    print(f"Using device: {device}")

    print(f"Divided image into {len(image_units)} units.")
    print(f"Comparing units with SSIM threshold: {threshold}")

    num_comparisons = len(image_units) * (len(image_units) - 1) // 2
    print(f"Total comparision to be done: {num_comparisons}")

    print("This may take a while...")

    positions = [pos for pos, unit in image_units]
    units = [unit for pos, unit in image_units]

    units_tensor = torch.from_numpy(np.array(units)).float().to(device)
    units_tensor = units_tensor.unsqueeze(1)  # Add channel dimension: (N, 1, H, W)

    indices = list(itertools.combinations(range(len(image_units)), 2))

    different_pairs = []
    batch_size = 1024  # Adjustable batch size

    start_time = time.time()
    ssim_module = SSIM(
        data_range=255.0, size_average=False, channel=1, nonnegative_ssim=True
    )
    for i in range(0, num_comparisons, batch_size):
        batch_indices = indices[i : i + batch_size]
        if not batch_indices:
            break

        indices1 = [idx[0] for idx in batch_indices]
        indices2 = [idx[1] for idx in batch_indices]

        tensor1 = units_tensor[indices1]
        tensor2 = units_tensor[indices2]

        ssim_scores = ssim_module(tensor1, tensor2)

        below_threshold_mask = ssim_scores < threshold
        below_threshold_indices_in_batch = torch.where(below_threshold_mask)[0]

        for idx_in_batch in below_threshold_indices_in_batch:
            original_pair_index = i + idx_in_batch.item()
            pair_indices = indices[original_pair_index]
            pos1 = positions[pair_indices[0]]
            pos2 = positions[pair_indices[1]]
            score = ssim_scores[idx_in_batch].item()
            different_pairs.append((pos1, pos2, score))
            print(f"  - Unit at {pos1} and Unit at {pos2}, SSIM: {score:.4f}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total comparision to be done: {num_comparisons}")
    print(f"Total comparison time: {elapsed_time:.2f} seconds.")

    print(f"Comparsion per second: {num_comparisons/elapsed_time:.2f}.")

    if not different_pairs:
        print("No significant differences found between any units.")
    else:
        print(f"\nFound {len(different_pairs)} pairs with SSIM below {threshold}:")
        # for pos1, pos2, ssim_score in different_pairs:
        #     print(f"  - Unit at {pos1} and Unit at {pos2}, SSIM: {ssim_score:.4f}")


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
