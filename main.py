import argparse
import concurrent.futures
import itertools
import os
import multiprocessing
from functools import partial
import shutil
import time
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from pytorch_msssim import SSIM

from utils.image_processor import divide_image_into_units


def save_different_pairs(
    different_pairs: List[Tuple[int, int, Tuple[int, int], Tuple[int, int], float]],
    units: List[np.ndarray],
    output_dir: str = "output",
) -> None:
    """
    Saves the different image unit pairs to the specified directory using multiple threads.

    Args:
        different_pairs (list): A list of tuples, where each tuple contains
                                (index1, index2, pos1, pos2, score).
        units (list): The list of all image units (numpy arrays).
        output_dir (str): The directory to save the output pairs.
    """
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    print(
        f"\nSaving {len(different_pairs)} different pairs to '{output_dir}' directory..."
    )

    def _save_single_pair(args):
        i, (idx1, idx2, _, _, _) = args
        pair_dir = os.path.join(output_dir, f"pair_{i+1}")
        os.makedirs(pair_dir, exist_ok=True)
        Image.fromarray(units[idx1]).save(os.path.join(pair_dir, "img1.png"))
        Image.fromarray(units[idx2]).save(os.path.join(pair_dir, "img2.png"))

    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(_save_single_pair, enumerate(different_pairs))

    print("Done saving.")


def _compare_pairs_worker(
    indices_chunk: List[Tuple[int, int]],
    units_tensor: torch.Tensor,
    positions: List[Tuple[int, int]],
    threshold: float,
    batch_size: int = 256,
) -> List[Tuple[int, int, Tuple[int, int], Tuple[int, int], float]]:
    """Worker function for CPU-based SSIM comparison."""
    different_pairs_chunk = []
    ssim_module = SSIM(
        data_range=255.0, size_average=False, channel=3, nonnegative_ssim=True
    )

    for i in range(0, len(indices_chunk), batch_size):
        batch_indices = indices_chunk[i : i + batch_size]
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
            original_pair_index_in_chunk = i + idx_in_batch.item()
            pair_indices = indices_chunk[original_pair_index_in_chunk]
            pos1 = positions[pair_indices[0]]
            pos2 = positions[pair_indices[1]]
            score = ssim_scores[idx_in_batch].item()
            different_pairs_chunk.append(
                (pair_indices[0], pair_indices[1], pos1, pos2, score)
            )
    return different_pairs_chunk


def find_different_units_gpu(
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

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        print("Error: No GPU (CUDA or MPS) available.")
        return

    print(f"Using device: {device}")

    print(f"Divided image into {len(image_units)} units.")
    print(f"Comparing units with SSIM threshold: {threshold}")

    num_comparisons = len(image_units) * (len(image_units) - 1) // 2
    print(f"Total comparision to be done: {num_comparisons}")

    print("This may take a while...")

    start_time = time.time()
    positions = [pos for pos, unit in image_units]
    units = [unit for pos, unit in image_units]

    units_tensor = torch.from_numpy(np.array(units)).float().to(device)
    # (N, H, W, C) -> (N, C, H, W)
    units_tensor = units_tensor.permute(0, 3, 1, 2)

    indices = list(itertools.combinations(range(len(image_units)), 2))

    different_pairs = []
    batch_size = 1024  # Adjustable batch size

    ssim_module = SSIM(
        data_range=255.0, size_average=False, channel=3, nonnegative_ssim=True
    )
    for i in range(0, len(indices), batch_size):
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
            different_pairs.append(
                (pair_indices[0], pair_indices[1], pos1, pos2, score)
            )

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total comparision to be done: {num_comparisons}")
    print(f"Total comparison time: {elapsed_time:.2f} seconds.")

    if num_comparisons > 0 and elapsed_time > 0:
        print(f"Comparsion per second: {num_comparisons/elapsed_time:.2f}.")

    if not different_pairs:
        print("No significant differences found between any units.")
    else:
        print(f"\nFound {len(different_pairs)} pairs with SSIM below {threshold}:")
        save_different_pairs(different_pairs, units)


def find_different_units_cpu(
    image_path: str, threshold: float = 0.9, unit_size: Tuple[int, int] = (512, 512)
) -> None:
    """
    Finds and reports image units that are different based on SSIM using CPU
    with multiprocessing.

    Args:
        image_path (str): Path to the image.
        threshold (float): SSIM threshold. Units with SSIM below this are
                         considered different.
        unit_size (tuple): The size of the image units to compare.
    """
    image_units = divide_image_into_units(image_path, unit_size)

    if not image_units:
        return

    print("Using device: CPU")

    print(f"Divided image into {len(image_units)} units.")
    print(f"Comparing units with SSIM threshold: {threshold}")

    num_comparisons = len(image_units) * (len(image_units) - 1) // 2
    print(f"Total comparision to be done: {num_comparisons}")

    print("This may take a while...")

    start_time = time.time()
    positions = [pos for pos, unit in image_units]
    units = [unit for pos, unit in image_units]

    units_tensor = torch.from_numpy(np.array(units)).float()
    # (N, H, W, C) -> (N, C, H, W)
    units_tensor = units_tensor.permute(0, 3, 1, 2)

    indices = list(itertools.combinations(range(len(image_units)), 2))

    num_processes = multiprocessing.cpu_count()
    print(f"Using {num_processes} CPU cores for parallel processing.")

    chunk_size = (len(indices) + num_processes - 1) // num_processes
    chunks = [indices[i : i + chunk_size] for i in range(0, len(indices), chunk_size)]

    with multiprocessing.Pool(processes=num_processes) as pool:
        worker = partial(
            _compare_pairs_worker,
            units_tensor=units_tensor,
            positions=positions,
            threshold=threshold,
        )
        results = pool.map(worker, chunks)

    different_pairs = [pair for sublist in results for pair in sublist]

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total comparision to be done: {num_comparisons}")
    print(f"Total comparison time: {elapsed_time:.2f} seconds.")

    if num_comparisons > 0 and elapsed_time > 0:
        print(f"Comparsion per second: {num_comparisons/elapsed_time:.2f}.")

    if not different_pairs:
        print("No significant differences found between any units.")
    else:
        print(f"\nFound {len(different_pairs)} pairs with SSIM below {threshold}:")
        save_different_pairs(different_pairs, units)


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
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="gpu",
        choices=["gpu", "cpu"],
        help="Device to use for computation ('gpu' or 'cpu'). Default is 'gpu'.",
    )
    args = parser.parse_args()

    if not (0.0 <= args.threshold <= 1.0):
        print("Error: Threshold must be between 0.0 and 1.0.")
        return

    if args.device == "gpu":
        find_different_units_gpu(args.image, args.threshold)
    else:
        find_different_units_cpu(args.image, args.threshold)


if __name__ == "__main__":
    main()
