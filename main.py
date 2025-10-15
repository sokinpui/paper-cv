import argparse
import concurrent.futures
import itertools
import multiprocessing
import os
import shutil
import time
import numba
from functools import partial
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from pytorch_msssim import SSIM
from tqdm import tqdm

from utils.image_processor import divide_image_into_units


@numba.jit(nopython=True)
def _collect_diff_pairs_from_batch_np(
    batch_indices_np, positions_np, ssim_scores_np, below_threshold_indices_np
):
    """
    Collects different pairs from a batch of comparisons using Numba for acceleration.
    This function is designed to be JIT-compiled by Numba in nopython mode.

    Args:
        batch_indices_np (np.ndarray): Numpy array of pair indices for the batch.
        positions_np (np.ndarray): Numpy array of all unit positions.
        ssim_scores_np (np.ndarray): Numpy array of SSIM scores for the batch.
        below_threshold_indices_np (np.ndarray): Numpy array of indices within the
                                                 batch that are below the threshold.

    Returns:
        np.ndarray: A numpy array where each row represents a different pair:
                    [idx1, idx2, pos1_row, pos1_col, pos2_row, pos2_col, score]
    """
    num_found = len(below_threshold_indices_np)
    # (idx1, idx2, pos1_row, pos1_col, pos2_row, pos2_col, score)
    results = np.empty((num_found, 7), dtype=np.float64)

    for i in range(num_found):
        idx_in_batch = below_threshold_indices_np[i]

        pair_indices = batch_indices_np[idx_in_batch]
        idx1 = pair_indices[0]
        idx2 = pair_indices[1]

        pos1 = positions_np[idx1]
        pos2 = positions_np[idx2]

        score = ssim_scores_np[idx_in_batch]

        results[i, 0] = idx1
        results[i, 1] = idx2
        results[i, 2] = pos1[0]
        results[i, 3] = pos1[1]
        results[i, 4] = pos2[0]
        results[i, 5] = pos2[1]
        results[i, 6] = score

    return results


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


def _compare_batch_worker(
    batch_indices: List[Tuple[int, int]],
    units_tensor: torch.Tensor,
    positions_np: np.ndarray,
    threshold: float,
) -> List[Tuple[int, int, Tuple[int, int], Tuple[int, int], float]]:
    """Worker function for CPU-based SSIM comparison on a batch of pairs."""
    ssim_module = SSIM(
        data_range=255.0, size_average=False, channel=3, nonnegative_ssim=True
    )

    if not batch_indices:
        return []

    batch_indices_np = np.array(batch_indices, dtype=np.int32)

    indices1 = batch_indices_np[:, 0].tolist()
    indices2 = batch_indices_np[:, 1].tolist()

    tensor1 = units_tensor[indices1]
    tensor2 = units_tensor[indices2]

    ssim_scores = ssim_module(tensor1, tensor2)

    below_threshold_mask = ssim_scores < threshold
    below_threshold_indices_in_batch_tensor = torch.where(below_threshold_mask)[0]

    different_pairs_batch = []
    if len(below_threshold_indices_in_batch_tensor) > 0:
        ssim_scores_np = ssim_scores.numpy()
        below_threshold_indices_np = below_threshold_indices_in_batch_tensor.numpy()

        results_np = _collect_diff_pairs_from_batch_np(
            batch_indices_np,
            positions_np,
            ssim_scores_np,
            below_threshold_indices_np,
        )
        for row in results_np:
            idx1, idx2, p1r, p1c, p2r, p2c, score = row
            different_pairs_batch.append(
                (int(idx1), int(idx2), (int(p1r), int(p1c)), (int(p2r), int(p2c)), score)
            )
    return different_pairs_batch


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
    positions_np = np.array(positions, dtype=np.int32)

    units_tensor = torch.from_numpy(np.array(units)).float().to(device)
    # (N, H, W, C) -> (N, C, H, W)
    units_tensor = units_tensor.permute(0, 3, 1, 2)

    indices = list(itertools.combinations(range(len(image_units)), 2))
    indices_np = np.array(indices, dtype=np.int32)

    different_pairs = []
    batch_size = 1024  # Adjustable batch size

    ssim_module = SSIM(
        data_range=255.0, size_average=False, channel=3, nonnegative_ssim=True
    )
    with tqdm(
        total=num_comparisons, desc="Processing", bar_format="{desc}: {n_fmt}/{total_fmt}"
    ) as pbar:
        for i in range(0, len(indices), batch_size):
            batch_indices_np_slice = indices_np[i : i + batch_size]
            if len(batch_indices_np_slice) == 0:
                break

            indices1 = batch_indices_np_slice[:, 0].tolist()
            indices2 = batch_indices_np_slice[:, 1].tolist()

            tensor1 = units_tensor[indices1]
            tensor2 = units_tensor[indices2]

            ssim_scores = ssim_module(tensor1, tensor2)

            below_threshold_mask = ssim_scores < threshold
            below_threshold_indices_in_batch_tensor = torch.where(below_threshold_mask)[
                0
            ]

            if len(below_threshold_indices_in_batch_tensor) > 0:
                ssim_scores_np = ssim_scores.cpu().numpy()
                below_threshold_indices_np = (
                    below_threshold_indices_in_batch_tensor.cpu().numpy()
                )

                results_np = _collect_diff_pairs_from_batch_np(
                    batch_indices_np_slice,
                    positions_np,
                    ssim_scores_np,
                    below_threshold_indices_np,
                )
                for row in results_np:
                    idx1, idx2, p1r, p1c, p2r, p2c, score = row
                    different_pairs.append(
                        (int(idx1), int(idx2), (int(p1r), int(p1c)), (int(p2r), int(p2c)), score)
                    )
            pbar.update(len(batch_indices_np_slice))

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
    positions_np = np.array(positions, dtype=np.int32)

    units_tensor = torch.from_numpy(np.array(units)).float()
    # (N, H, W, C) -> (N, C, H, W)
    units_tensor = units_tensor.permute(0, 3, 1, 2)

    indices = list(itertools.combinations(range(len(image_units)), 2))

    num_processes = multiprocessing.cpu_count()

    batch_size = 256
    batches = [
        indices[i : i + batch_size] for i in range(0, len(indices), batch_size)
    ]
    different_pairs = []

    with multiprocessing.Pool(processes=num_processes) as pool:
        worker = partial(
            _compare_batch_worker,
            units_tensor=units_tensor,
            positions_np=positions_np,
            threshold=threshold,
        )
        with tqdm(
            total=num_comparisons,
            desc="Processing",
            bar_format="{desc}: {n_fmt}/{total_fmt}",
        ) as pbar:
            for i, result_batch in enumerate(pool.imap(worker, batches)):
                if result_batch:
                    different_pairs.extend(result_batch)
                pbar.update(len(batches[i]))

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
