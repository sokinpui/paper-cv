# Image Unit Comparator

This program divides an image into 512x512 units and compares them using the Structural Similarity Index (SSIM) to find units that are different from each other.

## Features

- Divides any given image into 512x512 pixel units.
- Compares each unit with every other unit.
- Uses SSIM to quantify the similarity between units.
- Reports pairs of units whose SSIM score is below a specified threshold.
- Command-line interface for ease of use.

## Prerequisites

- Python 3.6+

## Setup

1.  **Clone the repository or download the source code.**

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the main script from the `image_comparator` directory, providing the path to your image.

```bash
python main.py /path/to/your/image.png
```
