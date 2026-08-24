"""
Part 2 Task 8: Export real test-split images as actual .png files.

torchvision's FashionMNIST stores data as raw IDX binary, not individual image
files. Part 3's classify_product_image(image_path: str) tool needs real files
to point at, so this script picks >=5 real test images (covering different
classes) and writes them out as .png via PIL, named so the true label is
obvious from the filename.
"""
import os
import numpy as np
from PIL import Image
from torchvision import datasets

from model import CLASS_NAMES

DATA_ROOT = "part2/data/FashionMNIST"
OUTPUT_DIR = "data/sample_images"
N_SAMPLES_PER_CLASS = 1  # 1 per class x 10 classes = 10 total (>= 5 required)


def main():
    test_dataset = datasets.FashionMNIST(root=DATA_ROOT, train=False, download=True)
    # test_dataset[i] returns (PIL.Image, label) with NO transform applied here
    # (we want the raw 28x28 image bytes for a clean, human-viewable .png)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    labels = np.array(test_dataset.targets)
    exported = 0

    for class_idx, class_name in enumerate(CLASS_NAMES):
        candidate_positions = np.where(labels == class_idx)[0]
        chosen_idx = candidate_positions[0]  # first real test image of this class

        img, label = test_dataset[chosen_idx]  # PIL.Image (28x28, mode 'L')
        safe_name = class_name.replace("/", "_").replace(" ", "_").lower()
        filename = f"{exported:02d}_{safe_name}.png"
        filepath = os.path.join(OUTPUT_DIR, filename)

        img.save(filepath)
        print(f"Saved {filepath}  (true label: {class_name}, test-set index: {chosen_idx})")
        exported += 1

    print(f"\nExported {exported} real test-split images to {OUTPUT_DIR}")
    print("Commit these exact .png files -- Part 3's tool reads from this folder.")


if __name__ == "__main__":
    main()
