"""
Part 2 Tasks 1-2: Load Fashion-MNIST, carve a stratified validation split out
of train, and apply ResNet-18-ready transforms (3-channel, 224x224,
ImageNet-normalized).

Pinned source: https://github.com/zalandoresearch/fashion-mnist
Fetched via torchvision.datasets.FashionMNIST(download=True) -- same canonical
dataset, zero configuration, no login/API key needed.
"""
import numpy as np
from torch.utils.data import Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

from model import IMAGENET_MEAN, IMAGENET_STD, INPUT_SIZE

DATA_ROOT = "part2/data"  # run scripts from part2/src/


def build_transform():
    """Resize to 224x224, replicate the single grayscale channel to 3,
    normalize with the ImageNet mean/std the ResNet-18 backbone expects."""
    return transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_datasets(val_size: int = 6000, random_state: int = 42):
    """
    Returns (train_subset, val_subset, test_dataset).
    - Standard 60,000-image train split, standard 10,000-image test split.
    - A stratified validation split of `val_size` images (>= 5,000 required)
      is carved OUT of the 60,000 train images, leaving test fully untouched.
    """
    tfm = build_transform()

    full_train = datasets.FashionMNIST(root=DATA_ROOT, train=True, download=True, transform=tfm)
    test_dataset = datasets.FashionMNIST(root=DATA_ROOT, train=False, download=True, transform=tfm)

    targets = np.array(full_train.targets)
    indices = np.arange(len(full_train))

    train_idx, val_idx = train_test_split(
        indices, test_size=val_size, stratify=targets, random_state=random_state
    )

    train_subset = Subset(full_train, train_idx)
    val_subset = Subset(full_train, val_idx)

    print(f"Train split: {len(train_subset)} images")
    print(f"Validation split: {len(val_subset)} images (stratified)")
    print(f"Test split: {len(test_dataset)} images (untouched until final eval)")

    return train_subset, val_subset, test_dataset


if __name__ == "__main__":
    load_datasets()
