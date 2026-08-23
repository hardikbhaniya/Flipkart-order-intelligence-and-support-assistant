"""
Part 2 Task 3 (speed tip): Cache the frozen ResNet-18 backbone's output
features for every image ONCE, so head-only training doesn't need to re-run
the backbone's forward pass every epoch.

Mathematically identical to re-running the frozen backbone every epoch, but
turns an hours-long CPU loop into: one feature-extraction pass (a few minutes
on GPU, well under an hour on CPU) + a near-instant head-only training step.

Batch size: 64 (documented per Task 3 requirement).
Run this once; train_head.py and finetune.py both consume its cached output.
"""
import os
import torch
from torch.utils.data import DataLoader

from model import build_model
from dataset import load_datasets

BATCH_SIZE = 64
FEATURES_DIR = "part2/data/features"


def extract_and_cache(subset, name, model, device):
    loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    all_feats, all_labels = [], []

    model.eval()
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(loader):
            imgs = imgs.to(device)
            feats = model.extract_features(imgs)
            all_feats.append(feats.cpu())
            all_labels.append(labels)
            if i % 20 == 0:
                print(f"  [{name}] batch {i}/{len(loader)}")

    all_feats = torch.cat(all_feats, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    os.makedirs(FEATURES_DIR, exist_ok=True)
    torch.save(all_feats, os.path.join(FEATURES_DIR, f"{name}_features.pt"))
    torch.save(all_labels, os.path.join(FEATURES_DIR, f"{name}_labels.pt"))
    print(f"Saved {name}: features {all_feats.shape}, labels {all_labels.shape}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_subset, val_subset, test_dataset = load_datasets()
    model = build_model(pretrained=True).to(device)

    print("\nExtracting TRAIN features...")
    extract_and_cache(train_subset, "train", model, device)

    print("\nExtracting VAL features...")
    extract_and_cache(val_subset, "val", model, device)

    print("\nExtracting TEST features...")
    extract_and_cache(test_dataset, "test", model, device)

    print("\nDone. Cached features are in", FEATURES_DIR)


if __name__ == "__main__":
    main()
