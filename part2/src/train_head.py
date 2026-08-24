"""
Part 2 Task 3: Train only the new classifier head (feature extraction stage),
using the cached backbone features from extract_features.py.

Documented hyperparameters:
- Batch size: 128 (fast, since these are just 512-dim vectors, not images)
- Optimizer: Adam
- Learning rate: 1e-3
- Epochs: 15
"""
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from model import build_model

FEATURES_DIR = "part2/data/features"
BATCH_SIZE = 128
LR = 1e-3
EPOCHS = 15


def load_cached(name):
    feats = torch.load(f"{FEATURES_DIR}/{name}_features.pt")
    labels = torch.load(f"{FEATURES_DIR}/{name}_labels.pt")
    return TensorDataset(feats, labels)


def evaluate(model, loader, device):
    model.head.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for feats, labels in loader:
            feats, labels = feats.to(device), labels.to(device)
            logits = model.forward_from_features(feats)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Hyperparameters: batch_size={BATCH_SIZE}, optimizer=Adam, lr={LR}, epochs={EPOCHS}")

    train_ds = load_cached("train")
    val_ds = load_cached("val")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = build_model(pretrained=True).to(device)  # backbone frozen, unused here except head
    optimizer = torch.optim.Adam(model.head.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.head.train()
        total_loss = 0.0
        for feats, labels in train_loader:
            feats, labels = feats.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model.forward_from_features(feats)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * feats.size(0)

        train_loss = total_loss / len(train_ds)
        val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:2d}/{EPOCHS} | train_loss={train_loss:.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.head.state_dict(), "part2/models/head_feature_extraction.pt")

    print(f"\nBest feature-extraction validation accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    if best_val_acc < 0.80:
        print("Validation accuracy is BELOW 80% -> run finetune.py next (Task 4 requires this).")
    else:
        print("Validation accuracy is >= 80% -> feature extraction alone is sufficient; "
              "fine-tuning (finetune.py) is optional.")


if __name__ == "__main__":
    import os
    os.makedirs("part2/models", exist_ok=True)
    main()
