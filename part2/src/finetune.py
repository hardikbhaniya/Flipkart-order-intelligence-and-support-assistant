"""
Part 2 Task 4: Conditional fine-tuning.

Run this ONLY if train_head.py reported feature-extraction validation
accuracy below 80%. Unfreezes ResNet-18's late layer (layer4) while keeping
early/middle layers frozen, and continues training end-to-end (full images,
not cached features -- fine-tuning changes the backbone's own weights, so
cached features from the old frozen backbone are no longer valid) at a lower
learning rate. This is the standard gradual-unfreezing strategy.

Documented hyperparameters:
- Batch size: 32 (smaller than head-only stage since we now backprop through
  part of the CNN, which is more memory-hungry)
- Optimizer: Adam
- Learning rate: 1e-4 (10x lower than head-only stage, standard fine-tuning practice)
- Epochs: 5
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import build_model
from dataset import load_datasets

BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 5


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Hyperparameters: batch_size={BATCH_SIZE}, optimizer=Adam, lr={LR}, epochs={EPOCHS}")

    train_subset, val_subset, _ = load_datasets()
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = build_model(pretrained=True).to(device)
    # Load the head weights already trained in train_head.py, then unfreeze layer4.
    model.head.load_state_dict(torch.load("../models/head_feature_extraction.pt", map_location=device))
    model.unfreeze_late_layers()

    before_acc = evaluate(model, val_loader, device)
    print(f"Validation accuracy BEFORE fine-tuning: {before_acc:.4f} ({before_acc*100:.2f}%)")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = before_acc
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)

        train_loss = total_loss / len(train_subset)
        val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}/{EPOCHS} | train_loss={train_loss:.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "../models/finetuned_full_model.pt")

    print(f"\nValidation accuracy AFTER fine-tuning: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"Before: {before_acc:.4f} -> After: {best_val_acc:.4f} "
          f"(change: {(best_val_acc - before_acc)*100:+.2f} pts)")


if __name__ == "__main__":
    main()
