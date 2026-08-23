"""
Part 2 Task 7: Save the final trained model to models/product_classifier.pt.

This is the ONE artifact Part 3's classify_product_image tool will load.
Run this AFTER train_head.py (and finetune.py, if fine-tuning was needed).

Logic:
- If ../models/finetuned_full_model.pt exists (i.e. finetune.py was run and
  it improved on the feature-extraction-only model), that becomes the final
  artifact.
- Otherwise, the feature-extraction backbone + trained head from train_head.py
  becomes the final artifact.
Either way, the SAME FlipkartProductClassifier architecture is saved, so
predict.py / Part 3's tool never need to know which path was taken.
"""
import os
import torch
from model import build_model

MODELS_DIR = "part2/models"
FINETUNED_PATH = os.path.join(MODELS_DIR, "finetuned_full_model.pt")
HEAD_ONLY_PATH = os.path.join(MODELS_DIR, "head_feature_extraction.pt")
FINAL_PATH = os.path.join(MODELS_DIR, "product_classifier.pt")


def main():
    device = torch.device("cpu")  # save in CPU-portable form
    model = build_model(pretrained=True).to(device)

    if os.path.exists(FINETUNED_PATH):
        print(f"Found fine-tuned model at {FINETUNED_PATH} -- using this as final artifact.")
        state_dict = torch.load(FINETUNED_PATH, map_location=device)
        model.load_state_dict(state_dict)
    elif os.path.exists(HEAD_ONLY_PATH):
        print(f"No fine-tuned model found -- using feature-extraction-only head "
              f"from {HEAD_ONLY_PATH} on top of the frozen pretrained backbone.")
        head_state = torch.load(HEAD_ONLY_PATH, map_location=device)
        model.head.load_state_dict(head_state)
    else:
        raise FileNotFoundError(
            "No trained head found. Run train_head.py first (and finetune.py if needed)."
        )

    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.save(model.state_dict(), FINAL_PATH)
    print(f"Saved final artifact to {FINAL_PATH}")

    # Sanity check: reload and confirm it loads cleanly
    reloaded = build_model(pretrained=True)
    reloaded.load_state_dict(torch.load(FINAL_PATH, map_location=device))
    reloaded.eval()
    print("Sanity check passed: model reloads correctly from product_classifier.pt")


if __name__ == "__main__":
    main()
