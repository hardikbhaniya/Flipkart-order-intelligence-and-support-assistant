"""
Part 2 Task 7: Documented one-function loading + single-image-prediction
snippet. This is EXACTLY what Part 3's classify_product_image tool imports
and calls -- it is not a hardcoded stand-in, it runs the real saved model.
"""
import torch
from PIL import Image

from model import build_model, CLASS_NAMES
from dataset import build_transform

MODEL_PATH = "part2/models/product_classifier.pt"  # adjust relative path from caller's cwd

_model_cache = {}


def load_model(model_path: str = MODEL_PATH):
    """Loads (and caches) the trained FlipkartProductClassifier once."""
    if "model" not in _model_cache:
        device = torch.device("cpu")
        model = build_model(pretrained=True)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        _model_cache["model"] = model
    return _model_cache["model"]


def predict_image(image_path: str, model_path: str = MODEL_PATH) -> dict:
    """
    Runs the real saved classifier on a single image file.
    Returns: {"predicted_category": str, "confidence": float}
    """
    model = load_model(model_path)
    transform = build_transform()

    img = Image.open(image_path)
    tensor = transform(img).unsqueeze(0)  # add batch dimension

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        confidence, pred_idx = torch.max(probs, dim=1)

    return {
        "predicted_category": CLASS_NAMES[pred_idx.item()],
        "confidence": round(confidence.item(), 4),
    }


if __name__ == "__main__":
    # Quick manual test against one of the exported sample images
    import sys
    test_path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_images/00_t-shirt_top.png"
    result = predict_image(test_path)
    print(f"Image: {test_path}")
    print(f"Prediction: {result}")
