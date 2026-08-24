"""
Part 3 Task 4: classify_product_image(image_path) -> dict

Loads Part 2's REAL saved models/product_classifier.pt and runs it against
one of the real .png files exported in Part 2 Task 8. NOT a hardcoded
stand-in -- this imports and calls Part 2's actual predict.py function.
"""
import sys
import os

# Make Part 2's src/ importable (same repo, sibling folder)
PART2_SRC = os.path.join(os.path.dirname(__file__), "../../part2/src")
sys.path.insert(0, PART2_SRC)

from predict import predict_image  # noqa: E402  (Part 2's real load+predict function)

PART2_MODEL_PATH = os.path.join(PART2_SRC, "../models/product_classifier.pt")


def classify_product_image(image_path: str) -> dict:
    """
    image_path is pointing at one of the real .png files exported in
    Part 2 Task 8 (e.g. ../../data/sample_images/03_sneaker.png).
    """
    result = predict_image(image_path, model_path=PART2_MODEL_PATH)
    return {
        "predicted_category": result["predicted_category"],
        "confidence": result["confidence"],
    }


if __name__ == "__main__":
    # Manual smoke test against one of the exported sample images
    sample_path = "../../data/sample_images/00_t-shirt_top.png"
    print("Image:", sample_path)
    print("Result:", classify_product_image(sample_path))
