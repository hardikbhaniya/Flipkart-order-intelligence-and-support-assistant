"""
Part 3 Task 3: check_return_risk(order_features) -> dict

Loads Part 1's REAL saved models/return_risk_model.pkl (the tuned Random
Forest Pipeline) and returns predict_proba's actual output plus a risk
bucket. NOT a hardcoded stand-in -- this calls the real artifact.

Bucket cut points are anchored to t*_rf (the F1-maximising threshold computed
on THIS model's own predict_proba during Part 1 Task 11), not fixed values:
  - "Low"    if probability <  t*_rf
  - "High"   if probability >= t*_rf + 0.15
  - "Medium" otherwise

Fixed cut points like 0.3/0.6 are not self-calibrating -- two equally valid
Random Forest models can produce completely different probability ranges, so
anchoring to this specific model's own t*_rf keeps the buckets meaningful.
"""
import os
import joblib
import pandas as pd

# Adjust these relative paths if your folder structure differs
PART1_MODEL_PATH = "../../part1/models/return_risk_model.pkl"
PART1_TSTAR_PATH = "../../part1/models/t_star_rf.txt"

_cache = {}


def _load():
    if "model" not in _cache:
        _cache["model"] = joblib.load(PART1_MODEL_PATH)
        with open(PART1_TSTAR_PATH) as f:
            _cache["t_star_rf"] = float(f.read().strip())
    return _cache["model"], _cache["t_star_rf"]


def check_return_risk(order_features: dict) -> dict:
    """
    order_features must contain the same columns Part 1's pipeline expects:
    price_inr, discount_pct, customer_tenure_days, num_previous_orders,
    num_previous_returns, delivery_distance_km, delivery_days,
    is_weekend_order, rating_given, product_category, payment_method
    """
    model, t_star_rf = _load()

    X = pd.DataFrame([order_features])
    probability = float(model.predict_proba(X)[:, 1][0])

    if probability < t_star_rf:
        bucket = "Low"
    elif probability >= t_star_rf + 0.15:
        bucket = "High"
    else:
        bucket = "Medium"

    return {
        "return_probability": round(probability, 4),
        "risk_bucket": bucket,
        "t_star_rf": t_star_rf,
        "cut_points": f"Low < {t_star_rf:.2f} <= Medium < {t_star_rf + 0.15:.2f} <= High",
    }


if __name__ == "__main__":
    # Manual smoke test with a realistic order
    example_order = {
        "price_inr": 1800, "discount_pct": 35.0, "customer_tenure_days": 120,
        "num_previous_orders": 4, "num_previous_returns": 2,
        "delivery_distance_km": 400.0, "delivery_days": 6,
        "is_weekend_order": 1, "rating_given": 3.0,
        "product_category": "Apparel", "payment_method": "COD",
    }
    result = check_return_risk(example_order)
    print("Example order:", example_order)
    print("Result:", result)
