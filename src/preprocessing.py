"""
Part 1 - Task 4: Preprocessing pipeline (no leakage).

- Numeric missing values -> median impute
- Categorical missing values -> mode impute
- product_category, payment_method -> one-hot encode
- Numeric features -> standard scale
- Pipeline is fit on the TRAIN split only; test is only transformed.

Import build_preprocessor() and load_split() from this module in every
downstream training script so Parts 5-10 all use the identical, leakage-safe
preprocessing definition.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

NUMERIC_FEATURES = [
    "price_inr", "discount_pct", "customer_tenure_days", "num_previous_orders",
    "num_previous_returns", "delivery_distance_km", "delivery_days",
    "is_weekend_order", "rating_given",
]
CATEGORICAL_FEATURES = ["product_category", "payment_method"]
TARGET = "returned"


def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipe, NUMERIC_FEATURES),
        ("cat", categorical_pipe, CATEGORICAL_FEATURES),
    ])
    return preprocessor


def load_split(csv_path: str = "orders_dataset.csv", test_size: float = 0.2, random_state: int = 42):
    """Load the CSV and return a stratified 80/20 train/test split (X, y each)."""
    df = pd.read_csv(csv_path)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def get_feature_names_out(preprocessor: ColumnTransformer) -> list:
    """Helper for Task 8 (feature importance) - readable post-transform feature names."""
    return list(preprocessor.get_feature_names_out())
