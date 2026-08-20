"""
Task 4: Leakage-safe preprocessing pipeline.

Builds a ColumnTransformer + Pipeline that:
- imputes missing numeric values with the median
- imputes missing categorical values with the mode
- one-hot encodes product_category and payment_method
- standard-scales numeric features

This is a single fitted object shared across baseline / logistic regression /
random forest, so it is always fit on the TRAIN split only and only ever
.transform()-ed (never re-fit) on validation/test data.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

NUMERIC_FEATURES = [
    "price_inr", "discount_pct", "customer_tenure_days", "num_previous_orders",
    "num_previous_returns", "delivery_distance_km", "delivery_days",
    "is_weekend_order", "rating_given",
]
CATEGORICAL_FEATURES = ["product_category", "payment_method"]
TARGET = "returned"


def load_data(path="orders_dataset.csv"):
    df = pd.read_csv(path)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    return X, y


def make_splits(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def build_preprocessor():
    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipe, NUMERIC_FEATURES),
        ("cat", categorical_pipe, CATEGORICAL_FEATURES),
    ])
    return preprocessor


def get_feature_names(preprocessor):
    """Return the expanded feature names after the ColumnTransformer runs
    (numeric names unchanged, categorical names expanded by OneHotEncoder)."""
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    return NUMERIC_FEATURES + cat_names


if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = make_splits(X, y)
    pre = build_preprocessor()
    pre.fit(X_train)  # fit on TRAIN ONLY
    Xt_train = pre.transform(X_train)
    Xt_test = pre.transform(X_test)  # transform only, never fit
    print("Train shape:", Xt_train.shape, "| Test shape:", Xt_test.shape)
    print("Feature names:", get_feature_names(pre))
    