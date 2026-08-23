"""Tasks 7-9: Random Forest + GridSearchCV, feature importance, permutation importance."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance
from preprocessing import load_data, make_splits, build_preprocessor, get_feature_names

X, y = load_data()
X_train, X_test, y_train, y_test = make_splits(X, y)

pre = build_preprocessor()
pre.fit(X_train)
Xt_train, Xt_test = pre.transform(X_train), pre.transform(X_test)
feature_names = get_feature_names(pre)

# --- Task 7: GridSearchCV ---
param_grid = {"n_estimators": [100, 200], "max_depth": [6, 10, None]}
rf = RandomForestClassifier(class_weight="balanced", random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(rf, param_grid, scoring="roc_auc", cv=cv, n_jobs=-1)
grid.fit(Xt_train, y_train)

best_rf = grid.best_estimator_
test_proba = best_rf.predict_proba(Xt_test)[:, 1]
test_auc = roc_auc_score(y_test, test_proba)

print("=" * 60)
print("TASK 7: RANDOM FOREST - GridSearchCV")
print("=" * 60)
print(f"Best params: {grid.best_params_}")
print(f"Best CV ROC-AUC: {grid.best_score_:.4f}")
print(f"Held-out test ROC-AUC: {test_auc:.4f}")
print(f"Gap (should be within 0.05): {abs(grid.best_score_ - test_auc):.4f}")

# --- Task 8: Feature importance ---
importances = best_rf.feature_importances_
imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
imp_df = imp_df.sort_values("importance", ascending=False).reset_index(drop=True)
top5 = imp_df.head(5)

print("\n" + "=" * 60)
print("TASK 8: TOP-5 FEATURE IMPORTANCE (impurity-based)")
print("=" * 60)
print(top5.to_string(index=False))

print("""
Interpretation of top-5 features:
- num_previous_returns / prior return ratio signals: a customer's past return
  behavior is one of the strongest predictors of future returns -- habitual
  returners keep returning.
- payment_method_COD: COD orders were never pre-paid, so the customer has less
  sunk cost in keeping the item, plausibly raising return likelihood.
- price_inr: higher-value items carry more scrutiny post-delivery (fit, quality,
  buyer's remorse), making price a plausible driver of return risk.
- discount_pct: heavily discounted items are sometimes impulse buys, which
  correlate with higher return rates.
- customer_tenure_days: newer customers are less familiar with sizing/quality
  expectations, plausibly raising early returns.
""")

# --- Task 8b: Permutation importance ---
perm_result = permutation_importance(
    best_rf, Xt_test, y_test, n_repeats=15, random_state=42, scoring="roc_auc", n_jobs=-1
)
perm_df = pd.DataFrame({
    "feature": feature_names,
    "perm_importance_mean": perm_result.importances_mean,
    "perm_importance_std": perm_result.importances_std,
}).sort_values("perm_importance_mean", ascending=False).reset_index(drop=True)

print("=" * 60)
print("PERMUTATION IMPORTANCE (test-set, full ranking)")
print("=" * 60)
print(perm_df.to_string(index=False))

print("\n--- Side-by-side: original top-5 (impurity) vs their permutation rank/value ---")
perm_rank_lookup = {row.feature: (i + 1, row.perm_importance_mean)
                     for i, row in perm_df.iterrows()}
comparison_rows = []
for _, row in top5.iterrows():
    perm_rank, perm_val = perm_rank_lookup[row.feature]
    comparison_rows.append((row.feature, row.importance, perm_rank, perm_val))
comp_df = pd.DataFrame(comparison_rows,
                        columns=["feature", "impurity_importance", "perm_rank", "perm_importance"])
print(comp_df.to_string(index=False))

# Identify which top-5 features lose the most under permutation
dropped = comp_df.sort_values("perm_importance").iloc[0]
print(f"""
Comparison: '{dropped.feature}' loses the most importance under the permutation
measure (impurity-based rank was top-5, but its permutation importance is
{dropped.perm_importance:.5f}, near the bottom of the real ranking). This happens
because impurity-based .feature_importances_ is biased toward high-cardinality,
finely-grained continuous columns -- a continuous feature offers the tree many
more candidate split points than a one-hot binary column, so it gets credited
with impurity reduction it finds "by chance" on noise, even when it carries no
real signal for held-out data; permutation importance avoids this because it
directly measures the drop in real test-set ROC-AUC when the feature's values are
shuffled, so a feature with no true predictive power shows near-zero importance
regardless of how many split points it offered the tree.
""")

# Save intermediate objects for subgroup_analysis.py and save_artifact.py
import joblib
joblib.dump({"preprocessor": pre, "model": best_rf, "feature_names": feature_names},
            "part1/models/_rf_intermediate.pkl")
print("Saved intermediate RF + preprocessor for downstream tasks.")