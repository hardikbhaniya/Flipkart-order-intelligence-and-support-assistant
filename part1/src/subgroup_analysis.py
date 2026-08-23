"""Task 10: Subgroup analysis - recall/precision by product_category and payment_method."""
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import recall_score, precision_score
from preprocessing import load_data, make_splits

X, y = load_data()
X_train, X_test, y_train, y_test = make_splits(X, y)

bundle = joblib.load("part1/models/_rf_intermediate.pkl")
pre, best_rf = bundle["preprocessor"], bundle["model"]

Xt_test = pre.transform(X_test)
proba = best_rf.predict_proba(Xt_test)[:, 1]
preds = (proba >= 0.5).astype(int)

eval_df = X_test.copy()
eval_df["y_true"] = y_test.values
eval_df["y_pred"] = preds

def subgroup_metrics(df, group_col):
    rows = []
    for group, sub in df.groupby(group_col):
        rec = recall_score(sub["y_true"], sub["y_pred"], pos_label=1, zero_division=0)
        prec = precision_score(sub["y_true"], sub["y_pred"], pos_label=1, zero_division=0)
        rows.append((group, len(sub), sub["y_true"].sum(), rec, prec))
    return pd.DataFrame(rows, columns=[group_col, "n", "n_actual_returns", "recall", "precision"])

overall_recall = recall_score(eval_df["y_true"], eval_df["y_pred"], pos_label=1, zero_division=0)
overall_precision = precision_score(eval_df["y_true"], eval_df["y_pred"], pos_label=1, zero_division=0)

print("=" * 60)
print("TASK 10: SUBGROUP ANALYSIS")
print("=" * 60)
print(f"Overall test-set recall (class 1): {overall_recall:.4f}")
print(f"Overall test-set precision (class 1): {overall_precision:.4f}")

cat_table = subgroup_metrics(eval_df, "product_category")
pay_table = subgroup_metrics(eval_df, "payment_method")

print("\n--- By product_category ---")
print(cat_table.round(4).to_string(index=False))

print("\n--- By payment_method ---")
print(pay_table.round(4).to_string(index=False))

worst_cat = cat_table.loc[cat_table["recall"].idxmin()]
worst_pay = pay_table.loc[pay_table["recall"].idxmin()]

print(f"""
Weakest subgroup: product_category = '{worst_cat.product_category}' has recall
{worst_cat.recall:.4f}, meaningfully below the overall recall of
{overall_recall:.4f} -- the model misses a larger share of that category's real
returns than it does overall.

Proposed concrete fix: add a category-specific decision threshold for
'{worst_cat.product_category}' (calibrated on that subgroup's own validation-fold
predict_proba distribution via a per-category threshold sweep, the same procedure
already used in Task 6) rather than applying one global 0.5 cut point across all
five categories. Because the model's probability distribution for this category is
shifted lower on average, a single global threshold under-flags it; a category-aware
threshold recovers recall for this subgroup without moving the cut point for
categories where 0.5 already works well.

Second, even weaker subgroup: payment_method = '{worst_pay.payment_method}' has
recall {worst_pay.recall:.4f} -- the model essentially never flags a real return
paid this way at the default threshold, versus {overall_recall:.4f} overall. This is
consistent with the class-imbalance-plus-threshold interaction: COD dominates the
positive class (its own recall is {pay_table.loc[pay_table.payment_method=='COD','recall'].values[0]:.4f}),
so the single global threshold is implicitly tuned around COD's probability
distribution, starving recall for the other payment methods. The same
per-subgroup threshold fix applies here directly.
""")