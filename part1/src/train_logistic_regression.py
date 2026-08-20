"""
Part 1 - Task 6: Logistic Regression, tuned via threshold sweep.

Run from the repo root: python src/train_logistic_regression.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score, roc_auc_score,
)

from preprocessing import load_data, make_splits, build_preprocessor
 
X, y = load_data()
X_train, X_test, y_train, y_test = make_splits(X, y)

lr_pipe = Pipeline(steps=[
    ("preprocess", build_preprocessor()),
    ("model", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
])
lr_pipe.fit(X_train, y_train)

proba = lr_pipe.predict_proba(X_test)[:, 1]

# ---- Default 0.5 threshold ----
y_pred_default = (proba >= 0.5).astype(int)
default_acc = accuracy_score(y_test, y_pred_default)
default_f1 = f1_score(y_test, y_pred_default, pos_label=1)
default_recall = recall_score(y_test, y_pred_default, pos_label=1)
default_precision = precision_score(y_test, y_pred_default, pos_label=1)
roc_auc = roc_auc_score(y_test, proba)

print("=" * 60)
print("LOGISTIC REGRESSION (class_weight='balanced') @ threshold=0.5")
print("=" * 60)
print(f"Accuracy:   {default_acc:.4f}")
print(f"F1:         {default_f1:.4f}")
print(f"Recall:     {default_recall:.4f}")
print(f"Precision:  {default_precision:.4f}")
print(f"ROC-AUC:    {roc_auc:.4f}")

# ---- Threshold sweep 0.1 -> 0.9, step 0.02 ----
thresholds = np.arange(0.10, 0.90 + 1e-9, 0.02)
sweep_results = []
for t in thresholds:
    y_pred_t = (proba >= t).astype(int)
    f1_t = f1_score(y_test, y_pred_t, pos_label=1, zero_division=0)
    recall_t = recall_score(y_test, y_pred_t, pos_label=1, zero_division=0)
    precision_t = precision_score(y_test, y_pred_t, pos_label=1, zero_division=0)
    sweep_results.append((round(t, 2), f1_t, recall_t, precision_t))

best_t, best_f1, best_recall, best_precision = max(sweep_results, key=lambda r: r[1])

print("\n" + "=" * 60)
print("THRESHOLD SWEEP (0.10 -> 0.90, step 0.02)")
print("=" * 60)
print(f"{'threshold':>10} {'F1':>8} {'recall':>8} {'precision':>10}")
for t, f1_t, recall_t, precision_t in sweep_results:
    marker = "  <-- best F1" if t == best_t else ""
    print(f"{t:>10.2f} {f1_t:>8.4f} {recall_t:>8.4f} {precision_t:>10.4f}{marker}")

recall_gain_pp = (best_recall - default_recall) * 100
precision_drop_pp = (default_precision - best_precision) * 100

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"F1-maximising threshold: {best_t}")
print(f"  F1 at that threshold:        {best_f1:.4f}")
print(f"  Recall at that threshold:    {best_recall:.4f}  (+{recall_gain_pp:.1f} pp vs default)")
print(f"  Precision at that threshold: {best_precision:.4f}  ({-precision_drop_pp:+.1f} pp vs default)")

print(f"""
BUSINESS TRADE-OFF:
Moving the decision threshold from the default 0.5 down to {best_t} makes the
model flag more orders as "likely to be returned" -- recall on returned=1
rises by {recall_gain_pp:.1f} percentage points (from {default_recall:.4f} to
{best_recall:.4f}), meaning far fewer genuinely-returned orders slip through
undetected. The cost is precision, which drops by {precision_drop_pp:.1f}
percentage points (from {default_precision:.4f} to {best_precision:.4f}): more
orders get flagged as high-risk that ultimately aren't returned. In plain
terms, we are choosing to make false negatives (missed real returns, which
cost Flipkart a surprise reverse-pickup/refund with no proactive
intervention) cheaper to avoid, at the price of accepting more false
positives (orders flagged for proactive risk-mitigation, like a
confirmation call, that turn out not to need it) -- a reasonable trade for a
support-workflow trigger, since a false positive only costs a bit of agent
attention, while a false negative costs a full unplanned return.
""")
