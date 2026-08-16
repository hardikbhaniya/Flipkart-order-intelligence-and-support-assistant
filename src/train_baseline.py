"""
Part 1 - Task 5: Baseline DummyClassifier.

Run from the repo root: python3 src/train_baseline.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report

from preprocessing import build_preprocessor, load_split

X_train, X_test, y_train, y_test = load_split("orders_dataset.csv")

baseline_pipe = Pipeline(steps=[
    ("preprocess", build_preprocessor()),
    ("model", DummyClassifier(strategy="most_frequent", random_state=42)),
])
baseline_pipe.fit(X_train, y_train)
y_pred = baseline_pipe.predict(X_test)

acc = accuracy_score(y_test, y_pred)
f1_class1 = f1_score(y_test, y_pred, pos_label=1)

print("=" * 60)
print("BASELINE: DummyClassifier (most_frequent)")
print("=" * 60)
print(f"Accuracy:            {acc:.4f}")
print(f"F1 (returned=1):     {f1_class1:.4f}")
print("\nFull classification report:")
print(classification_report(y_test, y_pred, digits=4))

print("""
WHY HIGH ACCURACY IS MISLEADING HERE:
The DummyClassifier always predicts the majority class (returned=0), so it
scores a deceptively high accuracy purely because ~77% of orders in this
dataset are not returned -- it never actually identifies a single returned
order. Its F1-score for the returned=1 class is 0.0 (precision and recall
are both 0, since it never predicts class 1 at all), which is the real
story: "high accuracy, zero recall" -- the classic failure mode of judging
an imbalanced-classification model by accuracy alone. Any usable return-risk
model must beat this trivial baseline's F1/recall on the minority class, not
just its accuracy, since accuracy alone would reward doing nothing.
""")
