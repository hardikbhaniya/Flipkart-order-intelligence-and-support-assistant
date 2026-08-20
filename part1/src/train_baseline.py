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

from preprocessing import load_data, make_splits, build_preprocessor

# X_train, X_test, y_train, y_test = load_split("orders_dataset.csv")
X, y = load_data()
X_train, X_test, y_train, y_test = make_splits(X, y)

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

print(f"""
Why high accuracy is misleading here:
The dataset's negative class (returned=0) makes up about {(1 - y.mean())*100:.1f}% of
orders, so a classifier that always predicts "not returned" scores {acc*100:.1f}%
accuracy while being completely useless -- it never once identifies an actual
return. This is the classic "high accuracy, zero recall" trap: on an imbalanced
dataset, accuracy rewards a model for parroting the majority class, and a metric
that ignores class balance (accuracy) makes a model with F1={f1_class1:.1f} for the
class we actually care about look deceptively strong. The two honest-evaluation
fixes this task is built on are (1) always compare against a baseline like this
one, so an F1 of 0.30 elsewhere is judged against a floor of 0.0, not in a vacuum,
and (2) pick metrics aligned to the real business problem -- here that's
recall/F1 on returned=1, not raw accuracy, because missing real returns is the
costly error the business actually wants to avoid.
""")