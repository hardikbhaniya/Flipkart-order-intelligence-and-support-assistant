"""
Task 11: Save the final artifact.

Combines the fitted preprocessor + tuned RandomForest into ONE fitted sklearn
Pipeline, re-runs the Task 6 threshold-sweep procedure on the RF's OWN
predict_proba output (not Logistic Regression's) to get t*_rf, and persists
everything to models/return_risk_model.pkl via joblib.
"""
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score
from preprocessing import load_data, make_splits, build_preprocessor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

X, y = load_data()
X_train, X_test, y_train, y_test = make_splits(X, y)

# Rebuild preprocessor + winning RF as ONE combined Pipeline (fit on train only)
preprocessor = build_preprocessor()
param_grid = {"n_estimators": [100, 200], "max_depth": [6, 10, None]}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rf = RandomForestClassifier(class_weight="balanced", random_state=42)
grid = GridSearchCV(rf, param_grid, scoring="roc_auc", cv=cv, n_jobs=-1)

full_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", grid),
])
full_pipeline.fit(X_train, y_train)

# Unwrap the GridSearchCV inside the pipeline down to the best RF for saving
best_rf = full_pipeline.named_steps["classifier"].best_estimator_
final_pipeline = Pipeline(steps=[
    ("preprocessor", full_pipeline.named_steps["preprocessor"]),
    ("classifier", best_rf),
])

# --- Threshold sweep on THIS model's own predict_proba (Task 11 requirement) ---
proba_test = final_pipeline.predict_proba(X_test)[:, 1]
thresholds = np.arange(0.1, 0.9 + 1e-9, 0.02)
results = []
for t in thresholds:
    p = (proba_test >= t).astype(int)
    f1_t = f1_score(y_test, p, pos_label=1, zero_division=0)
    rec_t = recall_score(y_test, p, pos_label=1, zero_division=0)
    prec_t = precision_score(y_test, p, pos_label=1, zero_division=0)
    results.append((round(t, 2), f1_t, rec_t, prec_t))

best = max(results, key=lambda r: r[1])
t_star_rf, f1_star, recall_star, precision_star = best
test_auc = roc_auc_score(y_test, proba_test)

print("=" * 60)
print("TASK 11: FINAL ARTIFACT + t*_rf")
print("=" * 60)
print(f"Final RF test ROC-AUC: {test_auc:.4f}")
print(f"t*_rf (F1-maximising threshold on RF's own predict_proba): {t_star_rf}")
print(f"  F1 at t*_rf:        {f1_star:.4f}")
print(f"  Recall at t*_rf:    {recall_star:.4f}")
print(f"  Precision at t*_rf: {precision_star:.4f}")

# Save artifact
import os
os.makedirs("part1/models", exist_ok=True)
joblib.dump(final_pipeline, "part1/models/return_risk_model.pkl")
print("\nSaved: part1/models/return_risk_model.pkl")

# Sanity check: reload and confirm predict_proba matches
reloaded = joblib.load("part1/models/return_risk_model.pkl")
reload_proba = reloaded.predict_proba(X_test)[:, 1]
assert np.allclose(proba_test, reload_proba), "Mismatch after reload!"
print("Sanity check passed: reloaded model's predict_proba matches in-memory model.")

# Persist t*_rf alongside the model for Part 3 to read
with open("part1/models/t_star_rf.txt", "w") as f:
    f.write(str(t_star_rf))
print(f"Saved t*_rf = {t_star_rf} to part1/models/t_star_rf.txt")