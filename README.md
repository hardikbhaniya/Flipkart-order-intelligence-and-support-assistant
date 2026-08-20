# Part 1 — Return-Risk Scoring Pipeline

## How to regenerate the dataset and model

```bash
# 1. Create + activate a virtual environment (Windows PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install numpy pandas scikit-learn joblib

# 2. Generate the exact seeded dataset (do not edit the seed / probabilities)
python generate_orders.py
# -> writes orders_dataset.csv (6000 rows, return rate ~22.75%)

# 3. Verify the data (Task 3: row counts, missingness, MAR justification)
python verify_data.py

# 4. Sanity-check the leakage-safe preprocessing pipeline (Task 4)
python preprocessing.py

# 5. Baseline DummyClassifier (Task 5)
python train_baseline.py

# 6. Logistic Regression + threshold sweep (Task 6)
python train_logistic_regression.py

# 7. Random Forest + GridSearchCV + feature/permutation importance (Tasks 7-9)
python train_random_forest.py

# 8. Subgroup analysis (Task 10)
python subgroup_analysis.py

# 9. Save final artifact + compute t*_rf (Task 11)
python save_artifact.py
```

## Outputs produced
- `orders_dataset.csv` — the seeded 6,000-row dataset
- `models/return_risk_model.pkl` — final tuned Random Forest **Pipeline**
  (preprocessing + classifier together), loadable via `joblib.load(...)`
- `models/t_star_rf.txt` — the F1-maximising threshold computed on this exact
  model's own `predict_proba` output, for Part 3's risk-bucket cut points

## Results summary
| Metric | Value |
|---|---|
| Dataset rows | 6,000 |
| Overall return rate | 22.75% |
| `rating_given` missing | 13.05% (MAR, conditional on `payment_method`) |
| Baseline F1 (class 1) | 0.0000 |
| Logistic Regression AUC / F1 (default thr.) | 0.6253 / 0.3921 |
| Logistic Regression best threshold | 0.44 (recall +17.9 pts vs default) |
| Random Forest best CV ROC-AUC | 0.6179 |
| Random Forest test ROC-AUC | 0.6143 |
| **t\*\_rf** (saved artifact) | **0.46** |