"""
Part 1 - Task 2: Verify the generated data.

Reports: row/column count, overall return rate, % missing rating_given,
return rate by product_category and by payment_method, and the MAR
classification with numeric evidence.

Run: python3 verify_data.py
"""

import pandas as pd

df = pd.read_csv("orders_dataset.csv")

print("=" * 60)
print("BASIC SHAPE")
print("=" * 60)
print(f"Rows: {len(df)}  Columns: {df.shape[1]}")
assert len(df) == 6000 and df.shape[1] == 13, "Dataset shape does not match spec!"

overall_return_rate = df["returned"].mean()
print(f"Overall return rate: {overall_return_rate:.4f} ({overall_return_rate*100:.2f}%)")
assert 0.18 <= overall_return_rate <= 0.27, "Return rate outside required 18-27% band!"

pct_missing_rating = df["rating_given"].isna().mean()
print(f"% missing rating_given: {pct_missing_rating*100:.2f}%")
assert 0.08 <= pct_missing_rating <= 0.18, "Missing-rating % outside required 8-18% band!"

print("\n" + "=" * 60)
print("RETURN RATE BY product_category")
print("=" * 60)
by_cat = df.groupby("product_category")["returned"].agg(["mean", "count"])
by_cat.columns = ["return_rate", "n_orders"]
by_cat["return_rate"] = by_cat["return_rate"].round(4)
print(by_cat.sort_values("return_rate", ascending=False))

print("\n" + "=" * 60)
print("RETURN RATE BY payment_method")
print("=" * 60)
by_pay = df.groupby("payment_method")["returned"].agg(["mean", "count"])
by_pay.columns = ["return_rate", "n_orders"]
by_pay["return_rate"] = by_pay["return_rate"].round(4)
print(by_pay.sort_values("return_rate", ascending=False))

print("\n" + "=" * 60)
print("MISSINGNESS MECHANISM FOR rating_given")
print("=" * 60)
missing_by_pay = df.assign(is_missing=df["rating_given"].isna()).groupby("payment_method")["is_missing"].mean()
cod_missing = missing_by_pay["COD"]
non_cod_missing = df.assign(is_missing=df["rating_given"].isna()).loc[df["payment_method"] != "COD", "is_missing"].mean()
gap = cod_missing - non_cod_missing

print(missing_by_pay.round(4))
print(f"\nCOD missing rate:      {cod_missing:.4f} ({cod_missing*100:.2f}%)")
print(f"Non-COD missing rate:  {non_cod_missing:.4f} ({non_cod_missing*100:.2f}%)")
print(f"Gap (COD - non-COD):   {gap:.4f} ({gap*100:.2f} percentage points)")

print(f"""
CLASSIFICATION: Missing At Random (MAR).

Justification: rating_given's missingness is NOT independent of the other
observed columns (which would be MCAR), and it does NOT depend on the
unobserved rating value itself (which would be MNAR). Instead, the
generator sets P(missing) = 0.22 for COD orders and 0.06 for all other
payment methods -- a rule conditioned entirely on the observed
payment_method column. The measured gap above ({gap*100:.1f} percentage
points between COD and non-COD orders) is the direct empirical evidence:
COD orders are missing a rating roughly {cod_missing/non_cod_missing:.1f}x
more often than non-COD orders, and this is exactly the definition of MAR
-- missingness that depends on another *observed* variable, not on the
missing value itself.
""")
