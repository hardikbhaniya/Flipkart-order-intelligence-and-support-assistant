"""
Part 1 - Task 2: Verify the generated data.

Reports: row/column count, overall return rate, % missing rating_given,
return rate by product_category and by payment_method, and the MAR
classification with numeric evidence.

"""

import pandas as pd

df = pd.read_csv("part1/src/orders_dataset.csv")

print("=" * 60)
print("TASK 3: DATA VERIFICATION")
print("=" * 60)

print(f"\nTotal rows: {len(df)}")
print(f"Total columns: {df.shape[1]}")
print(f"Overall return rate: {df['returned'].mean():.4f} ({df['returned'].mean()*100:.2f}%)")

missing_pct = df["rating_given"].isna().mean() * 100
print(f"Missing rating_given: {missing_pct:.2f}%")

print("\n--- Return rate by product_category ---")
cat_table = df.groupby("product_category")["returned"].agg(["mean", "count"])
cat_table.columns = ["return_rate", "n_orders"]
print(cat_table.round(4))

print("\n--- Return rate by payment_method ---")
pay_table = df.groupby("payment_method")["returned"].agg(["mean", "count"])
pay_table.columns = ["return_rate", "n_orders"]
print(pay_table.round(4))

print("\n--- Missingness rate: rating_given by payment_method ---")
miss_by_pay = df.groupby("payment_method")["rating_given"].apply(lambda s: s.isna().mean())
print(miss_by_pay.round(4))

cod_missing = df.loc[df["payment_method"] == "COD", "rating_given"].isna().mean()
non_cod_missing = df.loc[df["payment_method"] != "COD", "rating_given"].isna().mean()
gap = (cod_missing - non_cod_missing) * 100

print("\n" + "=" * 60)
print("MISSINGNESS CLASSIFICATION: MAR (Missing At Random)")
print("=" * 60)
print(f"""
rating_given is MAR, not MCAR and not MNAR.

Evidence: the missing rate for rating_given is {cod_missing*100:.2f}% for COD orders
versus {non_cod_missing*100:.2f}% for non-COD orders -- a gap of {gap:.2f} percentage
points. This is exactly how the generator built it: missing_mask uses probability
0.22 when payment_method == 'COD' and 0.06 otherwise, so whether a rating is missing
depends entirely on another OBSERVED column (payment_method).

- Not MCAR: MCAR would require the missing rate to be roughly equal across all
  payment methods (no dependency on any other column). Here it clearly is not --
  COD orders are missing at ~3.5x the rate of non-COD orders.
- Not MNAR: MNAR would mean the missingness depends on the unobserved rating value
  itself (e.g. unhappy customers with low ratings skip leaving one). The generator
  never looks at the rating_given value when deciding to blank it out -- the mask is
  drawn purely from payment_method, which we DO observe. So it's not MNAR.
- MAR fits: the probability of missingness is fully explained by an observed
  covariate (payment_method), which is the textbook definition of MAR.
""")