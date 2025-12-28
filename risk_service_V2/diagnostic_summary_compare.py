# diagnostic_summary_compare.py
import pandas as pd

print("=" * 80)
print("🔍 SUMMARY vs REAL DATA DIAGNOSTIC")
print("=" * 80)

path = "output/dataset_full.csv"  # ajuste se o seu output_dir for outro
print(f"📂 Loading: {path}")

df = pd.read_csv(path, usecols=["is_fraud"])
n = len(df)
fraud_count = df["is_fraud"].sum()
fraud_rate = fraud_count / n * 100

print(f"\nTotal rows: {n:,}")
print(f"Fraudulent: {fraud_count:,} ({fraud_rate:.2f}%)")
print(f"Legitimate: {n - fraud_count:,} ({100 - fraud_rate:.2f}%)")

print("\n" + "=" * 80)