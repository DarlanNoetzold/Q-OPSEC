import pandas as pd
from src.common.config_loader import default_config_loader

config = default_config_loader.load("dataset_config.yaml")

print("=" * 80)
print("🔍 FRAUD RATE DIAGNOSTIC")
print("=" * 80)

fraud_config = config.get("fraud", {})
print(f"\n📋 Config values:")
print(f"  global_fraud_rate: {fraud_config.get('global_fraud_rate', 'NOT FOUND')}")
print(f"  user_fraudster_rate: {fraud_config.get('user_fraudster_rate', 'NOT FOUND')}")

print(f"\n📂 Loading dataset...")
df = pd.read_csv("output/dataset_full.csv", nrows=100000)
print(f"  Loaded {len(df):,} rows")

if "is_fraud" in df.columns:
    fraud_count = df["is_fraud"].sum()
    fraud_rate = fraud_count / len(df) * 100
    print(f"\n🎯 Fraud in dataset:")
    print(f"  Fraudulent: {fraud_count:,} ({fraud_rate:.2f}%)")
    print(f"  Legitimate: {len(df) - fraud_count:,} ({100 - fraud_rate:.2f}%)")
else:
    print(f"\n❌ Column 'is_fraud' not found!")

if "fraud_probability" in df.columns:
    print(f"\n📊 Fraud probability stats:")
    print(f"  Mean: {df['fraud_probability'].mean():.4f}")
    print(f"  Median: {df['fraud_probability'].median():.4f}")
    print(f"  Min: {df['fraud_probability'].min():.4f}")
    print(f"  Max: {df['fraud_probability'].max():.4f}")
    print(f"  Std: {df['fraud_probability'].std():.4f}")

    print(f"\n📈 Fraud probability distribution:")
    print(df['fraud_probability'].describe())
else:
    print(f"\n❌ Column 'fraud_probability' not found!")

print("\n" + "=" * 80)