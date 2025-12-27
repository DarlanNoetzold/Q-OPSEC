"""
Dataset Explorer - Comprehensive analysis of generated fraud detection dataset
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")

# Configuration
OUTPUT_DIR = Path("output")
REPORT_DIR = Path("output/analysis")
REPORT_DIR.mkdir(exist_ok=True)


def load_datasets():
    """Load all parquet files."""
    print("=" * 80)
    print("📂 LOADING DATASETS")
    print("=" * 80)

    datasets = {}
    files = {
        "full": "dataset_full.parquet",
        "train": "dataset_train.parquet",
        "val": "dataset_val.parquet",
        "test": "dataset_test.parquet"
    }

    for name, filename in files.items():
        filepath = OUTPUT_DIR / filename
        if filepath.exists():
            df = pd.read_parquet(filepath)
            datasets[name] = df
            print(
                f"✅ {name:6s}: {len(df):>10,} rows | {len(df.columns):>3} columns | {df.memory_usage(deep=True).sum() / 1024 ** 2:>6.1f} MB")
        else:
            print(f"⚠️  {name:6s}: File not found")

    return datasets


def analyze_basic_info(datasets):
    """Basic dataset information."""
    print("\n" + "=" * 80)
    print("📊 DATASET OVERVIEW")
    print("=" * 80)

    df_full = datasets.get("full")
    if df_full is None:
        print("⚠️  Full dataset not available")
        return

    print(f"\nTotal events: {len(df_full):,}")
    print(f"Total columns: {len(df_full.columns)}")
    print(f"Memory usage: {df_full.memory_usage(deep=True).sum() / 1024 ** 2:.1f} MB")
    print(f"Date range: {df_full['timestamp_utc'].min()} to {df_full['timestamp_utc'].max()}")

    # Data types
    print("\n📋 Column Types:")
    dtype_counts = df_full.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"  {str(dtype):20s}: {count:3d} columns")


def analyze_columns(datasets):
    """Detailed column analysis."""
    print("\n" + "=" * 80)
    print("📋 COLUMN DETAILS")
    print("=" * 80)

    df_full = datasets.get("full")
    if df_full is None:
        return

    # Group columns by category
    categories = {
        "Event Identification": ["event_id", "event_type", "event_source", "timestamp"],
        "User/Account": ["user_id", "account_id", "account_type", "user_segment", "user_risk_class"],
        "Temporal": ["hour_of_day", "day_of_week", "is_weekend", "seconds_since"],
        "Transaction": ["amount", "currency", "channel", "recipient", "merchant"],
        "Location/Network": ["ip_address", "country", "city", "asn", "isp", "vpn", "proxy", "tor"],
        "Device": ["device_id", "os_name", "os_version", "browser", "user_agent", "screen_resolution"],
        "Security": ["mfa_enabled", "biometric", "tls_version", "waf", "av_version", "firewall"],
        "Fraud/Risk": ["is_fraud", "fraud_type", "fraud_score", "risk_score", "blacklist"],
        "Text/Message": ["message_text", "message_length", "message_language", "contains_url"],
        "LLM Features": ["llm_risk_score", "llm_sentiment", "llm_urgency", "llm_phishing"]
    }

    for category, keywords in categories.items():
        matching_cols = [col for col in df_full.columns if any(kw in col.lower() for kw in keywords)]
        if matching_cols:
            print(f"\n{category} ({len(matching_cols)} columns):")
            for col in matching_cols[:10]:  # Show first 10
                dtype = df_full[col].dtype
                nulls = df_full[col].isna().sum()
                null_pct = (nulls / len(df_full)) * 100
                unique = df_full[col].nunique()
                print(f"  • {col:40s} | {str(dtype):15s} | Nulls: {null_pct:5.1f}% | Unique: {unique:>8,}")

            if len(matching_cols) > 10:
                print(f"  ... and {len(matching_cols) - 10} more")


def analyze_fraud_distribution(datasets):
    """Analyze fraud distribution."""
    print("\n" + "=" * 80)
    print("🎯 FRAUD DISTRIBUTION")
    print("=" * 80)

    for name, df in datasets.items():
        if "is_fraud" not in df.columns:
            continue

        fraud_count = df["is_fraud"].sum()
        fraud_rate = df["is_fraud"].mean()

        print(f"\n{name.upper()}:")
        print(f"  Total events:      {len(df):>10,}")
        print(f"  Fraudulent:        {fraud_count:>10,} ({fraud_rate:>6.2%})")
        print(f"  Legitimate:        {len(df) - fraud_count:>10,} ({1 - fraud_rate:>6.2%})")

        # Fraud types
        if "fraud_type" in df.columns:
            print(f"\n  Fraud Types:")
            fraud_types = df[df["is_fraud"] == True]["fraud_type"].value_counts()
            for fraud_type, count in fraud_types.items():
                pct = (count / fraud_count) * 100
                print(f"    • {fraud_type:30s}: {count:>8,} ({pct:>5.1f}%)")


def analyze_temporal_patterns(datasets):
    """Analyze temporal patterns."""
    print("\n" + "=" * 80)
    print("⏰ TEMPORAL PATTERNS")
    print("=" * 80)

    df_full = datasets.get("full")
    if df_full is None or "timestamp_utc" not in df_full.columns:
        return

    df_full["hour"] = pd.to_datetime(df_full["timestamp_utc"]).dt.hour
    df_full["day"] = pd.to_datetime(df_full["timestamp_utc"]).dt.day_name()

    # Events by hour
    print("\nEvents by Hour of Day:")
    hourly = df_full.groupby("hour").size()
    for hour, count in hourly.items():
        bar = "█" * int(count / hourly.max() * 50)
        print(f"  {hour:02d}:00 | {bar} {count:,}")

    # Events by day
    print("\nEvents by Day of Week:")
    daily = df_full["day"].value_counts()
    for day, count in daily.items():
        bar = "█" * int(count / daily.max() * 50)
        print(f"  {day:10s} | {bar} {count:,}")


def analyze_user_behavior(datasets):
    """Analyze user behavior patterns."""
    print("\n" + "=" * 80)
    print("👤 USER BEHAVIOR")
    print("=" * 80)

    df_full = datasets.get("full")
    if df_full is None or "user_id" not in df_full.columns:
        return

    # Events per user
    events_per_user = df_full.groupby("user_id").size()
    print(f"\nEvents per User:")
    print(f"  Mean:   {events_per_user.mean():>10.1f}")
    print(f"  Median: {events_per_user.median():>10.1f}")
    print(f"  Min:    {events_per_user.min():>10,}")
    print(f"  Max:    {events_per_user.max():>10,}")

    # User risk classes
    if "user_risk_class" in df_full.columns:
        print(f"\nUser Risk Classes:")
        risk_classes = df_full["user_risk_class"].value_counts()
        for risk_class, count in risk_classes.items():
            pct = (count / len(df_full)) * 100
            print(f"  • {risk_class:15s}: {count:>10,} ({pct:>5.1f}%)")

    # User segments
    if "user_segment" in df_full.columns:
        print(f"\nUser Segments:")
        segments = df_full["user_segment"].value_counts()
        for segment, count in segments.items():
            pct = (count / len(df_full)) * 100
            print(f"  • {segment:15s}: {count:>10,} ({pct:>5.1f}%)")


def analyze_transaction_patterns(datasets):
    """Analyze transaction patterns."""
    print("\n" + "=" * 80)
    print("💰 TRANSACTION PATTERNS")
    print("=" * 80)

    df_full = datasets.get("full")
    if df_full is None or "amount" not in df_full.columns:
        return

    # Filter transactions only
    transactions = df_full[df_full["event_type"] == "transaction"]

    if len(transactions) == 0:
        print("⚠️  No transactions found")
        return

    print(f"\nTransaction Amounts:")
    print(f"  Count:  {len(transactions):>12,}")
    print(f"  Mean:   ${transactions['amount'].mean():>12,.2f}")
    print(f"  Median: ${transactions['amount'].median():>12,.2f}")
    print(f"  Min:    ${transactions['amount'].min():>12,.2f}")
    print(f"  Max:    ${transactions['amount'].max():>12,.2f}")
    print(f"  Std:    ${transactions['amount'].std():>12,.2f}")

    # Percentiles
    print(f"\nPercentiles:")
    for p in [10, 25, 50, 75, 90, 95, 99]:
        val = transactions["amount"].quantile(p / 100)
        print(f"  {p:2d}th: ${val:>12,.2f}")

    # Channels
    if "channel" in transactions.columns:
        print(f"\nTransaction Channels:")
        channels = transactions["channel"].value_counts()
        for channel, count in channels.items():
            pct = (count / len(transactions)) * 100
            print(f"  • {channel:20s}: {count:>10,} ({pct:>5.1f}%)")


def analyze_llm_features(datasets):
    """Analyze LLM-derived features."""
    print("\n" + "=" * 80)
    print("🤖 LLM FEATURES ANALYSIS")
    print("=" * 80)

    df_full = datasets.get("full")
    if df_full is None:
        return

    llm_cols = [c for c in df_full.columns if "llm_" in c.lower()]

    if not llm_cols:
        print("⚠️  No LLM features found (columns starting with 'llm_')")
        return

    print(f"\nFound {len(llm_cols)} LLM features:")

    for col in llm_cols:
        non_null = df_full[col].notna().sum()
        non_null_pct = (non_null / len(df_full)) * 100

        print(f"\n  {col}:")
        print(f"    Non-null: {non_null:>10,} ({non_null_pct:>5.1f}%)")

        if non_null > 0:
            if df_full[col].dtype in [np.float64, np.int64]:
                print(f"    Mean:     {df_full[col].mean():>10.4f}")
                print(f"    Median:   {df_full[col].median():>10.4f}")
                print(f"    Min:      {df_full[col].min():>10.4f}")
                print(f"    Max:      {df_full[col].max():>10.4f}")
            else:
                unique = df_full[col].nunique()
                print(f"    Unique:   {unique:>10,}")


def analyze_missing_data(datasets):
    """Analyze missing data patterns."""
    print("\n" + "=" * 80)
    print("❓ MISSING DATA ANALYSIS")
    print("=" * 80)

    df_full = datasets.get("full")
    if df_full is None:
        return

    missing = df_full.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if len(missing) == 0:
        print("\n✅ No missing data found!")
        return

    print(f"\nColumns with missing data ({len(missing)} total):")

    for col, count in missing.head(20).items():
        pct = (count / len(df_full)) * 100
        bar = "█" * int(pct / 2)
        print(f"  {col:40s} | {bar:50s} {count:>10,} ({pct:>5.1f}%)")

    if len(missing) > 20:
        print(f"\n  ... and {len(missing) - 20} more columns with missing data")


def generate_visualizations(datasets):
    """Generate visualization plots."""
    print("\n" + "=" * 80)
    print("📊 GENERATING VISUALIZATIONS")
    print("=" * 80)

    df_full = datasets.get("full")
    if df_full is None:
        return

    # 1. Fraud distribution
    if "is_fraud" in df_full.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        fraud_counts = df_full["is_fraud"].value_counts()
        ax.bar(["Legitimate", "Fraudulent"], fraud_counts.values, color=["green", "red"], alpha=0.7)
        ax.set_ylabel("Count")
        ax.set_title("Fraud vs Legitimate Events")
        for i, v in enumerate(fraud_counts.values):
            ax.text(i, v + len(df_full) * 0.01, f"{v:,}\n({v / len(df_full):.1%})", ha="center")
        plt.tight_layout()
        plt.savefig(REPORT_DIR / "fraud_distribution.png", dpi=150)
        plt.close()
        print("  ✅ fraud_distribution.png")

    # 2. Fraud types
    if "fraud_type" in df_full.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        fraud_types = df_full[df_full["is_fraud"] == True]["fraud_type"].value_counts()
        fraud_types.plot(kind="barh", ax=ax, color="coral")
        ax.set_xlabel("Count")
        ax.set_title("Fraud Types Distribution")
        plt.tight_layout()
        plt.savefig(REPORT_DIR / "fraud_types.png", dpi=150)
        plt.close()
        print("  ✅ fraud_types.png")

    # 3. Transaction amounts
    if "amount" in df_full.columns:
        transactions = df_full[df_full["event_type"] == "transaction"]
        if len(transactions) > 0:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # Histogram
            axes[0].hist(transactions["amount"], bins=50, color="skyblue", edgecolor="black", alpha=0.7)
            axes[0].set_xlabel("Amount")
            axes[0].set_ylabel("Frequency")
            axes[0].set_title("Transaction Amount Distribution")

            # Box plot by fraud status
            if "is_fraud" in transactions.columns:
                transactions.boxplot(column="amount", by="is_fraud", ax=axes[1])
                axes[1].set_xlabel("Is Fraud")
                axes[1].set_ylabel("Amount")
                axes[1].set_title("Amount by Fraud Status")
                plt.suptitle("")

            plt.tight_layout()
            plt.savefig(REPORT_DIR / "transaction_amounts.png", dpi=150)
            plt.close()
            print("  ✅ transaction_amounts.png")

    # 4. Temporal patterns
    if "timestamp_utc" in df_full.columns:
        df_full["hour"] = pd.to_datetime(df_full["timestamp_utc"]).dt.hour

        fig, ax = plt.subplots(figsize=(12, 6))
        hourly = df_full.groupby("hour").size()
        ax.plot(hourly.index, hourly.values, marker="o", linewidth=2, markersize=6)
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Number of Events")
        ax.set_title("Events by Hour of Day")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(REPORT_DIR / "temporal_patterns.png", dpi=150)
        plt.close()
        print("  ✅ temporal_patterns.png")

    # 5. Correlation heatmap (numeric features only)
    numeric_cols = df_full.select_dtypes(include=[np.number]).columns[:20]  # First 20 numeric
    if len(numeric_cols) > 1:
        fig, ax = plt.subplots(figsize=(12, 10))
        corr = df_full[numeric_cols].corr()
        sns.heatmap(corr, annot=False, cmap="coolwarm", center=0, ax=ax, square=True)
        ax.set_title("Feature Correlation Heatmap (Top 20 Numeric Features)")
        plt.tight_layout()
        plt.savefig(REPORT_DIR / "correlation_heatmap.png", dpi=150)
        plt.close()
        print("  ✅ correlation_heatmap.png")

    print(f"\n📁 Visualizations saved to: {REPORT_DIR}/")


def export_summary_report(datasets):
    """Export summary report to text file."""
    print("\n" + "=" * 80)
    print("📄 EXPORTING SUMMARY REPORT")
    print("=" * 80)

    df_full = datasets.get("full")
    if df_full is None:
        return

    report_path = REPORT_DIR / "dataset_summary.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("FRAUD DETECTION DATASET - SUMMARY REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Total Events: {len(df_full):,}\n")
        f.write(f"Total Columns: {len(df_full.columns)}\n")
        f.write(f"Memory Usage: {df_full.memory_usage(deep=True).sum() / 1024 ** 2:.1f} MB\n\n")

        f.write("Dataset Splits:\n")
        for name, df in datasets.items():
            f.write(f"  {name:6s}: {len(df):>10,} rows\n")

        f.write("\nFraud Distribution:\n")
        if "is_fraud" in df_full.columns:
            fraud_count = df_full["is_fraud"].sum()
            fraud_rate = df_full["is_fraud"].mean()
            f.write(f"  Fraudulent: {fraud_count:>10,} ({fraud_rate:>6.2%})\n")
            f.write(f"  Legitimate: {len(df_full) - fraud_count:>10,} ({1 - fraud_rate:>6.2%})\n")

        f.write("\nColumn List:\n")
        for i, col in enumerate(df_full.columns, 1):
            f.write(f"  {i:3d}. {col}\n")

    print(f"  ✅ Summary report saved to: {report_path}")


def main():
    """Main execution."""
    print("\n" + "🔍" * 40)
    print("FRAUD DETECTION DATASET EXPLORER")
    print("🔍" * 40 + "\n")

    # Load datasets
    datasets = load_datasets()

    if not datasets:
        print("\n❌ No datasets found in output/ directory")
        return

    # Run analyses
    analyze_basic_info(datasets)
    analyze_columns(datasets)
    analyze_fraud_distribution(datasets)
    analyze_temporal_patterns(datasets)
    analyze_user_behavior(datasets)
    analyze_transaction_patterns(datasets)
    analyze_llm_features(datasets)
    analyze_missing_data(datasets)

    # Generate visualizations
    generate_visualizations(datasets)

    # Export report
    export_summary_report(datasets)

    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\n📁 Reports and visualizations saved to: {REPORT_DIR}/")
    print("\nGenerated files:")
    print("  • fraud_distribution.png")
    print("  • fraud_types.png")
    print("  • transaction_amounts.png")
    print("  • temporal_patterns.png")
    print("  • correlation_heatmap.png")
    print("  • dataset_summary.txt")
    print("\n")


if __name__ == "__main__":
    main()