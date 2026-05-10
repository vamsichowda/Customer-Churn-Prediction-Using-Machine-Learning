"""
================================================================================
  CUSTOMER CHURN PREDICTION — STEP 1: DATA PREPROCESSING
================================================================================
  Tasks:
    1. Load raw dataset
    2. Inspect shape, dtypes, and basic statistics
    3. Handle missing values
    4. Fix data type issues
    5. Encode categorical variables
    6. Feature engineering
    7. Save cleaned dataset for next steps
================================================================================
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ── Path setup ────────────────────────────────────────────────────────────────
DATA_RAW    = "data/customer_churn.csv"
DATA_CLEAN  = "data/customer_churn_clean.csv"
os.makedirs("data", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

print("=" * 65)
print("  STEP 1: DATA PREPROCESSING")
print("=" * 65)

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1] Loading raw dataset...")
df = pd.read_csv(DATA_RAW)
print(f"    Shape     : {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"    Memory    : {df.memory_usage(deep=True).sum() / 1024:.1f} KB")

# ══════════════════════════════════════════════════════════════════════════════
# 2. INITIAL INSPECTION
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2] Column overview:")
type_summary = df.dtypes.reset_index()
type_summary.columns = ["Column", "dtype"]
type_summary["unique"] = [df[c].nunique() for c in df.columns]
type_summary["nulls"]  = [df[c].isna().sum() for c in df.columns]
print(type_summary.to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# 3. HANDLE MISSING VALUES
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3] Missing values:")
null_counts = df.isnull().sum()
null_pct    = (null_counts / len(df) * 100).round(2)
missing     = pd.DataFrame({"count": null_counts, "pct%": null_pct})
missing     = missing[missing["count"] > 0]

if missing.empty:
    print("    ✅ No missing values found.")
else:
    print(missing)
    # TotalCharges is missing for tenure==0 customers (no charges yet)
    # Strategy: fill with MonthlyCharges (first month approximation)
    mask = df["TotalCharges"].isna()
    df.loc[mask, "TotalCharges"] = df.loc[mask, "MonthlyCharges"]
    print(f"\n    ✅ Filled {mask.sum()} missing TotalCharges with MonthlyCharges.")

# ══════════════════════════════════════════════════════════════════════════════
# 4. FIX DATA TYPES
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4] Fixing data types...")

# TotalCharges may be read as object if it contained whitespace → cast to float
if df["TotalCharges"].dtype == object:
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["MonthlyCharges"], inplace=True)
    print("    Fixed TotalCharges → float64")

# SeniorCitizen is 0/1 int — keep as-is but note it's binary
print("    SeniorCitizen already numeric (0/1) ✅")

# Drop customerID — identifier, not a predictor
df.drop(columns=["customerID"], inplace=True)
print("    Dropped customerID column.")

# ══════════════════════════════════════════════════════════════════════════════
# 5. REMOVE DUPLICATES
# ══════════════════════════════════════════════════════════════════════════════
dupes = df.duplicated().sum()
print(f"\n[5] Duplicate rows: {dupes}")
if dupes:
    df.drop_duplicates(inplace=True)
    print(f"    Removed {dupes} duplicates.")
else:
    print("    ✅ No duplicates found.")

# ══════════════════════════════════════════════════════════════════════════════
# 6. ENCODE TARGET VARIABLE
# ══════════════════════════════════════════════════════════════════════════════
print("\n[6] Encoding target variable (Churn)...")
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
churn_rate  = df["Churn"].mean()
print(f"    Churn = 1 (Yes): {df['Churn'].sum():,}  ({churn_rate:.1%})")
print(f"    Churn = 0 (No) : {(df['Churn']==0).sum():,}  ({1-churn_rate:.1%})")

# ══════════════════════════════════════════════════════════════════════════════
# 7. ENCODE BINARY CATEGORICAL COLUMNS (Yes/No → 1/0)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[7] Encoding binary Yes/No columns...")
binary_cols = [
    "Partner", "Dependents", "PhoneService", "PaperlessBilling",
]
for col in binary_cols:
    df[col] = df[col].map({"Yes": 1, "No": 0})
    print(f"    {col}: Yes→1, No→0")

# gender: Male→1, Female→0
df["gender"] = df["gender"].map({"Male": 1, "Female": 0})
print("    gender: Male→1, Female→0")

# ══════════════════════════════════════════════════════════════════════════════
# 8. ENCODE MULTI-CLASS CATEGORICAL COLUMNS (One-Hot Encoding)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[8] One-hot encoding multi-class categoricals...")
multi_cols = [
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaymentMethod",
]
df_encoded = pd.get_dummies(df, columns=multi_cols, drop_first=False)
df_encoded = df_encoded.astype({col: int for col in df_encoded.select_dtypes("bool").columns})

new_cols = set(df_encoded.columns) - set(df.columns)
print(f"    Added {len(new_cols)} dummy columns (from {len(multi_cols)} original).")
print(f"    Final shape: {df_encoded.shape[0]:,} rows × {df_encoded.shape[1]} columns")

# ══════════════════════════════════════════════════════════════════════════════
# 9. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
print("\n[9] Feature engineering...")

# Tenure buckets
bins   = [0, 6, 12, 24, 48, 72]
labels = [0, 1, 2, 3, 4]
df_encoded["tenure_group"] = pd.cut(df_encoded["tenure"], bins=bins, labels=labels, include_lowest=True).astype(int)
print("    tenure_group: 5 bins (0-6, 7-12, 13-24, 25-48, 49-72 months)")

# Average monthly spend proxy
df_encoded["avg_charge_per_month"] = np.where(
    df_encoded["tenure"] > 0,
    (df_encoded["TotalCharges"] / df_encoded["tenure"]).round(2),
    df_encoded["MonthlyCharges"]
)
print("    avg_charge_per_month: TotalCharges / tenure")

# High-value customer flag
charge_75th = df_encoded["MonthlyCharges"].quantile(0.75)
df_encoded["high_value"]  = (df_encoded["MonthlyCharges"] >= charge_75th).astype(int)
print(f"    high_value: 1 if MonthlyCharges ≥ {charge_75th:.2f} (75th pct)")

print(f"\n    Final feature count: {df_encoded.shape[1] - 1} (excl. target)")

# ══════════════════════════════════════════════════════════════════════════════
# 10. SAVE CLEAN DATASET
# ══════════════════════════════════════════════════════════════════════════════
df_encoded.to_csv(DATA_CLEAN, index=False)
print(f"\n[10] ✅ Clean dataset saved → {DATA_CLEAN}")
print(f"     Shape: {df_encoded.shape[0]:,} rows × {df_encoded.shape[1]} columns")

# Save a summary report
summary = df_encoded.describe(include="all").T
summary.to_csv("outputs/preprocessing_summary.csv")
print("     Summary stats → outputs/preprocessing_summary.csv")

print("\n" + "=" * 65)
print("  Preprocessing complete. Run 02_eda.py next.")
print("=" * 65)
