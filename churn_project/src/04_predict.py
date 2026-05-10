"""
================================================================================
  CUSTOMER CHURN PREDICTION — STEP 4: PREDICTION INTERFACE
================================================================================
  Usage:
    python src/04_predict.py                         # batch predict on test set
    python src/04_predict.py --single                # predict a single customer
    python src/04_predict.py --csv data/new_customers.csv  # predict from CSV
================================================================================
"""

import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd

MODEL_PATH = "models/best_model_rf.pkl"
DATA_CLEAN = "data/customer_churn_clean.csv"

def load_model():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found at {MODEL_PATH}")
        print("   Run: python src/03_model_training.py  first.")
        sys.exit(1)
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

def predict_customers(X: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    model   = bundle["model"]
    cols    = bundle["feature_cols"]

    # Align columns (fill missing with 0)
    for c in cols:
        if c not in X.columns:
            X[c] = 0
    X = X[cols]

    proba  = model.predict_proba(X)[:, 1]
    pred   = (proba >= 0.5).astype(int)
    risk   = pd.cut(proba, bins=[0, 0.3, 0.6, 1.0],
                    labels=["Low Risk", "Medium Risk", "High Risk"])

    result = pd.DataFrame({
        "Churn_Prediction": pred,
        "Churn_Label":      np.where(pred == 1, "YES — Will Churn", "NO  — Will Stay"),
        "Churn_Probability": proba.round(4),
        "Risk_Category":    risk,
    })
    return result

def batch_predict():
    """Predict on the last 20% (simulated test batch)."""
    bundle = load_model()
    df     = pd.read_csv(DATA_CLEAN)
    X      = df.drop(columns=["Churn"])
    y_true = df["Churn"]

    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(X, y_true, test_size=0.20,
                                             random_state=42, stratify=y_true)

    results = predict_customers(X_test.reset_index(drop=True), bundle)
    combined = pd.concat([X_test.reset_index(drop=True), results], axis=1)
    combined["Actual_Churn"] = y_test.reset_index(drop=True)

    out = "outputs/batch_predictions.csv"
    combined.to_csv(out, index=False)

    print("=" * 55)
    print("  BATCH PREDICTION RESULTS")
    print("=" * 55)
    print(f"  Records evaluated : {len(results):,}")
    print(f"  Predicted churners: {results['Churn_Prediction'].sum():,}")
    print(f"  Predicted stayers : {(results['Churn_Prediction']==0).sum():,}")
    print(f"\n  Risk breakdown:")
    for cat in ["Low Risk", "Medium Risk", "High Risk"]:
        n = (results["Risk_Category"] == cat).sum()
        print(f"    {cat:15s}: {n:,}  ({n/len(results):.1%})")
    print(f"\n  ✅ Full predictions saved → {out}")
    print("=" * 55)
    return combined

def single_predict():
    """Interactively predict churn for one customer."""
    bundle = load_model()
    df     = pd.read_csv(DATA_CLEAN)
    sample = df.drop(columns=["Churn"]).iloc[0:1].copy()

    print("\n" + "=" * 55)
    print("  SINGLE CUSTOMER CHURN PREDICTOR")
    print("=" * 55)
    print("  (Press Enter to keep example values)\n")

    def ask(prompt, example):
        val = input(f"  {prompt} [{example}]: ").strip()
        return val if val else str(example)

    # Build a fresh row using the cleaned feature set
    # (simplified: set known high-risk vs low-risk profiles)
    print("  Choose a profile to test:")
    print("  1 — High risk  (Month-to-month, Fiber, short tenure)")
    print("  2 — Low risk   (Two year, DSL, long tenure)")
    choice = input("  Enter 1 or 2 [1]: ").strip() or "1"

    row = pd.DataFrame(0, index=[0], columns=bundle["feature_cols"])

    if choice == "2":
        row["tenure"]              = 48
        row["MonthlyCharges"]      = 45.0
        row["TotalCharges"]        = 2160.0
        row["avg_charge_per_month"]= 45.0
        row["tenure_group"]        = 3
        row["Partner"]             = 1
        row["Dependents"]          = 1
        row.get("Contract_Two year") # may exist
        if "Contract_Two year" in row.columns:
            row["Contract_Two year"]   = 1
        if "InternetService_DSL" in row.columns:
            row["InternetService_DSL"] = 1
        print("\n  Profile: Long-tenure, two-year contract customer")
    else:
        row["tenure"]              = 3
        row["MonthlyCharges"]      = 89.0
        row["TotalCharges"]        = 267.0
        row["avg_charge_per_month"]= 89.0
        row["tenure_group"]        = 0
        if "Contract_Month-to-month" in row.columns:
            row["Contract_Month-to-month"] = 1
        if "InternetService_Fiber optic" in row.columns:
            row["InternetService_Fiber optic"] = 1
        if "PaymentMethod_Electronic check" in row.columns:
            row["PaymentMethod_Electronic check"] = 1
        print("\n  Profile: New Fiber customer, month-to-month, e-check")

    result = predict_customers(row, bundle)
    print("\n" + "─" * 55)
    print(f"  Prediction        : {result['Churn_Label'].iloc[0]}")
    print(f"  Churn Probability : {result['Churn_Probability'].iloc[0]:.1%}")
    print(f"  Risk Category     : {result['Risk_Category'].iloc[0]}")
    print("─" * 55)

# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Churn Predictor")
    parser.add_argument("--single", action="store_true", help="Predict single customer")
    parser.add_argument("--csv",    type=str,            help="Path to input CSV")
    args = parser.parse_args()

    if args.single:
        single_predict()
    elif args.csv:
        bundle = load_model()
        df_in  = pd.read_csv(args.csv)
        result = predict_customers(df_in, bundle)
        out    = args.csv.replace(".csv", "_predictions.csv")
        pd.concat([df_in, result], axis=1).to_csv(out, index=False)
        print(f"✅ Predictions saved → {out}")
    else:
        batch_predict()
