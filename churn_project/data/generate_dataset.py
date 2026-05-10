"""
Generate a realistic Telco Customer Churn dataset (~7,000 rows).
Run once to produce: data/customer_churn.csv
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 7043

# ── Demographics ──────────────────────────────────────────────
gender         = np.random.choice(["Male", "Female"], N)
senior_citizen = np.random.choice([0, 1], N, p=[0.84, 0.16])
partner        = np.random.choice(["Yes", "No"], N)
dependents     = np.where(
    partner == "Yes",
    np.random.choice(["Yes", "No"], N, p=[0.45, 0.55]),
    np.random.choice(["Yes", "No"], N, p=[0.17, 0.83]),
)

# ── Account info ──────────────────────────────────────────────
tenure = np.random.choice(range(0, 73), N)

contract = np.where(
    tenure < 12,
    np.random.choice(["Month-to-month", "One year", "Two year"], N, p=[0.75, 0.15, 0.10]),
    np.where(
        tenure < 36,
        np.random.choice(["Month-to-month", "One year", "Two year"], N, p=[0.40, 0.35, 0.25]),
        np.random.choice(["Month-to-month", "One year", "Two year"], N, p=[0.20, 0.30, 0.50]),
    ),
)

paperless_billing = np.random.choice(["Yes", "No"], N, p=[0.59, 0.41])
payment_method    = np.random.choice(
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    N, p=[0.34, 0.23, 0.22, 0.21],
)

# ── Services ──────────────────────────────────────────────────
phone_service    = np.random.choice(["Yes", "No"], N, p=[0.90, 0.10])
multiple_lines   = np.where(phone_service == "No", "No phone service",
                            np.random.choice(["Yes", "No"], N, p=[0.50, 0.50]))
internet_service = np.random.choice(["DSL", "Fiber optic", "No"], N, p=[0.34, 0.44, 0.22])

def internet_addon(no_label):
    return np.where(
        internet_service == "No", no_label,
        np.random.choice(["Yes", "No"], N, p=[0.45, 0.55]),
    )

online_security  = internet_addon("No internet service")
online_backup    = internet_addon("No internet service")
device_protection= internet_addon("No internet service")
tech_support     = internet_addon("No internet service")
streaming_tv     = internet_addon("No internet service")
streaming_movies = internet_addon("No internet service")

# ── Charges ───────────────────────────────────────────────────
base_charge = np.where(internet_service == "No", 20,
              np.where(internet_service == "DSL", 45, 70))
addon_charge = (
    (multiple_lines == "Yes").astype(float) * 10 +
    (online_security == "Yes").astype(float) * 8 +
    (online_backup == "Yes").astype(float) * 8 +
    (device_protection == "Yes").astype(float) * 8 +
    (tech_support == "Yes").astype(float) * 8 +
    (streaming_tv == "Yes").astype(float) * 8 +
    (streaming_movies == "Yes").astype(float) * 8
)
monthly_charges = (base_charge + addon_charge + np.random.normal(0, 3, N)).clip(18.25, 118.75)
monthly_charges = monthly_charges.round(2)

total_charges = (monthly_charges * tenure + np.random.normal(0, 10, N)).clip(0)
total_charges  = np.where(tenure == 0, np.nan, total_charges.round(2))  # intentional missing

# ── Churn (target) ────────────────────────────────────────────
# Build churn probability based on known risk factors
churn_prob = 0.10
churn_prob += np.where(contract == "Month-to-month", 0.22, 0)
churn_prob += np.where(contract == "One year",       0.06, 0)
churn_prob += np.where(internet_service == "Fiber optic", 0.12, 0)
churn_prob += np.where(payment_method == "Electronic check", 0.08, 0)
churn_prob += np.where(senior_citizen == 1, 0.07, 0)
churn_prob += np.where(tenure < 6, 0.15, 0)
churn_prob += np.where(tenure > 48, -0.12, 0)
churn_prob += np.where(tech_support == "No", 0.05, 0)
churn_prob += np.where(online_security == "No", 0.05, 0)
churn_prob += np.where(partner == "No", 0.04, 0)
churn_prob  = churn_prob.clip(0.01, 0.97)

churn_raw   = np.random.binomial(1, churn_prob, N)
churn       = np.where(churn_raw == 1, "Yes", "No")

customer_id = [f"TLC-{str(i).zfill(5)}" for i in range(1, N + 1)]

df = pd.DataFrame({
    "customerID":        customer_id,
    "gender":            gender,
    "SeniorCitizen":     senior_citizen,
    "Partner":           partner,
    "Dependents":        dependents,
    "tenure":            tenure,
    "PhoneService":      phone_service,
    "MultipleLines":     multiple_lines,
    "InternetService":   internet_service,
    "OnlineSecurity":    online_security,
    "OnlineBackup":      online_backup,
    "DeviceProtection":  device_protection,
    "TechSupport":       tech_support,
    "StreamingTV":       streaming_tv,
    "StreamingMovies":   streaming_movies,
    "Contract":          contract,
    "PaperlessBilling":  paperless_billing,
    "PaymentMethod":     payment_method,
    "MonthlyCharges":    monthly_charges,
    "TotalCharges":      total_charges,
    "Churn":             churn,
})

out = "data/customer_churn.csv"
df.to_csv(out, index=False)
print(f"✅ Dataset saved → {out}")
print(f"   Rows: {len(df):,}  |  Churn rate: {(df['Churn']=='Yes').mean():.1%}")
print(f"   Missing TotalCharges: {df['TotalCharges'].isna().sum()}")
