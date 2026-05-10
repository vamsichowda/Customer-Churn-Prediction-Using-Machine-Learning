"""
================================================================================
  CUSTOMER CHURN PREDICTION — STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
================================================================================
  Generates 6 analysis charts saved to outputs/eda/
  Analyses:
    1. Churn distribution (pie chart)
    2. Churn by Contract type (bar)
    3. Churn by Internet Service (bar)
    4. Tenure distribution by churn (histogram)
    5. Monthly Charges distribution (KDE + box)
    6. Correlation heatmap of numeric features
================================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ── Style setup ───────────────────────────────────────────────────────────────
NAVY    = "#0D1B2A"
TEAL    = "#00B4D8"
TEAL2   = "#0077B6"
RED     = "#EF4444"
AMBER   = "#F59E0B"
GREEN   = "#22C55E"
GRAY    = "#94A3B8"
LIGHT   = "#F0F7FA"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "#F8FAFC",
    "axes.edgecolor":   "#CBD5E1",
    "axes.labelcolor":  NAVY,
    "axes.titlesize":   14,
    "axes.titleweight": "bold",
    "axes.titlecolor":  NAVY,
    "xtick.color":      "#475569",
    "ytick.color":      "#475569",
    "text.color":       NAVY,
    "font.family":      "DejaVu Sans",
    "grid.color":       "#E2E8F0",
    "grid.linestyle":   "--",
    "grid.linewidth":   0.6,
})

DATA_CLEAN = "data/customer_churn.csv"  # use raw for readable labels in EDA
OUT_DIR    = "outputs/eda"
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 65)
print("  STEP 2: EXPLORATORY DATA ANALYSIS")
print("=" * 65)

# Load raw data (labels intact for readable charts)
df = pd.read_csv(DATA_CLEAN)

# Fix TotalCharges
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].fillna(df["MonthlyCharges"], inplace=True)

print(f"\n    Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
churn_counts = df["Churn"].value_counts()
print(f"    Churn — Yes: {churn_counts.get('Yes',0):,}  |  No: {churn_counts.get('No',0):,}")

CHURN_COLORS = {
    "Yes": RED,
    "No":  TEAL,
    1: RED,
    0: TEAL,
}

# ══════════════════════════════════════════════════════════════════════════════
# CHART 1 — CHURN DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1] Churn distribution chart...")
fig, ax = plt.subplots(figsize=(7, 5))
labels  = ["No Churn", "Churned"]
values  = [churn_counts.get("No", 0), churn_counts.get("Yes", 0)]
colors  = [TEAL, RED]
explode = [0, 0.06]
wedges, texts, autotexts = ax.pie(
    values, labels=labels, colors=colors, explode=explode,
    autopct="%1.1f%%", startangle=140,
    textprops={"fontsize": 13, "color": NAVY},
    wedgeprops={"edgecolor": "white", "linewidth": 2},
)
for at in autotexts:
    at.set_fontsize(13)
    at.set_fontweight("bold")
    at.set_color("white")
ax.set_title("Customer Churn Distribution", pad=16, fontsize=16)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/01_churn_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("    ✅ Saved: 01_churn_distribution.png")

# ══════════════════════════════════════════════════════════════════════════════
# CHART 2 — CHURN BY CONTRACT TYPE
# ══════════════════════════════════════════════════════════════════════════════
print("[2] Churn by Contract type...")
ct  = df.groupby(["Contract", "Churn"]).size().unstack(fill_value=0)
ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: counts
ct.plot(kind="bar", ax=axes[0], color=[TEAL, RED], edgecolor="white", width=0.6)
axes[0].set_title("Churn Count by Contract Type")
axes[0].set_xlabel("Contract Type")
axes[0].set_ylabel("Number of Customers")
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=25, ha="right")
axes[0].legend(["No Churn", "Churned"], framealpha=0.8)
axes[0].yaxis.grid(True)
axes[0].set_axisbelow(True)
for p in axes[0].patches:
    if p.get_height() > 0:
        axes[0].annotate(f"{int(p.get_height()):,}", (p.get_x() + p.get_width()/2, p.get_height()),
                         ha="center", va="bottom", fontsize=9, color=NAVY)

# Right: churn rate %
ct_pct["Yes"].plot(kind="bar", ax=axes[1], color=RED, edgecolor="white", width=0.5)
axes[1].set_title("Churn Rate (%) by Contract Type")
axes[1].set_xlabel("Contract Type")
axes[1].set_ylabel("Churn Rate (%)")
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=25, ha="right")
axes[1].yaxis.grid(True)
axes[1].set_axisbelow(True)
for p in axes[1].patches:
    axes[1].annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width()/2, p.get_height()),
                     ha="center", va="bottom", fontsize=10, fontweight="bold", color=NAVY)

fig.suptitle("Contract Type vs Churn", fontsize=15, fontweight="bold", color=NAVY, y=1.01)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/02_churn_by_contract.png", dpi=150, bbox_inches="tight")
plt.close()
print("    ✅ Saved: 02_churn_by_contract.png")

# ══════════════════════════════════════════════════════════════════════════════
# CHART 3 — CHURN BY INTERNET SERVICE
# ══════════════════════════════════════════════════════════════════════════════
print("[3] Churn by Internet Service...")
isg = df.groupby(["InternetService", "Churn"]).size().unstack(fill_value=0)
isg_pct = isg.div(isg.sum(axis=1), axis=0) * 100

fig, ax = plt.subplots(figsize=(8, 5))
x     = np.arange(len(isg.index))
width = 0.35
b1 = ax.bar(x - width/2, isg["No"],  width, label="No Churn", color=TEAL, edgecolor="white")
b2 = ax.bar(x + width/2, isg.get("Yes", pd.Series(0, index=isg.index)),
            width, label="Churned", color=RED, edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels(isg.index, fontsize=11)
ax.set_title("Churn by Internet Service Type")
ax.set_xlabel("Internet Service")
ax.set_ylabel("Number of Customers")
ax.legend(framealpha=0.8)
ax.yaxis.grid(True)
ax.set_axisbelow(True)

# Add churn rate labels above red bars
for i, bar in enumerate(b2):
    rate = isg_pct.get("Yes", pd.Series(0, index=isg.index)).iloc[i]
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
            f"{rate:.1f}%", ha="center", va="bottom", fontsize=9.5,
            color=RED, fontweight="bold")

fig.tight_layout()
fig.savefig(f"{OUT_DIR}/03_churn_by_internet.png", dpi=150, bbox_inches="tight")
plt.close()
print("    ✅ Saved: 03_churn_by_internet.png")

# ══════════════════════════════════════════════════════════════════════════════
# CHART 4 — TENURE DISTRIBUTION BY CHURN
# ══════════════════════════════════════════════════════════════════════════════
print("[4] Tenure distribution by churn...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Histogram
for label, color in [("No", TEAL), ("Yes", RED)]:
    subset = df[df["Churn"] == label]["tenure"]
    axes[0].hist(subset, bins=30, alpha=0.65, color=color, edgecolor="white",
                 label=f"{'No Churn' if label=='No' else 'Churned'}")
axes[0].set_title("Tenure Distribution (Histogram)")
axes[0].set_xlabel("Tenure (months)")
axes[0].set_ylabel("Count")
axes[0].legend(framealpha=0.8)
axes[0].yaxis.grid(True)
axes[0].set_axisbelow(True)

# Box plot
bp_data  = [df[df["Churn"] == g]["tenure"].values for g in ["No", "Yes"]]
bp_labels= ["No Churn", "Churned"]
bp = axes[1].boxplot(bp_data, labels=bp_labels, patch_artist=True,
                     medianprops={"color": NAVY, "linewidth": 2},
                     whiskerprops={"color": "#475569"},
                     capprops={"color": "#475569"},
                     flierprops={"marker": "o", "markersize": 3, "alpha": 0.4})
for patch, color in zip(bp["boxes"], [TEAL, RED]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
axes[1].set_title("Tenure Distribution (Box Plot)")
axes[1].set_ylabel("Tenure (months)")
axes[1].yaxis.grid(True)
axes[1].set_axisbelow(True)

# Annotate medians
for i, data in enumerate(bp_data):
    median = np.median(data)
    axes[1].text(i + 1, median + 1.5, f"Med={median:.0f}m", ha="center",
                 fontsize=9.5, color=NAVY, fontweight="bold")

fig.suptitle("Tenure vs Churn", fontsize=15, fontweight="bold", color=NAVY)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/04_tenure_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("    ✅ Saved: 04_tenure_distribution.png")

# ══════════════════════════════════════════════════════════════════════════════
# CHART 5 — MONTHLY CHARGES ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print("[5] Monthly Charges analysis...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# KDE
for label, color, name in [("No", TEAL, "No Churn"), ("Yes", RED, "Churned")]:
    subset = df[df["Churn"] == label]["MonthlyCharges"]
    axes[0].hist(subset, bins=35, density=True, alpha=0.5, color=color, edgecolor="white")
    subset.plot.kde(ax=axes[0], color=color, lw=2.5, label=name)
axes[0].set_title("Monthly Charges — KDE + Histogram")
axes[0].set_xlabel("Monthly Charges ($)")
axes[0].set_ylabel("Density")
axes[0].legend(framealpha=0.8)
axes[0].yaxis.grid(True)
axes[0].set_axisbelow(True)

# Violin
parts = axes[1].violinplot(
    [df[df["Churn"] == g]["MonthlyCharges"].values for g in ["No", "Yes"]],
    positions=[1, 2], showmedians=True, showmeans=False,
)
for pc, color in zip(parts["bodies"], [TEAL, RED]):
    pc.set_facecolor(color)
    pc.set_alpha(0.6)
parts["cmedians"].set_color(NAVY)
parts["cmedians"].set_linewidth(2)
parts["cbars"].set_color("#475569")
parts["cmaxes"].set_color("#475569")
parts["cmins"].set_color("#475569")
axes[1].set_xticks([1, 2])
axes[1].set_xticklabels(["No Churn", "Churned"])
axes[1].set_title("Monthly Charges — Violin Plot")
axes[1].set_ylabel("Monthly Charges ($)")
axes[1].yaxis.grid(True)
axes[1].set_axisbelow(True)

fig.suptitle("Monthly Charges vs Churn", fontsize=15, fontweight="bold", color=NAVY)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/05_monthly_charges.png", dpi=150, bbox_inches="tight")
plt.close()
print("    ✅ Saved: 05_monthly_charges.png")

# ══════════════════════════════════════════════════════════════════════════════
# CHART 6 — CORRELATION HEATMAP
# ══════════════════════════════════════════════════════════════════════════════
print("[6] Correlation heatmap...")

# Load cleaned (encoded) dataset for numeric-only correlations
df_clean_path = "data/customer_churn_clean.csv"
if os.path.exists(df_clean_path):
    df_enc = pd.read_csv(df_clean_path)
else:
    # Fallback: encode on the fly
    df_enc = df.copy()
    df_enc["Churn"] = df_enc["Churn"].map({"Yes": 1, "No": 0})

numeric_cols = df_enc.select_dtypes(include=[np.number]).columns.tolist()

# Keep top features most correlated with Churn for readability
corr_full = df_enc[numeric_cols].corr()
top_feat  = corr_full["Churn"].abs().sort_values(ascending=False).head(16).index.tolist()
corr      = df_enc[top_feat].corr()

fig, ax = plt.subplots(figsize=(13, 10))
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(220, 15, as_cmap=True)
sns.heatmap(
    corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0,
    annot=True, fmt=".2f", annot_kws={"size": 8},
    square=True, linewidths=0.5, linecolor="#E2E8F0",
    ax=ax, cbar_kws={"shrink": 0.8},
)
ax.set_title("Feature Correlation Heatmap\n(Top 16 features most correlated with Churn)",
             fontsize=14, pad=16)
plt.xticks(rotation=40, ha="right", fontsize=9)
plt.yticks(rotation=0, fontsize=9)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/06_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("    ✅ Saved: 06_correlation_heatmap.png")

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY STATISTICS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 65)
print("  KEY INSIGHTS FROM EDA")
print("─" * 65)

churn_yes = df[df["Churn"] == "Yes"]
churn_no  = df[df["Churn"] == "No"]

print(f"\n  Overall churn rate    : {(df['Churn']=='Yes').mean():.1%}")
print(f"  Month-to-month churn  : {(churn_yes['Contract']=='Month-to-month').mean():.1%} of churners")
print(f"  Fiber optic churn     : {(churn_yes['InternetService']=='Fiber optic').mean():.1%} of churners")
print(f"  Avg tenure (churned)  : {churn_yes['tenure'].mean():.1f} months")
print(f"  Avg tenure (retained) : {churn_no['tenure'].mean():.1f} months")
print(f"  Avg monthly (churned) : ${churn_yes['MonthlyCharges'].mean():.2f}")
print(f"  Avg monthly (retained): ${churn_no['MonthlyCharges'].mean():.2f}")
print(f"\n  Charts saved to: {OUT_DIR}/")

print("\n" + "=" * 65)
print("  EDA complete. Run 03_model_training.py next.")
print("=" * 65)
