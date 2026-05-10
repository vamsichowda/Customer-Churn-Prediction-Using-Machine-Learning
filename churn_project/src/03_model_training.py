"""
================================================================================
  CUSTOMER CHURN PREDICTION — STEP 3: MODEL TRAINING & EVALUATION
================================================================================
  Models trained:
    • Logistic Regression (baseline)
    • Decision Tree Classifier
    • Random Forest Classifier  ← Best model
  Evaluation:
    • Accuracy, Precision, Recall, F1
    • Confusion Matrix
    • ROC-AUC Curve
    • Feature Importance (Random Forest)
================================================================================
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing   import StandardScaler
from sklearn.linear_model    import LogisticRegression
from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import RandomForestClassifier
from sklearn.metrics         import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report,
)

# ── Colours ───────────────────────────────────────────────────────────────────
NAVY  = "#0D1B2A"; TEAL  = "#00B4D8"; RED   = "#EF4444"
AMBER = "#F59E0B"; GREEN = "#22C55E"; GRAY  = "#94A3B8"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "#F8FAFC",
    "axes.edgecolor": "#CBD5E1", "axes.labelcolor": NAVY,
    "axes.titlesize": 13, "axes.titleweight": "bold",
    "xtick.color": "#475569", "ytick.color": "#475569",
    "font.family": "DejaVu Sans", "grid.color": "#E2E8F0",
    "grid.linestyle": "--", "grid.linewidth": 0.6,
})

DATA_CLEAN = "data/customer_churn_clean.csv"
OUT_DIR    = "outputs/models"
MODEL_DIR  = "models"
os.makedirs(OUT_DIR,   exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

print("=" * 65)
print("  STEP 3: MODEL TRAINING & EVALUATION")
print("=" * 65)

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA & SPLIT
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1] Loading clean dataset...")
df = pd.read_csv(DATA_CLEAN)

TARGET = "Churn"
X = df.drop(columns=[TARGET])
y = df[TARGET]

print(f"    Features : {X.shape[1]}")
print(f"    Samples  : {X.shape[0]:,}")
print(f"    Churn=1  : {y.sum():,} ({y.mean():.1%})")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\n    Train: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}")

# Scale features (for Logistic Regression)
scaler  = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ══════════════════════════════════════════════════════════════════════════════
# 2. DEFINE MODELS
# ══════════════════════════════════════════════════════════════════════════════
models = {
    "Logistic Regression": (
        LogisticRegression(max_iter=1000, random_state=42, C=1.0),
        X_train_sc, X_test_sc
    ),
    "Decision Tree": (
        DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, random_state=42),
        X_train, X_test
    ),
    "Random Forest": (
        RandomForestClassifier(n_estimators=150, max_depth=10,
                               min_samples_leaf=10, random_state=42, n_jobs=-1),
        X_train, X_test
    ),
}

# ══════════════════════════════════════════════════════════════════════════════
# 3. TRAIN & EVALUATE
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2] Training models...\n")
results = {}

for name, (model, X_tr, X_te) in models.items():
    print(f"  ── {name} ──")
    model.fit(X_tr, y_train)
    y_pred  = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc     = auc(fpr, tpr)

    # 5-fold CV on training set
    cv_scores = cross_val_score(model, X_tr, y_train, cv=5, scoring="f1")

    results[name] = {
        "model":   model,
        "y_pred":  y_pred,
        "y_proba": y_proba,
        "acc":     acc,
        "prec":    prec,
        "rec":     rec,
        "f1":      f1,
        "auc":     roc_auc,
        "fpr":     fpr,
        "tpr":     tpr,
        "cv_f1":   cv_scores.mean(),
        "cv_std":  cv_scores.std(),
    }
    print(f"    Accuracy : {acc:.4f}")
    print(f"    Precision: {prec:.4f}")
    print(f"    Recall   : {rec:.4f}")
    print(f"    F1 Score : {f1:.4f}")
    print(f"    ROC-AUC  : {roc_auc:.4f}")
    print(f"    CV F1    : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n")

# ══════════════════════════════════════════════════════════════════════════════
# 4. RESULTS TABLE
# ══════════════════════════════════════════════════════════════════════════════
print("[3] Results Summary:")
rows = []
for name, r in results.items():
    rows.append({
        "Model":      name,
        "Accuracy":   f"{r['acc']:.4f}",
        "Precision":  f"{r['prec']:.4f}",
        "Recall":     f"{r['rec']:.4f}",
        "F1 Score":   f"{r['f1']:.4f}",
        "ROC-AUC":    f"{r['auc']:.4f}",
        "CV F1 (5k)": f"{r['cv_f1']:.4f}",
    })
results_df = pd.DataFrame(rows)
print("\n" + results_df.to_string(index=False))
results_df.to_csv("outputs/model_results.csv", index=False)

# ══════════════════════════════════════════════════════════════════════════════
# 5. CONFUSION MATRICES
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4] Plotting confusion matrices...")
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
model_colors = [TEAL, AMBER, GREEN]

for ax, (name, r), color in zip(axes, results.items(), model_colors):
    cm = confusion_matrix(y_test, r["y_pred"])
    sns.heatmap(cm, annot=True, fmt="d", ax=ax, cmap="Blues",
                xticklabels=["Pred: No", "Pred: Yes"],
                yticklabels=["Act: No", "Act: Yes"],
                linewidths=1, linecolor="white",
                annot_kws={"size": 14, "weight": "bold"})
    ax.set_title(f"{name}\nAcc={r['acc']:.2%}  F1={r['f1']:.2%}", color=NAVY)
    ax.set_xlabel("Predicted", color=NAVY)
    ax.set_ylabel("Actual", color=NAVY)

fig.suptitle("Confusion Matrices — All Models", fontsize=15, fontweight="bold", color=NAVY)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/07_confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close()
print("    ✅ Saved: 07_confusion_matrices.png")

# ══════════════════════════════════════════════════════════════════════════════
# 6. ROC CURVES
# ══════════════════════════════════════════════════════════════════════════════
print("[5] Plotting ROC curves...")
fig, ax = plt.subplots(figsize=(7, 6))
line_styles = [("-", TEAL), ("--", AMBER), ("-.", GREEN)]
for (name, r), (ls, color) in zip(results.items(), line_styles):
    ax.plot(r["fpr"], r["tpr"], lw=2.5, linestyle=ls, color=color,
            label=f"{name}  (AUC = {r['auc']:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.5, label="Random (AUC = 0.500)")
ax.fill_between(results["Random Forest"]["fpr"],
                results["Random Forest"]["tpr"], alpha=0.08, color=GREEN)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve — Model Comparison")
ax.legend(loc="lower right", framealpha=0.9, fontsize=10)
ax.yaxis.grid(True); ax.xaxis.grid(True)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/08_roc_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("    ✅ Saved: 08_roc_curves.png")

# ══════════════════════════════════════════════════════════════════════════════
# 7. FEATURE IMPORTANCE (Random Forest)
# ══════════════════════════════════════════════════════════════════════════════
print("[6] Feature importance (Random Forest)...")
rf_model  = results["Random Forest"]["model"]
feat_imp  = pd.Series(rf_model.feature_importances_, index=X.columns)
feat_imp  = feat_imp.sort_values(ascending=False).head(15)

fig, ax = plt.subplots(figsize=(9, 6))
colors_bar = [GREEN if i == 0 else TEAL if i < 5 else GRAY for i in range(len(feat_imp))]
feat_imp.plot(kind="barh", ax=ax, color=colors_bar[::-1], edgecolor="white")
ax.invert_yaxis()
ax.set_title("Top 15 Feature Importances — Random Forest")
ax.set_xlabel("Importance Score")
ax.xaxis.grid(True); ax.set_axisbelow(True)
for i, (val, name) in enumerate(zip(feat_imp.values, feat_imp.index)):
    ax.text(val + 0.001, len(feat_imp)-1-i, f"{val:.4f}", va="center", fontsize=8.5, color=NAVY)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/09_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("    ✅ Saved: 09_feature_importance.png")

# ══════════════════════════════════════════════════════════════════════════════
# 8. MODEL COMPARISON BAR CHART
# ══════════════════════════════════════════════════════════════════════════════
print("[7] Model comparison chart...")
metrics = ["acc", "prec", "rec", "f1", "auc"]
labels  = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
x       = np.arange(len(labels))
width   = 0.25

fig, ax = plt.subplots(figsize=(11, 5))
offsets = [-width, 0, width]
bar_colors = [TEAL, AMBER, GREEN]
for i, (name, r) in enumerate(results.items()):
    vals = [r[m] for m in metrics]
    bars = ax.bar(x + offsets[i], vals, width, label=name, color=bar_colors[i], edgecolor="white")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8, color=NAVY)

ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
ax.set_ylim(0, 1.12)
ax.set_ylabel("Score"); ax.set_title("Model Performance Comparison")
ax.legend(framealpha=0.9); ax.yaxis.grid(True); ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/10_model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("    ✅ Saved: 10_model_comparison.png")

# ══════════════════════════════════════════════════════════════════════════════
# 9. SAVE BEST MODEL
# ══════════════════════════════════════════════════════════════════════════════
print("\n[8] Saving best model (Random Forest)...")
best_model = {
    "model":        results["Random Forest"]["model"],
    "scaler":       scaler,
    "feature_cols": list(X.columns),
}
with open(f"{MODEL_DIR}/best_model_rf.pkl", "wb") as f:
    pickle.dump(best_model, f)
print(f"    ✅ Saved → {MODEL_DIR}/best_model_rf.pkl")

# ══════════════════════════════════════════════════════════════════════════════
# 10. CLASSIFICATION REPORT
# ══════════════════════════════════════════════════════════════════════════════
print("\n[9] Classification Report — Random Forest:")
print(classification_report(y_test, results["Random Forest"]["y_pred"],
                             target_names=["No Churn", "Churned"]))

print("=" * 65)
print("  Training complete.")
print("  Best model: Random Forest  ✅")
best = results["Random Forest"]
print(f"  Accuracy : {best['acc']:.4f}")
print(f"  F1 Score : {best['f1']:.4f}")
print(f"  ROC-AUC  : {best['auc']:.4f}")
print("=" * 65)
