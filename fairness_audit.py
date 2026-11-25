"""
Fairness Audit of an AI Model for Pre-Approved Credit-Card Offers
Subject: Responsible and Safe AI (AIML4001) - External Practical, Set C
Dataset: Adult Census Income (adult.csv)  |  Sensitive attribute: sex
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from fairlearn.metrics import (MetricFrame, selection_rate, true_positive_rate,
    false_positive_rate, demographic_parity_difference, demographic_parity_ratio,
    equalized_odds_difference)
from fairlearn.postprocessing import ThresholdOptimizer

SEED = 42
DATA = "adult.csv"

# ---------------- Task 1: load, clean, encode, train ----------------
df = pd.read_csv(DATA, skipinitialspace=True)
print("Raw shape:", df.shape)
df = df.replace("?", np.nan).dropna().reset_index(drop=True)   # remove rows with '?'
print("Shape after removing '?' rows:", df.shape)

y = (df["income"].str.strip() == ">50K").astype(int)
sex = df["sex"].str.strip()
X = df.drop(columns=["income", "sex"])                          # sex NOT a feature

cat_cols = X.select_dtypes(exclude="number").columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]

X_tr, X_te, y_tr, y_te, s_tr, s_te = train_test_split(
    X, y, sex, test_size=0.30, random_state=SEED, stratify=y)
print("Train:", X_tr.shape, " Test:", X_te.shape)

pre = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)])
model = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000))])
model.fit(X_tr, y_tr)

# ---------------- Task 2: accuracy + MetricFrame ----------------
y_pred = model.predict(X_te)
base_acc = accuracy_score(y_te, y_pred)
print(f"\nBaseline model : Logistic Regression   Accuracy : {base_acc:.2f}")

def audit(y_true, y_hat, sens):
    mf = MetricFrame(
        metrics={"accuracy": accuracy_score, "selection_rate": selection_rate,
                 "tpr": true_positive_rate, "fpr": false_positive_rate},
        y_true=y_true, y_pred=y_hat, sensitive_features=sens)
    return mf

mf = audit(y_te, y_pred, s_te)
print("\nGROUP     ACCURACY  SELECTION_RATE   TPR    FPR")
print("-" * 52)
for g in ["Male", "Female"]:
    r = mf.by_group.loc[g]
    print(f"{g:<8}  {r.accuracy:.2f}      {r.selection_rate:.2f}            {r.tpr:.2f}   {r.fpr:.2f}")
print("-" * 52)

# ---------------- Task 3: fairness metrics ----------------
def fair_metrics(y_true, y_hat, sens):
    return (demographic_parity_difference(y_true, y_hat, sensitive_features=sens),
            demographic_parity_ratio(y_true, y_hat, sensitive_features=sens),
            equalized_odds_difference(y_true, y_hat, sensitive_features=sens))

dpd, dpr, eod = fair_metrics(y_te, y_pred, s_te)
print(f"\nDemographic parity difference : {dpd:.2f}")
print(f"Demographic parity ratio (DI) : {dpr:.2f} ({'< 0.80 -> BIAS' if dpr < 0.8 else '>= 0.80 -> PASS'})")
print(f"Equalized odds difference     : {eod:.2f}")

# ---------------- Tasks 4 & 5: ThresholdOptimizer ----------------
Xtr_t = model.named_steps["pre"].transform(X_tr)
Xte_t = model.named_steps["pre"].transform(X_te)
clf = model.named_steps["clf"]

results = {"Baseline": (y_pred, mf)}
for name, cons in [("TO: demographic_par.", "demographic_parity"),
                   ("TO: equalized_odds", "equalized_odds")]:
    to = ThresholdOptimizer(estimator=clf, constraints=cons, objective="accuracy_score",
                            prefit=True, predict_method="predict_proba")
    to.fit(Xtr_t, y_tr, sensitive_features=s_tr)
    yp = to.predict(Xte_t, sensitive_features=s_te, random_state=SEED)
    results[name] = (yp, audit(y_te, yp, s_te))
    d, r_, e = fair_metrics(y_te, yp, s_te)
    print(f"\n=== ThresholdOptimizer: {cons} ===")
    print(f"Accuracy {accuracy_score(y_te, yp):.2f}")
    print(audit(y_te, yp, s_te).by_group.round(2).to_string())
    print(f"DP diff {d:.2f} | DP ratio {r_:.2f} | EO diff {e:.2f}")

# ---------------- Task 6: comparison table + plot ----------------
rows = []
for name, (yp, m) in results.items():
    d, r_, e = fair_metrics(y_te, yp, s_te)
    rows.append(dict(MODEL=name, ACC=accuracy_score(y_te, yp),
                     SEL_M=m.by_group.loc["Male", "selection_rate"],
                     SEL_F=m.by_group.loc["Female", "selection_rate"],
                     DP_DIFF=d, DP_RATIO=r_, EO_DIFF=e))
comp = pd.DataFrame(rows).round(2)
comp.to_csv("comparison_table.csv", index=False)
print("\nMODEL                 ACC   SEL_M  SEL_F  DP_DIFF  DP_RATIO  EO_DIFF")
print("-" * 70)
for _, r in comp.iterrows():
    print(f"{r.MODEL:<20}  {r.ACC:.2f}  {r.SEL_M:.2f}   {r.SEL_F:.2f}   {r.DP_DIFF:.2f}     {r.DP_RATIO:.2f}      {r.EO_DIFF:.2f}")
print("-" * 70)

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(comp)); w = 0.35
b1 = ax.bar(x - w/2, comp.SEL_M, w, label="Male", color="#1f77b4")
b2 = ax.bar(x + w/2, comp.SEL_F, w, label="Female", color="#e377c2")
for b in list(b1) + list(b2):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005, f"{b.get_height():.2f}", ha="center")
ax.set_xticks(x); ax.set_xticklabels(comp.MODEL)
ax.set_ylabel("Selection rate (predicted high income)")
ax.set_title("Selection Rate by Sex: Baseline vs Mitigated Models")
ax.legend(); plt.tight_layout()
plt.savefig("selection_rate_by_sex.png", dpi=150)
print("\nPlot saved : selection_rate_by_sex.png")

# extra plot: accuracy vs fairness
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - w/2, comp.DP_DIFF, w, label="DP difference", color="#2ca02c")
ax.bar(x + w/2, comp.EO_DIFF, w, label="EO difference", color="#d62728")
ax2 = ax.twinx(); ax2.plot(x, comp.ACC, "ko-", label="Accuracy"); ax2.set_ylim(0.7, 0.9)
ax2.set_ylabel("Accuracy"); ax.set_ylabel("Disparity (lower = fairer)")
ax.set_xticks(x); ax.set_xticklabels(comp.MODEL); ax.set_title("Accuracy vs Fairness Trade-off")
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="upper left"); plt.tight_layout()
plt.savefig("accuracy_vs_fairness.png", dpi=150)
