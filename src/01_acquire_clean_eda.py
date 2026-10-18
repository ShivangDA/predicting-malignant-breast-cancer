"""Step 1 - Acquisition, cleaning, duplicate-impact study, EDA, multicollinearity.

Dataset: Breast Cancer Wisconsin (Diagnostic) - UCI Machine Learning Repository.
569 patients, 30 numeric features computed from digitised images of fine-needle
aspirates (FNA) of breast masses. Target: M = malignant, B = benign.
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from common import *

UCI_URL = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
           "breast-cancer-wisconsin/wdbc.data")
BASES = ["radius", "texture", "perimeter", "area", "smoothness", "compactness",
         "concavity", "concave_points", "symmetry", "fractal_dimension"]
COLS = (["id", "diagnosis"] + [f"{b}_mean" for b in BASES]
        + [f"{b}_se" for b in BASES] + [f"{b}_worst" for b in BASES])
out = {}

# ------------------------------------------------------------------ 1. ACQUIRE
raw_path = RAW / "wdbc.data"
source = None
try:
    import urllib.request
    urllib.request.urlretrieve(UCI_URL, raw_path)
    df = pd.read_csv(raw_path, header=None, names=COLS)
    source = "UCI repository (direct download)"
except Exception as e:  # offline / blocked network -> identical bundled copy
    from sklearn.datasets import load_breast_cancer
    d = load_breast_cancer(as_frame=True)
    X = d.data.copy()
    X.columns = COLS[2:]
    # sklearn target: 0 = malignant, 1 = benign
    X.insert(0, "diagnosis", np.where(d.target == 0, "M", "B"))
    X.insert(0, "id", np.arange(1, len(X) + 1))  # bundled copy has no patient ids
    df = X
    df.to_csv(raw_path, header=False, index=False)
    source = f"scikit-learn bundled copy of the UCI WDBC data ({type(e).__name__} on download)"
print("Source:", source)
out["source"] = source
out["raw_shape"] = list(df.shape)

# ------------------------------------------------------------------ 2. AUDIT
audit = {
    "rows": int(len(df)),
    "columns": int(df.shape[1]),
    "missing_cells": int(df.isna().sum().sum()),
    "duplicate_rows_all_cols": int(df.duplicated().sum()),
    "duplicate_rows_excluding_id": int(df.drop(columns="id").duplicated().sum()),
    "duplicate_ids": int(df["id"].duplicated().sum()),
    "non_numeric_feature_cols": [c for c in COLS[2:] if not pd.api.types.is_numeric_dtype(df[c])],
    "zero_values_in_concavity_cols": {c: int((df[c] == 0).sum()) for c in COLS[2:]
                                      if (df[c] == 0).sum() > 0},
    "negative_values": int((df[COLS[2:]] < 0).sum().sum()),
}
out["audit"] = audit
print(json.dumps(audit, indent=2))

# ------------------------------------------------------------------ 3. CLEAN
clean = df.copy()
clean["diagnosis"] = clean["diagnosis"].astype("category")
clean["target"] = (clean["diagnosis"] == "M").astype(int)        # 1 = malignant
clean["label"] = clean["diagnosis"].map({"B": "Benign", "M": "Malignant"})
clean = clean.drop_duplicates(subset=COLS[2:]).reset_index(drop=True)
feat = COLS[2:]
clean[feat] = clean[feat].astype("float64")
clean.to_csv(PROC / "wdbc_clean.csv", index=False)
out["clean_shape"] = list(clean.shape)
out["class_counts"] = clean["label"].value_counts().to_dict()
out["malignant_pct"] = round(100 * clean["target"].mean(), 2)

# Outliers (IQR 1.5x rule) - flagged, NOT removed
q1, q3 = clean[feat].quantile(.25), clean[feat].quantile(.75)
iqr = q3 - q1
is_out = (clean[feat] < q1 - 1.5 * iqr) | (clean[feat] > q3 + 1.5 * iqr)
out_counts = is_out.sum().sort_values(ascending=False)
out["outlier_rows_any_feature"] = int(is_out.any(axis=1).sum())
out["outlier_top"] = out_counts.head(8).to_dict()
# Do outliers concentrate in malignant tumours?
out["outlier_share_malignant"] = round(float(
    is_out.any(axis=1)[clean.target == 1].mean() * 100), 1)
out["outlier_share_benign"] = round(float(
    is_out.any(axis=1)[clean.target == 0].mean() * 100), 1)

# Skewness
skew = clean[feat].skew().sort_values(ascending=False)
out["skew_top"] = skew.head(6).round(2).to_dict()

# ----------------------------------------- 4. DUPLICATE IMPACT (simulation)
# The raw data has no exact duplicates, so we quantify *what would happen* if
# duplicates were present and left in: leaked copies inflate test performance.
rng = np.random.default_rng(SEED)
sim = []
for rate in [0.0, 0.05, 0.10, 0.20, 0.30]:
    accs, recs, aucs = [], [], []
    for rep in range(30):
        n_dup = int(rate * len(clean))
        dup = clean.sample(n_dup, random_state=rep, replace=False) if n_dup else clean.iloc[:0]
        dirty = pd.concat([clean, dup], ignore_index=True)
        Xd, yd = dirty[feat].values, dirty["target"].values
        # use a deliberately flexible model: it can memorise leaked rows
        from sklearn.neighbors import KNeighborsClassifier
        Xtr, Xte, ytr, yte = train_test_split(Xd, yd, test_size=.25, random_state=rep, stratify=yd)
        m = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=1)).fit(Xtr, ytr)
        p = m.predict(Xte)
        accs.append(accuracy_score(yte, p)); recs.append(recall_score(yte, p))
        aucs.append(roc_auc_score(yte, m.predict_proba(Xte)[:, 1]))
    sim.append({"dup_rate_pct": int(rate * 100), "accuracy": np.mean(accs),
                "recall": np.mean(recs), "auc": np.mean(aucs),
                "acc_sd": np.std(accs)})
sim = pd.DataFrame(sim)
sim.to_csv(RES / "duplicate_impact.csv", index=False)
out["duplicate_sim"] = sim.round(4).to_dict("records")

# also: duplicates distort class prior & summary statistics
dup_stats = []
for rate in [0.0, 0.10, 0.30]:
    n_dup = int(rate * len(clean))
    d_ = pd.concat([clean, clean[clean.target == 1].sample(n_dup, random_state=1)], ignore_index=True) \
        if n_dup else clean
    dup_stats.append({"scenario": f"{int(rate*100)}% extra malignant dupes",
                      "malignant_pct": round(100 * d_.target.mean(), 2),
                      "mean_area": round(d_["area_mean"].mean(), 1)})
out["duplicate_prior_shift"] = dup_stats

# Fig: duplicate impact
fig, ax = plt.subplots(1, 2, figsize=(10.5, 4))
ax[0].plot(sim.dup_rate_pct, sim.accuracy * 100, "o-", color=C_M, lw=2, label="Test accuracy")
ax[0].fill_between(sim.dup_rate_pct, (sim.accuracy - sim.acc_sd) * 100,
                   (sim.accuracy + sim.acc_sd) * 100, color=C_M, alpha=.15)
ax[0].set(xlabel="Duplicate rows left in data (%)", ylabel="Test accuracy (%)",
          title="Duplicates inflate scores modestly")
for x, y in zip(sim.dup_rate_pct, sim.accuracy * 100):
    ax[0].annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
ax[1].bar(sim.dup_rate_pct.astype(str) + "%", (sim.accuracy - sim.accuracy.iloc[0]) * 100,
          color=C_B)
ax[1].set(xlabel="Duplicate rows left in data (%)", ylabel="Optimism vs clean data (pp)",
          title="Optimism from train/test leakage (within ~1 SD)")
fig.suptitle("Impact of duplicate records (1-NN, 30 repeated splits)", y=1.03, fontsize=12)
save(fig, "fig01_duplicate_impact.png")

# ------------------------------------------------------------------ 5. EDA
# Fig 2: class balance + missing/validity audit
fig, ax = plt.subplots(1, 2, figsize=(11.5, 3.8), gridspec_kw=dict(wspace=.75))
cc = clean["label"].value_counts()
ax[0].bar(cc.index, cc.values, color=[PALETTE[i] for i in cc.index])
for i, v in enumerate(cc.values):
    ax[0].text(i, v + 5, f"{v}\n({100*v/len(clean):.1f}%)", ha="center")
ax[0].set(title="Class balance", ylabel="Patients", ylim=(0, cc.max() * 1.2))
miss = df.isna().sum()
labels = ["Missing cells", "Exact duplicate rows", "Duplicate IDs", "Negative values", "Non-numeric feature cols"]
vals = [audit["missing_cells"], audit["duplicate_rows_all_cols"], audit["duplicate_ids"],
        audit["negative_values"], len(audit["non_numeric_feature_cols"])]
ax[1].barh(labels, [max(v, .02) for v in vals], color=C_B)
for i, v in enumerate(vals):
    ax[1].text(.05, i, f"{v}  (none found)" if v == 0 else str(v), va="center", fontweight="bold")
ax[1].set(title="Data-quality audit: issues found", xlim=(0, 5), xlabel="Count")
ax[1].invert_yaxis()
save(fig, "fig02_class_balance_quality.png")

# Fig 3: distributions of 6 key features
key = ["radius_mean", "concave_points_mean", "area_worst", "concavity_worst",
       "texture_mean", "smoothness_mean"]
fig, axes = plt.subplots(2, 3, figsize=(12, 7), gridspec_kw=dict(hspace=.45, wspace=.3))
for a, c in zip(axes.ravel(), key):
    sns.kdeplot(data=clean, x=c, hue="label", palette=PALETTE, fill=True, alpha=.4,
                common_norm=False, ax=a, legend=a is axes[0, 0])
    a.set_title(c.replace("_", " ").title())
    a.set_ylabel("")
fig.suptitle("Malignant tumours are shifted to larger, more irregular shapes", y=1.01)
save(fig, "fig03_distributions.png")

# Fig 4: box plots (standardised) of all _mean features by class
fig, ax = plt.subplots(figsize=(11, 4.6))
mean_cols = [f"{b}_mean" for b in BASES]
z = (clean[mean_cols] - clean[mean_cols].mean()) / clean[mean_cols].std()
zl = z.assign(label=clean.label).melt("label", var_name="feature", value_name="z")
zl["feature"] = zl["feature"].str.replace("_mean", "").str.replace("_", " ")
sns.boxplot(data=zl, x="feature", y="z", hue="label", palette=PALETTE, fliersize=2, ax=ax)
ax.set(xlabel="", ylabel="Standardised value (z-score)", title="Box plots of mean features by diagnosis")
ax.tick_params(axis="x", rotation=30)
ax.legend(title="")
save(fig, "fig04_boxplots.png")

# Fig 5: correlation heatmap (hierarchically ordered)
corr = clean[feat].corr()
dist = 1 - corr.abs()
link = hierarchy.linkage(squareform(dist.values, checks=False), "average")
order = hierarchy.leaves_list(link)
corr_o = corr.iloc[order, order]
fig, ax = plt.subplots(figsize=(10.5, 9))
sns.heatmap(corr_o, cmap="RdBu_r", vmin=-1, vmax=1, center=0, square=True, ax=ax,
            xticklabels=True, yticklabels=True, cbar_kws={"shrink": .6, "label": "Pearson r"})
ax.tick_params(labelsize=7)
ax.set_title("Feature correlation (clustered): large blocks of redundant features")
save(fig, "fig05_correlation_heatmap.png")

# correlation analysis
pairs = (corr.where(np.triu(np.ones(corr.shape), 1).astype(bool)).stack().dropna()
         .rename("r").reset_index())
pairs.columns = ["f1", "f2", "r"]
pairs["abs_r"] = pairs.r.abs()
out["n_pairs_gt_0.9"] = int((pairs.abs_r > .9).sum())
out["n_pairs_gt_0.8"] = int((pairs.abs_r > .8).sum())
out["n_pairs_total"] = int(len(pairs))
out["top_pairs"] = pairs.sort_values("abs_r", ascending=False).head(8).round(4).to_dict("records")
# correlation with target (point-biserial = pearson with binary)
ct = clean[feat].corrwith(clean.target).sort_values(key=np.abs, ascending=False)
out["top_target_corr"] = ct.head(10).round(3).to_dict()
out["weak_target_corr"] = ct.tail(5).round(3).to_dict()

fig, ax = plt.subplots(figsize=(7.5, 8))
ctp = ct.sort_values()
ax.barh(ctp.index.str.replace("_", " "), ctp.values,
        color=[C_M if v > 0 else C_B for v in ctp.values])
ax.set(xlabel="Correlation with malignancy (point-biserial r)",
       title="Which features track malignancy?")
ax.tick_params(axis="y", labelsize=8)
save(fig, "fig06_target_correlation.png")

# Scatter: radius vs concave points with class
fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
sns.scatterplot(data=clean, x="radius_mean", y="concave_points_mean", hue="label",
                palette=PALETTE, alpha=.7, s=28, ax=ax[0])
ax[0].set_title("Two features alone separate most cases")
sns.scatterplot(data=clean, x="radius_mean", y="area_mean", hue="label", palette=PALETTE,
                alpha=.7, s=28, ax=ax[1], legend=False)
ax[1].set_title(f"Redundancy: radius vs area (r = {corr.loc['radius_mean','area_mean']:.2f})")
save(fig, "fig07_scatter.png")

# ------------------------------------------------------- 6. MULTICOLLINEARITY
Xs = StandardScaler().fit_transform(clean[feat])
vif = pd.Series([variance_inflation_factor(Xs, i) for i in range(Xs.shape[1])],
                index=feat).sort_values(ascending=False)
vif.to_csv(RES / "vif_all_features.csv", header=["VIF"])
out["vif_top"] = vif.head(10).round(1).to_dict()
out["vif_n_gt_10"] = int((vif > 10).sum())
out["vif_n_gt_5"] = int((vif > 5).sum())
out["condition_number"] = float(np.linalg.cond(Xs))

# Iterative VIF pruning (threshold 10)
keep = list(feat)
while True:
    v = pd.Series([variance_inflation_factor(clean[keep].pipe(lambda d: (d - d.mean()) / d.std()).values, i)
                   for i in range(len(keep))], index=keep)
    if v.max() <= 10:
        break
    keep.remove(v.idxmax())
out["vif_pruned_features"] = keep
out["vif_pruned_n"] = len(keep)
json.dump(keep, open(RES / "vif_pruned_features.json", "w"), indent=1)

pca = PCA().fit(Xs)
cum = np.cumsum(pca.explained_variance_ratio_)
out["pca_pc1_pct"] = round(100 * pca.explained_variance_ratio_[0], 1)
out["pca_pc2_pct"] = round(100 * pca.explained_variance_ratio_[1], 1)
out["pca_n_90"] = int(np.argmax(cum >= .90) + 1)
out["pca_n_95"] = int(np.argmax(cum >= .95) + 1)

fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
vt = vif.head(12).sort_values()
ax[0].barh(vt.index.str.replace("_", " "), vt.values, color=[C_M if v > 10 else C_B for v in vt.values])
ax[0].axvline(10, ls="--", c="k"); ax[0].set_xscale("log")
ax[0].set(xlabel="VIF (log scale)", title=f"{out['vif_n_gt_10']} of 30 features have VIF > 10")
ax[0].tick_params(axis="y", labelsize=8)
ax[1].plot(range(1, 31), cum * 100, "o-", color=C_B, ms=4)
ax[1].axhline(95, ls="--", c="grey"); ax[1].axvline(out["pca_n_95"], ls="--", c="grey")
ax[1].set(xlabel="Principal components", ylabel="Cumulative variance (%)",
          title=f"{out['pca_n_95']} components explain 95%")
pcs = PCA(2).fit_transform(Xs)
for lab in ["Benign", "Malignant"]:
    m_ = clean.label.values == lab
    ax[2].scatter(pcs[m_, 0], pcs[m_, 1], s=14, alpha=.65, c=PALETTE[lab], label=lab)
ax[2].set(xlabel=f"PC1 ({out['pca_pc1_pct']}%)", ylabel=f"PC2 ({out['pca_pc2_pct']}%)",
          title="PCA: classes are almost linearly separable")
ax[2].legend()
save(fig, "fig08_multicollinearity_pca.png")

# Duplicate sim pretty table + summary stats table
desc = clean.groupby("label")[["radius_mean", "texture_mean", "area_mean",
                               "concavity_mean", "concave_points_mean"]].agg(["mean", "std"]).round(3)
desc.to_csv(RES / "summary_by_class.csv")
clean[feat].describe().T.round(3).to_csv(RES / "summary_stats.csv")

json.dump(out, open(RES / "step1_metrics.json", "w"), indent=2, default=float)
print("Step 1 done.")
