"""
Week 4 - Machine Learning Model Development and Evaluation
Dataset : Breast Cancer Wisconsin (Diagnostic)  [UCI ML Repository; bundled with scikit-learn]
Task    : Binary classification  (1 = malignant, 0 = benign)

Run (from anywhere):  python src/week4_ml_pipeline.py
Outputs: figures/*.png, results/results.json, results/extras.json  (relative to the repo root)
"""
import os, json, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import pdist

import sklearn
from sklearn.base import clone
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, matthews_corrcoef,
                             brier_score_loss, confusion_matrix, roc_curve,
                             precision_recall_curve, fbeta_score)
from sklearn.model_selection import (train_test_split, StratifiedKFold, RepeatedStratifiedKFold,
                                     GridSearchCV, cross_validate, cross_val_predict,
                                     learning_curve, validation_curve)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

SEED = 42
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures")
RES = os.path.join(ROOT, "results")
os.makedirs(FIG, exist_ok=True); os.makedirs(RES, exist_ok=True)
T0 = time.time()

plt.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.labelsize": 9,
    "savefig.dpi": 200, "savefig.bbox": "tight", "legend.frameon": False,
})
C_B, C_M, C_N, C_D, C_A = "#2A9D8F", "#E76F51", "#8D99AE", "#264653", "#E9C46A"
R = {}   # every number quoted in the report is stored here


def log(msg):
    print(f"[{time.time()-T0:6.1f}s] {msg}", flush=True)


def save(name):
    plt.savefig(f"{FIG}/{name}.png")
    plt.close("all")


# =============================================================================
# 1. LOAD + AUDIT
# =============================================================================
raw = load_breast_cancer(as_frame=True)
df = raw.frame.copy()
df["malignant"] = 1 - df["target"]          # sklearn codes malignant as 0 -> recode so positive class = malignant
FEATS = list(raw.feature_names)
X, y = df[FEATS].copy(), df["malignant"].copy()

R["shape"] = [int(X.shape[0]), int(X.shape[1])]
R["n_missing"] = int(X.isna().sum().sum())
R["n_dup_rows"] = int(df[FEATS].duplicated().sum())
R["n_malignant"] = int(y.sum()); R["n_benign"] = int((1 - y).sum())
R["pct_malignant"] = float(y.mean() * 100)
Xs_all = StandardScaler().fit_transform(X)
R["min_pair_dist"] = float(pdist(Xs_all).min())
R["zero_counts"] = {c: int((X[c] == 0).sum()) for c in FEATS if (X[c] == 0).sum() > 0}
R["dtypes"] = sorted(set(str(t) for t in X.dtypes))

# --- outliers (Tukey fences) ---
q1, q3 = X.quantile(.25), X.quantile(.75); iqr = q3 - q1
omask = (X < q1 - 1.5 * iqr) | (X > q3 + 1.5 * iqr)
row_out = omask.any(axis=1)
R["outlier_cells"] = int(omask.values.sum())
R["outlier_cell_pct"] = float(omask.values.sum() / omask.size * 100)
R["rows_with_outlier"] = int(row_out.sum())
R["rows_with_outlier_pct"] = float(row_out.mean() * 100)
R["outlier_rows_malignant_pct"] = float(y[row_out].mean() * 100)
R["outlier_rate_malignant"] = float(row_out[y == 1].mean() * 100)
R["outlier_rate_benign"] = float(row_out[y == 0].mean() * 100)
R["outliers_top"] = omask.sum().sort_values(ascending=False).head(5).astype(int).to_dict()
skew = X.skew().sort_values(ascending=False)
R["skew_top"] = skew.head(5).round(2).to_dict()
R["n_skew_gt1"] = int((skew.abs() > 1).sum())

fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.0))
ax[0].bar(["Benign", "Malignant"], [R["n_benign"], R["n_malignant"]], color=[C_B, C_M])
for i, v in enumerate([R["n_benign"], R["n_malignant"]]):
    ax[0].text(i, v + 6, f"{v}  ({v/len(y)*100:.1f}%)", ha="center")
ax[0].set_title("(a) Class balance"); ax[0].set_ylabel("Patients"); ax[0].set_ylim(0, 420)
ax[1].bar(["Benign", "Malignant"], [R["outlier_rate_benign"], R["outlier_rate_malignant"]], color=[C_B, C_M])
for i, v in enumerate([R["outlier_rate_benign"], R["outlier_rate_malignant"]]):
    ax[1].text(i, v + 1, f"{v:.0f}%", ha="center")
ax[1].set_title("(b) Rows with ≥1 IQR outlier, by class"); ax[1].set_ylabel("% of class")
ax[1].set_ylim(0, max(R["outlier_rate_benign"], R["outlier_rate_malignant"]) * 1.25)
plt.tight_layout(); save("fig01_class_outliers")
log("audit done")

# =============================================================================
# 2. EDA that matters for modelling
# =============================================================================
# --- univariate separability ---
rows = []
for c in FEATS:
    auc = roc_auc_score(y, X[c]); auc = max(auc, 1 - auc)
    u, p = stats.mannwhitneyu(X.loc[y == 1, c], X.loc[y == 0, c], alternative="two-sided")
    rows.append(dict(feature=c, uni_auc=auc, p=p,
                     med_benign=X.loc[y == 0, c].median(), med_malig=X.loc[y == 1, c].median()))
uni = pd.DataFrame(rows).sort_values("uni_auc", ascending=False).reset_index(drop=True)
uni["ratio"] = uni["med_malig"] / uni["med_benign"]
R["uni_top"] = uni.head(8).to_dict("records")
R["uni_n_auc_gt_09"] = int((uni.uni_auc > 0.9).sum())
R["uni_n_auc_lt_060"] = int((uni.uni_auc < 0.60).sum())
R["uni_weakest"] = uni.tail(4).to_dict("records")
R["uni_n_sig_bonf"] = int((uni.p < 0.05 / len(FEATS)).sum())

# --- correlation / multicollinearity ---
corr = X.corr()
iu = np.triu_indices(len(FEATS), 1)
pairs = pd.DataFrame({"a": np.array(FEATS)[iu[0]], "b": np.array(FEATS)[iu[1]], "r": corr.values[iu]})
pairs["absr"] = pairs.r.abs()
R["n_pairs"] = int(len(pairs))
R["n_pairs_gt09"] = int((pairs.absr > 0.9).sum())
R["n_pairs_gt095"] = int((pairs.absr > 0.95).sum())
R["top_pairs"] = pairs.sort_values("absr", ascending=False).head(6).to_dict("records")
R["corr_median_abs"] = float(pairs.absr.median())


def vif_table(Xdf):
    Z = (Xdf - Xdf.mean()) / Xdf.std()
    out = {}
    for c in Z.columns:
        others = [o for o in Z.columns if o != c]
        r2 = LinearRegression().fit(Z[others], Z[c]).score(Z[others], Z[c])
        out[c] = 1 / (1 - r2) if r2 < 1 - 1e-12 else np.inf
    return pd.Series(out).sort_values(ascending=False)


vif = vif_table(X)
R["vif_max"] = float(vif.max()); R["vif_max_feat"] = vif.idxmax()
R["vif_gt10"] = int((vif > 10).sum()); R["vif_gt100"] = int((vif > 100).sum())
R["vif_top"] = vif.head(5).round(0).to_dict()


def prune_corr(Xdf, thr=0.9):
    """Unsupervised (label-free) pruning: repeatedly drop one member of the most correlated pair."""
    cols = list(Xdf.columns)
    while True:
        cm = Xdf[cols].corr().abs().to_numpy().copy(); np.fill_diagonal(cm, 0)
        if cm.max() <= thr: break
        i, j = np.unravel_index(cm.argmax(), cm.shape)
        drop = cols[i] if cm[i].mean() > cm[j].mean() else cols[j]
        cols.remove(drop)
    return cols


PRUNED = prune_corr(X, 0.9)
R["pruned_n"] = len(PRUNED); R["pruned_feats"] = PRUNED
vif_p = vif_table(X[PRUNED])
R["vif_pruned_max"] = float(vif_p.max()); R["vif_pruned_gt10"] = int((vif_p > 10).sum())
MEANS = [c for c in FEATS if c.startswith("mean ")]

cg = sns.clustermap(corr, cmap="RdBu_r", vmin=-1, vmax=1, figsize=(7.2, 7.2),
                    xticklabels=True, yticklabels=True, dendrogram_ratio=0.09,
                    cbar_pos=(0.02, 0.83, 0.02, 0.13))
cg.ax_heatmap.tick_params(labelsize=6)
plt.setp(cg.ax_heatmap.get_xticklabels(), rotation=90)
save("fig03_corr_clustermap")

# --- distributions of the most informative, non-redundant features ---
top6 = []          # greedy, one-per-family: strongest features whose |r| with every already-chosen feature is < 0.9 (descriptive EDA only)
for f in uni.feature:
    if all(abs(corr.loc[f, g]) < 0.9 for g in top6): top6.append(f)
    if len(top6) == 6: break
R["top6_pruned"] = top6
fig, axes = plt.subplots(2, 3, figsize=(7.6, 4.3))
for ax, c in zip(axes.ravel(), top6):
    sns.kdeplot(X.loc[y == 0, c], ax=ax, color=C_B, fill=True, alpha=.35, label="Benign", cut=0)
    sns.kdeplot(X.loc[y == 1, c], ax=ax, color=C_M, fill=True, alpha=.35, label="Malignant", cut=0)
    auc = uni.loc[uni.feature == c, "uni_auc"].iloc[0]
    ax.set_title(f"{c}\nunivariate AUC = {auc:.3f}", fontsize=8.5)
    ax.set_xlabel(""); ax.set_ylabel("")
    ax.set_yticks([])
axes[0, 0].legend(loc="upper right", fontsize=8)
plt.tight_layout(); save("fig02_distributions")
log("EDA done")

# =============================================================================
# 3. SPLIT
# =============================================================================
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.20, stratify=y, random_state=SEED)
R["n_train"], R["n_test"] = int(len(Xtr)), int(len(Xte))
R["train_malig"], R["test_malig"] = int(ytr.sum()), int(yte.sum())

# =============================================================================
# 4. DUPLICATE-LEAKAGE EXPERIMENT  (dataset has 0 duplicates -> quantify what they WOULD do)
# =============================================================================
DUP_MODELS = {
    "Decision Tree (unpruned)": lambda s: DecisionTreeClassifier(random_state=s),
    "1-Nearest Neighbour": lambda s: make_pipeline(StandardScaler(), KNeighborsClassifier(1)),
    "Random Forest": lambda s: RandomForestClassifier(n_estimators=60, random_state=s, n_jobs=1),
    "Logistic Regression": lambda s: make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
}
RATES = [0.0, 0.1, 0.2, 0.3, 0.5]
SEEDS = range(20)
dup = {m: {r: {"naive": [], "clean": []} for r in RATES} for m in DUP_MODELS}
for s in SEEDS:
    rng = np.random.RandomState(s)
    Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(X, y, test_size=.2, stratify=y, random_state=s)
    clean_acc = {}
    for m, f in DUP_MODELS.items():
        clean_acc[m] = accuracy_score(yc_te, f(s).fit(Xc_tr, yc_tr).predict(Xc_te))
    for r in RATES:
        idx = rng.choice(len(X), int(round(r * len(X))), replace=False)
        Xa = pd.concat([X, X.iloc[idx]], ignore_index=True)
        ya = pd.concat([y, y.iloc[idx]], ignore_index=True)
        Xa_tr, Xa_te, ya_tr, ya_te = train_test_split(Xa, ya, test_size=.2, stratify=ya, random_state=s)
        for m, f in DUP_MODELS.items():
            dup[m][r]["naive"].append(accuracy_score(ya_te, f(s).fit(Xa_tr, ya_tr).predict(Xa_te)))
            dup[m][r]["clean"].append(clean_acc[m])
R["dup"] = {m: {str(r): {k: [float(np.mean(v)), float(np.std(v, ddof=1) / np.sqrt(len(v)))]
                         for k, v in dup[m][r].items()} for r in RATES} for m in DUP_MODELS}
R["dup_infl"] = {m: {str(r): float((np.mean(dup[m][r]["naive"]) - np.mean(dup[m][r]["clean"])) * 100)
                     for r in RATES} for m in DUP_MODELS}

fig, ax = plt.subplots(1, 2, figsize=(8.0, 3.6), gridspec_kw={"width_ratios": [1.15, 1]})
names = list(DUP_MODELS); xp = np.arange(len(names)); w = .36
naive30 = [np.mean(dup[m][0.3]["naive"]) * 100 for m in names]
clean30 = [np.mean(dup[m][0.3]["clean"]) * 100 for m in names]
ax[0].bar(xp - w/2, clean30, w, color=C_B, label="Clean protocol (deduplicate, then split)")
ax[0].bar(xp + w/2, naive30, w, color=C_M, label="Naive protocol (duplicates straddle split)")
for i, (a, b) in enumerate(zip(clean30, naive30)):
    ax[0].text(i - w/2, a + .25, f"{a:.1f}", ha="center", fontsize=8)
    ax[0].text(i + w/2, b + .25, f"{b:.1f}", ha="center", fontsize=8)
ax[0].set_xticks(xp); ax[0].set_xticklabels([n.replace(" (", "\n(") .replace("Logistic Regression", "Logistic\nRegression").replace("Random Forest", "Random\nForest").replace("1-Nearest Neighbour", "1-Nearest\nNeighbour") for n in names], fontsize=8.5)
ax[0].set_ylim(88, 103.5); ax[0].set_yticks([88, 90, 92, 94, 96, 98, 100]); ax[0].set_ylabel("Test accuracy (%)  [axis starts at 88]")
ax[0].set_title("(a) 30% duplicated rows: reported accuracy"); ax[0].legend(fontsize=7.5, loc="upper left")
cols = [C_M, C_D, C_A, C_B]
for m, c in zip(names, cols):
    infl = [R["dup_infl"][m][str(r)] for r in RATES]
    ax[1].plot([r * 100 for r in RATES], infl, marker="o", color=c, label=m)
ax[1].axhline(0, color="grey", lw=.8)
ax[1].set_xlabel("Duplicated rows added (% of dataset)"); ax[1].set_ylabel("Accuracy inflation (pp)")
ax[1].set_title("(b) Inflation grows with duplication rate"); ax[1].legend(fontsize=7.5)
plt.tight_layout(); save("fig04_duplicate_leakage")
log("duplicate experiment done")

# =============================================================================
# 5. MODELS: tuning (inner CV) + nested CV (honest estimate) + one-shot test
# =============================================================================
MODELS = {
    "Dummy (class prior)": (Pipeline([("clf", DummyClassifier(strategy="prior"))]), None),
    "Logistic Regression": (Pipeline([("scale", StandardScaler()),
                                      ("clf", LogisticRegression(solver="liblinear", random_state=SEED, max_iter=5000))]),
                            {"clf__penalty": ["l1", "l2"], "clf__C": list(np.logspace(-3, 3, 13))}),
    "Decision Tree": (Pipeline([("clf", DecisionTreeClassifier(random_state=SEED))]),
                      {"clf__max_depth": [2, 3, 4, 5, 6, 8, None], "clf__min_samples_leaf": [1, 3, 5, 10],
                       "clf__criterion": ["gini", "entropy"]}),
    "Random Forest": (Pipeline([("clf", RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=1))]),
                      {"clf__max_depth": [None, 6], "clf__min_samples_leaf": [1, 3], "clf__max_features": ["sqrt", 0.3]}),
    "SVM (RBF)": (Pipeline([("scale", StandardScaler()),
                            ("clf", SVC(probability=True, random_state=SEED))]),
                  {"clf__C": [0.1, 1, 10, 100], "clf__gamma": ["scale", 0.003, 0.01, 0.03]}),
}
R["grid_sizes"] = {k: (int(np.prod([len(v) for v in g.values()])) if g else 0) for k, (_, g) in MODELS.items()}
inner = StratifiedKFold(5, shuffle=True, random_state=SEED + 1)
outer = StratifiedKFold(5, shuffle=True, random_state=SEED)
SCORING = {"accuracy": "accuracy", "precision": "precision", "recall": "recall", "f1": "f1", "roc_auc": "roc_auc"}

nested, fitted, test_res = {}, {}, {}
for name, (pipe, grid) in MODELS.items():
    if grid is None:
        cvres = cross_validate(pipe, Xtr, ytr, cv=outer, scoring=SCORING)
        est = clone(pipe).fit(Xtr, ytr); best_params, inner_auc = {}, 0.5
    else:
        gs = GridSearchCV(pipe, grid, cv=inner, scoring="roc_auc", n_jobs=1)
        cvres = cross_validate(gs, Xtr, ytr, cv=outer, scoring=SCORING)
        gs = GridSearchCV(pipe, grid, cv=inner, scoring="roc_auc", n_jobs=1, refit=True).fit(Xtr, ytr)
        est, best_params, inner_auc = gs.best_estimator_, gs.best_params_, gs.best_score_
    nested[name] = {k: [float(np.mean(cvres[f"test_{k}"])), float(np.std(cvres[f"test_{k}"], ddof=1))] for k in SCORING}
    fitted[name] = est
    p = est.predict_proba(Xte)[:, 1]; yhat = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(yte, yhat).ravel()
    test_res[name] = dict(
        params={k: (None if v is None else (float(v) if isinstance(v, (float, np.floating)) else v)) for k, v in best_params.items()},
        inner_cv_auc=float(inner_auc),
        train_acc=float(accuracy_score(ytr, est.predict(Xtr))),
        acc=float(accuracy_score(yte, yhat)), precision=float(precision_score(yte, yhat, zero_division=0)),
        recall=float(recall_score(yte, yhat)), f1=float(f1_score(yte, yhat)),
        roc_auc=float(roc_auc_score(yte, p)), pr_auc=float(average_precision_score(yte, p)),
        mcc=float(matthews_corrcoef(yte, yhat)), brier=float(brier_score_loss(yte, p)),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))
    test_res[name]["_p"] = p
    log(f"{name}: nested AUC {nested[name]['roc_auc'][0]:.4f} | test AUC {test_res[name]['roc_auc']:.4f} | params {best_params}")
R["nested"] = nested
R["test"] = {k: {kk: vv for kk, vv in v.items() if kk != "_p"} for k, v in test_res.items()}

# --- choose final model: best nested AUC, then prefer simplest within 1 SE ---
cand = {k: v for k, v in nested.items() if not k.startswith("Dummy")}
best_k = max(cand, key=lambda k: cand[k]["roc_auc"][0])
se = cand[best_k]["roc_auc"][1] / np.sqrt(5)
within = [k for k in cand if cand[k]["roc_auc"][0] >= cand[best_k]["roc_auc"][0] - se]
SIMPLICITY = ["Logistic Regression", "Decision Tree", "SVM (RBF)", "Random Forest"]
FINAL = [k for k in SIMPLICITY if k in within][0]
R["best_by_nested"] = best_k; R["within_1se"] = within; R["final_model"] = FINAL
R["nested_best_se"] = float(se)
log(f"FINAL MODEL = {FINAL}  (best by nested AUC: {best_k}; within 1 SE: {within})")
final_est = fitted[FINAL]

# --- Fig: confusion matrices (test) ---
show = ["Logistic Regression", "Decision Tree", "Random Forest", "SVM (RBF)"]
fig, axes = plt.subplots(2, 2, figsize=(6.4, 5.6))
for ax, m in zip(axes.ravel(), show):
    t = test_res[m]; cm = np.array([[t["tn"], t["fp"]], [t["fn"], t["tp"]]])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax, annot_kws={"size": 14, "weight": "bold"},
                xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"], linewidths=1, linecolor="white")
    ax.set_title(f"{m}\nacc {t['acc']*100:.1f}% | recall {t['recall']*100:.1f}%", fontsize=9)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
plt.tight_layout(); save("fig06_confusion_matrices")

# --- Fig: ROC (full + zoom) + PR (test) ---
palette = {"Dummy (class prior)": C_N, "Logistic Regression": C_D, "Decision Tree": C_A,
           "Random Forest": C_B, "SVM (RBF)": C_M}
fig, axx = plt.subplots(2, 2, figsize=(7.6, 6.4)); ax = [axx[0, 0], axx[0, 1], axx[1, 0], axx[1, 1]]
for m, c in palette.items():
    p = test_res[m]["_p"]
    kw = dict(color=c, lw=2.4 if m == FINAL else 1.5, ls="--" if m.startswith("Dummy") else "-")
    fpr, tpr, _ = roc_curve(yte, p)
    ax[0].plot(fpr, tpr, label=f"{m} (AUC {test_res[m]['roc_auc']:.3f})", **kw); ax[1].plot(fpr, tpr, **kw)
    pr, rc, _ = precision_recall_curve(yte, p)
    ax[2].plot(rc, pr, label=f"{m} (AP {test_res[m]['pr_auc']:.3f})", **kw); ax[3].plot(rc, pr, **kw)
ax[0].plot([0, 1], [0, 1], color="lightgrey", lw=.8)
ax[0].set_xlabel("False positive rate"); ax[0].set_ylabel("True positive rate (recall)"); ax[0].set_title("(a) ROC, held-out test set")
ax[0].legend(fontsize=7, loc="lower right"); ax[0].set_xlim(-.01, 1); ax[0].set_ylim(0, 1.02)
ax[1].set_xlim(-.005, 0.25); ax[1].set_ylim(0.70, 1.005); ax[1].set_xlabel("False positive rate"); ax[1].set_title("(b) ROC zoom: top-left corner")
ax[2].set_xlabel("Recall"); ax[2].set_ylabel("Precision"); ax[2].set_title("(c) Precision-recall, held-out test set")
ax[2].legend(fontsize=7, loc="lower left"); ax[2].set_ylim(0.3, 1.02); ax[2].set_xlim(0, 1.01)
ax[3].set_xlim(0.7, 1.005); ax[3].set_ylim(0.7, 1.005); ax[3].set_xlabel("Recall"); ax[3].set_title("(d) PR zoom: top-right corner")
plt.tight_layout(); save("fig07_roc_pr")

# =============================================================================
# 6. BIAS / VARIANCE DIAGNOSTICS
# =============================================================================
cv5 = StratifiedKFold(5, shuffle=True, random_state=SEED)
depths = list(range(1, 13))
tr_s, va_s = validation_curve(DecisionTreeClassifier(random_state=SEED), Xtr, ytr, param_name="max_depth",
                              param_range=depths, cv=cv5, scoring="accuracy")
Cs = np.logspace(-3, 3, 13)
lr_pipe = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(solver="liblinear", random_state=SEED, penalty="l2", max_iter=5000))])
tr_c, va_c = validation_curve(lr_pipe, Xtr, ytr, param_name="clf__C", param_range=Cs, cv=cv5, scoring="accuracy")
R["vc_dt"] = {"depths": depths, "train": tr_s.mean(1).tolist(), "cv": va_s.mean(1).tolist()}
R["vc_lr"] = {"C": Cs.tolist(), "train": tr_c.mean(1).tolist(), "cv": va_c.mean(1).tolist()}
sizes = np.linspace(.12, 1.0, 8)
ls_f = learning_curve(clone(final_est), Xtr, ytr, train_sizes=sizes, cv=cv5, scoring="accuracy", random_state=SEED, shuffle=True)
ls_t = learning_curve(DecisionTreeClassifier(random_state=SEED), Xtr, ytr, train_sizes=sizes, cv=cv5, scoring="accuracy", random_state=SEED, shuffle=True)
R["lc_final"] = {"n": ls_f[0].tolist(), "train": ls_f[1].mean(1).tolist(), "cv": ls_f[2].mean(1).tolist()}
R["lc_tree"] = {"n": ls_t[0].tolist(), "train": ls_t[1].mean(1).tolist(), "cv": ls_t[2].mean(1).tolist()}

fig, ax = plt.subplots(2, 2, figsize=(7.6, 6.0))
a = ax[0, 0]
a.plot(depths, tr_s.mean(1) * 100, "-o", color=C_D, label="Training accuracy", ms=4)
a.plot(depths, va_s.mean(1) * 100, "-o", color=C_M, label="Cross-validated accuracy", ms=4)
a.fill_between(depths, (va_s.mean(1) - va_s.std(1)) * 100, (va_s.mean(1) + va_s.std(1)) * 100, color=C_M, alpha=.15)
gap12 = (tr_s.mean(1)[-1] - va_s.mean(1)[-1]) * 100
a.annotate("", xy=(12.35, va_s.mean(1)[-1] * 100), xytext=(12.35, tr_s.mean(1)[-1] * 100), arrowprops=dict(arrowstyle="<->", color="k", lw=1))
a.text(12.0, 96.2, f"gap {gap12:.1f} pp\n(variance)", ha="right", fontsize=8.5)
a.annotate("depth 1:\nunder-fit", xy=(1, va_s.mean(1)[0] * 100), xytext=(1.8, 88.7), fontsize=8, arrowprops=dict(arrowstyle="->", color="grey"))
a.set_xlim(0.5, 12.9); a.set_xlabel("Decision-tree max depth"); a.set_ylabel("Accuracy (%)"); a.set_title("(a) Decision tree: complexity curve"); a.legend(fontsize=7.5, loc="lower right")
a.set_ylim(88, 101)
a = ax[0, 1]
a.semilogx(Cs, tr_c.mean(1) * 100, "-o", color=C_D, label="Training accuracy", ms=4)
a.semilogx(Cs, va_c.mean(1) * 100, "-o", color=C_M, label="Cross-validated accuracy", ms=4)
a.fill_between(Cs, (va_c.mean(1) - va_c.std(1)) * 100, (va_c.mean(1) + va_c.std(1)) * 100, color=C_M, alpha=.15)
a.set_xlabel("Inverse regularisation strength C (log scale)"); a.set_ylabel("Accuracy (%)")
a.set_title("(b) Logistic regression: regularisation curve"); a.legend(fontsize=7.5, loc="lower right"); a.set_ylim(85, 101)
for k, (a, lc, ttl) in enumerate([(ax[1, 0], ls_f, f"(c) Learning curve: {FINAL} (final)"), (ax[1, 1], ls_t, "(d) Learning curve: unpruned decision tree")]):
    a.plot(lc[0], lc[1].mean(1) * 100, "-o", color=C_D, label="Training", ms=4)
    a.plot(lc[0], lc[2].mean(1) * 100, "-o", color=C_M, label="Cross-validated", ms=4)
    a.fill_between(lc[0], (lc[2].mean(1) - lc[2].std(1)) * 100, (lc[2].mean(1) + lc[2].std(1)) * 100, color=C_M, alpha=.15)
    a.set_xlabel("Training-set size"); a.set_ylabel("Accuracy (%)"); a.set_title(ttl); a.legend(fontsize=7.5, loc="center" if k == 1 else "lower right", bbox_to_anchor=(0.5, 0.66) if k == 1 else None); a.set_ylim(84 if k == 1 else 88, 101.5)
plt.tight_layout(); save("fig08_bias_variance")
log("bias/variance done")

# =============================================================================
# 7. THRESHOLD ANALYSIS (chosen on out-of-fold predictions ONLY, then applied once to test)
# =============================================================================
oof = cross_val_predict(clone(final_est), Xtr, ytr, cv=StratifiedKFold(5, shuffle=True, random_state=SEED + 2),
                        method="predict_proba")[:, 1]
ths = np.linspace(0.02, 0.98, 97)
prec_t = [precision_score(ytr, oof >= t, zero_division=1) for t in ths]
rec_t = [recall_score(ytr, oof >= t) for t in ths]
f2_t = [fbeta_score(ytr, oof >= t, beta=2) for t in ths]
TARGET_RECALL = 0.98
ok = [t for t, r in zip(ths, rec_t) if r >= TARGET_RECALL]
THR = float(max(ok)) if ok else 0.5
R["threshold"] = THR
R["oof_at_default"] = dict(recall=float(recall_score(ytr, oof >= .5)), precision=float(precision_score(ytr, oof >= .5)))
R["oof_at_thr"] = dict(recall=float(recall_score(ytr, oof >= THR)), precision=float(precision_score(ytr, oof >= THR)))
p_fin = test_res[FINAL]["_p"]
def cm_metrics(yt, pr):
    tn, fp, fn, tp = confusion_matrix(yt, pr).ravel()
    return dict(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp), acc=float(accuracy_score(yt, pr)),
                precision=float(precision_score(yt, pr)), recall=float(recall_score(yt, pr)), f1=float(f1_score(yt, pr)))
R["test_default"] = cm_metrics(yte, (p_fin >= .5).astype(int))
R["test_thr"] = cm_metrics(yte, (p_fin >= THR).astype(int))

fig, ax = plt.subplots(1, 3, figsize=(8.4, 3.2), gridspec_kw={"width_ratios": [1.6, 1, 1]})
ax[0].plot(ths, rec_t, color=C_M, label="Recall (malignant)"); ax[0].plot(ths, prec_t, color=C_B, label="Precision")
ax[0].plot(ths, f2_t, color=C_D, ls="--", label="F2 (recall-weighted)")
t_f2 = float(np.mean(np.array(ths)[np.isclose(f2_t, np.max(f2_t), atol=1e-9)]))
ax[0].axvline(.5, color="grey", ls=":", lw=1); ax[0].axvline(THR, color=C_D, lw=1); ax[0].axvline(t_f2, color=C_A, lw=1.4, ls="--")
ax[0].text(.51, 1.005, "default 0.50", fontsize=7.5, color="grey"); ax[0].text(THR - .01, 1.005, f"recall≥98%: {THR:.2f}", fontsize=7.5, color=C_D, ha="right"); ax[0].text(t_f2 - .01, .585, f"F2-opt {t_f2:.2f}", fontsize=7.5, color="#B8860B", ha="right")
ax[0].set_xlabel("Decision threshold on P(malignant)"); ax[0].set_ylabel("Score"); ax[0].set_ylim(0.55, 1.035)
ax[0].set_title("(a) Out-of-fold trade-off (training data)"); ax[0].legend(fontsize=7.5, loc="lower center", bbox_to_anchor=(0.62, 0.02))
for a, key, ttl in [(ax[1], "test_default", "(b) Test @ 0.50"), (ax[2], "test_thr", f"(c) Test @ {THR:.2f}")]:
    t = R[key]; cm = np.array([[t["tn"], t["fp"]], [t["fn"], t["tp"]]])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=a, annot_kws={"size": 14, "weight": "bold"},
                xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"], linewidths=1, linecolor="white")
    a.set_title(ttl); a.set_xlabel("Predicted"); a.set_ylabel("Actual" if a is ax[1] else "")
plt.tight_layout(); save("fig09_threshold")

# =============================================================================
# 8. UNCERTAINTY ON THE TEST SET (bootstrap) - only 114 patients!
# =============================================================================
rng = np.random.RandomState(SEED)
yt = yte.values; pt = p_fin; B = 2000
bs = {k: [] for k in ["acc", "recall", "precision", "auc"]}
for _ in range(B):
    i = rng.randint(0, len(yt), len(yt))
    if yt[i].min() == yt[i].max(): continue
    ph = (pt[i] >= .5).astype(int)
    bs["acc"].append(accuracy_score(yt[i], ph)); bs["recall"].append(recall_score(yt[i], ph))
    bs["precision"].append(precision_score(yt[i], ph, zero_division=0)); bs["auc"].append(roc_auc_score(yt[i], pt[i]))
R["boot"] = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in bs.items()}
def cp_ci(k, n, a=0.05):
    lo = 0.0 if k == 0 else stats.beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - a / 2, k + 1, n - k)
    return [float(lo), float(hi)]
_t = R["test_default"] if "test_default" in R else None
R["exact_ci"] = dict(accuracy=cp_ci(_t["tp"] + _t["tn"], len(yt)), recall=cp_ci(_t["tp"], _t["tp"] + _t["fn"]),
                     precision=cp_ci(_t["tp"], _t["tp"] + _t["fp"]), specificity=cp_ci(_t["tn"], _t["tn"] + _t["fp"]))

# =============================================================================
# 9. MULTICOLLINEARITY: what it does to INTERPRETATION vs PREDICTION
# =============================================================================
best_lr = fitted["Logistic Regression"]
lr_params = {k.replace("clf__", ""): v for k, v in test_res["Logistic Regression"]["params"].items()}
R["lr_params"] = lr_params
COLL = ["mean radius", "mean perimeter", "mean area"]
coef_unreg, coef_tuned = [], []
rs = np.random.RandomState(SEED)
for _ in range(200):
    i = rs.randint(0, len(Xtr), len(Xtr))
    Xb, yb = Xtr.iloc[i], ytr.iloc[i]
    sc = StandardScaler().fit(Xb); Z = pd.DataFrame(sc.transform(Xb), columns=FEATS)
    m1 = LogisticRegression(C=1e3, max_iter=50000).fit(Z, yb)
    m2 = LogisticRegression(solver="liblinear", random_state=SEED, max_iter=5000, **lr_params).fit(Z, yb)
    coef_unreg.append([m1.coef_[0][FEATS.index(c)] for c in COLL]); coef_tuned.append([m2.coef_[0][FEATS.index(c)] for c in COLL])
coef_unreg, coef_tuned = np.array(coef_unreg), np.array(coef_tuned)
def flip(a):
    maj = np.sign(np.median(a, axis=0)); return (np.sign(a) != maj).mean(axis=0) * 100
R["coef_stab"] = dict(features=COLL,
                      unreg_sd=coef_unreg.std(0).tolist(), tuned_sd=coef_tuned.std(0).tolist(),
                      unreg_flip=flip(coef_unreg).tolist(), tuned_flip=flip(coef_tuned).tolist(),
                      unreg_med=np.median(coef_unreg, 0).tolist(), tuned_med=np.median(coef_tuned, 0).tolist())
fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.3), sharey=False)
for a, arr, ttl, col in [(ax[0], coef_unreg, "(a) Weakly regularised (C = 1000)", C_M), (ax[1], coef_tuned, f"(b) Tuned ({lr_params.get('penalty')}, C = {lr_params.get('C'):.3g})", C_B)]:
    bp = a.boxplot(arr, labels=[c.replace("mean ", "") for c in COLL], patch_artist=True, widths=.5, showfliers=False)
    for b in bp["boxes"]: b.set_facecolor(col); b.set_alpha(.55)
    for mline in bp["medians"]: mline.set_color(C_D)
    a.axhline(0, color="grey", lw=.8); a.set_title(ttl); a.set_ylabel("Standardised coefficient (200 bootstraps)")
plt.tight_layout(); save("fig05_coef_stability")

# --- feature-set ablation (repeated CV on training data; selection inside CV -> no leakage) ---
rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
def lr_clf(): return LogisticRegression(solver="liblinear", random_state=SEED, max_iter=5000, **lr_params)
variants = {
    f"All {len(FEATS)} features (baseline)": (list(FEATS), Pipeline([("scale", StandardScaler()), ("clf", lr_clf())])),
    f"Correlation-pruned (|r| ≤ 0.9) → {len(PRUNED)} features": (PRUNED, Pipeline([("scale", StandardScaler()), ("clf", lr_clf())])),
    "'Mean' block only → 10 features": (MEANS, Pipeline([("scale", StandardScaler()), ("clf", lr_clf())])),
    "Top-10 by ANOVA F (selected inside each fold)": (list(FEATS), Pipeline([("scale", StandardScaler()), ("sel", SelectKBest(f_classif, k=10)), ("clf", lr_clf())])),
    "All 30 + Yeo-Johnson power transform": (list(FEATS), Pipeline([("pt", PowerTransformer()), ("clf", lr_clf())])),
}
abl = {}
for k, (cols, pp) in variants.items():
    r_ = cross_validate(pp, Xtr[cols], ytr, cv=rcv, scoring={"roc_auc": "roc_auc", "recall": "recall", "accuracy": "accuracy"})
    abl[k] = {m: [float(r_[f"test_{m}"].mean()), float(r_[f"test_{m}"].std(ddof=1))] for m in ["roc_auc", "recall", "accuracy"]}
R["ablation"] = abl
log("ablation done")

# =============================================================================
# 10. ERROR ANALYSIS (out-of-fold predictions on ALL 569 patients -> more error cases to study)
# =============================================================================
oof_all = cross_val_predict(clone(final_est), X, y, cv=StratifiedKFold(10, shuffle=True, random_state=SEED),
                            method="predict_proba")[:, 1]
pred_all = (oof_all >= .5).astype(int)
grp = np.where((y == 1) & (pred_all == 1), "TP", np.where((y == 0) & (pred_all == 0), "TN", np.where((y == 0), "FP", "FN")))
Z_all = pd.DataFrame(Xs_all, columns=FEATS)
sev = Z_all[top6].mean(axis=1)                 # composite severity score = mean z of six key, non-redundant features
R["err"] = {g: dict(n=int((grp == g).sum()), mean_p=float(oof_all[grp == g].mean()) if (grp == g).any() else None,
                    mean_sev=float(sev[grp == g].mean()) if (grp == g).any() else None) for g in ["TN", "FP", "FN", "TP"]}
R["err_total"] = int(((grp == "FP") | (grp == "FN")).sum())
R["err_acc"] = float((grp == "TP").sum() + (grp == "TN").sum()) / len(y)
R["err_uncertain"] = int((((oof_all > .2) & (oof_all < .8)) & ((grp == "FP") | (grp == "FN"))).sum())
R["err_confident_wrong"] = int(((((oof_all >= .9) & (y == 0)) | ((oof_all <= .1) & (y == 1)))).sum())
R["oof_all_auc"] = float(roc_auc_score(y, oof_all))
R["oof_all_recall"] = float(recall_score(y, pred_all)); R["oof_all_precision"] = float(precision_score(y, pred_all))
# error profile of the top-6 features
prof = pd.DataFrame({g: Z_all.loc[grp == g, top6].mean() for g in ["TN", "FP", "FN", "TP"]})
R["err_profile"] = prof.round(2).to_dict()
# how many of the errors have every key feature within the other class's inter-quartile range?
# Test-set errors
te_pred = (p_fin >= .5).astype(int)
te_err = Xte.index[te_pred != yte.values]
R["test_errors"] = [dict(row=int(i), actual="Malignant" if yte.loc[i] == 1 else "Benign",
                         p_malignant=float(p_fin[list(Xte.index).index(i)])) for i in te_err]

pca = PCA(2, random_state=SEED).fit(Xs_all); P = pca.transform(Xs_all)
R["pca_var"] = pca.explained_variance_ratio_.tolist()
fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.6))
a = ax[0]
a.scatter(P[y == 0, 0], P[y == 0, 1], s=14, c=C_B, alpha=.5, label="Benign")
a.scatter(P[y == 1, 0], P[y == 1, 1], s=14, c=C_M, alpha=.5, label="Malignant")
fnm, fpm = grp == "FN", grp == "FP"
a.scatter(P[fnm, 0], P[fnm, 1], s=70, marker="X", c="k", label=f"False negative (n={fnm.sum()})")
a.scatter(P[fpm, 0], P[fpm, 1], s=70, marker="^", facecolors="none", edgecolors="k", linewidths=1.4, label=f"False positive (n={fpm.sum()})")
a.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.0f}% of variance)"); a.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.0f}%)")
a.set_title("(a) Errors sit in the class-overlap zone"); a.legend(fontsize=7.5, loc="upper right")
a = ax[1]
bins = np.linspace(0, 1, 21)
a.hist(oof_all[y == 0], bins=bins, color=C_B, alpha=.65, label="Actually benign")
a.hist(oof_all[y == 1], bins=bins, color=C_M, alpha=.65, label="Actually malignant")
a.set_yscale("log"); a.axvline(.5, color="grey", ls=":", label="default threshold 0.50"); a.axvline(THR, color=C_D, label=f"chosen threshold {THR:.2f}")
a.set_xlabel("Out-of-fold P(malignant)"); a.set_ylabel("Patients (log scale)"); a.set_title("(b) Model confidence by true class"); a.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.55, 1.0))
plt.tight_layout(); save("fig10_error_analysis")

# =============================================================================
# 11. Final-model coefficients (for interpretation) + versions
# =============================================================================
if FINAL == "Logistic Regression":
    co = pd.Series(final_est.named_steps["clf"].coef_[0], index=FEATS).sort_values(key=np.abs, ascending=False)
    R["coef_top"] = co.head(8).round(2).to_dict(); R["coef_nonzero"] = int((co != 0).sum())
R["versions"] = dict(python=".".join(map(str, __import__("sys").version_info[:3])), sklearn=sklearn.__version__,
                     pandas=pd.__version__, numpy=np.__version__)
R["runtime_s"] = float(time.time() - T0)
with open(os.path.join(RES, "results.json"), "w") as f:
    json.dump(R, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
log("ALL DONE")


# =============================================================================
# 12. SUPPLEMENTARY: operating points, review-band numbers, workflow diagram
# =============================================================================
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
R = json.load(open(os.path.join(RES, "results.json")))
raw = load_breast_cancer(as_frame=True)
X = raw.frame[list(raw.feature_names)]
y = 1 - raw.frame["target"]

E = {}
# zero-concavity patients: which class?
z = X["mean concavity"] == 0
E["zero_conc_n"] = int(z.sum()); E["zero_conc_malignant"] = int(y[z].sum())

# out-of-fold probabilities on all 569 with the final (tuned) logistic regression
p = R["lr_params"]
model = make_pipeline(StandardScaler(), LogisticRegression(solver="liblinear", random_state=SEED, max_iter=5000, **p))
oof = cross_val_predict(model, X, y, cv=StratifiedKFold(10, shuffle=True, random_state=SEED), method="predict_proba")[:, 1]
pred = (oof >= .5).astype(int)
err = pred != y.values
E["oof_acc_check"] = float(1 - err.mean()); E["oof_errors_check"] = int(err.sum())
for lo, hi in [(0.2, 0.8), (0.1, 0.9)]:
    band = (oof > lo) & (oof < hi)
    E[f"band_{lo}_{hi}"] = dict(n=int(band.sum()), pct=float(band.mean() * 100), errors=int((band & err).sum()),
                               err_share=float((band & err).sum() / err.sum() * 100))
    out = ~band
    E[f"band_{lo}_{hi}"]["acc_outside"] = float((pred[out] == y.values[out]).mean())
# same at chosen threshold
thr = R["threshold"]
predt = (oof >= thr).astype(int)
E["oof_all_thr"] = dict(fn=int(((predt == 0) & (y.values == 1)).sum()), fp=int(((predt == 1) & (y.values == 0)).sum()))
E["oof_all_def"] = dict(fn=int(((pred == 0) & (y.values == 1)).sum()), fp=int(((pred == 1) & (y.values == 0)).sum()))
# ---------------- operating points (thresholds chosen on TRAINING out-of-fold data only) ----------------
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, fbeta_score, confusion_matrix, accuracy_score
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.20, stratify=y, random_state=SEED)
oof_tr = cross_val_predict(model, Xtr, ytr, cv=StratifiedKFold(5, shuffle=True, random_state=SEED + 2), method="predict_proba")[:, 1]
p_te = model.fit(Xtr, ytr).predict_proba(Xte)[:, 1]
ths = np.linspace(0.02, 0.98, 97)
rec = np.array([recall_score(ytr, oof_tr >= t) for t in ths])
prec = np.array([precision_score(ytr, oof_tr >= t, zero_division=1) for t in ths])
f2 = np.array([fbeta_score(ytr, oof_tr >= t, beta=2) for t in ths])
t_f2 = float(np.mean(ths[np.isclose(f2, f2.max(), atol=1e-9)]))
t_r95 = float(max(t for t, r in zip(ths, rec) if r >= 0.95))
ops = {"default": 0.5, "f2_optimal": round(t_f2, 3), "recall_95": round(t_r95, 3), "recall_98": R["threshold"]}
E["ops"] = {}
for k, t in ops.items():
    o = oof_tr >= t; tp_ = ((o == 1) & (ytr.values == 1)).sum(); fp_ = ((o == 1) & (ytr.values == 0)).sum(); fn_ = ((o == 0) & (ytr.values == 1)).sum()
    pt = (p_te >= t).astype(int); tn, fp, fn, tp = confusion_matrix(yte, pt).ravel()
    E["ops"][k] = dict(thr=float(t), oof_recall=float(recall_score(ytr, o)), oof_precision=float(precision_score(ytr, o)),
                       oof_fp=int(fp_), oof_fn=int(fn_), te_tp=int(tp), te_fp=int(fp), te_fn=int(fn), te_tn=int(tn),
                       te_recall=float(tp / (tp + fn)), te_precision=float(tp / (tp + fp)) if tp + fp else None,
                       te_acc=float((tp + tn) / len(yte)))
json.dump(E, open(os.path.join(RES, "extras.json"), "w"), indent=1)
print(json.dumps(E, indent=1))

# ---------------- workflow diagram ----------------
plt.rcParams.update({"savefig.dpi": 200, "savefig.bbox": "tight"})
fig, ax = plt.subplots(figsize=(10.6, 4.1)); ax.set_xlim(0, 10.6); ax.set_ylim(0, 4.1); ax.axis("off")
TEAL, DARK, CORAL = "#2A9D8F", "#264653", "#E76F51"
W, H = 1.78, 1.0
xs = [0.1 + i * 2.12 for i in range(5)]
row1 = [("Raw data", "569 patients\n× 30 features", TEAL), ("Audit & EDA", "missing, duplicates,\noutliers, correlation", TEAL),
        ("Stratified split", "80 / 20, seed 42\ntest set locked away", TEAL), ("Training set", "455 patients\n(170 malignant)", DARK),
        ("Pipeline", "scale → model\n(fit inside each fold)", DARK)]
row2 = [("Inner 5-fold CV", "grid search, ROC-AUC", DARK), ("Outer 5-fold CV", "nested → honest\nestimate", DARK),
        ("Choose model", "best AUC, then simplest\nwithin 1 SE; threshold\nfrom out-of-fold data", DARK),
        ("One-shot test", "114 locked patients,\nexact CIs", CORAL), ("Error analysis", "who is missed and why;\nimprovement roadmap", CORAL)]
def box(x, y0, t, s, c):
    ax.add_patch(FancyBboxPatch((x, y0), W, H, boxstyle="round,pad=0.02,rounding_size=0.12", fc=c, ec="none"))
    ax.text(x + W / 2, y0 + H - 0.23, t, ha="center", va="center", color="white", fontsize=10.5, fontweight="bold")
    ax.text(x + W / 2, y0 + 0.36, s, ha="center", va="center", color="white", fontsize=8, linespacing=1.25)
def arrow(p0, p1, rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=11, lw=1.5, color="#555", connectionstyle=f"arc3,rad={rad}"))
y1, y2 = 2.75, 0.55
for i, (t, s, c) in enumerate(row1):
    box(xs[i], y1, t, s, c)
    if i: arrow((xs[i - 1] + W + 0.02, y1 + H / 2), (xs[i] - 0.02, y1 + H / 2))
for i, (t, s, c) in enumerate(row2):
    box(xs[i], y2, t, s, c)
    if i: arrow((xs[i - 1] + W + 0.02, y2 + H / 2), (xs[i] - 0.02, y2 + H / 2))
ym = 2.32
ax.plot([xs[4] + W / 2, xs[4] + W / 2, xs[0] + W / 2], [y1 - 0.02, ym, ym], color="#555", lw=1.5, solid_capstyle="round")
arrow((xs[0] + W / 2, ym), (xs[0] + W / 2, y2 + H + 0.03))
ax.text(5.3, 2.12, "training data only", ha="center", fontsize=8.5, color="#555", style="italic")
ax.text(xs[3] + W / 2, y2 - 0.22, "touched exactly once", ha="center", fontsize=8.5, color=CORAL, style="italic")
plt.savefig(os.path.join(FIG, "fig00_workflow.png")); plt.close()
