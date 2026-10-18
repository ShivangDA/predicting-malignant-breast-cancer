"""Step 3 - Machine-learning, multicollinearity handling, feature selection, thresholds.

All preprocessing (scaling, selection) lives INSIDE sklearn Pipelines so it is
fitted on training folds only -> no data leakage.
"""
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve,
                             average_precision_score, f1_score, precision_score, recall_score,
                             accuracy_score, brier_score_loss, ConfusionMatrixDisplay)
from sklearn.model_selection import (StratifiedKFold, cross_validate, cross_val_predict,
                                     train_test_split, learning_curve, GridSearchCV)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from common import *

warnings.filterwarnings("ignore")
df = pd.read_csv(PROC / "wdbc_clean.csv")
feat = [c for c in df.columns if c.endswith(("_mean", "_se", "_worst"))]
vif_keep = json.load(open(RES / "vif_pruned_features.json"))
X, y = df[feat], df.target.values
out = {}

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.25, stratify=y, random_state=SEED)
out["n_train"], out["n_test"] = len(Xtr), len(Xte)
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

# ------------------------------------------------------- A. model comparison
models = {
    "Logistic Regression (L2)": make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)),
    "Logistic Regression (L1)": make_pipeline(StandardScaler(), LogisticRegression(penalty="l1", solver="liblinear", C=0.5, random_state=SEED)),
    "SVM (RBF)": make_pipeline(StandardScaler(), SVC(C=1, probability=True, random_state=SEED)),
    "Random Forest": RandomForestClassifier(400, random_state=SEED, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(random_state=SEED),
}
scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
rows = []
for name, m in models.items():
    r = cross_validate(m, Xtr, ytr, cv=cv, scoring=scoring, return_train_score=True)
    rows.append({"model": name, **{s: r[f"test_{s}"].mean() for s in scoring},
                 **{f"{s}_sd": r[f"test_{s}"].std() for s in scoring},
                 "train_auc": r["train_roc_auc"].mean(), "train_acc": r["train_accuracy"].mean()})
CV = pd.DataFrame(rows).set_index("model")
CV.round(4).to_csv(RES / "cv_model_comparison.csv")
out["cv"] = CV.round(4).reset_index().to_dict("records")

# --------------------------- B. multicollinearity & feature selection study
base = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
fs_sets = {
    "All 30 features": feat,
    f"VIF-pruned ({len(vif_keep)})": vif_keep,
    "Mean features only (10)": [c for c in feat if c.endswith("_mean")],
    "Top-5 by ANOVA-F": None,
    "RFE-10 (logistic)": None,
    "L1-selected": None,
}
fsr = []
sel_info = {}
for name in fs_sets:
    if name == "Top-5 by ANOVA-F":
        pipe = make_pipeline(StandardScaler(), SelectKBest(f_classif, k=5), LogisticRegression(max_iter=5000))
    elif name == "RFE-10 (logistic)":
        pipe = make_pipeline(StandardScaler(), RFE(LogisticRegression(max_iter=5000), n_features_to_select=10),
                             LogisticRegression(max_iter=5000))
    elif name == "L1-selected":
        pipe = make_pipeline(StandardScaler(), LogisticRegression(penalty="l1", solver="liblinear", C=0.5, random_state=SEED))
    else:
        cols = fs_sets[name]
        from sklearn.compose import ColumnTransformer
        pipe = make_pipeline(ColumnTransformer([("s", StandardScaler(), cols)]), LogisticRegression(max_iter=5000))
    r = cross_validate(pipe, Xtr, ytr, cv=cv, scoring=["accuracy", "recall", "roc_auc", "f1"])
    fsr.append({"feature_set": name, "accuracy": r["test_accuracy"].mean(), "recall": r["test_recall"].mean(),
                "f1": r["test_f1"].mean(), "auc": r["test_roc_auc"].mean(), "auc_sd": r["test_roc_auc"].std()})
    pipe.fit(Xtr, ytr)
    if name == "RFE-10 (logistic)":
        sel_info[name] = [f for f, s in zip(feat, pipe[1].support_) if s]
    if name == "Top-5 by ANOVA-F":
        sel_info[name] = [f for f, s in zip(feat, pipe[1].get_support()) if s]
    if name == "L1-selected":
        co = pipe[-1].coef_.ravel()
        sel_info[name] = [f for f, c in zip(feat, co) if abs(c) > 1e-6]
FS = pd.DataFrame(fsr)
FS.round(4).to_csv(RES / "feature_selection_comparison.csv", index=False)
out["fs"] = FS.round(4).to_dict("records")
out["fs_selected"] = sel_info
out["l1_n_selected"] = len(sel_info["L1-selected"])

# Coefficient instability under multicollinearity: bootstrap coefficient SD / sign flips.
# Crossed design: feature set (all 30 vs VIF-pruned) x regularisation (weak C=100 vs tuned/strong C=0.3)
rng = np.random.default_rng(SEED)
def boot_coefs(cols, C, B=200):
    Xs = StandardScaler().fit_transform(Xtr[cols]); cs = []
    for _ in range(B):
        idx = rng.integers(0, len(Xs), len(Xs))
        m = LogisticRegression(C=C, max_iter=20000).fit(Xs[idx], ytr[idx]); cs.append(m.coef_.ravel())
    return np.array(cs)
def flips(bc):
    return float(np.mean([(np.sign(bc[:, j]) != np.sign(np.median(bc[:, j]))).mean() for j in range(bc.shape[1])]) * 100)
stab = {}
for fs_name, cols_ in [("All 30", feat), (f"VIF-pruned ({len(vif_keep)})", vif_keep)]:
    for C in (100.0, 0.3):
        bc = boot_coefs(cols_, C)
        stab[(fs_name, C)] = bc
out["coef_stability"] = [dict(feature_set=k[0], C=k[1], mean_coef_sd=round(float(v.std(0).mean()), 3),
                              mean_abs_coef=round(float(np.abs(np.median(v, 0)).mean()), 3),
                              sign_flip_pct=round(flips(v), 1)) for k, v in stab.items()]

fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
fs_plot = FS.set_index("feature_set")
ax[0].barh(fs_plot.index[::-1], fs_plot.auc[::-1], xerr=fs_plot.auc_sd[::-1], color=C_B, capsize=3)
ax[0].set_xlim(.96, 1.005); ax[0].set(xlabel="5-fold CV ROC-AUC (mean ± SD)", title="Feature subsets: AUC differences are within noise")
labels = [f"{k[0]}\nC={k[1]:g}" for k in stab]
bp = ax[1].boxplot([v.std(0) for v in stab.values()], tick_labels=labels, patch_artist=True)
for patch, k in zip(bp["boxes"], stab): patch.set_facecolor(C_M if k[1] == 100.0 else C_B)
ax[1].set_ylabel("Bootstrap SD of standardised coefficient"); ax[1].tick_params(axis="x", labelsize=8)
ax[1].set_title("Coefficient stability: regularisation matters more than pruning")
save(fig, "fig15_feature_selection.png")

# --------------------------------------- C. final model + tuning + test set
grid = GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
                    {"logisticregression__C": np.logspace(-2, 2, 9)}, scoring="roc_auc", cv=cv).fit(Xtr, ytr)
best = grid.best_estimator_
out["best_C"] = float(grid.best_params_["logisticregression__C"]); out["best_cv_auc"] = round(grid.best_score_, 4)
pte = best.predict_proba(Xte)[:, 1]
pred = (pte >= .5).astype(int)
tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
out["test_default"] = dict(accuracy=accuracy_score(yte, pred), precision=precision_score(yte, pred),
                           recall=recall_score(yte, pred), f1=f1_score(yte, pred), auc=roc_auc_score(yte, pte),
                           ap=average_precision_score(yte, pte), brier=brier_score_loss(yte, pte),
                           tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))
# bootstrap 95% CI for test AUC & recall
bt = []
for _ in range(2000):
    i = rng.integers(0, len(yte), len(yte))
    if len(set(yte[i])) < 2: continue
    bt.append((roc_auc_score(yte[i], pte[i]), recall_score(yte[i], pred[i]), accuracy_score(yte[i], pred[i])))
bt = np.array(bt)
out["test_ci"] = dict(auc=np.percentile(bt[:, 0], [2.5, 97.5]).round(4).tolist(),
                      recall=np.percentile(bt[:, 1], [2.5, 97.5]).round(4).tolist(),
                      accuracy=np.percentile(bt[:, 2], [2.5, 97.5]).round(4).tolist())

# held-out comparison of all models
rocs, hold = {}, []
for name, m in models.items():
    mm = clone(m).fit(Xtr, ytr); p = mm.predict_proba(Xte)[:, 1]
    rocs[name] = roc_curve(yte, p)
    hold.append(dict(model=name, auc=roc_auc_score(yte, p), accuracy=accuracy_score(yte, p > .5),
                     recall=recall_score(yte, p > .5), precision=precision_score(yte, p > .5)))
HOLD = pd.DataFrame(hold).round(4); HOLD.to_csv(RES / "heldout_comparison.csv", index=False)
out["heldout"] = HOLD.to_dict("records")

# Fig 16 ROC (all models) + PR (final)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
cols = sns.color_palette("colorblind", len(models))
for (name, (f, t, _)), c in zip(rocs.items(), cols):
    ax[0].plot(f, t, lw=2, color=c, label=f"{name} (AUC {HOLD.set_index('model').loc[name,'auc']:.3f})")
ax[0].plot([0, 1], [0, 1], "k--", lw=1); ax[0].legend(fontsize=8, loc="lower right")
ax[0].set(xlabel="False-positive rate", ylabel="True-positive rate (recall)", title="ROC curves - held-out test set")
pr, rc, _ = precision_recall_curve(yte, pte)
ax[1].plot(rc, pr, color=C_M, lw=2); ax[1].set(xlabel="Recall", ylabel="Precision", ylim=(.5, 1.02),
        title=f"Precision-recall - tuned logistic (AP {out['test_default']['ap']:.3f})")
save(fig, "fig16_roc_pr.png")

# Fig 17 confusion matrix + calibration
fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
ConfusionMatrixDisplay(confusion_matrix(yte, pred), display_labels=["Benign", "Malignant"]).plot(ax=ax[0], cmap="Blues", colorbar=False)
ax[0].set_title(f"Confusion matrix @ threshold 0.5 (n={len(yte)})"); ax[0].grid(False)
fp_, mp_ = calibration_curve(yte, pte, n_bins=6, strategy="quantile")
ax[1].plot(mp_, fp_, "o-", color=C_B, label="Tuned logistic"); ax[1].plot([0, 1], [0, 1], "k--", label="Perfect")
ax[1].set(xlabel="Mean predicted probability", ylabel="Observed fraction malignant",
          title=f"Calibration (Brier {out['test_default']['brier']:.3f})"); ax[1].legend()
save(fig, "fig17_confusion_calibration.png")

# Fig 18 over/underfitting: learning curve
tsz, trs, vas = learning_curve(best, X, y, cv=cv, train_sizes=np.linspace(.1, 1, 8), scoring="roc_auc", n_jobs=-1)
fig, ax = plt.subplots(figsize=(6.6, 4.4))
ax.plot(tsz, trs.mean(1), "o-", color=C_B, label="Training AUC"); ax.plot(tsz, vas.mean(1), "o-", color=C_M, label="Validation AUC")
ax.fill_between(tsz, vas.mean(1) - vas.std(1), vas.mean(1) + vas.std(1), color=C_M, alpha=.15)
ax.set(xlabel="Training examples", ylabel="ROC-AUC", ylim=(.9, 1.005), title="Learning curve: small, closing gap = low overfitting")
ax.legend(); save(fig, "fig18_learning_curve.png")
out["learning_gap_final"] = round(float(trs.mean(1)[-1] - vas.mean(1)[-1]), 4)

# --------------------------------------- D. threshold analysis (cost-sensitive)
oof = cross_val_predict(best, Xtr, ytr, cv=cv, method="predict_proba")[:, 1]   # threshold chosen on TRAIN OOF only
ths = np.linspace(.02, .98, 97)
def metrics_at(yt, p, t, cfn=10, cfp=1):
    pr_ = (p >= t).astype(int); tn, fp, fn, tp = confusion_matrix(yt, pr_, labels=[0, 1]).ravel()
    return dict(thr=t, recall=tp / max(tp + fn, 1), specificity=tn / max(tn + fp, 1), precision=tp / max(tp + fp, 1),
                f1=f1_score(yt, pr_), fn=int(fn), fp=int(fp), cost=cfn * fn + cfp * fp)
TH_oof = pd.DataFrame([metrics_at(ytr, oof, t) for t in ths])
TH_te = pd.DataFrame([metrics_at(yte, pte, t) for t in ths])
TH_te.round(4).to_csv(RES / "threshold_analysis_test.csv", index=False)
t_cost = float(TH_oof.loc[TH_oof.cost.idxmin(), "thr"])
t_f1 = float(TH_oof.loc[TH_oof.f1.idxmax(), "thr"])
# highest threshold with recall >=0.98 on OOF
t_rec = float(TH_oof[TH_oof.recall >= .98].thr.max())
out["thr_choice"] = dict(cost_opt_10to1=t_cost, f1_opt=t_f1, recall98=t_rec)
sc = {}
for nm, t in [("Default 0.50", .5), (f"Cost-optimal (10:1) {t_cost:.2f}", t_cost),
              (f"Recall>=98% {t_rec:.2f}", t_rec), (f"F1-optimal {t_f1:.2f}", t_f1)]:
    sc[nm] = metrics_at(yte, pte, t)
out["thr_scenarios_test"] = {k: {a: round(float(b), 4) for a, b in v.items()} for k, v in sc.items()}
# sensitivity to cost ratio
cost_ratio = []
for r in [1, 2, 5, 10, 20, 50]:
    cc = pd.DataFrame([metrics_at(ytr, oof, t, cfn=r) for t in ths]); tt = cc.loc[cc.cost.idxmin(), "thr"]
    m = metrics_at(yte, pte, tt, cfn=r); cost_ratio.append(dict(cost_ratio=r, thr=round(tt, 2), fn=m["fn"], fp=m["fp"], recall=round(m["recall"], 4)))
out["cost_ratio_table"] = cost_ratio

fig = plt.figure(figsize=(12, 9))
gs_ = fig.add_gridspec(2, 2, hspace=.38, wspace=.25)
ax = [fig.add_subplot(gs_[0, 0]), fig.add_subplot(gs_[0, 1]), fig.add_subplot(gs_[1, :])]
ax[0].plot(TH_te.thr, TH_te.recall, color=C_M, lw=2, label="Recall (sensitivity)")
ax[0].plot(TH_te.thr, TH_te.specificity, color=C_B, lw=2, label="Specificity")
ax[0].plot(TH_te.thr, TH_te.precision, color="#009E73", lw=2, label="Precision")
for nm, c in [("cost", t_cost), ("default", .5)]:
    ax[0].axvline(c, ls="--", c="k", lw=1)
ax[0].text(t_cost + .01, .02, f"cost-opt {t_cost:.2f}", fontsize=8, transform=ax[0].get_xaxis_transform()); ax[0].text(.51, .10, "default 0.5", fontsize=8, transform=ax[0].get_xaxis_transform())
ax[0].set(xlabel="Decision threshold", ylabel="Metric", title="Threshold trade-offs (test set)"); ax[0].legend(fontsize=8, loc="lower left")
ax[1].plot(TH_te.thr, TH_te.fn, color=C_M, lw=2, label="False negatives (missed cancers)")
ax[1].plot(TH_te.thr, TH_te.fp, color=C_B, lw=2, label="False positives (unnecessary biopsies)")
ax[1].set(xlabel="Decision threshold", ylabel="Number of patients", title="Error counts vs threshold"); ax[1].legend(fontsize=8)
for r, c in zip([1, 10, 50], ["#999", C_M, "k"]):
    cst = [metrics_at(yte, pte, t, cfn=r)["cost"] for t in ths]
    ax[2].plot(ths, np.array(cst) / max(cst), color=c, lw=2, label=f"FN costs {r}x FP")
ax[2].set(xlabel="Decision threshold", ylabel="Normalised total cost", title="Total cost vs threshold: the minimum shifts left as missed cancers get costlier"); ax[2].legend(fontsize=8)
save(fig, "fig19_threshold_analysis.png")

# --------------------------------------- E. interpretation
co = pd.Series(best[-1].coef_.ravel(), index=feat)
out["top_coefs"] = co.reindex(co.abs().sort_values(ascending=False).index).head(8).round(3).to_dict()
pi = permutation_importance(best, Xte, yte, scoring="roc_auc", n_repeats=30, random_state=SEED)
PI = pd.Series(pi.importances_mean, index=feat).sort_values(ascending=False)
PI.round(4).to_csv(RES / "permutation_importance.csv", header=["auc_drop"])
out["perm_top"] = PI.head(8).round(4).to_dict()
# Importance is split across correlated features: grouped permutation importance
groups = {"Size (radius, perimeter, area)": [c for c in feat if c.split("_")[0] in ("radius", "perimeter", "area")],
          "Shape (concavity, compactness)": [c for c in feat if c.startswith(("concav", "concave", "compact"))],
          "Texture": [c for c in feat if c.startswith("texture")],
          "Smooth/Symmetry/Fractal": [c for c in feat if c.startswith(("smooth", "symm", "fractal"))]}
rngp = np.random.default_rng(SEED); gi = {}
base_auc = roc_auc_score(yte, best.predict_proba(Xte)[:, 1])
for g, cols_ in groups.items():
    d = []
    for _ in range(40):
        Xp = Xte.copy(); perm = rngp.permutation(len(Xp)); Xp[cols_] = Xp[cols_].values[perm]
        d.append(base_auc - roc_auc_score(yte, best.predict_proba(Xp)[:, 1]))
    gi[g] = round(float(np.mean(d)), 4)
out["group_importance"] = gi
fig, ax = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw=dict(wspace=.75))
t8 = PI.head(10).iloc[::-1]; ax[0].barh(t8.index.str.replace("_", " "), t8.values, color=C_B)
ax[0].set(xlabel="Drop in test AUC when shuffled", title="Single-feature importance (diluted)")
gs = pd.Series(gi).sort_values()
ax[1].barh(gs.index, gs.values, color=C_M); ax[1].set(xlabel="Drop in test AUC when group is shuffled", title="Grouped importance: what drives the model")
ax[0].tick_params(axis="y", labelsize=8); ax[0].xaxis.set_major_locator(plt.MaxNLocator(4))
save(fig, "fig20_importance.png")

# errors
err = Xte.copy(); err["true"] = yte; err["p_malignant"] = pte
wrong = err[(pte >= .5).astype(int) != yte]
wrong.round(3).to_csv(RES / "misclassified_test_cases.csv")
out["n_wrong"] = len(wrong)


# human-review band: auto-clear below t_rec, auto-flag above 0.5, everything between -> human review
band = (pte >= t_rec) & (pte < .5)
out["review_band"] = dict(lo=t_rec, hi=.5, n=int(band.sum()), pct=round(100 * band.mean(), 1),
                          malignant_in_band=int(yte[band].sum()),
                          missed_below=int(((pte < t_rec) & (yte == 1)).sum()),
                          cleared_below=int((pte < t_rec).sum()),
                          flagged_above=int((pte >= .5).sum()), fp_above=int(((pte >= .5) & (yte == 0)).sum()))
pd.DataFrame({"y_true": yte, "p_malignant": pte}).round(5).to_csv(RES / "test_predictions.csv", index=False)
json.dump(out, open(RES / "step3_metrics.json", "w"), indent=2, default=float)
print(json.dumps({k: out[k] for k in ["cv", "fs", "best_C", "test_default", "test_ci", "heldout", "thr_choice", "thr_scenarios_test",
                                        "cost_ratio_table", "coef_stability", "group_importance", "learning_gap_final", "l1_n_selected"]}, indent=1, default=float))
