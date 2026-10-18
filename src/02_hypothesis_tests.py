"""Step 2 - Statistical hypothesis testing (every test is paired with a chart).

H1: Mean tumour size/shape features differ between malignant and benign masses.
H2: Malignant tumours are more variable (heteroscedastic) than benign ones.
H3: Feature distributions are non-normal -> non-parametric test needed (robustness).
H4: Diagnosis is associated with high concavity (chi-square on binned feature).
H5: Mean area differs across texture-tertile groups (one-way ANOVA / Kruskal).
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests
from common import *

df = pd.read_csv(PROC / "wdbc_clean.csv")
feat = [c for c in df.columns if c.endswith(("_mean", "_se", "_worst"))]
B, M = df[df.target == 0], df[df.target == 1]
out = {}
rng = np.random.default_rng(SEED)

# ---------------------------------------------------------- H1  Welch t-test
rows = []
for c in feat:
    a, b = M[c], B[c]
    t, p = stats.ttest_ind(a, b, equal_var=False)
    u, pu = stats.mannwhitneyu(a, b, alternative="two-sided")
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    d = (a.mean() - b.mean()) / sp
    # Welch-Satterthwaite CI for mean difference
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    dfw = se**4 / ((a.var(ddof=1) / len(a))**2 / (len(a) - 1) + (b.var(ddof=1) / len(b))**2 / (len(b) - 1))
    diff = a.mean() - b.mean()
    ci = (diff - stats.t.ppf(.975, dfw) * se, diff + stats.t.ppf(.975, dfw) * se)
    auc = u / (len(a) * len(b))        # = P(malignant > benign) (common-language effect)
    rows.append(dict(feature=c, mean_M=a.mean(), mean_B=b.mean(), diff=diff,
                     ci_lo=ci[0], ci_hi=ci[1], t=t, p_welch=p, U=u, p_mwu=pu,
                     cohen_d=d, auc=auc))
T = pd.DataFrame(rows)
T["p_welch_holm"] = multipletests(T.p_welch, method="holm")[1]
T["p_mwu_holm"] = multipletests(T.p_mwu, method="holm")[1]
T["sig_after_holm"] = T.p_welch_holm < .05
T = T.sort_values("cohen_d", key=np.abs, ascending=False).reset_index(drop=True)
T.to_csv(RES / "hypothesis_tests_all_features.csv", index=False)
out["H1_n_tests"] = len(T)
out["H1_n_sig_holm"] = int(T.sig_after_holm.sum())
out["H1_n_sig_raw"] = int((T.p_welch < .05).sum())
out["H1_not_sig"] = T[~T.sig_after_holm].feature.tolist()
out["H1_top"] = T.head(6)[["feature", "mean_M", "mean_B", "cohen_d", "t", "p_welch", "auc"]].round(4).to_dict("records")
out["H1_area_mean_ci"] = T[T.feature == "area_mean"][["diff", "ci_lo", "ci_hi"]].round(1).iloc[0].to_dict()
out["H1_area_worst_p"] = float(T[T.feature == "area_worst"].p_welch.iloc[0])
out["H1_welch_vs_mwu_agree"] = int(((T.p_welch_holm < .05) == (T.p_mwu_holm < .05)).sum())

# Fig 9: Test results: effect size forest with CI-ish + volcano-like
fig, ax = plt.subplots(1, 2, figsize=(13, 6))
Tt = T.head(15).iloc[::-1]
ax[0].barh(Tt.feature.str.replace("_", " "), Tt.cohen_d,
           color=[C_M if s else "#999" for s in Tt.sig_after_holm])
ax[0].axvline(.8, ls="--", c="k", lw=1); ax[0].text(.82, -0.6, "large effect (d=0.8)", fontsize=8)
ax[0].set(xlabel="Cohen's d (malignant - benign)", title="H1: Top-15 effect sizes")
ax[0].tick_params(axis="y", labelsize=8)
ax[1].scatter(T.cohen_d, -np.log10(T.p_welch_holm.clip(lower=1e-300)), c=[C_M if s else "#999" for s in T.sig_after_holm], s=40)
ax[1].axhline(-np.log10(.05), ls="--", c="k"); ax[1].text(T.cohen_d.min(), -np.log10(.05) + 2, "Holm-adjusted p = 0.05", fontsize=8)
ax[1].set(xlabel="Cohen's d", ylabel="-log10 (Holm-adjusted p)", title=f"H1: {out['H1_n_sig_holm']}/{len(T)} features significant after correction")
for _, r in T.head(4).iterrows():
    ax[1].annotate(r.feature.replace("_", " "), (r.cohen_d, -np.log10(max(r.p_welch_holm, 1e-300))), fontsize=7,
                   xytext=(-8, 5), textcoords="offset points", ha="right")
save(fig, "fig09_h1_effect_sizes.png")

# Fig 10: worked example - area_mean means with 95% CI & bootstrap distribution of diff
def boot(col, n=5000):
    return np.array([rng.choice(M[col], len(M)).mean() - rng.choice(B[col], len(B)).mean() for _ in range(n)])
bs = boot("area_mean")
out["H1_area_mean_boot_ci"] = [round(float(np.percentile(bs, 2.5)), 1), round(float(np.percentile(bs, 97.5)), 1)]
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
sns.histplot(bs, bins=50, color=C_B, ax=ax[0])
ax[0].axvline(0, c="k", ls="--"); ax[0].axvline(np.percentile(bs, 2.5), c=C_M); ax[0].axvline(np.percentile(bs, 97.5), c=C_M)
ax[0].set(xlabel="Mean area difference (malignant - benign), bootstrap", title="H1: Bootstrap 95% CI excludes 0")
sns.pointplot(data=df, x="label", y="area_mean", hue="label", palette=PALETTE, errorbar=("ci", 95),
              linestyles="", ax=ax[1], legend=False)
ax[1].set(xlabel="", ylabel="Mean tumour area (95% CI)", title="H1: Group means with 95% CI")
save(fig, "fig10_h1_bootstrap_ci.png")

# ---------------------------------------------------------- H2  Variance (Levene)
lev = []
for c in ["radius_mean", "area_mean", "texture_mean", "concavity_mean", "smoothness_mean", "area_worst"]:
    s, p = stats.levene(M[c], B[c], center="median")
    lev.append(dict(feature=c, var_M=M[c].var(), var_B=B[c].var(), ratio=M[c].var() / B[c].var(), levene_W=s, p=p))
L = pd.DataFrame(lev)
L["p_holm"] = multipletests(L.p, method="holm")[1]
L.to_csv(RES / "levene_tests.csv", index=False)
out["H2"] = L.to_dict("records")
fig, ax = plt.subplots(figsize=(7.5, 4.2))
ax.bar(L.feature.str.replace("_", " "), L.ratio, color=[C_M if p < .05 else "#999" for p in L.p_holm])
ax.axhline(1, c="k", ls="--"); ax.set_ylabel("Variance ratio (malignant / benign)")
ax.set_title("H2: Malignant tumours are far more variable"); ax.tick_params(axis="x", rotation=25)
for i, (r, p) in enumerate(zip(L.ratio, L.p_holm)):
    ax.text(i, r + .1, f"{r:.1f}x\np={p:.1e}", ha="center", fontsize=7)
save(fig, "fig11_h2_variance.png")

# ---------------------------------------------------------- H3  Normality
norm = []
for c in ["radius_mean", "area_mean", "concavity_mean", "area_se"]:
    for lab, g in [("Benign", B), ("Malignant", M)]:
        w, p = stats.shapiro(g[c])
        norm.append(dict(feature=c, group=lab, W=w, p=p, skew=stats.skew(g[c])))
N = pd.DataFrame(norm); N.to_csv(RES / "shapiro_tests.csv", index=False)
out["H3_n_reject"] = int((N.p < .05).sum()); out["H3_n_total"] = len(N)
fig, ax = plt.subplots(2, 4, figsize=(14, 7), gridspec_kw=dict(hspace=.5, wspace=.3))
for j, c in enumerate(["radius_mean", "area_mean", "concavity_mean", "area_se"]):
    for i, (lab, g) in enumerate([("Benign", B), ("Malignant", M)]):
        stats.probplot(g[c], dist="norm", plot=ax[i, j])
        ax[i, j].get_lines()[0].set(color=PALETTE[lab], ms=3)
        ax[i, j].get_lines()[1].set(color="k")
        p = N[(N.feature == c) & (N.group == lab)].p.iloc[0]
        ax[i, j].set_title(f"{c.replace('_',' ')} - {lab}\nShapiro p={p:.1e}", fontsize=9)
        ax[i, j].set_xlabel(""); ax[i, j].set_ylabel("")
fig.suptitle("H3: Q-Q plots - heavy right tails violate normality, so we confirm with Mann-Whitney", y=1.01)
save(fig, "fig12_h3_qq.png")

# ---------------------------------------------------------- H4  Chi-square
df["concavity_bin"] = pd.qcut(df.concavity_worst, 4, labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"])
ct = pd.crosstab(df.concavity_bin, df.label)
chi2, p, dof, exp = stats.chi2_contingency(ct)
cramer = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))
out["H4"] = dict(chi2=round(chi2, 1), dof=int(dof), p=float(p), cramers_v=round(cramer, 3),
                 min_expected=round(float(exp.min()), 1),
                 pct_malignant_by_quartile=(100 * ct["Malignant"] / ct.sum(axis=1)).round(1).to_dict())
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ct.plot(kind="bar", stacked=True, color=[C_B, C_M], ax=ax[0], rot=0)
ax[0].set(xlabel="Worst-concavity quartile", ylabel="Patients", title="H4: Counts by concavity quartile")
pct = 100 * ct["Malignant"] / ct.sum(axis=1)
ax[1].bar(pct.index, pct.values, color=C_M)
for i, v in enumerate(pct.values): ax[1].text(i, v + 1.5, f"{v:.0f}%", ha="center")
ax[1].set(ylabel="% malignant", ylim=(0, 110),
          title=f"Malignancy rate rises steeply (chi2={chi2:.0f}, p={p:.1e}, V={cramer:.2f})")
ax[1].tick_params(axis="x", rotation=0)
save(fig, "fig13_h4_chi_square.png")

# ---------------------------------------------------------- H5  ANOVA
df["texture_grp"] = pd.qcut(df.texture_mean, 3, labels=["Low", "Medium", "High"])
groups = [g.area_mean.values for _, g in df.groupby("texture_grp", observed=True)]
F, pa = stats.f_oneway(*groups)
H, pk = stats.kruskal(*groups)
grand = df.area_mean.mean()
eta2 = sum(len(g) * (g.mean() - grand) ** 2 for g in groups) / ((df.area_mean - grand) ** 2).sum()
from statsmodels.stats.multicomp import pairwise_tukeyhsd
tuk = pairwise_tukeyhsd(df.area_mean, df.texture_grp.astype(str))
tk = pd.DataFrame(tuk._results_table.data[1:], columns=tuk._results_table.data[0])
tk.to_csv(RES / "tukey_h5.csv", index=False)
out["H5"] = dict(F=round(F, 2), p_anova=float(pa), H=round(H, 2), p_kruskal=float(pk), eta_sq=round(eta2, 3),
                 tukey=tk.round(3).astype(str).to_dict("records"))
fig, ax = plt.subplots(figsize=(7, 4.4))
sns.violinplot(data=df, x="texture_grp", y="area_mean", hue="texture_grp", palette="Blues", inner="quartile",
               ax=ax, legend=False, order=["Low", "Medium", "High"])
ax.set(xlabel="Texture tertile", ylabel="Mean area",
       title=f"H5: ANOVA F={F:.1f}, p={pa:.1e}, eta²={eta2:.3f} (medium effect)")
save(fig, "fig14_h5_anova.png")

json.dump(out, open(RES / "step2_metrics.json", "w"), indent=2, default=float)
print(json.dumps({k: out[k] for k in ["H1_n_sig_holm", "H1_n_sig_raw", "H1_not_sig", "H3_n_reject", "H4", "H5"]}, indent=1, default=str))
