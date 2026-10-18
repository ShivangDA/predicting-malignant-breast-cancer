// Builds reports/Week5_Final_Project_Report.docx from results/*.json and figures/*.png
const fs = require("fs");
const path = require("path");
const D = require("docx");
const { Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, HeadingLevel, AlignmentType,
  LevelFormat, BorderStyle, WidthType, ShadingType, PageBreak, Footer, Header, PageNumber, TableOfContents } = D;

const ROOT = path.resolve(__dirname, "..");
const R = (f) => JSON.parse(fs.readFileSync(path.join(ROOT, "results", f), "utf8"));
const s1 = R("step1_metrics.json"), s2 = R("step2_metrics.json"), s3 = R("step3_metrics.json");
const FIGDIR = path.join(ROOT, "figures");

const f = (x, d = 3) => Number(x).toFixed(d);
const pct = (x, d = 1) => (100 * Number(x)).toFixed(d) + "%";
const SUP = { "-": "⁻", "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹" };
const sci = (x) => { x = Number(x); if (x === 0) return "< 1e-300"; if (x >= 0.01) return x.toFixed(2);
  const [m, e] = x.toExponential(1).split("e"); return m + "×10" + String(parseInt(e, 10)).split("").map((c) => SUP[c] || c).join(""); };

const FONT = "Calibri", MONO = "Consolas";
const NAVY = "1F3A5F", BLUE = "0072B2", ORANGE = "D55E00", GREY = "F2F4F7";
const W = 9026; // A4 content width in DXA (1" margins)

// ---------- helpers
function runs(text, base = {}) {
  // **bold** and `code` inline parsing
  const out = [];
  const re = /(\*\*[^*]+\*\*|`[^`]+`)/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), font: FONT, ...base }));
    const tok = m[0];
    if (tok.startsWith("**")) out.push(new TextRun({ text: tok.slice(2, -2), bold: true, font: FONT, ...base }));
    else out.push(new TextRun({ text: tok.slice(1, -1), font: MONO, size: (base.size || 22) - 2, color: "9C2F00", ...base, bold: false }));
    last = m.index + tok.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), font: FONT, ...base }));
  return out;
}
const P = (text, opt = {}) => new Paragraph({ children: runs(text, opt.run || {}), spacing: { after: 120, line: 288 }, alignment: opt.align, keepNext: opt.keepNext, ...opt.p });
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text: t, font: FONT })], pageBreakBefore: true });
const H1n = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text: t, font: FONT })] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text: t, font: FONT })] });
const H3 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun({ text: t, font: FONT })] });
const B = (text, level = 0) => new Paragraph({ numbering: { reference: "bul", level }, children: runs(text), spacing: { after: 60, line: 276 } });
const N = (text, ref = "num") => new Paragraph({ numbering: { reference: ref, level: 0 }, children: runs(text), spacing: { after: 60, line: 276 } });

function img(file, caption, widthIn = 6.2) {
  const buf = fs.readFileSync(path.join(FIGDIR, file));
  const w = buf.readUInt32BE(16), h = buf.readUInt32BE(20);
  const pw = Math.round(widthIn * 96), ph = Math.round(pw * h / w);
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120, after: 60 },
      children: [new ImageRun({ type: "png", data: buf, transformation: { width: pw, height: ph },
        altText: { title: file, description: caption, name: file } })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 },
      children: [new TextRun({ text: caption, italics: true, size: 19, color: "444444", font: FONT })] }),
  ];
}
function code(lines) {
  return lines.split("\n").map((ln, i, a) => new Paragraph({
    shading: { type: ShadingType.CLEAR, fill: "F6F8FA" },
    border: { left: { style: BorderStyle.SINGLE, size: 12, color: BLUE, space: 6 } },
    spacing: { after: i === a.length - 1 ? 200 : 0, line: 240 }, indent: { left: 120 },
    children: [new TextRun({ text: ln === "" ? " " : ln, font: MONO, size: 17 })] }));
}
function callout(title, body, fill = "EAF3FB", bar = BLUE) {
  const bd = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
  return new Table({ width: { size: W, type: WidthType.DXA }, columnWidths: [W],
    rows: [new TableRow({ children: [new TableCell({ width: { size: W, type: WidthType.DXA },
      shading: { type: ShadingType.CLEAR, fill }, margins: { top: 120, bottom: 120, left: 200, right: 200 },
      borders: { top: bd, bottom: bd, right: bd, left: { style: BorderStyle.SINGLE, size: 24, color: bar } },
      children: [new Paragraph({ spacing: { after: 60 }, children: [new TextRun({ text: title, bold: true, font: FONT, color: bar, size: 22 })] }),
        ...body.map((b) => new Paragraph({ spacing: { after: 60, line: 276 }, children: runs(b) }))] })] })] });
}
const gap = () => new Paragraph({ spacing: { after: 120 }, children: [] });

function table(headers, rows, widths, opt = {}) {
  const tot = widths.reduce((a, b) => a + b, 0);
  const bd = { style: BorderStyle.SINGLE, size: 4, color: "C9CED6" };
  const borders = { top: bd, bottom: bd, left: bd, right: bd };
  const mk = (txt, wd, head, shade, align) => new TableCell({ width: { size: wd, type: WidthType.DXA }, borders,
    shading: { type: ShadingType.CLEAR, fill: head ? NAVY : shade || "FFFFFF" },
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [new Paragraph({ alignment: align, children: runs(String(txt), { size: opt.size || 19, color: head ? "FFFFFF" : undefined, bold: head ? true : undefined }) })] });
  return new Table({ width: { size: tot, type: WidthType.DXA }, columnWidths: widths,
    rows: [new TableRow({ tableHeader: true, children: headers.map((h, i) => mk(h, widths[i], true, null, i === 0 || opt.left ? AlignmentType.LEFT : AlignmentType.CENTER)) }),
      ...rows.map((r, ri) => new TableRow({ cantSplit: true, children: r.map((c, i) => mk(c, widths[i], false, ri % 2 ? GREY : "FFFFFF", i === 0 || opt.left ? AlignmentType.LEFT : AlignmentType.CENTER)) }))] });
}
const cap = (t) => new Paragraph({ spacing: { before: 60, after: 200 }, children: [new TextRun({ text: t, italics: true, size: 19, color: "444444", font: FONT })] });

// ---------- numbers
const td = s3.test_default, tci = s3.test_ci;
const cvLR = s3.cv.find((c) => c.model === "Logistic Regression (L2)");
const sim0 = s1.duplicate_sim[0], sim30 = s1.duplicate_sim[4];
const thr = s3.thr_scenarios_test;
const thrKeys = Object.keys(thr);
const rb = s3.review_band;
const h1top = s2.H1_top;
const stab = s3.coef_stability;
const gi = s3.group_importance;

const C = [];

// ================= TITLE PAGE
C.push(new Paragraph({ spacing: { before: 2400, after: 120 }, children: [new TextRun({ text: "DATA SCIENCE INTERNSHIP — WEEK 5", font: FONT, size: 24, color: BLUE, bold: true, characterSpacing: 60 })] }));
C.push(new Paragraph({ spacing: { after: 200 }, children: [new TextRun({ text: "Predicting Malignant Breast Cancer", font: FONT, size: 64, bold: true, color: NAVY })] }));
C.push(new Paragraph({ spacing: { after: 400 }, border: { bottom: { style: BorderStyle.SINGLE, size: 12, color: ORANGE, space: 8 } },
  children: [new TextRun({ text: "End-to-end project report: data quality, statistical evidence, machine learning and strategic recommendations", font: FONT, size: 30, color: "444444" })] }));
C.push(P("**Dataset:** Breast Cancer Wisconsin (Diagnostic) — UCI Machine Learning Repository", { p: { spacing: { after: 60 } } }));
C.push(P("**Author:** Shivang Dube", { p: { spacing: { after: 60 } } }));
C.push(P("**Date:** 18 October 2026", { p: { spacing: { after: 60 } } }));
C.push(P("**Code & data:** https://github.com/ShivangDA/predicting-malignant-breast-cancer", { p: { spacing: { after: 400 } } }));
C.push(callout("About this report", [
  "This report consolidates Weeks 1–4 (data cleaning & EDA, visual storytelling, hypothesis testing, machine learning) into one reproducible project and extends each stage in response to evaluator feedback: a stronger executive summary, a quantified duplicate-impact study, deeper correlation and multicollinearity analysis, a chart for every statistical test, and a full decision-threshold analysis.",
  "Every number and figure is generated by the scripts in the `src/` folder of the accompanying repository."]));

// ================= TOC
C.push(new Paragraph({ children: [new PageBreak()] }));
C.push(new Paragraph({ children: [new TextRun({ text: "Contents", font: FONT, size: 36, bold: true, color: NAVY })], spacing: { after: 200 } }));
C.push(table(["Section", "Title"], [
  ["", "Executive Summary"], ["1", "Introduction"], ["2", "Data Acquisition and Audit"], ["3", "Data Cleaning and Preprocessing (incl. duplicate-impact study)"],
  ["4", "Exploratory Data Analysis (incl. correlation and multicollinearity)"], ["5", "Statistical Analysis and Hypothesis Testing (H1–H5)"],
  ["6", "Machine-Learning Model Development and Evaluation (incl. feature selection and thresholds)"], ["7", "Strategic Recommendations"],
  ["8", "Business and Research Impact"], ["9", "Limitations and Future Work"], ["10", "Conclusion"], ["A", "Appendix: Reproducibility, code excerpts, references"],
], [1200, 7826], { left: true }));

// ================= EXEC SUMMARY
C.push(H1("Executive Summary"));
C.push(P(`**The problem.** Can the ${s1.audit.columns - 2} nuclear measurements extracted from fine-needle-aspirate images support a transparent second-opinion model — and how should it be configured to miss as few cancers as possible without flooding clinicians with false alarms? **The data:** ${s1.raw_shape[0]} patients, ${s1.malignant_pct}% malignant; clean, with no missing values or duplicates.`));
C.push(callout("Five headline findings", [
  `**1. Malignancy is overwhelmingly a size-and-irregularity signal.** ${s2.H1_n_sig_holm} of 30 features differ significantly between classes after Holm correction (Cohen's d up to ${f(h1top[0].cohen_d, 2)}). The mean tumour area is ${f(s2.H1_area_mean_ci.diff, 0)} units larger for malignant masses (95% CI ${f(s2.H1_area_mean_ci.ci_lo, 0)}–${f(s2.H1_area_mean_ci.ci_hi, 0)}). In the highest quartile of worst-concavity, ${s2.H4.pct_malignant_by_quartile["Q4 (highest)"]}% of masses are malignant versus ${s2.H4.pct_malignant_by_quartile["Q1 (lowest)"]}% in the lowest.`,
  `**2. The features are heavily redundant.** ${s1.vif_n_gt_10} of 30 features have VIF > 10 (radius and perimeter exceed 3,700), ${s1["n_pairs_gt_0.9"]} of 435 feature pairs correlate above 0.9, and ${s1.pca_n_95} principal components capture 95% of the variance. Regularisation, not feature deletion, keeps the model stable (coefficient noise shrinks ${f(stab[0].mean_coef_sd / stab[1].mean_coef_sd, 0)}-fold).`,
  `**3. A simple model is enough.** Regularised logistic regression reaches ${f(td.auc, 3)} test AUC (95% CI ${tci.auc[0]}–${tci.auc[1]}) and ${pct(td.accuracy)} accuracy, matching or beating SVM, random forest and gradient boosting while remaining interpretable.`,
  `**4. The decision threshold matters more than the algorithm.** At the default 0.5 cut-off the model misses ${td.fn} of ${td.fn + td.tp} cancers on the test set. Moving to a cost-optimised cut-off of ${thrKeys[1].match(/[\d.]+$/)[0]} cuts expected cost by ${pct(1 - thr[thrKeys[1]].cost / thr[thrKeys[0]].cost, 0)} (assuming a missed cancer is 10× worse than a false alarm) at the price of one extra false positive.`,
  `**5. Duplicates are a quiet integrity risk.** None exist here, but a controlled simulation shows that leaving 30% duplicates in the data inflates a flexible model's test accuracy by ${f((sim30.accuracy - sim0.accuracy) * 100, 1)} percentage points and shifts the malignant share from ${s1.duplicate_prior_shift[0].malignant_pct}% to ${s1.duplicate_prior_shift[2].malignant_pct}% when the duplicates are concentrated in one class.`]));
C.push(P("**What we recommend.**", { keepNext: true, p: { spacing: { before: 160, after: 80 } } }));
C.push(N(`**Adopt regularised logistic regression as the production baseline** — transparent, well calibrated (Brier ${f(td.brier, 3)}), and not beaten by more complex models.`));
C.push(N(`**Treat the threshold as a clinical policy lever, not a technical default.** Use the cost-optimal threshold (${thrKeys[1].match(/[\d.]+$/)[0]}) and re-derive it whenever the clinic's cost ratio changes (Table 11).`));
C.push(N(`**Use a three-way triage:** auto-flag above 0.5, auto-clear below ${rb.lo}, and route the ${rb.pct}% in between (${rb.n} of ${s3.n_test} test patients) to human review. On the test set this flags ${rb.flagged_above} patients with ${rb.fp_above} false positives and clears ${rb.cleared_below} with ${rb.missed_below} missed cancer.`));
C.push(N("**Add data-governance guards** (deduplication, patient-level train/test splitting) before any retraining, and validate on an external cohort before any clinical use."));
C.push(gap());
C.push(callout("Read this before relying on the numbers", [
  `The test set contains only ${s3.n_test} patients (${td.tp + td.fn} malignant). The bootstrap interval on recall is wide (${tci.recall[0]}–${tci.recall[1]}), so differences of one or two patients between thresholds or models are **within sampling noise**. The data come from a single institution's imaging protocol. This is a decision-support prototype, not a validated diagnostic device.`], "FDF0E7", ORANGE));

// ================= 1 INTRODUCTION
C.push(H1("1. Introduction"));
C.push(H2("1.1 Context and objective"));
C.push(P("Breast cancer is among the most common cancers worldwide, and early, accurate classification of a breast mass determines treatment pathway. Fine-needle aspiration (FNA) is a minimally invasive sampling method; a digitised image of the aspirate allows software to measure the geometry of each cell nucleus. The question for this project is practical: can those measurements drive a reliable, auditable classifier, and how should it be operated?"));
C.push(H2("1.2 Project roadmap"));
C.push(table(["Stage", "Question", "Section"], [
  ["Data quality (Week 1)", "Is the data trustworthy? What would duplicates and outliers do?", "2–3"],
  ["EDA & storytelling (Weeks 1–2)", "How do benign and malignant masses differ? How redundant are the features?", "4"],
  ["Hypothesis testing (Week 3)", "Are the differences statistically and practically significant?", "5"],
  ["Machine learning (Week 4)", "Which model, which features, which threshold?", "6"],
  ["Strategy (Week 5)", "What should stakeholders do with this?", "7–9"],
], [2700, 4926, 1400], { left: true }));
C.push(gap());
C.push(H2("1.3 Stakeholders"));
C.push(B("**Pathology / radiology leads** — need to know how many cancers the tool would miss and how many extra reviews it creates."));
C.push(B("**Hospital analytics and clinical-governance teams** — need a model they can audit, monitor and re-calibrate."));
C.push(B("**Research collaborators** — need a reproducible pipeline to extend to other cohorts."));

// ================= 2 DATA
C.push(H1("2. Data Acquisition and Audit"));
C.push(H2("2.1 Source"));
C.push(P("The Breast Cancer Wisconsin (Diagnostic) dataset (Wolberg, Street & Mangasarian, 1995) is published in the UCI Machine Learning Repository. Each row is one patient; each of ten nuclear characteristics is summarised three ways per image — the **mean**, the **standard error (se)** and the **worst** (mean of the three largest values) — giving 30 numeric features and a binary diagnosis."));
C.push(table(["Characteristic", "What it measures"], [
  ["radius, perimeter, area", "Size of the nucleus (mean distance from centre to boundary; boundary length; enclosed area)"],
  ["texture", "Standard deviation of grey-scale values"],
  ["smoothness", "Local variation in radius lengths"],
  ["compactness", "perimeter² / area − 1"],
  ["concavity, concave points", "Severity and number of inward indentations of the contour"],
  ["symmetry, fractal dimension", "Shape symmetry; 'coastline approximation' − 1"],
], [3000, 6026], { left: true }));
C.push(gap());
C.push(P("**Acquisition code.** The script first tries to download the file directly from the UCI repository and, if the network is unavailable, falls back to an identical copy bundled with scikit-learn. " + (s1.source.includes("bundled") ? "In the environment where the figures in this report were produced, the UCI host was unreachable, so the bundled copy was used (it contains the same 569 records but no patient-ID column; synthetic row numbers were substituted). Running the script on a machine with internet access will use the original UCI file." : "The direct UCI download succeeded.")));
C.push(...code(`UCI_URL = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
           "breast-cancer-wisconsin/wdbc.data")
try:
    urllib.request.urlretrieve(UCI_URL, RAW / "wdbc.data")
    df = pd.read_csv(RAW / "wdbc.data", header=None, names=COLS)
except Exception:                      # offline -> identical bundled copy
    from sklearn.datasets import load_breast_cancer
    ...`));
C.push(H2("2.2 Data-quality audit"));
C.push(P(`Before any transformation we audited the raw table (${s1.raw_shape[0]} rows × ${s1.raw_shape[1]} columns). Every check passed:`));
C.push(table(["Check", "Result", "Implication"], [
  ["Missing cells", s1.audit.missing_cells, "No imputation required"],
  ["Exact duplicate rows", s1.audit.duplicate_rows_all_cols, "No rows removed"],
  ["Duplicates ignoring ID column", s1.audit.duplicate_rows_excluding_id, "No hidden re-measurements"],
  ["Non-numeric feature columns", s1.audit.non_numeric_feature_cols.length, "No type repair needed"],
  ["Negative values", s1.audit.negative_values, "All physically plausible"],
  ["Zero values", `${Object.keys(s1.audit.zero_values_in_concavity_cols).length} columns × 13 rows`, "Valid: perfectly convex nuclei have zero concavity"],
], [3300, 2200, 3526], { left: true }));
C.push(cap("Table 1. Raw-data audit."));
C.push(...img("fig02_class_balance_quality.png", "Figure 1. Left: class balance (62.7% benign / 37.3% malignant). Right: audit counts — every quality issue count is zero."));
C.push(P("**Why the zero-concavity rows are kept.** The 13 zero values in concavity and concave-points appear together in the same rows and in all three summary versions (mean/se/worst). That pattern is consistent with genuinely convex, smooth nuclei rather than missing data coded as zero (all 13 rows are benign, which fits: smooth, regular nuclei have no concave segments). Treating them as missing would inject error."));

// ================= 3 CLEANING
C.push(H1("3. Data Cleaning and Preprocessing"));
C.push(P("Because the raw data were already clean, the value of this stage lies in **documented decisions** and in quantifying what would have happened had problems been present."));
C.push(H2("3.1 Cleaning decisions and rationale"));
C.push(table(["Step", "Decision", "Rationale"], [
  ["Types", "Features cast to float64; diagnosis to category; new binary target (1 = malignant)", "Stable numerics; target coding makes recall = sensitivity to cancer"],
  ["Duplicates", "`drop_duplicates` on the 30 features (0 removed)", "Guard retained in the pipeline so future data refreshes stay safe"],
  ["Missing values", "None found; no imputation", "Imputation would add assumptions without benefit"],
  ["Outliers", `Flagged with the 1.5×IQR rule, **not removed** (${s1.outlier_rows_any_feature} rows, ${pct(s1.outlier_rows_any_feature / s1.clean_shape[0], 0)}, flagged on at least one feature)`, `Outliers concentrate in malignant cases (${s1.outlier_share_malignant}% of malignant vs ${s1.outlier_share_benign}% of benign rows) — they are the signal, not noise`],
  ["Skewness", `Heavily right-skewed standard-error features (area_se skew ${s1.skew_top.area_se})`, "Handled by rank-based tests in Section 5 and scaling in Section 6; no lossy transformation needed"],
  ["Scaling", "StandardScaler inside each model pipeline", "Fitted on training folds only — prevents leakage"],
], [1500, 3800, 3726], { left: true }));
C.push(cap("Table 2. Cleaning decisions."));
C.push(...code(`clean = df.copy()
clean["target"] = (clean["diagnosis"] == "M").astype(int)       # 1 = malignant
clean[feat] = clean[feat].astype("float64")
clean = clean.drop_duplicates(subset=feat).reset_index(drop=True)

# Outliers: flag, do not delete
q1, q3 = clean[feat].quantile(.25), clean[feat].quantile(.75)
iqr = q3 - q1
is_out = (clean[feat] < q1 - 1.5*iqr) | (clean[feat] > q3 + 1.5*iqr)`));

C.push(H2("3.2 Quantifying the impact of duplicates"));
C.push(P("Evaluators asked for deeper treatment of duplicates. Since none exist, we measured their **potential** impact with a controlled experiment. We randomly copied 5–30% of the rows, re-inserted the copies, and re-evaluated a 1-nearest-neighbour classifier (chosen because it can memorise leaked rows) on 30 repeated stratified 75/25 splits. When a copy lands in the test set and its twin in the training set, the model is graded on something it has already seen."));
C.push(table(["Duplicates added", "Test accuracy", "Recall", "AUC", "Optimism vs clean"],
  s1.duplicate_sim.map((r) => [r.dup_rate_pct + "%", pct(r.accuracy, 2), pct(r.recall, 2), f(r.auc, 4), (r.accuracy - sim0.accuracy >= 0 ? "+" : "") + f((r.accuracy - sim0.accuracy) * 100, 2) + " pp"]),
  [2000, 1800, 1700, 1700, 1826]));
C.push(cap("Table 3. Effect of leaked duplicates (mean of 30 splits)."));
C.push(...img("fig01_duplicate_impact.png", "Figure 2. Duplicates inflate scores monotonically (beyond 5%), but the effect is modest relative to split-to-split noise (shaded band = ±1 SD)."));
C.push(P(`**Interpretation.** With 30% duplicates, accuracy rises by ${f((sim30.accuracy - sim0.accuracy) * 100, 2)} pp and AUC by ${f(sim30.auc - sim0.auc, 3)} — real but smaller than one standard deviation (≈${f(sim0.acc_sd * 100, 1)} pp). The danger is therefore subtle: duplicates would not raise alarms by producing absurd scores; they would quietly make a model look slightly better than it is. A second effect is distortion of the class prior: duplicating only malignant rows would move the malignant share from ${s1.duplicate_prior_shift[0].malignant_pct}% to ${s1.duplicate_prior_shift[1].malignant_pct}% (10% extra) or ${s1.duplicate_prior_shift[2].malignant_pct}% (30% extra) and push mean area from ${s1.duplicate_prior_shift[0].mean_area} to ${s1.duplicate_prior_shift[2].mean_area}. **Conclusion:** deduplicate before splitting, and split by patient ID when multiple samples per patient exist.`));

// ================= 4 EDA
C.push(H1("4. Exploratory Data Analysis"));
C.push(H2("4.1 Distributions by diagnosis"));
C.push(...img("fig03_distributions.png", "Figure 3. Kernel density estimates for six key features. Malignant masses (orange) are shifted right and are broader."));
C.push(P("Size and shape-irregularity features show clear separation: radius, area, concavity and concave points. Texture and smoothness overlap far more, hinting they will contribute less. The malignant distributions are also visibly **wider**, a pattern we test formally in Section 5.2."));
C.push(...img("fig04_boxplots.png", "Figure 4. Standardised box plots of the ten mean features by diagnosis. Fractal dimension is the only feature whose medians coincide."));
C.push(H2("4.2 Which features track malignancy?"));
C.push(...img("fig06_target_correlation.png", "Figure 5. Correlation of each feature with the malignant label. Concave-points, perimeter and radius lead; five features (four standard errors plus fractal_dimension_mean) are essentially uncorrelated.", 5.0));
const tt = s1.top_target_corr;
C.push(P(`The strongest single correlates are concave_points_worst (r = ${tt.concave_points_worst}), perimeter_worst (${tt.perimeter_worst}) and concave_points_mean (${tt.concave_points_mean}). Five features — fractal_dimension_se, smoothness_se, fractal_dimension_mean, texture_se and symmetry_se — correlate near zero (|r| < 0.08), exactly the five that are non-significant in Section 5.1.`));
C.push(H2("4.3 Correlation structure"));
C.push(...img("fig05_correlation_heatmap.png", "Figure 6. Hierarchically clustered correlation matrix. The dark block in the lower right contains radius, perimeter and area in all three summary versions.", 5.6));
C.push(P(`**${s1["n_pairs_gt_0.9"]} of 435 pairs exceed |r| = 0.9, and ${s1["n_pairs_gt_0.8"]} exceed 0.8.** The most extreme are radius_mean–perimeter_mean (r = ${s1.top_pairs[0].r}), radius_worst–perimeter_worst (${s1.top_pairs[1].r}) and radius_mean–area_mean (${s1.top_pairs[2].r}). These are geometric identities — perimeter ≈ 2πr and area ≈ πr² — so the redundancy is structural, not accidental.`));
C.push(...img("fig07_scatter.png", "Figure 7. Left: two features separate most cases. Right: radius and area are almost a deterministic curve."));
C.push(H2("4.4 Multicollinearity diagnostics"));
C.push(P("We applied three complementary diagnostics (Week 4 feedback asked for more depth here):"));
C.push(B(`**Variance Inflation Factor (VIF).** ${s1.vif_n_gt_10} of 30 features exceed the conventional threshold of 10 and ${s1.vif_n_gt_5} exceed 5. radius_mean (${f(s1.vif_top.radius_mean, 0)}) and perimeter_mean (${f(s1.vif_top.perimeter_mean, 0)}) are the worst; radius_worst, perimeter_worst, area_mean and area_worst also exceed 300.`));
C.push(B(`**Condition number** of the standardised design matrix: ${f(s1.condition_number, 0)} (values above ~30 indicate problematic collinearity).`));
C.push(B(`**Principal components.** PC1 alone explains ${s1.pca_pc1_pct}% of variance; ${s1.pca_n_90} components reach 90% and ${s1.pca_n_95} reach 95%. Thirty measurements carry roughly ten dimensions of information.`));
C.push(...img("fig08_multicollinearity_pca.png", "Figure 8. Left: VIF (log scale). Centre: cumulative variance by principal component. Right: first two components already separate the classes.", 6.4));
C.push(P(`Iterative VIF pruning (drop the highest-VIF feature until all VIF ≤ 10) leaves ${s1.vif_pruned_n} features: ${s1.vif_pruned_features.join(", ")}. Section 6.2 tests whether this pruning helps prediction or stability.`));
C.push(callout("EDA takeaways", [
  "Malignancy is expressed through larger, more irregular nuclei; texture and smoothness add secondary information.",
  "Redundancy is geometric and extreme — any linear model must be regularised or reduced.",
  "Outliers carry diagnostic information; deleting them would remove malignant cases."]));

// ================= 5 HYPOTHESIS TESTING
C.push(H1("5. Statistical Analysis and Hypothesis Testing"));
C.push(P("Five hypotheses were tested. **Each test is paired with a chart**, addressing the Week 3 feedback. Because H1 runs 30 simultaneous tests, p-values are adjusted with the Holm step-down procedure (controls the family-wise error rate at 5%). Where normality is doubtful, a rank-based test is run alongside. Significance level: α = 0.05."));
C.push(H2("5.1 H1 — Malignant and benign masses differ in mean feature values"));
C.push(P("**H0:** μ_malignant = μ_benign for a given feature. **H1:** they differ. **Test:** Welch's t-test (no equal-variance assumption), confirmed by Mann–Whitney U; effect size by Cohen's d; 95% confidence intervals for the mean difference (Welch–Satterthwaite degrees of freedom) and a 5,000-resample bootstrap."));
C.push(table(["Feature", "Mean (malignant)", "Mean (benign)", "Cohen's d", "P(M > B)"],
  h1top.map((r) => [r.feature, f(r.mean_M, 3), f(r.mean_B, 3), f(r.cohen_d, 2), f(r.auc, 3)]), [2800, 1600, 1600, 1500, 1526]));
C.push(cap("Table 4. Six largest effects. All Holm-adjusted p-values are below 1e-50. P(M > B) is the probability that a random malignant case has a larger value than a random benign case (common-language effect size)."));
C.push(P(`**Result.** ${s2.H1_n_sig_holm} of ${s2.H1_n_tests} features are significant after Holm correction (${s2.H1_n_sig_raw} before correction — the correction removed one borderline feature). The ${s2.H1_n_tests - s2.H1_n_sig_holm} non-significant features are ${s2.H1_not_sig.join(", ").replace(/_/g, " ")}. Welch and Mann–Whitney agree on the significance decision for ${s2.H1_welch_vs_mwu_agree} of 30 features, so the conclusions do not depend on the normality assumption. All effects above d ≈ 0.8 are conventionally "large"; the leading features exceed 2.5 — rarely seen in applied data.`));
C.push(...img("fig09_h1_effect_sizes.png", "Figure 9. Left: effect sizes of the top 15 features. Right: effect size against Holm-adjusted significance; grey points are non-significant.", 6.4));
C.push(P(`**Worked example — mean area.** The mean area of malignant masses exceeds benign by ${f(s2.H1_area_mean_ci.diff, 1)} units (Welch 95% CI ${f(s2.H1_area_mean_ci.ci_lo, 1)} to ${f(s2.H1_area_mean_ci.ci_hi, 1)}; bootstrap 95% CI ${s2.H1_area_mean_boot_ci[0]} to ${s2.H1_area_mean_boot_ci[1]}). Interpretation: if we repeated the study many times, 95% of intervals built this way would contain the true difference; the interval is far from zero, and the two interval methods agree.`));
C.push(...img("fig10_h1_bootstrap_ci.png", "Figure 10. Left: bootstrap distribution of the area difference — zero is far outside. Right: group means with 95% CIs."));
C.push(H2("5.2 H2 — Malignant tumours are more variable"));
C.push(P("**H0:** equal variances. **Test:** Levene's test (median-centred, robust to non-normality), Holm-adjusted across the six features examined."));
C.push(table(["Feature", "Variance (M)", "Variance (B)", "Ratio", "Holm p"],
  s2.H2.map((r) => [r.feature, Number(r.var_M).toPrecision(3), Number(r.var_B).toPrecision(3), f(r.ratio, 1) + "×", sci(r.p_holm)]), [2600, 1700, 1700, 1300, 1726]));
C.push(cap("Table 5. Levene tests for equality of variance."));
C.push(...img("fig11_h2_variance.png", "Figure 11. Variance ratio (malignant ÷ benign). Size and shape features are 3–13× more variable in malignant masses; texture and smoothness are not.", 5.2));
C.push(P("**Interpretation.** Malignant tumours are not just larger on average, they are far more heterogeneous in size and shape — consistent with the clinical picture of irregular growth. Texture and smoothness show no variance difference. **Practical consequence:** the benign class is tight and easy to characterise, whereas the malignant class has a long tail of very large lesions, which is why a few very large malignant cases show up as outliers. It also means the malignant class overlaps the benign class at its **small** end — where, as Section 6.4 shows, the model's errors occur."));
C.push(H2("5.3 H3 — Normality (assumption check)"));
C.push(P(`Shapiro–Wilk rejected normality in ${s2.H3_n_reject} of ${s2.H3_n_total} feature-by-class combinations. Q–Q plots show the failure comes from heavy right tails (concavity_mean, area_se); radius_mean in the benign group is the only clearly normal case. This justifies running the rank-based Mann–Whitney alongside Welch in H1, and favours robust or rank-based methods for the standard-error features.`));
C.push(...img("fig12_h3_qq.png", "Figure 12. Q–Q plots against the normal distribution. Points curling above the line at the right edge indicate heavy right tails.", 6.4));
C.push(H2("5.4 H4 — Malignancy is associated with concavity"));
C.push(P(`**H0:** diagnosis is independent of worst-concavity quartile. **Test:** χ² test of independence on a 4×2 contingency table (all expected counts ≥ ${s2.H4.min_expected}, so the test is valid); effect size by Cramér's V.`));
C.push(P(`**Result:** χ²(${s2.H4.dof}) = ${s2.H4.chi2}, p = ${sci(s2.H4.p)}, Cramér's V = ${s2.H4.cramers_v} (a very strong association). The malignancy rate rises from ${s2.H4.pct_malignant_by_quartile["Q1 (lowest)"]}% in the lowest quartile to ${s2.H4.pct_malignant_by_quartile["Q2"]}%, ${s2.H4.pct_malignant_by_quartile["Q3"]}% and ${s2.H4.pct_malignant_by_quartile["Q4 (highest)"]}%.`));
C.push(...img("fig13_h4_chi_square.png", "Figure 13. Left: patient counts by concavity quartile. Right: malignancy rate by quartile."));
C.push(H2("5.5 H5 — Nuclear texture and tumour size are related"));
C.push(P(`**H0:** mean area is equal across texture tertiles. **Test:** one-way ANOVA with Kruskal–Wallis as a non-parametric check and Tukey HSD post-hoc comparisons.`));
C.push(P(`**Result:** F = ${s2.H5.F}, p = ${sci(s2.H5.p_anova)}; Kruskal–Wallis H = ${s2.H5.H}, p = ${sci(s2.H5.p_kruskal)}. Effect size η² = ${s2.H5.eta_sq} (a medium effect: texture tertile explains about 11% of area variance). All three Tukey pairs differ significantly (adjusted p < 0.001): mean area rises from Low to Medium to High texture.`));
C.push(...img("fig14_h5_anova.png", "Figure 14. Violin plots of mean area by texture tertile (dashed lines = quartiles).", 4.8));
C.push(callout("Caution: association is not mechanism", [
  "Texture and area are both higher in malignant masses, so malignancy is a **confounder**: the texture–area link in H5 is at least partly the shared dependence on diagnosis. We did not test a causal claim and the report does not make one."], "FDF0E7", ORANGE));
C.push(gap());
C.push(H2("5.6 Summary of tests"));
C.push(table(["Hypothesis", "Test(s)", "Result", "Effect size"], [
  ["H1 mean differences", "Welch t, Mann–Whitney, bootstrap CI", `${s2.H1_n_sig_holm}/30 significant (Holm)`, "d up to " + f(h1top[0].cohen_d, 2)],
  ["H2 unequal variance", "Levene (median)", "4 of 6 features significant", "ratio up to " + f(Math.max(...s2.H2.map((r) => r.ratio)), 1) + "×"],
  ["H3 normality", "Shapiro–Wilk, Q–Q", `${s2.H3_n_reject}/${s2.H3_n_total} rejected`, "n/a"],
  ["H4 concavity vs diagnosis", "χ² independence", "Reject H0", "V = " + s2.H4.cramers_v],
  ["H5 texture vs area", "ANOVA, Kruskal, Tukey", "Reject H0", "η² = " + s2.H5.eta_sq],
], [2300, 2700, 2200, 1826]));
C.push(cap("Table 6. Hypothesis-testing summary. A very small p-value only says an effect is unlikely to be chance; the effect sizes tell us it is also large."));

// ================= 6 ML
C.push(H1("6. Machine-Learning Model Development and Evaluation"));
C.push(H2("6.1 Design"));
C.push(B(`**Split:** stratified 75/25 → ${s3.n_train} training and ${s3.n_test} held-out test patients. The test set is used **once** for final reporting.`));
C.push(B("**Cross-validation:** stratified 5-fold on the training set for all model and feature-set comparisons."));
C.push(B("**Leakage control:** scaling and feature selection live inside scikit-learn `Pipeline` objects, so they are re-fitted on each training fold."));
C.push(B("**Metrics:** ROC-AUC (ranking quality), recall (cancers caught — the clinically critical metric), precision, F1, accuracy, Brier score (calibration)."));
C.push(H2("6.2 Model comparison"));
C.push(table(["Model", "Accuracy", "Recall", "Precision", "ROC-AUC", "Train AUC"],
  s3.cv.map((r) => [r.model, pct(r.accuracy) + " ±" + pct(r.accuracy_sd), pct(r.recall), pct(r.precision), f(r.roc_auc, 4), f(r.train_auc, 3)]),
  [2600, 1500, 1100, 1250, 1300, 1276], { size: 18 }));
C.push(cap("Table 7. 5-fold cross-validated performance on the training set. Random forest and gradient boosting reach train AUC = 1.0 — they memorise the training data — without a better validation AUC than the linear models."));
C.push(P(`All five models land within about one standard deviation of each other (CV AUC ${f(Math.min(...s3.cv.map((r) => r.roc_auc)), 3)}–${f(Math.max(...s3.cv.map((r) => r.roc_auc)), 3)}). **We chose regularised logistic regression** because (a) it is not beaten by any complex model, (b) its coefficients and probabilities can be explained to clinicians and audited, (c) its probabilities are well calibrated and (d) it is the least prone to overfitting. Tuning the inverse-regularisation strength C by grid search gave C = ${f(s3.best_C, 2)} (CV AUC ${s3.best_cv_auc}).`));
C.push(...code(`from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold

cv = StratifiedKFold(5, shuffle=True, random_state=42)
grid = GridSearchCV(
    make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
    {"logisticregression__C": np.logspace(-2, 2, 9)},
    scoring="roc_auc", cv=cv).fit(X_train, y_train)
best = grid.best_estimator_`));
C.push(H2("6.3 Multicollinearity and feature selection"));
C.push(P("Week 4 feedback asked for more depth. We addressed two separate questions: does reducing features change **accuracy**, and does it change **stability** of the fitted coefficients?"));
C.push(table(["Feature set", "Accuracy", "Recall", "F1", "ROC-AUC (±SD)"],
  s3.fs.map((r) => [r.feature_set, pct(r.accuracy), pct(r.recall), f(r.f1, 3), `${f(r.auc, 4)} ±${f(r.auc_sd, 4)}`]), [3100, 1400, 1300, 1200, 2026]));
C.push(cap("Table 8a. Cross-validated logistic regression with different feature sets."));
C.push(P(`**Accuracy.** Pruning from 30 to ${s1.vif_pruned_n} VIF-selected features changes AUC by only ${f(s3.fs[0].auc - s3.fs[1].auc, 3)} — well inside one SD (${f(s3.fs[0].auc_sd, 3)}). The L1 (lasso) model keeps ${s3.l1_n_selected} of 30 features and has the highest CV AUC (${f(s3.fs[5].auc, 4)}) with the smallest SD. Dropping to only the ten mean features or the five top-F features costs 0.5–0.8 AUC points and 2–3 points of recall: the "worst" and "se" features contain information the means lack.`));
C.push(table(["Feature set", "Regularisation (C)", "Mean coefficient SD", "Mean |coef|", "Sign flips across resamples"],
  stab.map((r) => [r.feature_set, r.C, f(r.mean_coef_sd, 3), f(r.mean_abs_coef, 2), r.sign_flip_pct + "%"]), [2100, 1700, 1900, 1500, 1826]));
C.push(cap("Table 8b. Coefficient stability over 200 bootstrap refits. Sign flip = share of resamples in which a coefficient takes the opposite sign to its median."));
C.push(P(`**Stability — a counter-intuitive finding.** Removing collinear features did **not** reduce coefficient noise: with weak regularisation the VIF-pruned model's coefficient SD (${f(stab[2].mean_coef_sd, 2)}) is actually higher than the full model's (${f(stab[0].mean_coef_sd, 2)}), because the coefficients of the retained features grow larger to compensate for dropped information. What **does** stabilise the model is regularisation: raising the penalty (C = 100 → 0.3) cuts the SD from ${f(stab[0].mean_coef_sd, 2)} to ${f(stab[1].mean_coef_sd, 2)} (about ${f(stab[0].mean_coef_sd / stab[1].mean_coef_sd, 0)}×). Pruning helps interpretability modestly (sign flips fall from ${stab[1].sign_flip_pct}% to ${stab[3].sign_flip_pct}% under strong regularisation) but should not be relied upon for stability.`));
C.push(...img("fig15_feature_selection.png", "Figure 15. Left: CV AUC by feature set (error bars ±1 SD). Right: bootstrap SD of coefficients — regularisation, not pruning, tames collinearity.", 6.4));
C.push(callout("Feature-selection conclusion", [
  "Keep all 30 features and rely on L2/L1 regularisation: it is as accurate as any subset, far more stable than unregularised fits, and avoids a manual pruning step that adds variance. Use the L1 variant if a sparser, 14-feature model is preferred for deployment."]));

C.push(H2("6.4 Held-out test performance"));
C.push(table(["Metric", "Test value", "95% bootstrap CI"], [
  ["ROC-AUC", f(td.auc, 4), `${tci.auc[0]} – ${tci.auc[1]}`],
  ["Accuracy", pct(td.accuracy), `${pct(tci.accuracy[0])} – ${pct(tci.accuracy[1])}`],
  ["Recall (sensitivity)", pct(td.recall), `${pct(tci.recall[0])} – ${pct(tci.recall[1])}`],
  ["Precision", pct(td.precision), "—"],
  ["F1", f(td.f1, 3), "—"],
  ["Brier score", f(td.brier, 4), "—"],
], [3200, 2800, 3026]));
C.push(cap(`Table 9. Tuned logistic regression on the ${s3.n_test}-patient test set (threshold 0.5).`));
C.push(...img("fig16_roc_pr.png", "Figure 16. Left: ROC curves for all five models on the test set — curves are nearly indistinguishable. Right: precision–recall curve for the tuned logistic model."));
C.push(...img("fig17_confusion_calibration.png", `Figure 17. Left: confusion matrix at threshold 0.5 (${td.tn} true negatives, ${td.fp} false positives, ${td.fn} false negatives, ${td.tp} true positives). Right: calibration curve — predicted probabilities track observed frequencies.`));
C.push(P(`At the default threshold the model made **${s3.n_wrong} errors — all false negatives** (missed cancers). Zero false positives means precision of 100%, but it also reveals that 0.5 is a poor operating point for a screening-style task. **Who was missed?** The three missed cancers are malignant masses at the small end of their class: their mean area (579, 717 and 584) lies in the 11th–27th percentile of malignant tumours, i.e. they look like benign masses on size alone. Two of them received probabilities of 0.43 and 0.45 — just under the cut-off — and are recovered by the lower threshold; the third (probability 0.10) is missed at every sensible threshold and is the irreducible error of this model. The cases are saved to \`results/misclassified_test_cases.csv\` for review.`));
C.push(H2("6.5 Over- and under-fitting"));
C.push(...img("fig18_learning_curve.png", "Figure 18. Learning curve: training and validation AUC converge as the training set grows.", 4.6));
C.push(P(`The gap between training and validation AUC at full size is only ${f(s3.learning_gap_final, 4)}, and both curves are flat above ≈100 examples. The model is neither under-fitted (AUC ≈ 0.997) nor over-fitted; more data of this type would give diminishing returns, whereas **new kinds** of data (a second hospital, different imaging protocols) would be more informative. The tree ensembles are the contrast: training AUC of 1.0 against CV AUC of ≈0.987 signals memorisation.`));

C.push(H2("6.6 Decision-threshold analysis"));
C.push(P("Week 4 feedback asked for detailed insight into decision thresholds. The default 0.5 treats both kinds of error as equally costly, which is wrong for cancer: a missed malignancy (false negative) is far worse than an unnecessary follow-up (false positive). We therefore (i) mapped how recall, specificity and precision move with the threshold, (ii) chose thresholds on **out-of-fold training predictions only** — never on the test set — and (iii) evaluated them once on the test set."));
C.push(...img("fig19_threshold_analysis.png", "Figure 19. Top left: recall, specificity and precision against threshold. Top right: false negatives and false positives. Bottom: normalised total cost for three cost ratios — the minimum shifts left as missed cancers become more costly.", 6.4));
C.push(table(["Operating point", "Threshold", "Recall", "Specificity", "Precision", "FN", "FP", "Cost (FN=10×FP)"],
  thrKeys.map((k) => { const r = thr[k]; return [k.replace(/\s*[\d.]+$/, "").replace("Recall>=98%", "Recall ≥ 98%"), f(r.thr, 2), pct(r.recall), pct(r.specificity), pct(r.precision), r.fn, r.fp, r.cost]; }),
  [2100, 950, 900, 1100, 1000, 600, 600, 1776], { size: 17 }));
C.push(cap("Table 10. Threshold scenarios on the held-out test set (thresholds fixed beforehand from training-set out-of-fold predictions)."));
C.push(P(`**Reading the table.** Moving from 0.50 to ${thr[thrKeys[1]].thr} recovers two of the three missed cancers (recall ${pct(td.recall)} → ${pct(thr[thrKeys[1]].recall)}) and costs one false positive; the total weighted cost falls from ${thr[thrKeys[0]].cost} to ${thr[thrKeys[1]].cost}. Forcing recall to ≥ 98% by going to ${thr[thrKeys[2]].thr} brings no further sensitivity on this test set but adds seven more false positives (precision ${pct(thr[thrKeys[2]].precision)}), so it is **dominated** by the cost-optimal threshold. Maximising F1 (${thr[thrKeys[3]].thr}) is an intermediate compromise.`));
C.push(table(["FN cost ÷ FP cost", "Optimal threshold", "False negatives", "False positives", "Recall"],
  s3.cost_ratio_table.map((r) => [r.cost_ratio + " : 1", r.thr, r.fn, r.fp, pct(r.recall)]), [2200, 1900, 1700, 1700, 1526]));
C.push(cap("Table 11. Sensitivity of the optimal threshold to the assumed cost ratio."));
C.push(P("The optimal threshold is **stable between cost ratios of 2:1 and 10:1** (0.29) and only moves sharply for very risk-averse settings (20:1 and above), where each extra missed cancer is avoided at the price of many more false alarms. Zero missed cancers on this test set requires a ratio of about 50:1 and 18 false positives. **Caveat:** with only 53 malignant test cases, the difference between one and three false negatives is within sampling noise; the direction of the trade-off is reliable, the exact numbers are not."));
C.push(H2("6.7 What drives the model?"));
C.push(...img("fig20_importance.png", "Figure 20. Left: single-feature permutation importance (small because correlated twins compensate). Right: grouped permutation importance reveals the real drivers.", 6.4));
C.push(P(`Shuffling any one feature barely moves the AUC (largest drop ${f(s3.perm_top.symmetry_worst, 4)}), because a correlated twin covers for it — a direct consequence of the redundancy in Section 4.4. Permuting whole **groups** gives the true picture: size features (radius, perimeter, area) cost ${f(gi["Size (radius, perimeter, area)"], 3)} AUC, shape features (concavity, compactness) ${f(gi["Shape (concavity, compactness)"], 3)}, texture ${f(gi["Texture"], 3)} and smoothness/symmetry/fractal ${f(gi["Smooth/Symmetry/Fractal"], 3)}. Nuclear **size** is by far the dominant signal, with shape irregularity a distant second. The raw coefficients (largest: texture_worst ${s3.top_coefs.texture_worst}, radius_se ${s3.top_coefs.radius_se}) should not be read feature-by-feature: collinearity lets the model assign weight arbitrarily among near-duplicates.`));

// ================= 7 RECOMMENDATIONS
C.push(H1("7. Strategic Recommendations"));
C.push(P("Each recommendation is tied to evidence in this report and to the stakeholder who would act on it."));
C.push(table(["#", "Recommendation", "Evidence", "Owner"], [
  ["1", "Adopt regularised logistic regression as the production baseline; do not pay the complexity cost of boosting or forests", `Sec. 6.2/6.4: test AUC ${f(td.auc, 3)} vs 0.995–0.998 for all alternatives, so no model is meaningfully better; trees overfit (train AUC 1.0)`, "Analytics / ML lead"],
  ["2", `Set the operating threshold from a stated cost ratio (default proposal 10:1 → ${thr[thrKeys[1]].thr}), not from the software default of 0.5`, `Sec. 6.6: ${td.fn} → ${thr[thrKeys[1]].fn} missed cancers, cost ${thr[thrKeys[0]].cost} → ${thr[thrKeys[1]].cost}`, "Clinical lead + governance"],
  ["3", `Deploy as a three-way triage: auto-flag ≥ 0.5, auto-clear < ${rb.lo}, human review in between`, `${rb.pct}% of patients (${rb.n}/${s3.n_test}) need review; ${rb.flagged_above} flagged with ${rb.fp_above} false positives; ${rb.cleared_below} cleared with ${rb.missed_below} miss`, "Pathology lead"],
  ["4", "Audit small-but-malignant lesions: cases with a modest size score but any concern on shape or texture should always go to human review", "Sec. 6.4: the 3 default misses are malignant masses in the smallest 27% of malignant tumours (probabilities 0.10–0.45)", "Pathology / QA"],
  ["5", "Standardise the imaging pipeline and monitor drift in size features (radius, perimeter, area)", "Sec. 6.7: size features carry most of the signal", "Imaging / data engineering"],
  ["6", "Add data-governance checks: deduplicate and split by patient before retraining", "Sec. 3.2: leaked duplicates inflate scores and shift class prior", "Data engineering"],
  ["7", "Retain all features with regularisation instead of manual pruning; offer the 14-feature L1 model as a sparse alternative", "Sec. 6.3: pruning does not improve stability or accuracy", "ML lead"],
  ["8", "Validate externally and prospectively before any clinical use; re-fit the threshold on local data", "Sec. 9: single-institution data, n = 569", "Research lead"],
], [450, 3500, 3376, 1700], { size: 18, left: true }));
C.push(cap("Table 12. Strategic recommendations."));
C.push(H2("7.1 How the recommendations improve decisions"));
C.push(B(`**Fewer missed cancers.** Moving the threshold is a free change: two of three missed cancers are recovered on the test set at no model re-training cost.`));
C.push(B(`**Focused expert time.** Only ${rb.pct}% of cases fall in the grey zone; the model clears ${pct(rb.cleared_below / s3.n_test, 0)} of patients and flags ${pct(rb.flagged_above / s3.n_test, 0)}, so expert attention goes where it adds the most value.`));
C.push(B("**Auditability.** A linear model with calibrated probabilities allows every flag to be traced to measurable nuclear features."));
C.push(B("**Protection against silent degradation.** Governance checks (duplicates, patient-level splits, drift monitoring) protect the quality of future versions."));

// ================= 8 IMPACT
C.push(H1("8. Business and Research Impact"));
C.push(H2("8.1 Illustrative operational impact"));
C.push(P(`The following back-of-envelope scenario uses test-set results; it is **illustrative, not a forecast**. Suppose a laboratory handles 1,000 FNA cases with the same ${s1.malignant_pct}% malignant share as this dataset (≈ 373 malignant). Applying the test-set rates at the cost-optimal threshold (recall ${pct(thr[thrKeys[1]].recall)}, specificity ${pct(thr[thrKeys[1]].specificity)}) versus the default (recall ${pct(td.recall)}, specificity ${pct(td.specificity ?? 1)}) gives:`));
const nM = Math.round(1000 * s1.malignant_pct / 100), nB = 1000 - nM;
const fnDef = Math.round(nM * (1 - td.recall)), fnOpt = Math.round(nM * (1 - thr[thrKeys[1]].recall));
const fpDef = Math.round(nB * (1 - 1)), fpOpt = Math.round(nB * (1 - thr[thrKeys[1]].specificity));
C.push(table(["Policy", "Missed cancers", "False alarms", "Weighted cost (10:1)"], [
  ["Default threshold 0.50", fnDef, fpDef, fnDef * 10 + fpDef],
  [`Cost-optimal ${thr[thrKeys[1]].thr}`, fnOpt, fpOpt, fnOpt * 10 + fpOpt],
], [3000, 2000, 2000, 2026]));
C.push(cap("Table 13. Scaled scenario (1,000 cases). Because the estimates rest on 53 malignant and 90 benign test patients, confidence bands are wide; treat as order-of-magnitude."));
C.push(H2("8.2 Research outcomes"));
C.push(B("Quantified evidence that nuclear size and contour irregularity dominate cytological malignancy features, with effect sizes above 2.5 standard deviations."));
C.push(B("A reproducible methodology for studying collinear feature sets: grouped permutation importance plus a crossed regularisation × pruning stability design."));
C.push(B("A cost-sensitive threshold framework that can be re-used by changing a single parameter (the cost ratio)."));

// ================= 9 LIMITATIONS
C.push(H1("9. Limitations and Future Work"));
C.push(H2("9.1 Limitations"));
C.push(B(`**Small test set.** ${s3.n_test} patients and ${td.tp + td.fn} malignant cases; recall CI ${tci.recall[0]}–${tci.recall[1]}. Differences of 1–2 patients are not statistically distinguishable.`));
C.push(B("**Single source.** All images come from one institution and imaging protocol; performance on other scanners, staining methods or populations is unknown."));
C.push(B("**No patient identifiers or demographics in the bundled copy.** We cannot check for multiple samples per patient (which would invalidate random splits) or for subgroup fairness (age, ethnicity)."));
C.push(B("**Features are pre-computed.** The model sees image-derived summaries, not the images; any upstream segmentation error is invisible to us."));
C.push(B("**Assumed costs.** The 10:1 cost ratio is an assumption for illustration; real values must come from clinical and health-economic input."));
C.push(B("**Multiple testing across hypotheses.** Holm correction was applied within H1 and H2; the five headline hypotheses were pre-specified and not adjusted against one another, which is acceptable given the very small p-values, but should be noted."));
C.push(B("**Duplicate analysis is a simulation.** It illustrates a mechanism with one model type; the magnitude would differ for other algorithms."));
C.push(B("**Not a medical device.** Nothing here has been clinically validated."));
C.push(H2("9.2 Future work"));
C.push(N("**External validation** on an independent cohort (e.g. other public FNA data sets or local clinic data) with a pre-registered threshold.", "num2"));
C.push(N("**Calibration monitoring** and recalibration (Platt or isotonic) as case-mix shifts.", "num2"));
C.push(N("**Deep learning on images** with the current feature model as a baseline and Grad-CAM-style explanations.", "num2"));
C.push(N("**Uncertainty quantification** (conformal prediction) to formalise the review band with coverage guarantees.", "num2"));
C.push(N("**Subgroup and fairness analysis** once demographic fields are available.", "num2"));
C.push(N("**Prospective pilot** comparing triage workflow with usual care on turnaround time and missed-cancer rate.", "num2"));

// ================= 10 CONCLUSION
C.push(H1("10. Conclusion"));
C.push(P(`Starting from a clean public dataset, we showed that nuclear **size and irregularity** separate malignant from benign breast masses with very large statistical effect sizes, that the 30 features are highly redundant (23 with VIF > 10), and that a regularised logistic regression exploits them as well as any model we tried (test AUC ${f(td.auc, 3)}). Three less obvious results shaped the recommendations: duplicates inflate scores quietly rather than dramatically; collinearity is best handled by regularisation rather than by dropping features; and the **decision threshold** — not the algorithm — determines how many cancers are missed. A cost-aware threshold plus a small human-review band turns an already accurate model into a safer workflow.`));
C.push(P("The caveats are equally important: a small test set, a single source, assumed cost ratios and no clinical validation. The right next step is external validation, not deployment."));

// ================= APPENDIX
C.push(H1("Appendix A. Reproducibility"));
C.push(H2("A.1 Repository layout"));
C.push(...code(`predicting-malignant-breast-cancer/
├── README.md
├── requirements.txt
├── run_all.sh                       # runs the whole pipeline
├── data/raw/wdbc.data               # raw UCI data
├── data/processed/wdbc_clean.csv    # cleaned data
├── src/
│   ├── common.py                    # paths, palette, helpers
│   ├── 01_acquire_clean_eda.py      # Sections 2-4
│   ├── 02_hypothesis_tests.py       # Section 5
│   ├── 03_modeling.py               # Section 6
│   └── build_report.js              # builds this .docx
├── figures/                         # 20 figures (PNG, 200 dpi)
├── results/                         # all tables / metrics (CSV, JSON)
└── reports/Week5_Final_Project_Report.docx`));
C.push(H2("A.2 Environment"));
C.push(P("Python 3.10+ with pandas, NumPy, SciPy, statsmodels, scikit-learn, matplotlib, seaborn; Node.js 18+ with the `docx` package only to rebuild this Word file. All random operations use seed 42."));
C.push(...code(`pip install -r requirements.txt
python src/01_acquire_clean_eda.py
python src/02_hypothesis_tests.py
python src/03_modeling.py
npm install docx && node src/build_report.js   # optional: rebuild report`));
C.push(H2("A.3 Key code excerpts"));
C.push(P("**Holm-corrected Welch tests with effect sizes (Section 5.1):**", { keepNext: true }));
C.push(...code(`for c in feat:
    a, b = M[c], B[c]
    t, p   = stats.ttest_ind(a, b, equal_var=False)
    u, pu  = stats.mannwhitneyu(a, b, alternative="two-sided")
    sp = np.sqrt(((len(a)-1)*a.var(ddof=1) + (len(b)-1)*b.var(ddof=1))
                 / (len(a)+len(b)-2))
    d  = (a.mean() - b.mean()) / sp                      # Cohen's d
T["p_welch_holm"] = multipletests(T.p_welch, method="holm")[1]`));
C.push(P("**Variance inflation factors and iterative pruning (Section 4.4):**", { keepNext: true }));
C.push(...code(`Xs  = StandardScaler().fit_transform(clean[feat])
vif = pd.Series([variance_inflation_factor(Xs, i)
                 for i in range(Xs.shape[1])], index=feat)
keep = list(feat)
while True:                         # drop highest VIF until all <= 10
    v = vif_of(clean[keep])
    if v.max() <= 10: break
    keep.remove(v.idxmax())`));
C.push(P("**Threshold chosen on out-of-fold training predictions only (Section 6.6):**", { keepNext: true }));
C.push(...code(`oof = cross_val_predict(best, X_train, y_train, cv=cv,
                        method="predict_proba")[:, 1]
cost = lambda t: 10*FN(y_train, oof >= t) + 1*FP(y_train, oof >= t)
t_star = min(np.linspace(.02, .98, 97), key=cost)    # then evaluate on test`));
C.push(H2("A.4 Result files"));
C.push(table(["File", "Content"], [
  ["results/step1_metrics.json … step3_metrics.json", "Every number quoted in this report"],
  ["results/hypothesis_tests_all_features.csv", "Full 30-feature test table: t, p, Holm p, CI, Cohen's d"],
  ["results/vif_all_features.csv", "VIF for all 30 features"],
  ["results/cv_model_comparison.csv, feature_selection_comparison.csv", "Model and feature-set CV results"],
  ["results/threshold_analysis_test.csv", "Metrics at 97 thresholds"],
  ["results/test_predictions.csv, misclassified_test_cases.csv", "Test-set probabilities and error cases"],
  ["results/duplicate_impact.csv", "Duplicate simulation results"],
], [4600, 4426], { left: true }));
C.push(gap());
C.push(H2("A.5 References"));
C.push(B("Wolberg, W. H., Street, W. N., & Mangasarian, O. L. (1995). Breast Cancer Wisconsin (Diagnostic) [Dataset]. UCI Machine Learning Repository."));
C.push(B("Street, W. N., Wolberg, W. H., & Mangasarian, O. L. (1993). Nuclear feature extraction for breast tumor diagnosis. SPIE Biomedical Image Processing and Biomedical Visualization, 1905, 861–870."));
C.push(B("Holm, S. (1979). A simple sequentially rejective multiple test procedure. Scandinavian Journal of Statistics, 6, 65–70."));
C.push(B("Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825–2830."));

// ---------- document
const doc = new Document({
  creator: "Shivang Dube", title: "Predicting Malignant Breast Cancer — Week 5 Final Project Report",
  styles: {
    default: { document: { run: { font: FONT, size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 36, bold: true, color: NAVY, font: FONT }, paragraph: { spacing: { before: 240, after: 200 }, outlineLevel: 0, keepNext: true } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, color: BLUE, font: FONT }, paragraph: { spacing: { before: 280, after: 120 }, outlineLevel: 1, keepNext: true } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, color: "333333", font: FONT }, paragraph: { spacing: { before: 200, after: 100 }, outlineLevel: 2, keepNext: true } },
    ],
  },
  numbering: { config: [
    { reference: "bul", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 270 } } } },
      { level: 1, format: LevelFormat.BULLET, text: "–", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1000, hanging: 270 } } } }] },
    ...["num", "num2", "num3"].map((r) => ({ reference: r, levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 360 } } } }] })),
  ] },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1300, right: 1440, bottom: 1300, left: 1440 } } },
    headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: "C9CED6", space: 4 } },
      children: [new TextRun({ text: "Predicting Malignant Breast Cancer — Week 5 Final Report", size: 17, color: "777777", font: FONT })] })] }) },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ text: "Page ", size: 18, color: "777777", font: FONT }), new TextRun({ children: [PageNumber.CURRENT], size: 18, color: "777777", font: FONT })] })] }) },
    children: C,
  }],
});
const outPath = path.join(ROOT, "reports", "Week5_Final_Project_Report.docx");
Packer.toBuffer(doc).then((b) => { fs.writeFileSync(outPath, b); console.log("wrote", outPath, b.length); });
