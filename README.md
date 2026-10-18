# Predicting Malignant Breast Cancer — End-to-End Data Science Project

Reproducible analysis of the **Breast Cancer Wisconsin (Diagnostic)** dataset (UCI): data audit and cleaning,
exploratory analysis, statistical hypothesis testing, machine-learning models, decision-threshold analysis and
strategic recommendations. The full write-up is in [`reports/Week5_Final_Project_Report.docx`](reports/Week5_Final_Project_Report.docx)
(a PDF copy is alongside it).

## Headline results

| Question | Answer |
|---|---|
| Data quality | 569 patients, 37.26% malignant; 0 missing values, 0 duplicates |
| Do classes differ? | 25/30 features significant after Holm correction; Cohen's d up to 2.69 |
| Multicollinearity | 23/30 features with VIF > 10; 10 PCs explain 95% of variance |
| Best model | Regularised logistic regression: test AUC 0.998, accuracy 97.9% (n = 143) |
| Threshold | Cost-optimal cut-off (0.29) cuts weighted error cost from 30 to 11 vs the 0.5 default |

> **Caveat:** small test set (143 patients), single-institution data, assumed 10:1 cost ratio.
> This is a decision-support prototype, **not** a validated medical device.

## Repository layout

```
├── README.md
├── requirements.txt
├── run_all.sh                     # runs the whole pipeline
├── data/
│   ├── raw/wdbc.data              # raw data (UCI format)
│   └── processed/wdbc_clean.csv   # cleaned data + binary target
├── src/
│   ├── common.py                  # paths, palette, helpers
│   ├── 01_acquire_clean_eda.py    # acquisition, audit, cleaning, duplicate-impact study, EDA, VIF/PCA
│   ├── 02_hypothesis_tests.py     # H1-H5 (Welch/Mann-Whitney, Levene, Shapiro, chi-square, ANOVA) + charts
│   ├── 03_modeling.py             # 5 models, feature selection, coefficient stability, thresholds, importance
│   └── build_report.js            # rebuilds the .docx from results/ and figures/
├── figures/                       # 20 figures (PNG)
├── results/                       # every table / metric (CSV, JSON)
└── reports/                       # final report (.docx and .pdf)
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
bash run_all.sh                                        # or run the three scripts in order
```

Python 3.10+. All randomness uses seed 42, so results are reproducible. The acquisition script first tries to
download the dataset from the UCI repository and falls back to the identical copy bundled with scikit-learn when offline.

To rebuild the Word report (optional): `npm install docx && node src/build_report.js`.

## Method notes

* **No leakage:** scaling and feature selection live inside scikit-learn `Pipeline`s; thresholds are chosen on out-of-fold
  *training* predictions and evaluated once on the held-out test set.
* **Multiple testing:** Holm correction across the 30 feature tests; Welch t-test confirmed by Mann-Whitney.
* **Duplicates:** none in the data; a controlled simulation quantifies what leaked duplicates would do (results/duplicate_impact.csv).
* **Multicollinearity:** VIF, condition number, PCA, and a crossed *feature-set x regularisation* coefficient-stability study.
  Finding: regularisation, not feature pruning, stabilises the model.

## Data source

Wolberg, W. H., Street, W. N., & Mangasarian, O. L. (1995). *Breast Cancer Wisconsin (Diagnostic)*. UCI Machine Learning Repository.

## License

MIT (code). The dataset retains its original UCI licence/terms.
