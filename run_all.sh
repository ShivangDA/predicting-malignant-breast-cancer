#!/usr/bin/env bash
# Reproduce every table, figure and metric. Usage: bash run_all.sh
set -euo pipefail
cd "$(dirname "$0")"
PY=$(command -v python3 || command -v python)
"$PY" src/01_acquire_clean_eda.py     # acquisition, cleaning, duplicate study, EDA, VIF/PCA
"$PY" src/02_hypothesis_tests.py      # H1-H5 with charts
"$PY" src/03_modeling.py              # models, feature selection, thresholds
# Optional: rebuild the Word report (needs Node.js):  npm install docx && node src/build_report.js
echo "Done. See figures/, results/ and reports/."
