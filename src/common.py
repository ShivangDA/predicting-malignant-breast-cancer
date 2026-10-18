"""Shared helpers: paths, plotting style, data loading."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
FIG = ROOT / "figures"
RES = ROOT / "results"
for p in (RAW, PROC, FIG, RES):
    p.mkdir(parents=True, exist_ok=True)

SEED = 42
# Colour-blind-safe palette: benign = blue, malignant = vermilion
C_B, C_M = "#0072B2", "#D55E00"
PALETTE = {"Benign": C_B, "Malignant": C_M}

sns.set_theme(style="whitegrid", context="notebook", font_scale=1.0)
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "savefig.bbox": "tight",
    "axes.titleweight": "bold", "axes.titlesize": 12, "axes.spines.top": False,
    "axes.spines.right": False,
})


def save(fig, name):
    path = FIG / name
    fig.savefig(path)
    plt.close(fig)
    return path
