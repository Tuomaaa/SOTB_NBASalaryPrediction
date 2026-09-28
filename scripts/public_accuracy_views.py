"""Public-facing accuracy views of the champion's out-of-sample predictions.

Reads outputs/models/oof_reference.csv, written by src/model/evaluate_suite.py,
so every number is out-of-fold (A1) or rolling-forward (B1). Never feed this
script the export_web refit: those predictions are in-sample.

Reporting only. Nothing here enters an accept/reject decision.

Outputs:
  outputs/diagnostics/public_accuracy_views.csv   metric table per layer
  outputs/diagnostics/public_accuracy_misses.csv  top misses with R2 cost
  outputs/diagnostics/public_accuracy_views.png   hit-rate curves (B1)

Run after the suite:

    python src/model/evaluate_suite.py
    python scripts/public_accuracy_views.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from config import CAP_BY_SEASON, OUTPUTS_DIR

OOF_PATH = OUTPUTS_DIR / "models" / "oof_reference.csv"
DIAG_DIR = OUTPUTS_DIR / "diagnostics"

CAP_PTS = (0.5, 1, 2, 3, 5, 10)
DOLLARS_M = (0.5, 1, 2, 3, 5, 8)
REL_PCT = (10, 20, 30, 50)
QUANTILES = (0.25, 0.5, 0.75, 0.9, 0.95)
TRIM_K = (1, 3, 5, 10)
BIG_CONTRACT_M = 10.0


def load_layers() -> dict[str, pd.DataFrame]:
    """Return the A1, B1 and B1-2026 row sets with a common `pred` column."""
    ref = pd.read_csv(OOF_PATH)
    ref["cap"] = ref["season"].map(CAP_BY_SEASON)
    fwd = ref["fwd_champion"].notna()
    layers = {
        "A1 pooled CV": ref.assign(pred=ref["oof_champion"]),
        "B1 forward 2024-26": ref[fwd].assign(pred=ref.loc[fwd, "fwd_champion"]),
    }
    b1 = layers["B1 forward 2024-26"]
    layers["B1 2026"] = b1[b1["season"] == 2026]
    return layers


def add_errors(d: pd.DataFrame) -> pd.DataFrame:
    """Add dollar, cap-point and relative absolute errors."""
    d = d.copy()
    d["actual_m"] = d["cap_pct"] * d["cap"] / 1e6
    d["pred_m"] = d["pred"] * d["cap"] / 1e6
    d["err_m"] = d["pred_m"] - d["actual_m"]
    d["abs_m"] = d["err_m"].abs()
    d["abs_pts"] = (d["pred"] - d["cap_pct"]).abs() * 100
    d["abs_rel"] = d["abs_m"] / d["actual_m"] * 100
    ss_tot = ((d["cap_pct"] - d["cap_pct"].mean()) ** 2).sum()
    d["r2_cost"] = (d["pred"] - d["cap_pct"]) ** 2 / ss_tot
    return d


def metrics(d: pd.DataFrame) -> dict:
    """Metric table for one row set."""
    big = d["actual_m"] >= BIG_CONTRACT_M
    non_min = d["signing_cat"] != "Minimum"
    out = {"n": len(d), "R2": r2_score(d["cap_pct"], d["pred"]),
           "MAE $M": d["abs_m"].mean(), "exact hit % (error 0)": (d["abs_pts"] < 1e-6).mean() * 100}
    for t in DOLLARS_M:
        out[f"within ${t}M %"] = (d["abs_m"] <= t).mean() * 100
    for t in CAP_PTS:
        out[f"within {t} cap pts %"] = (d["abs_pts"] <= t).mean() * 100
    for t in (1, 3, 5):
        out[f"non-minimum within {t} cap pts %"] = (d.loc[non_min, "abs_pts"] <= t).mean() * 100
    for t in REL_PCT:
        out[f"within {t}% relative %"] = (d["abs_rel"] <= t).mean() * 100
        out[f">=${BIG_CONTRACT_M:.0f}M within {t}% relative %"] = (d.loc[big, "abs_rel"] <= t).mean() * 100
    for q in QUANTILES:
        out[f"q{int(q * 100)} $M"] = d["abs_m"].quantile(q)
        out[f"q{int(q * 100)} cap pts"] = d["abs_pts"].quantile(q)
    order = d["abs_pts"].sort_values(ascending=False).index
    for k in TRIM_K:
        keep = d.drop(order[:k])
        out[f"R2 without top {k}"] = r2_score(keep["cap_pct"], keep["pred"])
    return out


def plot_curves(d: pd.DataFrame, path: Path) -> None:
    """Hit-rate curves in cap points and relative error, one panel each."""
    surf, ink, ink2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e7e6e2"
    s1, s2 = "#2a78d6", "#eb6834"
    big = d["actual_m"] >= BIG_CONTRACT_M
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=surf)
    panels = [
        (axes[0], np.linspace(0, 8, 161), [(d["abs_pts"], "all signings", s1)],
         "Error in cap percentage points", "error threshold (% of cap)"),
        (axes[1], np.linspace(0, 100, 201),
         [(d["abs_rel"], "all signings", s1),
          (d.loc[big, "abs_rel"], f"salary >= ${BIG_CONTRACT_M:.0f}M", s2)],
         "Relative error", "error threshold (% of actual salary)"),
    ]
    for ax, xs, series, title, xlabel in panels:
        ax.set_facecolor(surf)
        for vals, label, color in series:
            ys = [(vals <= x).mean() * 100 for x in xs]
            ax.plot(xs, ys, color=color, lw=2, label=label)
            ax.annotate(label, (xs[-1], ys[-1]), xytext=(-4, 6),
                        textcoords="offset points", ha="right", color=ink2, fontsize=9)
        ax.set_title(title, color=ink, fontsize=11, loc="left")
        ax.set_xlabel(xlabel, color=ink2, fontsize=9)
        ax.set_ylim(0, 100)
        ax.set_xlim(xs[0], xs[-1])
        ax.grid(color=grid, lw=0.8)
        ax.tick_params(colors=ink2, labelsize=9)
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0].set_ylabel("% of signings within threshold", color=ink2, fontsize=9)
    axes[1].legend(frameon=False, fontsize=9, labelcolor=ink2, loc="lower right")
    fig.suptitle("Forward accuracy (B1, 2024-26)", color=ink, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=surf)
    plt.close(fig)


def main():
    """Write the metric table, the misses table and the curve figure."""
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    frames = {name: add_errors(d) for name, d in load_layers().items()}
    table = pd.DataFrame({name: metrics(d) for name, d in frames.items()})
    pd.set_option("display.width", 200)
    print(table.round(3).to_string())
    table.round(4).to_csv(DIAG_DIR / "public_accuracy_views.csv")

    cols = ["player_name_norm", "season", "signing_cat", "actual_m", "pred_m",
            "err_m", "abs_pts", "r2_cost"]
    misses = []
    for name, d in frames.items():
        top = d.sort_values("r2_cost", ascending=False).head(10)[cols].copy()
        top.insert(0, "layer", name)
        misses.append(top)
        print(f"\n{name}: top misses (r2_cost = R2 lost to this row)")
        print(top[cols].round(3).to_string(index=False))
    pd.concat(misses).round(4).to_csv(DIAG_DIR / "public_accuracy_misses.csv", index=False)

    plot_curves(frames["B1 forward 2024-26"], DIAG_DIR / "public_accuracy_views.png")
    print(f"\nSaved {DIAG_DIR / 'public_accuracy_views.csv'}, "
          f"public_accuracy_misses.csv, public_accuracy_views.png")


if __name__ == "__main__":
    main()
