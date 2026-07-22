"""Model evaluation: residual analysis and prediction vs actual plots."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupKFold
from config import OUTPUTS_DIR


def plot_predictions_vs_actual(y_true, y_pred, title="Predictions vs Actual"):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_true, y_pred, alpha=0.4, s=20)
    lims = [0, max(max(y_true), max(y_pred)) * 1.05]
    ax.plot(lims, lims, "r--", linewidth=1, label="Perfect")
    ax.set_xlabel("Actual cap_pct")
    ax.set_ylabel("Predicted cap_pct")
    ax.set_title(title)
    ax.legend()
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    plt.tight_layout()
    return fig


def plot_residuals(y_true, y_pred, title="Residual Distribution"):
    residuals = y_pred - y_true
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(y_true, residuals, alpha=0.4, s=20)
    axes[0].axhline(0, color="r", linestyle="--")
    axes[0].set_xlabel("Actual cap_pct")
    axes[0].set_ylabel("Residual (pred - actual)")
    axes[0].set_title("Residuals vs Actual")

    axes[1].hist(residuals, bins=40, edgecolor="black", alpha=0.7)
    axes[1].axvline(0, color="r", linestyle="--")
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Residual Distribution")

    fig.suptitle(title)
    plt.tight_layout()
    return fig


def generate_cv_predictions(pipeline, X, y, groups):
    """Generate out-of-fold predictions using GroupKFold."""
    cv = GroupKFold(n_splits=5)
    y_pred = np.zeros_like(y)

    for train_idx, test_idx in cv.split(X, y, groups):
        pipeline.fit(X.iloc[train_idx], y[train_idx])
        y_pred[test_idx] = pipeline.predict(X.iloc[test_idx])

    return y_pred


def find_outliers(df, y_true, y_pred, n=20):
    """Find biggest over/underpays by residual magnitude."""
    residuals = y_pred - y_true
    df_out = df.copy()
    df_out["predicted_cap_pct"] = y_pred
    df_out["residual"] = residuals
    df_out["abs_residual"] = np.abs(residuals)

    print(f"\nTop {n} OVERPAID (model says worth less):")
    overpaid = df_out.nlargest(n, "residual")
    cols = ["player_name", "season", "cap_pct", "predicted_cap_pct", "residual", "base_rating"]
    print(overpaid[[c for c in cols if c in overpaid.columns]].to_string(index=False))

    print(f"\nTop {n} UNDERPAID (model says worth more):")
    underpaid = df_out.nsmallest(n, "residual")
    print(underpaid[[c for c in cols if c in underpaid.columns]].to_string(index=False))

    return df_out


if __name__ == "__main__":
    from src.model.train import load_training_data, train_ridge, _prepare_Xy

    df = load_training_data()
    ridge_results, pipeline = train_ridge(df)

    X, y, groups, features = _prepare_Xy(df)
    y_pred = generate_cv_predictions(pipeline, X, y, groups)

    out_dir = OUTPUTS_DIR / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig1 = plot_predictions_vs_actual(y, y_pred)
    fig1.savefig(out_dir / "pred_vs_actual.png", dpi=150)

    fig2 = plot_residuals(y, y_pred)
    fig2.savefig(out_dir / "residuals.png", dpi=150)

    find_outliers(df, y, y_pred)
    print(f"\nPlots saved to {out_dir}")
