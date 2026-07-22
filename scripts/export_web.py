"""Export every web-facing artifact from a single Grabit fit.

The portfolio site shows a valuation table, a per-player SHAP breakdown, and
diagnostic charts. All three must agree to the dollar: a waterfall that sums to
a different number than the row above it is worse than no waterfall at all.
So this script fits the model once and derives everything from that one fit —
it never reads the prediction CSVs in outputs/, which were produced by
different model versions on different days.

Writes to the portfolio site's public/data/nba/ directory:
    valuations.json  — one row per player-season (table data, no SHAP)
    shap.json        — per-row feature attributions in dollars (lazy-loaded)
    meta.json        — CV metrics, feature list, caps, generated timestamp
    charts/*.png     — diagnostics restyled to the site palette

Usage:
    python scripts/export_web.py [--out PATH]
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from config import CAP_BY_SEASON, OUTPUTS_DIR, PROCESSED_DIR
from src.model.train import (
    FEATURE_COLS,
    TARGET,
    _compute_max_eligible,
    _filter_rookie_scale,
    _filter_year1,
    _load_rookie_scale_set,
    _prepare_Xy,
    load_training_data,
    train_grabit,
)

DEFAULT_OUT = (
    Path(__file__).resolve().parents[2] / "WebPage" / "public" / "data" / "nba"
)

# Site theme tokens, mirrored from WebPage/app/globals.css :root
PALETTE = {
    "bg": "#FAF9F5",
    "bg_soft": "#F2F0E9",
    "ink": "#191917",
    "ink_soft": "#57564F",
    "green": "#4E7A5A",
    "pink": "#C98CA0",
    "gold": "#B09840",
    "teal": "#4A8E96",
    "line": "#E4E1D7",
}

# Display labels for the 14 model features, used by the SHAP waterfall.
FEATURE_LABELS = {
    "darko_dpm_z": "DARKO DPM",
    "lebron_z": "LEBRON",
    "rapm_z": "RAPM",
    "age": "Age",
    "age_squared": "Age²",
    "mpg": "Minutes / game",
    "availability_3yr": "Availability (3yr)",
    "usage_pct": "Usage rate",
    "height_inches": "Height",
    "cba_era": "CBA era",
    "ast_pct": "Assist %",
    "award_score_cum": "Awards",
    "draft_pick": "Draft pick",
    "prev_cap_pct": "Previous contract",
}


# Spotrac writes signing mechanisms in two styles — title case for some, a
# hyphenated slug for others. Canonicalise so the filter and the chart agree.
SIGNING_LABELS = {
    "Minimum": "Minimum",
    "Bird Rights": "Bird Rights",
    "Early Bird Rights": "Early Bird",
    "Non-Bird Rights": "Non-Bird",
    "Hardship": "Hardship",
    "cap-space": "Cap Space",
    "sign-and-trade": "Sign & Trade",
    "rookie-scale-exception": "Rookie Scale",
    "non-taxpayer-mid-level-exception": "Non-Taxpayer MLE",
    "taxpayer-mid-level-exception": "Taxpayer MLE",
    "room-mid-level-exception": "Room MLE",
    "bi-annual-exception": "Bi-Annual Exception",
    "qualifying-offer": "Qualifying Offer",
    "disabled-player-exception": "Disabled Player",
}


def _row_key(name_norm: str, season: int) -> str:
    return f"{name_norm}|{int(season)}"


def _load_signing_types() -> pd.DataFrame:
    """Signing mechanism per player-season, deduplicated.

    spotrac_signing_types.csv has duplicate (player, season) keys — merging it
    raw inflates the row count. Keep the first record per key.
    """
    path = PROCESSED_DIR / "spotrac_signing_types.csv"
    if not path.exists():
        return pd.DataFrame(columns=["player_name_norm", "season", "signing_type"])
    st = pd.read_csv(path)
    st = st.drop_duplicates(subset=["player_name_norm", "season"], keep="first")
    unknown = set(st["signing_type"].dropna()) - set(SIGNING_LABELS)
    if unknown:
        raise SystemExit(f"unmapped signing mechanism(s): {sorted(unknown)}")
    st["signing_type"] = st["signing_type"].map(SIGNING_LABELS)
    return st[["player_name_norm", "season", "signing_type"]]


def _training_medians(df: pd.DataFrame) -> tuple[list[str], pd.Series]:
    """Feature list and fill values as seen by the fitted model.

    _prepare_Xy drops near-constant columns and fills NaN with the median of
    the *filtered* training set. Predicting the full dataset has to reuse both,
    or the full-set rows get imputed against a different distribution.
    """
    tr = _filter_rookie_scale(_filter_year1(df))
    X_tr, _, _, features = _prepare_Xy(tr)
    return features, X_tr.median()


def build_frame(df: pd.DataFrame, model, features: list[str],
                medians: pd.Series) -> tuple[pd.DataFrame, np.ndarray]:
    """Predict latent + capped value for every player-season, with SHAP.

    The model is fit on year-1 non-rookie-scale rows only, but it is applied to
    all rows here. Escalator years (year_in_contract >= 2) are outside the
    training domain by construction — the site labels those separately as
    Contract Surplus rather than Signing Residual.
    """
    import shap

    full = _compute_max_eligible(df.copy())

    X = full.reindex(columns=features).copy()
    X = X.fillna(medians).fillna(0)

    latent = model.predict(X)
    max_elig = full["max_eligible_pct"].values
    capped = np.minimum(latent, max_elig)

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X)
    expected = float(np.ravel(explainer.expected_value)[0])

    # Sanity: SHAP decomposes the *latent* (pre-cap) prediction.
    recon = shap_vals.sum(axis=1) + expected
    drift = np.abs(recon - latent).max()
    if drift > 1e-4:
        raise SystemExit(
            f"SHAP reconstruction drifts from prediction by {drift:.2e} — "
            "the waterfall would not sum to the displayed value."
        )

    cap = full["season"].map(CAP_BY_SEASON).astype(float)
    if cap.isna().any():
        missing = sorted(full.loc[cap.isna(), "season"].unique())
        raise SystemExit(f"No salary cap on record for season(s): {missing}")

    out = pd.DataFrame({
        "player_name": full["player_name"],
        "player_name_norm": full["player_name_norm"],
        "season": full["season"].astype(int),
        "team": full["team_abbreviation"],
        "position": full["position"],
        "age": full["age"],
        "year_in_contract": full["year_in_contract"],
        "contract_years": full["contract_years"],
        "actual_cap_pct": full[TARGET],
        "latent_cap_pct": latent,
        "pred_cap_pct": capped,
        "max_eligible_pct": max_elig,
        "cap": cap,
    })
    # A first-round pick's slotted years often carry year_in_contract == 1 in
    # the contract-structure data, which would put a rookie-scale salary on the
    # Signing Board as though a team had freshly negotiated it. Flag them so the
    # site can hold that board to the model's actual training domain.
    rookie = _load_rookie_scale_set()
    out["is_rookie_scale"] = [
        (n, s) in rookie for n, s in zip(out["player_name_norm"], out["season"])
    ]

    out["actual_salary"] = out["actual_cap_pct"] * out["cap"]
    out["pred_salary"] = out["pred_cap_pct"] * out["cap"]
    out["latent_salary"] = out["latent_cap_pct"] * out["cap"]
    out["surplus"] = out["pred_salary"] - out["actual_salary"]
    out["is_capped"] = out["latent_cap_pct"] > out["max_eligible_pct"] + 1e-9

    for col in ("darko_dpm_z", "lebron_z", "rapm_z", "mpg", "usage_pct",
                "availability_3yr", "ast_pct", "height_inches", "draft_pick",
                "award_score_cum"):
        out[col] = full[col] if col in full.columns else np.nan

    st = _load_signing_types()
    out = out.merge(st, on=["player_name_norm", "season"], how="left")
    if len(out) != len(full):
        raise SystemExit("signing_type merge changed the row count")

    out["base_salary"] = expected * out["cap"]
    return out, shap_vals


def _round(x, nd=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return None
    return round(float(x), nd)


def write_json(out: pd.DataFrame, shap_vals: np.ndarray, features: list[str],
               results: dict, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    M = 1e6

    rows = []
    for r in out.itertuples(index=False):
        rows.append({
            "k": _row_key(r.player_name_norm, r.season),
            "n": r.player_name,
            "t": None if pd.isna(r.team) else r.team,
            "p": None if pd.isna(r.position) else r.position,
            "s": int(r.season),
            "a": _round(r.age, 0),
            "yc": None if pd.isna(r.year_in_contract) else int(r.year_in_contract),
            "cy": None if pd.isna(r.contract_years) else int(r.contract_years),
            "act": _round(r.actual_salary / M),
            "pred": _round(r.pred_salary / M),
            "lat": _round(r.latent_salary / M),
            "sur": _round(r.surplus / M),
            "cap": bool(r.is_capped),
            "rs": bool(r.is_rookie_scale),
            "st": None if pd.isna(r.signing_type) else r.signing_type,
            "dk": _round(r.darko_dpm_z),
            "lb": _round(r.lebron_z),
            "rp": _round(r.rapm_z),
            "mp": _round(r.mpg, 1),
            "us": _round(r.usage_pct, 1),
            "av": _round(r.availability_3yr, 3),
            "as": _round(r.ast_pct, 1),
            "ht": _round(r.height_inches, 0),
            "dp": None if pd.isna(r.draft_pick) else int(r.draft_pick),
            "aw": _round(r.award_score_cum),
        })

    caps = out["cap"].values
    shap_dollars = shap_vals * caps[:, None] / M
    keys = [_row_key(n, s) for n, s in
            zip(out["player_name_norm"], out["season"])]

    shap_out = {}
    for i, key in enumerate(keys):
        shap_out[key] = {
            f: _round(shap_dollars[i, j])
            for j, f in enumerate(features)
        }

    payload = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "model": results["model"],
        "rows": rows,
    }
    (dest / "valuations.json").write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    (dest / "shap.json").write_text(
        json.dumps({"base": None, "rows": shap_out},
                   ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )

    base_by_season = (
        out.groupby("season")["base_salary"].first() / M
    ).round(2).to_dict()

    meta = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "model": results["model"],
        "sigma": results.get("sigma"),
        "nTrain": results["n_samples"],
        "nFeatures": results["n_features"],
        "nCensored": results.get("n_censored"),
        "cvR2": _round(results["cv_r2_mean"], 4),
        "cvR2Std": _round(results.get("cv_r2_std"), 4),
        "cvMae": _round(results["cv_mae_mean"], 4),
        "cvR2Recent": _round(results.get("cv_r2_recent"), 4),
        "cvMaeRecent": _round(results.get("cv_mae_recent"), 4),
        "recentN": results.get("recent_n"),
        "features": [{"key": f, "label": FEATURE_LABELS.get(f, f)}
                     for f in features],
        "capBySeason": {str(k): v for k, v in CAP_BY_SEASON.items()},
        "baseSalaryBySeason": {str(k): v for k, v in base_by_season.items()},
        "seasons": sorted(int(s) for s in out["season"].unique()),
        "teams": sorted(t for t in out["team"].dropna().unique()),
        "positions": sorted(p for p in out["position"].dropna().unique()),
        "signingTypes": sorted(s for s in out["signing_type"].dropna().unique()),
        "nRows": len(out),
        "nYear1": int((out["year_in_contract"] == 1).sum()),
        # Rows the model is actually fit on: year-1 and not rookie-scale.
        "nSigning": int(((out["year_in_contract"] == 1)
                         & ~out["is_rookie_scale"]).sum()),
    }
    (dest / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    for name in ("valuations.json", "shap.json", "meta.json"):
        kb = (dest / name).stat().st_size / 1024
        print(f"  {name:20s} {kb:8.1f} KB")


def _style_axes(ax):
    ax.set_facecolor(PALETTE["bg"])
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(PALETTE["line"])
    ax.tick_params(colors=PALETTE["ink_soft"], labelsize=9)
    ax.xaxis.label.set_color(PALETTE["ink_soft"])
    ax.yaxis.label.set_color(PALETTE["ink_soft"])
    ax.title.set_color(PALETTE["ink"])
    ax.grid(True, color=PALETTE["line"], linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)


def write_charts(out: pd.DataFrame, shap_vals: np.ndarray,
                 features: list[str], dest: Path) -> None:
    """Diagnostics restyled to the site palette so they read as native."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    plt.rcParams.update({
        "figure.facecolor": PALETTE["bg"],
        "savefig.facecolor": PALETTE["bg"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
    })
    dest.mkdir(parents=True, exist_ok=True)
    M = 1e6

    # 1. Predicted vs actual, year-1 rows (the model's training domain).
    y1 = out[out["year_in_contract"] == 1]
    fig, ax = plt.subplots(figsize=(7, 6.2))
    _style_axes(ax)
    plain = y1[~y1["is_capped"]]
    capped = y1[y1["is_capped"]]
    ax.scatter(plain["actual_salary"] / M, plain["pred_salary"] / M, s=16,
               color=PALETTE["teal"], alpha=0.55, linewidths=0,
               label="Uncensored")
    ax.scatter(capped["actual_salary"] / M, capped["pred_salary"] / M, s=34,
               color=PALETTE["pink"], alpha=0.9, linewidths=0,
               label="CBA-capped (Grabit)")
    lim = max(y1["actual_salary"].max(), y1["pred_salary"].max()) / M * 1.05
    ax.plot([0, lim], [0, lim], color=PALETTE["ink_soft"], linewidth=1,
            linestyle="--", alpha=0.6)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Actual salary ($M)")
    ax.set_ylabel("Predicted salary ($M)")
    ax.set_title("Predicted vs actual — year-1 contracts")
    leg = ax.legend(frameon=False, loc="upper left")
    for t in leg.get_texts():
        t.set_color(PALETTE["ink_soft"])
    fig.tight_layout()
    fig.savefig(dest / "pred_vs_actual.png", dpi=160)
    plt.close(fig)

    # 2. Mean |SHAP| per feature.
    imp = np.abs(shap_vals).mean(axis=0)
    order = np.argsort(imp)
    fig, ax = plt.subplots(figsize=(7, 5.4))
    _style_axes(ax)
    ax.grid(True, axis="y", alpha=0)
    labels = [FEATURE_LABELS.get(features[i], features[i]) for i in order]
    ax.barh(labels, imp[order] * 100, color=PALETTE["green"], height=0.68)
    ax.set_xlabel("Mean |SHAP| (percentage points of the cap)")
    ax.set_title("Feature contribution to predicted value")
    fig.tight_layout()
    fig.savefig(dest / "shap_importance.png", dpi=160)
    plt.close(fig)

    # 3. Signing Residual by signing mechanism (year-1 rows only).
    st = y1.dropna(subset=["signing_type"])
    if len(st):
        agg = (st.groupby("signing_type")
                 .agg(bias=("surplus", "mean"), n=("surplus", "size"))
                 .query("n >= 5")
                 .sort_values("bias"))
        if len(agg):
            fig, ax = plt.subplots(figsize=(7.6, 4.8))
            _style_axes(ax)
            ax.grid(True, axis="y", alpha=0)
            colors = [PALETTE["gold"] if v < 0 else PALETTE["teal"]
                      for v in agg["bias"] / M]
            ax.barh([f"{i}  (n={int(n)})" for i, n in
                     zip(agg.index, agg["n"])],
                    agg["bias"] / M, color=colors, height=0.66)
            ax.axvline(0, color=PALETTE["ink_soft"], linewidth=1)
            ax.set_xlabel("Mean Signing Residual ($M)"
                          "\n← model underprices        overprices →")
            ax.set_title("Systematic bias by CBA signing mechanism", loc="left")
            fig.tight_layout()
            fig.savefig(dest / "residual_by_signing_type.png", dpi=160)
            plt.close(fig)

    # 4. SHAP beeswarm on the top features, site colormap.
    cmap = LinearSegmentedColormap.from_list(
        "site", [PALETTE["teal"], PALETTE["bg_soft"], PALETTE["pink"]]
    )
    top = np.argsort(imp)[-8:]
    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    _style_axes(ax)
    ax.grid(True, axis="y", alpha=0)
    rng = np.random.default_rng(42)
    for row, fi in enumerate(top):
        vals = shap_vals[:, fi] * 100
        raw = out[features[fi]].values if features[fi] in out.columns else None
        if raw is None or not np.isfinite(pd.to_numeric(raw, errors="coerce")).any():
            color = PALETTE["ink_soft"]
            sc_kw = dict(color=color)
        else:
            raw = pd.to_numeric(raw, errors="coerce").astype(float)
            lo, hi = np.nanpercentile(raw, [5, 95])
            norm = np.clip((raw - lo) / (hi - lo + 1e-12), 0, 1)
            sc_kw = dict(c=norm, cmap=cmap, vmin=0, vmax=1)
        jitter = rng.normal(0, 0.11, len(vals))
        ax.scatter(vals, np.full(len(vals), row) + jitter, s=7,
                   alpha=0.5, linewidths=0, **sc_kw)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([FEATURE_LABELS.get(features[i], features[i])
                        for i in top])
    ax.axvline(0, color=PALETTE["ink_soft"], linewidth=1, alpha=0.7)
    ax.set_xlabel("SHAP value (percentage points of the cap)")
    ax.set_title("Per-player feature impact  ·  colour = feature value")
    fig.tight_layout()
    fig.savefig(dest / "shap_beeswarm.png", dpi=160)
    plt.close(fig)

    for p in sorted(dest.glob("*.png")):
        print(f"  {p.name:32s} {p.stat().st_size / 1024:7.1f} KB")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help=f"output directory (default: {DEFAULT_OUT})")
    args = ap.parse_args()

    df = load_training_data()
    print(f"Loaded {len(df)} player-seasons")

    results, model, features = train_grabit(df, sigma=0.02)
    print(f"\nGrabit v3: CV R² {results['cv_r2_mean']:.4f}  "
          f"(2024-26: {results['cv_r2_recent']:.4f}, n={results['recent_n']})")

    # The site quotes CV metrics from train.py's own run. If this fit has
    # drifted from that one, the headline numbers and the table would describe
    # different models.
    canon = OUTPUTS_DIR / "models" / "grabit_results.json"
    if canon.exists():
        prev = json.loads(canon.read_text(encoding="utf-8"))
        gap = abs(prev["cv_r2_mean"] - results["cv_r2_mean"])
        if gap > 1e-6:
            raise SystemExit(
                f"CV R² differs from {canon.name} by {gap:.2e} "
                f"({prev['cv_r2_mean']:.6f} vs {results['cv_r2_mean']:.6f}). "
                "Re-run src/model/train.py so the quoted metrics match."
            )
        print(f"  matches {canon.name}")

    tr_features, medians = _training_medians(df)
    if tr_features != features:
        raise SystemExit("feature list drifted between fit and export")

    out, shap_vals = build_frame(df, model, features, medians)
    print(f"Scored {len(out)} rows  "
          f"({int((out['year_in_contract'] == 1).sum())} year-1, "
          f"{int(out['is_capped'].sum())} CBA-capped)")

    print("\nJSON:")
    write_json(out, shap_vals, features, results, args.out)
    print("\nCharts:")
    write_charts(out, shap_vals, features, args.out / "charts")

    # Keep a copy in-repo so the export is reproducible without the site.
    snap = OUTPUTS_DIR / "web"
    snap.mkdir(parents=True, exist_ok=True)
    out.to_csv(snap / "valuations_export.csv", index=False, encoding="utf-8")
    print(f"\nSnapshot: {snap / 'valuations_export.csv'}")
    print(f"Done -> {args.out}")


if __name__ == "__main__":
    main()
