"""Phase 3C: Parse Spotrac FA data, derive stayed_with_team, test as feature."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import unicodedata, re
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from xgboost import XGBRegressor
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.metrics import r2_score, mean_absolute_error

from config import PROCESSED_DIR, OUTPUTS_DIR, CAP_BY_SEASON, CACHE_DIR
from src.model.train import load_training_data, _filter_year1, _filter_rookie_scale, FEATURE_COLS, TARGET

PARAMS = dict(
    n_estimators=500, max_depth=4, learning_rate=0.01,
    subsample=0.7, colsample_bytree=0.7, min_child_weight=10,
    tree_method="hist",
)
SEEDS = list(range(42, 52))
V2_FEATURES = list(FEATURE_COLS)

SPOTRAC_TEAM_MAP = {
    "BKN": "BRK", "PHX": "PHO", "CHA": "CHO", "NOR": "NOP",
    "NO": "NOP", "GS": "GSW", "SA": "SAS", "NY": "NYK",
}


def norm(name):
    name = str(name).strip().lower()
    nfkd = unicodedata.normalize("NFKD", name)
    out = "".join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", out.replace(".", "").replace("-", " ")).strip()


def parse_spotrac():
    all_rows = []
    for year in range(2019, 2026):
        path = CACHE_DIR / f"spotrac_fa_{year}.html"
        if not path.exists():
            continue
        soup = BeautifulSoup(
            open(path, "r", encoding="utf-8").read(), "html.parser"
        )
        tables = soup.find_all("table")
        if not tables:
            continue
        t = tables[0]
        for row in t.find_all("tr")[1:]:
            cells = [td.get_text(strip=True) for td in row.find_all("td")]
            if len(cells) < 8:
                continue
            from_team = SPOTRAC_TEAM_MAP.get(cells[0], cells[0])
            to_team = SPOTRAC_TEAM_MAP.get(cells[2], cells[2])
            all_rows.append({
                "player_name_norm": norm(cells[3]),
                "season": year,
                "from_team": from_team,
                "to_team": to_team,
                "stayed_with_team": int(from_team == to_team and from_team != ""),
            })
    return pd.DataFrame(all_rows)


def evaluate(df, features, seeds=SEEDS):
    cv_scores, holdouts = [], {}
    for hs in [2024, 2025, 2026]:
        train_raw = df[df["season"] < hs].copy()
        train_f = _filter_year1(train_raw)
        train_f = _filter_rookie_scale(train_f)
        test_raw = df[df["season"] == hs].copy()
        test_f = _filter_year1(test_raw)
        test_f = _filter_rookie_scale(test_f)
        if len(test_f) < 5:
            continue
        avail = [f for f in features if f in train_f.columns]
        X_tr = train_f[avail].copy()
        med = X_tr.median()
        X_tr = X_tr.fillna(med).fillna(0)
        X_te = test_f[avail].copy().fillna(med).fillna(0)
        r2s, maes = [], []
        for seed in seeds:
            xgb = XGBRegressor(random_state=seed, **PARAMS)
            xgb.fit(X_tr, train_f[TARGET].values)
            pred = xgb.predict(X_te)
            cap = CAP_BY_SEASON.get(hs, 153_000_000)
            r2s.append(r2_score(test_f[TARGET].values, pred))
            maes.append(mean_absolute_error(test_f[TARGET].values, pred) * cap)
        holdouts[hs] = (np.mean(r2s), np.mean(maes))

    df_full = _filter_year1(df.copy())
    df_full = _filter_rookie_scale(df_full)
    avail = [f for f in features if f in df_full.columns]
    X = df_full[avail].fillna(df_full[avail].median()).fillna(0)
    cv_r2s = []
    for seed in seeds:
        xgb = XGBRegressor(random_state=seed, **PARAMS)
        cv = GroupKFold(n_splits=5)
        scores = cross_validate(xgb, X, df_full[TARGET].values,
                                groups=df_full["player_name_norm"].values,
                                cv=cv, scoring="r2")
        cv_r2s.append(np.mean(scores["test_score"]))
    return {"cv_r2": np.mean(cv_r2s), "holdouts": holdouts}


def main():
    df = load_training_data()
    print(f"Loaded {len(df)} rows")

    # Parse Spotrac
    fa = parse_spotrac()
    print(f"\nSpotrac FA signings: {len(fa)}")
    print(f"  stayed_with_team rate: {fa['stayed_with_team'].mean():.2%}")

    # Merge onto training data
    df = df.merge(
        fa[["player_name_norm", "season", "stayed_with_team"]],
        on=["player_name_norm", "season"],
        how="left",
    )
    matched = df["stayed_with_team"].notna().sum()
    print(f"  Matched to training data: {matched}/{len(df)} ({matched/len(df):.1%})")

    # Fill unmatched: for year_in_contract=1, if not in FA data, assume stayed (Bird extension)
    # For year_in_contract>1, not applicable (mid-contract)
    df["stayed_with_team"] = df["stayed_with_team"].fillna(0).astype(int)

    # Correlation
    yr1 = df[df["year_in_contract"] == 1]
    corr = yr1["stayed_with_team"].corr(yr1["cap_pct"])
    print(f"  Correlation with cap_pct (yr1 only): {corr:+.3f}")

    # V2 Baseline
    print(f"\n{'='*60}")
    print("  V2 BASELINE (13 features)")
    baseline = evaluate(df, V2_FEATURES)
    print(f"  CV R²: {baseline['cv_r2']:.4f}")
    for hs, (r2, mae) in baseline["holdouts"].items():
        print(f"  {hs} HO: R²={r2:.4f}  MAE=${mae/1e6:.1f}M")

    # Test stayed_with_team
    feats_stayed = V2_FEATURES + ["stayed_with_team"]
    result = evaluate(df, feats_stayed)
    dcv = result["cv_r2"] - baseline["cv_r2"]
    print(f"\n{'='*60}")
    print("  3C: + stayed_with_team")
    print(f"  CV R²: {result['cv_r2']:.4f}  (Δ={dcv:+.4f})")
    for hs in [2024, 2025, 2026]:
        if hs in result["holdouts"] and hs in baseline["holdouts"]:
            r2, mae = result["holdouts"][hs]
            r2b, maeb = baseline["holdouts"][hs]
            print(f"  {hs} HO: R²={r2:.4f} (Δ={r2-r2b:+.4f})  MAE=${mae/1e6:.1f}M (Δ={mae/1e6-maeb/1e6:+.1f}M)")
    passed = dcv >= 0.003
    print(f"  VERDICT: {'PASS' if passed else 'FAIL'} (threshold=0.003, ΔCV={dcv:+.4f})")

    # Also test combined with prev_cap_pct (the 3B winner)
    # Build prev_cap_pct
    df_with_prev = df.sort_values(["player_name_norm", "season"]).copy()
    df_with_prev["prev_cap_pct"] = np.nan
    yr1_rows = df_with_prev[df_with_prev["year_in_contract"] == 1].sort_values(
        ["player_name_norm", "season"]
    )
    for player in yr1_rows["player_name_norm"].unique():
        player_yr1 = yr1_rows[yr1_rows["player_name_norm"] == player].sort_values("season")
        cap_pcts = player_yr1["cap_pct"].values
        for i in range(1, len(cap_pcts)):
            idx = player_yr1.index[i]
            df_with_prev.loc[idx, "prev_cap_pct"] = cap_pcts[i - 1]
    median_rookie = df_with_prev[df_with_prev["draft_pick"] <= 30]["cap_pct"].median()
    df_with_prev["prev_cap_pct"] = df_with_prev["prev_cap_pct"].fillna(median_rookie)

    # Test: prev_cap_pct + stayed_with_team
    feats_combined = V2_FEATURES + ["prev_cap_pct", "stayed_with_team"]
    result_comb = evaluate(df_with_prev, feats_combined)
    dcv_c = result_comb["cv_r2"] - baseline["cv_r2"]
    print(f"\n{'='*60}")
    print("  3B+3C: + prev_cap_pct + stayed_with_team")
    print(f"  CV R²: {result_comb['cv_r2']:.4f}  (Δ={dcv_c:+.4f})")
    for hs in [2024, 2025, 2026]:
        if hs in result_comb["holdouts"] and hs in baseline["holdouts"]:
            r2, mae = result_comb["holdouts"][hs]
            r2b, maeb = baseline["holdouts"][hs]
            print(f"  {hs} HO: R²={r2:.4f} (Δ={r2-r2b:+.4f})  MAE=${mae/1e6:.1f}M (Δ={mae/1e6-maeb/1e6:+.1f}M)")


if __name__ == "__main__":
    main()
