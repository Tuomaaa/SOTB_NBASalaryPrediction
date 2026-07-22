"""Phase 3: Three independent feature experiments.

Each phase starts from v2 baseline (13 features), tests independently.
Winners (ΔCV >= 0.003) are combined into final v3 model.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.metrics import r2_score, mean_absolute_error

from config import PROCESSED_DIR, OUTPUTS_DIR, CAP_BY_SEASON
from src.model.train import load_training_data, _filter_year1, _filter_rookie_scale, FEATURE_COLS, TARGET

PARAMS = dict(
    n_estimators=500, max_depth=4, learning_rate=0.01,
    subsample=0.7, colsample_bytree=0.7, min_child_weight=10,
    tree_method="hist",
)
SEEDS = list(range(42, 52))
V2_FEATURES = list(FEATURE_COLS)  # 13 features


def evaluate(df, features, seeds=SEEDS):
    """Run CV + 2024/2025/2026 temporal holdouts."""
    cv_scores = []
    holdouts = {}

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

    # Full CV
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

    return {
        "cv_r2": np.mean(cv_r2s),
        "holdouts": holdouts,
    }


def print_comparison(name, result, baseline):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    dcv = result["cv_r2"] - baseline["cv_r2"]
    print(f"  CV R²:  {result['cv_r2']:.4f}  (Δ={dcv:+.4f})")
    for hs in [2024, 2025, 2026]:
        if hs in result["holdouts"] and hs in baseline["holdouts"]:
            r2, mae = result["holdouts"][hs]
            r2b, maeb = baseline["holdouts"][hs]
            print(f"  {hs} HO: R²={r2:.4f} (Δ={r2-r2b:+.4f})  MAE=${mae/1e6:.1f}M (Δ={mae/1e6-maeb/1e6:+.1f}M)")
    passed = dcv >= 0.003
    print(f"  VERDICT: {'PASS' if passed else 'FAIL'} (threshold=0.003, ΔCV={dcv:+.4f})")
    return passed


# ─── Phase 3A: Team cap space ───────────────────────────────────────
def build_team_cap_space(df):
    """Compute team cap space from salaries.csv."""
    sal = pd.read_csv(PROCESSED_DIR / "salaries.csv")
    team_sal = sal.groupby(["team", "season"])["salary"].sum().reset_index()
    team_sal.columns = ["team_abbr", "season", "team_salary"]

    team_sal["cap"] = team_sal["season"].map(CAP_BY_SEASON)
    team_sal["team_cap_space_pct"] = (team_sal["cap"] - team_sal["team_salary"]) / team_sal["cap"]
    team_sal["team_over_cap"] = (team_sal["team_salary"] > team_sal["cap"]).astype(int)

    # Merge onto df by team_abbreviation + season
    df = df.merge(
        team_sal[["team_abbr", "season", "team_cap_space_pct", "team_over_cap"]],
        left_on=["team_abbreviation", "season"],
        right_on=["team_abbr", "season"],
        how="left",
    ).drop(columns=["team_abbr"], errors="ignore")

    matched = df["team_cap_space_pct"].notna().sum()
    print(f"  Team cap space: {matched}/{len(df)} rows matched")
    print(f"  Cap space range: [{df['team_cap_space_pct'].min():.2f}, {df['team_cap_space_pct'].max():.2f}]")
    print(f"  Over cap: {df['team_over_cap'].sum()} rows")
    return df


# ─── Phase 3B: Contract year + prev_cap_pct ────────────────────────
def build_contract_features(df):
    """Derive is_contract_year and prev_cap_pct from contract structure."""
    df = df.sort_values(["player_name_norm", "season"]).copy()

    # is_contract_year: was the player in the FINAL year of their old contract
    # when producing the stats that map to this row?
    # For season S with year_in_contract=1 (new contract), check if at season S-1
    # the player had year_in_contract == contract_years (final year).
    df["is_contract_year"] = 0
    prev = df.set_index(["player_name_norm", "season"])

    for idx, row in df.iterrows():
        if row["year_in_contract"] != 1:
            continue
        pname = row["player_name_norm"]
        prev_season = row["season"] - 1
        key = (pname, prev_season)
        if key in prev.index:
            prev_row = prev.loc[key]
            if isinstance(prev_row, pd.DataFrame):
                prev_row = prev_row.iloc[0]
            if prev_row["year_in_contract"] == prev_row["contract_years"]:
                df.loc[idx, "is_contract_year"] = 1

    # prev_cap_pct: year-1 cap_pct of the PREVIOUS contract
    # For each player at year_in_contract=1, look back to find the most recent
    # year_in_contract=1 row and use its cap_pct.
    df["prev_cap_pct"] = np.nan
    yr1_rows = df[df["year_in_contract"] == 1].sort_values(["player_name_norm", "season"])

    for player in yr1_rows["player_name_norm"].unique():
        player_yr1 = yr1_rows[yr1_rows["player_name_norm"] == player].sort_values("season")
        seasons = player_yr1["season"].values
        cap_pcts = player_yr1["cap_pct"].values

        for i in range(1, len(seasons)):
            idx = player_yr1.index[i]
            df.loc[idx, "prev_cap_pct"] = cap_pcts[i - 1]

    # Fill missing prev_cap_pct with median rookie scale value for first contracts
    median_rookie = df[df["draft_pick"] <= 30]["cap_pct"].median()
    df["prev_cap_pct"] = df["prev_cap_pct"].fillna(median_rookie)

    ct_sum = df["is_contract_year"].sum()
    prev_filled = df["prev_cap_pct"].notna().sum()
    print(f"  is_contract_year=1: {ct_sum}/{len(df)} rows")
    print(f"  prev_cap_pct filled: {prev_filled}/{len(df)} rows")
    print(f"  prev_cap_pct corr with cap_pct: {df['prev_cap_pct'].corr(df['cap_pct']):.3f}")
    return df


# ─── Phase 3C: Spotrac signing type ────────────────────────────────
def try_spotrac_scrape():
    """Attempt to scrape Spotrac. Returns None if blocked."""
    import requests
    import time

    endpoints = [
        "https://www.spotrac.com/nba/free-agents/_/year/2024",
        "https://www.spotrac.com/nba/transactions",
        "https://www.spotrac.com/nba/free-agents",
    ]
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) Gecko/20100101 Firefox/126.0",
    ]

    for url in endpoints:
        for ua in user_agents:
            try:
                resp = requests.get(url, headers={"User-Agent": ua}, timeout=15)
                print(f"  {url[:60]}... UA={ua[:30]}... -> {resp.status_code}")
                if resp.status_code == 200:
                    return url, ua, resp.text
                time.sleep(3)
            except Exception as e:
                print(f"  {url[:60]}... -> ERROR: {e}")
                time.sleep(3)

    # Try HoopsHype as backup
    hh_urls = [
        "https://hoopshype.com/free-agents/",
        "https://hoopshype.com/free-agents/2024/",
    ]
    for url in hh_urls:
        try:
            resp = requests.get(url, headers={"User-Agent": user_agents[0]}, timeout=15)
            print(f"  {url} -> {resp.status_code}")
            if resp.status_code == 200:
                return url, user_agents[0], resp.text
            time.sleep(3)
        except Exception as e:
            print(f"  {url} -> ERROR: {e}")

    return None


# ─── Main ───────────────────────────────────────────────────────────
def main():
    df_raw = load_training_data()
    print(f"Loaded {len(df_raw)} rows\n")

    # ── V2 Baseline ──
    print("=" * 60)
    print("  V2 BASELINE (13 features)")
    print("=" * 60)
    baseline = evaluate(df_raw, V2_FEATURES)
    print(f"  CV R²: {baseline['cv_r2']:.4f}")
    for hs, (r2, mae) in baseline["holdouts"].items():
        print(f"  {hs} HO: R²={r2:.4f}  MAE=${mae/1e6:.1f}M")

    winners = []

    # ── Phase 3A: Team cap space ──
    print(f"\n{'#'*60}")
    print("  PHASE 3A: Team Cap Space")
    print(f"{'#'*60}")
    df_3a = build_team_cap_space(df_raw.copy())

    # Test continuous
    feats_3a_cont = V2_FEATURES + ["team_cap_space_pct"]
    result_3a_cont = evaluate(df_3a, feats_3a_cont)
    passed_cont = print_comparison("3A: + team_cap_space_pct (continuous)", result_3a_cont, baseline)

    # Test binary
    feats_3a_bin = V2_FEATURES + ["team_over_cap"]
    result_3a_bin = evaluate(df_3a, feats_3a_bin)
    passed_bin = print_comparison("3A: + team_over_cap (binary)", result_3a_bin, baseline)

    # Test both
    feats_3a_both = V2_FEATURES + ["team_cap_space_pct", "team_over_cap"]
    result_3a_both = evaluate(df_3a, feats_3a_both)
    passed_both = print_comparison("3A: + both cap space features", result_3a_both, baseline)

    if passed_cont or passed_bin or passed_both:
        best_3a = max(
            [("team_cap_space_pct", result_3a_cont, feats_3a_cont),
             ("team_over_cap", result_3a_bin, feats_3a_bin),
             ("both", result_3a_both, feats_3a_both)],
            key=lambda x: x[1]["cv_r2"],
        )
        winners.append(("3A", best_3a[0], best_3a[1], best_3a[2]))

    # ── Phase 3B: Contract year + prev_cap_pct ──
    print(f"\n{'#'*60}")
    print("  PHASE 3B: Contract Year + Previous Contract Value")
    print(f"{'#'*60}")
    df_3b = build_contract_features(df_raw.copy())

    # Test is_contract_year only
    feats_3b_cy = V2_FEATURES + ["is_contract_year"]
    result_3b_cy = evaluate(df_3b, feats_3b_cy)
    passed_cy = print_comparison("3B: + is_contract_year", result_3b_cy, baseline)

    # Test prev_cap_pct only
    feats_3b_prev = V2_FEATURES + ["prev_cap_pct"]
    result_3b_prev = evaluate(df_3b, feats_3b_prev)
    passed_prev = print_comparison("3B: + prev_cap_pct", result_3b_prev, baseline)

    # Test both
    feats_3b_both = V2_FEATURES + ["is_contract_year", "prev_cap_pct"]
    result_3b_both = evaluate(df_3b, feats_3b_both)
    passed_3b_both = print_comparison("3B: + both contract features", result_3b_both, baseline)

    if passed_cy or passed_prev or passed_3b_both:
        best_3b = max(
            [("is_contract_year", result_3b_cy, feats_3b_cy),
             ("prev_cap_pct", result_3b_prev, feats_3b_prev),
             ("both", result_3b_both, feats_3b_both)],
            key=lambda x: x[1]["cv_r2"],
        )
        winners.append(("3B", best_3b[0], best_3b[1], best_3b[2]))

    # ── Phase 3C: Spotrac ──
    print(f"\n{'#'*60}")
    print("  PHASE 3C: Spotrac Signing Type")
    print(f"{'#'*60}")
    spotrac_result = try_spotrac_scrape()
    if spotrac_result is None:
        print("\n  All Spotrac and HoopsHype endpoints blocked (403).")
        print("  Phase 3C: SKIPPED — manual curation needed.")
    else:
        print(f"\n  Got data from {spotrac_result[0][:60]}...")
        print("  TODO: parse and integrate signing type data")

    # ── Final combination ──
    print(f"\n{'#'*60}")
    print("  FINAL: Combining winners")
    print(f"{'#'*60}")

    if not winners:
        print("  No features passed the 0.003 threshold. V2 is final.")
        return

    # Build combined feature set
    combined_feats = list(V2_FEATURES)
    df_combined = df_raw.copy()

    for phase, feat_name, result, feats in winners:
        print(f"  {phase}: {feat_name} (ΔCV={result['cv_r2']-baseline['cv_r2']:+.4f})")
        for f in feats:
            if f not in combined_feats:
                combined_feats.append(f)

    # Rebuild df with all winner features
    if any(w[0] == "3A" for w in winners):
        df_combined = build_team_cap_space(df_combined)
    if any(w[0] == "3B" for w in winners):
        df_combined = build_contract_features(df_combined)

    print(f"\n  Combined features ({len(combined_feats)}): {combined_feats}")
    result_final = evaluate(df_combined, combined_feats)
    print_comparison("V3 COMBINED", result_final, baseline)


if __name__ == "__main__":
    main()
