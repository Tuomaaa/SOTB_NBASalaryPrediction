"""Fixed-row paired evaluation of the player-row missingness repair."""

import argparse
import json
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

from config import CAP_BY_SEASON, OUTPUTS_DIR, PROCESSED_DIR, RAW_DIR
from scripts.diagnostics import attach_signing_labels
from src.features.impact_identity import coalesce_impact_rows, fill_age_from_player_history
from src.features.build_dataset import _resolve_merged_age
from src.model.evaluate_suite import (
    DEFAULT_SEEDS, N_SPLITS, _in_confirmation_set, make_champion_fitter,
    paired_delta,
)
from src.model.extension_cap import attach_extension_cap
from src.model.route_mixture import attach_clf_features
from src.model.train import (
    FEATURE_COLS, TARGET, _compute_floor, _compute_max_eligible,
    _filter_continuations, _filter_mislabeled_year1, _filter_prorated,
    _filter_rookie_scale, _filter_year1, _prepare_Xy,
)

KEY = ["player_name_norm", "season"]
NEW_ZERO_KEYS = {
    ("wenyen gabriel", 2019), ("jontay porter", 2020),
    ("jonathan isaac", 2021), ("omer yurtseven", 2021),
    ("zach collins", 2021), ("jason preston", 2022),
    ("t.j. warren", 2022), ("e.j. liddell", 2023),
    ("tristan thompson", 2023),
}
LEGACY_GAMES = {
    2018: 82, 2019: 82, 2020: 72, 2021: 82,
    2022: 82, 2023: 82, 2024: 82, 2025: 82, 2026: 82,
}
WEIGHTS = [0.5, 0.3, 0.2]


def normalize_name(name: str) -> str:
    """Match the dataset's lowercase, diacritic-free name key."""
    text = unicodedata.normalize("NFKD", str(name).strip().lower())
    return "".join(char for char in text if not unicodedata.combining(char))


def apply_legacy_corrections(impact: pd.DataFrame) -> pd.DataFrame:
    """Apply the pre-missingness-repair correction register."""
    corrections = pd.read_csv(
        RAW_DIR / "raw_external" / "impact_metric_corrections.csv"
    )
    keep = [
        (str(row.player_name_norm), int(row.season)) not in NEW_ZERO_KEYS
        for row in corrections.itertuples(index=False)
    ]
    corrections = corrections[keep]
    out = impact.set_index(KEY, drop=False)
    meta = {"player_name_norm", "season", "source", "note"}
    for row in corrections.itertuples(index=False):
        key = (str(row.player_name_norm), int(row.season))
        for column in corrections.columns.difference(meta):
            value = getattr(row, column)
            if pd.notna(value):
                out.loc[key, column] = value
    return out.reset_index(drop=True)


def legacy_availability(impact: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the old last-three-records calculation, including NaN spread."""
    out = impact.copy()
    out["_max_games"] = out["season"].map(LEGACY_GAMES).fillna(82)
    out["_gp_pct"] = out["games"] / out["_max_games"]
    out = out.sort_values(KEY)
    values = {}
    for player, group in out.groupby("player_name_norm"):
        group = group.sort_values("season")
        seasons = group["season"].tolist()
        gp = group["_gp_pct"].tolist()
        for index, season in enumerate(seasons):
            prior = gp[max(0, index - 2):index + 1][::-1]
            weights = WEIGHTS[:len(prior)]
            values[(player, int(season))] = round(
                sum(value * weight for value, weight in zip(prior, weights))
                / sum(weights), 4
            )
    out["availability_3yr"] = [
        values[(player, int(season))]
        for player, season in zip(out["player_name_norm"], out["season"])
    ]
    return out.drop(columns=["_max_games", "_gp_pct"])


def legacy_feature_map() -> pd.DataFrame:
    """Rebuild only the pre-repair feature columns from unchanged raw sources."""
    salaries = pd.read_csv(PROCESSED_DIR / "salaries.csv")
    impact = pd.read_csv(PROCESSED_DIR / "impact_metrics.csv")
    salaries["player_name_norm"] = salaries["player"].map(normalize_name)
    salary_keys = set(zip(salaries["player_name_norm"], salaries["season"].astype(int)))
    impact = coalesce_impact_rows(impact, salary_keys)
    impact = apply_legacy_corrections(impact)
    impact = fill_age_from_player_history(impact)
    impact = legacy_availability(impact)

    frame = salaries.merge(impact, on=KEY, how="inner", suffixes=("_sal", "_imp"))
    frame["age"] = _resolve_merged_age(frame)
    if "games_imp" in frame:
        frame["games"] = frame["games_imp"]
    frame = frame.drop(
        columns=[column for column in frame if column.endswith(("_sal", "_imp"))],
        errors="ignore",
    )
    frame = fill_age_from_player_history(frame)
    computed_mpg = frame["minutes"] / frame["games"].replace(0, np.nan)
    frame["mpg"] = computed_mpg.fillna(frame.get("mpg"))
    frame["age_squared"] = frame["age"] ** 2
    columns = KEY + [
        "age", "age_squared", "mpg", "availability_3yr", "usage_pct", "ast_pct",
    ]
    return frame[columns]


def baseline_training_data(repaired: pd.DataFrame) -> pd.DataFrame:
    """Overlay legacy feature values onto the current waiver-enabled table."""
    legacy = legacy_feature_map()
    out = repaired.merge(legacy, on=KEY, how="left", suffixes=("", "_legacy"),
                         validate="one_to_one")
    for column in ["age", "age_squared", "mpg", "availability_3yr", "usage_pct", "ast_pct"]:
        out[column] = out.pop(f"{column}_legacy")
    return out


def prepare(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Apply the current production filter and pre-fit preparation chain."""
    df = _filter_rookie_scale(_filter_year1(raw.copy()))
    df = _filter_prorated(df)
    df = _compute_max_eligible(df)
    df = _filter_mislabeled_year1(df)
    df = _filter_continuations(df).reset_index(drop=True)
    df["cap"] = df["season"].map(CAP_BY_SEASON)
    df["salary_m"] = df[TARGET] * df["cap"] / 1e6
    _, _, _, features = _prepare_Xy(df)
    df[features] = df[features].fillna(df[features].median()).fillna(0)
    df["is_confirmation"] = df["player_name_norm"].map(_in_confirmation_set)
    df = attach_signing_labels(df, salary_dollars=df[TARGET] * df["cap"])
    df = _compute_floor(df)
    df = attach_extension_cap(df, verbose=False)
    return attach_clf_features(df)


def score(df, y, pred, mask) -> dict:
    """R2 and dollar error summary for one fixed-row mask."""
    mask = np.asarray(mask, dtype=bool)
    error = (pred[mask] - y[mask]) * df["cap"].to_numpy()[mask] / 1e6
    return {
        "n": int(mask.sum()), "r2": float(r2_score(y[mask], pred[mask])),
        "mae_m": float(np.abs(error).mean()), "bias_m": float(error.mean()),
    }


def run(seeds) -> tuple[dict, pd.DataFrame]:
    """Fit old/new states on identical CV folds and rolling origins."""
    repaired_raw = pd.read_csv(PROCESSED_DIR / "training_data_v2.csv")
    if TARGET not in repaired_raw:
        repaired_raw[TARGET] = repaired_raw["salary"] / repaired_raw["season"].map(CAP_BY_SEASON)
    baseline_raw = baseline_training_data(repaired_raw)
    old, old_clf = prepare(baseline_raw)
    new, new_clf = prepare(repaired_raw)
    if old_clf != new_clf or not old[KEY].equals(new[KEY]):
        raise AssertionError("old/new feature lists or evaluation keys differ")
    features = [feature for feature in FEATURE_COLS if feature in old.columns]
    y = new[TARGET].to_numpy()
    groups = new["player_name_norm"].to_numpy()
    folds = list(GroupKFold(N_SPLITS).split(new, y, groups))
    old_seed = np.zeros((len(new), len(seeds)))
    new_seed = np.zeros((len(new), len(seeds)))
    old_fold = np.zeros((N_SPLITS, len(seeds)))
    new_fold = np.zeros((N_SPLITS, len(seeds)))
    old_sel = np.zeros((N_SPLITS, len(seeds)))
    new_sel = np.zeros((N_SPLITS, len(seeds)))
    fitter = make_champion_fitter(new_clf)
    selection = ~new["is_confirmation"].to_numpy(dtype=bool)
    for fi, (_, va) in enumerate(folds):
        held = set(groups[va])
        old_train = old[~old["player_name_norm"].isin(held)]
        new_train = new[~new["player_name_norm"].isin(held)]
        print(f"fold {fi + 1}/{N_SPLITS}: {len(va)} validation rows")
        for si, seed in enumerate(seeds):
            old_seed[va, si] = fitter(old_train, old.iloc[va], features, seed)
            new_seed[va, si] = fitter(new_train, new.iloc[va], features, seed)
            old_fold[fi, si] = r2_score(y[va], old_seed[va, si])
            new_fold[fi, si] = r2_score(y[va], new_seed[va, si])
            take = selection[va]
            old_sel[fi, si] = r2_score(y[va][take], old_seed[va, si][take])
            new_sel[fi, si] = r2_score(y[va][take], new_seed[va, si][take])

    old_oof, new_oof = old_seed.mean(axis=1), new_seed.mean(axis=1)
    recent = new["season"].to_numpy() >= 2024
    metrics = {}
    for name, mask in {
        "A1_pooled": np.ones(len(new), bool), "A1_selection": selection,
        "A2_recent": recent, "A2_recent_selection": recent & selection,
    }.items():
        metrics[name] = {
            "old": score(new, y, old_oof, mask),
            "new": score(new, y, new_oof, mask),
        }
    metrics["A1_pooled"]["paired"] = paired_delta(old_fold, new_fold)
    metrics["A1_selection"]["paired"] = paired_delta(old_sel, new_sel)

    old_fwd = np.full(len(new), np.nan)
    new_fwd = np.full(len(new), np.nan)
    for season in (2024, 2025, 2026):
        train = new["season"].to_numpy() < season
        test = new["season"].to_numpy() == season
        old_acc = np.zeros(int(test.sum()))
        new_acc = np.zeros(int(test.sum()))
        for seed in seeds:
            old_acc += fitter(old[train], old[test], features, seed)
            new_acc += fitter(new[train], new[test], features, seed)
        old_fwd[test] = old_acc / len(seeds)
        new_fwd[test] = new_acc / len(seeds)
    fwd_mask = ~np.isnan(new_fwd)
    metrics["B1_forward"] = {
        "old": score(new, y, old_fwd, fwd_mask),
        "new": score(new, y, new_fwd, fwd_mask),
    }

    ref = new[KEY + [TARGET, "salary_m", "is_confirmation"]].copy()
    ref["oof_old"] = old_oof
    ref["oof_new"] = new_oof
    payload = {
        "n": len(new), "seeds": list(seeds), "n_splits": N_SPLITS,
        "metrics": metrics,
    }
    return payload, ref


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=len(DEFAULT_SEEDS))
    args = parser.parse_args()
    seeds = DEFAULT_SEEDS[:args.seeds]
    payload, ref = run(seeds)
    for name, result in payload["metrics"].items():
        old, new = result["old"], result["new"]
        line = f"{name}: R2 {old['r2']:.4f} -> {new['r2']:.4f}; MAE ${old['mae_m']:.3f}M -> ${new['mae_m']:.3f}M"
        if "paired" in result:
            delta = result["paired"]
            line += f"; paired {delta['delta']:+.4f} +/- {delta['se']:.4f}, t={delta['t']:+.2f}"
        print(line)
    out = OUTPUTS_DIR / "models"
    out.mkdir(parents=True, exist_ok=True)
    (out / "missingness_repair_evaluation.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    ref.to_csv(out / "missingness_repair_oof.csv", index=False)


if __name__ == "__main__":
    main()
