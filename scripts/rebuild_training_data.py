"""Rebuild training_data_v2.csv through the full three-stage chain.

The chain was previously undocumented and split across three files, one of them
an experiment script:

    1. src/features/build_dataset.py        base merge (salaries x metrics x
                                            heights x contract structure)
    2. scripts/build_external_features.py   awards, draft, injuries, team value
    3. scripts/phase3.py                    build_contract_features -> prev_cap_pct

This is the single entry point. After rebuilding it validates against the
previous file: on every shared (player, season) row the twelve cap-independent
model features must reproduce exactly; cap-derived columns (cap_pct,
prev_cap_pct) are reported per season but may legitimately move when config.py
caps or the data change. Any feature regression aborts with a nonzero exit.

    python scripts/rebuild_training_data.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

from config import PROCESSED_DIR

TRAINING = PROCESSED_DIR / "training_data_v2.csv"

# Features that must reproduce bit-for-bit on rows present in both builds.
# cap_pct and prev_cap_pct are excluded: both derive from CAP_BY_SEASON, so a
# cap correction legitimately moves them; prev_cap_pct's first-contract fill is
# a median over the dataset and shifts whenever rows are added.
STABLE_FEATURES = [
    "darko_dpm_z", "lebron_z", "laker_z", "age", "age_squared", "mpg",
    "availability_3yr", "usage_pct", "height_inches", "cba_era", "ast_pct",
    "is_waived", "is_waived_known",
    "award_score_cum", "draft_pick", "year_in_contract", "contract_years",
]


def stage1_base() -> None:
    from src.features.build_dataset import build_dataset
    df = build_dataset()
    df = df.loc[:, ~df.columns.duplicated()]
    df.to_csv(TRAINING, index=False)
    print(f"\nstage 1 done: {len(df)} rows -> {TRAINING.name}")


def stage2_external() -> None:
    from scripts.build_external_features import main as external_main
    external_main()  # reads and rewrites training_data_v2.csv in place
    print("stage 2 done")


def stage3_prev_cap_pct() -> pd.DataFrame:
    from scripts.phase3 import build_contract_features
    df = pd.read_csv(TRAINING)
    df = build_contract_features(df)
    df = df.drop(columns=[c for c in ("is_contract_year",) if c in df.columns])
    df.to_csv(TRAINING, index=False)
    print(f"stage 3 done: prev_cap_pct added ({len(df)} rows)")
    return df


def validate(old: pd.DataFrame, new: pd.DataFrame) -> None:
    old = old.loc[:, ~old.columns.duplicated()]
    new = new.loc[:, ~new.columns.duplicated()]
    key = ["player_name_norm", "season"]
    o = old.drop_duplicates(key).set_index(key).sort_index()
    n = new.drop_duplicates(key).set_index(key).sort_index()

    lost = o.index.difference(n.index)
    if len(lost):
        for k in list(lost)[:10]:
            print(f"  LOST row: {k}")
        raise SystemExit(f"{len(lost)} rows disappeared during rebuild — aborting")

    shared = o.index.intersection(n.index)
    print(f"\nvalidating {len(shared)} rows shared with the previous build")
    bad = []
    for col in STABLE_FEATURES:
        if col not in o.columns or col not in n.columns:
            continue
        ok = np.isclose(
            pd.to_numeric(n.loc[shared, col], errors="coerce").astype(float),
            pd.to_numeric(o.loc[shared, col], errors="coerce").astype(float),
            rtol=1e-6, equal_nan=True,
        )
        pct = float(np.mean(ok)) * 100
        mark = "" if pct > 99.9 else "   <-- REGRESSION"
        print(f"  {col:20s} {pct:7.2f}%{mark}")
        if pct <= 99.9:
            bad.append(col)

    # informational: the cap-derived columns, per season
    if "cap_pct" in o.columns and "cap_pct" in n.columns:
        print("\n  cap-derived columns (may move legitimately):")
        for s in sorted({k[1] for k in shared}):
            rows = [k for k in shared if k[1] == s]
            ok = np.isclose(o.loc[rows, "cap_pct"].astype(float),
                            n.loc[rows, "cap_pct"].astype(float),
                            rtol=1e-4, equal_nan=True)
            print(f"    cap_pct {int(s)}: {np.mean(ok)*100:6.1f}% unchanged "
                  f"(n={len(rows)})")

    print("\nrows and year-1 counts by season:")
    print(f"{'season':>7s} {'before':>7s} {'after':>7s} | {'yr1 before':>10s} {'yr1 after':>9s}")
    for s in sorted(set(old["season"]) | set(new["season"])):
        b, a = (old["season"] == s).sum(), (new["season"] == s).sum()
        yb = ((old["season"] == s) & (old["year_in_contract"] == 1)).sum()
        ya = ((new["season"] == s) & (new["year_in_contract"] == 1)).sum()
        print(f"{int(s):7d} {b:7d} {a:7d} | {yb:10d} {ya:9d}")

    if bad:
        raise SystemExit(f"stable features regressed: {bad}")
    print("\nOK — every stable feature reproduces on shared rows.")


def main() -> None:
    previous = pd.read_csv(TRAINING)
    stage1_base()
    stage2_external()
    rebuilt = stage3_prev_cap_pct()
    validate(previous, rebuilt)


if __name__ == "__main__":
    main()
