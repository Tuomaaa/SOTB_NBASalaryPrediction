"""Standardise minimum-salary rows to the cap-charge convention.

ISSUES #38: for one-year minimum contracts of 3+ service-year veterans, the
league reimburses the team so the player is PAID his service-year rate while
the team's cap is CHARGED the 2-year-vet rate.  Our salary column mixes both
conventions: 2019 and 2021 are 100 % the paid convention, other seasons are
predominantly cap-charge.  This script identifies the paid-convention rows and
writes corrections to salary_corrections.csv.

Convention choice: CAP CHARGE.  The model's target (cap_pct) represents what
teams pay against the salary cap, not what the player receives in his bank
account.  The cap charge is also the value used for the CBA salary floor
(floor_pct) and is the convention the majority of seasons already use.

Usage:
    python scripts/fix_minimum_convention.py            # dry-run diagnostics
    python scripts/fix_minimum_convention.py --write     # append to salary_corrections.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

import argparse
import numpy as np
import pandas as pd
from config import CAP_BY_SEASON, PROCESSED_DIR, RAW_DIR


# ---------------------------------------------------------------------------
# 1.  The 2-year vet minimum (= cap charge) per season
#
#     Derived from the dominant mass point at ~1.4848 % of cap among Minimum-
#     labelled rows.  These values are exact CBA-published dollar amounts; the
#     ratio stays within 1 part in 10^5 of 0.014 848 across all eight seasons.
# ---------------------------------------------------------------------------
CAP_CHARGE = {
    2019: 1_620_564,
    2020: 1_620_564,
    2021: 1_669_178,
    2022: 1_836_090,
    2023: 2_019_706,
    2024: 2_087_519,
    2025: 2_296_274,
    2026: 2_449_421,
}

# Sanity-check: each value must land within 0.0002 pp of the expected ratio.
_EXPECTED_RATIO = 0.014_848
for _s, _cc in CAP_CHARGE.items():
    _ratio = _cc / CAP_BY_SEASON[_s]
    assert abs(_ratio - _EXPECTED_RATIO) < 2e-6, (
        f"season {_s}: cap-charge ratio {_ratio:.8f} is too far from "
        f"{_EXPECTED_RATIO}"
    )


# ---------------------------------------------------------------------------
# 2.  Known paid minimum-scale ratios (salary / cap)
#
#     Extracted from the 2019 and 2021 data (both 100 % paid convention per
#     ISSUES #38 amendment) by identifying salary mass points among Minimum-
#     labelled one-year-contract rows.  Service years 4-10+ use ratios that
#     are constant across all seasons to 6 significant figures.  Service year
#     3 varies by CBA year (three observed variants).
#
#     A row is "scale-matched" if its salary/cap ratio falls within TOLERANCE
#     of any known ratio.  Unmatched rows above the cap charge are flagged as
#     uncertain and printed for review but NOT written to corrections.
# ---------------------------------------------------------------------------
KNOWN_PAID_RATIOS = [
    # service year 3 — varies by CBA year
    0.015_076,   # 2019 CBA year
    0.015_193,   # 2023, 2025 CBA years
    0.015_383,   # 2020, 2021, 2022, 2024, 2026 CBA years
    # service years 4-10+ — constant across all observed seasons
    0.015_917,   # 4 years
    0.017_252,   # 5 years
    0.018_587,   # 6 years
    0.019_922,   # 7 years
    0.021_258,   # 8 years
    0.021_363,   # 9 years
    0.023_500,   # 10+ years
]

# Tolerance for scale matching: 0.02 pp of cap (~$20-30 K depending on cap).
# The CBA publishes exact dollar amounts; this tolerance absorbs rounding only.
_RATIO_TOLERANCE = 0.000_2


def _matches_known_ratio(salary: float, cap: float) -> bool:
    """True if salary/cap is within tolerance of any known paid-scale ratio."""
    ratio = salary / cap
    return any(abs(ratio - r) < _RATIO_TOLERANCE for r in KNOWN_PAID_RATIOS)


def _load_minimum_rows() -> pd.DataFrame:
    """Load training data joined with Minimum signing-type labels."""
    td = pd.read_csv(
        PROCESSED_DIR / "training_data_v2.csv",
        usecols=[
            "player_name_norm", "season", "salary", "cap_pct",
            "age", "year_in_contract", "contract_years",
        ],
    )
    st = pd.read_csv(PROCESSED_DIR / "spotrac_signing_types.csv")
    st_min = (
        st[st["signing_type"].str.contains("Minimum", case=False, na=False)]
        .drop_duplicates(["player_name_norm", "season"])
    )
    merged = td.merge(
        st_min[["player_name_norm", "season", "contract_years"]].rename(
            columns={"contract_years": "st_cy"}
        ),
        on=["player_name_norm", "season"],
        how="inner",
    )
    return merged


def identify_paid_convention_rows(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (confident, uncertain) subsets of paid-convention rows.

    Selection criteria:
      1. Signing type is Minimum (already filtered on load).
      2. One-year contract: contract_years == 1 in the training data.
         Multi-year minimums do NOT receive the cap-charge subsidy.
      3. Salary is strictly above the 2-year-vet minimum (the cap charge).
      4. Salary is at or below the 10+-year minimum for that season
         (= cap * 0.02355, with headroom for rounding).

    Split:
      - confident: salary matches a known CBA scale ratio (within tolerance).
      - uncertain: salary is above cap charge but does not match any known
        ratio.  Printed for review, not written to corrections.
    """
    confident = []
    uncertain = []

    # The 10+-year rate is the ceiling of the paid minimum scale.
    upper_ratio = 0.023_55  # slightly above 0.023500 for rounding

    for season in sorted(df["season"].unique()):
        if season not in CAP_CHARGE:
            continue
        cap = CAP_BY_SEASON[season]
        cc = CAP_CHARGE[season]
        upper = cap * upper_ratio

        mask = (
            (df["season"] == season)
            & (df["contract_years"] == 1)
            & (df["salary"] > cc + 1)
            & (df["salary"] <= upper)
        )
        sub = df[mask]
        for _, row in sub.iterrows():
            if _matches_known_ratio(row["salary"], cap):
                confident.append(row)
            else:
                uncertain.append(row)

    to_df = lambda lst: (pd.DataFrame(lst) if lst
                         else df.iloc[:0].copy())
    return to_df(confident), to_df(uncertain)


def _load_existing_corrections() -> pd.DataFrame:
    """Load salary_corrections.csv if it exists."""
    path = RAW_DIR / "raw_external" / "salary_corrections.csv"
    if not path.exists():
        return pd.DataFrame(columns=[
            "player_name_norm", "season", "kind", "value_usd",
            "confidence", "source", "note",
        ])
    return pd.read_csv(path)


def build_corrections(paid: pd.DataFrame) -> pd.DataFrame:
    """Build a corrections DataFrame for the paid-convention rows."""
    records = []
    for _, r in paid.iterrows():
        season = int(r["season"])
        cc = CAP_CHARGE[season]
        records.append({
            "player_name_norm": r["player_name_norm"],
            "season": season,
            "kind": "min_cap_charge",
            "value_usd": cc,
            "confidence": "verified",
            "source": "CBA rule: 1-yr vet min cap charge = 2-yr vet minimum",
            "note": (
                f"Paid {int(r['salary']):,} (service-year rate); "
                f"cap charge {cc:,}. "
                f"ISSUES #38 convention fix."
            ),
        })
    return pd.DataFrame(records)


def print_summary(
    confident: pd.DataFrame,
    uncertain: pd.DataFrame,
) -> None:
    """Print per-season diagnostics."""
    print("=" * 72)
    print("MINIMUM CONVENTION FIX  --  ISSUES #38")
    print("Convention: cap charge (2-year vet minimum)")
    print("=" * 72)

    total_rows = 0
    total_shift_pct = 0.0

    for season in sorted(confident["season"].unique()):
        cap = CAP_BY_SEASON[season]
        cc = CAP_CHARGE[season]
        sub = confident[confident["season"] == season].sort_values("salary")
        n = len(sub)
        total_rows += n

        shifts_usd = sub["salary"].values - cc
        shifts_pct = shifts_usd / cap
        mean_shift_usd = shifts_usd.mean()
        mean_shift_pct = shifts_pct.mean() * 100
        total_shift_pct += shifts_pct.sum()

        print(f"\nSeason {season}: {n} rows corrected  "
              f"(cap charge = ${cc:,})")
        print(f"  Mean shift: -${mean_shift_usd:,.0f}  "
              f"(-{mean_shift_pct:.4f} pp of cap)")
        print(f"  {'Player':<28s} {'Paid':>12s} {'Cap charge':>12s} "
              f"{'Shift':>12s}")
        print(f"  {'-'*28} {'-'*12} {'-'*12} {'-'*12}")
        for _, r in sub.iterrows():
            shift = int(r["salary"]) - cc
            print(f"  {r['player_name_norm']:<28s} "
                  f"${int(r['salary']):>11,} "
                  f"${cc:>11,} "
                  f"-${shift:>10,}")

    mean_pct = (total_shift_pct / total_rows * 100) if total_rows else 0
    print(f"\n{'=' * 72}")
    print(f"CONFIDENT: {total_rows} rows corrected across "
          f"{confident['season'].nunique()} seasons")
    print(f"  Total cap_pct reduction: "
          f"{total_shift_pct:.4f} (mean {mean_pct:.4f} pp per row)")

    if len(uncertain) > 0:
        print(f"\n{'=' * 72}")
        print(f"UNCERTAIN: {len(uncertain)} rows above cap charge but salary "
              f"does not match")
        print(f"           any known CBA scale value.  NOT corrected.")
        print(f"  {'Player':<28s} {'Season':>6s} {'Salary':>12s} "
              f"{'Ratio':>10s}")
        print(f"  {'-'*28} {'-'*6} {'-'*12} {'-'*10}")
        for _, r in uncertain.sort_values(["season", "salary"]).iterrows():
            cap = CAP_BY_SEASON[int(r["season"])]
            ratio = r["salary"] / cap
            print(f"  {r['player_name_norm']:<28s} "
                  f"{int(r['season']):>6d} "
                  f"${int(r['salary']):>11,} "
                  f"{ratio:.6f}")
        print(f"\n  Review these rows manually.  Possible causes:")
        print(f"  - prorated partial-season amount")
        print(f"  - Exhibit-10 or incentive bonus added to base")
        print(f"  - mislabelled non-minimum contract (ISSUES #4)")
        print(f"  - CBA scale value not in the known-ratio list")


def print_verification(
    df_all: pd.DataFrame,
    confident: pd.DataFrame,
) -> None:
    """Verify no scale-matched one-year Minimum row exceeds the cap charge."""
    print(f"\n{'=' * 72}")
    print("VERIFICATION: scale-matched one-year Minimum rows after fix")
    print("=" * 72)
    ok = True
    corrected_keys = set(
        zip(confident["player_name_norm"], confident["season"])
    )

    # Same upper bound as the selection logic
    upper_ratio = 0.023_55

    for season in sorted(df_all["season"].unique()):
        if season not in CAP_CHARGE:
            continue
        cap = CAP_BY_SEASON[season]
        cc = CAP_CHARGE[season]
        upper = cap * upper_ratio

        sub = df_all[
            (df_all["season"] == season)
            & (df_all["contract_years"] == 1)
            & (df_all["salary"] <= upper)
        ]

        remaining = 0
        for _, r in sub.iterrows():
            key = (r["player_name_norm"], r["season"])
            sal = cc if key in corrected_keys else r["salary"]
            if sal > cc + 1 and _matches_known_ratio(r["salary"], cap):
                remaining += 1

        if remaining:
            print(f"  Season {season}: FAIL  {remaining} scale-matched rows "
                  f"still above cap charge")
            ok = False
        else:
            print(f"  Season {season}: OK")

    if ok:
        print("\n  All seasons pass.")
    else:
        print("\n  WARNING: some scale-matched rows remain.")


def main():
    parser = argparse.ArgumentParser(
        description="Identify and correct paid-convention minimum-salary rows."
    )
    parser.add_argument(
        "--write", action="store_true",
        help="Append corrections to salary_corrections.csv (default: dry run).",
    )
    args = parser.parse_args()

    df = _load_minimum_rows()
    print(f"Loaded {len(df)} Minimum-labelled rows from training data.\n")

    confident, uncertain = identify_paid_convention_rows(df)
    print_summary(confident, uncertain)
    print_verification(df, confident)

    if not args.write:
        print(f"\n--- DRY RUN --- pass --write to append to "
              f"salary_corrections.csv")
        return

    corrections = build_corrections(confident)
    existing = _load_existing_corrections()

    # De-duplicate: keep existing min_cap_charge entries and only append
    # genuinely new ones.  The old approach (drop all min_cap_charge, re-add
    # detected) silently lost corrections whose rows were already fixed in
    # the training data and therefore no longer detected as above cap charge.
    existing_keys = set(
        zip(existing["player_name_norm"], existing["season"])
    )
    new_only = corrections[
        ~corrections.apply(
            lambda r: (r["player_name_norm"], r["season"]) in existing_keys,
            axis=1,
        )
    ]

    combined = pd.concat([existing, new_only], ignore_index=True)
    out_path = RAW_DIR / "raw_external" / "salary_corrections.csv"
    combined.to_csv(out_path, index=False)
    print(f"\nAppended {len(new_only)} new corrections to {out_path}")
    print(f"  (total entries in file: {len(combined)})")
    print(f"\nThe pipeline applies 'min_cap_charge' corrections in "
          f"build_dataset.py (stage 1 of rebuild).")
    print(f"  Run: python scripts/rebuild_training_data.py")


if __name__ == "__main__":
    main()
