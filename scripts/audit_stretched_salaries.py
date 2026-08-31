"""Find training rows whose salary is a waiving team's dead money, not a signing.

The stretch provision spreads a waived player's remaining guaranteed money over
`2 * remaining years + 1` seasons, on the WAIVING team's books. Basketball
Reference reports that obligation on the player's row, so the salary chain can
ingest it as if the player had signed it — while the contract he actually signed
with another team is ignored. `cap_pct` is the model target, so such a row trains
the model on a price no market ever set (ISSUES #36).

The instrument is a disagreement between two sources that should agree: the
observed salary, and the AAV of the Spotrac contract block that best matches it.
A stretched obligation shows up as salary >> AAV, because the salary belongs to
the old contract and the AAV to the new one.

    python scripts/audit_stretched_salaries.py
    python scripts/audit_stretched_salaries.py --ratio 1.5

Exit status is 1 when any stretch-class row is found, so a refresh can gate on it.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

from config import PROCESSED_DIR

# A row is stretch-class when the salary/AAV disagreement is corroborated by a
# waiver that PRECEDES it. "Waived at some point in his career" is too loose — it
# flags Winslow 2019, a genuine rookie-extension salary whose waivers came in
# 2023 and 2024. Without that corroboration the disagreement is a block-matching
# artifact (a renegotiation, or an extension year matched against the contract it
# replaced), which is not a salary defect.
DEFAULT_RATIO = 2.0
MIN_SALARY_M = 2.0
# A stretch runs 2n+1 years, so an obligation can outlive the waiver by a while;
# five years covers every schedule the CBA permits for the contracts we hold.
STRETCH_LOOKBACK_YEARS = 5


def load_frame() -> pd.DataFrame:
    from src.model.evaluate_suite import load_evaluation_frame
    df, _ = load_evaluation_frame(
        verbose=False, allow_missing_computed=True
    )
    return df


def audit(df: pd.DataFrame, ratio: float = DEFAULT_RATIO) -> pd.DataFrame:
    """Join each row to its best-matching Spotrac block and score the gap."""
    st = pd.read_csv(PROCESSED_DIR / "spotrac_signing_types.csv")
    tx = pd.read_csv(PROCESSED_DIR / "spotrac_transactions.csv")
    sd = pd.read_csv(PROCESSED_DIR / "contract_signing_dates.csv")
    sal = pd.read_csv(PROCESSED_DIR / "salaries.csv")

    base = df[["player_name_norm", "season", "salary_m", "cap_pct",
               "is_waived", "prev_cap_pct"]].copy()
    m = base.merge(st, on=["player_name_norm", "season"], how="left")
    m["aav_m"] = m["aav"] / 1e6
    m["gap_m"] = m["salary_m"] - m["aav_m"]
    # give the source every chance to agree: keep the block closest to us
    m = m.loc[m.groupby(["player_name_norm", "season"])["gap_m"]
              .transform(lambda s: s.abs() == s.abs().min())]
    m = m.drop_duplicates(["player_name_norm", "season"])

    waivers = tx[tx["event_type"].eq("waived")].copy()
    waivers["transaction_date"] = pd.to_datetime(waivers["transaction_date"],
                                                 errors="coerce")

    flagged = m[m["aav_m"].notna() & (m["salary_m"] >= MIN_SALARY_M)
                & (m["salary_m"] / m["aav_m"] >= ratio)].copy()
    flagged["ratio"] = flagged["salary_m"] / flagged["aav_m"]

    def prior_waiver(player: str, season: int):
        """The latest waiver before this season's signing window opens."""
        cand = waivers[waivers["player_name_norm"].eq(player)]
        # season N is the N/N+1 season; its market opens on 1 July of year N
        cutoff = pd.Timestamp(year=int(season) + 1, month=1, day=1)
        start = pd.Timestamp(year=int(season) - STRETCH_LOOKBACK_YEARS,
                             month=7, day=1)
        cand = cand[cand["transaction_date"].between(start, cutoff)]
        if cand.empty:
            return None
        # prefer the waiver that names the stretch provision — it is the one
        # that put the money on the old team's books, which is not always the
        # most recent waiver (Noah 2020 was waived again three weeks later)
        named = cand[cand["tx_text"].astype(str)
                     .str.contains("stretch", case=False, na=False)]
        pick = named if not named.empty else cand
        return pick.sort_values("transaction_date").iloc[-1]

    hits = [prior_waiver(p, s) for p, s in
            zip(flagged["player_name_norm"], flagged["season"])]
    flagged["prior_waiver_date"] = [
        None if h is None else h["transaction_date"].date() for h in hits]
    flagged["prior_waiver_text"] = [
        None if h is None else h["tx_text"] for h in hits]

    # the salary row's own team, and the team he signed with that season
    sal_team = {(str(p).lower(), int(s)): t for p, s, t in
                zip(sal["player"], sal["season"], sal["team"])}
    sign_team = {(p, int(s)): t for p, s, t in
                 zip(sd["player_name_norm"], sd["signing_season"], sd["team"])}
    flagged["salary_team"] = [sal_team.get((p, s)) for p, s in
                              zip(flagged["player_name_norm"], flagged["season"])]
    flagged["signed_with"] = [sign_team.get((p, s)) for p, s in
                              zip(flagged["player_name_norm"], flagged["season"])]
    flagged["stretch_class"] = flagged["prior_waiver_date"].notna()
    return flagged.sort_values("gap_m", ascending=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ratio", type=float, default=DEFAULT_RATIO,
                    help="flag rows whose salary is at least this x the AAV")
    args = ap.parse_args()

    df = load_frame()
    flagged = audit(df, args.ratio)
    cols = ["player_name_norm", "season", "salary_m", "aav_m", "ratio",
            "signing_type", "salary_team", "signed_with", "prev_cap_pct",
            "prior_waiver_date"]
    print(f"\nevaluation rows {len(df)}; flagged at >= {args.ratio}x: "
          f"{len(flagged)}")
    print(flagged[cols].to_string(index=False))

    bad = flagged[flagged["stretch_class"]]
    print(f"\nstretch-class rows (a waiver PRECEDING the row corroborates the "
          f"disagreement): {len(bad)}")
    for _, r in bad.iterrows():
        print(f"  {r['player_name_norm']} {int(r['season'])}: "
              f"${r['salary_m']:.3f}M on our books vs ${r['aav_m']:.3f}M signed "
              f"({r['salary_team']} -> {r['signed_with']})")
        print(f"      {r['prior_waiver_date']}  {r['prior_waiver_text']}")
    others = flagged[~flagged["stretch_class"]]
    if len(others):
        print(f"\nbenign block mismatches (no waiver precedes the row; the "
              f"block predates the row's own contract): {len(others)}")
        for _, r in others.iterrows():
            print(f"  {r['player_name_norm']} {int(r['season'])} "
                  f"({r['signing_type']})")

    if len(bad):
        print("\nSee ISSUES #36. These rows are fabricated observations of the "
              "target; do not train on them without sourcing the signed salary.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
