"""Merge the Spotrac salary source onto the Basketball Reference chain.

Produces `data/processed/merged_salaries.csv`, the salary input consumed by
`src/features/build_dataset.py`, with one row per (player, season) in
`salaries.csv` for the configured training seasons. Offline — reads only the
HTML cache and the processed CSVs. `rebuild_training_data.py` runs this before
the feature build and refuses to continue if the resulting key set is stale.

WHY A MERGE LAYER EXISTS AT ALL

`build_spotrac_salaries.py` reports what Spotrac says; it deliberately refuses
to pick between the readings a cell allows. Somebody still has to pick, because
the model needs one number per player-season, and the right pick is different
in four distinguishable situations. This script is that decision, and every row
records which of the four produced it so no choice is silent.

The salary convention this project settled on is the CAP CHARGE (ISSUES #38) —
what the team is charged against the cap, not what the player banks. Every
branch below is an attempt to recover that quantity from what Spotrac prints.

WHY THE PAGES ARE RE-PARSED INSTEAD OF READING spotrac_salaries.csv

That file keeps ONE row per player-season and prefers the contract table
(`Cap Hit`) over the career table when both exist. For 2023-2025 that is most
of the frame — and it throws away exactly the career-table facts this merge
needs: the second dollar amount in a traded season's cell and the team-badge
count that says whether to add it. Re-parsing the same cached pages through the
same `parse_page` costs about two minutes and keeps both table shapes in hand.
Nothing is re-scraped and nothing is written back to that file.

THE MERGE POLICY

  season >= 2026 -> Spotrac `Cap Hit`.
      BBRef's future-season columns are unreliable in a way that is not
      random: they carry stale cap projections, dead money booked as salary,
      and declined options that were never removed. Spotrac's contract table
      is the live cap sheet.

  season <= 2025 -> computed from the CAREER table, four-way:

      vet_min  The career table shows CASH. For a one-year minimum contract of
               a 3+-year veteran the league reimburses the team, so the player
               is paid his service-year rate while the cap is charged the
               2-year rate. Spotrac prints the former; the repo carries the
               latter. Detected by matching the observed amount against the
               published CBA minimum scale (`mechanism_cap._get_vet_min_usd`)
               and converted with `fix_minimum_convention.CAP_CHARGE` — the two
               tables the repo already uses for this, so the merge cannot drift
               from the corrections already sitting in salary_corrections.csv.

      waived   A player waived during the season has his cell contaminated by
               dead money and set-off, and no arithmetic over the printed
               amounts recovers the contract. BBRef is kept for these rows.

      split    Two teams' badges on the row means the season's money was
               divided between them; the season figure is the SUM.

      single   One badge and two amounts means the second figure is an
               adjustment to the first, not a second team's share; the FIRST
               amount is already the season figure.

  Then: season 2019 career amounts are divided by 0.9375. Spotrac's 2019-20
  career `Base` is the COVID-reduced payout (15/16 of the contracted amount —
  Al Horford's contracted $28,000,000 prints as $26,250,000). The model wants
  the contracted amount. The division is applied to EVERY 2019 row, which is
  not quite true of the source: on the evaluation frame, 83 of 128 rows land
  on the BBRef value after dividing and 7 land on it without dividing, so a
  handful of 2019 rows were never reduced and this over-corrects them by
  6.7%. Kevin Durant, who missed the season, is one. No signal on the page
  separates the two populations, so the majority rule is applied and the
  minority is reported by `audit`.

  signing_contract  A season whose cell holds more than one amount (a trade,
               a waiver, or a buyout and re-signing) is priced by the contract
               that STARTS that season, not by the season's cash. A training
               row is one signing, so summing the season across teams prices a
               waived player's old deal as his new one: Dion Waiters 2019 read
               $13.44M for a rest-of-season minimum. The contract comes from
               the Spotrac contract blocks (`spotrac_signing_types.csv`,
               `contract_start == season`). When several start, the largest
               AAV wins, because a rest-of-season minimum is never the
               offseason deal. A one-year contract is priced at its total. A
               longer one takes the combination of the cell's figures closest
               to its AAV, and is left to the branches below when none is
               within `SIGNING_AAV_TOL`. NBA Cup prize money is excluded from
               the cell first. A season with any block above its CBA exception
               limit is left to the branches below as misread. A deal labelled
               Minimum takes the minimum cap-charge rule. A rest-of-season deal keeps its prorated
               amount so the prorated filter drops the row, as it did under
               BBRef. See ISSUES #50 and the migration queue item.

  Anything the above cannot resolve falls back to the BBRef value, branch
  `fallback_bbref`. A missing Spotrac page degrades to stale BBRef, never to a
  missing salary.

BRANCH ORDER is vet_min -> waived -> split -> single, and it matters: a
minimum-salary player who was also waived mid-season shows a partial payout
that will not match the minimum scale, so he falls through to `waived`, while
one who was paid his full minimum before a summer waiver is converted properly.

Usage:
    python scripts/build_merged_salaries.py            # dry run + full audit
    python scripts/build_merged_salaries.py --write
"""

import argparse
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

from config import CAP_BY_SEASON, PROCESSED_DIR, SEASONS
from scripts.build_spotrac_salaries import (
    PLAYER_CACHE,
    URLS,
    parse_page,
    parse_status,
    parse_transactions,
)
from scripts.fix_minimum_convention import CAP_CHARGE
from scripts.scrape_spotrac_players import page_defect
from src.model.mechanism_cap import _get_vet_min_usd

OUT = PROCESSED_DIR / "merged_salaries.csv"
TRAINING = PROCESSED_DIR / "training_data_v2.csv"
SALARIES = PROCESSED_DIR / "salaries.csv"
CONTRACT_STRUCTURE = PROCESSED_DIR / "contract_structure_v2.csv"
TRANSACTIONS = PROCESSED_DIR / "spotrac_transactions.csv"
SIGNING_TYPES = PROCESSED_DIR / "spotrac_signing_types.csv"

# Spotrac's 2019-20 career `Base` is 15/16 of the contracted salary — the
# COVID payout reduction. Verified anchor: Al Horford's PHI 4yr/$97M pays
# $28,000,000 in 2019-20 and Spotrac prints $26,250,000.
COVID_SEASON = 2019
COVID_FACTOR = 0.9375

# Minimum-scale match tolerance. The CBA publishes exact dollars; the only
# slack needed is rounding in the 2023-CBA tier ratios, which are quoted to
# three decimals and so land within about $2K of the published amount.
#
# Keep this tight. At $25K the band starts swallowing coincidences: one team's
# share of Tyus Jones's 2025 season is $2,655,172, which sits $13K from the
# 5-year minimum, and the row was priced as a minimum instead of summing to
# the $7,000,000 the two halves make.
MIN_MATCH_TOL_USD = 5_000

# (player, season) pairs the cached pages report as a TRADE, filled in during
# harvest. A mid-season trade splits the season's money between two teams and
# Spotrac labels the pre-trade half `Retained` — the same label a stretch
# annuity carries. Active-only is right for the annuity and wrong for the
# trade, and the label cannot tell them apart, so the transaction log decides.
# Worth +138 rows against -10 on the status_active branch (70% -> 79%).
PAGE_TRADED: set[tuple[str, int]] = set()

# How far a multi-year contract's year-1 amount may sit from its AAV. Raises
# are at most 8% a year under the CBA, so a five-year deal's first year sits
# at most about 15% below its AAV; the wider band absorbs likely bonuses.
SIGNING_AAV_TOL = 0.35


def _normalize_name(name: str) -> str:
    """Match the project-wide player key used by build_dataset."""
    name = str(name).strip().lower()
    nfkd = unicodedata.normalize("NFKD", name)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


# ---------------------------------------------------------------------------
# Spotrac side
# ---------------------------------------------------------------------------
def harvest_spotrac() -> pd.DataFrame:
    """Every per-season row on every cached player page, both table shapes.

    Deliberately NOT de-duplicated across table kinds: the caller needs the
    career row and the contract row for the same season to exist side by side,
    which is the one thing `spotrac_salaries.csv` cannot give it.
    """
    urls = pd.read_csv(URLS)
    urls["slug"] = (urls["url"].str.split("?").str[0].str.rstrip("/")
                    .str.rsplit("/", n=1).str[-1].str.lower())
    # One slug carries several names (Spotrac files a player under his legal
    # name while the box scores use the short one) — emit a row per name and
    # let the join take the one it needs. See build_spotrac_salaries.
    slug_to_names: dict[str, list[str]] = {}
    for slug, name in zip(urls["slug"], urls["player_name_norm"]):
        slug_to_names.setdefault(slug, []).append(name)

    records = []
    for p in sorted(PLAYER_CACHE.glob("*.html")):
        names = slug_to_names.get(p.stem.lower())
        if not names or page_defect(p):
            # page_defect rejects the NFL players who share a name with an NBA
            # player — see ISSUES #41.
            continue
        status = parse_status(p)
        for season, kind, _text in parse_transactions(p):
            if kind == "traded":
                for name in names:
                    PAGE_TRADED.add((name, int(season)))
        for r in parse_page(p):
            st = status.get(r["season"], {})
            for name in names:
                records.append({**r, **st, "player_name_norm": name})
    return pd.DataFrame(records)


def _career_table(raw: pd.DataFrame) -> pd.DataFrame:
    """One career-table row per player-season, richest first."""
    car = raw[raw["table_kind"] == "career"].copy()
    car = (car.sort_values(["player_name_norm", "season", "base"],
                           ascending=[True, True, False])
              .drop_duplicates(["player_name_norm", "season"], keep="first"))
    status_cols = ["active_total", "retained_total", "other_total",
                   "statuses", "n_active"]
    for col in status_cols:
        if col not in car:
            car[col] = np.nan
    return car[["player_name_norm", "season", "base", "cash_total",
                "n_teams", "split_total", "extra_amounts", *status_cols]]


def _contract_table(raw: pd.DataFrame) -> pd.DataFrame:
    """One contract-table row per player-season, largest cap hit first.

    Overlapping contracts (an extension listed beside the deal it replaces)
    can both print a row for the same season; the larger cap hit is the one
    that is actually on the books.
    """
    con = raw[(raw["table_kind"] == "contract") & raw["cap_hit"].notna()].copy()
    con = (con.sort_values(["player_name_norm", "season", "cap_hit"],
                           ascending=[True, True, False])
              .drop_duplicates(["player_name_norm", "season"], keep="first"))
    return con[["player_name_norm", "season", "cap_hit"]]


# ---------------------------------------------------------------------------
# The four-way branch for season <= 2025
# ---------------------------------------------------------------------------
def _min_scale(season: int) -> list[float]:
    """Every legal minimum-salary amount for a season, in dollars.

    Service tiers 0-10 from the published CBA scale, plus the cap charge
    itself (the 2-year rate) so a row already carrying the charged amount is
    recognised rather than treated as an ordinary contract.
    """
    amounts = [_get_vet_min_usd(season, e) for e in range(11)]
    if season in CAP_CHARGE:
        amounts.append(float(CAP_CHARGE[season]))
    return amounts


def _is_min_amount(amount: float, season: int) -> bool:
    """Does this dollar figure sit on the minimum scale for the season?"""
    if amount is None or not np.isfinite(amount):
        return False
    return any(abs(amount - m) <= MIN_MATCH_TOL_USD for m in _min_scale(season))


def _min_cap_charge(paid: float, season: int, contract_years) -> float:
    """The cap charge for a minimum-salary row.

    The league's reimbursement applies only to ONE-year minimum contracts, so
    a multi-year minimum is charged what it pays. This mirrors the selection
    rule in `fix_minimum_convention.identify_paid_convention_rows`; keeping the
    two in step is why that module's table is imported rather than restated.
    """
    if season not in CAP_CHARGE:
        return paid
    cy = pd.to_numeric(contract_years, errors="coerce")
    if pd.notna(cy) and int(cy) != 1:
        return paid
    return float(min(paid, CAP_CHARGE[season]))


def _minimum_labelled() -> set:
    """(player, season) pairs Spotrac labels as a Minimum signing.

    This is the same evidence `fix_minimum_convention.py` uses to decide which
    BBRef rows needed the cap-charge correction, so the merge and the existing
    corrections agree by construction rather than by luck. It is a signing
    LABEL used to recover a CBA-fixed dollar amount, not a feature — nothing
    here reaches the model's feature list, and the rule in CLAUDE.md about
    Minimum being determined by the contract value is exactly why the amount
    can be read off the CBA scale instead of off the page.
    """
    if not SIGNING_TYPES.exists():
        return set()
    st = pd.read_csv(SIGNING_TYPES)
    st = st[st["signing_type"].str.contains("Minimum", case=False, na=False)]
    return set(zip(st["player_name_norm"], st["season"].astype(int)))


def _waived_in_season(transactions: pd.DataFrame) -> set:
    """(player, season) pairs carrying a waiver DURING the priced season.

    Not the same thing as the model's `is_waived` feature, which asks about a
    waiver in the year BEFORE the signing. Here the question is whether this
    season's money was interrupted, so the season containing the waiver is the
    one that is contaminated. A season runs July Y to June Y+1 — the same
    convention as `parse_signing_dates.signing_season`.
    """
    if transactions.empty:
        return set()
    w = transactions[transactions["event_type"].eq("waived")].copy()
    w["transaction_date"] = pd.to_datetime(w["transaction_date"],
                                           errors="coerce")
    w = w.dropna(subset=["transaction_date"])
    season = np.where(w["transaction_date"].dt.month >= 7,
                      w["transaction_date"].dt.year,
                      w["transaction_date"].dt.year - 1)
    return set(zip(w["player_name_norm"], season.astype(int)))


def _career_candidates(row) -> tuple[float | None, float | None]:
    """(first amount, split sum) from a career row, COVID-adjusted.

    Both are returned because the minimum-scale test has to try each: a
    minimum-salary player traded mid-season has his full-season minimum split
    across two cells, so only the SUM lands on the scale.
    """
    base = row.get("base")
    if base is None or not np.isfinite(base):
        return None, None
    total = row.get("split_total")
    total = float(total) if total is not None and np.isfinite(total) else None
    if int(row["season"]) == COVID_SEASON:
        base = base / COVID_FACTOR
        total = total / COVID_FACTOR if total is not None else None
    return float(base), total


def _all_amounts(row, season: int) -> list[float]:
    """Every dollar figure printed in the career cell, COVID-adjusted.

    Used only by the dead-money rescue below, never by the split/first choice:
    reading the individual halves of an ordinary traded season is how a share
    that happens to land near a minimum tier gets mistaken for a contract.
    """
    out = []
    base = row.get("base")
    if base is not None and np.isfinite(base):
        out.append(float(base))
    for part in str(row.get("extra_amounts") or "").split(";"):
        if part.strip().isdigit():
            out.append(float(part))
    if season == COVID_SEASON:
        out = [a / COVID_FACTOR for a in out]
    return out


def _status_base_candidate(
    row,
) -> tuple[float | None, str | None, float | None]:
    """Select Active Base components when Spotrac labels every amount.

    Spotrac's status table labels each amount Active, Retained, Buyout, etc.
    The labels are usable only when their count matches the Base cell's amount
    count. Active components are this player's live signed salary; a row with
    no Active component is dead money only and must not enter training.

    The third return value is the Active/Reserve-only amount. It is kept apart
    from the selected salary because a traded row can add a Retained component,
    while the Active component itself still identifies a paid veteran minimum.
    """
    statuses = row.get("statuses")
    if statuses is None or pd.isna(statuses):
        return None, None, None
    labels = [s.strip().lower() for s in str(statuses).split(";") if s.strip()]
    amounts = _all_amounts(row, int(row["season"]))
    if not labels or len(labels) != len(amounts):
        return None, None, None

    # Pair first, THEN filter. Dropping a label without dropping its amount
    # would misalign every pair after it.
    pairs = list(zip(labels, amounts))

    # In-season tournament prize money is not salary under any reading, so it
    # is removed before the live/dead decision rather than counted on either
    # side. Leaving it in also made rows look impure that are not: ten of the
    # fourteen mislabelled `retained_only` rows were plain dead money wearing
    # an `NBA Cup` bonus beside it (Terry Rozier 2025, Robin Lopez 2023).
    pairs = [(lab, amt) for lab, amt in pairs if not lab.startswith("nba cup")]
    if not pairs:
        return None, None, None

    live = [amt for lab, amt in pairs
            if lab.startswith("active") or lab.startswith("reserve")]
    live_value = float(sum(live)) if live else None

    # `Reserve` is a ROSTER designation on a live contract, not dead money.
    # Treating "not Active" as dead swept four genuine salaries into the
    # dropped set, two of them inside the evaluation frame: Spencer Dinwiddie
    # 2019 ($10.6M, 64 games for Brooklyn) and DeAndre Jordan 2019 ($9.9M).
    # A traded season's money is genuinely split across two teams, so the
    # `Retained` half is this player's salary too and the season total is the
    # sum. Only when there was no trade does `Retained` mean somebody else's
    # dead money.
    if (str(row["player_name_norm"]), int(row["season"])) in PAGE_TRADED:
        # `Buyout` is still excluded. A trade and a buyout are not exclusive —
        # a player can be dealt in July and bought out in September, and the
        # buyout money belongs to the team that ate it, not to him. Summing it
        # with the trade halves priced Danilo Gallinari's 2022 season at
        # $19,479,000 against $6,479,000, the largest single error in the
        # frame. `Retained` stays IN: on a traded season that is the pre-trade
        # half of his own salary.
        traded_pairs = [amt for lab, amt in pairs
                        if not lab.startswith("buyout")]
        if traded_pairs:
            return float(sum(traded_pairs)), "status_active", live_value

    if live:
        return live_value, "status_active", live_value
    return None, "retained_only", None


def _signing_starts() -> dict:
    """(player, season) -> contracts whose first season is that season.

    Each contract is a dict with `signing_type`, `contract_years`,
    `total_value` and `aav`, read from the Spotrac contract blocks.
    """
    if not SIGNING_TYPES.exists():
        return {}
    st = pd.read_csv(SIGNING_TYPES)
    if "contract_start" not in st.columns:
        return {}
    st = st[st["season"] == st["contract_start"]]
    starts: dict = {}
    for r in st.itertuples():
        starts.setdefault((r.player_name_norm, int(r.season)), []).append({
            "signing_type": r.signing_type,
            "contract_years": r.contract_years,
            "total_value": r.total_value,
            "aav": r.aav,
        })
    return starts


# NBA Cup prize money per player, by season and finishing tier. The 2023
# amounts are the league's announced $500K / $200K / $100K / $50K. Later
# seasons scale with the cap; the dump shows 51,497 and 102,994 for 2024 and
# 53,093, 106,187 and 530,933 for 2025, and the remaining tiers follow the
# same ratio.
CUP_PRIZES = {
    2023: (500_000, 200_000, 100_000, 50_000),
    2024: (514_971, 205_988, 102_994, 51_497),
    2025: (530_933, 212_373, 106_187, 53_093),
}


def _is_cup_prize(amount: float, season: int) -> bool:
    """Is this figure exactly one of the season's NBA Cup prize tiers?"""
    return any(abs(amount - p) <= 2 for p in CUP_PRIZES.get(season, ()))


def _salary_amounts(row, season: int) -> list[float]:
    """The cell's dollar figures without NBA Cup prize money.

    Cup money is never salary (see `_status_base_candidate`). The status label
    identifies it when the label count matches the amount count. Single-team
    cells often carry no labels, so a figure after the first that equals a
    prize tier to the dollar is dropped too. The first figure is the contract's
    own and is never dropped.
    """
    amounts = _all_amounts(row, season)
    statuses = row.get("statuses")
    labels = ([s.strip().lower() for s in str(statuses).split(";") if s.strip()]
              if statuses is not None and not pd.isna(statuses) else [])
    if labels and len(labels) == len(amounts):
        amounts = [a for lab, a in zip(labels, amounts)
                   if not lab.startswith("nba cup")]
        return amounts
    return amounts[:1] + [a for a in amounts[1:]
                          if not _is_cup_prize(a, season)]


def _has_multiple_amounts(row) -> bool:
    """Does the season cell carry more than one contract's money?

    NBA Cup prize money beside a single salary does not count: that row is
    one contract, and the season-cash branches already price it.
    """
    n_teams = pd.to_numeric(row.get("n_teams"), errors="coerce")
    if pd.notna(n_teams) and n_teams > 1:
        return True
    return len(_salary_amounts(row, int(row["season"]))) > 1


# Spotrac signing labels that name a CBA exception, mapped to the
# `mle_exception_amounts.csv` key that caps the exception's first-year salary.
_EXCEPTION_KEYS = {
    "Bi-Annual": "bae",
    "Non-Taxpayer MLE": "non_taxpayer_mle",
    "Taxpayer MLE": "taxpayer_mle",
    "room-mid-level-exception": "room_mle",
}
_EXCEPTION_CAPS: dict | None = None


def _exception_cap(signing_type, season: int) -> float | None:
    """First-year salary cap of a CBA exception, or None when not one."""
    global _EXCEPTION_CAPS
    key = _EXCEPTION_KEYS.get(str(signing_type))
    if key is None:
        return None
    if _EXCEPTION_CAPS is None:
        from config import RAW_DIR
        path = RAW_DIR / "raw_external" / "mle_exception_amounts.csv"
        _EXCEPTION_CAPS = {}
        if path.exists():
            ex = pd.read_csv(path)
            _EXCEPTION_CAPS = {(str(r.exception_type), int(r.season)):
                               float(r.amount_usd) for r in ex.itertuples()}
    return _EXCEPTION_CAPS.get((key, season))


def _subset_sums(amounts: list[float]) -> list[float]:
    """Every non-empty subset sum of a cell's figures (cells hold <= 6)."""
    from itertools import combinations
    return [sum(c) for n in range(1, len(amounts) + 1)
            for c in combinations(amounts, n)]


def _signing_year1_amount(row, season: int, contracts: list
                          ) -> tuple[float, float, str] | None:
    """Year-1 amount, length and signing type of the contract that starts
    this season.

    Returns None when no amount can be tied to the contract; the caller then
    falls through to the season-cash branches.
    """
    usable = [c for c in contracts
              if pd.notna(c.get("aav")) and float(c["aav"]) > 0]
    if not usable:
        return None
    amounts = _salary_amounts(row, season)
    sums = _subset_sums(amounts) if amounts else []

    def one_year_total(c):
        years = pd.to_numeric(c.get("contract_years"), errors="coerce")
        total = pd.to_numeric(c.get("total_value"), errors="coerce")
        if pd.notna(years) and int(years) == 1 and pd.notna(total):
            return float(total)
        return None

    # An exception cannot pay more than its first-year limit. A block above it
    # was misread, and a season with one misread block cannot be trusted to
    # say which deal is the signing: Markieff Morris's 2019 Bi-Annual prints
    # "1 yr / $6.56M" against a $3,623,000 limit, so the $6.56M is a two-year
    # total and his offseason deal is not recoverable from the blocks.
    for k in usable:
        total = one_year_total(k)
        limit = _exception_cap(k.get("signing_type"), season)
        if (total is not None and limit is not None
                and total > limit + MIN_MATCH_TOL_USD):
            return None

    # The largest AAV is the offseason deal: a rest-of-season minimum signed
    # after a buyout never outranks it. Patrick Beverley's $13,000,000 year is
    # paid in two teams' halves beside a $801,614 rest-of-season minimum.
    c = max(usable, key=lambda c: float(c["aav"]))
    years = pd.to_numeric(c.get("contract_years"), errors="coerce")
    total = one_year_total(c)
    if total is not None:
        picked = total
    else:
        # A longer deal's year 1 is the combination of the cell's figures
        # nearest its AAV. The whole-cell sum also carries any rest-of-season
        # deal signed after a buyout: Shake Milton's 2023 NT-MLE year is
        # $1,925,287 + $3,074,713 = $5,000,000, not the $5,552,938 cell total.
        if not sums:
            return None
        aav = float(c["aav"])
        picked = min(sums, key=lambda a: abs(a - aav))
        if abs(picked - aav) > SIGNING_AAV_TOL * aav:
            return None
    # The same limit applies to a multi-year deal's year-1 amount.
    cap = _exception_cap(c.get("signing_type"), season)
    if cap is not None and picked > cap + MIN_MATCH_TOL_USD:
        return None
    return (float(picked), (float(years) if pd.notna(years) else np.nan),
            str(c.get("signing_type")))


def resolve_row(row, waived_keys: set, min_labels: set,
                starts: dict | None = None) -> tuple[float | None, str]:
    """The merged salary for one player-season, and the branch that made it.

    `starts` maps (player, season) to the contracts that begin that season
    (`_signing_starts`). When it is omitted, the signing_contract branch is off.

    Returns (None, branch) when Spotrac cannot answer and the caller must fall
    back to BBRef.
    """
    season = int(row["season"])
    key = (row["player_name_norm"], season)

    # Cap hit wins wherever it exists, in EVERY season — not just 2026+.
    #
    # The season split was the wrong axis. Spotrac's contract tables cover
    # every year of a contract it still lists, so cap hit reaches back as far
    # as the oldest contract on the page: it is available for 96% of 2026
    # training rows but also 63% of 2025 and 35% of 2024, thinning to ~1% by
    # 2021. Gating on the season threw away the 2024-25 half of that for no
    # reason.
    #
    # And cap hit is the better answer where both exist. On the 163 frame rows
    # carrying a career row AND a contract row for a season <= 2025, cap hit
    # reconciles with BBRef 149 times against the career-derived value's 147,
    # and it is right where the career value is wrong on 8 rows versus 6 the
    # other way. That is a thin margin on its own, but it is not the argument:
    # `Base` and `Cap Hit` are different quantities and the TARGET is defined
    # in cap-hit terms (ISSUES #38). Dejounte Murray 2024 is the clean case —
    # career Base $24,799,600, contract Cap Hit $28,817,135, BBRef $29,517,135.
    #
    # It also lands ahead of the vet_min branch deliberately: a minimum
    # contract's cap hit is ALREADY the charge, so reading it beats inferring
    # the charge from the paid amount.
    cap_hit = row.get("cap_hit")
    if cap_hit is not None and np.isfinite(cap_hit):
        return float(cap_hit), "spotrac_cap_hit"

    # A multi-amount season is priced by the contract that starts in it. This
    # runs before the dead-money and trade-sum readings below, because those
    # answer "what was he paid this season", and a training row asks "what did
    # this signing cost".
    signing = (starts or {}).get(key)
    if signing and season < 2026 and _has_multiple_amounts(row):
        picked = _signing_year1_amount(row, season, signing)
        if picked is not None:
            amount, years, signing_type = picked
            # A deal Spotrac labels Minimum is charged by the minimum rule even
            # when its amount misses the scale by more than the match band
            # (Isaiah Thomas 2019: $2,320,044 on a one-year minimum).
            if ("minimum" in signing_type.lower()
                    or _is_min_amount(amount, season)):
                return _min_cap_charge(amount, season, years), "vet_min"
            return amount, "signing_contract"

    status_value, status_branch, status_live_value = _status_base_candidate(row)
    if status_branch == "retained_only":
        return np.nan, status_branch
    if status_value is not None:
        # A traded row sums Active + Retained, but the Active component can be
        # a paid veteran minimum by itself. `_status_base_candidate` returns
        # that component after the 2019 COVID adjustment so the minimum-scale
        # test runs before accepting the larger trade sum.
        minimum_value = None
        if key in min_labels and _is_min_amount(status_live_value, season):
            minimum_value = status_live_value
        elif _is_min_amount(status_value, season):
            # Preserve the existing whole-row minimum path. Some traded rows'
            # components add to the full-season minimum even when neither
            # component matches the scale by itself.
            minimum_value = status_value
        if minimum_value is not None:
            return (_min_cap_charge(minimum_value, season,
                                    row.get("contract_years")),
                    "vet_min")
        return status_value, status_branch
    if season >= 2026:
        return None, "fallback_bbref"

    first, split = _career_candidates(row)
    if first is None:
        return None, "fallback_bbref"

    n_teams = pd.to_numeric(row.get("n_teams"), errors="coerce")
    is_split = pd.notna(n_teams) and n_teams > 1 and split is not None
    preferred = split if is_split else first

    # vet_min first: a minimum row must never go through the split/first
    # logic, because both readings give the PAID amount and the repo carries
    # the CHARGED one.
    #
    # ONLY the preferred candidate is tested. Probing the first amount of a
    # split row as well looks harmless and is not: one half of a traded
    # season's money lands inside the minimum band by coincidence often enough
    # to matter — Isaiah Jackson's 2025 LAC share of $2,882,759 sits $7,859
    # from the 6-year minimum — and the row was then priced as a minimum
    # instead of summed, a $4.7M error on a player earning $7.6M. Four of the
    # twenty largest changes in the frame came from that one extra probe.
    if preferred is not None and _is_min_amount(preferred, season):
        return (_min_cap_charge(preferred, season, row.get("contract_years")),
                "vet_min")

    # Dead money outlives the waiver. A player waived in season Y-1 whose
    # remaining salary was not stretched still prints beside his new team's
    # figure in season Y, and the team-badge count reads it as a trade split:
    # Blake Griffin's 2021 cell is Detroit's $30.7M of dead money next to
    # Brooklyn's minimum, and summing them prices a minimum-salary veteran at
    # $32.4M. So a prior-season waiver disqualifies a MULTI-amount row, while
    # a single-amount row from the same player is left alone — one figure
    # cannot be a sum of two contracts.
    prior = (row["player_name_norm"], season - 1)
    extra = row.get("extra_amounts")
    # NaN, not "", is what a left merge leaves behind, and str(nan) is the
    # truthy "nan" — test for missingness explicitly.
    has_extra = pd.notna(extra) and bool(str(extra).strip())
    if key in waived_keys or (prior in waived_keys and has_extra):
        # One rescue before giving up. A minimum contract's amount is fixed by
        # the CBA, so if Spotrac labels this signing a Minimum AND one of the
        # figures in the contaminated cell IS a full-season minimum, that
        # figure is the contract and the rest is somebody else's dead money.
        # Blake Griffin's 2021 cell is Detroit's $30.7M beside Brooklyn's
        # $1,669,178; the second number is the whole answer.
        #
        # Both conditions are required. The label alone spans every season of
        # a multi-year deal and mislabels besides — taking it at face value
        # repriced Luguentz Dort's 2022 season from $15.3M to the minimum —
        # and a scale-shaped figure alone is often just one team's share of a
        # traded season.
        if key in min_labels:
            for cand in _all_amounts(row, season):
                if _is_min_amount(cand, season):
                    return (_min_cap_charge(cand, season,
                                            row.get("contract_years")),
                            "vet_min")
        return None, "waived"

    if is_split:
        return float(split), "split"
    return float(first), "single"


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def build() -> pd.DataFrame:
    """The merged salary table, one row per configured base-salary key."""
    base = pd.read_csv(SALARIES, usecols=["player", "season", "salary"])
    base = base[base["season"].isin(SEASONS)].copy()
    base["player_name_norm"] = base["player"].map(_normalize_name)
    if base.duplicated(["player_name_norm", "season"]).any():
        dup = base[base.duplicated(["player_name_norm", "season"], keep=False)]
        raise RuntimeError(
            "salaries.csv has duplicate normalized player-season keys:\n"
            + dup.head(20).to_string(index=False)
        )

    td = base[["player_name_norm", "season", "salary"]]
    if CONTRACT_STRUCTURE.exists():
        cs = pd.read_csv(
            CONTRACT_STRUCTURE,
            usecols=["player_name_norm", "season", "contract_years"],
        ).drop_duplicates(["player_name_norm", "season"])
        td = td.merge(cs, on=["player_name_norm", "season"], how="left",
                      validate="one_to_one")
    else:
        td = td.assign(contract_years=np.nan)
    # The BBRef fallback must be the value the pipeline actually uses, which
    # is the corrected one: salary_override repairs stretched dead money
    # (ISSUES #36) and min_cap_charge already converts some minimum rows to
    # the charge convention (ISSUES #38). Comparing against the raw CSV column
    # would invent differences the model never sees.
    from src.model.train import _load_salary_corrections
    corr = _load_salary_corrections()
    fix = corr[corr["kind"].isin(["salary_override", "min_cap_charge"])]
    fix_map = {(str(r.player_name_norm), int(r.season)): float(r.value_usd)
               for r in fix.itertuples()}
    salary_override_map = {
        (str(r.player_name_norm), int(r.season)): float(r.value_usd)
        for r in corr[corr["kind"].eq("salary_override")].itertuples()
    }
    keys = list(zip(td["player_name_norm"], td["season"].astype(int)))
    patched = pd.Series([fix_map.get(k) for k in keys], index=td.index)
    td["bbref_salary"] = patched.fillna(td["salary"])

    print("harvesting cached Spotrac pages (offline, ~2 min) ...")
    raw = harvest_spotrac()
    print(f"  raw season rows parsed : {len(raw)}")
    career, contract = _career_table(raw), _contract_table(raw)
    print(f"  career-table seasons   : {len(career)}")
    print(f"  contract-table seasons : {len(contract)}")

    df = (td.merge(career, on=["player_name_norm", "season"], how="left")
            .merge(contract, on=["player_name_norm", "season"], how="left"))

    tx = (pd.read_csv(TRANSACTIONS) if TRANSACTIONS.exists()
          else pd.DataFrame(columns=["player_name_norm", "transaction_date",
                                     "event_type"]))
    waived_keys = _waived_in_season(tx)
    min_labels = _minimum_labelled()
    starts = _signing_starts()

    salaries, branches, season_cash = [], [], []
    for row in df.to_dict("records"):
        val, branch = resolve_row(row, waived_keys, min_labels, starts)
        # The pre-signing_contract reading, kept for the audit only.
        season_cash.append(resolve_row(row, waived_keys, min_labels)[0])
        key = (str(row["player_name_norm"]), int(row["season"]))
        if key in salary_override_map:
            # A sourced correction is authoritative over every computed
            # branch. This covers rows where Spotrac exposes the components
            # but its Active/Retained labels cannot distinguish traded salary
            # from dead money, plus the documented 2019 rows whose career
            # Base was not reduced by the otherwise league-wide COVID factor.
            val = salary_override_map[key]
            branch = "salary_override"
            source = "curated"
        elif branch == "retained_only":
            source = "spotrac"
        elif val is None:
            val, branch = row["bbref_salary"], (
                "waived" if branch == "waived" else "fallback_bbref")
            source = "bbref"
        else:
            source = "spotrac"
        salaries.append(val)
        branches.append((branch, source))

    df["salary_merged"] = salaries
    df["salary_season_cash"] = season_cash
    df["has_signing_start"] = [
        (str(k), int(s)) in starts
        for k, s in zip(df["player_name_norm"], df["season"])]
    df["branch"] = [b for b, _ in branches]
    df["source"] = [s for _, s in branches]
    df["cap"] = df["season"].map(CAP_BY_SEASON)
    return df


def output_table(df: pd.DataFrame) -> pd.DataFrame:
    """Select and name the tracked salary-migration artifact columns."""
    return (df[["player_name_norm", "season", "salary_merged", "source",
                "branch", "bbref_salary", "n_teams", "extra_amounts",
                "contract_years"]]
            .rename(columns={"salary_merged": "salary"})
            .sort_values(["player_name_norm", "season"]))


def _eval_frame() -> pd.DataFrame:
    """The 880-row evaluation frame — the rows a change to salary can move.

    Scoring runs over every row, but only these carry a negotiated year-1
    price, so they are where a salary migration must be audited.
    """
    from src.model import train as T
    df = T.load_training_data()
    for f in (T._filter_year1, T._filter_rookie_scale, T._filter_prorated):
        df = f(df)
    df = T._compute_max_eligible(df)
    df = T._filter_mislabeled_year1(df)
    df = T._filter_continuations(df)
    df = T._filter_rookie_contracts(df)
    return df


def audit(df: pd.DataFrame) -> None:
    """Print the reconciliation against the current BBRef values."""
    frame = _eval_frame()[["player_name_norm", "season", "salary"]]
    frame = frame.rename(columns={"salary": "bbref"})
    m = frame.merge(df[["player_name_norm", "season", "salary_merged",
                        "salary_season_cash", "has_signing_start",
                        "branch", "source", "n_teams",
                        "extra_amounts", "cap", "cap_hit"]],
                    on=["player_name_norm", "season"], how="left")
    print(f"\n{'='*72}\nEVALUATION FRAME: {len(m)} rows\n{'='*72}")

    m["delta"] = m["salary_merged"] - m["bbref"]
    m["exact"] = m["delta"].abs() <= 1.0
    m["delta_pct"] = m["delta"] / m["cap"]
    m["multi"] = (m["extra_amounts"].notna()
                  & (m["extra_amounts"].astype(str).str.len() > 0))

    print("\nrows per branch, and agreement with BBRef:")
    print(f"{'branch':18s} {'rows':>6s} {'exact':>7s} {'rate':>7s} "
          f"{'|Δ| median':>12s} {'|Δ| max':>14s}")
    for br, g in m.groupby("branch"):
        nz = g.loc[~g["exact"], "delta"].abs()
        print(f"{br:18s} {len(g):6d} {int(g['exact'].sum()):7d} "
              f"{g['exact'].mean():6.1%} "
              f"{(nz.median() if len(nz) else 0):12,.0f} "
              f"{(nz.max() if len(nz) else 0):14,.0f}")

    sub = m[m["multi"]]
    print(f"\nsame table, restricted to the {len(sub)} multi-amount rows "
          f"(the prototype's calibration set):")
    for br, g in sub.groupby("branch"):
        print(f"{br:18s} {len(g):6d} {int(g['exact'].sum()):7d} "
              f"{g['exact'].mean():6.1%}")

    changed = m[~m["exact"] & m["salary_merged"].notna()]
    print(f"\nrows the merge CHANGES: {len(changed)} of {len(m)} "
          f"({len(changed)/len(m):.1%})")
    print("\nchange in dollars:")
    print(changed["delta"].describe(
        percentiles=[.05, .25, .5, .75, .95]).to_string())
    print("\nchange in cap_pct (percentage points):")
    print((changed["delta_pct"] * 100).describe(
        percentiles=[.05, .25, .5, .75, .95]).to_string())

    print("\nlargest 20 changes (a human must eyeball these):")
    big = changed.reindex(changed["delta"].abs().sort_values(
        ascending=False).index).head(20)
    print(f"{'player':26s} {'szn':>5s} {'old':>14s} {'new':>14s} "
          f"{'delta':>14s} {'Δpp':>7s} {'branch':>16s}")
    for r in big.itertuples():
        print(f"{r.player_name_norm:26s} {r.season:5d} {r.bbref:14,.0f} "
              f"{r.salary_merged:14,.0f} {r.delta:14,.0f} "
              f"{r.delta_pct*100:7.2f} {r.branch:>16s}")

    # How well the flat 2019 COVID rule holds. Reported, not acted on: the
    # page carries nothing that says which rows were reduced.
    s19 = m[(m["season"] == 2019) & m["salary_merged"].notna()]
    if len(s19):
        undiv = (s19["salary_merged"] * COVID_FACTOR - s19["bbref"]).abs() <= 1
        print(f"\n2019 COVID rule on {len(s19)} frame rows: "
              f"{int(s19['exact'].sum())} agree with BBRef after dividing by "
              f"{COVID_FACTOR}, {int(undiv.sum())} would have agreed WITHOUT "
              f"dividing (over-corrected by 6.7%)")

    # The stated policy sends every season <= 2025 through the career table.
    # Where Spotrac ALSO publishes a contract row for those seasons, that row
    # carries `Cap Hit` — the convention the target is defined in — while the
    # career table carries `Base`. The two are different quantities, so this
    # block reports which one the current BBRef column agrees with. It is a
    # measurement, not a branch; nothing above reads it.
    both = m[(m["season"] <= 2025) & m["cap_hit"].notna()
             & m["salary_merged"].notna()]
    if len(both):
        cap_exact = (both["cap_hit"] - both["bbref"]).abs() <= 1.0
        car_exact = both["exact"]
        print(f"\ncareer `Base` vs contract `Cap Hit`, season <= 2025 "
              f"({len(both)} frame rows have both):")
        print(f"  career-derived agrees with BBRef : {int(car_exact.sum()):4d}"
              f" ({car_exact.mean():.1%})")
        print(f"  contract Cap Hit agrees w/ BBRef : {int(cap_exact.sum()):4d}"
              f" ({cap_exact.mean():.1%})")
        print(f"  cap hit right where career wrong : "
              f"{int((cap_exact & ~car_exact).sum())}")
        print(f"  career right where cap hit wrong : "
              f"{int((car_exact & ~cap_exact).sum())}")

    # Every frame row the signing_contract branch moves, beside the season-cash
    # reading it replaced. A human must check these against the contract block.
    moved = m[(m["salary_merged"] - m["salary_season_cash"]).abs() > 1.0]
    print(f"\nsigning_contract branch moves {len(moved)} frame rows "
          f"(season cash -> signing year 1):")
    print(f"{'player':26s} {'szn':>5s} {'bbref':>14s} {'season cash':>14s} "
          f"{'signing':>14s} {'branch':>16s}")
    for r in moved.sort_values(["season", "player_name_norm"]).itertuples():
        cash = r.salary_season_cash
        cash_s = f"{cash:14,.0f}" if pd.notna(cash) else f"{'bbref':>14s}"
        print(f"{r.player_name_norm:26s} {r.season:5d} {r.bbref:14,.0f} "
              f"{cash_s} {r.salary_merged:14,.0f} {r.branch:>16s}")
    multi = m[(m["n_teams"] > 1) & (m["season"] < 2026)]
    no_start = multi[~multi["has_signing_start"].fillna(False).astype(bool)]
    print(f"\nmulti-team frame rows with no Spotrac contract starting that "
          f"season: {len(no_start)} of {len(multi)} (reported, not acted on; "
          "such a row may not be a signing)")
    print(no_start[["player_name_norm", "season", "branch"]]
          .to_string(index=False, max_rows=40))

    unres = m[m["branch"].isin(["fallback_bbref", "waived"])]
    print(f"\nunreconciled (no Spotrac answer, BBRef kept): {len(unres)} "
          f"({len(unres)/len(m):.1%})  "
          f"[fallback_bbref {int((m['branch']=='fallback_bbref').sum())}, "
          f"waived {int((m['branch']=='waived').sum())}]")
    print(unres[["player_name_norm", "season", "branch"]]
          .to_string(index=False, max_rows=40))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    df = build()
    print("\nbranch counts over the full table:")
    print(df["branch"].value_counts().to_string())
    audit(df)

    out = output_table(df)
    if not args.write:
        print("\ndry run — nothing written. Re-run with --write.")
        return
    out.to_csv(OUT, index=False)
    print(f"\nwrote {OUT} ({len(out)} rows)")


if __name__ == "__main__":
    main()
