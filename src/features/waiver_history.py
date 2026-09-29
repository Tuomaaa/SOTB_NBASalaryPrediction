"""Previous-waiver feature from dated Spotrac transaction logs."""

import re
from pathlib import Path

import numpy as np
import pandas as pd

from config import CAP_BY_SEASON, PROCESSED_DIR


TRANSACTIONS = PROCESSED_DIR / "spotrac_transactions.csv"
SIGNING_DATES = PROCESSED_DIR / "contract_signing_dates.csv"
WAIVER_LOOKBACK_DAYS = 365
# Spotrac wording for a waiver that leaves the old team paying the player.
# "reduced waived amount to $0" says nothing is owed and does not match.
MONEY_OWED = re.compile(
    r"buyout|bought out|stretch|gave back|giving back|dead cap"
    r"|reducing salary|reducing amount owed", re.IGNORECASE)


def money_owed(waivers: pd.DataFrame) -> float:
    """1.0 when any of these waiver events leaves money owed, else 0.0."""
    return float(waivers["tx_text"].astype(str).str.contains(MONEY_OWED).any())


def classify_transaction(text: str) -> str:
    """Classify the transaction types needed by the waiver feature."""
    low = str(text).strip().lower()
    if low.startswith("waived by") or "bought out" in low or "buyout" in low:
        return "waived"
    if low.startswith("signed"):
        return "signed"
    if low.startswith("claimed"):
        return "claimed"
    if low.startswith("traded"):
        return "traded"
    if low.startswith("released"):
        return "released"
    return "other"


def build_transaction_events(cache_dir: Path) -> pd.DataFrame:
    """Parse every dated transaction from cached Spotrac player pages."""
    from scripts.parse_signing_dates import (
        parse_date,
        read_transactions,
        signing_season,
        slug_to_training_name,
    )

    raw, _ = read_transactions(cache_dir)
    columns = [
        "player_slug", "player_name_norm", "transaction_date",
        "transaction_season", "event_type", "tx_text",
    ]
    if raw.empty:
        return pd.DataFrame(columns=columns)

    names = slug_to_training_name()
    out = raw.rename(columns={"text": "tx_text"}).copy()
    out["player_name_norm"] = out["player_slug"].map(names).fillna(
        out["player_slug"].str.replace("-", " ", regex=False)
    )
    out["transaction_date"] = out["date_raw"].map(parse_date)
    out["transaction_season"] = out["transaction_date"].map(signing_season)
    out["event_type"] = out["tx_text"].map(classify_transaction)
    out = out.drop_duplicates(
        ["player_slug", "transaction_date", "tx_text"]
    ).sort_values(["player_name_norm", "transaction_date", "tx_text"])
    return out[columns].reset_index(drop=True)


# Opening night by season start year. A deal waived before it is a camp cut
# that never priced the season.
SEASON_OPENERS = {
    2015: "2015-10-27", 2016: "2016-10-25", 2017: "2017-10-17",
    2018: "2018-10-16", 2019: "2019-10-22", 2020: "2020-12-22",
    2021: "2021-10-19", 2022: "2022-10-18", 2023: "2023-10-24",
    2024: "2024-10-22", 2025: "2025-10-21",
}


def _season_opener(season: int) -> pd.Timestamp:
    """Opening night of a season, October 20 when the date is not listed."""
    return pd.Timestamp(SEASON_OPENERS.get(season, f"{season}-10-20"))


def _is_preseason_cut(signing_date, next_date, waivers, season: int) -> bool:
    """True when a waiver ends this deal before opening night and before the
    player's next signing."""
    if pd.isna(signing_date) or waivers is None or waivers.empty:
        return False
    end = min(_season_opener(season),
              next_date if pd.notna(next_date) else pd.Timestamp.max)
    hit = waivers[(waivers["transaction_date"] >= signing_date)
                  & (waivers["transaction_date"] <= end)]
    return not hit.empty


def _choose_fallback_signing(
    rows: pd.DataFrame, season: int, salary: float,
    player_tx: pd.DataFrame | None = None,
):
    """Choose a same-season signing when no dated span covers the row.

    Without a priced candidate, take the season's first dated signing that was
    not cut before opening night. A later in-season deal (10-day,
    rest-of-season) follows any in-season waiver, so anchoring the lookback on
    it reads events after the contract that priced the row. A camp deal cut
    before the opener did not price the row, so the waiver that ended it is
    prior information for the deal that did.
    """
    cand = rows[rows["signing_season"] == season].copy()
    if cand.empty:
        return None
    years = pd.to_numeric(cand["contract_years"], errors="coerce")
    total = pd.to_numeric(cand["total_value"], errors="coerce")
    cand["_aav"] = total / years
    priced = cand[cand["_aav"].notna()]
    if not priced.empty:
        return priced.loc[(priced["_aav"] - salary).abs().idxmin()]
    dated = cand.dropna(subset=["signing_date"]).sort_values("signing_date")
    if dated.empty:
        return None
    waivers = None
    if player_tx is not None:
        waivers = player_tx[player_tx["event_type"].eq("waived")
                            & player_tx["transaction_date"].notna()]
    nxt = dated["signing_date"].shift(-1)
    for (_, row), next_date in zip(dated.iterrows(), nxt):
        if not _is_preseason_cut(row["signing_date"], next_date, waivers,
                                 season):
            return row
    return dated.iloc[0]


def _resolve_waiver_no_signing(
    player_tx: pd.DataFrame,
    season: int,
    lookback_days: int = WAIVER_LOOKBACK_DAYS,
) -> tuple[float, pd.Timestamp | None, str | None, float] | None:
    """Try to resolve waiver status when signing date is unknown.

    Uses conservative season-based date windows.  A season-X signing happens
    roughly July–October of year X (`signing_season`), so the 365-day lookback
    spans roughly July of year X-1 to October of year X (ISSUES #58: the
    windows previously sat one year early).

    Three outcomes:
    - No waiver events at all → (0.0, None, None, 0.0)   (definitively not waived)
    - All waivers outside the widest possible window → (0.0, None, None, 0.0)
    - A waiver clearly inside the tightest window    → (1.0, date, text, owed)
    - Ambiguous (waiver between tight and wide)      → None  (leave unknown)

    `owed` is `money_owed` over every waiver in the tight window.

    The wide window brackets the earliest-possible lookback start (signing
    on July 1, lookback starts July 1 of the prior year) through the latest
    plausible signing date (Oct 25).  The tight window is the intersection of
    every possible 365-day lookback: Oct 25 of year X-1 through July 1 of
    year X.  A waiver in the tight window is inside any possible lookback;
    one outside the wide window is outside every possible lookback; one in
    between depends on the exact signing date we don't have.
    """
    waivers = player_tx[
        player_tx["event_type"].eq("waived")
        & player_tx["transaction_date"].notna()
    ]
    if waivers.empty:
        return (0.0, None, None, 0.0)

    # Wide window: earliest possible lookback start → latest possible signing
    wide_start = pd.Timestamp(f"{season - 1}-07-01")
    wide_end = pd.Timestamp(f"{season}-10-25")
    in_wide = waivers[
        (waivers["transaction_date"] >= wide_start)
        & (waivers["transaction_date"] <= wide_end)
    ]
    if in_wide.empty:
        return (0.0, None, None, 0.0)

    # Tight window: inside every possible 365-day lookback
    tight_start = pd.Timestamp(f"{season - 1}-10-25")
    tight_end = pd.Timestamp(f"{season}-07-01")
    in_tight = waivers[
        (waivers["transaction_date"] >= tight_start)
        & (waivers["transaction_date"] <= tight_end)
    ]
    if not in_tight.empty:
        hit = in_tight.sort_values("transaction_date").iloc[-1]
        return (1.0, hit["transaction_date"], hit["tx_text"],
                money_owed(in_tight))

    # Waiver is in the wide window but not the tight window — ambiguous.
    return None


def attach_waiver_history(
    df: pd.DataFrame,
    transactions: pd.DataFrame | None = None,
    signing_dates: pd.DataFrame | None = None,
    lookback_days: int = WAIVER_LOOKBACK_DAYS,
) -> pd.DataFrame:
    """Attach whether a waiver occurred in the year before the current signing.

    is_waived is 1/0 only when both the player's transaction page and the
    signing date that prices the row are observed. It stays NaN otherwise.
    is_waived_known makes that source coverage explicit and is kept for the
    required coverage-control evaluation. prior_waiver_owed is 1 when any
    waiver in the lookback left the old team paying the player (a buyout,
    stretch or dead money); it scans every waiver, not only the last one.

    When a player has a transaction page but no signing date can be matched,
    the function attempts conservative resolution: if no waiver events exist
    or all waivers fall clearly outside any possible lookback window for the
    season, the row is marked known.  See _resolve_waiver_no_signing.
    """
    out = df.copy()
    tx = (
        pd.read_csv(TRANSACTIONS, parse_dates=["transaction_date"])
        if transactions is None and TRANSACTIONS.exists()
        else transactions
    )
    sd = (
        pd.read_csv(SIGNING_DATES, parse_dates=["signing_date"])
        if signing_dates is None and SIGNING_DATES.exists()
        else signing_dates
    )
    out["is_waived"] = np.nan
    out["is_waived_known"] = 0.0
    out["prior_waiver_date"] = pd.NaT
    out["prior_waiver_text"] = pd.NA
    out["prior_waiver_owed"] = np.nan
    if tx is None or tx.empty:
        return out

    tx = tx.copy()
    tx["transaction_date"] = pd.to_datetime(
        tx["transaction_date"], errors="coerce"
    )
    # sd may be None or empty — the signing-date lookup simply finds nothing,
    # and conservative resolution (_resolve_waiver_no_signing) fills what it can.
    if sd is not None and not sd.empty:
        sd = sd.copy()
        sd["signing_date"] = pd.to_datetime(sd["signing_date"], errors="coerce")
    else:
        sd = pd.DataFrame(columns=[
            "player_name_norm", "signing_date", "signing_season",
            "contract_years", "total_value", "is_extension",
            "contract_class", "team", "fa_year_matched",
            "match_confidence", "tx_text",
        ])
    tx_by_player = {p: g for p, g in tx.groupby("player_name_norm")}
    sd_by_player = {p: g for p, g in sd.groupby("player_name_norm")}
    page_coverage = set(tx_by_player)

    from scripts.parse_signing_dates import contract_spans, covering_contract

    spans = contract_spans(sd)
    n_resolved = 0
    for i, row in out.iterrows():
        player = str(row["player_name_norm"])
        season = int(row["season"])
        salary = float(
            row["salary"]
            if "salary" in out.columns
            else row["cap_pct"] * CAP_BY_SEASON[season]
        )
        signing = None
        covered = covering_contract(spans, player, season, salary=salary)
        if covered is not None:
            signing = covered["signing_date"]
        if signing is None or pd.isna(signing):
            candidates = sd_by_player.get(player)
            if candidates is not None:
                fallback = _choose_fallback_signing(
                    candidates, season, salary, tx_by_player.get(player)
                )
                if fallback is not None:
                    signing = fallback["signing_date"]

        if player not in page_coverage:
            continue

        if signing is None or pd.isna(signing):
            # Player has a transaction page but no signing date for this
            # season.  Try conservative window-based resolution.
            resolved = _resolve_waiver_no_signing(
                tx_by_player[player], season, lookback_days
            )
            if resolved is not None:
                is_w, w_date, w_text, owed = resolved
                out.at[i, "is_waived_known"] = 1.0
                out.at[i, "is_waived"] = is_w
                out.at[i, "prior_waiver_owed"] = owed
                if w_date is not None:
                    out.at[i, "prior_waiver_date"] = w_date
                    out.at[i, "prior_waiver_text"] = w_text
                n_resolved += 1
            continue

        signing = pd.Timestamp(signing)
        out.at[i, "is_waived_known"] = 1.0
        prior = tx_by_player[player]
        prior = prior[
            prior["event_type"].eq("waived")
            & prior["transaction_date"].notna()
            & (prior["transaction_date"] < signing)
        ]
        prior = prior[
            prior["transaction_date"]
            >= signing - pd.Timedelta(days=lookback_days)
        ]
        out.at[i, "is_waived"] = float(not prior.empty)
        out.at[i, "prior_waiver_owed"] = (money_owed(prior)
                                          if not prior.empty else 0.0)
        if not prior.empty:
            hit = prior.sort_values("transaction_date").iloc[-1]
            out.at[i, "prior_waiver_date"] = hit["transaction_date"]
            out.at[i, "prior_waiver_text"] = hit["tx_text"]
    if n_resolved:
        print(f"  waiver history: resolved {n_resolved} rows via conservative "
              f"window (no signing date)")
    return out

def attach_waiver_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Compute waiver interaction features from existing columns.

    prev_cap_pct_x_waived = prev_cap_pct * is_waived
    mpg_x_waived          = mpg * is_waived

    NaN semantics: if is_waived is NaN (unknown coverage), the product is NaN
    too (not 0). NaN * x = NaN in pandas handles this natively, but both
    columns are coerced to numeric first so a stray string cannot silently
    zero out the product.
    """
    out = df.copy()
    waived = pd.to_numeric(out.get("is_waived"), errors="coerce")
    prev = pd.to_numeric(out.get("prev_cap_pct"), errors="coerce")
    mpg = pd.to_numeric(out.get("mpg"), errors="coerce")
    out["prev_cap_pct_x_waived"] = prev * waived
    out["mpg_x_waived"] = mpg * waived
    return out


def attach_waiver_status_as_of(
    df: pd.DataFrame,
    as_of_date,
    transactions: pd.DataFrame | None = None,
    lookback_days: int = WAIVER_LOOKBACK_DAYS,
) -> pd.DataFrame:
    """Attach current recent-waiver status for an unsigned prediction frame.

    Training anchors the lookback on each historical contract's signing date.
    An unsigned player has no such date yet, so inference anchors the identical
    365-day window on the caller's explicit data-as-of date.
    """
    out = df.copy()
    tx = (
        pd.read_csv(TRANSACTIONS, parse_dates=["transaction_date"])
        if transactions is None and TRANSACTIONS.exists()
        else transactions
    )
    out["is_waived"] = np.nan
    out["is_waived_known"] = 0.0
    out["prior_waiver_date"] = pd.NaT
    out["prior_waiver_text"] = pd.NA
    out["prior_waiver_owed"] = np.nan
    if tx is None or tx.empty:
        return out

    tx = tx.copy()
    tx["transaction_date"] = pd.to_datetime(
        tx["transaction_date"], errors="coerce"
    )
    cutoff = pd.Timestamp(as_of_date)
    start = cutoff - pd.Timedelta(days=lookback_days)
    for i, player in out["player_name_norm"].astype(str).items():
        rows = tx[tx["player_name_norm"].eq(player)]
        if rows.empty:
            continue
        out.at[i, "is_waived_known"] = 1.0
        prior = rows[
            rows["event_type"].eq("waived")
            & rows["transaction_date"].notna()
            & (rows["transaction_date"] <= cutoff)
            & (rows["transaction_date"] >= start)
        ]
        out.at[i, "is_waived"] = float(not prior.empty)
        out.at[i, "prior_waiver_owed"] = (money_owed(prior)
                                          if not prior.empty else 0.0)
        if not prior.empty:
            hit = prior.sort_values("transaction_date").iloc[-1]
            out.at[i, "prior_waiver_date"] = hit["transaction_date"]
            out.at[i, "prior_waiver_text"] = hit["tx_text"]
    return out

def transaction_data_as_of(
    transactions: pd.DataFrame | None = None,
) -> pd.Timestamp:
    """Latest dated source event, used as the reproducible inference cutoff."""
    tx = pd.read_csv(TRANSACTIONS) if transactions is None else transactions
    dates = pd.to_datetime(tx["transaction_date"], errors="coerce").dropna()
    if dates.empty:
        raise ValueError("Spotrac transaction source has no dated events")
    return pd.Timestamp(dates.max()).normalize()
