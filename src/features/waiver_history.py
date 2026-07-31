"""Previous-waiver feature from dated Spotrac transaction logs."""

from pathlib import Path

import numpy as np
import pandas as pd

from config import CAP_BY_SEASON, PROCESSED_DIR


TRANSACTIONS = PROCESSED_DIR / "spotrac_transactions.csv"
SIGNING_DATES = PROCESSED_DIR / "contract_signing_dates.csv"
WAIVER_LOOKBACK_DAYS = 365


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


def _choose_fallback_signing(
    rows: pd.DataFrame, season: int, salary: float
):
    """Choose a same-season signing when no dated span covers the row."""
    cand = rows[rows["signing_season"] == season].copy()
    if cand.empty:
        return None
    years = pd.to_numeric(cand["contract_years"], errors="coerce")
    total = pd.to_numeric(cand["total_value"], errors="coerce")
    cand["_aav"] = total / years
    priced = cand[cand["_aav"].notna()]
    if not priced.empty:
        return priced.loc[(priced["_aav"] - salary).abs().idxmin()]
    dated = cand.dropna(subset=["signing_date"])
    return None if dated.empty else dated.sort_values("signing_date").iloc[-1]


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
    required coverage-control evaluation.
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
    if tx is None or sd is None or tx.empty or sd.empty:
        return out

    tx = tx.copy()
    tx["transaction_date"] = pd.to_datetime(
        tx["transaction_date"], errors="coerce"
    )
    sd = sd.copy()
    sd["signing_date"] = pd.to_datetime(sd["signing_date"], errors="coerce")
    tx_by_player = {p: g for p, g in tx.groupby("player_name_norm")}
    sd_by_player = {p: g for p, g in sd.groupby("player_name_norm")}
    page_coverage = set(tx_by_player)

    from scripts.parse_signing_dates import contract_spans, covering_contract

    spans = contract_spans(sd)
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
                fallback = _choose_fallback_signing(candidates, season, salary)
                if fallback is not None:
                    signing = fallback["signing_date"]

        if player not in page_coverage or signing is None or pd.isna(signing):
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
        if not prior.empty:
            hit = prior.sort_values("transaction_date").iloc[-1]
            out.at[i, "prior_waiver_date"] = hit["transaction_date"]
            out.at[i, "prior_waiver_text"] = hit["tx_text"]
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
