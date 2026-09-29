"""Estimate how likely a signing is the player's last contract.

A veteran who expects this to be his last contract may trade salary for a
chance to win (LeBron James 2026, Chris Paul 2025, Stephen Curry's 2026
extension). The quantity is the belief at signing, so it is estimated from
completed careers that were already over before the evaluation seasons begin.

The hazard h(age, BPM, BPM trend, minutes, games) = P(season T is the
player's last) is
fitted on Basketball Reference seasons 1997-2016 only, with retirement read
from seasons up to 2018. Nothing from 2019 onward enters the fit, so every
evaluation row (2019-2026) is scored by a model frozen before it.

A row priced after season S describes a signing for S+1 onward, so

    P(last contract within k seasons) = 1 - prod_{j=1..k} (1 - h(age + j, x))

with the player's season-S BPM, trend, minutes and games held fixed.

Writes `data/processed/retirement_hazard.csv`: player_name_norm, season,
p_last_1y, p_last_2y, p_last_3y for every `advanced_stats.csv` player-season.

    python scripts/build_retirement_hazard.py
"""

import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from config import PROCESSED_DIR

HISTORY = PROCESSED_DIR / "advanced_stats_history.csv"
CURRENT = PROCESSED_DIR / "advanced_stats.csv"
OUT = PROCESSED_DIR / "retirement_hazard.csv"

FIT_LAST_SEASON = 2016       # fit rows
LABEL_LAST_SEASON = 2018     # retirement is read from seasons up to here
MIN_FIT_AGE = 26
FEATURES = ["age", "bpm_s", "d_bpm", "mpg", "games"]
# Age raises the hazard; quality, its trend, role and availability lower it.
# d_bpm separates a stable veteran from one in decline at the same level.
MONOTONE = [1, -1, -1, -1, -1]
# BPM is shrunk toward replacement level (-2) with a 500-minute prior, so a
# 60-minute cameo does not read as a star.
BPM_PRIOR, PRIOR_MIN = -2.0, 500.0


def _norm(name: str) -> str:
    """Project-wide player key: lower case, accents stripped."""
    s = str(name).strip().lower()
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def player_seasons(df: pd.DataFrame) -> pd.DataFrame:
    """One row per player-season: the multi-team total row, else the only row."""
    df = df.copy()
    df["is_total"] = df["team"].astype(str).str.match(r"^(TOT|\dTM)$")
    df = (df.sort_values(["player_url", "season", "is_total", "minutes"],
                         ascending=[True, True, False, False])
            .drop_duplicates(["player_url", "season"]))
    df["mpg"] = df["minutes"] / df["games"].clip(lower=1)
    df["bpm_s"] = ((df["bpm"] * df["minutes"] + BPM_PRIOR * PRIOR_MIN)
                   / (df["minutes"] + PRIOR_MIN))
    df = df.dropna(subset=["age", "minutes", "games", "bpm"])
    return add_trend(df)


def add_trend(df: pd.DataFrame) -> pd.DataFrame:
    """d_bpm: change in shrunk BPM from the player's previous season.

    A first season, or a gap of more than one season, has no trend and reads 0.
    """
    df = df.sort_values(["player_url", "season"]).copy()
    prev = df.groupby("player_url")[["season", "bpm_s"]].shift(1)
    consecutive = prev["season"].eq(df["season"] - 1)
    df["d_bpm"] = np.where(consecutive, df["bpm_s"] - prev["bpm_s"], 0.0)
    return df


def fit_hazard(history: pd.DataFrame) -> HistGradientBoostingClassifier:
    """Fit P(this season is the last) on seasons <= FIT_LAST_SEASON."""
    h = history[history["season"] <= LABEL_LAST_SEASON]
    last = h.groupby("player_url")["season"].max()
    fit = h[(h["season"] <= FIT_LAST_SEASON) & (h["age"] >= MIN_FIT_AGE)].copy()
    fit["last"] = fit["player_url"].map(last).eq(fit["season"])
    print(f"hazard fit: {len(fit)} player-seasons age >= {MIN_FIT_AGE}, "
          f"{int(fit['last'].sum())} final seasons, "
          f"{int((fit['age'] >= 38).sum())} at 38+")
    model = HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=300,
        monotonic_cst=MONOTONE, random_state=0)
    return model.fit(fit[FEATURES], fit["last"])


def p_last(model, rows: pd.DataFrame, k: int) -> np.ndarray:
    """P(the contract signed after this season is the last within k seasons)."""
    survive = np.ones(len(rows))
    for j in range(1, k + 1):
        x = rows[FEATURES].copy()
        x["age"] = x["age"] + j
        survive *= 1.0 - model.predict_proba(x)[:, 1]
    return 1.0 - survive


def main() -> None:
    """Fit on history, score every current player-season, write the table."""
    history = player_seasons(pd.read_csv(HISTORY))
    model = fit_hazard(history)
    # Score 2019+ rows with a trend that can reach back into 2018.
    both = pd.concat([pd.read_csv(HISTORY), pd.read_csv(CURRENT)],
                     ignore_index=True)
    cur = player_seasons(both)
    cur = cur[cur["season"] >= 2019]
    out = pd.DataFrame({
        "player_name_norm": cur["player"].map(_norm),
        "season": cur["season"].astype(int),
        "age": cur["age"],
        "bpm_s": cur["bpm_s"].round(3),
    })
    for k in (1, 2, 3):
        out[f"p_last_{k}y"] = p_last(model, cur, k).round(4)
    out = (out.sort_values(["player_name_norm", "season"])
              .drop_duplicates(["player_name_norm", "season"]))
    out.to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(out)} rows)")


if __name__ == "__main__":
    main()
