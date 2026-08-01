"""The `playoff_mpg_diff` feature: playoff minutes minus regular-season minutes.

A coach's playoff rotation is a public, dated judgement of a player made by the
one party with full information, on the games that matter most, and it is made
BEFORE the contract this row prices. A player whose minutes collapse in the
playoffs is priced down by the market the following summer; the regular-season
box score does not see it.

Definition (v8.6x, one column):

    playoff_mpg_diff = po_mpg - mpg,  and 0.0 where the player did not appear

Zero is the right fill rather than NaN: "did not play in the playoffs" is a
statement about the player's team, not a gap in his record, and a zero
difference is exactly the neutral reading the model should take from it. 55% of
the evaluation frame fills this way, and the untouched-rows check in the v8.6x
gate run confirms the model leaves them alone ($3.161M -> $3.162M).

Only the DIFFERENCE ships. `po_games` was tested alongside it and rejected as a
team-success proxy — permuting it within (season, playoff team) retains its
entire edge, which is the same failure that put `win_pct` / `made_playoffs` in
METHODOLOGY's rejected table. `playoff_mpg_diff` survives that same
falsification (the within-team permutation retains 6% of its gain).

Attached at load time (`train.py::load_training_data`) rather than baked into
`training_data_v2.csv`, the same way `attach_waiver_interactions` is: the column
is a pure function of one external table plus the existing `mpg`, so a rebuild
would buy nothing and would move unrelated columns on the way past.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd

from config import PROCESSED_DIR, RAW_DIR

PLAYOFF_MPG_PATH = PROCESSED_DIR / "playoff_mpg.csv"
_ALIAS_PATH = RAW_DIR / "raw_external" / "player_name_aliases.csv"


def _alias_map() -> dict[str, str]:
    """old -> current normalized names, so a renamed player still joins."""
    if not _ALIAS_PATH.exists():
        return {}
    al = pd.read_csv(_ALIAS_PATH)
    return dict(zip(al["old_name"], al["current_name"]))


def load_playoff_mpg(path: Path | None = None) -> pd.DataFrame:
    """(player_name_norm, season, po_mpg, po_games), alias-resolved and unique."""
    path = PLAYOFF_MPG_PATH if path is None else Path(path)
    po = pd.read_csv(path)
    po["player_name_norm"] = po["player_name_norm"].replace(_alias_map())
    return po.drop_duplicates(["player_name_norm", "season"], keep="first")[
        ["player_name_norm", "season", "po_mpg", "po_games"]]


def attach_playoff_mpg(df: pd.DataFrame, path: Path | None = None) -> pd.DataFrame:
    """Attach `playoff_mpg_diff` (and the raw `po_mpg` / `po_games` it comes from).

    Missing playoff table, or a frame without the join keys, leaves the column
    absent so a caller that does not need it is unaffected; a frame that HAS the
    keys always gets the column, filled 0.0 for non-playoff rows.

    Idempotent: re-attaching recomputes from the same source, never doubles.
    """
    out = df.copy()
    keys = {"player_name_norm", "season", "mpg"}
    path = PLAYOFF_MPG_PATH if path is None else Path(path)
    if not keys.issubset(out.columns) or not path.exists():
        return out

    po = load_playoff_mpg(path)
    out = out.drop(columns=[c for c in ("po_mpg", "po_games") if c in out.columns])
    n0 = len(out)
    out = out.merge(po, on=["player_name_norm", "season"], how="left",
                    validate="many_to_one")
    assert len(out) == n0, "attach_playoff_mpg changed row count"

    mpg = pd.to_numeric(out["mpg"], errors="coerce")
    po_mpg = pd.to_numeric(out["po_mpg"], errors="coerce")
    out["playoff_mpg_diff"] = (po_mpg - mpg).fillna(0.0)
    return out
