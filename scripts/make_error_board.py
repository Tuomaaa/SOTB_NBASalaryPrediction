"""Build the curated top-40 error board from the champion OOF reference.

The board's row universe is outputs/models/oof_reference.csv, written by
src/model/evaluate_suite.py from load_evaluation_frame() — the exact membership
chain train.py fits on. Building the board from anything looser puts rows the
model never trains on onto it: the pre-fix board carried seven such phantoms
(Hayward 2019/2023, Graham 2020, Millsap 2019, Love 2022, Camara 2025,
Wall 2020), all left-censored or rookie-contract rows the filter chain drops.

Two inputs persist review state across regenerations, keyed (player, season):
  - the existing board CSV: category + comment carry forward;
  - error_board_kicked.csv: entries reviewed in prior sessions and judged
    model-correct — these never return to the board.
Rows on neither list get category "new".
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

from config import CAP_BY_SEASON, OUTPUTS_DIR, PROCESSED_DIR

TOP_N = 40
DIAG_DIR = OUTPUTS_DIR / "diagnostics"
BOARD_PATH = DIAG_DIR / "v84_top40_annotated.csv"
KICKED_PATH = DIAG_DIR / "error_board_kicked.csv"
OOF_PATH = OUTPUTS_DIR / "models" / "oof_reference.csv"


def _key(df: pd.DataFrame) -> pd.Series:
    return list(zip(df["player_name_norm"], df["season"].astype(int)))


def main() -> None:
    ref = pd.read_csv(OOF_PATH)
    cap = ref["season"].map(CAP_BY_SEASON)
    ref["pred_m"] = ref["oof_champion"] * cap / 1e6
    ref["err_m"] = ref["pred_m"] - ref["salary_m"]
    ref["direction"] = ref["err_m"].map(lambda e: "OVER" if e > 0 else "UNDER")

    td = pd.read_csv(PROCESSED_DIR / "training_data_v2.csv",
                     usecols=["player_name_norm", "season", "games", "is_waived"])
    td = td.drop_duplicates(["player_name_norm", "season"])
    ref = ref.merge(td, on=["player_name_norm", "season"], how="left")

    kicked = set()
    if KICKED_PATH.exists():
        k = pd.read_csv(KICKED_PATH)
        kicked = set(_key(k))

    # Annotation sources, oldest first — the current board wins on conflict.
    # Legacy category names map to the current taxonomy ("model wrong" and
    # "not modelable" are retired; TBD is the conservative translation).
    legacy = {"model wrong": "TBD", "not modelable": "TBD"}
    prior = {}
    for src in (DIAG_DIR / "v84_top30_annotated.csv", BOARD_PATH):
        if not src.exists():
            continue
        old = pd.read_csv(src)
        for key, (_, row) in zip(_key(old), old.iterrows()):
            if isinstance(row["category"], str) and row["category"] not in ("", "new"):
                comment = row["comment"] if isinstance(row["comment"], str) else ""
                prior[key] = (legacy.get(row["category"], row["category"]), comment)

    ref["_key"] = _key(ref)
    pool = ref[~ref["_key"].isin(kicked)].copy()
    pool["abs_err"] = pool["err_m"].abs()
    board = pool.sort_values("abs_err", ascending=False).head(TOP_N).copy()

    board["category"] = [prior.get(k, ("new", ""))[0] for k in board["_key"]]
    board["comment"] = [prior.get(k, ("new", ""))[1] for k in board["_key"]]

    out = board[["player_name_norm", "season", "direction", "salary_m",
                 "pred_m", "err_m", "is_waived", "games", "category", "comment"]].copy()
    out = out.rename(columns={"games": "gp"})
    for col in ("salary_m", "pred_m"):
        out[col] = out[col].round(2)
    out["err_m"] = out["err_m"].map(lambda e: f"{e:+.2f}")
    out["is_waived"] = out["is_waived"].fillna(0).astype(int)
    out["gp"] = out["gp"].fillna(0).astype(int)

    dropped = [k for k in prior if k not in set(board["_key"])]
    out.to_csv(BOARD_PATH, index=False)
    print(f"Board written: {BOARD_PATH.name} ({len(out)} rows)")
    counts = out["category"].value_counts()
    print("Categories: " + ", ".join(f"{c}={n}" for c, n in counts.items()))
    if dropped:
        print("Annotated entries no longer on the board:")
        for p, s in sorted(dropped, key=lambda x: x[1]):
            print(f"  {p} {s}  ({prior[(p, s)][0]})")


if __name__ == "__main__":
    main()
