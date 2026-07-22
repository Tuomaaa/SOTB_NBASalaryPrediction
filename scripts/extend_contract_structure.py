"""Extend contract_structure_v2.csv to cover refreshed salary rows.

The script that originally produced this table was never committed. A
reconstruction from CBA escalator ratios reaches only 88% agreement on the
year-1 flag — far too low to regenerate history without invalidating every
published version number. So this script never recomputes anything: a row whose
salary is unchanged keeps its existing assignment byte-for-byte, and only rows
that are new or whose salary moved get assigned, as fresh contracts starting at
the first dirty season (a deal signed in July that starts next season is year 1
by definition, and BBRef team pages list the guaranteed years of that deal as
the consecutive seasons that follow).

Reads the pre-refresh table from salaries_prev.csv (written by
scripts/refresh_salaries.py), falling back to git HEAD when the sidecar is
missing. Hard-fails if any row with an unchanged salary would move.
"""

import io
import subprocess
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

from config import PROCESSED_DIR

SALARIES = PROCESSED_DIR / "salaries.csv"
PREV_SIDECAR = PROCESSED_DIR / "salaries_prev.csv"
STRUCTURE = PROCESSED_DIR / "contract_structure_v2.csv"


def norm(s: str) -> str:
    """lower + strip accents — the key convention the structure table uses."""
    s = str(s).strip().lower()
    return "".join(c for c in unicodedata.normalize("NFKD", s)
                   if not unicodedata.combining(c))


def load_previous_salaries() -> pd.DataFrame:
    if PREV_SIDECAR.exists():
        return pd.read_csv(PREV_SIDECAR)
    print(f"{PREV_SIDECAR.name} not found, falling back to git HEAD")
    blob = subprocess.run(
        ["git", "show", "HEAD:data/processed/salaries.csv"],
        cwd=ROOT, capture_output=True, check=True,
    ).stdout
    return pd.read_csv(io.BytesIO(blob))


def main() -> None:
    new_sal = pd.read_csv(SALARIES)
    old_sal = load_previous_salaries()
    struct = pd.read_csv(STRUCTURE)

    for d in (old_sal, new_sal):
        d["k"] = d["player"].map(norm)

    old_map = old_sal.set_index(["k", "season"])["salary"].to_dict()
    struct_map = struct.set_index(["player_name_norm", "season"]).to_dict("index")

    status = {}
    for k, s, sal in zip(new_sal["k"], new_sal["season"], new_sal["salary"]):
        prev = old_map.get((k, s))
        status[(k, s)] = ("added" if prev is None
                          else "changed" if prev != sal else "same")
    counts = pd.Series(list(status.values())).value_counts()
    print("refreshed salary rows by status:")
    for label, n in counts.items():
        print(f"  {label:8s} {n:5d}")
    if counts.get("added", 0) + counts.get("changed", 0) == 0:
        print("\nnothing new — structure table already covers every row")

    present: dict[str, set[int]] = {}
    for k, s in zip(new_sal["k"], new_sal["season"]):
        present.setdefault(k, set()).add(int(s))
    cap_pct = new_sal.set_index(["k", "season"])["cap_pct"].to_dict()

    rows, assigned = [], set()
    carried = 0

    # 1) unchanged rows keep their assignment exactly
    for (k, s), st in status.items():
        if st == "same" and (k, s) in struct_map:
            src = struct_map[(k, s)]
            rows.append({"player_name_norm": k, "season": s,
                         "starting_cap_pct": src["starting_cap_pct"],
                         "contract_years": src["contract_years"],
                         "year_in_contract": src["year_in_contract"]})
            assigned.add((k, s))
            carried += 1

    # 2) new/changed rows become fresh contracts over consecutive dirty seasons
    dirty = sorted(key for key, st in status.items() if st != "same")
    new_contracts = 0
    for k, s in dirty:
        if (k, s) in assigned:
            continue
        run = []
        t = int(s)
        while (k, t) in status and t in present.get(k, set()):
            if (k, t) in assigned:
                break
            run.append(t)
            t += 1
        start_pct = cap_pct.get((k, run[0]))
        for i, season in enumerate(run, start=1):
            rows.append({"player_name_norm": k, "season": season,
                         "starting_cap_pct": start_pct,
                         "contract_years": len(run), "year_in_contract": i})
            assigned.add((k, season))
        new_contracts += 1

    out = (pd.DataFrame(rows)
           .sort_values(["player_name_norm", "season"])
           .reset_index(drop=True))
    print(f"\ncarried over unchanged      : {carried}")
    print(f"newly assigned contracts    : {new_contracts} "
          f"({len(out) - carried} rows)")

    # ---- gate: an unchanged-salary row must never move -------------------
    before = struct.set_index(["player_name_norm", "season"])
    after = out.set_index(["player_name_norm", "season"])
    common = before.index.intersection(after.index)
    moved = common[
        (before.loc[common, "year_in_contract"]
         != after.loc[common, "year_in_contract"])
        | (before.loc[common, "contract_years"]
           != after.loc[common, "contract_years"])
    ]
    illegal = [key for key in moved if status.get(key) == "same"]
    print(f"\nvalidation: {len(common)} pre-existing rows, {len(moved)} moved, "
          f"{len(illegal)} moved WITHOUT a salary change")
    if illegal:
        for key in illegal[:10]:
            print(f"  ILLEGAL move: {key}")
        raise SystemExit("aborting — structure table left unwritten")

    y1_before = (struct["year_in_contract"] == 1).sum()
    y1_after = (out["year_in_contract"] == 1).sum()
    print(f"year-1 rows: {y1_before} -> {y1_after} ({y1_after - y1_before:+d})")

    out.to_csv(STRUCTURE, index=False)
    print(f"\nwrote {STRUCTURE} ({len(out)} rows)")


if __name__ == "__main__":
    main()
