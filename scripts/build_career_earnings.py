"""Career earnings banked before season T, read straight off Spotrac.

Produces `data/processed/career_earnings.csv`. Offline — reads only the HTML
cache. This is the "has banked enough money that winning beats the marginal
dollar" leg of the ring-chasing hypothesis; see
`docs/briefs/2026-08-29-ring-chasing-handoff.md`.

## What changed, and why the old method is gone

The first version could not see a per-season career total, only Spotrac's
single "Career Earnings thru YYYY" headline, so it worked backwards:

    career_earnings_thru_prev(T) = anchor - sum(salaries.csv, seasons T..2025)

That construction dragged in four problems, every one of which is now moot:

  * **A convention gap.** The anchor is cash paid; `salaries.csv` mixes cash
    and cap charge (ISSUES #36, #38), so every subtracted season injected a
    mean +$0.104M error. Harmless against Chris Paul's $403M, NOT harmless
    below ~$2M, which is where the two-way class lives.
  * **An anchor-season calibration.** Spotrac labels the headline by the
    season's ENDING year while this repo labels by the STARTING year, so the
    anchor had to be pinned to repo 2025 by measurement, and re-pinned if the
    cache moved.
  * **A dependency on `salaries.csv`**, which made the whole column a hostage
    of the Spotrac migration — it would have needed regenerating afterwards.
  * **40 rows clipped at zero** because the subtraction went negative.

The career table carries a **`Cash Cumulative` column, per season, from the
player's rookie year**. Chris Paul's runs 2005-2025 and ends at $404,526,572;
LeBron's runs from 2003 and reads $581,375,548 through 2025. Reading that cell
directly answers the question the subtraction was approximating, so all four
problems disappear at once and the column becomes migration-independent.

## Two things that look like defects and are not

**The 2019-20 COVID reduction is kept.** Spotrac's cash figures for that season
are 93.75% of the contracted amount because that is what players were actually
paid. The `salary` target elsewhere in this project wants the CONTRACTED
amount, but this column asks how much money is in the bank, so the reduced
figure is the correct one. Do not "fix" it here.

**A split season prints two cumulative values** ("$403,148,100 $404,526,572").
Take the LAST. The page's own "Total" row settles it — Chris Paul's reads
$404,526,572 — and note this is the OPPOSITE of the rule the per-season `Base`
column needs, where the first figure is the contract's own and the rest belong
to the other team. Same cell shape, two different correct answers.

## Semantics

`career_earnings_thru_prev(T)` is the cumulative at season **T-1**, so nothing
in it is knowable only after season T is signed. Where the page has no T-1 row
(a gap year, or a season abroad) the nearest earlier season is used and the row
is flagged `gap_filled` rather than silently interpolated.

Usage:
    python scripts/build_career_earnings.py            # dry run, prints checks
    python scripts/build_career_earnings.py --write
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
from bs4 import BeautifulSoup

from config import CACHE_DIR, PROCESSED_DIR, CAP_BY_SEASON
from scripts.scrape_spotrac_players import page_defect

PLAYER_CACHE = CACHE_DIR / "spotrac_players"
OUT_PATH = PROCESSED_DIR / "career_earnings.csv"
URLS = PROCESSED_DIR / "spotrac_player_urls.csv"
TRAINING = PROCESSED_DIR / "training_data_v2.csv"
SALARIES = PROCESSED_DIR / "salaries.csv"

FIRST_SEASON = 2019
LAST_SEASON = 2026

MONEY_RE = re.compile(r"\$[\d,]+")
CAREER_YEAR_RE = re.compile(r"^(19\d{2}|20\d{2})$")

# Anchors that must survive any parser change. Values are the cumulative at the
# stated season, read from the page by hand on 2026-08-29.
# Each value is the page's OWN "Total" row, which is the authority on the
# first-vs-last question above: Chris Paul's Total reads $404,526,572 — the
# LAST of the two figures his 2025 cumulative cell prints, not the first.
KNOWN = {
    ("chris paul", 2025): 404_526_572,
    ("lebron james", 2025): 581_375_548,
    ("udonis haslem", 2022): 71_005_646,
}


def _last_amount(cell: str) -> int | None:
    """The LAST dollar figure in a cell — see the split-season note above."""
    found = MONEY_RE.findall(cell or "")
    return int(found[-1].replace("$", "").replace(",", "")) if found else None


def _col(headers: list[str], wanted: str) -> int | None:
    for i, h in enumerate(headers):
        if h.strip().lower().startswith(wanted):
            return i
    return None


def parse_cumulative(path: Path) -> dict[int, int]:
    """{season: cash cumulative} from a player page's career table."""
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"),
                         "html.parser")
    out: dict[int, int] = {}
    for table in soup.find_all("table"):
        headers = [th.get_text(" ", strip=True) for th in table.find_all("th")]
        if not headers or _col(headers, "year") != 0:
            continue
        # Career tables only. A contract table also carries a cumulative but
        # restarts it at the contract, which is not a career total.
        if _col(headers, "cap hit") is not None:
            continue
        i_cum = _col(headers, "cash cumulative")
        if i_cum is None:
            continue
        for tr in table.find_all("tr"):
            cells = [td.get_text(" ", strip=True)
                     for td in tr.find_all(["td", "th"])]
            if not cells or not CAREER_YEAR_RE.match(cells[0].strip()):
                continue
            if i_cum >= len(cells):
                continue
            val = _last_amount(cells[i_cum])
            if val is None:
                continue          # future seasons print an empty cell
            season = int(cells[0])
            # Several career tables can appear; keep the largest reading for a
            # season, since a partial table would understate the running total.
            out[season] = max(val, out.get(season, 0))
    return out


def load_pages() -> dict[str, dict[int, int]]:
    """{player_name_norm: {season: cumulative}} over the whole cache."""
    urls = pd.read_csv(URLS)
    urls["slug"] = (urls["url"].str.split("?").str[0].str.rstrip("/")
                    .str.rsplit("/", n=1).str[-1].str.lower())
    # One slug, several names — Spotrac's legal name plus the repo's short one.
    # See build_spotrac_salaries.py; a plain dict lets the alias win and the
    # page's numbers land under a name no training row uses.
    slug_to_names: dict[str, list[str]] = {}
    for slug, name in zip(urls["slug"], urls["player_name_norm"]):
        slug_to_names.setdefault(slug, []).append(name)

    by_player: dict[str, dict[int, int]] = {}
    defective = 0
    for path in sorted(PLAYER_CACHE.glob("*.html")):
        names = slug_to_names.get(path.stem.lower())
        if not names:
            continue
        if page_defect(path):
            defective += 1
            continue
        cum = parse_cumulative(path)
        if not cum:
            continue
        for name in names:
            by_player[name] = cum
    print(f"pages with a career cumulative: {len(by_player)} "
          f"({defective} rejected by page_defect)")
    return by_player


def _assert_known(by_player: dict[str, dict[int, int]]) -> None:
    for (who, season), expected in KNOWN.items():
        got = by_player.get(who, {}).get(season)
        if got != expected:
            raise SystemExit(
                f"anchor check failed: {who} {season} cumulative reads "
                f"{got!r}, expected {expected:,}. Check the page's own "
                f"'Total' row before touching either side — it is the "
                f"authority on which figure a multi-value cumulative means."
            )
    print(f"anchor checks OK ({', '.join(w for w, _ in KNOWN)})")


def build(by_player: dict[str, dict[int, int]],
          keys: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, season in zip(keys["player_name_norm"], keys["season"]):
        cum = by_player.get(name)
        if not cum:
            rows.append((name, season, None, "no_spotrac_page"))
            continue
        prior = [s for s in cum if s < season]
        if not prior:
            rows.append((name, season, 0.0, "no_prior_season"))
            continue
        latest = max(prior)
        quality = "ok" if latest == season - 1 else "gap_filled"
        rows.append((name, season, float(cum[latest]), quality))

    out = pd.DataFrame(rows, columns=["player_name_norm", "season",
                                      "career_earnings_thru_prev", "quality"])
    cap = out["season"].map(CAP_BY_SEASON).astype(float)
    out["career_earnings_thru_prev_cap_pct"] = out["career_earnings_thru_prev"] / cap
    return out.sort_values(["player_name_norm", "season"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    by_player = load_pages()
    _assert_known(by_player)

    tr = pd.read_csv(TRAINING)
    tr = tr.loc[:, ~tr.columns.duplicated()]
    sal = pd.read_csv(SALARIES)
    sal["player_name_norm"] = sal["player"].str.strip().str.lower()
    keys = pd.concat([tr[["player_name_norm", "season"]],
                      sal[["player_name_norm", "season"]]])
    keys = keys.dropna(subset=["player_name_norm"]).astype({"season": int})
    keys = keys[(keys["season"] >= FIRST_SEASON) & (keys["season"] <= LAST_SEASON)]
    keys = keys.drop_duplicates().reset_index(drop=True)

    out = build(by_player, keys)
    print(f"\nrows {len(out)}, players {out['player_name_norm'].nunique()}")
    print(out["quality"].value_counts().to_string())

    if OUT_PATH.exists():
        old = pd.read_csv(OUT_PATH)
        cmp = old[["player_name_norm", "season", "career_earnings_thru_prev"]].merge(
            out[["player_name_norm", "season", "career_earnings_thru_prev"]],
            on=["player_name_norm", "season"], how="inner",
            suffixes=("_old", "_new")).dropna()
        d = cmp["career_earnings_thru_prev_new"] - cmp["career_earnings_thru_prev_old"]
        print(f"\nvs the subtract-forward version, {len(cmp)} shared rows:")
        print(f"  identical            : {(d.abs() < 1).sum()}")
        print(f"  within $1M           : {(d.abs() < 1e6).sum()}")
        print(f"  median signed delta  : ${d.median()/1e6:+.3f}M")
        print(f"  mean signed delta    : ${d.mean()/1e6:+.3f}M")
        big = cmp.assign(d=d).reindex(d.abs().sort_values(ascending=False).index)
        print("\n  largest 8 moves:")
        print(big.head(8).to_string(index=False, float_format=lambda x: f"{x:,.0f}"))

    if not args.write:
        print("\ndry run — nothing written. Re-run with --write.")
        return
    out.to_csv(OUT_PATH, index=False)
    print(f"\nwrote {OUT_PATH} — {len(out)} rows")


if __name__ == "__main__":
    main()
