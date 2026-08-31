"""Parse per-season salary from the cached Spotrac player pages.

Produces `data/processed/spotrac_salaries.csv`, the candidate replacement for
the Basketball Reference salary chain. Offline — reads only the HTML cache.

WHAT SPOTRAC EXPOSES, AND WHY IT TAKES TWO TABLES

A player page carries two shapes of per-year table and they do not overlap:

  * contract tables — `Year` reads "2026-27", columns include `Cap Hit`.
    These cover the CURRENT and FUTURE years of each listed contract, and
    `Cap Hit` is the cap-charge convention this project settled on in
    ISSUES #38. This is the authoritative column where it exists.

  * career tables — `Year` reads "2017", columns are `Base` / `Signing` /
    `Incentives` / `Cash Total`. These cover the whole career but carry no
    cap hit; `Base` is the nearest equivalent and is what past seasons have
    to fall back on.

So a full-history migration gets cap hit for recent/future seasons and base
salary for older ones. Those are different quantities for any player with a
signing bonus, likely incentives, or a minimum-salary reimbursement, and the
audit script is what decides whether that gap matters. This script only
reports both and never silently picks one: `cap_hit` and `base` are separate
columns, and `salary` prefers cap hit.

SEASON CONVENTION

Both tables label by the season's START year, matching this repo (season 2026
is 2026-27). That is asserted, not assumed: a contract table's "2026-27"
heading and a career table's "2026" row carry the same dollar figure for the
same player, and `_assert_season_alignment` fails the run if they stop
agreeing. Note this differs from the "Career Earnings thru YYYY" AGGREGATE,
which Spotrac labels by ENDING year — see `build_career_earnings.py`, which
calibrated that separately. Two fields on one page, two conventions.

CELL SHAPES, AND SPLIT SEASONS

A traded or amended season prints two amounts in one cell
("$37,096,620 $53,093"). The first is taken; the rest go to `extra_amounts`
rather than being dropped.

Whether those amounts should be SUMMED depends on why there are two, and the
two reasons are not distinguishable from the numbers alone. A player on two
teams has his money split between them and the season total is the sum; a
player with one team has a second figure that is an adjustment, and the first
is already the season figure. Measured against the BBRef chain, summing is
right for traded seasons (170 hits vs 38) and wrong otherwise (14 vs 253).

The switch is the team badge, not `spotrac_transactions.csv` — that table
covers about half the trades and produced errors in both directions
(Avery Bradley 2020 is flagged traded but takes the first amount; AJ Johnson
2025 is not flagged but needs the sum). `n_teams` and `split_total` are
therefore both emitted and the caller chooses; nothing is collapsed here.

An upper bound worth knowing: choosing perfectly between "first" and "sum"
reconciles only 68% of multi-amount rows with the BBRef chain. The residue is
incentives, dead money, and rows where BBRef itself is wrong — it is not a
parsing problem and no cleverer choice rule closes it.

Usage:
    python scripts/build_spotrac_salaries.py            # dry run, prints audit
    python scripts/build_spotrac_salaries.py --write
"""

import argparse
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
from bs4 import BeautifulSoup

from config import CACHE_DIR, PROCESSED_DIR
from scripts.scrape_spotrac_players import page_defect

PLAYER_CACHE = CACHE_DIR / "spotrac_players"
OUT = PROCESSED_DIR / "spotrac_salaries.csv"
URLS = PROCESSED_DIR / "spotrac_player_urls.csv"

MONEY_RE = re.compile(r"\$[\d,]+")
CONTRACT_YEAR_RE = re.compile(r"^(20\d{2})-(\d{2})$")
CAREER_YEAR_RE = re.compile(r"^(19\d{2}|20\d{2})$")


def _normalize_name(name: str) -> str:
    """The project-wide key — see build_dataset._normalize_name."""
    name = str(name).strip().lower()
    nfkd = unicodedata.normalize("NFKD", name)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _amounts(cell: str) -> list[int]:
    """Every dollar figure in a cell, in order."""
    return [int(m.replace("$", "").replace(",", ""))
            for m in MONEY_RE.findall(cell or "")]


# Badge filenames come in two shapes and BOTH must be read:
#   nba_den.png / nba_cle1.png   — league prefix, optional variant digit
#   orl_20251.png                — team code first, season-ish suffix
# A pattern that only knows the `nba_` form silently drops the other badge,
# and a split season then reports ONE team. That is not cosmetic: the split
# test keys on this count, and the miss put Aaron Gordon's 2020 ORL->DEN
# season (and 42 others) on the wrong branch.
TEAM_FILE_RE = re.compile(r"([a-z]{2,3})(?:\d*)(?:_\d+)?\.(?:png|svg|webp)$", re.I)


def _row_teams(tr) -> list[str]:
    """Team codes for a table row, read from the team-badge images.

    Spotrac renders the `Team(s)` column as logos, so the cell's TEXT is
    empty — a text-only parse reports zero teams for every row and cannot
    tell a split season from a whole one. Two badges means the player was on
    two teams that season, which is the only self-contained signal that the
    row's money is split; `spotrac_transactions.csv` covers barely half of
    the trades and errs in both directions.

    Only `logo-sm` images are considered, so a stray sponsor or headshot in
    the row cannot invent a team.
    """
    out = []
    for img in tr.find_all("img", class_="logo-sm"):
        stem = (img.get("src", "") or "").rsplit("/", 1)[-1]
        stem = re.sub(r"^nba_", "", stem, flags=re.I)
        m = TEAM_FILE_RE.match(stem)
        if m:
            code = m.group(1).upper()
            if code not in out:
                out.append(code)
    return out


def _headers(table) -> list[str]:
    return [th.get_text(" ", strip=True) for th in table.find_all("th")]


def _col(headers: list[str], *wanted: str) -> int | None:
    """Index of the first header whose text starts with one of `wanted`.

    Spotrac glues a subtitle onto the header text ("Cap Hit Annual",
    "Cap % League Cap"), so the match is a prefix, not equality.
    """
    for i, h in enumerate(headers):
        for w in wanted:
            if h.lower().startswith(w.lower()):
                return i
    return None


def parse_page(path: Path) -> list[dict]:
    """Every per-season row a player page exposes, from both table shapes."""
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"),
                         "html.parser")
    rows: list[dict] = []

    for table in soup.find_all("table"):
        headers = _headers(table)
        if not headers or _col(headers, "year") != 0:
            continue
        i_cap = _col(headers, "cap hit")
        i_base = _col(headers, "base")
        i_cash = _col(headers, "cash total")
        # Career tables label this plain "Incentives"; contract tables split
        # it into Likely/Unlikely. Only the plain one is wanted here — a
        # prefix match on "incentives" would grab "Incentives Likely" first.
        i_inc = next((i for i, h in enumerate(headers)
                      if h.strip().lower() == "incentives"), None)
        if i_cap is None and i_base is None:
            continue

        for tr in table.find_all("tr"):
            cells = [td.get_text(" ", strip=True)
                     for td in tr.find_all(["td", "th"])]
            if not cells:
                continue
            label = cells[0].strip()
            m_c, m_k = CONTRACT_YEAR_RE.match(label), CAREER_YEAR_RE.match(label)
            if not (m_c or m_k):
                continue                      # header, "Total", "UFA" tail
            season = int((m_c or m_k).group(1))

            def pick(idx):
                if idx is None or idx >= len(cells):
                    return None, []
                vals = _amounts(cells[idx])
                return (vals[0] if vals else None), vals[1:]

            cap_hit, cap_extra = pick(i_cap)
            base, base_extra = pick(i_base)
            cash, _ = pick(i_cash)
            inc, _ = pick(i_inc)
            if cap_hit is None and base is None:
                continue
            teams = _row_teams(tr)
            extra = cap_extra or base_extra
            # A split season prints one amount per team in the same cell.
            # Summing is only correct when the row actually names two teams,
            # so the sum is offered as its own column and the caller decides
            # — the amounts themselves are never silently collapsed.
            split_total = None
            if extra and len(teams) > 1:
                split_total = (cap_hit if cap_hit is not None else base) + sum(extra)
            rows.append({
                "season": season,
                "cap_hit": cap_hit,
                "base": base,
                "cash_total": cash,
                "incentives": inc,
                "teams": ";".join(teams),
                "n_teams": len(teams),
                "split_total": split_total,
                "extra_amounts": ";".join(str(x) for x in extra),
                "table_kind": "contract" if m_c else "career",
            })
    return rows


STATUS_LIVE = ("active",)


def parse_status(path: Path) -> dict[int, dict]:
    """Per-season money split by Spotrac's own `Status` label.

    There is a THIRD career table — `Year | Age | Team(s) | Status |
    Cash Total | Cash Cumulative | Awards` — that `parse_page` skips because
    it carries neither `Base` nor `Cap Hit`. It is the most useful table on
    the page, because its `Status` cell labels each amount in the row, in the
    same order as the amounts:

        Batum 2020   Retained | Active    $8,856,969 | $2,564,753
        Batum 2022   Active | Retained    $10,843,350 | $8,856,969

    That answers the question the team-badge count could only guess at. A row
    with two badges is not necessarily a trade: it is just as often a stretch
    annuity from a waiver years earlier, still being paid by the old team, and
    summing those two figures invents salary the player never earned. Batum's
    2022 season summed to $19.7M against a true $10.8M for exactly that reason.

    The observed label vocabulary, over a 250-page sample:

        Active (1006), Retained (407), Active / Club Exercised (42),
        NBA Cup (41), Active / Player Exercised (6), Buyout (3), Reserve (1)

    Only `Active*` is this player's live salary with this team. `Retained` and
    `Buyout` are another team's dead money. `NBA Cup` is in-season tournament
    prize money — not salary at all, and the badge rule would have added it.

    Returns {season: {"active_total", "retained_total", "other_total",
    "statuses", "n_active"}}. Seasons whose label and amount counts disagree
    are skipped rather than guessed at.
    """
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"),
                         "html.parser")
    out: dict[int, dict] = {}
    for table in soup.find_all("table"):
        headers = _headers(table)
        if not headers or headers[0].strip().lower() != "year":
            continue
        if "Status" not in headers:
            continue
        i_st = headers.index("Status")
        i_ct = _col(headers, "cash total")
        if i_ct is None:
            continue
        for tr in table.find_all("tr"):
            tds = tr.find_all("td")
            if not tds or not CAREER_YEAR_RE.match(tds[0].get_text(strip=True)):
                continue
            if i_st >= len(tds) or i_ct >= len(tds):
                continue
            divs = [d.get_text(strip=True)
                    for d in tds[i_st].find_all(["div", "span"])]
            labels = [s for s in divs if s] or [
                s for s in tds[i_st].get_text("|", strip=True).split("|") if s]
            amounts = _amounts(tds[i_ct].get_text(" ", strip=True))
            # One label per amount, or the pairing is meaningless.
            if not labels or len(labels) != len(amounts):
                continue
            season = int(tds[0].get_text(strip=True))
            live = [a for lab, a in zip(labels, amounts)
                    if lab.strip().lower().startswith(STATUS_LIVE)]
            held = [a for lab, a in zip(labels, amounts)
                    if lab.strip().lower().startswith(("retained", "buyout"))]
            out[season] = {
                "active_total": float(sum(live)) if live else None,
                "retained_total": float(sum(held)) if held else None,
                "other_total": float(sum(amounts) - sum(live) - sum(held)),
                "statuses": ";".join(labels),
                "n_active": len(live),
            }
        break
    return out


TX_DATE_RE = re.compile(r"([A-Z][a-z]{2})\s+(\d{1,2}),\s*(\d{4})")
_MONTHS = {m: i + 1 for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])}


def parse_transactions(path: Path) -> list[tuple[int, str, str]]:
    """(season, kind, text) for every transaction printed on a player page.

    `data/processed/spotrac_transactions.csv` is the existing source for this
    and it is INCOMPLETE: parsing `ul.player-transactions` off the cached
    pages yields 1,142 distinct traded player-seasons against that file's 878,
    and covers all but one of the file's own. The gap is not cosmetic — the
    missing trades are what made a mid-season trade look like dead money.
    Evan Turner, Marvin Williams and Brandon Knight were all traded in
    2019-20, none of them appear in the CSV, and each was priced off half a
    season as a result.

    Season follows the repo convention via the July boundary: a February trade
    belongs to the season that began the previous autumn.
    """
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"),
                         "html.parser")
    ul = soup.find("ul", class_=lambda c: c and "player-transactions" in c)
    if not ul:
        return []
    out = []
    for li in ul.find_all("li"):
        text = re.sub(r"\s+", " ", li.get_text(" ", strip=True))
        m = TX_DATE_RE.search(text)
        if not m:
            continue
        month, year = _MONTHS[m.group(1)], int(m.group(3))
        season = year if month >= 7 else year - 1
        body = text[m.end():].strip(" |")
        low = body.lower()
        kind = ("traded" if "traded" in low else
                "waived" if "waived" in low else
                "signed" if "signed" in low else "other")
        out.append((season, kind, body[:200]))
    return out


def _assert_season_alignment(df: pd.DataFrame) -> None:
    """Both table shapes must label the same season the same way.

    Where a player has a contract row and a career row for one season, the
    dollar figures must agree. If Spotrac ever changes either convention this
    fails loudly instead of shifting an entire source by one season.
    """
    both = df.dropna(subset=["cap_hit", "base"])
    both = both[both["cap_hit"] > 0]
    if both.empty:
        print("  season alignment: no overlapping rows to check")
        return
    agree = (both["cap_hit"] - both["base"]).abs() <= 1000
    rate = agree.mean()
    print(f"  season alignment: {agree.sum()}/{len(both)} overlapping rows "
          f"agree within $1K ({rate:.1%})")
    if rate < 0.80:
        worst = both[~agree].head(10)
        print(worst.to_string(index=False))
        raise SystemExit(
            "Contract and career tables disagree on more than 20% of "
            "overlapping seasons. The season convention may have moved — "
            "resolve before writing."
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    urls = pd.read_csv(URLS)
    urls["slug"] = (urls["url"].str.split("?").str[0].str.rstrip("/")
                    .str.rsplit("/", n=1).str[-1].str.lower())
    # One slug can carry SEVERAL names. Spotrac files a player under his legal
    # name while the box scores use the short one, so the FA-page harvest adds
    # `mohamed bamba` alongside the repo's `mo bamba` — both pointing at
    # /nba/player/mohamed-bamba. A plain dict(zip(...)) lets the alias win and
    # the page's rows are then filed under a name no training row uses: Bamba,
    # Claxton, Mykhailiuk, P.J. Washington, Jae'Sean Tate and A.J. Green all
    # went missing that way, every season of each. Emit a row per name and let
    # the join pick the one it needs.
    slug_to_names: dict[str, list[str]] = {}
    for slug, name in zip(urls["slug"], urls["player_name_norm"]):
        slug_to_names.setdefault(slug, []).append(name)

    pages = sorted(PLAYER_CACHE.glob("*.html"))
    print(f"cached pages: {len(pages)}")

    records, unmapped, empty, defective = [], 0, 0, []
    for p in pages:
        names = slug_to_names.get(p.stem.lower())
        if not names:
            unmapped += 1
            continue
        # A page that is not an NBA player page must never reach the parse.
        # Two of the search-redirect URLs in the URL table resolved to the
        # NFL players of the same name — Spotrac has an A.J. Green and a
        # Cam Thomas in both leagues — and 17 rows of NFL salary were
        # ingested as NBA salary before this guard existed. The page title
        # does NOT catch it (both pages are titled with the right name);
        # only the canonical URL's sport does, which is what page_defect
        # checks. See ISSUES #41.
        defect = page_defect(p)
        if defect:
            defective.append((names[0], p.name, str(defect)[:60]))
            continue
        rows = parse_page(p)
        status = parse_status(p)
        if not rows:
            empty += 1
            continue
        for r in rows:
            st = status.get(r["season"], {})
            for name in names:
                records.append({**r, **st, "player_name_norm": name})

    df = pd.DataFrame(records)
    print(f"  pages with no URL-table entry : {unmapped}")
    print(f"  pages rejected by page_defect : {len(defective)}")
    for who, fname, why in defective:
        print(f"      {who:24s} {fname:24s} {why}")
    print(f"  pages yielding no season rows : {empty}")
    print(f"  raw season rows parsed        : {len(df)}")

    # A season can appear in several tables (contract + career, or two
    # contracts overlapping a traded year). Keep the richest: a cap hit
    # beats none, then the larger base.
    df["_has_cap"] = df["cap_hit"].notna()
    df = (df.sort_values(["player_name_norm", "season", "_has_cap", "base"],
                         ascending=[True, True, False, False])
            .drop_duplicates(["player_name_norm", "season"], keep="first")
            .drop(columns="_has_cap"))

    df["salary"] = df["cap_hit"].fillna(df["base"])
    df["salary_basis"] = df["cap_hit"].notna().map(
        {True: "cap_hit", False: "base"})
    print(f"  distinct player-seasons       : {len(df)}")
    print(f"    from cap hit                : {(df.salary_basis=='cap_hit').sum()}")
    print(f"    from base salary            : {(df.salary_basis=='base').sum()}")

    _assert_season_alignment(pd.DataFrame(records))

    for c in ("active_total", "retained_total", "other_total",
              "statuses", "n_active"):
        if c not in df:
            df[c] = None
    df = df[["player_name_norm", "season", "salary", "salary_basis",
             "cap_hit", "base", "cash_total", "incentives", "teams", "n_teams",
             "split_total", "extra_amounts", "active_total", "retained_total",
             "other_total", "statuses", "n_active",
             "table_kind"]].sort_values(["player_name_norm", "season"])

    print("\nrows per season:")
    print(df[df.season.between(2019, 2029)].groupby("season").size().to_string())

    if not args.write:
        print("\ndry run — nothing written. Re-run with --write.")
        return
    df.to_csv(OUT, index=False)
    print(f"\nwrote {OUT} ({len(df)} rows)")


if __name__ == "__main__":
    main()
