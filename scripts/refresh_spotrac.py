"""Refresh Spotrac free-agency data for a new FA year.

Does four things, all cached and rate-limited:

  1. Fetch the FA-class page (free-agents/signed/_/year/YYYY) into the HTML
     cache and append its signings to spotrac_fa_signings.csv.
  2. Scrape player pages for signees not yet in the player cache.
  3. Re-parse every cached player page with the anchor-based season assignment
     in scrape_spotrac_players.parse_contracts and rewrite
     spotrac_signing_types.csv. This table is a diagnostic label source, never
     a model feature, so regenerating it cannot move any published metric.
  4. Report label quality: mislabeled-minimum count and training-row coverage.

    python scripts/refresh_spotrac.py                # FA year 2026
    python scripts/refresh_spotrac.py --year 2027
    python scripts/refresh_spotrac.py --reparse-only # no network at all
"""

import argparse
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import requests
from bs4 import BeautifulSoup

from config import CACHE_DIR, PROCESSED_DIR, USER_AGENT, CAP_BY_SEASON
from scripts.scrape_spotrac_players import (
    PLAYER_CACHE, norm, parse_contracts, slugify,
)

FA_URL = "https://www.spotrac.com/nba/free-agents/signed/_/year/{year}"
UPCOMING_FA_URL = "https://www.spotrac.com/nba/free-agents/_/year/{year}"
DELAY = 5
BACKOFF = 60


def _get(url: str, retries: int = 3) -> str | None:
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
            if resp.status_code == 200:
                return resp.text
            if resp.status_code == 403:
                wait = BACKOFF * (attempt + 1)
                print(f"    403, backing off {wait}s...")
                time.sleep(wait)
        except requests.exceptions.RequestException as e:
            print(f"    retry {attempt + 1}/{retries}: {e}")
            time.sleep(5)
    return None


def fetch_fa_page(year: int, force: bool = False) -> Path | None:
    path = CACHE_DIR / f"spotrac_fa_{year}.html"
    if path.exists() and not force:
        print(f"FA page for {year} already cached")
        return path
    html = _get(FA_URL.format(year=year))
    if html is None:
        print(f"could not fetch the {year} FA page")
        return None
    path.write_text(html, encoding="utf-8")
    print(f"cached {path.name} ({len(html) // 1024} KB)")
    return path


def parse_fa_page(path: Path, year: int) -> tuple[pd.DataFrame, dict[str, str]]:
    """FA table rows plus {player_name_norm: player-page URL}."""
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"),
                         "html.parser")
    tables = soup.find_all("table")
    if not tables:
        return pd.DataFrame(), {}
    rows, urls = [], {}
    for tr in tables[0].find_all("tr")[1:]:
        cells = tr.find_all("td")
        text = [td.get_text(strip=True) for td in cells]
        if len(text) < 8 or not text[3]:
            continue
        pname = text[3]
        rows.append({
            "player_name": pname,
            "player_name_norm": norm(pname),
            "fa_year": year,
            "season": year,
            "from_team": text[0],
            "to_team": text[2],
            "stayed_with_team": int(text[0] == text[2] and text[0] != ""),
            "contract_years": pd.to_numeric(text[5], errors="coerce"),
            "total_value_str": text[6],
            "aav_str": text[7],
        })
        for td in cells:
            link = td.find("a", href=True)
            if link and "/nba/player/" in link["href"]:
                urls[norm(pname)] = link["href"]
                break
    return pd.DataFrame(rows), urls


def scrape_new_players(urls: dict[str, str], force: bool = False) -> None:
    """Fetch signee pages. With force, refetch even cached ones — a page cached
    before the signing happened does not show the new contract, which is the
    normal state for every re-signing veteran after a refresh."""
    if force:
        missing = dict(urls)
    else:
        missing = {p: u for p, u in urls.items()
                   if not (PLAYER_CACHE / f"{slugify(p)}.html").exists()}
    print(f"player pages: {len(urls)} signees, {len(missing)} to fetch"
          f"{' (forced)' if force else ''}")
    for i, (pname, url) in enumerate(sorted(missing.items()), 1):
        if url.startswith("/"):
            url = "https://www.spotrac.com" + url
        html = _get(url)
        if html:
            (PLAYER_CACHE / f"{slugify(pname)}.html").write_text(
                html, encoding="utf-8")
            print(f"  [{i}/{len(missing)}] {pname}")
        else:
            print(f"  [{i}/{len(missing)}] {pname}: FAILED")
        time.sleep(DELAY)


def _slug_to_training_name() -> dict[str, str]:
    """Map cache-file slugs back to the training data's player_name_norm.

    Slugs strip dots and apostrophes ("a.j. green" -> "aj-green"), so reversing
    a slug produces a name that fails to merge with the training data. Building
    the map from the training side keeps every downstream join exact — this
    mismatch alone hid the labels for every dotted or hyphenated name.
    """
    train = pd.read_csv(PROCESSED_DIR / "training_data_v2.csv")
    train = train.loc[:, ~train.columns.duplicated()]
    return {slugify(p): p for p in train["player_name_norm"].unique()}


def harvest_upcoming_urls(years=range(2027, 2032)) -> dict[str, str]:
    """Player URLs from the upcoming-FA lists, keyed by slug.

    Players who never reached free agency (rookie-max extensions: Adebayo,
    Holmgren, the 2022 draft class) appear on no signed-FA page, so the signed
    pages cannot supply their URLs. Every rostered player eventually shows up
    on some future year's upcoming-FA list, which makes those five pages a
    near-complete URL directory. Pages are cached; at most five fetches.
    """
    urls: dict[str, str] = {}
    for y in years:
        path = CACHE_DIR / f"spotrac_fa_upcoming_{y}.html"
        if not path.exists():
            html = _get(UPCOMING_FA_URL.format(year=y))
            if html is None:
                continue
            path.write_text(html, encoding="utf-8")
            print(f"cached {path.name} ({len(html) // 1024} KB)")
            time.sleep(DELAY)
        soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"),
                             "html.parser")
        for a in soup.find_all("a", href=True):
            if "/nba/player/" in a["href"]:
                slug = slugify(a.get_text(strip=True))
                if slug:
                    urls.setdefault(slug, a["href"])
    return urls


def fill_missing_pages() -> None:
    """Fetch pages for unlabeled evaluation players that have no cache entry."""
    from scripts.diagnostics import attach_signing_labels
    from src.model.train import (
        load_training_data, _filter_year1, _filter_rookie_scale,
    )
    df = _filter_rookie_scale(_filter_year1(load_training_data()))
    df = attach_signing_labels(df, salary_dollars=df["salary"])
    unlabeled = df.loc[df["signing_cat"] == "Unknown", "player_name_norm"].unique()
    missing = [p for p in unlabeled
               if not (PLAYER_CACHE / f"{slugify(p)}.html").exists()]
    print(f"{len(unlabeled)} unlabeled evaluation players, "
          f"{len(missing)} with no cached page")
    if not missing:
        return
    urls = harvest_upcoming_urls()
    print(f"harvested {len(urls)} URLs from upcoming-FA pages")
    hit = {p: urls[slugify(p)] for p in missing if slugify(p) in urls}
    print(f"URL found for {len(hit)} of {len(missing)}")
    scrape_new_players(hit)


def rebuild_signing_types() -> pd.DataFrame:
    """Re-parse every cached player page into spotrac_signing_types.csv."""
    slug_map = _slug_to_training_name()
    all_rows = []
    pages = sorted(PLAYER_CACHE.glob("*.html"))
    for path in pages:
        pname = slug_map.get(path.stem, path.stem.replace("-", " "))
        for c in parse_contracts(path):
            for season in c.get("seasons", []):
                all_rows.append({
                    "player_name_norm": pname,
                    "season": season,
                    "signing_type": c.get("signing_type", ""),
                    "contract_years": c.get("contract_years"),
                    "total_value": c.get("total_value"),
                    "aav": c.get("aav"),
                })
    df = pd.DataFrame(all_rows)
    out = PROCESSED_DIR / "spotrac_signing_types.csv"
    df.to_csv(out, index=False)
    print(f"re-parsed {len(pages)} player pages -> {out.name} ({len(df)} rows)")
    return df


def validate(st: pd.DataFrame) -> None:
    """Label-quality report against the training data.

    Uses the same salary-aware attach the diagnostics use, so a mid-season
    buyout row is judged by the contract that produced its salary rather than
    whichever deal happens to share the season.
    """
    from scripts.diagnostics import attach_signing_labels

    train = pd.read_csv(PROCESSED_DIR / "training_data_v2.csv")
    train = train.loc[:, ~train.columns.duplicated()]
    m = attach_signing_labels(train, salary_dollars=train["salary"])
    m["salary_m"] = m["salary"] / 1e6

    bad = m[(m["signing_cat"] == "Minimum") & (m["salary_m"] > 6)]
    print(f"\ntraining rows labeled Minimum with salary > $6M: {len(bad)}"
          f"  (30 under the old season walk)")
    if len(bad):
        print(bad[["player_name_norm", "season", "salary_m"]]
              .sort_values("salary_m", ascending=False).head(8)
              .to_string(index=False))

    print("\nlabel coverage (all training rows / year-1 evaluation rows):")
    from src.model.train import _filter_year1, _filter_rookie_scale
    ev = _filter_rookie_scale(_filter_year1(m))
    for s in sorted(m["season"].unique()):
        sub, sub_ev = m[m["season"] == s], ev[ev["season"] == s]
        cov = (sub["signing_cat"] != "Unknown").mean() * 100
        cov_ev = ((sub_ev["signing_cat"] != "Unknown").mean() * 100
                  if len(sub_ev) else float("nan"))
        print(f"  {int(s)}: {cov:5.1f}%  /  {cov_ev:5.1f}%  "
              f"(n={len(sub)} / {len(sub_ev)})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--year", type=int, default=2026)
    ap.add_argument("--force-fa", action="store_true",
                    help="re-fetch the FA page even if cached")
    ap.add_argument("--refresh-signees", action="store_true",
                    help="refetch every signee page, not only uncached ones — "
                         "needed once per FA year, since pages cached before "
                         "a signing do not show it")
    ap.add_argument("--fill-missing", action="store_true",
                    help="fetch pages for unlabeled evaluation players via "
                         "the upcoming-FA URL directory (extensions never "
                         "appear on signed-FA pages)")
    ap.add_argument("--reparse-only", action="store_true",
                    help="skip all network work; just re-parse cached pages")
    args = ap.parse_args()

    if args.fill_missing:
        fill_missing_pages()

    if not args.reparse_only and not args.fill_missing:
        path = fetch_fa_page(args.year, force=args.force_fa)
        if path is not None:
            fa_rows, urls = parse_fa_page(path, args.year)
            print(f"parsed {len(fa_rows)} signings for {args.year}")
            if len(fa_rows):
                fa_csv = PROCESSED_DIR / "spotrac_fa_signings.csv"
                fa = pd.read_csv(fa_csv)
                fa = fa[fa["fa_year"] != args.year]
                fa = pd.concat([fa, fa_rows], ignore_index=True)
                fa.to_csv(fa_csv, index=False)
                print(f"updated {fa_csv.name}: {len(fa)} rows "
                      f"({(fa['fa_year'] == args.year).sum()} for {args.year})")
                scrape_new_players(urls, force=args.refresh_signees)

    st = rebuild_signing_types()
    validate(st)


if __name__ == "__main__":
    main()
