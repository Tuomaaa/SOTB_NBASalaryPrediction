"""Scrape Basketball Reference advanced tables for the retirement-hazard fit.

The hazard needs completed careers: every player-season from 1996-97 onward,
so that a player's last appearance can be read as his retirement. The model's
own `advanced_stats.csv` starts in 2019, which is too late to see how long
players of each tier and age kept playing.

Stop at the first run of 403s: Basketball Reference jails a client that keeps
requesting after a refusal (2026-09-28: a cloud session was jailed after 22
refused requests). Rerunning resumes from the cache.

Writes `data/processed/advanced_stats_history.csv` with the same columns as
`advanced_stats.csv`. `season` is Basketball Reference's END year, as there.
Pages are cached under `data/raw/html_cache/`; a failed fetch leaves the cache
untouched and the season is reported, not written.

    python scripts/scrape_advanced_history.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

import subprocess
import time

import pandas as pd

from config import CACHE_DIR, PROCESSED_DIR, USER_AGENT
from src.scraping.stats import BBREF_BASE, scrape_season_advanced
from src.scraping.utils import _cache_path

FIRST_SEASON = 1997
LAST_SEASON = 2018          # 2019 onward is advanced_stats.csv
OUT = PROCESSED_DIR / "advanced_stats_history.csv"
# Basketball Reference allows about 20 requests a minute; stay under it.
REQUEST_GAP_SECONDS = 4.0


def _prefetch(season: int, last: list[float]) -> None:
    """Fill the page cache with a plain HTTP fetch when it is missing.

    `scrape_season_advanced` fetches through Playwright, which cannot launch
    where its bundled browser does not match. These tables are static HTML, so
    a plain request returns the same page. Only a complete page is cached, so a
    failed fetch never replaces a valid cached one.
    """
    url = f"{BBREF_BASE}/leagues/NBA_{season}_advanced.html"
    path = _cache_path(url)
    if path.exists():
        return
    wait = REQUEST_GAP_SECONDS - (time.time() - last[0])
    if wait > 0:
        time.sleep(wait)
    # curl, not `requests`: Basketball Reference answers python-requests with
    # 403 while serving the same URL and User-Agent to curl.
    res = subprocess.run(
        ["curl", "-sS", "-m", "30", "-A", USER_AGENT, "-w", "\n%{http_code}",
         url], capture_output=True, text=True)
    last[0] = time.time()
    body, _, code = res.stdout.rpartition("\n")
    if res.returncode != 0 or code != "200" or 'id="advanced"' not in body:
        raise RuntimeError(f"HTTP {code or res.returncode} for {url}")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def main() -> None:
    """Scrape each season, keep what succeeds, and report what failed."""
    frames, failed = [], []
    last = [0.0]
    for season in range(FIRST_SEASON, LAST_SEASON + 1):
        if len(failed) >= 2 and failed[-2:] == [season - 2, season - 1]:
            print("  two refusals in a row; stopping to avoid a longer jail")
            break
        try:
            _prefetch(season, last)
        except RuntimeError as err:
            # Fall through to the project's Playwright fetch, which works on
            # machines whose Playwright browser matches its version.
            print(f"  [curl] {season}: {err}; trying Playwright")
        try:
            df = scrape_season_advanced(season)
        except RuntimeError as err:
            print(f"  [fail] {season}: {err}")
            failed.append(season)
            continue
        if df.empty:
            failed.append(season)
            continue
        print(f"  {season}: {len(df)} rows", flush=True)
        frames.append(df)
    if not frames:
        raise SystemExit("no season scraped")
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(out)} rows, seasons "
          f"{out['season'].min()}-{out['season'].max()})")
    if failed:
        print(f"failed seasons (rerun to retry from cache): {failed}")


if __name__ == "__main__":
    main()
