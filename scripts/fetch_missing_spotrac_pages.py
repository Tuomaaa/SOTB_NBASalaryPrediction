"""Fetch the Spotrac player pages the URL table names but the cache lacks.

Reads data/processed/spotrac_player_urls.csv, keeps only players who appear in
training_data_v2.csv, and fetches the ones with no cached page. Reuses
scrape_spotrac_players.scrape_player so the cache layout, rate limit and retry
policy stay in one place.

    python scripts/fetch_missing_spotrac_pages.py --dry-run
    python scripts/fetch_missing_spotrac_pages.py
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

from config import CACHE_DIR, PROCESSED_DIR
from scripts.scrape_spotrac_players import scrape_player

PLAYER_CACHE = CACHE_DIR / "spotrac_players"


def url_slug(url: str) -> str:
    return str(url).split("?")[0].rstrip("/").rsplit("/", 1)[-1].lower()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    urls = pd.read_csv(PROCESSED_DIR / "spotrac_player_urls.csv")
    td = pd.read_csv(PROCESSED_DIR / "training_data_v2.csv")
    wanted = set(td["player_name_norm"].dropna())
    urls = urls[urls["player_name_norm"].isin(wanted)].copy()
    urls["slug"] = urls["url"].map(url_slug)

    cached = {p.stem.lower() for p in PLAYER_CACHE.glob("*.html")}
    todo = urls[~urls["slug"].isin(cached) & (urls["slug"] != "")]
    print(f"in URL table and in training data: {len(urls)}")
    print(f"already cached                   : {len(urls) - len(todo)}")
    print(f"to fetch                         : {len(todo)}")
    if args.dry_run or todo.empty:
        return

    # scrape_player owns rate limiting, retry, and page validation. Callers do
    # not add sleeps here; keeping the policy in one place prevents bursts.
    ok = fail = 0
    for i, r in enumerate(todo.itertuples(), 1):
        if scrape_player(r.url, r.slug):
            ok += 1
        else:
            fail += 1
        if i % 20 == 0 or i == len(todo):
            print(f"  [{i}/{len(todo)}] ok={ok} fail={fail}", flush=True)
    print(f"\ndone: {ok} fetched, {fail} failed")
    if fail:
        print("Failures are usually throttling, not dead URLs — re-run "
              "before concluding a page does not exist.")


if __name__ == "__main__":
    main()
