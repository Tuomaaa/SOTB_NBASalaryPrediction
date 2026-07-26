"""Fetch Spotrac player transaction logs and build the waiver source table.

Raw HTML is cached under data/raw/html_cache/spotrac_players. The tracked,
reproducible artifact is data/processed/spotrac_transactions.csv.

    python scripts/build_waiver_features.py --fetch
    python scripts/build_waiver_features.py
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

from config import PROCESSED_DIR, SCRAPE_DELAY_SECONDS
from scripts.scrape_spotrac_players import (
    PLAYER_CACHE,
    scrape_player,
    slugify,
)
from src.features.waiver_history import (
    TRANSACTIONS,
    attach_waiver_history,
    build_transaction_events,
)


URLS = PROCESSED_DIR / "spotrac_player_urls.csv"


def _players(scope: str = "evaluation") -> list[str]:
    """Players needed for model training, or the complete scoring table."""
    if scope == "evaluation":
        from src.model.evaluate_suite import load_evaluation_frame
        df, _ = load_evaluation_frame(verbose=False)
    else:
        df = pd.read_csv(PROCESSED_DIR / "training_data_v2.csv")
    return sorted(df["player_name_norm"].dropna().astype(str).unique())


def _load_urls() -> dict[str, str]:
    if not URLS.exists():
        return {}
    df = pd.read_csv(URLS)
    return dict(zip(df["player_name_norm"], df["url"]))


def _save_urls(urls: dict[str, str]) -> None:
    pd.DataFrame(
        sorted(urls.items()), columns=["player_name_norm", "url"]
    ).to_csv(URLS, index=False)


def fetch_missing(players: list[str], limit: int | None = None) -> None:
    """Fetch uncached player pages with a delay after every live request."""
    urls = _load_urls()
    missing = [
        player for player in players
        if not (PLAYER_CACHE / f"{slugify(player)}.html").exists()
    ]
    if limit is not None:
        missing = missing[:limit]
    print(f"players {len(players)}, cached {len(players) - len(missing)}, "
          f"to fetch {len(missing)}")

    for i, player in enumerate(missing, 1):
        slug = slugify(player)
        url = urls.get(player, f"https://www.spotrac.com/nba/player/{slug}")
        urls[player] = url
        _save_urls(urls)
        page = scrape_player(url, slugify(player))
        if page is None:
            print(f"[{i}/{len(missing)}] {player}: FETCH FAILED; stopping")
            print("Rerun the same command after the source recovers.")
            break
        print(f"[{i}/{len(missing)}] {player}: OK")
        time.sleep(SCRAPE_DELAY_SECONDS)


def build() -> pd.DataFrame:
    """Parse cached pages and write the transaction and feature audit tables."""
    events = build_transaction_events(PLAYER_CACHE)
    TRANSACTIONS.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(TRANSACTIONS, index=False)
    print(
        f"wrote {TRANSACTIONS} ({len(events)} events, "
        f"{events['player_name_norm'].nunique() if len(events) else 0} players, "
        f"{int(events['event_type'].eq('waived').sum()) if len(events) else 0} waivers)"
    )

    training = pd.read_csv(PROCESSED_DIR / "training_data_v2.csv")
    audit = attach_waiver_history(training, transactions=events)
    known = int(audit["is_waived_known"].sum())
    positives = int(audit["is_waived"].fillna(0).sum())
    print(
        f"full-table feature coverage {known}/{len(audit)} "
        f"({known / max(len(audit), 1):.1%}), positives {positives}"
    )
    return events


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fetch", action="store_true",
        help="search and fetch uncached Spotrac player pages",
    )
    parser.add_argument(
        "--scope", choices=("evaluation", "all"), default="evaluation",
        help="fetch the model-training players (default) or every scoring player",
    )
    parser.add_argument(
        "--limit", type=int,
        help="fetch at most this many missing pages (for smoke tests)",
    )
    args = parser.parse_args()

    players = _players(args.scope)
    if args.fetch:
        fetch_missing(players, limit=args.limit)
    build()


if __name__ == "__main__":
    main()
