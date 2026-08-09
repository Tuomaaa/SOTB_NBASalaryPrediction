"""Scrape per-season Pure RAPM data from nbarapm.com /scposs endpoint.

For each player, fetches all window sizes (1y–5y) of actual regularized
adjusted plus-minus computed from play-by-play data.  Results are cached
per-player and compiled into data/processed/pure_rapm.csv.
"""

import json
import sys
import time
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
from playwright.sync_api import sync_playwright
from tqdm import tqdm

from config import CACHE_DIR, PROCESSED_DIR

SCPOSS_CACHE = CACHE_DIR / "nbarapm_scposs"


def _normalize_name(name: str) -> str:
    name = name.strip().lower()
    nfkd = unicodedata.normalize("NFKD", name)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _fetch_scposs(page, player_name: str) -> list[dict]:
    """Fetch /scposs data for one player from within a Playwright page context."""
    return page.evaluate(
        """async (name) => {
            try {
                const body = new URLSearchParams();
                body.append("player_name", name);
                const r = await fetch("/scposs", {
                    method: "POST",
                    headers: {"Content-Type": "application/x-www-form-urlencoded"},
                    body: body.toString()
                });
                const data = await r.json();
                if (data.error) return [];
                const rows = Array.isArray(data) ? data : (data.rs || []);
                return rows;
            } catch(e) {
                return [];
            }
        }""",
        player_name,
    )


def _normalize_rows(rows: list[dict], player_name_norm: str) -> list[dict]:
    """Extract the fields we need from raw /scposs response."""
    out = []
    for row in rows:
        length = str(row.get("Year_Interval") or row.get("length_of_rapm") or "")
        length = "".join(c for c in length if c.isdigit())
        if length not in ("1", "2", "3", "4", "5"):
            continue

        ending_season = row.get("Latest_Year") or row.get("ending_season")
        if ending_season is None:
            continue

        out.append({
            "player_name_norm": player_name_norm,
            "season": int(ending_season),
            "rapm_window": int(length),
            "pure_rapm": float(row.get("OVR_RAPM") or row.get("rapm") or row.get("net_rapm") or 0),
            "pure_rapm_off": float(row.get("Off_RAPM") or row.get("off") or 0),
            "pure_rapm_def": float(row.get("Def_RAPM") or row.get("def") or 0),
            "rapm_poss": int(row.get("possessions") or row.get("Off_Poss") or 0),
        })
    return out


def scrape_pure_rapm(player_names: list[str]) -> pd.DataFrame:
    """Scrape Pure RAPM for all players, caching per-player."""
    SCPOSS_CACHE.mkdir(parents=True, exist_ok=True)

    all_rows = []
    uncached = []

    for name in player_names:
        norm = _normalize_name(name)
        cache_file = SCPOSS_CACHE / f"{norm.replace(' ', '_')}.json"
        if cache_file.exists():
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            all_rows.extend(_normalize_rows(data, norm))
        else:
            uncached.append((name, norm))

    if uncached:
        print(f"  Fetching Pure RAPM for {len(uncached)} uncached players...")
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            ctx = browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/125.0.0.0 Safari/537.36"
                )
            )
            page = ctx.new_page()
            page.goto(
                "https://nbarapm.com/player/nikola-jokic",
                wait_until="domcontentloaded",
                timeout=60_000,
            )
            time.sleep(10)

            for i, (name, norm) in enumerate(tqdm(uncached, desc="Pure RAPM")):
                try:
                    data = _fetch_scposs(page, name)
                    cache_file = SCPOSS_CACHE / f"{norm.replace(' ', '_')}.json"
                    cache_file.write_text(json.dumps(data), encoding="utf-8")
                    all_rows.extend(_normalize_rows(data, norm))
                except Exception as e:
                    print(f"  [error] {name}: {e}")

                if (i + 1) % 50 == 0:
                    time.sleep(2)

            browser.close()

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    return df


def main():
    td = pd.read_csv(PROCESSED_DIR / "training_data_v2.csv")
    players = td["player_name_norm"].dropna().unique().tolist()
    print(f"Scraping Pure RAPM for {len(players)} players...")

    df = scrape_pure_rapm(players)
    if df.empty:
        print("No data scraped.")
        return

    print(f"  {len(df)} total rows, {df['player_name_norm'].nunique()} players")
    print(f"  Seasons: {df['season'].min()}–{df['season'].max()}")
    print(f"  Windows: {sorted(df['rapm_window'].unique())}")

    out_path = PROCESSED_DIR / "pure_rapm.csv"
    df.to_csv(out_path, index=False)
    print(f"  Saved to {out_path}")


if __name__ == "__main__":
    main()
