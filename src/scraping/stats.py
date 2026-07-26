"""Scrape priced-season advanced stats from Basketball Reference.

One page per season at /leagues/NBA_{year}_advanced.html gives all players
(~750 rows) with BPM, VORP, WS, PER, USG%, TS%, and more.

The model keys a 2023 signing to the 2022-23 performance that priced it, so its
``season`` equals Basketball Reference's END year. Do not add one to the URL.
"""

import argparse
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
from bs4 import BeautifulSoup
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright
from tqdm import tqdm

from config import CACHE_DIR, SCRAPE_DELAY_SECONDS, SEASONS, USER_AGENT
from src.scraping.utils import _cache_path

BBREF_BASE = "https://www.basketball-reference.com"
_last_request_time = 0.0

KEEP_COLUMNS = {
    "name_display": "player",
    "age": "age",
    "team_name_abbr": "team",
    "pos": "position",
    "games": "games",
    "games_started": "games_started",
    "mp": "minutes",
    "per": "per",
    "ts_pct": "ts_pct",
    "usg_pct": "usg_pct",
    "ast_pct": "ast_pct",
    "ows": "ows",
    "dws": "dws",
    "ws": "ws",
    "ws_per_48": "ws_per_48",
    "obpm": "obpm",
    "dbpm": "dbpm",
    "bpm": "bpm",
    "vorp": "vorp",
}


def _parse_float(text: str) -> float | None:
    text = text.strip()
    if not text or text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def fetch_static_html(url: str) -> str:
    """Fetch one static BBRef table through Playwright, with caching."""
    path = _cache_path(url)
    if path.exists():
        return path.read_text(encoding="utf-8")

    global _last_request_time
    wait = SCRAPE_DELAY_SECONDS - (time.time() - _last_request_time)
    if wait > 0:
        time.sleep(wait)
    with sync_playwright() as playwright:
        try:
            browser = playwright.chromium.launch(headless=True)
        except PlaywrightError as chromium_error:
            try:
                browser = playwright.chromium.launch(
                    headless=True, channel="msedge"
                )
            except PlaywrightError as edge_error:
                raise RuntimeError(
                    "could not launch Playwright Chromium or Microsoft Edge"
                ) from edge_error
            print(f"  [warn] Playwright Chromium unavailable: {chromium_error}")
        context = browser.new_context(user_agent=USER_AGENT)
        page = context.new_page()
        page.route(
            "**/*",
            lambda route: route.abort()
            if route.request.resource_type in {"image", "stylesheet", "font", "media"}
            else route.continue_(),
        )
        response = page.goto(url, wait_until="domcontentloaded", timeout=30_000)
        _last_request_time = time.time()
        if response is None or response.status >= 400:
            status = "no response" if response is None else response.status
            browser.close()
            raise RuntimeError(f"HTTP {status} for {url}")
        html = page.content()
        browser.close()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return html


def scrape_season_advanced(season: int) -> pd.DataFrame:
    """Scrape the regular season priced by a model-season signing row.

    Model season 2024 represents a deal signed after 2023-24, and BBRef calls
    that statistics page ``NBA_2024_advanced.html``.
    """
    bbref_year = season
    url = f"{BBREF_BASE}/leagues/NBA_{bbref_year}_advanced.html"
    html = fetch_static_html(url)
    soup = BeautifulSoup(html, "lxml")

    table = soup.find("table", {"id": "advanced"})
    if table is None:
        print(f"  [warn] No advanced table for {season}")
        return pd.DataFrame()

    thead = table.find("thead")
    header_row = thead.find_all("tr")[-1] if thead else None
    if not header_row:
        return pd.DataFrame()

    col_stats = [th.get("data-stat", "") for th in header_row.find_all("th")]

    tbody = table.find("tbody")
    if not tbody:
        return pd.DataFrame()

    rows = []
    for tr in tbody.find_all("tr"):
        if tr.get("class") and "thead" in tr.get("class", []):
            continue
        cells = tr.find_all(["th", "td"])
        if len(cells) < 5:
            continue

        row = {}
        player_url = None
        for cell in cells:
            ds = cell.get("data-stat", "")
            if ds not in KEEP_COLUMNS:
                continue
            col_name = KEEP_COLUMNS[ds]
            text = cell.get_text(strip=True)

            if col_name == "player":
                row[col_name] = text
                link = cell.find("a")
                if link:
                    player_url = link.get("href", "")
            elif col_name in ("team", "position"):
                row[col_name] = text
            else:
                row[col_name] = _parse_float(text)

        if not row.get("player"):
            continue

        row["player_url"] = player_url
        row["season"] = season
        rows.append(row)

    return pd.DataFrame(rows)


def scrape_all_advanced(seasons: list[int] | None = None) -> pd.DataFrame:
    """Scrape advanced stats for all configured seasons."""
    if seasons is None:
        seasons = SEASONS
    frames = []
    for season in tqdm(seasons, desc="Advanced stats"):
        try:
            df = scrape_season_advanced(season)
            if not df.empty:
                frames.append(df)
        except Exception as e:
            print(f"  [error] Season {season}: {e}")
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)

    # Players traded mid-season appear multiple times (once per team + TOT).
    # Keep TOT only for that player-season. Keying on the player alone would
    # delete every non-traded season of anyone who was ever traded.
    is_total = combined["team"].astype(str).str.fullmatch(r"(?:TOT|\d+TM)")
    tot_keys = set(map(tuple, combined.loc[
        is_total, ["player_url", "season"]
    ].to_numpy()))
    keys = list(zip(combined["player_url"], combined["season"]))
    mask = [(key not in tot_keys) or total
            for key, total in zip(keys, is_total)]
    combined = combined[mask].reset_index(drop=True)

    return combined


def _merge_refreshed_advanced(
    existing: pd.DataFrame,
    refreshed: pd.DataFrame,
    requested_seasons: list[int],
) -> tuple[pd.DataFrame, list[int]]:
    """Replace successful seasons and preserve stale rows for failed ones."""
    successful = (
        set(refreshed["season"].astype(int).unique())
        if not refreshed.empty else set()
    )
    requested = set(requested_seasons)
    missing = sorted(requested - successful)
    existing_seasons = (
        set(existing["season"].astype(int).unique())
        if not existing.empty else set()
    )
    unavailable = sorted(set(missing) - existing_seasons)
    if unavailable:
        raise RuntimeError(
            "refresh failed with no stale data for seasons: "
            + ", ".join(map(str, unavailable))
        )

    if existing.empty:
        combined = refreshed.copy()
    else:
        stale = existing[~existing["season"].astype(int).isin(successful)]
        combined = pd.concat([stale, refreshed], ignore_index=True)
    return combined, missing


if __name__ == "__main__":
    from config import PROCESSED_DIR

    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--season", type=int, action="append",
        help="refresh one model priced-season; repeat for several seasons",
    )
    args = parser.parse_args()
    seasons = args.season or SEASONS
    refreshed = scrape_all_advanced(seasons)
    out = PROCESSED_DIR / "advanced_stats.csv"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    existing = pd.read_csv(out) if out.exists() else pd.DataFrame()
    try:
        df, stale_seasons = _merge_refreshed_advanced(
            existing, refreshed, seasons
        )
    except RuntimeError as error:
        raise SystemExit(f"{error}; existing file left unchanged") from error
    if stale_seasons:
        print(
            "  [warn] Kept stale advanced stats for seasons: "
            + ", ".join(map(str, stale_seasons))
        )
    df = df.sort_values(["season", "player"]).reset_index(drop=True)
    df.to_csv(out, index=False)
    print(f"\nSaved {len(df)} rows ({len(refreshed)} refreshed) to {out}")
    print(f"Seasons: {sorted(df['season'].unique())}")
    print(f"Players: {df['player'].nunique()}")
    print(f"\nTop 10 BPM (2024):")
    top = df[df["season"] == 2024].nlargest(10, "bpm")[
        ["player", "team", "bpm", "vorp", "usg_pct"]
    ]
    print(top.to_string(index=False))
