"""Scrape per-season impact metrics from nbarapm.com.

Endpoints (POST with player_name, called from within Playwright page context):
  /search/DARKO?allow_empty=1  → per-season DPM (DARKO Plus-Minus)
  /search/lebron?allow_empty=1 → per-season LEBRON
  /search/LAKER_history?allow_empty=1 → per-season RAPM, BPM, WAR, usage, etc.

All endpoints return per-season data with year/season columns.
We batch player lookups inside a single Playwright browser session.
"""

import json
import sys
import time
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
from playwright.sync_api import sync_playwright
from tqdm import tqdm

from config import CACHE_DIR, PROCESSED_DIR, SEASONS


def _normalize_name(name: str) -> str:
    """Lowercase + strip diacritics: 'Nikola Jokić' → 'nikola jokic'."""
    name = name.strip().lower()
    nfkd = unicodedata.normalize("NFKD", name)
    return "".join(c for c in nfkd if not unicodedata.combining(c))

METRICS_CACHE = CACHE_DIR / "nbarapm_metrics"


def _fetch_player_metrics(page, player_name: str) -> dict:
    """Fetch DARKO, LEBRON, and RAPM for one player from within a page context."""
    return page.evaluate(
        """async (name) => {
            const out = {};
            const endpoints = ["DARKO", "lebron", "LAKER_history"];
            for (const ep of endpoints) {
                try {
                    const body = new URLSearchParams();
                    body.append("player_name", name);
                    const r = await fetch("/search/" + ep + "?allow_empty=1", {
                        method: "POST",
                        headers: {"Content-Type": "application/x-www-form-urlencoded"},
                        body: body.toString()
                    });
                    const data = await r.json();
                    out[ep] = Array.isArray(data) ? data : [];
                } catch(e) {
                    out[ep] = [];
                }
            }
            return out;
        }""",
        player_name,
    )


def scrape_player_impact(player_names: list[str], batch_size: int = 50) -> dict[str, pd.DataFrame]:
    """Scrape impact metrics for a list of players.

    Returns dict with keys 'darko', 'lebron', 'rapm' each containing a DataFrame.
    Caches raw JSON per player to avoid re-fetching.
    """
    METRICS_CACHE.mkdir(parents=True, exist_ok=True)

    all_darko, all_lebron, all_rapm = [], [], []
    uncached = []

    for name in player_names:
        cache_file = METRICS_CACHE / f"{name.lower().replace(' ', '_')}.json"
        if cache_file.exists():
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            all_darko.extend(data.get("DARKO", []))
            all_lebron.extend(data.get("lebron", []))
            all_rapm.extend(data.get("LAKER_history", []))
        else:
            uncached.append(name)

    if uncached:
        print(f"  Fetching metrics for {len(uncached)} uncached players...")
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

            for i, name in enumerate(tqdm(uncached, desc="Impact metrics")):
                try:
                    data = _fetch_player_metrics(page, name)
                    cache_file = METRICS_CACHE / f"{name.lower().replace(' ', '_')}.json"
                    cache_file.write_text(json.dumps(data), encoding="utf-8")

                    all_darko.extend(data.get("DARKO", []))
                    all_lebron.extend(data.get("lebron", []))
                    all_rapm.extend(data.get("LAKER_history", []))
                except Exception as e:
                    print(f"  [error] {name}: {e}")

                if (i + 1) % batch_size == 0:
                    time.sleep(2)

            browser.close()

    darko_df = pd.DataFrame(all_darko) if all_darko else pd.DataFrame()
    lebron_df = pd.DataFrame(all_lebron) if all_lebron else pd.DataFrame()
    rapm_df = pd.DataFrame(all_rapm) if all_rapm else pd.DataFrame()

    return {"darko": darko_df, "lebron": lebron_df, "rapm": rapm_df}


def build_impact_dataset(player_names: list[str]) -> pd.DataFrame:
    """Scrape and merge impact metrics into a single per-player-season DataFrame.

    Returns columns: player_name, season, darko_dpm, lebron, rapm, plus offense/defense splits.
    """
    metrics = scrape_player_impact(player_names)

    frames = []

    if not metrics["darko"].empty:
        df = metrics["darko"].copy()
        df = df.rename(columns={
            "season": "season", "player_name": "player_name", "nba_id": "nba_id",
            "dpm": "darko_dpm", "o_dpm": "darko_odpm", "d_dpm": "darko_ddpm",
            "dpm_rank": "darko_rank",
        })
        keep = ["player_name", "nba_id", "season", "darko_dpm", "darko_odpm", "darko_ddpm", "darko_rank"]
        df = df[[c for c in keep if c in df.columns]]
        df = df.drop_duplicates(subset=["player_name", "season"], keep="first")
        df["player_name_norm"] = df["player_name"].apply(_normalize_name)
        frames.append(df)

    if not metrics["lebron"].empty:
        df = metrics["lebron"].copy()
        df = df.rename(columns={
            "year": "season", "player_name": "player_name", "nba_id": "nba_id",
            "LEBRON": "lebron", "O-LEBRON": "lebron_off", "D-LEBRON": "lebron_def",
            "LEBRON_Rank": "lebron_rank",
        })
        keep = ["player_name", "nba_id", "season", "lebron", "lebron_off", "lebron_def", "lebron_rank"]
        df = df[[c for c in keep if c in df.columns]]
        df = df.drop_duplicates(subset=["player_name", "season"], keep="first")
        df["player_name_norm"] = df["player_name"].apply(_normalize_name)
        frames.append(df)

    if not metrics["rapm"].empty:
        df = metrics["rapm"].copy()
        # Filter to regular season only (season_type "RS" vs "PO" for playoffs)
        if "season_type" in df.columns:
            df = df[df["season_type"] == "RS"]
        # Traded players have multiple RS rows — keep the one with most minutes
        if "minutes" in df.columns and "player_name" in df.columns and "year" in df.columns:
            df = df.sort_values("minutes", ascending=False).drop_duplicates(
                subset=["player_name", "year"], keep="first"
            )
        df = df.rename(columns={
            "year": "season", "player_name": "player_name", "nba_id": "nba_id",
            "rapm": "rapm", "orapm": "rapm_off", "drapm": "rapm_def",
            "rapm_rank": "rapm_rank", "war": "war", "usage_pct": "usage_pct",
        })
        keep = ["player_name", "nba_id", "season", "rapm", "rapm_off", "rapm_def",
                "rapm_rank", "war", "usage_pct", "games", "minutes", "position",
                "age",
                "bpm", "ts_plus", "efg_plus", "fg3_plus", "threepar_plus",
                "ast_pct", "reb_pct", "stl_pct", "blk_pct", "tov_pct",
                "points_per_100", "ws_48", "ortg", "drtg",
                "team_abbreviation"]
        df = df[[c for c in keep if c in df.columns]]
        df["player_name_norm"] = df["player_name"].apply(_normalize_name)
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    # Merge on normalized name + season (handles diacritics + case mismatches)
    # Drop per-source player_name and nba_id before merge to avoid conflicts
    for i, df in enumerate(frames):
        suffix = f"_{i}"
        frames[i] = df.rename(columns={"player_name": f"player_name{suffix}", "nba_id": f"nba_id{suffix}"})

    combined = frames[0]
    for df in frames[1:]:
        combined = combined.merge(df, on=["player_name_norm", "season"], how="outer")

    # Consolidate player_name / nba_id from whichever source had it
    name_cols = [c for c in combined.columns if c.startswith("player_name_") and c != "player_name_norm"]
    id_cols = [c for c in combined.columns if c.startswith("nba_id_")]
    combined["player_name"] = combined[name_cols].bfill(axis=1).iloc[:, 0]
    combined["nba_id"] = combined[id_cols].bfill(axis=1).iloc[:, 0]
    combined = combined.drop(columns=name_cols + id_cols)

    # Filter to our target seasons
    combined = combined[combined["season"].isin(SEASONS)].reset_index(drop=True)

    # Reorder columns
    front = ["player_name", "player_name_norm", "nba_id", "season"]
    rest = [c for c in combined.columns if c not in front]
    combined = combined[front + rest]

    return combined


if __name__ == "__main__":
    import sys

    # Load player names from salary data
    salary_path = PROCESSED_DIR / "salaries.csv"
    if salary_path.exists():
        salaries = pd.read_csv(salary_path)
        player_names = sorted(salaries["player"].dropna().unique().tolist())
        print(f"Found {len(player_names)} players from salary data")
    else:
        print("No salary data found — using test players")
        player_names = ["Nikola Jokic", "Jayson Tatum", "Luka Doncic", "Shai Gilgeous-Alexander"]

    df = build_impact_dataset(player_names)
    out = PROCESSED_DIR / "impact_metrics.csv"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nSaved {len(df)} rows to {out}")
    print(f"Players: {df['player_name_norm'].nunique()}")
    print(f"Seasons: {sorted(df['season'].unique())}")
    print(f"\nSample (top 10 by DARKO DPM, 2024):")
    if "darko_dpm" in df.columns:
        top = df[df["season"] == 2024].nlargest(10, "darko_dpm")
        print(top[["player_name", "season", "darko_dpm", "lebron", "rapm"]].to_string(index=False))
