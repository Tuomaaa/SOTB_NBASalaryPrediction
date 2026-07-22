"""Scrape the 9 missing teams + historical per-player salaries from BBRef."""

import sys
import io
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import pandas as pd
from config import PROCESSED_DIR
from src.scraping.contracts import (
    scrape_team_contracts, scrape_historical_salaries, _add_cap_pct
)

MISSING_TEAMS = ["CHO", "CHI", "DEN", "HOU", "LAC", "MEM", "MIL", "NOP", "NYK"]

# Load existing data
existing = pd.read_csv(PROCESSED_DIR / "salaries.csv")
print(f"Existing: {len(existing)} rows, {existing['team'].nunique()} teams")

# 1) Scrape 9 missing teams
print(f"\n=== Scraping {len(MISSING_TEAMS)} missing teams ===")
new_team_frames = []
for team in MISSING_TEAMS:
    try:
        df = scrape_team_contracts(team)
        if not df.empty:
            new_team_frames.append(df)
            print(f"  {team}: {len(df)} rows")
    except Exception as e:
        print(f"  [error] {team}: {e}")

if new_team_frames:
    new_teams = pd.concat(new_team_frames, ignore_index=True)
    existing = pd.concat([existing, new_teams], ignore_index=True)
    existing = existing.drop_duplicates(subset=["player_url", "season"], keep="first")
    print(f"After adding missing teams: {len(existing)} rows, {existing['team'].nunique()} teams")

# 2) Historical per-player salary scrape
player_urls = existing["player_url"].dropna().unique().tolist()
player_urls = [u for u in player_urls if u]
print(f"\n=== Scraping salary history for {len(player_urls)} players ===")
history = scrape_historical_salaries(player_urls)
print(f"  Got {len(history)} historical rows")

if not history.empty:
    # Add player names from URL for historical rows
    history["player"] = history["player_url"]
    # Merge with existing to fill player names
    url_to_name = existing.dropna(subset=["player", "player_url"]).drop_duplicates("player_url")[["player_url", "player"]]
    history = history.drop(columns=["player"]).merge(url_to_name, on="player_url", how="left")

    combined = pd.concat([existing, history], ignore_index=True)
    combined = combined.drop_duplicates(subset=["player_url", "season"], keep="first")
else:
    combined = existing

# Re-add cap_pct
if "salary_cap" in combined.columns:
    combined = combined.drop(columns=["salary_cap", "cap_pct"])
combined = _add_cap_pct(combined)

out = PROCESSED_DIR / "salaries.csv"
combined.to_csv(out, index=False)
print(f"\n=== DONE ===")
print(f"Saved {len(combined)} rows to {out}")
print(f"Teams: {combined['team'].nunique()} — {sorted(combined['team'].dropna().unique())}")
print(f"Players: {combined['player'].nunique()}")
print(f"Seasons: {sorted(combined['season'].unique())}")
print(f"cap_pct range: {combined['cap_pct'].min():.3f} - {combined['cap_pct'].max():.3f}")
