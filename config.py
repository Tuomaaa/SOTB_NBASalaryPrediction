from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REFERENCE_DIR = DATA_DIR / "reference"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

SEASONS = list(range(2019, 2027))  # 2019-20 through 2025-26

# Salary cap by season — keyed by START year (e.g. 2024 = 2024-25 season)
#
# These values are load-bearing: cap_pct is the model target, so a wrong cap
# silently rescales the target for a whole season. Each figure below is
# corroborated by max contracts landing on exactly 25/30/35% of it — see
# scripts/check_caps.py, which fails if a season stops reconciling.
CAP_BY_SEASON = {
    2018: 101_869_000,
    2019: 109_140_000,
    2020: 109_140_000,
    2021: 112_414_000,
    2022: 123_655_000,
    2023: 136_021_000,
    2024: 140_588_000,
    2025: 154_647_000,
    2026: 164_961_000,  # official (pr.nba.com); was the 166.0M projection — ISSUES #16
    2027: 174_300_000,  # projected
    2028: 183_000_000,  # projected
    2029: 192_150_000,  # projected
}

# CBA eras
CBA_NEW_ERA_SEASON = 2024  # 2023 CBA took effect for 2023-24 season

# Scraping
SCRAPE_DELAY_SECONDS = 3.0
REQUEST_TIMEOUT = 30
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0.0.0 Safari/537.36"
)
CACHE_DIR = RAW_DIR / "html_cache"
