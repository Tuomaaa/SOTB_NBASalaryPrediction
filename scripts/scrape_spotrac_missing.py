"""Scrape missing Spotrac player pages.

Strategy:
  1. WebSearch (google) "spotrac {name} NBA contract" to find player URL+ID
  2. Fetch the player page via Playwright (renders JS)
  3. Cache HTML to data/raw/html_cache/spotrac_players/{slug}.html

Handles Spotrac rate-limiting with automatic backoff.

Usage:
    python scripts/scrape_spotrac_missing.py          # scrape all missing
    python scripts/scrape_spotrac_missing.py --dry     # list missing players
    python scripts/scrape_spotrac_missing.py --limit 5 # scrape first 5 only
"""
import sys, io, time, re, unicodedata, argparse, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from pathlib import Path
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
from config import CACHE_DIR

PLAYER_CACHE = CACHE_DIR / "spotrac_players"
PLAYER_CACHE.mkdir(parents=True, exist_ok=True)

DELAY = 6
RATE_LIMIT_BACKOFF = 120

# Pre-built URL map from cached FA pages
_FA_URLS = None


def norm(name):
    name = str(name).strip().lower()
    nfkd = unicodedata.normalize("NFKD", name)
    out = "".join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", out.replace(".", "").replace("-", " ")).strip()


def slugify(name):
    return re.sub(r"[^a-z0-9-]", "", norm(name).replace(" ", "-"))


def _load_fa_urls():
    global _FA_URLS
    if _FA_URLS is not None:
        return _FA_URLS
    _FA_URLS = {}
    for year in range(2019, 2026):
        path = CACHE_DIR / f"spotrac_fa_{year}.html"
        if not path.exists():
            continue
        soup = BeautifulSoup(open(path, "r", encoding="utf-8").read(), "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "/nba/player/" in href:
                pname = norm(a.get_text(strip=True))
                if pname:
                    _FA_URLS[pname] = href
    return _FA_URLS


def _google_spotrac_url(display_name):
    """Use Google to find the Spotrac player URL. Returns URL or None."""
    import requests
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/125.0.0.0 Safari/537.36"
        )
    }
    q = f"site:spotrac.com {display_name} NBA contract"
    try:
        resp = requests.get(
            "https://www.google.com/search",
            params={"q": q},
            headers=headers,
            timeout=15,
        )
        # Google obfuscates URLs, look for spotrac pattern
        matches = re.findall(
            r"spotrac\.com/nba/player/_/id/(\d+)/([a-z0-9-]+)", resp.text
        )
        if matches:
            pid, slug = matches[0]
            return f"https://www.spotrac.com/nba/player/_/id/{pid}/{slug}"

        # Also check team-based URLs
        matches2 = re.findall(
            r"spotrac\.com/nba/[a-z-]+/([a-z-]+-\d+)", resp.text
        )
        if matches2:
            return f"https://www.spotrac.com/nba/player/_/id/{matches2[0].split('-')[-1]}/{'-'.join(matches2[0].split('-')[:-1])}"
    except Exception:
        pass
    return None


def get_missing_players():
    """Return list of (player_name_norm, display_name, known_url)."""
    import pandas as pd
    from src.model.train import load_training_data, _filter_year1, _filter_rookie_scale

    df = load_training_data()
    df = _filter_year1(df)
    df = _filter_rookie_scale(df)

    name_map = {}
    for _, row in df.drop_duplicates("player_name_norm").iterrows():
        pn = row.get("player_name", row["player_name_norm"])
        name_map[row["player_name_norm"]] = pn

    fa_urls = _load_fa_urls()

    missing = []
    for p in sorted(df["player_name_norm"].unique()):
        slug = slugify(p)
        if not (PLAYER_CACHE / f"{slug}.html").exists():
            url = fa_urls.get(p)
            missing.append((p, name_map.get(p, p), url))
    return missing


def _is_rate_limited(page):
    title = page.title()
    if "slow down" in title.lower():
        return True
    content = page.content()
    if len(content) < 5000 and "slow down" in content.lower():
        return True
    return False


def _fetch_page(page, url):
    """Navigate to URL, return (ok|rate_limited|empty, html)."""
    if not url.startswith("http"):
        url = "https://www.spotrac.com" + url
    page.goto(url, timeout=30_000)
    page.wait_for_load_state("domcontentloaded")
    time.sleep(3)
    if _is_rate_limited(page):
        return "rate_limited", None
    html = page.content()
    if len(html) < 1000:
        return "empty", None
    return "ok", html


def scrape_one(page, display_name, slug, known_url=None):
    """Scrape one player. Returns status string."""
    cache_path = PLAYER_CACHE / f"{slug}.html"
    if cache_path.exists():
        return "cached"

    # Step 1: find URL
    url = known_url
    url_source = "fa_cache"

    if not url:
        url = _google_spotrac_url(display_name)
        url_source = "google"

    if not url:
        return "no_url"

    # Step 2: fetch page
    status, html = _fetch_page(page, url)
    if status == "rate_limited":
        return "rate_limited"
    if not html:
        return "empty"

    cache_path.write_text(html, encoding="utf-8")
    return f"ok ({url_source})"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    missing = get_missing_players()
    has_url = sum(1 for _, _, u in missing if u)
    print(f"Missing: {len(missing)} players ({has_url} have FA URL, {len(missing)-has_url} need Google)")

    if args.dry:
        for pn, dn, url in missing:
            tag = "URL" if url else "GOOGLE"
            print(f"  [{tag:6s}] {pn:35s} ({dn})")
        return

    to_scrape = missing[:args.limit] if args.limit > 0 else missing
    print(f"Scraping {len(to_scrape)} players (delay={DELAY}s)")
    print(f"Estimated: {len(to_scrape) * (DELAY + 4) / 60:.0f} minutes\n")

    stats = {"ok": 0, "cached": 0, "no_url": 0, "empty": 0,
             "error": 0, "rate_limited": 0}

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1920, "height": 1080},
        )
        page = context.new_page()

        consecutive_rl = 0
        i = 0
        while i < len(to_scrape):
            pn, dn, url = to_scrape[i]
            slug = slugify(pn)

            try:
                result = scrape_one(page, dn, slug, url)
            except Exception as e:
                result = "error"
                print(f"    Exception: {e}")

            if result == "rate_limited":
                consecutive_rl += 1
                backoff = RATE_LIMIT_BACKOFF * consecutive_rl
                if consecutive_rl >= 5:
                    print(f"\n  Rate-limited {consecutive_rl}x. Stopping. Re-run later.")
                    break
                print(f"  [{i+1:3d}/{len(to_scrape)}] ! {dn:30s} -> RATE LIMITED, waiting {backoff}s...")
                time.sleep(backoff)
                continue

            consecutive_rl = 0
            key = "ok" if result.startswith("ok") else (result if result in stats else "error")
            stats[key] += 1

            sym = {"ok": "+", "cached": ".", "no_url": "?", "empty": "!"}.get(key, "X")
            print(f"  [{i+1:3d}/{len(to_scrape)}] {sym} {dn:30s} -> {result}")

            if result != "cached":
                time.sleep(DELAY)
            i += 1

        context.close()
        browser.close()

    total = len(list(PLAYER_CACHE.glob("*.html")))
    print(f"\nResults: {stats}")
    print(f"Cache: {total} player pages")


if __name__ == "__main__":
    main()
