"""Scrape Spotrac player pages for signing type data. Cache all HTML, 3s rate limit."""
import sys, io, time, re, unicodedata
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import requests
import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path

from config import CACHE_DIR, USER_AGENT, PROCESSED_DIR, SCRAPE_DELAY_SECONDS

PLAYER_CACHE = CACHE_DIR / "spotrac_players"
PLAYER_CACHE.mkdir(parents=True, exist_ok=True)


def norm(name):
    name = str(name).strip().lower()
    nfkd = unicodedata.normalize("NFKD", name)
    out = "".join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", out.replace(".", "").replace("-", " ")).strip()


def slugify(name):
    """Convert player name to Spotrac URL slug."""
    name = norm(name).replace(" ", "-")
    return re.sub(r"[^a-z0-9-]", "", name)


# ─── Step 1: Collect player URLs from cached FA pages ───────────────
def collect_fa_urls():
    """Extract player name → Spotrac URL from cached FA pages."""
    urls = {}
    for year in range(2019, 2026):
        path = CACHE_DIR / f"spotrac_fa_{year}.html"
        if not path.exists():
            continue
        soup = BeautifulSoup(open(path, "r", encoding="utf-8").read(), "html.parser")
        for table in soup.find_all("table"):
            for row in table.find_all("tr")[1:]:
                cells = row.find_all("td")
                for cell in cells:
                    link = cell.find("a", class_="link")
                    if link and link.get("href") and "/nba/player/" in link["href"]:
                        pname = norm(link.get_text(strip=True))
                        urls[pname] = link["href"]
    return urls


# ─── Step 2: Build URL for players not in FA tables ─────────────────
def search_spotrac_player(name):
    """Try Spotrac search API to find player URL."""
    search_url = "https://www.spotrac.com/search"
    try:
        resp = requests.get(
            search_url,
            params={"q": name, "sport": "nba"},
            headers={"User-Agent": USER_AGENT},
            timeout=15,
        )
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "html.parser")
            # Look for first player link
            for a in soup.find_all("a"):
                href = a.get("href", "")
                if "/nba/player/" in href:
                    return href
    except Exception:
        pass
    return None


# ─── Step 3: Scrape player pages ────────────────────────────────────
def scrape_player(url, slug, retries=3):
    """Fetch and cache a player page."""
    cache_path = PLAYER_CACHE / f"{slug}.html"
    if cache_path.exists():
        return cache_path

    for attempt in range(retries):
        try:
            resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
            if resp.status_code == 200:
                cache_path.write_text(resp.text, encoding="utf-8")
                return cache_path
            elif resp.status_code == 403:
                print(f"    403 for {slug}, skipping")
                return None
        except requests.exceptions.RequestException as e:
            print(f"    Retry {attempt+1}/{retries} for {slug}: {e}")
            time.sleep(5)
    return None


# ─── Step 4: Parse signing type from player page ────────────────────
def parse_contracts(html_path):
    """Extract contract signing types from a player page."""
    soup = BeautifulSoup(
        open(html_path, "r", encoding="utf-8").read(), "html.parser"
    )

    contracts = []
    signed_labels = soup.find_all(
        "div", class_="label",
        string=lambda t: t and "Signed Using" in t,
    )

    for sl in signed_labels:
        contract = {}
        # Walk up to contract container
        container = sl.parent
        for _ in range(5):
            if container.parent and container.parent.name not in ("body", "html", "[document]"):
                container = container.parent
                labels = container.find_all("div", class_="label")
                if len(labels) >= 3:
                    break

        for label in container.find_all("div", class_="label"):
            val_div = label.find_next_sibling("div", class_="value")
            val = val_div.get_text(strip=True) if val_div else ""
            ltext = label.get_text(strip=True)
            if ltext == "Signed Using:":
                contract["signing_type"] = val
            elif ltext == "Contract Terms:":
                contract["terms"] = val
                m = re.match(r"(\d+)\s*yr", val)
                if m:
                    contract["contract_years"] = int(m.group(1))
                m2 = re.search(r"\$[\d,]+", val.replace(" ", ""))
                if m2:
                    contract["total_value"] = int(m2.group().replace("$", "").replace(",", ""))
            elif ltext == "Average Salary:":
                raw = val.replace("$", "").replace(",", "")
                try:
                    contract["aav"] = int(raw)
                except ValueError:
                    pass

        if contract.get("signing_type"):
            contracts.append(contract)

    # Get career earnings for year mapping
    years = []
    tables = soup.find_all("table")
    # Find career earnings table (has 'Year', 'Age', 'CashTotal' or similar)
    for t in tables:
        headers = [th.get_text(strip=True) for th in t.find_all("th")]
        if "Year" in headers and "Age" in headers and any("Cash" in h for h in headers):
            for row in t.find_all("tr")[1:]:
                cells = [td.get_text(strip=True) for td in row.find_all("td")]
                if cells and cells[0].isdigit():
                    years.append(int(cells[0]))
            break

    # Map contracts to years (contracts are most-recent-first, years are chronological)
    # Work backwards from end of career
    year_idx = len(years)
    for c in contracts:
        n = c.get("contract_years", 1)
        start_idx = year_idx - n
        if start_idx >= 0:
            c["seasons"] = years[start_idx:year_idx]
        else:
            c["seasons"] = years[:year_idx]
        year_idx = start_idx

    return contracts


# ─── Main ───────────────────────────────────────────────────────────
def main():
    # Get all unique players from training data
    td = pd.read_csv(PROCESSED_DIR / "training_data_v2.csv")
    players = sorted(td["player_name_norm"].unique())
    # Also get player_name for search
    name_map = {}
    for _, row in td.drop_duplicates("player_name_norm").iterrows():
        pn = row.get("player_name", row.get("player_name_norm", ""))
        name_map[row["player_name_norm"]] = pn

    print(f"Training data players: {len(players)}")

    # Collect URLs from FA pages
    fa_urls = collect_fa_urls()
    print(f"URLs from FA pages: {len(fa_urls)}")

    # Match training data players to URLs
    url_map = {}
    missing = []
    for p in players:
        if p in fa_urls:
            url_map[p] = fa_urls[p]
        else:
            missing.append(p)

    print(f"Matched from FA pages: {len(url_map)}")
    print(f"Missing (need search): {len(missing)}")

    # Scrape matched players first
    scraped = 0
    failed = 0
    total = len(url_map) + len(missing)

    for i, (pname, url) in enumerate(url_map.items()):
        slug = slugify(pname)
        cache_path = PLAYER_CACHE / f"{slug}.html"
        if cache_path.exists():
            scraped += 1
            continue
        result = scrape_player(url, slug)
        if result:
            scraped += 1
        else:
            failed += 1
        if (scraped + failed) % 25 == 0:
            print(f"  Progress: {scraped + failed}/{total} (scraped={scraped}, failed={failed})")
        time.sleep(SCRAPE_DELAY_SECONDS)

    # Try search for missing players
    print(f"\nSearching for {len(missing)} missing players...")
    search_found = 0
    for i, pname in enumerate(missing):
        slug = slugify(pname)
        cache_path = PLAYER_CACHE / f"{slug}.html"
        if cache_path.exists():
            scraped += 1
            search_found += 1
            continue

        display_name = name_map.get(pname, pname)
        url = search_spotrac_player(display_name)
        if url:
            result = scrape_player(url, slug)
            if result:
                scraped += 1
                search_found += 1
                url_map[pname] = url
        else:
            failed += 1

        if (i + 1) % 25 == 0:
            print(f"  Search progress: {i+1}/{len(missing)} (found={search_found})")
        time.sleep(SCRAPE_DELAY_SECONDS)

    print(f"\nScraping complete: {scraped} scraped, {failed} failed")

    # Parse all cached player pages
    print("\nParsing signing types...")
    all_rows = []
    parsed = 0
    no_data = 0

    for pname in players:
        slug = slugify(pname)
        cache_path = PLAYER_CACHE / f"{slug}.html"
        if not cache_path.exists():
            continue

        contracts = parse_contracts(cache_path)
        if not contracts:
            no_data += 1
            continue

        parsed += 1
        for c in contracts:
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
    print(f"\nParsed: {parsed} players, {len(df)} player-season rows")
    print(f"No signing data: {no_data} players")

    if len(df) > 0:
        print(f"\nSigning type distribution:")
        print(df["signing_type"].value_counts().to_string())

        # Save
        out_path = PROCESSED_DIR / "spotrac_signing_types.csv"
        df.to_csv(out_path, index=False)
        print(f"\nSaved: {out_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
