"""Scrape Spotrac player pages for signing type data. Cache all HTML, 3s rate limit."""
import sys, time, re, unicodedata
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import requests
import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path

from config import CACHE_DIR, USER_AGENT, PROCESSED_DIR, SCRAPE_DELAY_SECONDS

PLAYER_CACHE = CACHE_DIR / "spotrac_players"
PLAYER_CACHE.mkdir(parents=True, exist_ok=True)
_LAST_SPOTRAC_REQUEST_AT = None


def _wait_for_spotrac_request() -> None:
    """Keep every Spotrac request at least the configured delay apart."""
    global _LAST_SPOTRAC_REQUEST_AT
    now = time.monotonic()
    if _LAST_SPOTRAC_REQUEST_AT is not None:
        remaining = SCRAPE_DELAY_SECONDS - (now - _LAST_SPOTRAC_REQUEST_AT)
        if remaining > 0:
            time.sleep(remaining)
    _LAST_SPOTRAC_REQUEST_AT = time.monotonic()


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
        _wait_for_spotrac_request()
        resp = requests.get(
            search_url,
            params={"q": name, "sport": "nba"},
            headers={"User-Agent": USER_AGENT},
            timeout=15,
        )
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "html.parser")
            wanted = norm(name)
            redirect_match = False
            for a in soup.find_all("a", href=True):
                href = a.get("href", "")
                label = re.sub(r"\s*\([^)]*\).*$", "", a.get_text(" ", strip=True))
                if "/nba/player/" in href and norm(label) == wanted:
                    return href
                if "/redirect/player/" in href and norm(label) == wanted:
                    redirect_match = True
            if redirect_match:
                return f"https://www.spotrac.com/nba/player/{slugify(name)}"
    except Exception:
        pass
    return None


# ─── Step 3: Scrape player pages ────────────────────────────────────
def scrape_player(url, slug, retries=3):
    """Fetch, validate, and cache a player page with rate-limited retries."""
    cache_path = PLAYER_CACHE / f"{slug}.html"
    if cache_path.exists():
        defect = page_defect(cache_path, expected_url=url)
        if defect is None:
            return cache_path
        print(f"    Invalid cached page for {slug}: {defect}; refetching")
        cache_path.unlink()

    for attempt in range(retries):
        try:
            _wait_for_spotrac_request()
            resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
            if resp.status_code == 200:
                cache_path.write_text(resp.text, encoding="utf-8")
                defect = page_defect(cache_path, expected_url=url)
                if defect is None:
                    return cache_path
                print(f"    Invalid page for {slug}: {defect}")
                cache_path.unlink()
            elif resp.status_code in (403, 429):
                print(f"    HTTP {resp.status_code} for {slug}; "
                      f"retry {attempt + 1}/{retries}")
                if attempt + 1 < retries:
                    time.sleep(SCRAPE_DELAY_SECONDS * (2 ** (attempt + 1)))
            else:
                print(f"    HTTP {resp.status_code} for {slug}; "
                      f"retry {attempt + 1}/{retries}")
        except requests.exceptions.RequestException as e:
            print(f"    Retry {attempt+1}/{retries} for {slug}: {e}")
    return None


SEASON_SPAN_RE = re.compile(r"^\s*(\d{4})\s*[-–]\s*(\d{4})\b")
NBA_PLAYER_URL_RE = re.compile(r"spotrac\.com/nba/player/", re.I)
PLAYER_ID_RE = re.compile(r"/id/(\d+)(?:/|$)", re.I)


def _player_slug(url: str) -> str:
    """Trailing player slug from a direct Spotrac player URL."""
    if not NBA_PLAYER_URL_RE.search(url or "") or "/redirect/player/" in url:
        return ""
    return url.split("?")[0].rstrip("/").rsplit("/", 1)[-1].lower()


def page_defect(html_path, expected_url: str | None = None):
    """Why a cached player page is not usable, or None if it is fine.

    A fetch that lands somewhere other than the NBA player page it asked for
    is a FAILED fetch, not data. Two ways it has happened: the generic
    /nba landing page (ISSUES #41 — sixteen of those, whose homepage markup
    the parser mined for 1,064 contract rows belonging to 173 unrelated
    players), and a same-name athlete in another sport, which the page title
    cannot catch because the title carries the name we asked for. Both are
    visible in the canonical URL, which names the sport and the player id.
    """
    soup = BeautifulSoup(
        open(html_path, "r", encoding="utf-8").read(), "html.parser"
    )
    link = soup.find("link", rel="canonical")
    url = link.get("href", "") if link else ""
    if not url:
        og = soup.find("meta", property="og:url")
        url = og.get("content", "") if og else ""
    if not url:
        return "no canonical URL"
    if not NBA_PLAYER_URL_RE.search(url):
        return f"canonical URL is not an NBA player page: {url}"
    if expected_url and NBA_PLAYER_URL_RE.search(expected_url):
        expected_id = PLAYER_ID_RE.search(expected_url)
        actual_id = PLAYER_ID_RE.search(url)
        if expected_id and actual_id:
            if expected_id.group(1) != actual_id.group(1):
                return ("canonical player id does not match requested URL: "
                        f"expected {expected_id.group(1)}, got "
                        f"{actual_id.group(1)}")
        else:
            expected_slug = _player_slug(expected_url)
            actual_slug = _player_slug(url)
            if (expected_slug and actual_slug
                    and expected_slug != actual_slug):
                return ("canonical player slug does not match requested URL: "
                        f"expected {expected_slug}, got {actual_slug}")
    return None


def _is_contract_details(cls):
    return bool(cls) and "contract-details" in cls


def _is_contract_wrapper(cls):
    return bool(cls) and "contract-wrapper" in cls


def _wrapper_season_span(wrapper):
    """The 'YYYY-YYYY' season span Spotrac prints as each contract's heading.

    Spotrac labels every contract block with the seasons it actually covered,
    in our own start-year convention ("2025-2027" = 2025-26 through 2027-28).
    That heading is a direct anchor and beats the derived one below wherever
    the two disagree — see parse_contracts.
    """
    m = SEASON_SPAN_RE.match(wrapper.get_text(" ", strip=True))
    if not m:
        return None
    start, end = int(m.group(1)), int(m.group(2))
    if not (1980 <= start <= 2100 and start <= end <= 2100):
        return None
    return start, end


# ─── Step 4: Parse signing type from player page ────────────────────
def parse_contracts(html_path):
    """Extract contract signing types from a player page.

    Season assignment reads the season span Spotrac prints in each contract
    block's heading ("2019-2019 Free Agent", "2021-2023 Free Agent"), clamped
    to the block's own nominal length. The derived anchor it replaced —
    [fa_year - n, fa_year - 1] from the "Free Agent:" field and the contract
    length — is kept as the fallback for the ~0.5% of blocks with no heading
    span, but it is wrong whenever those two fields count different things:
    "Contract Terms" counts GUARANTEED years while "Free Agent" is the year
    the player actually reached the market, so an option year or a
    non-guaranteed tail shifts the whole span late. Bobby Portis's 2019 NYK
    deal (1 yr guaranteed, team option, "Free Agent: 2021") landed on season
    2020 under the derived rule and on 2019 — correct — under the heading.
    143 of 2,528 cached blocks disagree, and every one spot-checked favours
    the heading.

    Both are better than the original implementation, which walked the career
    earnings table backwards and so assumed the listed contracts tile the
    career exactly — one missing or superseded deal shifted every assignment
    below it (Hassan Whiteside's $27M 2019 season landed on a minimum deal).

    Contracts appear newest-first, so a season already claimed by a more
    recent contract is never reassigned: when a player is waived and re-signs,
    the new deal wins the overlap and the abandoned tail of the old deal is
    dropped. Contracts without any anchor are chained immediately before the
    earliest season claimed so far (pre-2015 pages often omit the Free Agent
    field).

    A block whose "Signed Using" value is EMPTY — the normal state of a
    veteran extension, which uses no exception — yields no label but still
    takes part in the chain. It used to be discarded before season assignment,
    which broke the chain for every older contract beneath it: Josh Hart's
    Bird-Rights deal and Kevin Durant's Brooklyn extension both came back with
    no seasons at all because the extension above them had been removed.
    """
    soup = BeautifulSoup(
        open(html_path, "r", encoding="utf-8").read(), "html.parser"
    )

    contracts = []
    wrappers = [w for w in soup.find_all("div", class_=_is_contract_wrapper)
                if w.find("div", class_=_is_contract_details)]

    for wrapper in wrappers:
        contract = {"season_span": _wrapper_season_span(wrapper)}
        container = wrapper.find("div", class_=_is_contract_details)

        for label in container.find_all("div", class_="label"):
            val_div = label.find_next_sibling("div", class_="value")
            val = val_div.get_text(strip=True) if val_div else ""
            ltext = label.get_text(strip=True)
            if ltext == "Signed Using:":
                contract["signing_type"] = val
            elif ltext == "Contract Terms:":
                contract["terms"] = val
                # A 10-day deal's terms read "10 yr(s) / $151,821": Spotrac
                # mechanically appends "yr(s)" to a day count, so a naive year
                # parse records a phantom 10-YEAR contract (ISSUES #11). Guard
                # it two ways — never read a year count from a field that says
                # "day", and (at the append below) reject any parsed length
                # above the 5-year CBA maximum, whose 10-day money ($41k-$176k)
                # betrays the misread.
                if "day" not in val.lower():
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
            elif ltext == "Free Agent:":
                m = re.search(r"(20\d{2}|19\d{2})", val)
                if m:
                    contract["fa_year"] = int(m.group(1))

        # Reject contracts whose parsed length exceeds the 5-year CBA maximum:
        # these are 10-day deals misread as decade-long ones (ISSUES #11), and
        # their phantom spans (Tolliver "starting" in 1980) feed
        # _filter_continuations a spurious "starts earlier" signal.
        yrs = contract.get("contract_years")
        if not (yrs is not None and yrs > 5):
            contracts.append(contract)

    # Career earnings years — only needed as a last-resort anchor when the
    # newest contract has no Free Agent field.
    years = []
    for t in soup.find_all("table"):
        headers = [th.get_text(strip=True) for th in t.find_all("th")]
        if "Year" in headers and "Age" in headers and any("Cash" in h for h in headers):
            for row in t.find_all("tr")[1:]:
                cells = [td.get_text(strip=True) for td in row.find_all("td")]
                if cells and cells[0].isdigit():
                    years.append(int(cells[0]))
            break

    # Anchored contracts keep their FULL nominal span even when spans overlap:
    # a mid-season buyout puts two real contracts on one season (Westbrook
    # 2022-23 — supermax cash, then a minimum signing), and which one a
    # diagnostic wants depends on the salary being explained. Consumers
    # disambiguate by matching AAV against the observed salary; suppressing the
    # overlap here would silently pick one side and mislabel the other.
    cursor = None  # earliest anchored/assigned start so far, for chaining only
    for c in contracts:
        n = c.get("contract_years", 1)
        span = c.get("season_span")
        fa = c.get("fa_year")
        if span is not None:
            start, end = span
            # Never claim more seasons than the block's own nominal length, and
            # never claim past Spotrac's own span. Kemba Walker's Boston deal
            # heads "2019-2023" for 4 guaranteed years; the clamp keeps 2022 as
            # its last season instead of handing 2023 a sign-and-trade label
            # his rest-of-season minimum actually earned.
            if n:
                end = min(end, start + n - 1)
            # A heading span runs to the contract's last SCHEDULED season, which
            # includes an option year the player declined; "Free Agent: YYYY"
            # says when he actually reached the market, so his last paid season
            # is at most YYYY-1. Harrison Barnes's Dallas maximum heads
            # "2016-2019" and has "Free Agent: 2019" — he opted out in June
            # 2019 and re-signed in Sacramento, so its claim on 2019 would
            # outbid the Sacramento deal on AAV distance and label his SAC
            # re-signing with Dallas's cap-space mechanism.
            if fa is not None:
                end = min(end, fa - 1)
            c["seasons"] = list(range(start, end + 1))
        elif fa is not None:
            c["seasons"] = list(range(fa - n, fa))
        elif cursor is not None:
            c["seasons"] = list(range(cursor - n, cursor))
        elif years:
            c["seasons"] = years[-n:]
        else:
            c["seasons"] = []
        if c["seasons"]:
            cursor = min(c["seasons"]) if cursor is None else min(cursor, min(c["seasons"]))

    # Typeless blocks exist only to keep the chain honest (see docstring); they
    # carry no label and must not reach the signing-types table.
    return [c for c in contracts if c.get("signing_type")]


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
