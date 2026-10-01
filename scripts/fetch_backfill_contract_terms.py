"""Extract Spotrac contract terms for the spotrac_fa_backfill salary rows.

ISSUES #55: the backfill rows in salaries.csv carry only one season, so
extend_contract_structure.py stores every backfill contract as one year. This
script reads the Spotrac "Contract Terms" field and the option years for the
contract that covers each backfill season. It writes a lookup table only; it
does not change targets or contract_structure_v2.csv.

Output: data/processed/spotrac_backfill_contract_terms.csv
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
from bs4 import BeautifulSoup

from config import PROCESSED_DIR
from scrape_spotrac_players import (
    PLAYER_CACHE, _is_contract_details, _is_contract_wrapper, _player_slug,
    _wrapper_season_span, norm, page_defect, parse_contracts, scrape_player,
    slugify,
)

SOURCE = "spotrac_fa_backfill"
OUT_PATH = PROCESSED_DIR / "spotrac_backfill_contract_terms.csv"
COLUMNS = [
    "player_name_norm", "season", "team", "spotrac_url", "contract_terms_raw",
    "contract_first_season", "contract_years", "option_type", "option_season",
    "total_value", "signing_type", "contract_type", "page_source",
]
ROW_SEASON_RE = re.compile(r"(\d{4})-\d{2}")
HEADING_SPAN_RE = re.compile(r"(\d{4})\s*[-–]\s*(\d{4})")


def _canonical_url(html_path: Path) -> str:
    """Return the canonical URL of a cached player page."""
    soup = BeautifulSoup(html_path.read_text(encoding="utf-8"), "html.parser")
    link = soup.find("link", rel="canonical")
    return link.get("href", "") if link else ""


def _terms_fields(terms: str) -> dict:
    """Parse years and total value from a "Contract Terms" string.

    Uses the same rules as parse_contracts: no year count from a day count,
    and no length above the 5-year CBA maximum (ISSUES #11).
    """
    out = {"contract_years": None, "total_value": None}
    if "day" not in terms.lower():
        m = re.match(r"(\d+)\s*yr", terms)
        if m and int(m.group(1)) <= 5:
            out["contract_years"] = int(m.group(1))
    m = re.search(r"\$[\d,]+", terms.replace(" ", ""))
    if m:
        out["total_value"] = int(m.group().replace("$", "").replace(",", ""))
    return out


def _blocks(html_path: Path) -> list[dict]:
    """Return every contract block on the page, newest first.

    parse_contracts drops blocks with an empty "Signed Using" value. Two-way
    and pending contracts have that empty value, so this reader keeps them.
    The season span comes from the <span class="years"> heading, which is
    present even when a "PENDING SIGNING" alert precedes it.
    """
    soup = BeautifulSoup(html_path.read_text(encoding="utf-8"), "html.parser")
    blocks = []
    for wrapper in soup.find_all("div", class_=_is_contract_wrapper):
        details = wrapper.find("div", class_=_is_contract_details)
        if details is None:
            continue
        years_el = wrapper.find("span", class_="years")
        span, contract_type = None, ""
        if years_el is not None:
            m = HEADING_SPAN_RE.search(years_el.get_text(" ", strip=True))
            if m:
                span = (int(m.group(1)), int(m.group(2)))
            small = years_el.find("small")
            if small is not None:
                contract_type = small.get_text(" ", strip=True)
        fields = {"terms": "", "signing_type": ""}
        for label in details.find_all("div", class_="label"):
            val = label.find_next_sibling("div", class_="value")
            val = val.get_text(strip=True) if val else ""
            ltext = label.get_text(strip=True)
            if ltext == "Contract Terms:":
                fields["terms"] = val
            elif ltext == "Signed Using:":
                fields["signing_type"] = val
        blocks.append({
            "wrapper": wrapper,
            "span": span,
            "heading_span": _wrapper_season_span(wrapper),
            "contract_type": contract_type,
            **fields,
            **_terms_fields(fields["terms"]),
        })
    return blocks


def _schedule(wrapper) -> list[tuple[int, str]]:
    """Return (season, status) for each year row of the block's cap table.

    Status is "player" or "team" for an option year, "ufa"/"rfa" for the
    free-agency row after the contract, "" for a normal contract year, and
    another lowercase label (for example "mutual") when Spotrac shows one.
    """
    table = wrapper.find("table") if wrapper is not None else None
    if table is None or table.find("tbody") is None:
        return []
    rows = []
    for tr in table.find("tbody").find_all("tr", recursive=False):
        cells = tr.find_all("td", recursive=False)
        if not cells:
            continue
        m = ROW_SEASON_RE.search(cells[0].get_text(" ", strip=True))
        if not m:
            continue
        # Spotrac marks option years with div.option: class option-player*
        # or option-club* (also -exercised/-declined) and text Player/Team.
        opt = tr.find("div", class_=lambda c: c and "option" in c.split())
        status = opt.get_text(strip=True).lower() if opt is not None else ""
        rows.append((int(m.group(1)), status))
    return rows


def _select_contract(html_path: Path, season: int):
    """Return (contract, block) for the newest contract covering `season`.

    parse_contracts is the primary reader. When no labelled contract covers
    the season, fall back to an unlabelled block whose heading span covers
    it. Return (None, None) when neither covers the season.
    """
    blocks = _blocks(html_path)
    for c in parse_contracts(html_path):
        if season not in c.get("seasons", []):
            continue
        block = next((b for b in blocks
                      if b["heading_span"] == c.get("season_span")
                      and b["terms"] == c.get("terms")), None)
        return c, block
    for b in blocks:
        if b["span"] and b["span"][0] <= season <= b["span"][1] \
                and not b["signing_type"]:
            return {"terms": b["terms"],
                    "contract_years": b["contract_years"],
                    "total_value": b["total_value"],
                    "signing_type": "",
                    "season_span": b["span"]}, b
    return None, None


def _refetch(url: str, slug: str) -> bool:
    """Replace a stale cached page with a live one.

    The old page is kept aside and restored when the fetch fails, so a failed
    fetch never removes the last valid cached page.
    """
    cache_path = PLAYER_CACHE / f"{slug}.html"
    backup = cache_path.with_suffix(".html.bak")
    cache_path.replace(backup)
    if scrape_player(url, slug) is None:
        backup.replace(cache_path)
        return False
    backup.unlink()
    return True


def resolve_row(name_norm: str, season: int, team: str, url: str | None):
    """Return (record, None) for a parsed row or (None, reason) otherwise."""
    slug = _player_slug(url) if url else slugify(name_norm)
    cache_path = PLAYER_CACHE / f"{slug}.html"

    if cache_path.exists():
        page_source = "cache"
        defect = page_defect(cache_path, expected_url=url)
        if defect is None and url is None:
            actual = _player_slug(_canonical_url(cache_path))
            if actual and actual != slug:
                defect = f"canonical slug {actual} does not match {slug}"
        if defect is not None:
            return None, f"cached page rejected: {defect}"
    else:
        page_source = "live"
        if url is None:
            return None, "no Spotrac URL and no cached page"
        # scrape_player waits SCRAPE_DELAY_SECONDS between requests and
        # writes only a page that passes page_defect.
        if scrape_player(url, slug) is None:
            return None, "live fetch failed or returned a defective page"

    contract, block = _select_contract(cache_path, season)
    if contract is None and page_source == "cache":
        if url is None:
            return None, (f"cached page has no contract for {season}; "
                          "no Spotrac URL to refetch")
        print(f"  {name_norm}: cached page has no {season} contract; "
              "refetching")
        if not _refetch(url, slug):
            return None, (f"cached page has no contract for {season}; "
                          "live refetch failed")
        page_source = "live"
        contract, block = _select_contract(cache_path, season)
    if contract is None:
        return None, f"no contract on the {page_source} page covers {season}"

    # _wrapper_season_span misses a heading that follows a PENDING alert.
    span = contract.get("season_span") or (block["span"] if block else None)
    schedule = _schedule(block["wrapper"] if block else None)
    if span is not None:
        schedule = [(s, st) for s, st in schedule if span[0] <= s <= span[1]]
    options = [(s, st) for s, st in schedule if st in ("player", "team")]
    if len(options) > 1:
        print(f"  {name_norm}: {len(options)} option years; recording last")
    option_type, option_season = (options[-1][1], options[-1][0]) if options \
        else ("none", None)

    years = contract.get("contract_years")
    contract_seasons = [s for s, st in schedule if st not in ("ufa", "rfa")]
    if years is not None and contract_seasons \
            and len(contract_seasons) != years:
        print(f"  {name_norm}: terms say {years} yr but the cap table lists "
              f"{len(contract_seasons)} contract seasons")

    return {
        "player_name_norm": name_norm,
        "season": season,
        "team": team,
        "spotrac_url": url or _canonical_url(cache_path),
        "contract_terms_raw": contract.get("terms", ""),
        "contract_first_season": span[0] if span else None,
        "contract_years": years,
        "option_type": option_type,
        "option_season": option_season,
        "total_value": contract.get("total_value"),
        "signing_type": contract.get("signing_type", ""),
        "contract_type": block["contract_type"] if block else "",
        "page_source": page_source,
    }, None


def main():
    """Resolve every backfill row, write the CSV, and check LeBron James."""
    sal = pd.read_csv(PROCESSED_DIR / "salaries.csv", encoding="utf-8")
    rows = sal[sal["source"] == SOURCE]
    urls = pd.read_csv(PROCESSED_DIR / "spotrac_player_urls.csv")
    url_map = dict(zip(urls["player_name_norm"], urls["url"]))
    print(f"Backfill rows: {len(rows)}")

    records, unresolved = [], []
    for _, r in rows.iterrows():
        name_norm = norm(r["player"])
        season = int(r["season"])
        rec, reason = resolve_row(name_norm, season, r["team"],
                                  url_map.get(name_norm))
        if rec is None:
            unresolved.append((name_norm, season, reason))
        else:
            records.append(rec)

    out = pd.DataFrame(records, columns=COLUMNS)
    out["contract_years"] = out["contract_years"].astype("Int64")
    out["option_season"] = out["option_season"].astype("Int64")
    out["contract_first_season"] = out["contract_first_season"].astype("Int64")
    out["total_value"] = out["total_value"].astype("Int64")
    out.to_csv(OUT_PATH, index=False, encoding="utf-8")

    print(f"\nRows parsed: {len(out)}")
    print(f"Rows unresolved: {len(unresolved)}")
    for name_norm, season, reason in unresolved:
        print(f"  {name_norm} {season}: {reason}")
    print("\ncontract_years distribution:")
    print(out["contract_years"].value_counts(dropna=False).sort_index()
          .to_string())
    print("\noption_type distribution:")
    print(out["option_type"].value_counts().to_string())
    print("\ncontract_type distribution:")
    print(out["contract_type"].value_counts().to_string())
    print("\npage_source distribution:")
    print(out["page_source"].value_counts().to_string())
    print(f"\nSaved: {OUT_PATH}")

    lebron = out[(out["player_name_norm"] == "lebron james")
                 & (out["season"] == 2026)]
    ok = (len(lebron) == 1
          and lebron["contract_years"].iloc[0] == 2
          and lebron["option_type"].iloc[0] == "player"
          and lebron["option_season"].iloc[0] == 2027)
    if not ok:
        print("\nCHECK FAILED: LeBron James 2026 does not read 2 years with a "
              "2027 player option:")
        print(lebron.to_string(index=False))
        sys.exit(1)
    print("\nCheck passed: LeBron James 2026 = 2 years, player option 2027.")


if __name__ == "__main__":
    main()
