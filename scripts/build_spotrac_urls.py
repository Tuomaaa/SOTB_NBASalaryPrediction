"""Build data/processed/spotrac_player_urls.csv from the cached Spotrac FA pages.

The table had no producer in the repo — only `build_waiver_features.py` read it,
nothing wrote it, the same orphaned-artifact shape CLAUDE.md flags for
`contract_structure_v2.csv`. This script is that producer.

Two defects in the old ad-hoc collector inside `scrape_spotrac_players.py` are
fixed here, and both were costing coverage:

  1. It read only `spotrac_fa_{year}.html` for `range(2019, 2026)`. The cache
     also holds `spotrac_fa_upcoming_{year}.html` for 2027-2031, a different
     filename pattern the loop never matched. Those files carry every player
     under contract into a future season — which is exactly the 2025/2026
     draft cohort and the extended-only veterans (Curry, Paul George) that
     the table was missing.
  2. It keyed on a local `norm()` that strips periods and turns hyphens into
     spaces, while `player_name_norm` everywhere else comes from
     `_normalize_name` (lowercase + NFKD, punctuation KEPT). Players like
     `collin murray-boyles` and `craig porter jr.` were present in the cache
     and simply failed to join.

Extend-only, following `extend_contract_structure.py`: every row already in the
table is carried byte-for-byte and the script hard-fails if one would move.
`build_waiver_features.py` consumes this file, so a silently changed URL is a
silently changed feature.

Direct `/nba/player/<slug>` links are preferred over the search-redirect form
(`redirect/player/<id>?ref=search`) when adding a new player — ISSUES #41 traced
16 junk cached pages to those redirects. Existing redirect rows are reported,
not rewritten; pass --repair-redirects to replace all of them. The repair
prefers a validated canonical URL from the player-page cache, then a direct
FA-page URL, then Spotrac's direct slug form.

Usage:
    python scripts/build_spotrac_urls.py              # dry run, prints coverage
    python scripts/build_spotrac_urls.py --write
    python scripts/build_spotrac_urls.py --write --repair-redirects
"""

import argparse
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
from bs4 import BeautifulSoup

from config import CACHE_DIR, PROCESSED_DIR, RAW_DIR
from scripts.scrape_spotrac_players import page_defect, slugify

URLS = PROCESSED_DIR / "spotrac_player_urls.csv"
TRAINING = PROCESSED_DIR / "training_data_v2.csv"
REDIRECT_MARK = "redirect/player"
PLAYER_CACHE = CACHE_DIR / "spotrac_players"
URL_OVERRIDES = RAW_DIR / "raw_external" / "spotrac_url_overrides.csv"


def _normalize_name(name: str) -> str:
    """The project-wide key. Identical to `build_dataset._normalize_name`.

    Lowercase, NFKD, drop combining marks. Punctuation is KEPT — hyphens,
    periods and apostrophes are part of the key.
    """
    name = str(name).strip().lower()
    nfkd = unicodedata.normalize("NFKD", name)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def collect_cached_urls() -> tuple[dict[str, str], dict[str, int]]:
    """Harvest player -> Spotrac URL from every cached FA page.

    Returns (url_map, per_file_counts). Both the settled `spotrac_fa_{year}`
    and the forward-looking `spotrac_fa_upcoming_{year}` patterns are read.
    """
    urls: dict[str, str] = {}
    counts: dict[str, int] = {}
    for path in sorted(CACHE_DIR.glob("spotrac_fa_*.html")):
        soup = BeautifulSoup(
            path.read_text(encoding="utf-8", errors="ignore"), "html.parser"
        )
        found = 0
        for table in soup.find_all("table"):
            for row in table.find_all("tr")[1:]:
                for cell in row.find_all("td"):
                    link = cell.find("a", class_="link")
                    if not (link and link.get("href")):
                        continue
                    if "/nba/player/" not in link["href"]:
                        continue
                    key = _normalize_name(link.get_text(strip=True))
                    if not key:
                        continue
                    found += 1
                    # First direct link wins; never let a later page
                    # downgrade a key that already resolved.
                    urls.setdefault(key, link["href"])
        counts[path.name] = found
    return urls, counts


def _slug(url: str) -> str:
    """The player slug at the end of a Spotrac URL.

    Spotrac serves the same player under two path forms — the bare
    `/nba/player/<slug>` the table was built on, and the newer
    `/nba/player/_/id/<id>/<slug>` the current FA pages link to. They resolve
    to the same page, so identity is the trailing slug, not the whole string.
    Search redirects (`redirect/player/<id>?ref=search`) carry no slug at all
    and return "".
    """
    if REDIRECT_MARK in url:
        return ""
    tail = str(url).split("?")[0].rstrip("/").rsplit("/", 1)[-1]
    return tail.lower()


def _cached_canonical(player: str) -> str | None:
    """Return a validated cached page's canonical NBA URL, when available."""
    path = PLAYER_CACHE / f"{slugify(player)}.html"
    if not path.exists() or page_defect(path):
        return None
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
    link = soup.find("link", rel="canonical")
    return link.get("href", "") if link else None


def _coverage(keys: set[str], season: int = 2026) -> tuple[int, int]:
    """How many of `season`'s training rows have a URL. Returns (have, total)."""
    if not TRAINING.exists():
        return (0, 0)
    td = pd.read_csv(TRAINING)
    rows = td[td["season"] == season].dropna(subset=["player_name_norm"])
    names = rows["player_name_norm"].unique()
    return (sum(n in keys for n in names), len(names))


def _load_overrides() -> dict[str, str]:
    """Sourced identity URLs for players search or FA pages resolve wrongly."""
    if not URL_OVERRIDES.exists():
        return {}
    rows = pd.read_csv(URL_OVERRIDES)
    required = {"player_name_norm", "url"}
    if not required.issubset(rows.columns):
        raise SystemExit(
            f"{URL_OVERRIDES.name} must contain {sorted(required)}"
        )
    if rows.duplicated("player_name_norm").any():
        raise SystemExit(f"{URL_OVERRIDES.name} has duplicate player keys")
    out = dict(zip(rows["player_name_norm"], rows["url"]))
    bad = {k: v for k, v in out.items() if "/nba/player/" not in str(v)}
    if bad:
        raise SystemExit(f"non-NBA player URL overrides: {bad}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="write the merged table (default: dry run)")
    ap.add_argument("--repair-redirects", action="store_true",
                    help="replace every existing search redirect with a "
                         "direct /nba/player/ URL")
    args = ap.parse_args()

    existing = pd.read_csv(URLS)
    old_map = dict(zip(existing["player_name_norm"], existing["url"]))
    print(f"existing table: {len(existing)} rows "
          f"({sum(REDIRECT_MARK in u for u in old_map.values())} search-redirect)")

    harvested, counts = collect_cached_urls()
    overrides = _load_overrides()
    harvested.update(overrides)
    print("\ncached FA pages:")
    for name, n in counts.items():
        print(f"  {name:36s} {n:5d} player links")
    print(f"  {'-> distinct players':36s} {len(harvested):5d}")

    new_keys = sorted(set(harvested) - set(old_map))
    print(f"\nnew players the cache can add: {len(new_keys)}")

    # Extend-only: an existing key keeps its URL. Identity is the slug, so a
    # bare-form/id-form pair is a format difference, not a disagreement.
    overlap = [(k, old_map[k], harvested[k]) for k in old_map if k in harvested]
    reformat = [c for c in overlap if _slug(c[1]) == _slug(c[2]) and c[1] != c[2]]
    redirect_rows = [(k, v) for k, v in old_map.items() if REDIRECT_MARK in v]
    repairable = [c for c in overlap if REDIRECT_MARK in c[1]]
    genuine = [c for c in overlap if c[0] not in overrides
               if _slug(c[1]) and _slug(c[1]) != _slug(c[2])]

    print(f"existing keys the cache also covers: {len(overlap)}")
    print(f"  same player, newer URL form  : {len(reformat)}")
    print(f"  search-redirect, replaceable : {len(repairable)}")
    print(f"  search-redirect, total       : {len(redirect_rows)}")
    print(f"  DIFFERENT player slug        : {len(genuine)}")
    if genuine:
        print("\nSLUG CONFLICTS — refusing to move these:")
        for k, a, b in genuine[:20]:
            print(f"  {k:28s} table={a}\n  {'':28s} cache={b}")
        raise SystemExit(
            f"{len(genuine)} existing URLs point at a different player than "
            f"the cache does. This table feeds build_waiver_features.py; "
            f"resolve by hand before writing."
        )

    merged = dict(old_map)
    for k in new_keys:
        merged[k] = harvested[k]
    merged.update(overrides)
    repaired = 0
    if args.repair_redirects:
        for k, _old in redirect_rows:
            new = (_cached_canonical(k) or harvested.get(k)
                   or f"https://www.spotrac.com/nba/player/{slugify(k)}")
            merged[k] = new
            repaired += 1

    have_before, total = _coverage(set(old_map))
    have_after, _ = _coverage(set(merged))
    print(f"\n2026 training-row coverage: {have_before}/{total} "
          f"({have_before / total:.0%})  ->  {have_after}/{total} "
          f"({have_after / total:.0%})")

    if not args.write:
        print("\ndry run — nothing written. Re-run with --write.")
        return

    out = (pd.DataFrame({"player_name_norm": list(merged),
                         "url": [merged[k] for k in merged]})
           .sort_values("player_name_norm")
           .reset_index(drop=True))

    # The carry-forward guarantee, asserted rather than trusted.
    check = dict(zip(out["player_name_norm"], out["url"]))
    for k, v in old_map.items():
        if k in overrides or (args.repair_redirects and REDIRECT_MARK in v):
            continue
        if check.get(k) != v:
            raise SystemExit(f"carry-forward violated for {k!r}: "
                             f"{v} -> {check.get(k)}")

    out.to_csv(URLS, index=False)
    print(f"\nwrote {URLS} — {len(existing)} -> {len(out)} rows"
          + (f", {repaired} redirects repaired" if repaired else ""))


if __name__ == "__main__":
    main()
