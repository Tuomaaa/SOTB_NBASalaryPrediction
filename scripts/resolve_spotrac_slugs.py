"""Resolve Spotrac URLs for players the FA-page harvest cannot reach.

`build_spotrac_urls.py` can only find players who appear in a cached Spotrac
free-agent table. Anyone with no contract running into 2027 — retired, out of
the league, fringe — is structurally absent: 171 players and 335 training rows,
weighted toward the older seasons.

This script closes that gap by CONSTRUCTING the URL from the player's name
(`spotrac.com/nba/player/<slug>`, the bare form 490 rows of the existing table
already use and which fetches fine) and validating what comes back.

It deliberately does NOT use Spotrac's search endpoint. That path is how
`redirect/player/<id>?ref=search` URLs got into the table, and two of them
resolved to the NFL players of the same name — Spotrac carries an A.J. Green
and a Cam Thomas in both leagues, and 17 rows of NFL salary reached
`spotrac_salaries.csv` before `page_defect` was wired in. A wrong-sport page
is titled with the right name, so only the canonical URL catches it. See
ISSUES #41.

Construction cannot succeed for a player Spotrac files under a legal name the
box scores do not use (`mo bamba` -> `mohamed-bamba`, `svi mykhailiuk` ->
`sviatoslav-mykhailiuk`). Those come back 404 and are reported, not guessed at.

--dry-run first measures the construction rule against the 623 players whose
URL is already known, so the expected hit rate is known before any request is
sent.

Usage:
    python scripts/resolve_spotrac_slugs.py --dry-run
    python scripts/resolve_spotrac_slugs.py
    python scripts/resolve_spotrac_slugs.py --write   # also update the URL table
"""

import argparse
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

from config import CACHE_DIR, PROCESSED_DIR
from scripts.scrape_spotrac_players import page_defect, scrape_player

PLAYER_CACHE = CACHE_DIR / "spotrac_players"
URLS = PROCESSED_DIR / "spotrac_player_urls.csv"
TRAINING = PROCESSED_DIR / "training_data_v2.csv"
BASE_URL = "https://www.spotrac.com/nba/player"
REDIRECT_MARK = "redirect/player"


def slugify(display_name: str) -> str:
    """Player name -> Spotrac's URL slug.

    Strips accents and apostrophes/periods, keeps internal hyphens, and joins
    on hyphens: "De'Aaron Fox" -> "deaaron-fox", "Collin Murray-Boyles" ->
    "collin-murray-boyles", "Craig Porter Jr." -> "craig-porter-jr".
    """
    n = unicodedata.normalize("NFKD", str(display_name).strip().lower())
    n = "".join(c for c in n if not unicodedata.combining(c))
    n = n.replace("'", "").replace("’", "").replace(".", "")
    n = re.sub(r"[^a-z0-9\- ]", "", n)
    return re.sub(r"[\s-]+", "-", n).strip("-")


def url_slug(url: str) -> str:
    if REDIRECT_MARK in str(url):
        return ""
    return str(url).split("?")[0].rstrip("/").rsplit("/", 1)[-1].lower()


def _display_names() -> dict[str, str]:
    td = pd.read_csv(TRAINING).dropna(subset=["player_name_norm"])
    td = td.drop_duplicates("player_name_norm")
    return dict(zip(td["player_name_norm"], td["player_name"]))


def calibrate(urls: pd.DataFrame, names: dict[str, str]) -> None:
    """How often does the construction rule reproduce a URL we already have?"""
    known = urls[urls["url"].str.contains("/nba/player/", na=False)].copy()
    known["actual"] = known["url"].map(url_slug)
    known["built"] = known["player_name_norm"].map(
        lambda k: slugify(names.get(k, k)))
    ok = known["actual"] == known["built"]
    print(f"construction rule vs {len(known)} known URLs: "
          f"{ok.sum()} match ({ok.mean():.1%})")
    miss = known[~ok].head(8)
    if len(miss):
        print("  misses (Spotrac files these under another name):")
        for r in miss.itertuples():
            print(f"    {r.player_name_norm:24s} built={r.built:26s} actual={r.actual}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--write", action="store_true",
                    help="add resolved URLs to spotrac_player_urls.csv")
    args = ap.parse_args()

    urls = pd.read_csv(URLS)
    names = _display_names()
    calibrate(urls, names)

    td = pd.read_csv(TRAINING).dropna(subset=["player_name_norm"])
    have = set(urls["player_name_norm"])
    missing = sorted(set(td["player_name_norm"]) - have)

    # Players whose cached page is not an NBA page: their URL is wrong, so
    # they need resolving too even though the table has a row for them.
    contaminated = []
    for r in urls.itertuples():
        slug = url_slug(r.url) or str(r.url).rsplit("/", 1)[-1].split("?")[0]
        p = PLAYER_CACHE / f"{slug}.html"
        if p.exists() and page_defect(p):
            contaminated.append(r.player_name_norm)

    targets = missing + contaminated
    print(f"\nno URL at all        : {len(missing)}")
    print(f"URL resolves to a bad page: {len(contaminated)}  {contaminated}")
    print(f"to resolve           : {len(targets)}")

    if args.dry_run:
        print("\ndry run — no requests sent.")
        return

    resolved, dead, defective = {}, [], []
    for i, key in enumerate(targets, 1):
        slug = slugify(names.get(key, key))
        if not slug:
            dead.append((key, "(empty slug)"))
            continue
        url = f"{BASE_URL}/{slug}"
        got = scrape_player(url, slug)
        path = PLAYER_CACHE / f"{slug}.html"
        if not got or not path.exists():
            dead.append((key, slug))
            continue
        defect = page_defect(path)
        if defect:
            # Never leave a wrong-sport or landing page in the cache: a later
            # run would short-circuit on its existence and report success.
            defective.append((key, slug, str(defect)[:50]))
            path.unlink()
            continue
        resolved[key] = url
        if i % 20 == 0 or i == len(targets):
            print(f"  [{i}/{len(targets)}] ok={len(resolved)} "
                  f"dead={len(dead)} defect={len(defective)}", flush=True)

    print(f"\nresolved  : {len(resolved)}")
    print(f"unreachable: {len(dead)}   (404 or throttled — re-run before "
          f"concluding the page does not exist)")
    print(f"rejected   : {len(defective)}")
    for k, s, why in defective:
        print(f"    {k:24s} {s:26s} {why}")

    if not args.write or not resolved:
        if resolved and not args.write:
            print("\n--write not given; URL table untouched.")
        return

    out = urls.copy()
    fixed = 0
    for k, u in resolved.items():
        hit = out["player_name_norm"] == k
        if hit.any():
            out.loc[hit, "url"] = u          # replace a bad URL
            fixed += 1
        else:
            out.loc[len(out)] = {"player_name_norm": k, "url": u}
    out = out.sort_values("player_name_norm").reset_index(drop=True)
    out.to_csv(URLS, index=False)
    print(f"\nwrote {URLS}: {len(urls)} -> {len(out)} rows "
          f"({len(resolved) - fixed} added, {fixed} corrected)")


if __name__ == "__main__":
    main()
