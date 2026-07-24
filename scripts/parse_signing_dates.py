"""Parse a signing date for every contract on the cached Spotrac player pages.

Writes `data/processed/contract_signing_dates.csv`, one row per dated signing
transaction:

    player_slug, player_name_norm, signing_date, contract_years, total_value,
    is_extension, contract_class, team, fa_year_matched, match_confidence,
    tx_text

**Where the dates are.** Only the transactions list carries them. The contract
blocks the rest of the pipeline reads (`Signed Using:`, `Contract Terms:`,
`Free Agent:` — see `scripts/scrape_spotrac_players.parse_contracts`) have no
date field at all: a scan of all 627 cached pages finds zero `Signed:` labels,
so `ul.player-transactions` is the only source and no "whichever covers more"
choice arises.

**Offline by construction.** This script never fetches. The cache is populated
by `scripts/scrape_spotrac_players.py` and `scripts/refresh_spotrac.py`, which
own the rate limits; a player with no cached page simply contributes no rows,
which is the required degradation — missing for that player, never a deleted
row elsewhere.

`fa_year_matched` links a transaction back to the contract blocks already
parsed into `spotrac_signing_types.csv`, matching on contract length and total
value. See `match_transactions` for the tolerance and the confidence ladder.

    python scripts/parse_signing_dates.py
    python scripts/parse_signing_dates.py --cache-dir /path/to/spotrac_players
"""

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

from config import CACHE_DIR, PROCESSED_DIR

# ─── HTML extraction ────────────────────────────────────────────────
# Regex rather than BeautifulSoup: the pages are ~900 KB each and only one
# small list is wanted, so a full parse of 627 of them costs minutes for
# nothing. The markup below is fixed template output, not authored HTML.
UL_RE = re.compile(r'<ul class="player-transactions[^"]*">(.*?)</ul>', re.S)
LI_RE = re.compile(r'<li class="list-group-item.*?</li>', re.S)
DATE_RE = re.compile(r'<strong class="link">(.*?)</strong>', re.S)
TEXT_RE = re.compile(r'<small class="d-block">(.*?)</small>', re.S)
TAG_RE = re.compile(r"<[^>]+>")

# "Signed a 3 year $103.6 million veteran contract extension with Denver (DEN)"
#
# Both the year count and the amount are optional: two-way, 10-day and
# Exhibit-10 entries carry neither. The match is anchored on the trailing
# "with <Team> (ABB)" rather than on the word "contract", because Spotrac drops
# that word on exactly the deals this task exists to find — Towns reads "Signed
# a 4 year $220.44 Designated Veteran Player extension with Minnesota (MIN)".
# Every optional-looking piece below is paid for by a real entry: "Signed to a
# ...", "$ 7.58 million" and "$$43.3 million" (stray spaces and a doubled
# sign), "1year", "$838k" / "$127.2M" (suffix instead of the word), and one
# "contract wit New Orleans" typo.
_HEAD = (r"^Signed\s*(?:to\s+)?(?:an?\s+)?"
         r"(?:(?P<years>\d+)\s*year\s+)?"
         r"(?:\$+\s*(?P<amt>[\d.,]+)\s*"
         r"(?P<unit>million|billion|thousand|[mkb])?\s+)?")
SIGN_RE = re.compile(
    _HEAD + r"(?P<mid>.*?)\s*\bwit[h]?\s+[A-Za-z .'-]+?\s*\(\s*[A-Z]{2,4}\s*\)",
    re.I,
)
# Fallback for entries whose team parenthetical is missing or malformed
# ("... contract with Sacramento", "... contract Golden State (GSW)"). Only
# applied when the text states an amount, so it cannot invent a contract out
# of an unstructured note.
SIGN_LOOSE_RE = re.compile(_HEAD + r"(?P<mid>.*)$", re.I)
TEAM_RE = re.compile(r"wit[h]?\s+([A-Za-z .'-]+?)\s*\(\s*([A-Z]{2,4})\s*\)")
UNIT = {"billion": 1e9, "b": 1e9,
        "million": 1e6, "m": 1e6,
        "thousand": 1e3, "k": 1e3}

# Below this a bare "$4.34" cannot be a literal dollar total — no NBA contract
# is worth $4.34 — so Spotrac dropped the word "million", which it does on
# ~1% of entries ("$132.43 maximum contract extension"). Amounts written with
# thousands separators ("$854,389") are literal and sit far above the line.
BARE_MILLIONS_MAX = 100_000


def _strip(fragment: str) -> str:
    return re.sub(r"\s+", " ", TAG_RE.sub(" ", fragment)).strip()


def read_transactions(cache_dir: Path) -> pd.DataFrame:
    """Every transaction entry on every cached page, dates unparsed."""
    rows = []
    pages = sorted(cache_dir.glob("*.html"))
    for path in pages:
        html = path.read_text(encoding="utf-8", errors="replace")
        block = UL_RE.search(html)
        if block is None:
            continue
        for order, li in enumerate(LI_RE.findall(block.group(1))):
            text = TEXT_RE.search(li)
            if text is None:
                continue
            date = DATE_RE.search(li)
            rows.append({
                "player_slug": path.stem,
                "tx_order": order,
                "date_raw": _strip(date.group(1)) if date else "",
                "text": _strip(text.group(1)),
            })
    return pd.DataFrame(rows), len(pages)


# ─── Field parsing ──────────────────────────────────────────────────
def parse_date(raw: str):
    """Spotrac writes 'Jul 06, 2025' everywhere; the abbreviated and dotted
    forms the brief warns about do not occur in this cache. Anything that does
    not parse is returned as None and counted, never silently dropped."""
    raw = raw.strip()
    for fmt in ("%b %d, %Y", "%B %d, %Y", "%b. %d, %Y", "%b %d %Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    return None


def parse_amount(amt: str | None, unit: str | None):
    """Total contract value in dollars, or None when the text states none."""
    if not amt:
        return None
    try:
        value = float(amt.replace(",", ""))
    except ValueError:
        return None
    if unit:
        return value * UNIT[unit.lower()]
    return value * 1e6 if value < BARE_MILLIONS_MAX else value


def classify(text: str) -> str:
    """Coarse contract class so consumers can filter the non-market rows.

    A guess, deliberately: 10-day, two-way and Exhibit-10 deals are real
    transactions with real dates but they are not negotiated annual prices, and
    the training filters already remove most of them as prorated.
    """
    low = text.lower()
    if "two-way" in low or "two way" in low:
        return "two-way"
    if "10-day" in low or "10 day" in low:
        return "10-day"
    if "exhibit 10" in low:
        return "exhibit-10"
    if "rest-of-season" in low or "rest of season" in low:
        return "rest-of-season"
    if "hardship" in low:
        return "hardship"
    if "rookie scale" in low:
        return "rookie-scale"
    if "2nd-round exception" in low or "second round exception" in low:
        return "second-round"
    return "standard"


def parse_signings(tx: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Signing transactions only, with years / value / extension / class."""
    stats = {"signed_entries": 0, "unmatched_text": 0, "unparseable_date": 0,
             "loose_match": 0, "unmatched_samples": []}
    rows = []
    for r in tx.itertuples(index=False):
        if not r.text.startswith("Signed"):
            continue
        stats["signed_entries"] += 1
        m = SIGN_RE.match(r.text)
        if m is None:
            loose = SIGN_LOOSE_RE.match(r.text)
            if loose is not None and loose.group("amt"):
                m = loose
                stats["loose_match"] += 1
            else:
                stats["unmatched_text"] += 1
                if len(stats["unmatched_samples"]) < 25:
                    stats["unmatched_samples"].append(r.text[:140])
                continue
        date = parse_date(r.date_raw)
        if date is None:
            stats["unparseable_date"] += 1

        total = parse_amount(m.group("amt"), m.group("unit"))

        team = TEAM_RE.search(r.text)
        rows.append({
            "player_slug": r.player_slug,
            "tx_order": r.tx_order,
            "signing_date": date,
            "contract_years": int(m.group("years")) if m.group("years") else None,
            "total_value": total,
            "is_extension": int("extension" in r.text.lower()),
            "contract_class": classify(r.text),
            "team": team.group(2) if team else None,
            "tx_text": r.text[:300],
        })
    return pd.DataFrame(rows), stats


# ─── Season and block matching ──────────────────────────────────────
def signing_season(date) -> float:
    """The NBA season a signing date belongs to, keyed by start year.

    A season runs October Y to June Y+1, and the free-agency window that feeds
    it opens the previous July, so July Y through June Y+1 all belong to season
    Y. July is the boundary: a July 6 signing is for the season starting that
    October, not the one that just ended.
    """
    if date is None or pd.isna(date):
        return np.nan
    return date.year if date.month >= 7 else date.year - 1


def slug_to_training_name() -> dict[str, str]:
    """Cache slugs back to the training data's player_name_norm.

    Built from the training side, exactly as `refresh_spotrac._slug_to_training_name`
    does: slugs drop dots and apostrophes, so reversing one produces a name that
    fails to merge.
    """
    from scripts.scrape_spotrac_players import slugify

    train = pd.read_csv(PROCESSED_DIR / "training_data_v2.csv")
    train = train.loc[:, ~train.columns.duplicated()]
    return {slugify(p): p for p in train["player_name_norm"].unique()}


def contract_blocks() -> pd.DataFrame:
    """One row per contract block in spotrac_signing_types.csv.

    That file is stored exploded to player-seasons; collapsing on the span keys
    recovers the blocks. `fa_year` is `contract_start + contract_years` because
    `refresh_spotrac` writes `contract_start = min(seasons)` and the anchored
    span is `[fa_year - n, fa_year - 1]`.
    """
    st = pd.read_csv(PROCESSED_DIR / "spotrac_signing_types.csv")
    keys = ["player_name_norm", "contract_years", "total_value", "contract_start"]
    blocks = st.dropna(subset=["contract_start"]).drop_duplicates(keys)[
        keys + ["signing_type", "aav"]
    ].copy()
    blocks["fa_year"] = blocks["contract_start"] + blocks["contract_years"]
    return blocks


def match_transactions(sign: pd.DataFrame, blocks: pd.DataFrame,
                       value_tol: float = 0.02) -> pd.DataFrame:
    """Attach `fa_year_matched` / `match_confidence` to each signing.

    A transaction and a block describe the same contract when they agree on
    length and money. The transaction text rounds ("$103.6 million" against the
    block's $103,608,840), so the value test is relative: within `value_tol`
    (2%) of the block value, which is looser than the worst rounding
    (0.5% at one significant decimal on a small deal) and far tighter than the
    gap between two different contracts for the same player.

    Confidence ladder, most to least certain:

      exact       — value within tolerance and years equal, uniquely
      value       — value within tolerance, no year count in the text
      ambiguous   — several blocks agree; the earliest fa_year is taken
      none        — money in the text, but no block agrees
      unmatchable — no money in the text to key on

    The value is required. Two-way, 10-day and Exhibit-10 entries state neither
    years nor dollars, so a year-only test passes against every block the player
    has and returns a confident-looking answer built from nothing; they are
    reported as `unmatchable` instead, which is also honest about the fact that
    those deals have no contract block to match.

    A tie is left flagged rather than resolved: the pages carry superseded
    shells of renegotiated deals whose anchors are unreliable (ISSUES #2), and
    guessing between them would launder that into a confident-looking date.
    """
    out_fa, out_conf = [], []
    by_player = {p: g for p, g in blocks.groupby("player_name_norm")}

    for r in sign.itertuples(index=False):
        has_y = r.contract_years is not None and not pd.isna(r.contract_years)
        has_v = r.total_value is not None and not pd.isna(r.total_value)
        cand = by_player.get(r.player_name_norm)

        if not has_v:
            out_fa.append(np.nan)
            out_conf.append("unmatchable")
            continue
        if cand is None or len(cand) == 0:
            out_fa.append(np.nan)
            out_conf.append("none")
            continue

        val_ok = ((cand["total_value"] - r.total_value).abs()
                  <= value_tol * cand["total_value"].abs())
        yr_ok = (cand["contract_years"] == r.contract_years) if has_y \
            else pd.Series(True, index=cand.index)

        hit = cand[yr_ok & val_ok]
        if len(hit) == 0:
            out_fa.append(np.nan)
            out_conf.append("none")
        elif len(hit) == 1:
            out_fa.append(float(hit["fa_year"].iloc[0]))
            out_conf.append("exact" if has_y else "value")
        else:
            out_fa.append(float(hit["fa_year"].min()))
            out_conf.append("ambiguous")

    sign = sign.copy()
    sign["fa_year_matched"] = out_fa
    sign["match_confidence"] = out_conf
    return sign


# ─── Spans, for the consumers ───────────────────────────────────────
# Final-year option annotation, e.g. "2026-27 Player Option", "2025 Club
# Option". The year is the START year of the option season (2026-27 -> 2026).
# Only options carrying a 4-digit year are usable; a bare "Player Option"
# (Durant 2017) or "3rd Year Club Option" (Millsap 2017) yields None, which is
# safe — those rows never need the anchor correction (see contract_spans).
OPTION_YEAR_RE = re.compile(r"(\d{4})(?:-\d{2})?\s+(?:Player|Club|Team)\s+Option",
                            re.I)


def _option_year(tx_text: str | None) -> float:
    """Latest final-year-option season stated in a transaction, or NaN."""
    if not tx_text:
        return np.nan
    years = [int(y) for y in OPTION_YEAR_RE.findall(tx_text)]
    return float(max(years)) if years else np.nan


def contract_spans(sd: pd.DataFrame | None = None) -> pd.DataFrame:
    """Dated contracts with the seasons they cover.

    Every consumer of this table needs the same derivation, so it lives here
    once rather than in four analyses:

    1. Where a block matched, the span nominally starts at
       `fa_year_matched - years`. This is the only instrument that knows an
       extension takes effect after it is signed — Towns signed July 2022 for a
       span starting 2024.

       **Option-aware anchor.** When the deal's final year is a player/club
       option, Spotrac's `fa_year` is the option-DECISION summer, which sits one
       season inside the nominal end, so `[fa - years, fa - 1]` starts the deal
       a year early (Embiid 2023 read as "2 of 4" when $47.6M = 0.35x the 2023
       cap is the supermax's year 1). The correction keys on the option season
       carried in the transaction text: when a final-year option year exceeds
       the nominal span end, the span slides forward to end ON that option
       season, `[option - years + 1, option]`. The test `option_year >
       naive_end` self-restricts to exactly the deals Spotrac anchored on the
       option — a fresh signing's or a correctly-anchored extension's final-year
       option already lands on the nominal end (Trae's 2021 extension, Randle's
       2021 extension), so it is left untouched.

    2. Where nothing matched, the contract is assumed to start in its own
       signing season — EXCEPT an unmatched extension, which begins paying the
       season AFTER it is signed (`signing_season + 1`); Durant 2026 and
       Holmgren 2026 are first paying years that read as "2 of n" without this.
       `span_source` keeps the distinction visible.

    3. **Where block-anchor and date disagree, the date wins.** A block anchor
       earlier than the signing date describes a contract that began before it
       existed; the superseded shells of renegotiated deals ISSUES #2 warns the
       Free-Agent anchor cannot be trusted on. Resolving them to the signing
       season is what makes Butler 2019 read as year 1 again.

    `is_reneg` flags a renegotiation-and-extend: its signing season is a FRESH
    price even though an older deal's span still covers it (Markkanen 2024,
    Turner 2022), so a consumer must treat `(player, signing_season)` as fresh
    regardless of the covering span. The renegotiation re-prices the current
    season and adds later years, so the extension span this row carries does not
    itself cover the repriced season; the flag is how a consumer recovers it.

    Season `S` is a *fresh price* when `span_start == S` (or S is renegotiated-
    fresh), whether or not the ink dried that summer; it is a continuation when
    `span_start < S`.
    """
    if sd is None:
        sd = pd.read_csv(PROCESSED_DIR / "contract_signing_dates.csv",
                         parse_dates=["signing_date"])
    s = sd[sd["contract_years"].notna() & sd["signing_season"].notna()].copy()
    s["contract_years"] = s["contract_years"].astype(int)
    s["signing_season"] = s["signing_season"].astype(int)

    matched = s["fa_year_matched"].notna()
    is_ext = s["is_extension"].fillna(0).astype(int) == 1
    s["option_year"] = s["tx_text"].map(_option_year)

    yrs = s["contract_years"]
    fa = pd.to_numeric(s["fa_year_matched"], errors="coerce")
    naive_start = fa - yrs
    naive_end = fa - 1
    # option-aware slide: end the span on the option season when the option
    # sits beyond the nominal end (matched deals only; fallback rows have no fa)
    opt_shift = (matched & s["option_year"].notna()
                 & (s["option_year"] > naive_end))
    block_anchor = np.where(opt_shift, s["option_year"] - yrs + 1, naive_start)

    # fallback: signing season, +1 for an unmatched extension (fix 2)
    fallback_anchor = np.where(is_ext, s["signing_season"] + 1,
                               s["signing_season"])

    anchor = pd.to_numeric(pd.Series(np.where(matched, block_anchor,
                                              fallback_anchor), index=s.index),
                           errors="coerce")
    s = s[anchor.notna()].copy()
    matched = matched[s.index]
    opt_shift = opt_shift[s.index]
    anchor = anchor[anchor.notna()].astype(int)

    s["option_shifted"] = opt_shift.values
    s["span_conflict"] = anchor < s["signing_season"]
    s["span_start"] = np.where(s["span_conflict"], s["signing_season"], anchor)
    s["span_end"] = s["span_start"] + s["contract_years"] - 1
    s["span_source"] = np.where(
        s["span_conflict"], "date-resolved",
        np.where(opt_shift.values, "option-shifted",
                 np.where(matched.values, "block", "assumed")))
    s["is_reneg"] = s["tx_text"].str.contains("renegotiat", case=False,
                                              na=False)
    s["aav"] = s["total_value"] / s["contract_years"]
    # summers between signing and the contract taking effect; >= 2 is the
    # early-supermax class that no award window anchored on the start season
    # can reach (early_supermax.csv exists for exactly this)
    s["early_gap"] = s["span_start"] - s["signing_season"]
    return s


def renegotiated_seasons(spans: pd.DataFrame) -> set[tuple[str, int]]:
    """`(player_name_norm, season)` pairs re-priced by a renegotiation.

    A renegotiation-and-extend bumps the current (signing) season's salary to a
    fresh figure, so that season is a fresh price even though an older deal's
    span still covers it. See `contract_spans` `is_reneg`.
    """
    r = spans[spans["is_reneg"]]
    return set(zip(r["player_name_norm"], r["signing_season"].astype(int)))


def covering_contract(spans: pd.DataFrame, player: str, season: int,
                      salary: float | None = None):
    """The dated contract covering (player, season), or None.

    When several contracts touch one season — a mid-season buyout, or a deal
    superseded partway through — the one whose AAV explains the observed pay
    wins, the same disambiguation `diagnostics.attach_signing_labels` uses.
    """
    cand = spans[spans["player_name_norm"] == player]
    hit = cand[(cand["span_start"] <= season) & (cand["span_end"] >= season)]
    if len(hit) == 0:
        return None
    if salary is not None and hit["aav"].notna().any():
        return hit.loc[(hit["aav"] - salary).abs().fillna(np.inf).idxmin()]
    return hit.sort_values("signing_date").iloc[-1]


# ─── Entry point ────────────────────────────────────────────────────
COLUMNS = [
    "player_slug", "player_name_norm", "signing_date", "signing_season",
    "contract_years", "total_value", "is_extension", "contract_class", "team",
    "fa_year_matched", "match_confidence", "tx_text",
]


def build(cache_dir: Path) -> pd.DataFrame:
    tx, n_pages = read_transactions(cache_dir)
    print(f"pages read: {n_pages}, transaction entries: {len(tx)}")

    sign, stats = parse_signings(tx)
    print(f"signing entries: {stats['signed_entries']}, "
          f"loose fallback: {stats['loose_match']}, "
          f"text unparseable: {stats['unmatched_text']}, "
          f"date unparseable: {stats['unparseable_date']}")
    for s in stats["unmatched_samples"]:
        print(f"    UNPARSED: {s}")

    # Some pages list one transaction twice (Gordon's 2014 deal, Drummond's
    # 2012). Identical date, terms and wording is one contract, not two, and a
    # consumer counting contracts would otherwise double it.
    dup_keys = ["player_slug", "signing_date", "contract_years", "total_value",
                "tx_text"]
    n_dup = int(sign.duplicated(dup_keys).sum())
    sign = sign.drop_duplicates(dup_keys).reset_index(drop=True)
    print(f"duplicate transaction entries dropped: {n_dup}")

    name_map = slug_to_training_name()
    sign["player_name_norm"] = sign["player_slug"].map(name_map).fillna(
        sign["player_slug"].str.replace("-", " ", regex=False))
    sign["signing_season"] = [signing_season(d) for d in sign["signing_date"]]

    sign = match_transactions(sign, contract_blocks())
    print("\nmatch confidence (all signings):")
    print(sign["match_confidence"].value_counts().to_string())

    n_priced = int((sign["match_confidence"] != "unmatchable").sum())
    n_matched = int(sign["fa_year_matched"].notna().sum())
    print(f"\npriced signings (money stated): {n_priced}, "
          f"matched to a block: {n_matched} "
          f"({n_matched / max(n_priced, 1):.1%})")

    print("\ncontract class:")
    print(sign["contract_class"].value_counts().to_string())

    sign = sign.sort_values(["player_slug", "signing_date"],
                            na_position="last")
    return sign[COLUMNS]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", type=Path,
                    default=CACHE_DIR / "spotrac_players",
                    help="cached Spotrac player pages (raw HTML is gitignored, "
                         "so a fresh worktree must point at the main checkout)")
    ap.add_argument("--out", type=Path,
                    default=PROCESSED_DIR / "contract_signing_dates.csv")
    args = ap.parse_args()

    if not args.cache_dir.exists():
        raise SystemExit(f"no cache at {args.cache_dir}")

    df = build(args.cache_dir)
    df.to_csv(args.out, index=False)
    print(f"\nwrote {args.out} ({len(df)} rows)")


if __name__ == "__main__":
    main()
