"""Dated, ex-ante injury burden from cached Spotrac player pages.

Spotrac's injury table dates games missed, not the date of diagnosis.  A row is
therefore observable only from its first missed-game date.  Completed rows may
use the published games-missed total.  For a row still open at the decision
date, only one missed game is credited; using the eventual total would leak
future availability into a historical signing prediction.
"""

from __future__ import annotations

import html as html_lib
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from config import CACHE_DIR, PROCESSED_DIR


INJURY_EVENTS = PROCESSED_DIR / "spotrac_injuries.csv"
PLAYER_URLS = PROCESSED_DIR / "spotrac_player_urls.csv"
SIGNING_DATES = PROCESSED_DIR / "contract_signing_dates.csv"

INJURY_BLOCK_RE = re.compile(
    r'<div[^>]+id=["\']injuries["\'][^>]*>(.*?)(?=<div[^>]+class=["\'][^"\']*tab-pane|\Z)',
    re.I | re.S,
)
ROW_RE = re.compile(r"<tr\b[^>]*>(.*?)</tr>", re.I | re.S)
CELL_RE = re.compile(r"<td\b[^>]*>(.*?)</td>", re.I | re.S)
TAG_RE = re.compile(r"<[^>]+>")
DATE_RANGE_RE = re.compile(r"^(.+?)\s+-\s+(.+?)$")
PLAYER_ID_RE = re.compile(r"/id/(\d+)(?:/|$)", re.I)

SEVERE_RE = re.compile(
    r"achilles|acl|patellar|tendon|ligament|fracture|broken|surgery|spinal",
    re.I,
)
STRUCTURAL_RE = re.compile(
    r"knee|back|hip|foot|ankle|hamstring|groin|calf|shoulder|elbow|wrist|hand",
    re.I,
)
LOW_HEALTH_RE = re.compile(
    r"illness|concussion|soreness|sprain|strain|contusion|bruise|management",
    re.I,
)
NON_INJURY_RE = re.compile(
    r"rest|personal|suspension|coach|not with team|conditioning|protocol",
    re.I,
)


def _text(fragment: str) -> str:
    """Collapse one HTML fragment into plain text."""
    return re.sub(
        r"\s+", " ", html_lib.unescape(TAG_RE.sub(" ", fragment))
    ).strip()


def _parse_date(raw: str) -> pd.Timestamp | None:
    """Parse one date from the injury table."""
    value = pd.to_datetime(str(raw).strip(), format="mixed", errors="coerce")
    return None if pd.isna(value) else pd.Timestamp(value).normalize()


def classify_injury_reason(reason: str) -> tuple[str, float]:
    """Map a Spotrac reason to a fixed, clinically coarse severity weight.

    The mapping is deliberately broad.  It encodes recovery-risk families,
    without fitting weights to salary outcomes.
    """
    value = str(reason).strip()
    if NON_INJURY_RE.search(value):
        return "non_injury_dnp", 0.0
    if SEVERE_RE.search(value):
        return "severe_structural", 1.50
    if STRUCTURAL_RE.search(value):
        return "musculoskeletal", 1.15
    if LOW_HEALTH_RE.search(value):
        return "minor_or_management", 0.70
    return "other_health", 1.00


def parse_injury_page(path: Path, player_name_norm: str) -> pd.DataFrame:
    """Parse every injury/DNP row from one cached Spotrac player page."""
    columns = [
        "player_name_norm", "season_label", "team", "start_date",
        "end_date", "games_missed", "reason", "injury_category",
        "type_weight", "source_file",
    ]
    raw = path.read_text(encoding="utf-8", errors="replace")
    match = INJURY_BLOCK_RE.search(raw)
    if match is None:
        return pd.DataFrame(columns=columns)

    rows = []
    for row_html in ROW_RE.findall(match.group(1)):
        cells = [_text(cell) for cell in CELL_RE.findall(row_html)]
        if len(cells) < 5 or not re.fullmatch(r"\d{4}-\d{2}", cells[0]):
            continue
        date_match = DATE_RANGE_RE.match(cells[2])
        if date_match is None:
            start = end = _parse_date(cells[2])
        else:
            start = _parse_date(date_match.group(1))
            end = _parse_date(date_match.group(2))
        games_text = re.sub(r"[^0-9.]", "", cells[3])
        games = float(games_text) if games_text else np.nan
        category, weight = classify_injury_reason(cells[4])
        rows.append({
            "player_name_norm": player_name_norm,
            "season_label": cells[0],
            "team": cells[1],
            "start_date": start,
            "end_date": end,
            "games_missed": games,
            "reason": cells[4],
            "injury_category": category,
            "type_weight": weight,
            "source_file": path.name,
        })
    return pd.DataFrame(rows, columns=columns)


def _cache_path(url: str, player: str, cache_dir: Path) -> Path | None:
    """Resolve the current URL-table entry to its cached page."""
    match = PLAYER_ID_RE.search(str(url))
    if match:
        by_id = cache_dir / f"{match.group(1)}.html"
        if by_id.exists():
            return by_id
    from scripts.scrape_spotrac_players import slugify

    by_name = cache_dir / f"{slugify(player)}.html"
    return by_name if by_name.exists() else None


def build_injury_events(
    cache_dir: Path | None = None,
    player_urls: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Parse the latest mapped cache page for every known training player."""
    cache = cache_dir or (CACHE_DIR / "spotrac_players")
    urls = (
        pd.read_csv(PLAYER_URLS)
        if player_urls is None
        else player_urls.copy()
    )
    frames = []
    page_count = 0
    for row in urls.drop_duplicates("player_name_norm").itertuples(index=False):
        path = _cache_path(row.url, row.player_name_norm, cache)
        if path is None:
            continue
        page_count += 1
        parsed = parse_injury_page(path, row.player_name_norm)
        if not parsed.empty:
            frames.append(parsed)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not out.empty:
        out = out.drop_duplicates([
            "player_name_norm", "start_date", "end_date", "games_missed",
            "reason",
        ]).sort_values(["player_name_norm", "start_date", "reason"])
    stats = {
        "url_players": int(urls["player_name_norm"].nunique()),
        "cached_players": page_count,
        "players_with_events": int(out["player_name_norm"].nunique()) if not out.empty else 0,
        "events": len(out),
        "unparseable_start": int(out["start_date"].isna().sum()) if not out.empty else 0,
        "unparseable_end": int(out["end_date"].isna().sum()) if not out.empty else 0,
    }
    return out.reset_index(drop=True), stats


def _event_score(
    event: pd.Series,
    cutoff: pd.Timestamp,
    half_life_days: float,
    use_type_weight: bool,
    recurrence_count: int,
) -> float:
    """Score one event using only information observable by ``cutoff``."""
    start = pd.Timestamp(event["start_date"])
    end = pd.Timestamp(event["end_date"])
    completed = end < cutoff
    observed_games = float(event["games_missed"]) if completed else 1.0
    if not math.isfinite(observed_games):
        observed_games = 1.0
    impact = math.log1p(max(observed_games, 0.0)) / math.log(83.0)
    anchor = end if completed else start
    age_days = max((cutoff - anchor).days, 0)
    recency = 0.5 ** (age_days / half_life_days)
    type_weight = float(event["type_weight"]) if use_type_weight else 1.0
    recurrence_weight = 1.0 + 0.25 * min(recurrence_count, 3)
    return impact * recency * type_weight * recurrence_weight


def injury_score_as_of(
    events: pd.DataFrame,
    cutoff,
    half_life_days: float = 730.0,
    use_type_weight: bool = True,
    use_recurrence: bool = True,
) -> tuple[float, int, float]:
    """Return weighted burden, observable event count, and missed games.

    Events enter only after their first missed-game date.  Recurrence counts
    prior events in the same coarse category during the preceding three years.
    """
    when = pd.Timestamp(cutoff).normalize()
    prior = events[
        events["start_date"].notna()
        & events["end_date"].notna()
        & (events["start_date"] < when)
        & events["type_weight"].gt(0)
    ].sort_values("start_date")
    if prior.empty:
        return 0.0, 0, 0.0

    score = 0.0
    observed_games = 0.0
    history: list[pd.Series] = []
    for _, event in prior.iterrows():
        recurrence_count = 0
        if use_recurrence:
            window_start = pd.Timestamp(event["start_date"]) - pd.Timedelta(days=1095)
            recurrence_count = sum(
                old["injury_category"] == event["injury_category"]
                and pd.Timestamp(old["start_date"]) >= window_start
                for old in history
            )
        score += _event_score(
            event, when, half_life_days, use_type_weight, recurrence_count
        )
        completed = pd.Timestamp(event["end_date"]) < when
        games = float(event["games_missed"]) if completed else 1.0
        observed_games += games if math.isfinite(games) else 1.0
        history.append(event)
    return score, len(prior), observed_games


def attach_injury_history(
    df: pd.DataFrame,
    events: pd.DataFrame | None = None,
    signing_dates: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Attach fixed injury-score variants at each row's signing date."""
    from src.features.waiver_history import _choose_fallback_signing
    from scripts.parse_signing_dates import contract_spans, covering_contract

    out = df.copy()
    ev = (
        pd.read_csv(INJURY_EVENTS, parse_dates=["start_date", "end_date"])
        if events is None and INJURY_EVENTS.exists()
        else events
    )
    sd = (
        pd.read_csv(SIGNING_DATES, parse_dates=["signing_date"])
        if signing_dates is None and SIGNING_DATES.exists()
        else signing_dates
    )
    for name in (
        "injury_score_1y", "injury_score_2y", "injury_score_typed",
        "injury_score_recurrent", "injury_event_count", "injury_games_observed",
    ):
        out[name] = np.nan
    out["injury_history_known"] = 0.0
    out["injury_cutoff_date"] = pd.NaT
    if ev is None or sd is None or sd.empty:
        return out

    ev = ev.copy()
    ev["start_date"] = pd.to_datetime(ev["start_date"], errors="coerce")
    ev["end_date"] = pd.to_datetime(ev["end_date"], errors="coerce")
    sd = sd.copy()
    sd["signing_date"] = pd.to_datetime(sd["signing_date"], errors="coerce")
    events_by_player = {p: g for p, g in ev.groupby("player_name_norm")}
    signings_by_player = {p: g for p, g in sd.groupby("player_name_norm")}
    spans = contract_spans(sd)

    for idx, row in out.iterrows():
        player = str(row["player_name_norm"])
        season = int(row["season"])
        salary = float(row.get("salary", 0.0))
        signing = covering_contract(spans, player, season, salary=salary)
        cutoff = None if signing is None else signing["signing_date"]
        if cutoff is None or pd.isna(cutoff):
            candidates = signings_by_player.get(player)
            if candidates is not None:
                fallback = _choose_fallback_signing(candidates, season, salary)
                cutoff = None if fallback is None else fallback["signing_date"]
        if cutoff is None or pd.isna(cutoff):
            continue

        out.at[idx, "injury_history_known"] = 1.0
        out.at[idx, "injury_cutoff_date"] = pd.Timestamp(cutoff)
        player_events = events_by_player.get(player, ev.iloc[0:0])
        score_1y, count, games = injury_score_as_of(
            player_events, cutoff, half_life_days=365.0,
            use_type_weight=False, use_recurrence=False,
        )
        score_2y, _, _ = injury_score_as_of(
            player_events, cutoff, half_life_days=730.0,
            use_type_weight=False, use_recurrence=False,
        )
        typed, _, _ = injury_score_as_of(
            player_events, cutoff, half_life_days=730.0,
            use_type_weight=True, use_recurrence=False,
        )
        recurrent, _, _ = injury_score_as_of(
            player_events, cutoff, half_life_days=730.0,
            use_type_weight=True, use_recurrence=True,
        )
        out.at[idx, "injury_score_1y"] = score_1y
        out.at[idx, "injury_score_2y"] = score_2y
        out.at[idx, "injury_score_typed"] = typed
        out.at[idx, "injury_score_recurrent"] = recurrent
        out.at[idx, "injury_event_count"] = count
        out.at[idx, "injury_games_observed"] = games
    return out
