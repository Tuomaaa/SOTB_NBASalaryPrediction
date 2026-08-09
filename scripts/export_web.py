"""Export every web-facing artifact from a single Grabit fit.

The portfolio site shows a valuation table, a per-player SHAP breakdown, and
diagnostic charts. All three must agree to the dollar: a waterfall that sums to
a different number than the row above it is worse than no waterfall at all.
So this script fits the model once and derives everything from that one fit —
it never reads the prediction CSVs in outputs/, which were produced by
different model versions on different days.

The post-Stage-1 chain (push, CBA clip, extension raise cap) is not
re-implemented here: it is `src.model.stages.compose`, the same function the
evaluation suite scores and predict.py deploys. Stage 3 reads the realized
route, so the forward R² written into meta.json is a TOLD-ROUTE number.

Two boards, two membership rules — decided here so the site never re-derives a
looser one (ISSUES #32):

  * The **Signing Board** reports a *pricing-accuracy* number (forward R², MAE,
    the signing-mechanism residuals). A row belongs only if the model was asked
    to price it — i.e. only if it survives the training filter chain itself
    (`_signing_membership`, the `is_signing` / `sg` flag). This drops the same
    classes training drops: rookie-scale years, a player's first two seasons
    (second-round and undrafted first contracts, priced by convention not by the
    market), prorated partial seasons, tier mislabels, and pre-2019
    continuations. The old rule `(year_in_contract == 1) & ~is_rookie_scale`
    admitted 612 such rows and lifted the advertised forward R² by +0.016.
  * The **Value Board** answers "is this contract a bargain?" and may keep rows
    the Signing Board rejects: a second-round pick on the minimum genuinely IS a
    surplus asset (kept, with a real surplus), and a free agent with no contract
    on file (`is_fa`, null surplus) belongs here too. The one class barred from
    its surplus *ranking* is the prorated mid-season signing (`is_prorated` /
    `pr`): its partial-season pay against a full-season prediction is an
    arithmetic artifact ($1.55M paid, shown at $23.79M "surplus"), so it carries
    a null surplus exactly as a free agent does and cannot top the bargain list.

Writes to the portfolio site's public/data/nba/ directory:
    valuations.json  — one row per player-season (table data, no SHAP)
    shap.json        — per-row feature attributions in dollars (lazy-loaded)
    meta.json        — CV metrics, feature list, caps, generated timestamp
    model.json       — stripped Grabit trees for in-browser what-if traversal
    charts/*.png     — diagnostics restyled to the site palette

Usage:
    python scripts/export_web.py [--out PATH]
"""

import argparse
import gzip
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from config import CAP_BY_SEASON, OUTPUTS_DIR, PROCESSED_DIR, RAW_DIR
from src.model import stages
from src.model.extension_cap import attach_extension_cap, attach_extension_value
from src.features.base_rating import attach_od_diffs
from src.features.playoff_minutes import attach_playoff_mpg
from src.features.waiver_history import (
    attach_waiver_interactions,
    attach_waiver_status_as_of,
    transaction_data_as_of,
)
from src.model.train import (
    FEATURE_COLS,
    PRORATED_FLOOR,
    TARGET,
    _compute_floor,
    _compute_max_eligible,
    _filter_continuations,
    _filter_mislabeled_year1,
    _filter_prorated,
    _filter_rookie_contracts,
    _filter_rookie_scale,
    _filter_year1,
    _load_rookie_scale_set,
    _normalize_vetmin_caphold,
    _prepare_Xy,
    load_training_data,
    train_grabit,
)

DEFAULT_OUT = (
    Path(__file__).resolve().parents[2] / "WebPage" / "public" / "data" / "nba"
)

# The model is trained on every season before this one, so the holdout season's
# signings are a genuine forward prediction — the model never saw them. That is
# what makes the Signing Board's accuracy claim honest: scoring the rows the
# model was fit on would understate its error by roughly a third. Everything at
# or after this season is out-of-sample; earlier seasons are the training data.
HOLDOUT_SEASON = 2026

# Site theme tokens, mirrored from WebPage/app/globals.css :root
PALETTE = {
    "bg": "#FAF9F5",
    "bg_soft": "#F2F0E9",
    "ink": "#191917",
    "ink_soft": "#57564F",
    "green": "#4E7A5A",
    "pink": "#C98CA0",
    "gold": "#B09840",
    "teal": "#4A8E96",
    "line": "#E4E1D7",
}

# Display labels for the 14 model features, used by the SHAP waterfall.
FEATURE_LABELS = {
    "darko_dpm_z": "DARKO DPM",
    "lebron_z": "LEBRON",
    "laker_z": "LAKER",
    "age": "Age",
    "age_squared": "Age²",
    "mpg": "Minutes / game",
    "availability_3yr": "Availability (3yr)",
    "usage_pct": "Usage rate",
    "height_inches": "Height",
    "cba_era": "CBA era",
    "ast_pct": "Assist %",
    "award_score_cum": "Awards",
    "draft_pick": "Draft pick",
    "is_waived": "Recently waived",
    "prev_cap_pct": "Previous contract",
    "mpg_x_waived": "Minutes x waived",
    "playoff_mpg_diff": "Playoff minutes swing",
}


# Spotrac writes signing mechanisms in a mix of styles, and refresh_spotrac.py
# has partially normalised them, so the same mechanism appears both as a
# hyphenated slug and in title case. Map every observed form to one canonical
# label. If a new one shows up, _load_signing_types raises rather than dropping
# it silently — add it here.
SIGNING_LABELS = {
    "Minimum": "Minimum",
    "Bird Rights": "Bird Rights",
    "Early Bird Rights": "Early Bird",
    "Non-Bird Rights": "Non-Bird",
    "Hardship": "Hardship",
    "cap-space": "Cap Space",
    "sign-and-trade": "Sign & Trade",
    "extend-and-trade": "Extend & Trade",
    "rookie-scale-exception": "Rookie Scale",
    "second-round-exception": "Second Round",
    "non-taxpayer-mid-level-exception": "Non-Taxpayer MLE",
    "Non-Taxpayer MLE": "Non-Taxpayer MLE",
    "taxpayer-mid-level-exception": "Taxpayer MLE",
    "Taxpayer MLE": "Taxpayer MLE",
    "room-mid-level-exception": "Room MLE",
    "bi-annual-exception": "Bi-Annual Exception",
    "Bi-Annual": "Bi-Annual Exception",
    "qualifying-offer": "Qualifying Offer",
    "disabled-player-exception": "Disabled Player",
}


def _model_version() -> str | None:
    """The vN.Mx tag this export was built from, via `git describe`.

    Returns e.g. "v8.10x" on a tagged commit, or "v8.10x+3" three commits past
    one — the "+N" is deliberate, so a site built from an untagged working
    state cannot silently claim to be the released version. None if the repo
    has no tags or git is unavailable, in which case the site falls back to the
    architecture name.
    """
    import subprocess

    try:
        out = subprocess.run(
            ["git", "describe", "--tags"],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    described = out.stdout.strip()
    if not described:
        return None
    # `v8.10x-3-gabc1234` -> `v8.10x+3`; a clean tag passes through unchanged.
    parts = described.split("-")
    if len(parts) >= 3 and parts[-1].startswith("g"):
        return f"{'-'.join(parts[:-2])}+{parts[-2]}"
    return described


def _row_key(name_norm: str, season: int) -> str:
    return f"{name_norm}|{int(season)}"


def _load_signing_types() -> pd.DataFrame:
    """Signing mechanism per player-season, deduplicated.

    spotrac_signing_types.csv has duplicate (player, season) keys — merging it
    raw inflates the row count. Keep the first record per key.
    """
    path = PROCESSED_DIR / "spotrac_signing_types.csv"
    if not path.exists():
        return pd.DataFrame(columns=["player_name_norm", "season", "signing_type"])
    st = pd.read_csv(path)
    st = st.drop_duplicates(subset=["player_name_norm", "season"], keep="first")
    unknown = set(st["signing_type"].dropna()) - set(SIGNING_LABELS)
    if unknown:
        raise SystemExit(f"unmapped signing mechanism(s): {sorted(unknown)}")
    st["signing_type"] = st["signing_type"].map(SIGNING_LABELS)
    return st[["player_name_norm", "season", "signing_type"]]


def _training_medians(df: pd.DataFrame) -> tuple[list[str], pd.Series]:
    """Feature list and fill values as seen by the fitted model.

    _prepare_Xy drops near-constant columns and fills NaN with the median of
    the *filtered* training set. Predicting the full dataset has to reuse both,
    or the full-set rows get imputed against a different distribution.

    The filter chain here must stay identical to the one inside train_grabit,
    or the medians come from a different row set than the model was fit on. Pass
    the same season-restricted df that was handed to train_grabit — the medians
    must reflect only the seasons the model actually saw.
    """
    tr = _normalize_vetmin_caphold(_filter_rookie_contracts(
        _filter_continuations(_filter_mislabeled_year1(
            _compute_max_eligible(_filter_prorated(_filter_rookie_scale(
                _filter_year1(df))))))))
    X_tr, _, _, features = _prepare_Xy(tr)
    return features, X_tr.median()


def _signing_membership(df: pd.DataFrame) -> set[tuple[str, int]]:
    """Signing-Board membership = the training filter chain, byte-for-byte.

    The Signing Board reports a *pricing-accuracy* number, so a row belongs on it
    only if the model was actually asked to price it — i.e. only if it survives
    the exact five-plus-one filter chain `evaluate_suite.load_evaluation_frame`
    and `train_grabit` apply. The board must not re-derive a looser rule of its
    own; that is ISSUES #32, where `(year_in_contract == 1) & ~is_rookie_scale`
    admitted 612 rows the chain rejects — second-round rookies the rookie-scale
    filter cannot see, prorated mid-season signings, and pre-2019 continuations —
    which lifted the advertised forward R² by +0.016 on rows the model never
    trained on. Run over every season (the holdout included) so 2026 members are
    decided by the same rule as 2019's.
    """
    chain = _normalize_vetmin_caphold(_filter_rookie_contracts(
        _filter_continuations(_filter_mislabeled_year1(
            _compute_max_eligible(_filter_prorated(_filter_rookie_scale(
                _filter_year1(df.copy()))))))))
    return set(zip(chain["player_name_norm"], chain["season"].astype(int)))


def _zscore_basis(df: pd.DataFrame) -> dict[str, dict[str, dict[str, float]]]:
    """Raw-impact mean/sd by season over the salary-joined player frame.

    This is the exact population `build_dataset` used to create the three
    impact z-scores. Using every row in impact_metrics.csv instead would put a
    browser-entered raw value on a different scale from the fitted model.
    """
    metrics = (("darko_dpm", "darko_dpm_z"),
               ("lebron", "lebron_z"),
               ("laker", "laker_z"))
    basis: dict[str, dict[str, dict[str, float]]] = {}
    for season, group in df.groupby("season"):
        season_out = {}
        for raw, z_col in metrics:
            values = group[raw].dropna()
            mean = float(values.mean())
            sd = float(values.std())
            expected = (group[raw] - mean) / sd
            present = expected.notna() & group[z_col].notna()
            drift = float(
                (expected[present] - group.loc[present, z_col]).abs().max()
            )
            if drift > 1e-10:
                raise SystemExit(
                    f"{raw} {int(season)} z-score basis drifts by {drift:.2e}"
                )
            season_out[raw] = {"mean": mean, "sd": sd}
        basis[str(int(season))] = season_out
    return basis


def _mle_by_season() -> dict[str, dict[str, float]]:
    """First-year exception amounts as cap percentages, by season."""
    mle = pd.read_csv(RAW_DIR / "raw_external" / "mle_exception_amounts.csv")
    required = {"non_taxpayer_mle", "taxpayer_mle", "room_mle", "bae"}
    observed = set(mle["exception_type"].unique())
    if observed != required:
        raise SystemExit(
            f"MLE table kinds changed: expected {sorted(required)}, "
            f"got {sorted(observed)}"
        )

    out: dict[str, dict[str, float]] = {}
    for season, group in mle.groupby("season"):
        season = int(season)
        if season not in CAP_BY_SEASON:
            raise SystemExit(f"No salary cap for MLE season {season}")
        if len(group) != len(required):
            raise SystemExit(f"MLE season {season} does not contain all four amounts")
        cap = float(CAP_BY_SEASON[season])
        out[str(season)] = {
            str(r.exception_type): float(r.amount_usd) / cap
            for r in group.itertuples(index=False)
        }
    return out


def _experience_years(df: pd.DataFrame) -> pd.Series:
    """The same draft-year/age-fallback experience instrument as the model."""
    from scripts.build_external_features import norm
    from src.model.train import _load_draft_years

    clean = df["player_name_norm"].apply(norm)
    return (df["season"] - clean.map(_load_draft_years())).fillna(
        (df["age"].fillna(25) - 19).clip(lower=0)
    ).astype(int).clip(lower=0)


def _display_floor_table(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Display-only floor lookup measured once on the salary training frame.

    `_compute_floor` is frame-relative: calling it on a what-if or free-agent
    prediction frame makes the target-derived at-floor population circular and
    can produce a negative floor. Do not use this table in model training or
    mutate `_compute_floor`; it exists only for browser route displays.
    """
    measured = _compute_floor(df.copy())
    experience = _experience_years(measured)
    measured["experience_bucket"] = pd.cut(
        experience, [-1, 2, 5, 9, 99], labels=["0-2", "3-5", "6-9", "10+"]
    )
    at_floor = measured[measured["is_at_floor"]]
    grouped = at_floor.groupby(
        ["season", "experience_bucket"], observed=True
    )[TARGET].median()
    fallback = {
        int(season): float(value)
        for season, value in at_floor.groupby("season")[TARGET].min().items()
    }
    overall = float(at_floor[TARGET].min())
    buckets = ("0-2", "3-5", "6-9", "10+")
    seasons = sorted(int(season) for season in measured["season"].unique())
    return {
        str(season): {
            bucket: float(grouped.get(
                (season, bucket), fallback.get(season, overall)
            ))
            for bucket in buckets
        }
        for season in seasons
    }


_TREE_ARRAYS = (
    "left_children", "right_children", "split_indices", "split_conditions",
    "default_left", "base_weights",
)


def _compact_split_condition(value: float) -> float:
    """Shortest decimal in (previous float32, value] for an exact JS split."""
    exact = float(np.float32(value))
    previous = float(np.nextafter(
        np.float32(exact), np.float32(-np.inf), dtype=np.float32
    ))
    if exact == 0:
        return 0.0
    exponent = int(np.floor(np.log10(abs(exact))))
    for significant in range(1, 10):
        candidate = float(f"{exact:.{significant}g}")
        if candidate > exact:
            step = 10.0 ** (exponent - significant + 1)
            candidate = float(f"{candidate - step:.{significant}g}")
        if previous < candidate <= exact:
            return candidate
    return exact


def _strip_model(model, features: list[str], medians: pd.Series,
                 zscore_basis: dict, dest: Path) -> dict:
    """Save m_final, then retain only the arrays required by JS traversal."""
    full_path = dest / ".grabit-full.tmp.json"
    try:
        model.save_model(full_path)
        raw = json.loads(full_path.read_text(encoding="utf-8"))
    finally:
        full_path.unlink(missing_ok=True)

    learner = raw["learner"]
    feature_names = learner["feature_names"]
    if feature_names != features:
        raise SystemExit(
            f"serialized feature order drifted: {feature_names} != {features}"
        )
    raw_trees = learner["gradient_booster"]["model"]["trees"]
    trees = []
    for tree in raw_trees:
        stripped = {key: tree[key] for key in _TREE_ARRAYS}
        # XGBoost compares float32 features to float32 thresholds internally.
        # Its 3.3 JSON text can parse just beyond that boundary in JS, so make
        # the threshold's training-time precision explicit once at export.
        # The browser traversal can then use the brief's exact rule:
        # Math.fround(feature) < split_conditions[node].
        stripped["split_conditions"] = [
            (_compact_split_condition(value)
             if stripped["left_children"][i] != -1 else value)
            for i, value in enumerate(stripped["split_conditions"])
        ]
        trees.append(stripped)
    return {
        # Keep the XGBoost representation intact. In current JSON it is a
        # string containing a JSON array; the browser parses it and takes [0].
        "base_score": learner["learner_model_param"]["base_score"],
        "feature_names": feature_names,
        "training_medians": {
            feature: float(medians[feature]) for feature in features
        },
        "zscore_basis": zscore_basis,
        "trees": trees,
    }


def _assert_model_parity(stripped: dict, out: pd.DataFrame,
                         tolerance: float = 1e-6) -> float:
    """Reproduce browser traversal with float32 split inputs for every row."""
    base = float(json.loads(stripped["base_score"])[0])
    predictions = np.empty(len(out), dtype=float)
    for i, values in enumerate(out["model_x"]):
        total = base
        for tree in stripped["trees"]:
            node = 0
            while tree["left_children"][node] != -1:
                value = float(np.float32(values[tree["split_indices"][node]]))
                if np.isnan(value):
                    node = (tree["left_children"][node]
                            if tree["default_left"][node]
                            else tree["right_children"][node])
                elif value < tree["split_conditions"][node]:
                    node = tree["left_children"][node]
                else:
                    node = tree["right_children"][node]
            total += tree["base_weights"][node]
        predictions[i] = total

    drift = float(np.max(np.abs(predictions - out["latent_cap_pct"].values)))
    if drift > tolerance:
        raise SystemExit(
            f"stripped-model float32 traversal drifts by {drift:.2e} "
            f"(limit {tolerance:.1e})"
        )
    return drift


def build_frame(df: pd.DataFrame, model, features: list[str],
                medians: pd.Series, train_df: pd.DataFrame
                ) -> tuple[pd.DataFrame, np.ndarray]:
    """Predict latent + bounded value for every player-season, with SHAP.

    The model is fit on the training seasons only, but it is applied to all
    rows here. Two kinds of row are outside that training domain: escalator and
    rookie-scale years (labelled Contract Surplus, not Signing Residual), and
    everything in HOLDOUT_SEASON and later, which the model never saw at all —
    those carry a genuine forward prediction and are flagged is_forward.

    The post-Stage-1 chain is `src.model.stages.compose`, the same function the
    evaluation suite and predict.py call, so the board cannot drift from the
    scored model. `train_df` is needed because the push's route classifier must
    be fit on exactly the seasons the regression saw.
    """
    import shap

    full = _compute_max_eligible(df.copy())
    full = _normalize_vetmin_caphold(full)
    full = _compute_floor(full)
    # Stage-3 inputs. A row is only an extension row when a dated span STARTS on
    # its season, so escalator years and rookie-scale years come back False and
    # Stage 3 is inert on them.
    full = attach_extension_cap(full)
    full = attach_extension_value(full)

    # Missingness indicator before fill (ISSUES #39).
    if "laker_z" in full.columns and "laker_known" not in full.columns:
        full["laker_known"] = full["laker_z"].notna().astype(int)

    X = full.reindex(columns=features).copy()
    X = X.fillna(medians).fillna(0)

    latent = model.predict(X)
    max_elig = full["max_eligible_pct"].values
    floor_pct = full["floor_pct"].values
    # Stage 2 is two-sided in BOTH directions now: the push lifts a max-worthy
    # player the model prices below his ceiling, the clip caps one priced above
    # it, and the floor lifts an at-minimum player to what the CBA guarantees.
    # Stage 3 then returns an extension row to its own raise cap.
    p_max = stages.deployed_p_max(stages.training_route_frame(train_df), full,
                                  medians=medians)
    is_ext = full["is_extension"].values
    ext_cap = full["ext_cap_pct"].values
    capped = stages.compose(latent, lo=floor_pct, hi=max_elig, p_max=p_max,
                            is_extension=is_ext, ext_cap_pct=ext_cap)
    flags = stages.bound_flags(latent, capped, lo=floor_pct, hi=max_elig,
                               p_max=p_max, is_extension=is_ext,
                               ext_cap_pct=ext_cap)

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X)
    expected = float(np.ravel(explainer.expected_value)[0])

    # Sanity: SHAP decomposes the *latent* (pre-cap) prediction.
    recon = shap_vals.sum(axis=1) + expected
    drift = np.abs(recon - latent).max()
    if drift > 1e-4:
        raise SystemExit(
            f"SHAP reconstruction drifts from prediction by {drift:.2e} — "
            "the waterfall would not sum to the displayed value."
        )

    cap = full["season"].map(CAP_BY_SEASON).astype(float)
    if cap.isna().any():
        missing = sorted(full.loc[cap.isna(), "season"].unique())
        raise SystemExit(f"No salary cap on record for season(s): {missing}")

    out = pd.DataFrame({
        "player_name": full["player_name"],
        "player_name_norm": full["player_name_norm"],
        "season": full["season"].astype(int),
        "team": full["team_abbreviation"],
        "position": full["position"],
        "age": full["age"],
        "experience": _experience_years(full).values,
        "year_in_contract": full["year_in_contract"],
        "contract_years": full["contract_years"],
        "actual_cap_pct": full[TARGET],
        "latent_cap_pct": latent,
        "pred_cap_pct": capped,
        "tier_ceiling_pct": full["tier_ceiling_pct"].values,
        "max_eligible_pct": max_elig,
        "ext_cap_pct": ext_cap,
        "ext_value_pct": full["ext_value_pct"].values,
        "floor_pct": floor_pct,
        "p_max": p_max,
        "cap": cap,
    })
    out["is_forward"] = out["season"] >= HOLDOUT_SEASON
    # A first-round pick's slotted years often carry year_in_contract == 1 in
    # the contract-structure data, which would put a rookie-scale salary on the
    # Signing Board as though a team had freshly negotiated it. Flag them so the
    # site can hold that board to the model's actual training domain.
    rookie = _load_rookie_scale_set()
    out["is_rookie_scale"] = [
        (n, s) in rookie for n, s in zip(out["player_name_norm"], out["season"])
    ]
    # Signing-Board membership IS the training filter chain — not a looser rule
    # re-derived here (ISSUES #32). A row carries a pricing-accuracy number only
    # if the model was asked to price it.
    members = _signing_membership(df)
    out["is_signing"] = [
        (n, s) in members for n, s in zip(out["player_name_norm"], out["season"])
    ]
    # Prorated partial-season pay (< 1.2% of the cap): a fraction of a year's
    # salary against a full-season prediction. Already outside is_signing (the
    # chain drops it); flagged here so the Value Board can also keep it out of the
    # surplus ranking, where its inflated "surplus" is an arithmetic artifact.
    out["is_prorated"] = out["actual_cap_pct"] < PRORATED_FLOOR

    out["actual_salary"] = out["actual_cap_pct"] * out["cap"]
    out["pred_salary"] = out["pred_cap_pct"] * out["cap"]
    out["latent_salary"] = out["latent_cap_pct"] * out["cap"]
    out["surplus"] = out["pred_salary"] - out["actual_salary"]
    # Prorated rows get no surplus number at all — the same null a free agent
    # carries — so a partial-season deal cannot top the bargain list (ISSUES #32).
    out.loc[out["is_prorated"], "surplus"] = np.nan
    # Which CBA bound, if any, moved the prediction off its latent value. There
    # are now two ceilings and an upward push, so `is_capped` can no longer be
    # read off the latent alone: a row the push pinned to the tier ceiling is
    # capped and the old test would miss it, and a row Stage 3 lowered sits at a
    # legal ceiling that is NOT the max. Four flags, one question each — see
    # stages.bound_flags.
    out["is_pushed"] = flags["is_pushed"]
    out["is_capped"] = flags["is_capped"]
    out["is_ext_capped"] = flags["is_ext_capped"]
    out["is_floored"] = flags["is_floored"]

    for col in ("darko_dpm_z", "lebron_z", "laker_z", "mpg", "usage_pct",
                "availability_3yr", "ast_pct", "height_inches", "draft_pick",
                "award_score_cum"):
        out[col] = full[col] if col in full.columns else np.nan

    st = _load_signing_types()
    out = out.merge(st, on=["player_name_norm", "season"], how="left")
    if len(out) != len(full):
        raise SystemExit("signing_type merge changed the row count")
    # Second-round exception signings are convention-priced ($2.3M), not market-
    # negotiated. The experience filter catches most first contracts (exp<=1), but
    # extending it to exp<=2 for second-round picks would wrongly drop real market
    # deals (Austin Reaves 2023, Herbert Jones 2023, etc.). Exclude by the Spotrac
    # signing-type label instead — it is exact, and only affects board membership.
    n_2nd = int((out["is_signing"] & (out["signing_type"] == "Second Round")).sum())
    if n_2nd:
        out.loc[out["signing_type"] == "Second Round", "is_signing"] = False
        print(f"  Second-round exception exclusion: {n_2nd} rows removed from "
              "signing-board membership")

    out["base_salary"] = expected * out["cap"]
    out["is_fa"] = False
    # Full-precision, median-filled inputs in serialized feature order. The
    # browser joins these to model.json.feature_names rather than assuming a
    # hard-coded index, so `_prepare_Xy` may still drop a near-constant column.
    out["model_x"] = list(X.to_numpy(dtype=float))
    return out, shap_vals, expected


def _add_free_agents(out: pd.DataFrame, shap_vals: np.ndarray, model,
                     features: list[str], medians: pd.Series, expected: float,
                     df: pd.DataFrame, train_df: pd.DataFrame
                     ) -> tuple[pd.DataFrame, np.ndarray]:
    """Append holdout-season free agents the salary data does not yet cover.

    A player can have a full season of impact metrics and no contract row: a
    free agent who has not re-signed, or one who signed so recently the salary
    scrape has not caught it. The salary is the model's target, so those rows
    fall out of training and out of build_frame — and a valuation board that
    exists to price players should not drop the very players whose price is the
    open question. These carry a market value (latent) with no actual salary to
    compare against, so they belong on the Value Board only and are flagged
    is_fa. The Signing Board, which measures accuracy against a real contract,
    excludes them.

    Impact metrics supply the three ratings, usage, minutes and age; the
    remaining features are borrowed from the player's most recent training row.
    The ratings are z-scored against the holdout season's *training* players —
    the same basis build_dataset used — so an FA's z sits on the same scale as
    everyone already on the board.
    """
    import shap

    im = pd.read_csv(PROCESSED_DIR / "impact_metrics.csv")
    im = im[im["season"] == HOLDOUT_SEASON].drop_duplicates("player_name_norm")

    have = set(out.loc[out["season"] == HOLDOUT_SEASON, "player_name_norm"])
    fa = im[~im["player_name_norm"].isin(have)].copy()

    # Same z-score basis as the model saw: the holdout season's training players.
    t_hold = df[df["season"] == HOLDOUT_SEASON]
    for raw, zc in (("darko_dpm", "darko_dpm_z"), ("lebron", "lebron_z"),
                    ("laker", "laker_z")):
        mu, sd = t_hold[raw].mean(), t_hold[raw].std()
        fa[zc] = (fa[raw] - mu) / sd
    fa = attach_od_diffs(fa)

    fa["mpg"] = fa["minutes"] / fa["games"].replace(0, np.nan)
    fa["cba_era"] = 1

    # Everything else comes from the player's most recent season on record.
    hist = df.sort_values("season").groupby("player_name_norm").last()
    fa = fa[fa["player_name_norm"].isin(hist.index)].copy()
    # Position is stable and worth borrowing; team is not — a free agent's last
    # team is stale the moment he signs elsewhere, so it is left blank.
    for col in ("age", "height_inches", "draft_pick", "prev_cap_pct",
                "award_score_cum", "ast_pct", "availability_3yr", "position"):
        borrowed = fa["player_name_norm"].map(hist[col]) if col in hist else np.nan
        if col == "age":
            # Prefer this season's age from impact metrics; fall back to history.
            fa["age"] = fa["age"].where(fa["age"].notna(), borrowed) \
                if "age" in fa.columns else borrowed
        else:
            fa[col] = borrowed
    fa["age_squared"] = fa["age"] ** 2
    waiver_as_of = transaction_data_as_of()
    fa = attach_waiver_status_as_of(fa, waiver_as_of)
    print(f"  Waiver feature as of {waiver_as_of.date()}: "
          f"{int(fa['is_waived'].fillna(0).sum())} free-agent positives")
    fa = attach_waiver_interactions(fa)
    # The holdout season's playoffs are over by the time this board is built, so
    # a free agent's playoff minutes are observed, not forecast. A player whose
    # team missed the playoffs keeps the neutral 0.0 difference.
    fa = attach_playoff_mpg(fa)
    print(f"  Playoff minutes attached: "
          f"{int(fa['po_mpg'].notna().sum())} of {len(fa)} free agents played")

    # Missingness indicator before fill (ISSUES #39).
    if "laker_z" in fa.columns and "laker_known" not in fa.columns:
        fa["laker_known"] = fa["laker_z"].notna().astype(int)

    X = fa.reindex(columns=features).fillna(medians).fillna(0)
    latent = model.predict(X)
    shap_fa = shap.TreeExplainer(model).shap_values(X)

    # An unsigned free agent has no contract, so Stage 3 is inert (no extension
    # span) and the DOWNWARD half of Stage 2 stays off as before: the Value
    # Board's job for these rows is market value, and a fringe player whose
    # value sits under the minimum is exactly the finding, not an error to
    # round away. The ceiling is real though, so it is computed and used — both
    # as the push's target and as a cap on the pushed value, because a push
    # without its clip can land a player above his legal max.
    #
    # `_compute_floor` is deliberately NOT called here: it derives floor_pct
    # from the frame's OWN at-floor rows, and on a frame whose `cap_pct` is the
    # model's own latent that lookup is circular (it returns a NEGATIVE floor
    # for Biyombo 2026). See the RESULT's ISSUES entry.
    probe = fa.copy()
    probe[TARGET] = latent
    probe = _compute_max_eligible(probe)
    probe = attach_extension_value(probe)
    max_elig = probe["max_eligible_pct"].values
    p_max = stages.deployed_p_max(stages.training_route_frame(train_df), probe,
                                  medians=medians)
    no_floor = np.full(len(fa), -np.inf)
    value = stages.stage2(latent, lo=no_floor, hi=max_elig, p_max=p_max)
    fa_flags = stages.bound_flags(latent, value, lo=no_floor, hi=max_elig,
                                  p_max=p_max)
    n_push = int(fa_flags["is_pushed"].sum())
    if n_push:
        who = ", ".join(
            f"{fa['player_name'].values[i]} P={p_max[i]:.2f}"
            for i in np.flatnonzero(fa_flags["is_pushed"]))
        print(f"  Free agents moved by the max push: {n_push} ({who})")

    cap = float(CAP_BY_SEASON[HOLDOUT_SEASON])
    fa_out = pd.DataFrame({
        "player_name": fa["player_name"].values,
        "player_name_norm": fa["player_name_norm"].values,
        "season": HOLDOUT_SEASON,
        "team": np.nan,
        "position": fa["position"].values,
        "age": fa["age"].values,
        "experience": _experience_years(probe).values,
        "year_in_contract": np.nan,
        "contract_years": np.nan,
        "actual_cap_pct": np.nan,
        "latent_cap_pct": latent,
        # Value Board shows market value: the push applies (it is ex ante, and
        # a max-worthy unsigned FA is the case it exists for), the ceiling caps
        # the pushed value, and the floor still does not apply.
        "pred_cap_pct": value,
        "tier_ceiling_pct": probe["tier_ceiling_pct"].values,
        "max_eligible_pct": max_elig,
        "ext_cap_pct": np.nan,
        "ext_value_pct": probe["ext_value_pct"].values,
        "floor_pct": np.nan,
        "p_max": p_max,
        "cap": cap,
        "is_forward": True,
        "is_rookie_scale": False,
        # Free agents have no contract on file: not a signing (no pricing-accuracy
        # claim), not prorated, and already null-surplus (Value Board only).
        "is_signing": False,
        "is_prorated": False,
        "actual_salary": np.nan,
        "pred_salary": value * cap,
        "latent_salary": latent * cap,
        "surplus": np.nan,
        "is_pushed": fa_flags["is_pushed"],
        "is_capped": fa_flags["is_capped"],
        "is_ext_capped": False,
        "is_floored": False,
        "darko_dpm_z": fa["darko_dpm_z"].values,
        "lebron_z": fa["lebron_z"].values,
        "laker_z": fa["laker_z"].values,
        "mpg": fa["mpg"].values,
        "usage_pct": fa["usage_pct"].values,
        "availability_3yr": fa["availability_3yr"].values,
        "ast_pct": fa["ast_pct"].values,
        "height_inches": fa["height_inches"].values,
        "draft_pick": fa["draft_pick"].values,
        "award_score_cum": fa["award_score_cum"].values,
        "signing_type": np.nan,
        "base_salary": expected * cap,
        "is_fa": True,
        "model_x": list(X.to_numpy(dtype=float)),
    })

    print(f"  Free agents added: {len(fa_out)} "
          f"(impact metrics, no contract on file)")
    combined = pd.concat([out, fa_out], ignore_index=True)
    combined_shap = np.vstack([shap_vals, shap_fa])
    return combined, combined_shap


def _round(x, nd=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return None
    return round(float(x), nd)


def _json_number(x):
    """A full-precision JSON number, or null for NaN/inf."""
    value = float(x)
    return value if np.isfinite(value) else None


def write_json(out: pd.DataFrame, shap_vals: np.ndarray, features: list[str],
               medians: pd.Series, source_df: pd.DataFrame, model,
               results: dict, fwd_metrics: dict, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    M = 1e6

    rows = []
    for r in out.itertuples(index=False):
        rows.append({
            "k": _row_key(r.player_name_norm, r.season),
            "n": r.player_name,
            "t": None if pd.isna(r.team) else r.team,
            "p": None if pd.isna(r.position) else r.position,
            "s": int(r.season),
            "a": _round(r.age, 0),
            "experience": int(r.experience),
            "yc": None if pd.isna(r.year_in_contract) else int(r.year_in_contract),
            "cy": None if pd.isna(r.contract_years) else int(r.contract_years),
            "act": _round(r.actual_salary / M),
            "pred": _round(r.pred_salary / M),
            "lat": _round(r.latent_salary / M),
            # Exact cap-percentage value and exact model inputs are retained
            # for parity tests and the browser what-if. Display dollars above
            # stay rounded independently.
            "latent_cap_pct": _json_number(r.latent_cap_pct),
            "feature_values": {
                feature: _json_number(value)
                for feature, value in zip(features, r.model_x)
            },
            "sur": _round(r.surplus / M),
            "tier_ceiling_pct": _json_number(r.tier_ceiling_pct),
            "max_eligible_pct": _json_number(r.max_eligible_pct),
            "ext_value_pct": _json_number(r.ext_value_pct),
            "floor_pct": _json_number(r.floor_pct),
            "cap": bool(r.is_capped),
            # Stage 3: the prediction sits at this row's own extension raise
            # cap, a legal ceiling that is not the max tier. Separate key from
            # "cap" so the site can keep saying MAX only where MAX is true.
            "ec": bool(r.is_ext_capped),
            "psh": bool(r.is_pushed),
            "flr": bool(r.is_floored),
            "fwd": bool(r.is_forward),
            "rs": bool(r.is_rookie_scale),
            # sg: on the Signing Board (survives the training filter chain, so it
            #     carries a pricing-accuracy number). pr: prorated partial season,
            #     excluded from the surplus ranking. See ISSUES #32.
            "sg": bool(r.is_signing),
            "pr": bool(r.is_prorated),
            "fa": bool(r.is_fa),
            "st": None if pd.isna(r.signing_type) else r.signing_type,
            "dk": _round(r.darko_dpm_z),
            "lb": _round(r.lebron_z),
            "rp": _round(r.laker_z),
            "mp": _round(r.mpg, 1),
            "us": _round(r.usage_pct, 1),
            "av": _round(r.availability_3yr, 3),
            "as": _round(r.ast_pct, 1),
            "ht": _round(r.height_inches, 0),
            "dp": None if pd.isna(r.draft_pick) else int(r.draft_pick),
            "aw": _round(r.award_score_cum),
        })

    caps = out["cap"].values
    shap_dollars = shap_vals * caps[:, None] / M
    keys = [_row_key(n, s) for n, s in
            zip(out["player_name_norm"], out["season"])]

    shap_out = {}
    for i, key in enumerate(keys):
        shap_out[key] = {
            f: _round(shap_dollars[i, j])
            for j, f in enumerate(features)
        }

    payload = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "model": results["model"],
        "rows": rows,
    }
    (dest / "valuations.json").write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    (dest / "shap.json").write_text(
        json.dumps({"base": None, "rows": shap_out},
                   ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )

    base_by_season = (
        out.groupby("season")["base_salary"].first() / M
    ).round(2).to_dict()

    signing = out["is_signing"]

    meta = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "model": results["model"],
        # The vN.Mx tag of the commit this export was built from. The site used
        # to hardcode this and it went stale within one version — the model
        # moves faster than anyone remembers to edit a string in the front end.
        "modelVersion": _model_version(),
        "holdoutSeason": HOLDOUT_SEASON,
        # Forward accuracy — the model's error on the holdout season's signings,
        # which it never saw. This is the honest headline the Signing Board
        # quotes; scoring the training rows would understate it by a third.
        "forwardR2": _round(fwd_metrics["r2"], 4),
        "forwardMae": _round(fwd_metrics["mae_m"], 2),
        "forwardN": fwd_metrics["n"],
        # Pooled cross-validation on the training seasons (< holdout) — the
        # structural "selection" number, a different quantity from forward
        # accuracy and not to be quoted as if it were the same.
        "cvR2": _round(results["cv_r2_mean"], 4),
        "cvMae": _round(results["cv_mae_mean"], 4),
        "sigma": results.get("sigma"),
        "nTrain": results["n_samples"],
        "nFeatures": results["n_features"],
        "nCensored": results.get("n_censored"),
        "nLeftCensored": results.get("n_left_censored"),
        # Stage 2's push and Stage 3's extension clip. tau and margin are
        # pre-registered constants, not fitted parameters. forwardR2 above is a
        # TOLD-ROUTE number (Stage 3 reads the realized extension flag) — the
        # convention adopted 2026-07-26; v7.1x-v7.13x were ex ante.
        "tau": stages.TAU,
        "margin": stages.MARGIN,
        "route": "told",
        "nPushed": int(out["is_pushed"].sum()),
        "nExtCapped": int(out["is_ext_capped"].sum()),
        "features": [{"key": f, "label": FEATURE_LABELS.get(f, f)}
                     for f in features],
        "mleBySeason": _mle_by_season(),
        "floorBySeasonExperience": _display_floor_table(source_df),
        "capBySeason": {str(k): v for k, v in CAP_BY_SEASON.items()},
        "baseSalaryBySeason": {str(k): v for k, v in base_by_season.items()},
        "seasons": sorted(int(s) for s in out["season"].unique()),
        "teams": sorted(t for t in out["team"].dropna().unique()),
        "positions": sorted(p for p in out["position"].dropna().unique()),
        "signingTypes": sorted(s for s in out["signing_type"].dropna().unique()),
        "nRows": len(out),
        "nYear1": int((out["year_in_contract"] == 1).sum()),
        # Negotiated first years: year-1 and not rookie-scale.
        "nSigning": int(signing.sum()),
        # Of those, the ones the model never saw — the forward slice.
        "nForward": int((signing & out["is_forward"]).sum()),
        # Free agents priced with no contract on file (Value Board only).
        "nFa": int(out["is_fa"].sum()),
    }
    (dest / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    stripped = _strip_model(model, features, medians,
                            _zscore_basis(source_df), dest)
    drift = _assert_model_parity(stripped, out)
    model_bytes = json.dumps(
        stripped, ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")
    (dest / "model.json").write_bytes(model_bytes)
    print(f"  model parity max |delta| {drift:.2e}")

    for name in ("valuations.json", "shap.json", "meta.json", "model.json"):
        kb = (dest / name).stat().st_size / 1024
        if name == "model.json":
            gzip_kb = len(gzip.compress(model_bytes, mtime=0)) / 1024
            print(f"  {name:20s} {kb:8.1f} KB  ({gzip_kb:.1f} KB gzip)")
        else:
            print(f"  {name:20s} {kb:8.1f} KB")


def _style_axes(ax):
    ax.set_facecolor(PALETTE["bg"])
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(PALETTE["line"])
    ax.tick_params(colors=PALETTE["ink_soft"], labelsize=9)
    ax.xaxis.label.set_color(PALETTE["ink_soft"])
    ax.yaxis.label.set_color(PALETTE["ink_soft"])
    ax.title.set_color(PALETTE["ink"])
    ax.grid(True, color=PALETTE["line"], linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)


def write_charts(out: pd.DataFrame, shap_vals: np.ndarray,
                 features: list[str], dest: Path) -> None:
    """Diagnostics restyled to the site palette so they read as native."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    plt.rcParams.update({
        "figure.facecolor": PALETTE["bg"],
        "savefig.facecolor": PALETTE["bg"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
    })
    dest.mkdir(parents=True, exist_ok=True)
    M = 1e6

    # 1. Predicted vs actual on the holdout season's signings — forward,
    #    out-of-sample, so the scatter shows real predictive accuracy rather
    #    than the model recalling its own training rows.
    y1 = out[out["is_forward"] & out["is_signing"]]
    fig, ax = plt.subplots(figsize=(7, 6.2))
    _style_axes(ax)
    at_bound = y1["is_capped"] | y1["is_floored"] | y1["is_ext_capped"]
    plain = y1[~at_bound]
    bound = y1[at_bound]
    ax.scatter(plain["actual_salary"] / M, plain["pred_salary"] / M, s=16,
               color=PALETTE["teal"], alpha=0.55, linewidths=0,
               label="Priced by the model")
    if len(bound):
        ax.scatter(bound["actual_salary"] / M, bound["pred_salary"] / M, s=34,
                   color=PALETTE["pink"], alpha=0.9, linewidths=0,
                   label="At a CBA bound")
    lim = max(y1["actual_salary"].max(), y1["pred_salary"].max()) / M * 1.05
    ax.plot([0, lim], [0, lim], color=PALETTE["ink_soft"], linewidth=1,
            linestyle="--", alpha=0.6)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Actual salary ($M)")
    ax.set_ylabel("Predicted salary ($M)")
    ax.set_title(f"Predicted vs actual — {HOLDOUT_SEASON} signings (forward)")
    leg = ax.legend(frameon=False, loc="upper left")
    for t in leg.get_texts():
        t.set_color(PALETTE["ink_soft"])
    fig.tight_layout()
    fig.savefig(dest / "pred_vs_actual.png", dpi=160)
    plt.close(fig)

    # 2. Mean |SHAP| per feature.
    imp = np.abs(shap_vals).mean(axis=0)
    order = np.argsort(imp)
    fig, ax = plt.subplots(figsize=(7, 5.4))
    _style_axes(ax)
    ax.grid(True, axis="y", alpha=0)
    labels = [FEATURE_LABELS.get(features[i], features[i]) for i in order]
    ax.barh(labels, imp[order] * 100, color=PALETTE["green"], height=0.68)
    ax.set_xlabel("Mean |SHAP| (percentage points of the cap)")
    ax.set_title("Feature contribution to predicted value")
    fig.tight_layout()
    fig.savefig(dest / "shap_importance.png", dpi=160)
    plt.close(fig)

    # 3. Signing Residual by signing mechanism. Uses all negotiated first years,
    #    not just the forward slice, so each mechanism has enough rows for the
    #    bias to be stable — this chart shows the shape of the pattern, and the
    #    project page carries the precise out-of-fold figures alongside it.
    st_rows = out[out["is_signing"]]
    st = st_rows.dropna(subset=["signing_type"])
    if len(st):
        agg = (st.groupby("signing_type")
                 .agg(bias=("surplus", "mean"), n=("surplus", "size"))
                 .query("n >= 5")
                 .sort_values("bias"))
        if len(agg):
            fig, ax = plt.subplots(figsize=(7.6, 4.8))
            _style_axes(ax)
            ax.grid(True, axis="y", alpha=0)
            colors = [PALETTE["gold"] if v < 0 else PALETTE["teal"]
                      for v in agg["bias"] / M]
            ax.barh([f"{i}  (n={int(n)})" for i, n in
                     zip(agg.index, agg["n"])],
                    agg["bias"] / M, color=colors, height=0.66)
            ax.axvline(0, color=PALETTE["ink_soft"], linewidth=1)
            ax.set_xlabel("Mean Signing Residual ($M)"
                          "\n← model underprices        overprices →")
            ax.set_title("Systematic bias by CBA signing mechanism", loc="left")
            fig.tight_layout()
            fig.savefig(dest / "residual_by_signing_type.png", dpi=160)
            plt.close(fig)

    # 4. SHAP beeswarm on the top features, site colormap.
    cmap = LinearSegmentedColormap.from_list(
        "site", [PALETTE["teal"], PALETTE["bg_soft"], PALETTE["pink"]]
    )
    top = np.argsort(imp)[-8:]
    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    _style_axes(ax)
    ax.grid(True, axis="y", alpha=0)
    rng = np.random.default_rng(42)
    for row, fi in enumerate(top):
        vals = shap_vals[:, fi] * 100
        raw = out[features[fi]].values if features[fi] in out.columns else None
        if raw is None or not np.isfinite(pd.to_numeric(raw, errors="coerce")).any():
            color = PALETTE["ink_soft"]
            sc_kw = dict(color=color)
        else:
            raw = pd.to_numeric(raw, errors="coerce").astype(float)
            lo, hi = np.nanpercentile(raw, [5, 95])
            norm = np.clip((raw - lo) / (hi - lo + 1e-12), 0, 1)
            sc_kw = dict(c=norm, cmap=cmap, vmin=0, vmax=1)
        jitter = rng.normal(0, 0.11, len(vals))
        ax.scatter(vals, np.full(len(vals), row) + jitter, s=7,
                   alpha=0.5, linewidths=0, **sc_kw)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([FEATURE_LABELS.get(features[i], features[i])
                        for i in top])
    ax.axvline(0, color=PALETTE["ink_soft"], linewidth=1, alpha=0.7)
    ax.set_xlabel("SHAP value (percentage points of the cap)")
    ax.set_title("Per-player feature impact  ·  colour = feature value")
    fig.tight_layout()
    fig.savefig(dest / "shap_beeswarm.png", dpi=160)
    plt.close(fig)

    for p in sorted(dest.glob("*.png")):
        print(f"  {p.name:32s} {p.stat().st_size / 1024:7.1f} KB")


def _check_forward(fwd_r2: float, tol: float = 0.03) -> None:
    """Cross-check the forward R² against the evaluation suite's rolling-origin
    number for the same season.

    The two are computed by independent code paths — this exporter and
    evaluate_suite.py — so they will not agree to the dollar. A gap beyond tol
    means one of them is wrong, most likely that this export has slipped back to
    scoring rows the model was trained on (which would inflate the number).
    """
    suite = OUTPUTS_DIR / "models" / "evaluation_suite.json"
    if not suite.exists():
        print("  (no evaluation_suite.json — skipping forward cross-check)")
        return
    d = json.loads(suite.read_text(encoding="utf-8"))
    origin = (d.get("champion", {}).get("B1_by_origin", {})
              .get(str(HOLDOUT_SEASON)))
    if not origin or "r2" not in origin:
        print(f"  (suite has no rolling-origin R² for {HOLDOUT_SEASON})")
        return
    ref = origin["r2"]
    gap = abs(ref - fwd_r2)
    if gap > tol:
        raise SystemExit(
            f"Forward R² {fwd_r2:.4f} is {gap:.3f} off the evaluation suite's "
            f"rolling-origin {HOLDOUT_SEASON} R² of {ref:.4f}. Either the model "
            "changed and evaluate_suite.py needs a rerun, or this export is "
            "scoring rows the model was trained on."
        )
    print(f"  forward R² {fwd_r2:.4f} matches suite's {ref:.4f} (Δ{gap:.3f})")


def main() -> None:
    from sklearn.metrics import mean_absolute_error, r2_score

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help=f"output directory (default: {DEFAULT_OUT})")
    ap.add_argument("--no-snapshot", action="store_true",
                    help="do not update outputs/web/valuations_export.csv")
    args = ap.parse_args()

    df = load_training_data()
    print(f"Loaded {len(df)} player-seasons")

    # Fit on every season before the holdout, so the holdout's signings are a
    # true forward prediction. train_grabit filters the frame it is given, so a
    # season-restricted df is all it takes — no change to train.py.
    train_df = df[df["season"] < HOLDOUT_SEASON].copy()
    results, model, features = train_grabit(train_df, sigma=0.02)
    print(f"\n{results['model']} fit on seasons < {HOLDOUT_SEASON}: "
          f"{results['n_samples']} rows, pooled CV R² {results['cv_r2_mean']:.4f}")

    tr_features, medians = _training_medians(train_df)
    if tr_features != features:
        raise SystemExit("feature list drifted between fit and export")

    out, shap_vals, expected = build_frame(df, model, features, medians, train_df)
    out, shap_vals = _add_free_agents(out, shap_vals, model, features, medians,
                                      expected, df, train_df)

    # The headline the Signing Board quotes is the forward number: accuracy on
    # the signings the model never saw. Membership is the training filter chain
    # (is_signing), not a looser re-derivation — free agents, second-round
    # rookies, prorated and continuation rows are all outside it (ISSUES #32).
    fwd = out[out["is_forward"] & out["is_signing"]]
    fwd_metrics = {
        "r2": float(r2_score(fwd["actual_cap_pct"], fwd["pred_cap_pct"])),
        "mae_m": float(mean_absolute_error(fwd["actual_salary"],
                                           fwd["pred_salary"]) / 1e6),
        "n": int(len(fwd)),
    }
    print(f"Scored {len(out)} rows  "
          f"({int((out['year_in_contract'] == 1).sum())} year-1, "
          f"{int(out['is_pushed'].sum())} pushed, "
          f"{int(out['is_capped'].sum())} capped, "
          f"{int(out['is_ext_capped'].sum())} at an extension raise cap, "
          f"{int(out['is_floored'].sum())} floored)")
    print(f"Forward {HOLDOUT_SEASON}: R² {fwd_metrics['r2']:.4f}, "
          f"MAE ${fwd_metrics['mae_m']:.2f}M on {fwd_metrics['n']} unseen signings")
    _check_forward(fwd_metrics["r2"])

    print("\nJSON:")
    write_json(out, shap_vals, features, medians, df, model, results,
               fwd_metrics, args.out)
    print("\nCharts:")
    write_charts(out, shap_vals, features, args.out / "charts")

    # Keep a copy in-repo so the export is reproducible without the site.
    if not args.no_snapshot:
        snap = OUTPUTS_DIR / "web"
        snap.mkdir(parents=True, exist_ok=True)
        out.to_csv(snap / "valuations_export.csv", index=False, encoding="utf-8")
        print(f"\nSnapshot: {snap / 'valuations_export.csv'}")
    print(f"Done -> {args.out}")


if __name__ == "__main__":
    main()
