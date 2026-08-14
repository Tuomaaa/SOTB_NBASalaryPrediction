"""Nested-CV ablation: kf_market_value with the FULL champion model as measurement.

The feature: for a player pricing at season T whose previous Year-1 contract
was at season T0, anchor a Kalman filter at prev_cap_pct (the market's last
actual price signal) and update it with the champion model's predictions on the
player's intermediate seasons (T0+1 .. T-1, the escalator years in df_full).
The KF output is a smoothed trajectory between "where the market last priced
him" and "where his stats say he should be now".

A Phase-1 test used a 2-parameter linear predictor (base_rating + age,
R2=0.46) as the measurement: the trajectory concept showed signal (R2
0.25 -> 0.41 on 386 multi-contract pairs) but the feature failed the gate
(dSel=+0.00045, t=+0.61) because the weak predictor systematically dragged
star anchors down on long contracts. This script swaps in the full 21-feature
Grabit champion (R2~0.82 OOF) as the measurement, which requires nested CV
for fold-honesty:

    outer fold f (5-fold GroupKFold by player, 10 seeds):
        inner 4-fold GroupKFold OOF over the outer TRAINING rows
            -> fold-honest Year-1 predictions (diagnostic R2, and R)
            -> per-inner-model predictions on the df_full intermediate rows
        KF per row, using ONLY inner-world measurements:
            train row (player p in inner fold j): measurements from inner
                model j, which excluded ALL of p's Year-1 rows
            test row (player p in fold f): p never entered ANY inner model,
                measurements are the mean of the four inner models
        fit the 22-feature candidate on the outer training rows, score fold f

    The kf column is recomputed per OUTER fold: fold f's models never see a
    kf value that was informed by fold f's players. (A single per-seed kf
    vector reusing each row's own-fold value would let fold g's TRAINING rows
    carry inner-OOF information from fold g's players — the channel this
    per-fold recomputation closes.)

The intermediate predictor is always the 21-feature champion — kf_market_value
never feeds itself (two-stage, not feedback). The route classifier's feature
list is FROZEN (v8.6x) and does not grow with the regression's, so the
candidate adds kf_market_value to the regression alone.

KF parameters (Phase-1 values, kept):
    P0 = 0.0005          anchor is ground truth, near-zero prior variance
    R  = var(inner OOF residuals), per (seed, fold)
    Q  = max(var(year-over-year dcap_pct, outer-train multi-contract
             players) - 2R, 0.0005)

Fallbacks: no anchor value -> NaN (median-imputed); no intermediate seasons
(or no prior Year-1 contract on record) -> kf_market_value = the anchor.

Anchor design (--anchor):
    market (default) — three-tier anchor:
        tier 1  the player's most recent PRIOR row in the evaluation frame
                itself: the last provable fresh market pricing. Filtered
                year-1 labels (prorated stints, continuation mislabels,
                rookie scale) cannot anchor — their seasons become
                measurements inside the window instead. Anchor value is read
                from the EVAL frame (cap-hold normalized, salary-corrected);
                prior variance P0 = 0.0005 (a market price is near truth).
        tier 2  no market anchor, but the player has rookie-scale seasons
                before T (first-rounders): anchor at the EARLIEST observed
                rookie season (deal year 2, treated as the deal's pricing
                point), every later pre-T season is a measurement — the
                y3/y4 team-option trail is where a declining trajectory
                shows. A slotted price is not a market judgment, so its
                prior variance is R: the anchor gets the trust of ONE model
                measurement, no more. Fixed before this arm was ever scored;
                never re-tune it on a result.
        tier 3  neither -> kf = prev_cap_pct: the feature degrades to
                information the model already has.
    prev — variant A, kept reproducible: anchor = prev_cap_pct (v7.11x
        semantics, last season's pay) with the window drawn from the RAW
        year_in_contract labels, dirty anchors included. Superseded by
        'market'; its 10-seed result: dSel -0.00042, t -0.39, FAIL.

Two candidate arms per run, off the same inner OOF and the same kf values:
    ADD    21 champion features + kf_market_value (22)
    SWAP   prev_cap_pct -> kf_market_value (21) — the sharper test, since kf
           is a refinement of prev_cap_pct, identical wherever no
           intermediate season exists

Gate: paired selection-pool delta, t > 2 (CLAUDE.md, layer A).

Two pipeline depths are reported, because they are not interchangeable:

    Phase 3   push -> clip -> extension clip. The pre-registered gate, and the
              deepest an arm can go inside a fitter — `make_champion_fitter`
              sees one fold, while the signing offset is a function of the
              whole OOF vector.
    Phase 3b  the same four arms with Stage 3's signing offset attached
              leave-fold-out, which is the deployed pipeline entire. Quoting
              Phase 3 against an evaluate_suite number reads as a regression
              that is really a missing stage (~0.016 R2 at v8.13x), so
              VERSION_HISTORY takes Phase 3b.

The gate stays on Phase 3: a paired delta between two arms is near-insensitive
to a per-type constant both of them receive, and moving a decided gate after
the fact would re-open every ablation already recorded against it.

Run:  python scripts/ablation_kf_market_value_full.py [--seeds 10]
          [--anchor prev|market] [--prehistory] [--expand-anchors]
      (--seeds 1 is a ~4-minute smoke pass; the gate needs the full 10)

--prehistory: inject pre-2019 Year-1 cap_pct values as tier-1 anchors for
    players whose most recent Year-1 contract was signed before 2019. Uses
    BBRef salary data from salaries_prehistory.csv (2016-2018) cross-
    referenced with Spotrac signing_types for Year-1 identification, plus
    Spotrac AAV for 2015 where no BBRef salary exists. These values enter
    the anchor map only, never the training set.

--expand-anchors: two expansions of the anchor map (--anchor market only):
    (a) First-contract tier-2 mirror: undrafted / 2nd-round players whose
        first NBA contract (exp <= 1, not rookie-scale) is in the raw
        training data get a tier-2 anchor at that contract's cap_pct with
        P0 = R. These are the mirror image of the rookie-scale tier-2: a
        first-round pick gets a slotted-salary anchor, now the undrafted
        and second-rounders get a first-contract anchor. ~75 eval rows
        currently at tier 3 move to tier 2.
    (b) Rookie Year-1 anchor shift: tier-2 rookie anchors move from
        deal-year-2 (draft_year+1, the earliest observed salary) to
        deal-year-1 (the actual start of the rookie contract). Year-1
        salary does not exist in the training data (stats-salary lag), so
        it is back-calculated from Year-2: salary_y1 = salary_y2 / 1.05
        (the standard 5% rookie-scale annual raise). The deal-year-2 row
        that was the anchor becomes a measurement, giving the KF one more
        update step. These values enter the anchor map only, never the
        training set.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

from config import OUTPUTS_DIR, CAP_BY_SEASON, PROCESSED_DIR
from src.model.evaluate_suite import (
    N_SPLITS, FORWARD_ORIGINS, load_evaluation_frame, make_champion_fitter,
    paired_delta, _dollars, _fold_pass, _reduce_fold_pass,
    oof_groupkfold,
)
from src.model.route_mixture import (
    attach_clf_features, train_route_classifier, route_proba, MAX_IDX,
)
from src.model.stages import (
    compose, signing_offsets, stage3_signing, SIGNING_K,
)
from src.model.train import (
    FEATURE_COLS, TARGET, load_training_data, _compute_max_eligible,
    _make_tobit_obj, _XGB_BASE,
)

P0 = 0.0005
P0_FIRST_CONTRACT = None   # set from R at runtime; separate from rookie P0=R
Q_FLOOR = 0.04
N_INNER = 4
KF_COL = "kf_market_value"
PREV_COL = "prev_cap_pct"


# ---------------------------------------------------------------------------
# Pre-2019 anchor loader
# ---------------------------------------------------------------------------

def load_prehistory_anchors() -> dict[str, dict[int, tuple[float, bool]]]:
    """Load pre-2019 Year-1 cap_pct values for KF anchor injection.

    Sources:
      - salaries_prehistory.csv: BBRef salary data for seasons 2016-2018
      - spotrac_signing_types.csv: Year-1 identification (season == contract_start)
      - Spotrac AAV for season 2015 where no BBRef salary exists

    Returns dict[player_name_norm -> {season: (cap_pct, is_prorated)}].
    Pre-2019 rows enter the anchor map only, never the training set.
    """
    from src.model.train import PRORATED_FLOOR

    # 1. Load Spotrac signing types to identify Year-1 contracts
    spotrac_path = PROCESSED_DIR / "spotrac_signing_types.csv"
    if not spotrac_path.exists():
        print("WARNING: spotrac_signing_types.csv not found, no prehistory anchors")
        return {}
    st = pd.read_csv(spotrac_path)
    st["season"] = st["season"].astype(int)
    st["contract_start"] = st["contract_start"].astype(int)
    y1 = st[(st["season"] == st["contract_start"])
            & (st["season"] >= 2015) & (st["season"] <= 2018)].copy()
    y1_set = set(zip(y1["player_name_norm"], y1["season"]))
    print(f"  Spotrac Year-1 contracts 2015-2018: {len(y1_set)}")

    # 2. Load BBRef prehistory salaries (2016-2018)
    prehistory_path = PROCESSED_DIR / "salaries_prehistory.csv"
    events: dict[str, dict[int, tuple[float, bool]]] = {}
    bbref_covered = set()
    if prehistory_path.exists():
        ph = pd.read_csv(prehistory_path)
        ph["season"] = ph["season"].astype(int)
        for _, r in ph.iterrows():
            p, s, sal = r["player_name_norm"], r["season"], r["salary"]
            if (p, s) not in y1_set:
                continue
            cap = CAP_BY_SEASON.get(s)
            if cap is None:
                continue
            cap_pct = float(sal) / float(cap)
            is_prorated = cap_pct < PRORATED_FLOOR
            events.setdefault(p, {})[s] = (cap_pct, is_prorated)
            bbref_covered.add((p, s))
        print(f"  BBRef prehistory Year-1 matches: {len(bbref_covered)}")

    # 3. Spotrac AAV fallback for 2015 and any BBRef gaps
    aav_used = 0
    for _, r in y1.iterrows():
        p, s = r["player_name_norm"], int(r["season"])
        if (p, s) in bbref_covered:
            continue
        cap = CAP_BY_SEASON.get(s)
        aav = r.get("aav")
        if cap is None or pd.isna(aav) or float(aav) <= 0:
            continue
        cap_pct = float(aav) / float(cap)
        is_prorated = cap_pct < PRORATED_FLOOR
        events.setdefault(p, {})[s] = (cap_pct, is_prorated)
        aav_used += 1
    print(f"  Spotrac AAV fallback anchors: {aav_used}")

    total = sum(len(v) for v in events.values())
    players = len(events)
    print(f"  Total prehistory anchors: {total} "
          f"({players} players, seasons 2015-2018)")
    return events


# ---------------------------------------------------------------------------
# Champion pipeline, split into fit / predict
# ---------------------------------------------------------------------------

def fit_champion_models(train: pd.DataFrame, features: list[str],
                        clf_features: list[str], seed: int,
                        sigma: float = 0.02, gate_frac: float = 0.55,
                        floor_gate_k: float = 2.0):
    """Fit the champion Stage-1 Grabit + route classifier on one slice.

    Byte-for-byte the training path of route_mixture.grabit_latent (shipped
    defaults: censor_c=None, sigma_left=None) plus make_champion_fitter's
    classifier fit — split into fit and predict so ONE fitted pair can price
    several frames (the inner-val Year-1 rows AND the df_full intermediate
    rows) without refitting.
    """
    y_tr = train[TARGET].values
    max_elig_tr = train["max_eligible_pct"].values
    is_max_tr = train["is_max_contract"].values.astype(bool)

    base = XGBRegressor(**{**_XGB_BASE, "random_state": seed})
    base.fit(train[features], y_tr)
    bp = base.predict(train[features])
    gate = is_max_tr & (bp >= gate_frac * max_elig_tr)
    if floor_gate_k > 0 and "is_at_floor" in train.columns:
        gate_l = train["is_at_floor"].values & (bp <= floor_gate_k * y_tr)
    else:
        gate_l = np.zeros(len(train), bool)

    model = XGBRegressor(**{**_XGB_BASE, "random_state": seed,
                            "objective": _make_tobit_obj(gate, sigma,
                                                         left_mask=gate_l),
                            "base_score": float(y_tr.mean())})
    model.fit(train[features], y_tr)
    clf = train_route_classifier(train, clf_features, seed)
    return model, clf


def predict_champion(model, clf, test: pd.DataFrame, features: list[str],
                     clf_features: list[str]) -> np.ndarray:
    """The champion composition (push -> clip -> extension clip) on `test`.

    Identical to make_champion_fitter's output for the same fitted models. On
    df_full intermediate rows is_extension is False everywhere, so Stage 3 is
    a no-op there and the measurement is the ex-post market-value estimate.
    """
    latent = model.predict(test[features])
    lo = (test["floor_pct"].values if "floor_pct" in test.columns
          else np.zeros(len(test)))
    hi = test["max_eligible_pct"].values
    p_max = route_proba(clf, test, clf_features)[:, MAX_IDX]
    return compose(latent, lo=lo, hi=hi, p_max=p_max,
                   is_extension=test["is_extension"].values,
                   ext_cap_pct=test["ext_cap_pct"].values)


def signing_corrected(by_seed: np.ndarray, df: pd.DataFrame, folds, sel):
    """Attach Stage 3's signing offset to an already-collected OOF matrix.

    `predict_champion` above stops at the extension clip, for the reason
    `make_champion_fitter` documents: a fitter sees one fold, and the offset is
    a function of the WHOLE out-of-fold vector. So every arm this harness scores
    is the deployed pipeline minus its last stage, and its R2 is not comparable
    to evaluate_suite's headline — the gap ran ~0.016 R2 at v8.13x. This is the
    missing tail, `oof_groupkfold_signing`'s procedure applied to a matrix
    instead of to a `_fold_pass` store, so the two paths cannot drift.

    Fold f's offsets are a mean over the OOF residuals of folds g != f, so no
    row is corrected by a statistic its own target informed. Each (fold, seed)
    cell is corrected BEFORE the seed average, because the offset composes with
    clips that are not linear (evaluate_suite `_fold_pass`, same reasoning).

    Args:
        by_seed: (n_seeds, n_rows) predictions, each row filled from the fold
            that held it out.
        df: the evaluation frame — supplies `signing_cat` and the legality
            bounds the offset must be re-clipped into.
        folds: the (train, val) index pairs `by_seed` was filled from.
        sel: selection-pool mask (True where the row is NOT confirmation).

    Returns:
        (oof, fold_r2, fold_r2_sel, lfo) — the first three matching
        `_reduce_fold_pass`'s tuple, plus the per-fold offset detail.
    """
    y = df[TARGET].values
    cat = df["signing_cat"].values
    lo, hi = df["floor_pct"].values, df["max_eligible_pct"].values
    is_ext, ext_cap = df["is_extension"].values, df["ext_cap_pct"].values
    mech = df["mech_cap_pct"].values if "mech_cap_pct" in df.columns else None

    fold_of = np.empty(len(df), dtype=int)
    for fi, (_tr, va) in enumerate(folds):
        fold_of[va] = fi
    resid = y - by_seed.mean(axis=0)          # actual - predicted, cap_pct
    lfo = {fi: signing_offsets(resid, cat, pool=fold_of != fi,
                               k=SIGNING_K, detail=True)
           for fi in range(len(folds))}

    n_seeds = by_seed.shape[0]
    out = np.zeros_like(by_seed)
    fold_r2 = np.zeros((len(folds), n_seeds))
    fold_r2_sel = np.zeros((len(folds), n_seeds))
    for fi, (_tr, va) in enumerate(folds):
        vs = sel[va]
        for si in range(n_seeds):
            p = stage3_signing(
                by_seed[si, va], cat[va], lfo[fi], lo=lo[va], hi=hi[va],
                mech_cap_pct=mech[va] if mech is not None else None,
                is_extension=is_ext[va], ext_cap_pct=ext_cap[va])
            out[si, va] = p
            fold_r2[fi, si] = r2_score(y[va], p)
            fold_r2_sel[fi, si] = (r2_score(y[va][vs], p[vs])
                                   if vs.sum() > 10 else np.nan)
    return out.mean(axis=0), fold_r2, fold_r2_sel, lfo


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_full_frame(df_eval: pd.DataFrame, features: list[str]
                       ) -> pd.DataFrame:
    """load_training_data() made predictable by the champion pipeline.

    The intermediate rows only ever appear on the TEST side of a fit, so they
    need the prediction-time columns: the 21 features (median-filled with the
    evaluation frame's medians, the scale every model here was fit on), the
    Stage-2 bounds, inert Stage-3 route facts, and the classifier columns.
    """
    df_full = load_training_data()
    df_full = _compute_max_eligible(df_full)

    # Stage-2 lower bound: the evaluation frame's per-season floor. Exact
    # per-row floors need the experience bucket; for a KF measurement the
    # difference is noise, and grabit_latent would otherwise clip at zero.
    season_floor = df_eval.groupby("season")["floor_pct"].median()
    df_full["floor_pct"] = (df_full["season"].map(season_floor)
                            .fillna(float(df_eval["floor_pct"].min())))

    # Stage 3 is inert on intermediate rows: an escalator season is not a
    # first-paying-year extension, and the measurement wanted is the model's
    # market-value estimate, not a route-capped price.
    df_full["is_extension"] = False
    df_full["ext_cap_pct"] = np.nan

    med = df_eval[features].median()
    df_full[features] = df_full[features].fillna(med).fillna(0)

    # Classifier columns: base cols are a subset of `features` (already
    # filled); enrichment cols keep native NaN, their tested ship form.
    df_full, _ = attach_clf_features(df_full)
    return df_full.reset_index(drop=True)


def build_anchor_map(df_eval: pd.DataFrame, df_full: pd.DataFrame,
                     extra_events: dict[str, dict[int, tuple[float, bool]]]
                     | None = None, expand_anchors: bool = False):
    """The three-tier anchor map (--anchor market). Per eval row:

        tier[i]        1 market anchor, 2 rookie anchor, 0 fallback (kf=prev)
        anchor_val[i]  the anchor cap_pct (NaN on tier 0)
        inter_idx[i]   df_full positions of the measurement seasons, in
                       chronological order (empty on tier 0 / no seasons)
        anchor_kind[i] diagnostic sub-type: "market", "rookie_y2",
                       "rookie_y1", "first_contract", or "" (tier 0)

    Tier-1 candidates are MARKET PRICING EVENTS, two kinds:
      - the player's own PRIOR rows in the eval frame — eval membership IS
        the project's definition of a fresh market price. Anchor value =
        the EVAL row's cap_pct (cap-hold normalized, salary-corrected).
      - prorated year-1 stints (raw label, cap_pct < PRORATED_FLOOR): a
        10-day at the minimum RATE is a market verdict ("worth the
        minimum") whose observed cap_pct merely prorates playing time, so
        the anchor value is PULLED UP to that season's full-season minimum
        cap charge (the eval frame's per-season floor_pct).
    Continuation mislabels, rookie scale and first contracts still cannot
    anchor — their seasons become measurements inside the window. Tier 2
    anchors at the earliest observed rookie-scale season (deal year 2) and
    is trusted like one measurement (P0 = R, see main).

    extra_events: optional pre-2019 Year-1 cap_pct anchors (--prehistory).
        Merged BEFORE the eval-frame events so an eval-frame anchor at the
        same (player, season) wins; a pre-2019 anchor only fires for
        players whose most recent Year-1 was before the eval frame starts.

    expand_anchors: when True, two tier-2 expansions (--expand-anchors):
        (a) First-contract mirror: players whose first NBA contract
            (exp <= 1, not rookie-scale) is in df_full get a tier-2
            anchor at that contract's cap_pct.
        (b) Rookie Year-1 shift: tier-2 rookie anchors move from
            deal-year-2 to deal-year-1, with Year-1 cap_pct
            back-calculated from the Year-2 value via the 5% rookie raise.
        All expanded anchors get P0 = R; they enter the anchor map only,
        never the training set.
    """
    from src.model.train import _load_rookie_scale_set, PRORATED_FLOOR
    rs = _load_rookie_scale_set()
    season_floor = df_eval.groupby("season")["floor_pct"].median().to_dict()
    floor_min = float(df_eval["floor_pct"].min())

    # market pricing events: player -> {season: (anchor value, is_prorated)}
    # Pre-2019 events go in first; eval-frame events use setdefault so they
    # cannot be overwritten by the pre-2019 data, but a pre-2019 anchor at a
    # season the eval frame does not cover adds to the player's event list.
    events: dict[str, dict[int, tuple[float, bool]]] = {}
    if extra_events:
        for p, p_events in extra_events.items():
            for s, val in p_events.items():
                events.setdefault(p, {})[int(s)] = val
    for p, s, c in zip(df_eval["player_name_norm"],
                       df_eval["season"].astype(int), df_eval[TARGET].values):
        events.setdefault(p, {})[int(s)] = (float(c), False)

    full_cp: dict[tuple[str, int], float] = {}
    rows_by_player: dict[str, dict[int, int]] = {}
    yic = df_full["year_in_contract"] if "year_in_contract" in df_full.columns \
        else pd.Series(np.nan, index=df_full.index)
    for pos, (p, s, c, y1) in enumerate(zip(df_full["player_name_norm"],
                                            df_full["season"].astype(int),
                                            df_full[TARGET].values,
                                            yic.values)):
        if (p, s) not in full_cp:         # duplicates: keep the first row
            full_cp[(p, s)] = float(c)
            rows_by_player.setdefault(p, {})[s] = pos
            if y1 == 1 and c < PRORATED_FLOOR:
                events.setdefault(p, {}).setdefault(
                    s, (season_floor.get(s, floor_min), True))
    rookie_seasons: dict[str, list[int]] = {}
    for (p, s) in rs:
        if (p, s) in full_cp:
            rookie_seasons.setdefault(p, []).append(s)

    # --- expand_anchors: first-contract data (Task 4) -----------------------
    # Identify rows in df_full where exp <= 1 and NOT in the rookie-scale set.
    # These are undrafted / 2nd-round first contracts that _filter_rookie_contracts
    # removes from the eval frame. Their cap_pct provides a tier-2 anchor.
    first_contract_cap: dict[str, dict[int, float]] = {}
    if expand_anchors:
        from scripts.build_external_features import norm
        from src.model.train import _load_debut_seasons
        debut = _load_debut_seasons()
        if debut:
            for p, s, c in zip(df_full["player_name_norm"],
                               df_full["season"].astype(int),
                               df_full[TARGET].values):
                pn = norm(str(p))
                d = debut.get(pn)
                if d is None:
                    continue
                exp = int(s) - int(d)
                if exp <= 1 and (p, int(s)) not in rs:
                    first_contract_cap.setdefault(p, {})[int(s)] = float(c)
            print(f"  expand-anchors: {sum(len(v) for v in first_contract_cap.values())} "
                  f"first-contract rows found ({len(first_contract_cap)} players)")

    # --- expand_anchors: rookie Year-1 computed values (Task 5) -------------
    # For first-round picks, back-calculate Year-1 cap_pct from Year-2. The
    # Year-1 row does not exist in the training data (stats-salary lag), but
    # the rookie-scale 5% annual raise means salary_y1 = salary_y2 / 1.05.
    # This gives the KF one more measurement (deal-year-2 shifts from anchor
    # to measurement). Where salaries.csv has the actual Year-1 salary, it is
    # used instead; the back-calculation is the fallback.
    ROOKIE_RAISE = 1.05  # standard rookie-scale annual raise
    rookie_y1: dict[str, tuple[int, float]] = {}  # player -> (draft_year, y1_cap_pct)
    if expand_anchors:
        # Try salaries.csv first for actual Year-1 data
        sal_lookup: dict[tuple[str, int], float] = {}
        sal_path = PROCESSED_DIR / "salaries.csv"
        if sal_path.exists():
            sal = pd.read_csv(sal_path)
            for _, r in sal.iterrows():
                cap = CAP_BY_SEASON.get(int(r["season"]))
                if cap:
                    # Normalize name: lowercase, strip, remove periods (for
                    # "jr." -> "jr" matching against player_name_norm).
                    pn = str(r["player"]).lower().strip().replace(".", "")
                    sal_lookup[(pn, int(r["season"]))] = float(r["salary"]) / cap
        n_actual, n_computed = 0, 0
        for p, seasons in rookie_seasons.items():
            earliest = min(seasons)       # deal year 2 = draft_year + 1
            y1_season = earliest - 1      # deal year 1 = draft_year
            cap_y1 = CAP_BY_SEASON.get(y1_season)
            if cap_y1 is None:
                continue
            # Strip periods from player_name_norm for the salary lookup
            pn_key = p.replace(".", "")
            if (pn_key, y1_season) in sal_lookup:
                rookie_y1[p] = (y1_season, sal_lookup[(pn_key, y1_season)])
                n_actual += 1
            elif (p, earliest) in full_cp:
                # Back-calculate: salary_y1 = salary_y2 / 1.05
                y2_cap_pct = full_cp[(p, earliest)]
                cap_y2 = CAP_BY_SEASON.get(earliest)
                if cap_y2:
                    y1_salary = y2_cap_pct * cap_y2 / ROOKIE_RAISE
                    rookie_y1[p] = (y1_season, float(y1_salary / cap_y1))
                    n_computed += 1
        print(f"  expand-anchors: {len(rookie_y1)} rookie Year-1 values "
              f"({n_actual} from salaries.csv, {n_computed} back-calculated "
              f"from Year-2, of {len(rookie_seasons)} rookie players)")

    # --- Tier assignment ----------------------------------------------------
    tier = np.zeros(len(df_eval), dtype=int)
    anchor_val = np.full(len(df_eval), np.nan)
    anchor_prorated = np.zeros(len(df_eval), dtype=bool)
    anchor_kind = np.full(len(df_eval), "", dtype="U20")
    inter_idx, needed = [], set()
    for i, (p, T) in enumerate(zip(df_eval["player_name_norm"],
                                   df_eval["season"].astype(int))):
        market = [s for s in events.get(p, ()) if s < T]
        if market:
            t0 = max(market)
            tier[i] = 1
            anchor_val[i], anchor_prorated[i] = events[p][t0]
            anchor_kind[i] = "market"
        else:
            rook = [s for s in rookie_seasons.get(p, ()) if s < T]
            if rook:
                tier[i] = 2
                if expand_anchors and p in rookie_y1:
                    # Task 5: anchor at Year-1 (draft year) with computed value;
                    # deal-year-2 (the old anchor) becomes a measurement.
                    dy, y1_val = rookie_y1[p]
                    if dy < T:
                        t0 = dy
                        anchor_val[i] = y1_val
                        anchor_kind[i] = "rookie_y1"
                    else:
                        t0 = min(rook)
                        anchor_val[i] = full_cp[(p, t0)]
                        anchor_kind[i] = "rookie_y2"
                else:
                    t0 = min(rook)                # earliest observed = deal year 2
                    anchor_val[i] = full_cp[(p, t0)]
                    anchor_kind[i] = "rookie_y2"
            elif expand_anchors and p in first_contract_cap:
                # Task 4: first-contract anchor for undrafted / 2nd-round players.
                fc_seasons = [s for s in first_contract_cap[p] if s < T]
                if fc_seasons:
                    t0 = min(fc_seasons)
                    tier[i] = 2
                    fc_val = first_contract_cap[p][t0]
                    if fc_val < PRORATED_FLOOR:
                        # Prorated partial-season: pull up to the season minimum,
                        # same treatment as prorated market anchors.
                        anchor_val[i] = season_floor.get(t0, floor_min)
                        anchor_prorated[i] = True
                    else:
                        anchor_val[i] = fc_val
                    anchor_kind[i] = "first_contract"
                else:
                    inter_idx.append([])
                    continue
            else:
                inter_idx.append([])
                continue
        seasons = sorted(s for s in rows_by_player.get(p, {}) if t0 < s < T)
        idx = [rows_by_player[p][s] for s in seasons]
        inter_idx.append(idx)
        needed.update(idx)
    return inter_idx, sorted(needed), tier, anchor_val, anchor_prorated, anchor_kind


def build_intermediate_map_raw(df_eval: pd.DataFrame, df_full: pd.DataFrame):
    """Variant A's window builder (--anchor prev), kept for reproducibility.

    T0 is the player's most recent prior year_in_contract==1 season in
    df_full — the RAW label, dirty anchors included (52 prorated stints, 132
    rookie-scale rows, 197 continuations/first contracts of the 716 anchors).
    Returns (inter_idx, needed_idx, n_anchor, t0_pos).
    """
    year1: dict[str, list[int]] = {}
    year1_pos: dict[tuple[str, int], int] = {}
    yic = df_full["year_in_contract"] if "year_in_contract" in df_full.columns \
        else pd.Series(np.nan, index=df_full.index)
    rows_by_player: dict[str, dict[int, int]] = {}
    seen = set()
    for pos, (p, s, y1) in enumerate(zip(df_full["player_name_norm"],
                                         df_full["season"].astype(int),
                                         yic.values)):
        if (p, s) not in seen:            # duplicates (mid-season buyouts)
            seen.add((p, s))              # keep the first row per (p, s)
            rows_by_player.setdefault(p, {})[s] = pos
        if y1 == 1:
            year1.setdefault(p, []).append(s)
            year1_pos.setdefault((p, s), pos)

    inter_idx, needed = [], set()
    t0_pos = np.full(len(df_eval), -1, dtype=int)
    n_anchor = 0
    for i, (p, T) in enumerate(zip(df_eval["player_name_norm"],
                                   df_eval["season"].astype(int))):
        prior = [s for s in year1.get(p, ()) if s < T]
        if not prior:
            inter_idx.append([])
            continue
        n_anchor += 1
        t0 = max(prior)
        t0_pos[i] = year1_pos[(p, t0)]
        seasons = sorted(s for s in rows_by_player.get(p, {}) if t0 < s < T)
        idx = [rows_by_player[p][s] for s in seasons]
        inter_idx.append(idx)
        needed.update(idx)
    return inter_idx, sorted(needed), n_anchor, t0_pos


def estimate_q(df_full: pd.DataFrame, players: set, r_var: float
               ) -> tuple[float, float, int]:
    """Process noise from year-over-year dcap_pct of multi-contract players.

    Restricted to the OUTER TRAINING players so no test-fold target informs
    the constant. Under a random walk observed with variance R,
    var(y_t - y_{t-1}) = Q + 2R, hence the subtraction; the floor keeps the
    filter from freezing when escalator years make the raw variance small.
    """
    sub = df_full[df_full["player_name_norm"].isin(players)]
    if "year_in_contract" in sub.columns:
        counts = (sub[sub["year_in_contract"] == 1]
                  .groupby("player_name_norm").size())
        multi = set(counts[counts >= 2].index)
        sub = sub[sub["player_name_norm"].isin(multi)]
    sub = (sub.drop_duplicates(["player_name_norm", "season"])
           .sort_values(["player_name_norm", "season"]))
    deltas = []
    for _, g in sub.groupby("player_name_norm"):
        s = g["season"].values.astype(int)
        v = g[TARGET].values
        consec = s[1:] == s[:-1] + 1
        deltas.extend((v[1:] - v[:-1])[consec])
    var_d = float(np.var(deltas, ddof=1)) if len(deltas) > 2 else 0.0
    return max(var_d - 2.0 * r_var, Q_FLOOR), var_d, len(deltas)


# ---------------------------------------------------------------------------
# The Kalman filter
# ---------------------------------------------------------------------------

def kalman_update(anchor: float, measurements, q: float, r: float,
                  p0: float = P0) -> float:
    """Random-walk KF: anchor at prev_cap_pct, one update per season."""
    x, p = float(anchor), p0
    for z in measurements:
        p_pred = p + q
        k = p_pred / (p_pred + r)
        x = x + k * (float(z) - x)
        p = (1.0 - k) * p_pred
    return x


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global P0, Q_FLOOR

    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10,
                    help="number of seeds (10 = the canonical gate)")
    ap.add_argument("--anchor", choices=("prev", "market"), default="market",
                    help="'market' = three-tier anchor (eval-frame market "
                         "price / rookie deal-year-2 with P0=R / fallback to "
                         "prev_cap_pct); 'prev' = variant A, raw-label "
                         "windows anchored at prev_cap_pct, kept only for "
                         "reproducibility")
    ap.add_argument("--no-prehistory", action="store_true",
                    help="disable pre-2019 Year-1 anchors (BBRef 2016-18 + "
                         "Spotrac AAV 2015); on by default")
    ap.add_argument("--no-expand-anchors", action="store_true",
                    help="disable expanded tier-2 anchors (first-contract "
                         "mirror + rookie Year-1 shift); on by default")
    ap.add_argument("--p0", type=float, default=None,
                    help="override tier-1 initial state variance P0 "
                         f"(default {P0})")
    ap.add_argument("--q-floor", type=float, default=None,
                    help="override process noise floor Q_FLOOR "
                         f"(default {Q_FLOOR})")
    args = ap.parse_args()
    args.prehistory = not args.no_prehistory
    args.expand_anchors = not args.no_expand_anchors

    if args.p0 is not None:
        P0 = args.p0
    if args.q_floor is not None:
        Q_FLOOR = args.q_floor
    seeds = list(range(args.seeds))
    t_start = time.time()

    line = "=" * 70
    print(line)
    print("NESTED-CV kf_market_value ABLATION (full-model measurement)")
    print(f"anchor mode: {args.anchor} "
          f"({'three-tier: market price / rookie y2 with P0=R / prev fallback' if args.anchor == 'market' else 'variant A: prev_cap_pct, raw-label windows'})"
          f"{' + PREHISTORY (pre-2019 Year-1 anchors)' if args.prehistory else ' (no prehistory)'}"
          f"{' + EXPAND-ANCHORS (first-contract + rookie Year-1)' if args.expand_anchors else ' (no expand-anchors)'}")
    print(line)

    df_eval, frame_features = load_evaluation_frame(allow_missing_computed=True)
    df_eval, clf_features = attach_clf_features(df_eval)
    # The champion arm is the prev_cap_pct baseline. Reconstruct it from
    # FEATURE_COLS rather than trusting what the frame hands back: since v8.13x
    # shipped the SWAP, FEATURE_COLS *is* the kf list, and kf_market_value is
    # computed per fold rather than stored, so _prepare_Xy drops it and
    # load_evaluation_frame returns 20 columns carrying no price anchor at all
    # (ISSUES #50). Swapping kf back to prev yields the pre-v8.13x champion
    # whichever direction the ship went, and is a no-op on a list that still
    # has prev.
    features = [PREV_COL if f == KF_COL else f for f in FEATURE_COLS]
    missing = [f for f in features if f not in df_eval.columns]
    assert not missing, f"baseline features absent from the frame: {missing}"
    assert PREV_COL in features and KF_COL not in features
    # prev_cap_pct skipped the frame loader's median fill on its way out of
    # FEATURE_COLS. Every arm here reads it, so fill it on the same rule.
    unfilled = [f for f in features if f not in frame_features]
    if unfilled:
        df_eval[unfilled] = (df_eval[unfilled]
                             .fillna(df_eval[unfilled].median()).fillna(0))
        print(f"median-filled {unfilled}: outside FEATURE_COLS, so "
              f"load_evaluation_frame skipped them")
    y = df_eval[TARGET].values
    groups = df_eval["player_name_norm"].values
    sel = ~df_eval["is_confirmation"].values
    prev = df_eval["prev_cap_pct"].values
    seas_all = df_eval["season"].values.astype(int)

    print(f"\nPreparing the full frame (all year_in_contract rows)...")
    df_full = prepare_full_frame(df_eval, features)
    prehistory_events = None
    if args.prehistory and args.anchor == "market":
        print("\nLoading pre-2019 Year-1 anchors (--prehistory):")
        prehistory_events = load_prehistory_anchors()
    if args.anchor == "market":
        (inter_idx, needed_idx, tier, anchor_val,
         anchor_prorated, anchor_kind) = build_anchor_map(
            df_eval, df_full, extra_events=prehistory_events,
            expand_anchors=args.expand_anchors)
        anchor = np.where(tier > 0, anchor_val, prev)
        n_anchor = int((tier > 0).sum())
    else:
        inter_idx, needed_idx, n_anchor, _t0 = build_intermediate_map_raw(
            df_eval, df_full)
        tier = np.ones(len(df_eval), dtype=int)   # P0 everywhere (variant A)
        anchor_prorated = np.zeros(len(df_eval), dtype=bool)
        anchor_kind = np.full(len(df_eval), "prev", dtype="U20")
        anchor = prev.copy()
    pos_of = {fi: k for k, fi in enumerate(needed_idx)}
    inter_pos = [[pos_of[fi] for fi in lst] for lst in inter_idx]
    full_needed = df_full.iloc[needed_idx]

    n_inter = sum(1 for lst in inter_idx if lst)
    counts = pd.Series([len(lst) for lst in inter_idx])
    print(f"eval frame {len(df_eval)} rows x {len(features)} features "
          f"({len(clf_features)} classifier features, frozen — "
          f"{KF_COL} joins the regression only)")
    print(f"full frame {len(df_full)} rows; intermediate predict subset "
          f"{len(needed_idx)} unique (player, season) rows")
    print(f"anchor: {n_anchor}/{len(df_eval)} rows anchored; {n_inter} have "
          f">=1 measurement season "
          f"(counts 1/2/3+: {int((counts == 1).sum())}/"
          f"{int((counts == 2).sum())}/{int((counts >= 3).sum())})")
    if args.anchor == "market":
        has_i = np.array([bool(lst) for lst in inter_idx])
        m1 = tier == 1
        m2 = tier == 2
        print(f"  tier 1 market-price anchor (P0=0.0005): {int(m1.sum())} rows, "
              f"{int((m1 & has_i).sum())} with measurements")
        print(f"    of tier 1, prorated stints pulled to the season minimum: "
              f"{int(anchor_prorated[m1].sum())}")
        print(f"  tier 2 anchors (P0=R): {int(m2.sum())} rows, "
              f"{int((m2 & has_i).sum())} with measurements")
        for kind, lab in (("rookie_y2", "rookie deal-year-2 (original)"),
                          ("rookie_y1", "rookie Year-1 shift (expand-anchors)"),
                          ("first_contract", "first-contract mirror (expand-anchors)")):
            mk = anchor_kind == kind
            if mk.any():
                print(f"    {lab}: {int(mk.sum())} rows, "
                      f"{int((mk & has_i).sum())} with measurements")
                if kind == "first_contract":
                    print(f"      of which prorated (pulled to min): "
                          f"{int(anchor_prorated[mk].sum())}")
        print(f"  tier 3 fallback kf=prev_cap_pct: {int((tier == 0).sum())} rows")

    # Champion arm — the canonical protocol, 21 features. Run through
    # _fold_pass rather than oof_groupkfold so the per-(fold, seed)
    # predictions survive for Phase 4's fold-paired segment deltas;
    # _reduce_fold_pass then yields oof_groupkfold's exact outputs.
    n_seeds = len(seeds)
    print(f"\nChampion ({len(features)} features): GroupKFold over "
          f"{n_seeds} seeds x {N_SPLITS} folds...", flush=True)
    t0 = time.time()
    fitter_21 = make_champion_fitter(clf_features)
    store_c, folds_c, _ = _fold_pass(df_eval, features, fitter_21, seeds)
    oof_c, fr2_c, fr2sel_c = _reduce_fold_pass(df_eval, store_c, folds_c,
                                               seeds)[None]
    oof_c_by_seed = np.zeros((n_seeds, len(df_eval)))
    for rec in store_c:
        oof_c_by_seed[rec["si"], rec["va"]] = rec["pred"][None]
    print(f"  done in {time.time() - t0:.0f}s   "
          f"A1={r2_score(y, oof_c):.4f}", flush=True)

    # Candidate arms — nested CV, kf recomputed per outer fold. Two tests off
    # the same inner OOF and the same kf values:
    #   ADD   21 champion features + kf_market_value (22). Diluted by the
    #         kf <-> prev_cap_pct collinearity; kept as the reference the
    #         Phase-1 result (dSel +0.00045, t +0.61) was quoted on.
    #   SWAP  prev_cap_pct -> kf_market_value (21). The sharper test: kf IS a
    #         refinement of prev_cap_pct (identical wherever no intermediate
    #         season exists), so the hypothesis is "the refined price signal
    #         beats the raw one", not "the model wants both columns".
    folds = list(GroupKFold(n_splits=N_SPLITS).split(df_eval, y, groups))
    # The champion arm went through _fold_pass, which splits on the same frame
    # with the same grouping column; GroupKFold is deterministic, so the two
    # lists agree. Phase 3b's paired deltas read both matrices as one fold
    # index, so state it rather than trust it.
    assert all(np.array_equal(a[1], b[1]) for a, b in zip(folds_c, folds)), \
        "champion and candidate fold assignments diverged"
    features_add = list(features) + [KF_COL]
    features_swap = [KF_COL if f == PREV_COL else f for f in features]
    fitter_cand = make_champion_fitter(clf_features)

    oof_add_acc = np.zeros(len(df_eval))
    oof_swap_acc = np.zeros(len(df_eval))
    oof_ctrl_acc = np.zeros(len(df_eval))
    oof_add_by_seed = np.zeros((n_seeds, len(df_eval)))
    oof_swap_by_seed = np.zeros((n_seeds, len(df_eval)))
    oof_ctrl_by_seed = np.zeros((n_seeds, len(df_eval)))
    fr2_add = np.zeros((N_SPLITS, n_seeds))
    fr2sel_add = np.zeros((N_SPLITS, n_seeds))
    fr2_swap = np.zeros((N_SPLITS, n_seeds))
    fr2sel_swap = np.zeros((N_SPLITS, n_seeds))
    fr2_ctrl = np.zeros((N_SPLITS, n_seeds))
    fr2sel_ctrl = np.zeros((N_SPLITS, n_seeds))
    kf_heldout = np.zeros((n_seeds, len(df_eval)))   # each row's own-fold kf
    inner_r2_by_seed = []
    r_by_seed, q_by_seed = [], []

    print(f"\nCandidates (ADD {len(features_add)} / SWAP {len(features_swap)} "
          f"features): nested CV, {N_INNER}-fold inner OOF per outer fold...",
          flush=True)
    for si, seed in enumerate(seeds):
        t_seed = time.time()
        inner_r2s, r_vals, q_vals = [], [], []
        for fi, (tr, va) in enumerate(folds):
            train_df = df_eval.iloc[tr]
            y_tr, g_tr = y[tr], groups[tr]

            # Inner OOF over the outer training rows (grouped by player, so
            # a player's every Year-1 row sits in one inner fold).
            inner_folds = list(GroupKFold(n_splits=N_INNER)
                               .split(train_df, y_tr, g_tr))
            inner_oof = np.full(len(tr), np.nan)
            full_pred = np.zeros((N_INNER, len(needed_idx)))
            player_inner: dict[str, int] = {}
            for j, (itr, iva) in enumerate(inner_folds):
                model, clf = fit_champion_models(train_df.iloc[itr], features,
                                                 clf_features, seed)
                inner_oof[iva] = predict_champion(model, clf,
                                                  train_df.iloc[iva],
                                                  features, clf_features)
                full_pred[j] = predict_champion(model, clf, full_needed,
                                                features, clf_features)
                for p in np.unique(g_tr[iva]):
                    player_inner[p] = j
            full_mean = full_pred.mean(axis=0)

            resid = y_tr - inner_oof
            r_var = float(np.var(resid, ddof=1))
            q_var, _, _ = estimate_q(df_full, set(g_tr), r_var)
            inner_r2s.append(r2_score(y_tr, inner_oof))
            r_vals.append(r_var)
            q_vals.append(q_var)

            # kf for this fold's train AND test rows, from this fold's inner
            # world only.
            def kf_rows(rows):
                out = np.full(len(rows), np.nan)
                for k, i in enumerate(rows):
                    if not np.isfinite(anchor[i]):
                        continue                      # -> median impute below
                    pos_list = inter_pos[i]
                    if not pos_list:
                        out[k] = anchor[i]            # no intermediate seasons
                        continue
                    j = player_inner.get(groups[i])
                    zs = (full_pred[j, pos_list] if j is not None
                          else full_mean[pos_list])
                    # tier-2 anchors are non-market prices: prior variance
                    # depends on anchor kind. Rookie (slotted) and first-
                    # contract (undrafted/2nd-round) get separate P0.
                    if anchor_kind[i] == "first_contract":
                        p0 = P0_FIRST_CONTRACT if P0_FIRST_CONTRACT is not None else r_var * 2
                    elif tier[i] == 2:
                        p0 = r_var
                    else:
                        p0 = P0
                    out[k] = kalman_update(anchor[i], zs, q_var, r_var, p0)
                return out

            kf_tr, kf_va = kf_rows(tr), kf_rows(va)
            kf_heldout[si, va] = np.where(np.isfinite(kf_va), kf_va, prev[va])

            train_22 = train_df.copy()
            test_22 = df_eval.iloc[va].copy()
            train_22[KF_COL] = kf_tr
            test_22[KF_COL] = kf_va
            fill = float(np.nanmedian(kf_tr)) if np.isfinite(kf_tr).any() \
                else float(np.nanmedian(prev))
            train_22[KF_COL] = train_22[KF_COL].fillna(fill)
            test_22[KF_COL] = test_22[KF_COL].fillna(fill)

            vs = sel[va]
            pred_add = fitter_cand(train_22, test_22, features_add, seed)
            oof_add_acc[va] += pred_add
            oof_add_by_seed[si, va] = pred_add
            fr2_add[fi, si] = r2_score(y[va], pred_add)
            fr2sel_add[fi, si] = (r2_score(y[va][vs], pred_add[vs])
                                  if vs.sum() > 10 else np.nan)
            pred_swap = fitter_cand(train_22, test_22, features_swap, seed)
            oof_swap_acc[va] += pred_swap
            oof_swap_by_seed[si, va] = pred_swap
            fr2_swap[fi, si] = r2_score(y[va], pred_swap)
            fr2sel_swap[fi, si] = (r2_score(y[va][vs], pred_swap[vs])
                                   if vs.sum() > 10 else np.nan)

            # Season-dummy control: prev + the TRAIN fold's per-season mean of
            # (kf - prev). Same season/coverage structure as kf, zero
            # player-level content — the supply_samepos is2019 lesson: a
            # feature whose values align with seasons must beat this arm,
            # not just the champion.
            dkf_tr = train_22[KF_COL].values - prev[tr]
            sm = pd.Series(dkf_tr).groupby(seas_all[tr]).mean().to_dict()
            train_23 = train_df.copy()
            test_23 = df_eval.iloc[va].copy()
            train_23[KF_COL] = prev[tr] + np.array(
                [sm.get(s, 0.0) for s in seas_all[tr]])
            test_23[KF_COL] = prev[va] + np.array(
                [sm.get(s, 0.0) for s in seas_all[va]])
            pred_ctrl = fitter_cand(train_23, test_23, features_swap, seed)
            oof_ctrl_acc[va] += pred_ctrl
            oof_ctrl_by_seed[si, va] = pred_ctrl
            fr2_ctrl[fi, si] = r2_score(y[va], pred_ctrl)
            fr2sel_ctrl[fi, si] = (r2_score(y[va][vs], pred_ctrl[vs])
                                   if vs.sum() > 10 else np.nan)

        inner_r2_by_seed.append(float(np.mean(inner_r2s)))
        r_by_seed.append(float(np.mean(r_vals)))
        q_by_seed.append(float(np.mean(q_vals)))
        print(f"  seed {seed}: inner OOF R2={inner_r2_by_seed[-1]:.4f}  "
              f"R={r_by_seed[-1]:.5f}  Q={q_by_seed[-1]:.5f}  "
              f"add [{' '.join(f'{fr2_add[fi, si]:.3f}' for fi in range(N_SPLITS))}]  "
              f"swap [{' '.join(f'{fr2_swap[fi, si]:.3f}' for fi in range(N_SPLITS))}]  "
              f"{time.time() - t_seed:.0f}s", flush=True)

    oof_add = oof_add_acc / n_seeds
    oof_swap = oof_swap_acc / n_seeds
    oof_ctrl = oof_ctrl_acc / n_seeds
    kf_mean = kf_heldout.mean(axis=0)

    # ------------------------------------------------------------------ report
    print(f"\n{line}")
    print("Phase 1 — Inner OOF accuracy (the KF measurement's quality)")
    print(line)
    for si, seed in enumerate(seeds):
        print(f"  Seed {seed}: inner OOF R2 = {inner_r2_by_seed[si]:.4f}")
    print(f"  Mean inner OOF R2: {np.mean(inner_r2_by_seed):.4f}  "
          f"(simple predictor was 0.46)")
    print(f"  Mean R (measurement var): {np.mean(r_by_seed):.5f}  "
          f"(simple predictor was ~0.0037)")
    print(f"  Mean Q (process var):     {np.mean(q_by_seed):.5f}  "
          f"(floor {Q_FLOOR})")

    moved = np.abs(kf_mean - prev)
    has_inter = np.array([bool(lst) for lst in inter_idx])
    print(f"\n{line}")
    print("Phase 2 — KF diagnostic (held-out kf values, seed-averaged)")
    print(line)
    if args.anchor == "market":
        has_a = tier > 0
        print(f"  anchor <-> prev_cap_pct:  "
              f"r={np.corrcoef(anchor, prev)[0, 1]:.3f}   "
              f"|anchor - prev| mean={np.abs(anchor - prev)[has_a].mean():.4f} "
              f"(over the {int(has_a.sum())} anchored rows)")
    print(f"  {KF_COL} <-> prev_cap_pct:  r={np.corrcoef(kf_mean, prev)[0, 1]:.3f}")
    print(f"  {KF_COL} <-> {TARGET}:       r={np.corrcoef(kf_mean, y)[0, 1]:.3f}"
          f"   (prev_cap_pct <-> {TARGET}: r={np.corrcoef(prev, y)[0, 1]:.3f})")
    print(f"  |kf - prev_cap_pct|: mean={moved.mean():.4f}  "
          f"median={np.median(moved):.4f}  "
          f"(rows with intermediates only: mean={moved[has_inter].mean():.4f})")
    print(f"  Rows moved by >0.1% of cap: {int((moved > 0.001).sum())}"
          f"/{len(df_eval)}")
    if args.anchor == "market":
        for t_id, lab in ((1, "tier 1 (market anchor)"),
                          (2, "tier 2 (rookie anchor)")):
            m = tier == t_id
            if m.any():
                print(f"  {lab}: n={int(m.sum())}  "
                      f"|kf-prev| mean={moved[m].mean():.4f}  "
                      f"moved>0.1%cap {int((moved[m] > 0.001).sum())}  "
                      f"r(kf,y)={np.corrcoef(kf_mean[m], y[m])[0, 1]:.3f} "
                      f"(r(prev,y)={np.corrcoef(prev[m], y[m])[0, 1]:.3f})")

    recent = df_eval["season"].values >= 2024
    mae_c, _ = _dollars(df_eval, oof_c)

    def _arm_line(tag, oof_x, n_feat):
        mae_x, _ = _dollars(df_eval, oof_x)
        print(f"  {tag:28s} A1={r2_score(y, oof_x):.4f}  "
              f"sel={r2_score(y[sel], oof_x[sel]):.4f}  "
              f"A2={r2_score(y[recent], oof_x[recent]):.4f}  "
              f"MAE=${mae_x:.2f}M  ({n_feat} features)")
        return mae_x

    def _delta_lines(fr2sel_x, fr2_x):
        d_sel = paired_delta(fr2sel_c, fr2sel_x)
        d_pool = paired_delta(fr2_c, fr2_x)
        ok = d_sel["t"] > 2
        print(f"    dSel={d_sel['delta']:+.5f} +/- {d_sel['se']:.5f}  "
              f"t={d_sel['t']:+.2f}  [{'PASS' if ok else 'FAIL'}]  "
              f"dPooled={d_pool['delta']:+.5f} t={d_pool['t']:+.2f}")
        print(f"    folds (selection): "
              + " ".join(f"{v:+.5f}" for v in d_sel["per_fold"]))
        return d_sel, d_pool, ok

    print(f"\n{line}")
    print("Phase 3 — Ablation (paired, identical folds and seeds; "
          "gate: selection-pool t > 2)")
    print(line)
    print("  pipeline: push -> clip -> extension clip. Stage 3's signing "
          "offset is NOT attached here,")
    print("  so these R2 are the deployed pipeline minus its last stage — "
          "Phase 3b is the comparable one.")
    _arm_line("Champion (prev_cap_pct)", oof_c, len(features))
    mae_add = _arm_line(f"ADD  + {KF_COL}", oof_add, len(features_add))
    d_sel_add, d_pool_add, pass_add = _delta_lines(fr2sel_add, fr2_add)
    mae_swap = _arm_line(f"SWAP prev_cap_pct -> kf", oof_swap,
                         len(features_swap))
    d_sel_swap, d_pool_swap, pass_swap = _delta_lines(fr2sel_swap, fr2_swap)
    mae_ctrl = _arm_line("CTRL season-dummy", oof_ctrl, len(features_swap))
    d_sel_ctrl = paired_delta(fr2sel_c, fr2sel_ctrl)
    d_pool_ctrl = paired_delta(fr2_c, fr2_ctrl)
    share = (d_sel_ctrl["delta"] / d_sel_swap["delta"]
             if d_sel_swap["delta"] else float("nan"))
    print(f"    dSel={d_sel_ctrl['delta']:+.5f} t={d_sel_ctrl['t']:+.2f}  "
          f"dPooled={d_pool_ctrl['delta']:+.5f}")
    print(f"    season-structure-only arm recovers {share:.0%} of the swap "
          f"gain (is2019 recovered 70% of supply_samepos -> reject; the swap "
          f"must clearly beat this arm)")

    # Phase 3b — the same four arms with Stage 3's signing offset attached.
    # Phase 3 is the pre-registered gate and stays where it was decided; this
    # block exists because Phase 3's numbers are one stage short of what
    # evaluate_suite reports, so quoting them side by side reads as a
    # regression that is really a missing stage. VERSION_HISTORY should carry
    # THESE, and say so.
    print(f"\n{line}")
    print(f"Phase 3b — Same arms, Stage-3 signing offset attached "
          f"(leave-fold-out, k={SIGNING_K})")
    print(line)
    print("  pipeline: push -> clip -> signing offset -> mech cap -> extension "
          "clip -> re-clip.")
    print("  Comparable to evaluate_suite's champion row; Phase 3 above is not.")
    s3 = {}
    for tag, mat, lab, n_feat in (
            ("champion", oof_c_by_seed, "Champion (prev_cap_pct)", len(features)),
            ("add", oof_add_by_seed, f"ADD  + {KF_COL}", len(features_add)),
            ("swap", oof_swap_by_seed, "SWAP prev_cap_pct -> kf",
             len(features_swap)),
            ("control", oof_ctrl_by_seed, "CTRL season-dummy",
             len(features_swap))):
        oof_x, fr2_x, fr2sel_x, lfo_x = signing_corrected(mat, df_eval, folds,
                                                          sel)
        s3[tag] = {"oof": oof_x, "fold_r2": fr2_x, "fold_r2_sel": fr2sel_x,
                   "lfo": lfo_x, "mae_m": _arm_line(lab, oof_x, n_feat)}
        if tag != "champion":
            d_sel_x = paired_delta(s3["champion"]["fold_r2_sel"], fr2sel_x)
            d_pool_x = paired_delta(s3["champion"]["fold_r2"], fr2_x)
            s3[tag]["paired_selection"] = d_sel_x
            s3[tag]["paired_pooled"] = d_pool_x
            print(f"    dSel={d_sel_x['delta']:+.5f} +/- {d_sel_x['se']:.5f}  "
                  f"t={d_sel_x['t']:+.2f}  "
                  f"dPooled={d_pool_x['delta']:+.5f} t={d_pool_x['t']:+.2f}")
    print(f"  Stage-3 lift on the shipped arm (SWAP): "
          f"A1 {r2_score(y, oof_swap):.4f} -> {r2_score(y, s3['swap']['oof']):.4f}"
          f"   MAE ${mae_swap:.2f}M -> ${s3['swap']['mae_m']:.2f}M")
    print("  Offsets (fold 0, cap_pct; positive RAISES an underpriced type):")
    for t_name, d in sorted(s3["swap"]["lfo"][0].items()):
        print(f"    {t_name:14s} n={d['n']:4d}  raw={d['raw']:+.5f}  "
              f"shrunk={d['offset']:+.5f}")

    # Phase 4 — the 2x3 view: champion (prev_cap_pct) vs SWAP (kf) on all
    # rows, on the tier-3 fallback segment (kf == prev by construction — any
    # movement here is collateral from the shared model, not from the
    # feature's own values), and on the kf-active segment where the feature
    # actually differs. Fold-paired deltas; a segment R2 cell needs >= 10
    # validation rows.
    seg_json = {}
    if args.anchor == "market":
        cap_m = df_eval["cap"].values / 1e6

        def seg_stats(m):
            mat_c = np.full((N_SPLITS, n_seeds), np.nan)
            mat_s = np.full((N_SPLITS, n_seeds), np.nan)
            for fi, (_tr, va) in enumerate(folds):
                vm = va[m[va]]
                if len(vm) < 10:
                    continue
                for si in range(n_seeds):
                    mat_c[fi, si] = r2_score(y[vm], oof_c_by_seed[si, vm])
                    mat_s[fi, si] = r2_score(y[vm], oof_swap_by_seed[si, vm])
            ok = ~np.isnan(mat_c).any(axis=1)
            d = paired_delta(mat_c[ok], mat_s[ok]) if ok.sum() >= 3 else None
            return {
                "n": int(m.sum()),
                "champ_r2": float(r2_score(y[m], oof_c[m])),
                "swap_r2": float(r2_score(y[m], oof_swap[m])),
                "champ_mae_m": float(np.abs((oof_c[m] - y[m]) * cap_m[m]).mean()),
                "swap_mae_m": float(np.abs((oof_swap[m] - y[m]) * cap_m[m]).mean()),
                "paired": d, "n_folds_scored": int(ok.sum()),
            }

        print(f"\n{line}")
        print("Phase 4 — kf vs prev_cap_pct by segment "
              "(champion vs SWAP, fold-paired)")
        print(line)
        for lab, m in (("all rows", np.ones(len(df_eval), bool)),
                       ("tier 3: kf=prev fallback", tier == 0),
                       ("kf-active: tier 1+2", tier > 0)):
            s = seg_stats(m)
            d = s["paired"]
            dtxt = (f"dR2={d['delta']:+.5f} t={d['t']:+.2f} "
                    f"({s['n_folds_scored']} folds)" if d else "n too small")
            print(f"  {lab:28s} n={s['n']:3d}  "
                  f"R2 {s['champ_r2']:.4f} -> {s['swap_r2']:.4f}  {dtxt}")
            print(f"  {'':28s} MAE ${s['champ_mae_m']:.2f}M -> "
                  f"${s['swap_mae_m']:.2f}M "
                  f"({s['swap_mae_m'] - s['champ_mae_m']:+.2f})")
            seg_json[lab] = s

    # Phase 5 — C-layer guards for the SWAP candidate (evaluate_suite rules):
    # C1 relative calibration gate |slope-1| excess <= 0.005; C2 no signing
    # mechanism's |bias| may grow by more than $0.3M.
    cap_m_all = df_eval["cap"].values / 1e6
    slope_c = float(np.polyfit(oof_c, y, 1)[0])
    slope_s = float(np.polyfit(oof_swap, y, 1)[0])
    c1_excess = abs(slope_s - 1.0) - abs(slope_c - 1.0)
    print(f"\n{line}")
    print("Phase 5 — C-layer guards (champion vs SWAP)")
    print(line)
    print(f"  C1 calibration slope {slope_c:.4f} -> {slope_s:.4f}   "
          f"|slope-1| excess {c1_excess:+.4f}  "
          f"[{'PASS' if c1_excess <= 0.005 else 'FAIL'}] (gate <= +0.005)")
    c2_json, worst_cat, worst_growth = {}, None, -np.inf
    for cat, sub in df_eval.groupby("signing_cat"):
        m = (df_eval["signing_cat"] == cat).values
        if m.sum() < 10:
            continue
        err_c = (oof_c[m] - y[m]) * cap_m_all[m]
        err_s = (oof_swap[m] - y[m]) * cap_m_all[m]
        growth = abs(float(err_s.mean())) - abs(float(err_c.mean()))
        c2_json[str(cat)] = {
            "n": int(m.sum()), "champ_bias_m": float(err_c.mean()),
            "swap_bias_m": float(err_s.mean()), "abs_bias_growth_m": growth,
            "champ_mae_m": float(np.abs(err_c).mean()),
            "swap_mae_m": float(np.abs(err_s).mean())}
        if growth > worst_growth:
            worst_cat, worst_growth = str(cat), growth
    print(f"  C2 bias by signing mechanism (|bias| growth, gate <= +$0.3M):")
    for cat, d in sorted(c2_json.items(), key=lambda kv: -kv[1]["n"]):
        print(f"    {cat:14s} n={d['n']:4d}  bias ${d['champ_bias_m']:+5.2f}M "
              f"-> ${d['swap_bias_m']:+5.2f}M   growth "
              f"{d['abs_bias_growth_m']:+5.2f}")
    print(f"  worst: {worst_cat} {worst_growth:+.2f}M  "
          f"[{'PASS' if worst_growth <= 0.3 else 'FAIL'}]")

    # ------------------------------------------------------------------
    # Phase 6: B1 forward evaluation (rolling-origin)
    # ------------------------------------------------------------------
    print(f"\n{line}")
    print("Phase 6 — B1 forward (rolling-origin, train on seasons < T)")
    print(line)
    print("  The temporal split is naturally fold-honest for kf: the base")
    print("  model is trained on seasons < T only, so all intermediate")
    print("  predictions and KF values for season T are out-of-sample.")
    print("  Stage 3 signing offsets are from inner OOF within each")
    print("  training window (the rolling_forward_signing pattern).")

    FWD_TAGS = ("champion", "add", "swap", "control")
    fwd_s2 = {t: np.full(len(df_eval), np.nan) for t in FWD_TAGS}
    fwd_s3 = {t: np.full(len(df_eval), np.nan) for t in FWD_TAGS}
    fwd_b1_detail = {}
    cat_all = df_eval["signing_cat"].values
    lo_all = df_eval["floor_pct"].values
    hi_all = df_eval["max_eligible_pct"].values
    is_ext_all = df_eval["is_extension"].values
    ext_cap_all = df_eval["ext_cap_pct"].values
    mech_all = (df_eval["mech_cap_pct"].values
                if "mech_cap_pct" in df_eval.columns else None)

    fitter_fwd = make_champion_fitter(clf_features)

    for T in FORWARD_ORIGINS:
        te, tr = seas_all == T, seas_all < T
        n_te, n_tr = int(te.sum()), int(tr.sum())
        if n_te < 10 or n_tr < 200:
            print(f"  origin {T}: skipped (n_test={n_te}, n_train={n_tr})")
            continue
        t_o = time.time()
        print(f"\n  origin {T}: train {n_tr}, test {n_te}", flush=True)

        # --- KF measurement: base model on seasons < T ------------------
        # Fit the 21-feature prev_cap_pct model on training rows, predict
        # intermediate rows for KF updates and training rows for R/Q.
        full_p = np.zeros(len(needed_idx))
        tr_p = np.zeros(n_tr)
        for seed in seeds:
            mdl, clr = fit_champion_models(
                df_eval[tr], features, clf_features, seed)
            full_p += predict_champion(
                mdl, clr, full_needed, features, clf_features)
            tr_p += predict_champion(
                mdl, clr, df_eval[tr], features, clf_features)
        full_p /= n_seeds
        tr_p /= n_seeds

        r_fwd = float(np.var(y[tr] - tr_p, ddof=1))
        q_fwd, _, _ = estimate_q(df_full, set(groups[tr]), r_fwd)
        print(f"    KF params: R={r_fwd:.5f}  Q={q_fwd:.5f}", flush=True)

        # --- KF for all eval rows ----------------------------------------
        kf_fwd = np.full(len(df_eval), np.nan)
        for i in range(len(df_eval)):
            if not np.isfinite(anchor[i]):
                continue
            pl = inter_pos[i]
            if not pl:
                kf_fwd[i] = anchor[i]
                continue
            if anchor_kind[i] == "first_contract":
                p0_i = P0_FIRST_CONTRACT if P0_FIRST_CONTRACT is not None else r_fwd * 2
            elif tier[i] == 2:
                p0_i = r_fwd
            else:
                p0_i = P0
            kf_fwd[i] = kalman_update(
                anchor[i], full_p[pl], q_fwd, r_fwd, p0_i)

        fill_kf = (float(np.nanmedian(kf_fwd[tr]))
                   if np.isfinite(kf_fwd[tr]).any()
                   else float(np.nanmedian(prev)))

        # Augmented frames: SWAP/ADD with kf_market_value
        train_kf_fwd = df_eval[tr].copy()
        test_kf_fwd = df_eval[te].copy()
        train_kf_fwd[KF_COL] = np.where(
            np.isfinite(kf_fwd[tr]), kf_fwd[tr], fill_kf)
        test_kf_fwd[KF_COL] = np.where(
            np.isfinite(kf_fwd[te]), kf_fwd[te], fill_kf)

        # CONTROL: season-dummy kf
        dkf_fwd = train_kf_fwd[KF_COL].values - prev[tr]
        sm_fwd = pd.Series(dkf_fwd).groupby(seas_all[tr]).mean().to_dict()
        train_ctrl_fwd = df_eval[tr].copy()
        test_ctrl_fwd = df_eval[te].copy()
        train_ctrl_fwd[KF_COL] = prev[tr] + np.array(
            [sm_fwd.get(s, 0.0) for s in seas_all[tr]])
        test_ctrl_fwd[KF_COL] = prev[te] + np.array(
            [sm_fwd.get(s, 0.0) for s in seas_all[te]])

        # --- Stage 2 forward predictions ---------------------------------
        acc_fwd = {t: np.zeros(n_te) for t in FWD_TAGS}
        for seed in seeds:
            acc_fwd["champion"] += fitter_fwd(
                df_eval[tr], df_eval[te], features, seed)
            acc_fwd["add"] += fitter_fwd(
                train_kf_fwd, test_kf_fwd, features_add, seed)
            acc_fwd["swap"] += fitter_fwd(
                train_kf_fwd, test_kf_fwd, features_swap, seed)
            acc_fwd["control"] += fitter_fwd(
                train_ctrl_fwd, test_ctrl_fwd, features_swap, seed)
        for t in FWD_TAGS:
            fwd_s2[t][te] = acc_fwd[t] / n_seeds

        s2_r2 = {t: float(r2_score(y[te], fwd_s2[t][te])) for t in FWD_TAGS}
        print(f"    Stage 2: champ={s2_r2['champion']:.4f}  "
              f"add={s2_r2['add']:.4f}  swap={s2_r2['swap']:.4f}  "
              f"ctrl={s2_r2['control']:.4f}", flush=True)

        # --- Stage 3: signing offset from inner OOF ---------------------
        # Each arm gets its own signing offsets from its own inner-OOF
        # residuals within the training window (same pattern as Phase 3b
        # and rolling_forward_signing in evaluate_suite).
        origin_signing = {}
        for tag, train_aug, feats in (
            ("champion", df_eval[tr], features),
            ("add", train_kf_fwd, features_add),
            ("swap", train_kf_fwd, features_swap),
            ("control", train_ctrl_fwd, features_swap),
        ):
            inner_oof_fwd, _, _ = oof_groupkfold(
                train_aug, feats, fitter_fwd, seeds)
            offs_fwd = signing_offsets(
                y[tr] - inner_oof_fwd,
                train_aug["signing_cat"].values,
                k=SIGNING_K, detail=True)
            fwd_s3[tag][te] = stage3_signing(
                fwd_s2[tag][te], cat_all[te], offs_fwd,
                lo=lo_all[te], hi=hi_all[te],
                mech_cap_pct=(mech_all[te] if mech_all is not None
                              else None),
                is_extension=is_ext_all[te],
                ext_cap_pct=ext_cap_all[te])
            origin_signing[tag] = {
                t: {"offset": d["offset"], "n": d["n"]}
                for t, d in offs_fwd.items()
            }

        s3_r2 = {t: float(r2_score(y[te], fwd_s3[t][te])) for t in FWD_TAGS}
        print(f"    Stage 3: champ={s3_r2['champion']:.4f}  "
              f"add={s3_r2['add']:.4f}  swap={s3_r2['swap']:.4f}  "
              f"ctrl={s3_r2['control']:.4f}  "
              f"({time.time() - t_o:.0f}s)", flush=True)

        fwd_b1_detail[int(T)] = {
            "n_tr": n_tr, "n_te": n_te, "r": r_fwd, "q": q_fwd,
            "stage2": s2_r2, "stage3": s3_r2,
            "signing_offsets": origin_signing,
        }

    # --- Pooled B1 summary -----------------------------------------------
    scored_fwd = ~np.isnan(fwd_s2["champion"])
    m2026 = scored_fwd & (seas_all == 2026)
    b1_json = {"by_origin": fwd_b1_detail}

    print(f"\n  B1 summary -- Stage 2 (push -> clip -> extension clip):")
    b1_s2_json = {}
    for tag in FWD_TAGS:
        r2p = float(r2_score(y[scored_fwd], fwd_s2[tag][scored_fwd]))
        mae_p, _ = _dollars(df_eval, fwd_s2[tag], scored_fwd)
        r2_26 = (float(r2_score(y[m2026], fwd_s2[tag][m2026]))
                 if m2026.sum() >= 10 else float("nan"))
        print(f"    {tag:10s} rolling={r2p:.4f}  2026={r2_26:.4f}  "
              f"MAE=${mae_p:.2f}M")
        b1_s2_json[tag] = {"b1_rolling_r2": r2p, "b1_2026_r2": r2_26,
                           "b1_mae_m": mae_p}
    b1_json["stage2"] = b1_s2_json

    print(f"\n  B1 summary -- Stage 3 (+ signing offset from inner OOF):")
    b1_s3_json = {}
    for tag in FWD_TAGS:
        r2p = float(r2_score(y[scored_fwd], fwd_s3[tag][scored_fwd]))
        mae_p, _ = _dollars(df_eval, fwd_s3[tag], scored_fwd)
        r2_26 = (float(r2_score(y[m2026], fwd_s3[tag][m2026]))
                 if m2026.sum() >= 10 else float("nan"))
        print(f"    {tag:10s} rolling={r2p:.4f}  2026={r2_26:.4f}  "
              f"MAE=${mae_p:.2f}M")
        b1_s3_json[tag] = {"b1_rolling_r2": r2p, "b1_2026_r2": r2_26,
                           "b1_mae_m": mae_p}
    b1_json["stage3"] = b1_s3_json

    out_dir = OUTPUTS_DIR / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{args.anchor}{'_prehistory' if args.prehistory else ''}{'_expand' if args.expand_anchors else ''}"
    if args.p0 is not None:
        suffix += f"_p0{args.p0:g}"
    if args.q_floor is not None:
        suffix += f"_qf{args.q_floor:g}"
    payload = {
        "n": len(df_eval), "seeds": seeds, "n_inner": N_INNER,
        "anchor_mode": args.anchor,
        "prehistory": bool(args.prehistory),
        "expand_anchors": bool(args.expand_anchors),
        "p0": P0, "q_floor": Q_FLOOR,
        "anchor_rows": n_anchor, "rows_with_intermediates": int(n_inter),
        "tiers": {"tier1": int((tier == 1).sum()),
                  "tier1_prorated_min": int(anchor_prorated[tier == 1].sum()),
                  "tier2": int((tier == 2).sum()),
                  "tier2_rookie_y2": int((anchor_kind == "rookie_y2").sum()),
                  "tier2_rookie_y1": int((anchor_kind == "rookie_y1").sum()),
                  "tier2_first_contract": int((anchor_kind == "first_contract").sum()),
                  "tier3": int((tier == 0).sum())},
        "inner_oof_r2_by_seed": inner_r2_by_seed,
        "r_by_seed": r_by_seed, "q_by_seed": q_by_seed,
        "kf_diag": {
            "corr_kf_prev": float(np.corrcoef(kf_mean, prev)[0, 1]),
            "corr_kf_y": float(np.corrcoef(kf_mean, y)[0, 1]),
            "corr_prev_y": float(np.corrcoef(prev, y)[0, 1]),
            "abs_move_mean": float(moved.mean()),
            "abs_move_median": float(np.median(moved)),
            "rows_moved_gt_0p001": int((moved > 0.001).sum()),
        },
        "champion": {"a1": float(r2_score(y, oof_c)),
                     "sel": float(r2_score(y[sel], oof_c[sel])),
                     "a2": float(r2_score(y[recent], oof_c[recent])),
                     "mae_m": mae_c},
        "candidate_add": {"a1": float(r2_score(y, oof_add)),
                          "sel": float(r2_score(y[sel], oof_add[sel])),
                          "a2": float(r2_score(y[recent], oof_add[recent])),
                          "mae_m": mae_add,
                          "paired_selection": d_sel_add,
                          "paired_pooled": d_pool_add,
                          "gate_pass": bool(pass_add)},
        "candidate_swap": {"a1": float(r2_score(y, oof_swap)),
                           "sel": float(r2_score(y[sel], oof_swap[sel])),
                           "a2": float(r2_score(y[recent], oof_swap[recent])),
                           "mae_m": mae_swap,
                           "paired_selection": d_sel_swap,
                           "paired_pooled": d_pool_swap,
                           "gate_pass": bool(pass_swap)},
        "candidate_control": {"a1": float(r2_score(y, oof_ctrl)),
                              "sel": float(r2_score(y[sel], oof_ctrl[sel])),
                              "mae_m": mae_ctrl,
                              "paired_selection": d_sel_ctrl,
                              "paired_pooled": d_pool_ctrl,
                              "share_of_swap_gain": float(share)},
        "c_layer": {"c1_slope_champ": slope_c, "c1_slope_swap": slope_s,
                    "c1_excess": float(c1_excess),
                    "c1_pass": bool(c1_excess <= 0.005),
                    "c2_by_mechanism": c2_json,
                    "c2_worst": worst_cat,
                    "c2_worst_growth_m": float(worst_growth),
                    "c2_pass": bool(worst_growth <= 0.3)},
        # The same four arms carrying Stage 3's signing offset. `candidate_*`
        # above stop at the extension clip; these are what evaluate_suite's
        # champion row can be read against.
        "stage3_signing": {
            "k": SIGNING_K,
            "arms": {tag: {"a1": float(r2_score(y, d["oof"])),
                           "sel": float(r2_score(y[sel], d["oof"][sel])),
                           "a2": float(r2_score(y[recent], d["oof"][recent])),
                           "mae_m": d["mae_m"],
                           "paired_selection": d.get("paired_selection"),
                           "paired_pooled": d.get("paired_pooled")}
                     for tag, d in s3.items()},
            "offsets_by_fold": {str(fi): v
                                for fi, v in s3["swap"]["lfo"].items()},
        },
        "segments_kf_vs_prev": seg_json,
        "fold_r2_selection": {"champion": fr2sel_c.tolist(),
                              "candidate_add": fr2sel_add.tolist(),
                              "candidate_swap": fr2sel_swap.tolist()},
        "fold_r2": {"champion": fr2_c.tolist(),
                    "candidate_add": fr2_add.tolist(),
                    "candidate_swap": fr2_swap.tolist()},
        "forward_b1": b1_json,
        "runtime_s": round(time.time() - t_start, 1),
    }
    out = out_dir / f"ablation_kf_market_value_full_{suffix}.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    ref = df_eval[["player_name_norm", "season", TARGET, "signing_cat",
                   "is_confirmation"]].copy()
    ref["tier"] = tier
    ref["anchor_kind"] = anchor_kind
    ref["prev_cap_pct"] = prev
    ref["kf_mean_heldout"] = kf_mean
    ref["oof_champion"] = oof_c
    ref["oof_add"] = oof_add
    ref["oof_swap"] = oof_swap
    ref["oof_control"] = oof_ctrl
    # ..._s3 = the same arm with Stage 3's signing offset. Both are kept so a
    # later reader can see which stage a number came from without a rerun.
    for tag in ("champion", "add", "swap", "control"):
        ref[f"oof_{tag}_s3"] = s3[tag]["oof"]
    for tag in FWD_TAGS:
        ref[f"fwd_{tag}_s2"] = fwd_s2[tag]
        ref[f"fwd_{tag}_s3"] = fwd_s3[tag]
    ref_path = out_dir / f"ablation_kf_oof_{suffix}.csv"
    ref.to_csv(ref_path, index=False)
    print(f"\nSaved {out}")
    print(f"Saved {ref_path} ({len(ref)} rows)   "
          f"(total {time.time() - t_start:.0f}s)")


if __name__ == "__main__":
    main()
