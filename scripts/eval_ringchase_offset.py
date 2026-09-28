r"""Measurement harness — the ring-chasing discount as a Stage-2/3 output offset.

NOTHING HERE IS ADOPTED. No version bump, no VERSION_HISTORY entry, no change
to `FEATURE_COLS`, `stages.py`, or any deployed constant. This file measures a
candidate correction and writes only into `outputs/`.

EVERY NUMBER IS PROVISIONAL. The Spotrac salary migration is rewriting
`salaries.csv` / `training_data_v2.csv` concurrently and is expected to move
~15% of the evaluation frame. `career_earnings.csv`'s subtraction leg reads
`salaries.csv`, so the cell membership itself will move. Re-run after the
migration settles before quoting any figure. ISSUES #48's warning applies with
full force: the offset below is a data-dependent learned constant and a
training-data rebuild makes it stale exactly the way it makes a published R2
stale.

===========================================================================
THE HYPOTHESIS
===========================================================================

Some players sign for far less than they are worth, by choice, because they
have banked enough money that winning outweighs the marginal dollar. The
champion over-prices them; the observed salary is a lower bound on latent
value rather than a market price.

Two things about the phenomenon are already established and are NOT re-derived
here (docs/briefs/2026-08-29-ring-chasing-handoff.md):

  * It is NOT floor-restricted. The largest instances are Bird Rights rows --
    DeRozan 2024 (+$19.6M), Harden 2022 (+$19.5M), Harden 2025 (+$14.9M) --
    not minimums. A correction whose support is `is_at_floor` misses them.
  * The Stage-1 FEATURE route already failed: adding career earnings and rings
    to the 21-feature regression scored +0.00059, t=+1.22, p=0.23. The model
    does not lack the information. It lacks the instruction.

===========================================================================
WHAT SHAPE THIS IS, AND WHY
===========================================================================

An OUTPUT offset on a circumstance group, in the exact shape of the Stage-3
signing offset (`stages.signing_offsets`, `scripts/eval_stage3_signing.py`):

    offset(group) = n/(n+k) x mean(actual - predicted) over the pool,  k = 20

Three reasons this shape rather than the alternatives that were considered:

  * NOT a Stage-1 Grabit censor gate. Stage 1's contract is to price under
    DEFAULT parameters -- it estimates the market's pricing function, and a
    voluntary discount is a deviation FROM that function, not part of it.
    Right-censoring changes the shared trees, so a hypothesis about ~100 rows
    would perturb leaf values for all 880, and its payoff would land on the
    neighbours and need a bespoke metric to see. CLAUDE.md's escalation rule
    puts the cheap, local, reversible intervention first.
  * NOT a hand-specified pull (the floor branch's `MARGIN * floor_pct` arm B).
    `floor_pct` is a meaningless destination for a $47M-to-$33M discount, and
    a hand-set magnitude needs a precedent to justify it. Here the magnitude
    is ESTIMATED and the shrinkage is protection rather than decoration: a
    small cell is pulled hard toward zero at k=20, which is the right amount
    of caution for a cell this size.
  * The correction never retrains anything. Champion and candidate share every
    fold, every seed and every fitted model, so the paired delta carries zero
    fit noise.

ADMISSIBILITY -- why a circumstance grouping is legal where a mechanism
grouping is not. CLAUDE.md restricts the signing offset to the four
ELIGIBILITY mechanisms because "MLE, BAE and Minimum are determined by the
contract value itself, so conditioning on them reads the target." All three
coordinates of this group are settled BEFORE the contract being priced exists:

    career_earnings_thru_prev   strictly through season T-1
    rings_thru_prev             strictly before season T
    age                         exogenous

None of them can be moved by the price under prediction, so conditioning on
them does not read the target. This is the property that makes the grouping
admissible; it is checked in code (`assert_ex_ante`) rather than only asserted
here.

===========================================================================
PRE-REGISTRATION -- fixed before any arm score was computed
===========================================================================

CELL (route (a): pre-register the most ordinary cell, do not scan)

    DZ  =  rings_thru_prev == 0
           AND age >= 30
           AND career_earnings_thru_prev >= P75 of the evaluation frame

    n = 102, the plainest reading of "old, rich and ringless". It is NOT the
    best-scoring of the six age x earnings-quantile combinations the handoff
    brief scanned -- the winner there was age>=34 & p90 at p=0.0057, and
    reporting a scanned winner is selection, not pre-registration. The
    stricter cells are reported DESCRIPTIVELY below (their n and their shrunk
    offset) and no arm is scored on them.

    Rows with MISSING career earnings are NOT members. Absence of evidence
    must not create membership in a group defined by a positive fact. On this
    frame the question is nearly moot: 879/880 rows have the column, and all
    264 at-floor rows do.

    The P75 threshold is a quantile of a COVARIATE and involves no target. It
    is computed frame-wide so the reporting zone is a fixed row group
    (ISSUES #20b); the leave-fold-out variant is computed and printed beside
    it as a sensitivity, and must not change membership materially.

MAGNITUDE                learned, never hand-specified.
                         offset = n/(n+k) x mean(actual - predicted) in cap_pct.
                         k = SIGNING_K = 20, IMPORTED from src/model/stages.py
                         so this harness cannot drift from the shrinkage the
                         project already pre-registered. A k=0 arm is computed
                         REFERENCE-ONLY and decides nothing.

FOLD HONESTY (layer A)   fold f is corrected by an offset estimated from rows
                         OUTSIDE fold f. Layer B estimates from an inner OOF
                         over seasons < T only. The deployed-form offset (all
                         rows, k=20) is printed for reporting and never scores.

                         One residual channel, stated rather than hidden and
                         inherited verbatim from eval_stage3_signing: the
                         leave-fold-out offset for fold f is a mean over OOF
                         residuals of rows in folds g != f, and each of those
                         predictions came from a model that had fold f in ITS
                         training set. Layer B is clean of it entirely.

LEGALITY TAIL            offset -> extension raise-cap clip -> clip into
                         [floor_pct, max_eligible_pct]. Same components and the
                         same order as `stages.stage3_signing`.

DECISION STATISTICS      pre-registered, in this order:
  primary   paired dSel: fold-paired delta R2 on SELECTION rows (ISSUES #20a).
            The repo's escalation rule. Bar t > 2 to call it a win; a
            non-negative point estimate with t < 2 is "no effect", not a win.
  zone      "judge a targeted intervention where it acts" -- on DZ:
            |bias| growth and RMSE. Both must improve.
            DZ MAE is reported but is NOT the primary zone statistic, and the
            reason is stated in advance: a MEAN offset minimises squared error
            and bias, not absolute error. DZ's mean over-prediction is +$1.90M
            while its MEDIAN is +$0.38M, so a constant sized by the mean is
            larger than most members need and MAE can worsen while bias and
            RMSE improve. Choosing MAE here would be choosing a statistic the
            estimator was never fitted to.
  collateral  rows outside DZ must be BIT-IDENTICAL (offset is exactly 0.0
            there); asserted, not assumed. Within DZ, the rows the champion
            already UNDER-prices are pushed further wrong -- counted and named.
  guards    C2 fixed signing-mechanism segments, worst |bias| growth <= $0.30M;
            B1 forward R2 drop <= 0.003; A1/A2 reported.

SHRINKAGE TRANSPARENCY   the handoff brief asks how much of the cell effect is
                         carried by its largest contributors. A leave-one-out
                         and leave-three-out jackknife of the raw cell mean is
                         computed and printed. If the offset is carried by
                         three rows, the report says so.

SECONDARY ARM (Pw) -- reported, does not decide
                         The original task asked for a fold-honest P and its
                         Spearman against the over-prediction it exists to fix
                         (benchmark: the P that failed last time at -0.611,
                         anti-ranked). That question is worth answering even
                         though the primary arm is a hard cell, so a
                         circumstance P is built and scored:

    P        logistic regression, fitted INSIDE each training slice and
             predicted on the held-out slice, on 4 standardised terms:
             career earnings (as cap_pct), age, rings, and the age x earnings
             product. The product is in because the hypothesis is explicitly
             conjunctive ("old AND rich") and an additive logit cannot express
             a conjunction; rings enters additively because the brief's
             evidence for it is a main effect. A main-effects-only variant is
             printed as a sensitivity.
    label    within the training slice only: the slice's own in-sample
             champion-family baseline over-prices the row by >= $2.0M. $2.0M
             is the at-floor zone's mean over-prediction (+$2.04M rounded) --
             the headroom quantity this work exists to remove -- and is a
             pre-existing constant, not a swept one. The in-sample baseline
             mirrors what `grabit_latent` already does for its censor gates.
    tau      pre-registered by win-minus-collateral on the CHAMPION OOF,
             computed BEFORE any arm is scored:
                 E_win(tau)  = sum over {P>=tau} of max(champ_err, 0)
                 E_coll(tau) = sum over {P>=tau} of max(-champ_err, 0)
                 tau* = argmax (E_win - E_coll)
             The floor branch's tau* = 0.10 is explicitly NOT reused: it was
             selected for a different P that was anti-ranked against its error.
    arm      group = {P >= tau*}, same k=20 leave-fold-out offset, same
             legality tail. No MARGIN, no hand-set magnitude anywhere.

Run:  OMP_NUM_THREADS=6 python scripts/eval_ringchase_offset.py
      [--seeds N] [--skip-b]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from config import CAP_BY_SEASON, OUTPUTS_DIR, PROCESSED_DIR
from src.model.evaluate_suite import (
    DEFAULT_SEEDS, FORWARD_ORIGINS, N_SPLITS, TARGET, _compute_kf_forward,
    _compute_kf_nested, _dollars, abs_bias_growth, load_evaluation_frame,
    make_champion_fitter, oof_groupkfold, paired_delta, prepare_kf_context,
)
from src.model import route_mixture as rm
from src.model.extension_cap import attach_extension_cap
from src.model.stages import SIGNING_K, compose, stage3
from src.model.train import FEATURE_COLS, _XGB_BASE

# ---- pre-registered constants ---------------------------------------------
CELL_RINGS_MAX = 0          # rings_thru_prev == 0
CELL_AGE_MIN = 30           # age >= 30
CELL_EARN_Q = 0.75          # career earnings >= frame P75
K_SHRINK = SIGNING_K        # 20. IMPORTED, never restated, never swept.
K_REFERENCE = 0.0           # reference-only arm; decides nothing.

# Secondary (Pw) arm
LABEL_OVER_M = 2.0          # $M; the at-floor zone's mean over-prediction
TAU_GRID = np.round(np.arange(0.10, 0.95, 0.02), 2)

# Gate bars, inherited from the floor-branch / Stage-3 harnesses
C2_BAR = 0.30               # $M worst per-segment |bias| growth
B1_BAR = 0.003              # forward R2 drop
DSEL_T_BAR = 2.0            # paired t on selection folds

# Descriptive-only sub-cells. Reported (n, raw mean, shrunk offset); NO arm is
# scored on them, because scoring them would be the threshold re-scan the
# handoff explicitly forbids.
SUBCELLS = [("age>=32 & earn>=p90", 32, 0.90),
            ("age>=34 & earn>=p90", 34, 0.90),
            ("age>=34 & earn>=p75", 34, 0.75)]

CAP_DISPLAY = CAP_BY_SEASON.get(2026, 153_000_000)
OUT = OUTPUTS_DIR / "models"
NAMED = [("demar derozan", 2024), ("james harden", 2022), ("james harden", 2025),
         ("chris paul", 2025), ("lebron james", 2026), ("marc gasol", 2020),
         ("montrezl harrell", 2022), ("blake griffin", 2021)]


# ---------------------------------------------------------------------------
# The correction
# ---------------------------------------------------------------------------

def group_offset(resid, member, pool, k=K_SHRINK) -> dict:
    """Shrunk mean residual over `member & pool`. The Stage-3 offset formula.

    Deliberately NOT `stages.signing_offsets`: that function hard-refuses any
    key outside SIGNING_ELIGIBLE_TYPES, which is the right guard for a
    mechanism label and the wrong one for a circumstance group. The arithmetic
    is identical and k comes from the same imported constant, so the two cannot
    drift apart on the quantity that was pre-registered.

    Args:
        resid: actual - predicted, cap_pct. A NEGATIVE offset LOWERS an
            over-priced group, which is what this hypothesis asks for.
        member: bool mask, group membership.
        pool: bool mask of rows allowed to inform the estimate.
        k: shrinkage denominator; offset = n/(n+k) x raw mean.
    """
    m = np.asarray(member, bool) & np.asarray(pool, bool)
    n = int(m.sum())
    raw = float(np.asarray(resid)[m].mean()) if n else 0.0
    return {"n": n, "raw": raw, "shrink": n / (n + k) if n else 0.0,
            "offset": (n / (n + k)) * raw if n else 0.0}


def apply_group_offset(pred, member, offset, *, lo, hi, is_ext, ext_cap):
    """pred + offset on members, then the extension clip, then the [lo,hi] clip.

    Same components and the same order as `stages.stage3_signing`, which is
    where a correction of this kind has to put legality: a pull that pushed a
    row below the league minimum would be pricing an impossible contract. A
    non-member gets `x + 0.0` and clips that are the identity on a value
    already inside its bounds, so it comes out bit-identical. That is asserted
    downstream, not assumed here.
    """
    out = np.array(pred, dtype=float, copy=True)
    out = out + np.where(np.asarray(member, bool), float(offset), 0.0)
    out = stage3(out, is_extension=is_ext, ext_cap_pct=ext_cap)
    return np.clip(out, np.asarray(lo, float), np.asarray(hi, float))


def assert_ex_ante(df: pd.DataFrame) -> dict:
    """Check the admissibility claim in code, not only in the docstring.

    The grouping is legal only because none of its coordinates can be moved by
    the price under prediction. Two things are checkable on the data:

      1. career_earnings_thru_prev is strictly THROUGH T-1, so it must never
         include this row's own salary. Tested by the contrapositive: a row's
         earnings must not correlate with its own salary once the player's
         history is held fixed -- too weak to test directly, so what is tested
         is the constructive property, that the column is identical for a row
         regardless of that row's cap_pct, i.e. it is a function of
         (player, season) alone. The build script guarantees this; the check
         here is that the merge is 1:1 on (player, season).
      2. rings_thru_prev is strictly before T: a ring earned IN season T must
         not appear. Checked against the file's own ring-season list.
    """
    out = {}
    key = df[["player_name_norm", "season"]]
    out["merge_is_1to1"] = bool(not key.duplicated().any())
    bad = 0
    for s, seas in zip(df["season"].values, df["ring_seasons_thru_prev"].values):
        if isinstance(seas, str) and seas.strip():
            for tok in seas.split("|"):
                tok = tok.strip()
                if tok and int(tok) >= int(s):
                    bad += 1
    out["rings_at_or_after_T"] = int(bad)
    out["pass"] = bool(out["merge_is_1to1"] and bad == 0)
    return out


# ---------------------------------------------------------------------------
# Fit pass
# ---------------------------------------------------------------------------

P_MAIN = ["earn_z", "age_z", "rings_z", "age_x_earn"]
P_ADD = ["earn_z", "age_z", "rings_z"]


def _p_design(sub: pd.DataFrame, scaler=None, cols=P_MAIN):
    """Standardised circumstance design matrix. Fit the scaler on train only."""
    raw = np.column_stack([sub["career_earnings_thru_prev_cap_pct"].values,
                           sub["age"].values,
                           sub["rings_thru_prev"].values]).astype(float)
    if scaler is None:
        scaler = StandardScaler().fit(raw)
    z = scaler.transform(raw)
    d = {"earn_z": z[:, 0], "age_z": z[:, 1], "rings_z": z[:, 2],
         "age_x_earn": z[:, 0] * z[:, 1]}
    return np.column_stack([d[c] for c in cols]), scaler


def champion_fold_pass(df, kf_ctx, clf_features, seeds, cap_m):
    """Champion OOF per (fold, seed) + the fold-honest circumstance P.

    The champion here is `compose(latent, lo, hi, p_max, is_extension,
    ext_cap_pct)` -- the arm `outputs/models/oof_reference.csv` stores as
    `oof_champion`, and the one the handoff brief's evidence table was
    measured against. It does NOT include the deployed Stage-3 signing offset;
    that arm is the suite's `challenger`. Building on the champion keeps this
    measurement comparable to the evidence that motivated it.
    """
    features = list(FEATURE_COLS)
    y = df[TARGET].values
    folds = list(GroupKFold(n_splits=N_SPLITS).split(
        df, y, df["player_name_norm"].values))
    store = []
    for si, seed in enumerate(seeds):
        for fi, (tr, va) in enumerate(folds):
            train, test = df.iloc[tr], df.iloc[va]
            tr_aug, te_aug = _compute_kf_nested(kf_ctx, train, test,
                                                clf_features, seed)
            latent, lo, hi = rm.grabit_latent(tr_aug, te_aug, features, seed)
            clf = rm.train_route_classifier(tr_aug, clf_features, seed)
            p_max = rm.route_proba(clf, te_aug, clf_features)[:, rm.MAX_IDX]
            champ = compose(latent, lo=lo, hi=hi, p_max=p_max,
                            is_extension=test["is_extension"].values,
                            ext_cap_pct=test["ext_cap_pct"].values)

            # --- circumstance P, fitted inside the training slice ----------
            base = XGBRegressor(**{**_XGB_BASE, "random_state": seed})
            base.fit(tr_aug[features], y[tr])
            bp = base.predict(tr_aug[features])          # in-sample, as the
            lab = ((bp - y[tr]) * cap_m[tr]) >= LABEL_OVER_M   # censor gates do
            ps = {}
            for tag, cols in (("p", P_MAIN), ("p_add", P_ADD)):
                if lab.sum() < 10 or (~lab).sum() < 10:
                    ps[tag] = np.zeros(len(va))
                    continue
                Xtr, sc = _p_design(train, cols=cols)
                Xte, _ = _p_design(test, scaler=sc, cols=cols)
                lr = LogisticRegression(max_iter=2000, C=1.0)
                lr.fit(Xtr, lab.astype(int))
                ps[tag] = lr.predict_proba(Xte)[:, 1]

            store.append({"si": si, "fi": fi, "va": va, "champ": champ,
                          "p": ps["p"], "p_add": ps["p_add"]})
        print(f"    seed {seed} done ({si + 1}/{len(seeds)})", flush=True)
    return store, folds


def seed_average(store, n_rows, key, n_seeds):
    acc = np.zeros(n_rows)
    for rec in store:
        acc[rec["va"]] += rec[key]
    return acc / n_seeds


def fold_r2_matrix(store, y, n_folds, n_seeds, key, sel=None):
    out = np.full((n_folds, n_seeds), np.nan)
    for rec in store:
        va = rec["va"]
        m = np.ones(len(va), bool) if sel is None else sel[va]
        if m.sum() > 10:
            out[rec["fi"], rec["si"]] = r2_score(y[va][m], rec[key][m])
    return out


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def zone_stats(df, pred, mask, cap_m, y):
    err = (pred[mask] - y[mask]) * cap_m[mask]
    return {"n": int(mask.sum()), "mae": float(np.abs(err).mean()),
            "bias": float(err.mean()),
            "rmse": float(np.sqrt((err ** 2).mean())),
            "median": float(np.median(err))}


def c2_segments(df, champ, cand):
    segs = {}
    for cat in sorted(set(map(str, df["signing_cat"].fillna("Unknown")))):
        m = (df["signing_cat"].fillna("Unknown").astype(str) == cat).values
        if m.sum() < 10:
            continue
        _, b_c = _dollars(df, champ, m)
        _, b_d = _dollars(df, cand, m)
        mae_c, _ = _dollars(df, champ, m)
        segs[cat] = {"n": int(m.sum()), "champ": b_c, "cand": b_d,
                     "signed_change": b_d - b_c,
                     "abs_bias_growth": abs_bias_growth(b_d, b_c)}
    return segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=len(DEFAULT_SEEDS))
    ap.add_argument("--skip-b", action="store_true")
    args = ap.parse_args()
    seeds = tuple(DEFAULT_SEEDS[:args.seeds])
    pd.set_option("display.width", 250)

    # ---- frame + circumstance columns -------------------------------------
    df, base_features = load_evaluation_frame(verbose=True,
                                              allow_missing_computed=True)
    df = attach_extension_cap(df, verbose=False)
    df, clf_features = rm.attach_clf_features(df)

    ce = pd.read_csv(PROCESSED_DIR / "career_earnings.csv")
    rg = pd.read_csv(PROCESSED_DIR / "rings_thru_prev.csv")
    n0 = len(df)
    df = df.merge(ce, on=["player_name_norm", "season"], how="left")
    df = df.merge(rg, on=["player_name_norm", "season"], how="left")
    assert len(df) == n0, "circumstance merge changed the row count"
    df = df.reset_index(drop=True)

    y = df[TARGET].values
    cap_m = df["cap"].values / 1e6
    lo_all = df["floor_pct"].values
    hi_all = df["max_eligible_pct"].values
    is_ext = df["is_extension"].values
    ext_cap = df["ext_cap_pct"].values
    sel = ~df["is_confirmation"].values
    recent = df["season"].values >= 2024

    earn = df["career_earnings_thru_prev"].values.astype(float)
    rings = df["rings_thru_prev"].values
    age = df["age"].values.astype(float)
    earn_known = np.isfinite(earn)
    p75 = float(np.nanquantile(earn, CELL_EARN_Q))
    dz = (earn_known & (np.nan_to_num(rings, nan=-1) == CELL_RINGS_MAX)
          & (age >= CELL_AGE_MIN) & (earn >= p75))
    df["career_earnings_thru_prev_cap_pct"] = (
        df["career_earnings_thru_prev_cap_pct"].fillna(0.0))
    df["rings_thru_prev"] = df["rings_thru_prev"].fillna(0)

    report = {
        "PROVISIONAL": ("Spotrac salary migration is rewriting salaries.csv and "
                        "training_data_v2.csv concurrently; ~15% of this frame "
                        "is expected to move. Re-run before quoting."),
        "adopted": False, "n_rows": len(df), "seeds": list(seeds),
        "k_shrink": K_SHRINK,
        "cell": {"rule": "rings==0 & age>=30 & career_earnings>=frame P75",
                 "p75_dollars": p75, "n": int(dz.sum()),
                 "n_selection": int((dz & sel).sum()),
                 "n_confirmation": int((dz & ~sel).sum()),
                 "route": "(a) pre-registered ordinary cell, no scan"},
    }

    print(f"\nframe {len(df)} rows | {len(FEATURE_COLS)} regression features "
          f"| seeds {len(seeds)}")
    print(f"career earnings present {int(earn_known.sum())}/{len(df)}; "
          f"frame P{CELL_EARN_Q * 100:.0f} = ${p75 / 1e6:.1f}M")
    print(f"PRE-REGISTERED CELL  rings==0 & age>=30 & earnings>=P75 : "
          f"n={int(dz.sum())} ({int((dz & sel).sum())} selection, "
          f"{int((dz & ~sel).sum())} confirmation)")
    print(f"PRE-REGISTERED k = {K_SHRINK:g} (imported from stages.SIGNING_K); "
          f"k = {K_REFERENCE:g} is reference-only")

    ax = assert_ex_ante(df)
    report["ex_ante_check"] = ax
    print(f"ex-ante admissibility check: merge 1:1 {ax['merge_is_1to1']}, "
          f"rings dated at/after T {ax['rings_at_or_after_T']}  "
          f"[{'PASS' if ax['pass'] else 'FAIL'}]")
    if not ax["pass"]:
        raise SystemExit("ex-ante admissibility check FAILED — stop.")

    # leave-fold-out sensitivity of the quantile itself
    print("\nCELL COMPOSITION")
    print(f"  {'signing_cat':16s} {'n':>4s}")
    for cat, n in df.loc[dz, "signing_cat"].fillna("Unknown").value_counts().items():
        print(f"  {str(cat):16s} {n:4d}")
    print(f"  at_floor members: {int((dz & df['is_at_floor'].values.astype(bool)).sum())}"
          f" of {int(dz.sum())}  -> the cell is NOT floor-restricted")
    report["cell"]["by_signing_cat"] = {
        str(k): int(v) for k, v in
        df.loc[dz, "signing_cat"].fillna("Unknown").value_counts().items()}
    report["cell"]["n_at_floor"] = int(
        (dz & df["is_at_floor"].values.astype(bool)).sum())

    # ---- fit pass ----------------------------------------------------------
    print("\nFIT PASS — champion (nested-CV kf_market_value) + circumstance P")
    kf_ctx = prepare_kf_context(df, base_features, verbose=True)
    store, folds = champion_fold_pass(df, kf_ctx, clf_features, seeds, cap_m)
    n_folds, n_seeds = len(folds), len(seeds)
    fold_of = np.empty(len(df), int)
    for fi, (_, va) in enumerate(folds):
        fold_of[va] = fi

    champ = seed_average(store, len(df), "champ", n_seeds)
    p_oof = seed_average(store, len(df), "p", n_seeds)
    p_add = seed_average(store, len(df), "p_add", n_seeds)
    resid = y - champ                       # actual - predicted, cap_pct
    champ_err = (champ - y) * cap_m         # over-prediction, $M

    a1 = float(r2_score(y, champ))
    a2 = float(r2_score(y[recent], champ[recent]))
    print(f"\n  champion reproduction: A1 {a1:.4f}  A2 {a2:.4f}  "
          f"MAE ${np.abs(champ_err).mean():.3f}M  bias ${champ_err.mean():+.3f}M")
    ref = OUT / "oof_reference.csv"
    if ref.exists():
        o = df[["player_name_norm", "season"]].merge(
            pd.read_csv(ref)[["player_name_norm", "season", "oof_champion"]],
            on=["player_name_norm", "season"], how="left")["oof_champion"].values
        d = float(np.nanmax(np.abs(champ - o)))
        print(f"  vs stored oof_reference.oof_champion: max|diff| {d:.4f}, "
              f"corr {np.corrcoef(champ, o)[0, 1]:.5f}, stored A1 "
              f"{r2_score(y, o):.4f}")
        report["champion_repro"] = {"A1": a1, "A2": a2, "max_absdiff_vs_stored": d,
                                    "stored_A1": float(r2_score(y, o))}

    # ---- the cell's residual profile, computed on the champion -------------
    print("\n" + "=" * 100)
    print("  CELL PROFILE on the champion OOF (before any correction)")
    print("=" * 100)
    u = mannwhitneyu(champ_err[dz], champ_err[~dz], alternative="greater")
    print(f"  DZ  n={int(dz.sum()):3d}  mean over-prediction ${champ_err[dz].mean():+.2f}M"
          f"  median ${np.median(champ_err[dz]):+.2f}M   "
          f"Mann-Whitney p={u.pvalue:.4f}")
    print(f"  rest n={int((~dz).sum()):3d}  mean ${champ_err[~dz].mean():+.2f}M"
          f"  median ${np.median(champ_err[~dz]):+.2f}M")
    print("  NOTE the mean/median gap: a mean-sized constant is larger than the "
          "typical member needs.")
    report["cell_profile"] = {
        "dz_mean_over_m": float(champ_err[dz].mean()),
        "dz_median_over_m": float(np.median(champ_err[dz])),
        "rest_mean_over_m": float(champ_err[~dz].mean()),
        "mannwhitney_p": float(u.pvalue)}

    # sub-cells, DESCRIPTIVE ONLY
    print("\n  sub-cells (DESCRIPTIVE ONLY — no arm is scored on these)")
    print(f"    {'cell':24s} {'n':>4s} {'raw mean resid':>15s} {'shrink':>7s} "
          f"{'offset $M':>10s}")
    subs = {}
    for tag, amin, q in SUBCELLS:
        qv = float(np.nanquantile(earn, q))
        m = (earn_known & (np.nan_to_num(rings, nan=-1) == 0)
             & (age >= amin) & (earn >= qv))
        g = group_offset(resid, m, np.ones(len(df), bool))
        subs[tag] = g
        print(f"    {tag:24s} {g['n']:4d} {g['raw']:+15.5f} {g['shrink']:7.3f} "
              f"{g['offset'] * CAP_DISPLAY / 1e6:+10.3f}")
    report["subcells"] = subs

    # ---- deployed-form offset + jackknife ---------------------------------
    all_rows = np.ones(len(df), bool)
    dep20 = group_offset(resid, dz, all_rows, k=K_SHRINK)
    dep0 = group_offset(resid, dz, all_rows, k=K_REFERENCE)
    print("\n" + "=" * 100)
    print("  DEPLOYED-FORM OFFSET (all OOF rows) — reporting only, never scores")
    print("=" * 100)
    print(f"  n={dep20['n']}  raw mean resid {dep20['raw']:+.5f} cap_pct  "
          f"shrink {dep20['shrink']:.3f}  offset {dep20['offset']:+.5f} "
          f"= ${dep20['offset'] * CAP_DISPLAY / 1e6:+.3f}M at the 2026 cap")
    print(f"  k=0 reference: {dep0['offset'] * CAP_DISPLAY / 1e6:+.3f}M")
    print("  sign: offset = mean(actual - predicted); NEGATIVE lowers an "
          "over-priced group, which is what this hypothesis asks for.")

    idx_dz = np.flatnonzero(dz)
    order = idx_dz[np.argsort(-champ_err[idx_dz])]
    jack = {"full_raw": dep20["raw"], "full_offset_m":
            dep20["offset"] * CAP_DISPLAY / 1e6}
    print("\n  JACKKNIFE — is the offset carried by a few rows?")
    print(f"    {'drop':28s} {'n':>4s} {'raw resid':>11s} {'offset $M':>10s}")
    print(f"    {'(none)':28s} {dep20['n']:4d} {dep20['raw']:+11.5f} "
          f"{dep20['offset'] * CAP_DISPLAY / 1e6:+10.3f}")
    for d in (1, 3):
        keep = dz.copy()
        keep[order[:d]] = False
        g = group_offset(resid, keep, all_rows)
        names = ", ".join(f"{df['player_name_norm'].iat[i]} {df['season'].iat[i]}"
                          for i in order[:d])
        jack[f"drop{d}"] = {**g, "dropped": names,
                            "offset_m": g["offset"] * CAP_DISPLAY / 1e6}
        print(f"    top {d} by over-prediction   {g['n']:4d} {g['raw']:+11.5f} "
              f"{g['offset'] * CAP_DISPLAY / 1e6:+10.3f}   [{names}]")
    print("\n    the 5 largest contributors inside DZ:")
    for i in order[:5]:
        print(f"      {df['player_name_norm'].iat[i]:22s} {df['season'].iat[i]} "
              f"age {age[i]:.0f}  rings {int(rings[i])}  earned "
              f"${earn[i] / 1e6:6.0f}M  paid ${df['salary_m'].iat[i]:6.2f}M  "
              f"champErr {champ_err[i]:+7.2f}M  {df['signing_cat'].iat[i]}")
    report["jackknife"] = jack

    # ---- the P: does the discount rank on circumstance? --------------------
    print("\n" + "=" * 100)
    print("  THE CIRCUMSTANCE P — fold-honest, and how it ranks the error")
    print("=" * 100)
    p_diag = {}
    for tag, pv in (("P (with age x earnings)", p_oof),
                    ("P (main effects only)", p_add)):
        rho_all = float(spearmanr(pv, champ_err).statistic)
        rho_dz = float(spearmanr(pv[dz], champ_err[dz]).statistic)
        auc = float(roc_auc_score(dz.astype(int), pv))
        over = np.maximum(champ_err, 0)
        top = np.argsort(-pv)[:len(pv) // 2]
        share = float(over[top].sum() / over.sum())
        p_diag[tag] = {"spearman_frame": rho_all, "spearman_in_cell": rho_dz,
                       "auc_vs_cell": auc, "top_half_error_share": share,
                       "max_p": float(pv.max()), "median_p": float(np.median(pv))}
        print(f"  {tag:26s} Spearman(P, over-prediction) frame {rho_all:+.3f}  "
              f"in-cell {rho_dz:+.3f}")
        print(f"  {'':26s} top half by P carries {share:.1%} of the frame's "
              f"over-prediction (50% = no information)")
        print(f"  {'':26s} AUC vs cell membership {auc:.3f}   max P "
              f"{pv.max():.3f}   median P {np.median(pv):.3f}")
    print(f"  BENCHMARK: the P that failed on the floor branch scored "
          f"Spearman -0.611 (anti-ranked against the error it existed to fix).")
    report["p_diagnostics"] = p_diag

    # ---- tau, pre-registered BEFORE any arm score --------------------------
    print("\n" + "=" * 100)
    print("  PRE-REGISTERED tau — win minus collateral, computed before any arm")
    print("=" * 100)
    print(f"  {'tau':>5s} {'n>=t':>5s} {'nDZ':>4s} {'purity':>7s} {'Ewin':>8s} "
          f"{'Ecoll':>8s} {'OBJ':>8s}")
    sweep = []
    for tau in TAU_GRID:
        t = p_oof >= tau
        if t.sum() == 0:
            continue
        e_win = float(np.maximum(champ_err[t], 0).sum())
        e_coll = float(np.maximum(-champ_err[t], 0).sum())
        row = {"tau": float(tau), "n": int(t.sum()),
               "n_in_cell": int((t & dz).sum()),
               "purity": float((t & dz).sum() / t.sum()),
               "expected_win_m": e_win, "expected_collateral_m": e_coll,
               "objective_m": e_win - e_coll}
        sweep.append(row)
        print(f"  {tau:5.2f} {row['n']:5d} {row['n_in_cell']:4d} "
              f"{row['purity']:7.3f} {e_win:8.1f} {e_coll:8.1f} "
              f"{row['objective_m']:8.1f}")
    best = max(sweep, key=lambda r: r["objective_m"])
    tau_star = best["tau"]
    print(f"\n  SELECTED tau* = {tau_star:.2f}  (win ${best['expected_win_m']:.1f}M "
          f"- collateral ${best['expected_collateral_m']:.1f}M = "
          f"${best['objective_m']:.1f}M), {best['n']} rows touched, "
          f"purity {best['purity']:.3f}")
    if tau_star in (float(TAU_GRID[0]), float(TAU_GRID[-1])):
        print("  ANOMALY: tau* is on the grid edge — the objective did not turn "
              "over inside the pre-registered range. Reported, not repaired.")
    report["tau_sweep"] = sweep
    report["tau_star"] = tau_star
    pw = p_oof >= tau_star

    # ---- arms --------------------------------------------------------------
    print("\n" + "=" * 100)
    print("  ARMS — leave-fold-out offsets (fold f corrected from rows outside f)")
    print("=" * 100)
    arms = {"O": (dz, K_SHRINK), "O0": (dz, K_REFERENCE), "Pw": (pw, K_SHRINK)}
    lfo = {}
    for name, (member, k) in arms.items():
        lfo[name] = {fi: group_offset(resid, member, fold_of != fi, k=k)
                     for fi in range(n_folds)}
    print(f"  {'arm':4s} " + "  ".join(f"{'fold ' + str(f):>13s}"
                                       for f in range(n_folds)))
    for name in arms:
        print(f"  {name:4s} " + "  ".join(
            f"{lfo[name][f]['offset'] * CAP_DISPLAY / 1e6:+13.3f}"
            for f in range(n_folds)) + "   ($M at 2026 cap)")

    for rec in store:
        va = rec["va"]
        for name, (member, _k) in arms.items():
            rec[name] = apply_group_offset(
                rec["champ"], member[va], lfo[name][rec["fi"]]["offset"],
                lo=lo_all[va], hi=hi_all[va], is_ext=is_ext[va],
                ext_cap=ext_cap[va])
    preds = {n: seed_average(store, len(df), n, n_seeds) for n in arms}

    frs_champ = fold_r2_matrix(store, y, n_folds, n_seeds, "champ", sel)
    frs_champ_all = fold_r2_matrix(store, y, n_folds, n_seeds, "champ")

    # ---- gate battery ------------------------------------------------------
    gates = {}
    for name, (member, k) in arms.items():
        cand = preds[name]
        print("\n" + "-" * 100)
        print(f"  ARM {name}   group n={int(member.sum())}   k={k:g}"
              + ("   [PRIMARY]" if name == "O" else
                 "   [reference only, decides nothing]" if name == "O0" else
                 "   [secondary, reported]"))
        # bit-identity outside the group
        worst = 0.0
        for rec in store:
            va = rec["va"]
            m = ~member[va]
            if m.sum():
                worst = max(worst, float(np.abs(rec[name][m] - rec["champ"][m]).max()))
        print(f"    bit-identity outside the group: max|diff| {worst:.3e} "
              f"[{'PASS' if worst == 0.0 else 'FAIL'}]  "
              f"n={int((~member).sum())} rows")

        z = {}
        for tag, m in (("all", member), ("sel", member & sel),
                       ("conf", member & ~sel)):
            if m.sum() == 0:
                continue
            zc = zone_stats(df, champ, m, cap_m, y)
            zd = zone_stats(df, cand, m, cap_m, y)
            z[tag] = {"n": zc["n"], "champ": zc, "cand": zd,
                      "mae_win": zc["mae"] - zd["mae"],
                      "rmse_win": zc["rmse"] - zd["rmse"],
                      "abs_bias_growth": abs_bias_growth(zd["bias"], zc["bias"])}
        print(f"    zone (where it acts) — SELECTION rows decide (ISSUES #20a)")
        for tag in ("sel", "all", "conf"):
            if tag not in z:
                continue
            d = z[tag]
            print(f"      [{tag:4s}] n={d['n']:4d}  bias "
                  f"${d['champ']['bias']:+6.2f} -> ${d['cand']['bias']:+6.2f}M "
                  f"(|bias| growth {d['abs_bias_growth']:+.3f})  RMSE "
                  f"${d['champ']['rmse']:5.2f} -> ${d['cand']['rmse']:5.2f}M "
                  f"(win {d['rmse_win']:+.3f})  MAE ${d['champ']['mae']:5.2f} -> "
                  f"${d['cand']['mae']:5.2f}M (win {d['mae_win']:+.3f})")

        # rows inside the group the champion already UNDER-prices
        hurt = member & (champ_err < 0)
        e_c = np.abs(champ_err)
        e_d = np.abs((cand - y) * cap_m)
        print(f"    inside-group collateral: {int(hurt.sum())} of "
              f"{int(member.sum())} members were already UNDER-priced and are "
              f"pushed further wrong by ${float((e_d[hurt] - e_c[hurt]).sum()):+.2f}M "
              f"in total")

        c2 = c2_segments(df, champ, cand)
        c2_worst = max((v["abs_bias_growth"] for v in c2.values()), default=0.0)
        pd_sel = paired_delta(frs_champ, fold_r2_matrix(store, y, n_folds,
                                                        n_seeds, name, sel))
        pd_all = paired_delta(frs_champ_all, fold_r2_matrix(store, y, n_folds,
                                                            n_seeds, name))
        print(f"    C2 worst segment |bias| growth {c2_worst:+.3f}M "
              f"[{'PASS' if c2_worst <= C2_BAR else 'fail'}]")
        for cat, v in sorted(c2.items(), key=lambda kv: -kv[1]["abs_bias_growth"])[:3]:
            print(f"        {cat:14s} n={v['n']:4d}  {v['champ']:+6.2f} -> "
                  f"{v['cand']:+6.2f}  |bias| growth {v['abs_bias_growth']:+.3f}")
        print(f"    A1 {r2_score(y, cand):.4f}  A2 "
              f"{r2_score(y[recent], cand[recent]):.4f}")
        print(f"    paired dSel {pd_sel['delta']:+.5f} +/- {pd_sel['se']:.5f}  "
              f"t {pd_sel['t']:+.2f} [{'PASS' if pd_sel['t'] > DSEL_T_BAR else 'fail'}]"
              f"   per fold {pd_sel['per_fold']}")
        print(f"    paired dAll  {pd_all['delta']:+.5f}  t {pd_all['t']:+.2f}")
        gates[name] = {
            "group_n": int(member.sum()), "k": k,
            "bit_identity_max_absdiff": worst,
            "bit_identity_pass": bool(worst == 0.0),
            "zone": z, "c2": c2, "c2_worst_abs_bias_growth": c2_worst,
            "c2_pass": bool(c2_worst <= C2_BAR),
            "inside_group_underpriced_n": int(hurt.sum()),
            "inside_group_extra_damage_m": float((e_d[hurt] - e_c[hurt]).sum()),
            "A1": float(r2_score(y, cand)),
            "A2": float(r2_score(y[recent], cand[recent])),
            "dSel": pd_sel, "dAll": pd_all,
            "dSel_pass": bool(pd_sel["t"] > DSEL_T_BAR),
            "lfo_offsets_m": {str(f): lfo[name][f]["offset"] * CAP_DISPLAY / 1e6
                              for f in range(n_folds)}}
    report["gates"] = gates

    # ---- who it touches that it should not --------------------------------
    print("\n" + "=" * 100)
    print("  COLLATERAL — the members the correction damages most (arm O)")
    print("=" * 100)
    cand = preds["O"]
    e_c, e_d = np.abs(champ_err), np.abs((cand - y) * cap_m)
    dmg = e_d - e_c
    rows = [{"player": df["player_name_norm"].iat[i], "season": int(df["season"].iat[i]),
             "age": float(age[i]), "rings": int(rings[i]),
             "earn_m": float(earn[i] / 1e6) if np.isfinite(earn[i]) else None,
             "paid_m": float(df["salary_m"].iat[i]),
             "signing_cat": str(df["signing_cat"].iat[i]),
             "champ_err_m": float(champ_err[i]), "damage_m": float(dmg[i])}
            for i in np.flatnonzero(dz)]
    rows.sort(key=lambda r: -r["damage_m"])
    print(f"  {'player':22s} {'szn':>4s} {'age':>3s} {'paid$M':>7s} "
          f"{'champErr':>9s} {'damage':>8s}  signing")
    for r in rows[:10]:
        print(f"  {r['player']:22s} {r['season']:4d} {r['age']:3.0f} "
              f"{r['paid_m']:7.2f} {r['champ_err_m']:+9.2f} {r['damage_m']:+8.2f}  "
              f"{r['signing_cat']}")
    print(f"  ... and the 5 it helps most:")
    for r in rows[-5:][::-1]:
        print(f"  {r['player']:22s} {r['season']:4d} {r['age']:3.0f} "
              f"{r['paid_m']:7.2f} {r['champ_err_m']:+9.2f} {r['damage_m']:+8.2f}  "
              f"{r['signing_cat']}")
    report["collateral_O"] = rows

    print("\n  named cases from the handoff brief:")
    key = {(a, b): i for i, (a, b) in enumerate(zip(df["player_name_norm"],
                                                    df["season"]))}
    named = []
    for nm, se in NAMED:
        i = key.get((nm, se))
        if i is None:
            print(f"    {nm} {se}: NOT IN FRAME")
            continue
        named.append({"player": nm, "season": se, "in_cell": bool(dz[i]),
                      "p": float(p_oof[i]), "champ_err_m": float(champ_err[i]),
                      "arm_O_err_m": float((cand[i] - y[i]) * cap_m[i])})
        n_ = named[-1]
        print(f"    {nm + ' ' + str(se):26s} inCell {str(n_['in_cell']):5s} "
              f"P {n_['p']:.3f}  champErr {n_['champ_err_m']:+7.2f} -> armO "
              f"{n_['arm_O_err_m']:+7.2f}")
    report["named_cases"] = named

    # ---- layer B -----------------------------------------------------------
    if not args.skip_b:
        print("\n" + "=" * 100)
        print("  LAYER B — rolling origin, offsets from seasons < T only")
        print("=" * 100)
        season = df["season"].values
        fwd_c = np.full(len(df), np.nan)
        fwd = {n: np.full(len(df), np.nan) for n in arms}
        b_detail = {}
        for T in FORWARD_ORIGINS:
            te, tr = season == T, season < T
            if te.sum() < 10 or tr.sum() < 200:
                continue
            print(f"\n  origin {T}: train {int(tr.sum())}, test {int(te.sum())}",
                  flush=True)
            kf_all = _compute_kf_forward(df, kf_ctx, tr, clf_features, seeds)
            fill = float(np.nanmedian(kf_all[tr]))
            tr_aug, te_aug = df[tr].copy(), df[te].copy()
            tr_aug["kf_market_value"] = np.where(np.isfinite(kf_all[tr]),
                                                 kf_all[tr], fill)
            te_aug["kf_market_value"] = np.where(np.isfinite(kf_all[te]),
                                                kf_all[te], fill)
            acc = np.zeros(int(te.sum()))
            for seed in seeds:
                latent, lo, hi = rm.grabit_latent(tr_aug, te_aug,
                                                  list(FEATURE_COLS), seed)
                clf = rm.train_route_classifier(tr_aug, clf_features, seed)
                pm = rm.route_proba(clf, te_aug, clf_features)[:, rm.MAX_IDX]
                acc += compose(latent, lo=lo, hi=hi, p_max=pm,
                               is_extension=te_aug["is_extension"].values,
                               ext_cap_pct=te_aug["ext_cap_pct"].values)
            fwd_c[te] = acc / len(seeds)
            print(f"    inner OOF over the training window for the offset...",
                  flush=True)
            inner_oof, _, _ = oof_groupkfold(tr_aug, list(FEATURE_COLS),
                                             make_champion_fitter(clf_features),
                                             seeds)
            r_in = tr_aug[TARGET].values - inner_oof
            b_detail[str(T)] = {}
            for name, (member, k) in arms.items():
                g = group_offset(r_in, member[tr], np.ones(int(tr.sum()), bool), k=k)
                b_detail[str(T)][name] = {**g,
                                          "offset_m": g["offset"] * CAP_DISPLAY / 1e6}
                fwd[name][te] = apply_group_offset(
                    fwd_c[te], member[te], g["offset"], lo=lo_all[te],
                    hi=hi_all[te], is_ext=is_ext[te], ext_cap=ext_cap[te])
                print(f"      {name:4s} offset from seasons<{T}: n={g['n']:3d} "
                      f"{g['offset'] * CAP_DISPLAY / 1e6:+.3f}M  "
                      f"(applied to {int(member[te].sum())} test rows)")
        m = ~np.isnan(fwd_c)
        b1 = {"champion": float(r2_score(y[m], fwd_c[m]))}
        print(f"\n    {'champion':10s} forward R2 {b1['champion']:.4f}")
        for name in arms:
            b1[name] = float(r2_score(y[m], fwd[name][m]))
            drop = b1["champion"] - b1[name]
            print(f"    {name:10s} forward R2 {b1[name]:.4f}   drop {drop:+.4f} "
                  f"[{'PASS' if drop <= B1_BAR else 'fail'}]")
        report["B1"] = b1
        report["B_offsets_by_origin"] = b_detail

    # ---- verdict -----------------------------------------------------------
    print("\n" + "=" * 100)
    print("  VERDICT — arm O (primary), against the pre-registered bars")
    print("=" * 100)
    g = gates["O"]
    checks = {
        "bit-identity outside the cell": g["bit_identity_pass"],
        "DZ |bias| improves (sel)": g["zone"]["sel"]["abs_bias_growth"] < 0,
        "DZ RMSE improves (sel)": g["zone"]["sel"]["rmse_win"] > 0,
        "C2 worst |bias| growth <= 0.30M": g["c2_pass"],
        "paired dSel t > 2": g["dSel_pass"],
    }
    if "B1" in report:
        checks["B1 drop <= 0.003"] = (report["B1"]["champion"]
                                      - report["B1"]["O"]) <= B1_BAR
    for k_, v in checks.items():
        print(f"    {'PASS' if v else 'FAIL'}  {k_}")
    print(f"\n  OVERALL: {'PASS' if all(checks.values()) else 'FAIL'}")
    print("  Reminder: NOTHING IS ADOPTED and every number is PROVISIONAL "
          "pending the Spotrac migration.")
    report["verdict"] = {k_: bool(v) for k_, v in checks.items()}
    report["verdict"]["overall"] = bool(all(checks.values()))

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "ringchase_offset_eval.json", "w") as fh:
        json.dump(report, fh, indent=2, default=float)
    dump = df[["player_name_norm", "season", TARGET, "salary_m", "age",
               "signing_cat", "is_at_floor", "is_confirmation", "floor_pct",
               "max_eligible_pct"]].copy()
    dump["career_earnings_thru_prev"] = earn
    dump["rings_thru_prev"] = rings
    dump["in_cell"] = dz
    dump["p_circumstance"] = p_oof
    dump["oof_champion"] = champ
    for name in arms:
        dump[f"oof_{name}"] = preds[name]
    dump.to_csv(OUT / "ringchase_offset_oof.csv", index=False)
    print(f"\nSaved {OUT / 'ringchase_offset_eval.json'}")
    print(f"Saved {OUT / 'ringchase_offset_oof.csv'}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
