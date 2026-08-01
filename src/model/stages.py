r"""Stage 2 and Stage 3 of the prediction pipeline, in one place.

Stage 1 (`train.grabit` / `route_mixture.grabit_latent`) prices a row under
*default parameters* with a two-sided censored loss and emits a LATENT value.
Everything the CBA does to that latent afterwards lives here, so that the suite,
the web export and `predict.py` cannot drift apart — three consumers
re-implementing a clip is how the "$39.68M for Marcus Smart" class of bug
survives a code review.

    latent  ->  push  ->  clip(lo, hi)  ->  stage 3
              \________  stage 2  _______/

**Stage 2 — the two-sided CBA bound, both halves.** The clip caps a row at its
tier ceiling and lifts it to the league floor. That is only the downward half of
the bound: it cannot fix a max-worthy player the model prices *below* his
ceiling, and the max zone was underpredicted by $4.27M for exactly that reason.
The push is the upward half — where the route classifier says P(max) >= TAU, the
latent is moved a P-weighted fraction of the way to `MARGIN * ceiling`, and the
clip then lands it on the ceiling itself.

**Stage 3 — the told-route extension clip.** A row that IS a first-paying-year
extension additionally faces its own raise cap (`ext_cap_pct`, see
`extension_cap.py`): 120%/140% of the player's prior salary or of the league's
Estimated Average Player Salary, whichever is greater. That number is usually
far below the tier ceiling, so the push can send an extension row to a ceiling
it could not legally reach. Predicting $39.68M for Marcus Smart 2022 was not
inaccurate, it was impossible; Stage 3 returns him to the law.

Two constants are PRE-REGISTERED and must not be re-tuned:

  TAU = 0.52     chosen 2026-07-26 from the sweep's expected-win-minus-expected-
                 collateral rule, before any score on this arm was seen.
  MARGIN = 1.05  frozen since route-mixture phase 1. Never tuned on a zone
                 metric — Stage 2's clip makes the censored sides one-way
                 valves, so zone MAE is monotone in the margin.

**Convention (docs/QUEUE.md, 2026-07-26, refined 2026-07-27).** Stage 3 reads
the realized route, so numbers computed with it are TOLD-ROUTE numbers. The
route passes the convention's test — told "he extended", you still have to
compute 1.40 x prior pay to land on Brunson's $34.94M — so it is reported in the
same column as the ex-ante numbers. But v7.1x-v7.13x were computed under the old
"ignore the route" convention, so the ex-ante figure (push only, Stage 3 off) is
reported beside the headline wherever the series has to stay readable.

Stage 3 never reads the target, and never clamps a ceiling to observed pay
(standing decision, docs/QUEUE.md). `ext_cap_pct` never enters
`max_eligible_pct` or the Stage-1 censor mask: the raise cap binds only
CONDITIONAL on choosing to extend, which is the same "choice, not constraint"
that killed right-censoring good players on minimums.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd

# Pre-registered constants. Re-tuning either after seeing a score is the failure
# three experiments died on — do not touch them without a new pre-registration.
TAU = 0.52
MARGIN = 1.05

# Deployed-model seed, matching train._XGB_BASE's random_state so the shipped
# classifier is as reproducible as the shipped regression.
DEPLOY_SEED = 42


def stage2(latent, *, lo, hi, p_max=None, tau: float = TAU,
           margin: float = MARGIN) -> np.ndarray:
    """Push toward the ceiling where P(max) >= tau, then clip into [lo, hi].

    Args:
        latent: Stage-1 latent value, in cap_pct.
        lo: `floor_pct` — the CBA minimum for the row.
        hi: `max_eligible_pct` — the row's tier ceiling.
        p_max: fold-honest P(max) from the route classifier, or None to run the
            clip alone (the pre-push champion, reproduced bit-for-bit).
        tau, margin: the pre-registered constants; exposed only so a harness can
            state them explicitly, never to be swept on a score.

    Returns:
        The Stage-2 prediction. Rows with P(max) < tau (or NaN P) are clipped
        exactly as before, so the push is inert wherever the classifier is not
        confident.
    """
    latent = np.asarray(latent, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    if p_max is None:
        return np.clip(latent, lo, hi)
    p = np.asarray(p_max, dtype=float)
    pushed = latent + p * (margin * hi - latent)
    return np.clip(np.where(p >= tau, pushed, latent), lo, hi)


def stage3(pred, *, is_extension, ext_cap_pct) -> np.ndarray:
    """Clip a prediction at the row's own extension raise cap.

    The contract, in three clauses:

      1. It only ever LOWERS a prediction. It is a `minimum`, never a maximum;
         a row whose Stage-2 value already sits under its raise cap is returned
         untouched.
      2. It is a NO-OP where `is_extension` is false or `ext_cap_pct` is NaN.
         An unsigned free agent has neither, so the deployed free-agent path is
         bit-identical to the pre-Stage-3 pipeline.
      3. It never reads the target. `ext_cap_pct` is computed from the CBA's
         published raise multiple, the player's prior salary and the league's
         Estimated Average Player Salary — never from observed pay.

    Args:
        pred: Stage-2 prediction, in cap_pct.
        is_extension: bool per row — is this the first paying year of an
            extension? Produced by `extension_cap.attach_extension_cap`.
        ext_cap_pct: the row's legal ceiling as a share of the cap, NaN where no
            extension governs it. Same producer.

    Returns:
        The Stage-3 prediction.
    """
    out = np.array(pred, dtype=float, copy=True)
    ext = np.asarray(pd.Series(is_extension).fillna(False).values, dtype=bool)
    cap = np.asarray(ext_cap_pct, dtype=float)
    m = ext & ~np.isnan(cap)
    out[m] = np.minimum(out[m], cap[m])
    return out


def compose(latent, *, lo, hi, p_max=None, is_extension=None, ext_cap_pct=None,
            tau: float = TAU, margin: float = MARGIN) -> np.ndarray:
    """The whole post-Stage-1 chain: latent -> push -> clip(lo, hi) -> stage 3.

    Passing `p_max=None` drops the push; passing `is_extension=None` drops
    Stage 3. Both omitted reproduces the v7.13x champion exactly, which is what
    makes the ex-ante reference arm free to compute.

    The final re-clip into [lo, hi] after Stage 3 is a legal, not a statistical,
    step: a contract cannot pay below the league minimum even if a raise cap
    computed lower. It is a no-op on the current frame (verified: zero rows have
    `ext_cap_pct < floor_pct`), and it is the same order the measurement harness
    used, so the shipped composition is bit-identical to the arm that was gated.
    """
    pred = stage2(latent, lo=lo, hi=hi, p_max=p_max, tau=tau, margin=margin)
    if is_extension is not None and ext_cap_pct is not None:
        pred = stage3(pred, is_extension=is_extension, ext_cap_pct=ext_cap_pct)
        pred = np.clip(pred, np.asarray(lo, dtype=float),
                       np.asarray(hi, dtype=float))
    return pred


def bound_flags(latent, pred, *, lo, hi, p_max=None, is_extension=None,
                ext_cap_pct=None, tau: float = TAU, margin: float = MARGIN,
                tol: float = 1e-9) -> dict[str, np.ndarray]:
    """Which bound, if any, moved each row off its latent value.

    There are now TWO ceilings and an upward push, so the old
    `is_capped = latent > max_eligible` no longer says what it means: a row the
    push pinned to the tier ceiling is capped and would not be flagged, and a
    row Stage 3 lowered is at a legal ceiling that is not the max. Four flags,
    each answering exactly one question:

        is_pushed     the push fired (P(max) >= tau) and raised the row
        is_capped     the prediction sits at the TIER ceiling
        is_ext_capped Stage 3 lowered the prediction (the raise cap bound)
        is_floored    the prediction sits at the CBA floor
    """
    latent = np.asarray(latent, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    s2 = stage2(latent, lo=lo, hi=hi, p_max=p_max, tau=tau, margin=margin)

    if p_max is None:
        pushed_gate = np.zeros(len(latent), bool)
        cand = latent
    else:
        p = np.asarray(p_max, dtype=float)
        pushed_gate = p >= tau
        cand = np.where(pushed_gate, latent + p * (margin * hi - latent), latent)

    return {
        "is_pushed": pushed_gate & (s2 > np.clip(latent, lo, hi) + tol),
        "is_capped": cand > hi + tol,
        "is_ext_capped": np.asarray(pred, dtype=float) < s2 - tol,
        "is_floored": cand < lo - tol,
    }


# ---------------------------------------------------------------------------
# Route probability for the deployed (single-fit) consumers
# ---------------------------------------------------------------------------

def training_route_frame(df: pd.DataFrame) -> pd.DataFrame:
    """The exact frame `train_grabit` fits on, plus both CBA bounds.

    `train_grabit` filters internally and returns only the model, so a consumer
    that needs the route classifier trained on the SAME rows has to reproduce
    the chain. Keep this identical to `train_grabit`'s prologue — a classifier
    fit on a different row set than the regression is a silent inconsistency the
    suite cannot see.

    `_compute_floor` is included because the route labels read `is_at_floor`.
    """
    from src.model.train import (
        _filter_year1, _filter_rookie_scale, _filter_rookie_contracts,
        _filter_prorated, _compute_max_eligible, _filter_mislabeled_year1,
        _filter_continuations, _compute_floor,
    )
    tr = _filter_rookie_contracts(_filter_continuations(_filter_mislabeled_year1(
        _compute_max_eligible(_filter_prorated(_filter_rookie_scale(
            _filter_year1(df)))))))
    return _compute_floor(tr).reset_index(drop=True)


def deployed_p_max(train: pd.DataFrame, test: pd.DataFrame,
                   medians: pd.Series | None = None,
                   seed: int = DEPLOY_SEED) -> np.ndarray:
    """P(max) for `test`, from a route classifier fit on `train` alone.

    The single-fit counterpart of the suite's fold-honest classifier: the same
    four-class booster on the same enriched feature list, trained only on the
    seasons the regression saw. P(max) is an OUTPUT composition weight and never
    joins the regression's feature list.

    `medians` fills the classifier's base columns the way the fitted regression
    saw them; the classifier-only enrichment columns keep NATIVE NaN, which is
    their tested ship form (XGBoost hist learns a default split direction).

    The fill list is CLF_BASE_COLS, not FEATURE_COLS: since v8.6x the two are
    separate lists, and a regression-only feature is dropped by the reindex
    below, so filling it here would raise a KeyError on the test frame.
    """
    from src.model.route_mixture import (
        attach_clf_features, train_route_classifier, route_proba, MAX_IDX,
        CLF_BASE_COLS,
    )

    train_x, clf_features = attach_clf_features(train)
    test_x, _ = attach_clf_features(test)

    fill = [c for c in CLF_BASE_COLS if c in train_x.columns]
    if medians is not None:
        train_x[fill] = train_x[fill].fillna(medians).fillna(0)
    else:
        train_x[fill] = train_x[fill].fillna(train_x[fill].median()).fillna(0)

    test_x = test_x.reindex(columns=clf_features)
    if medians is not None:
        test_x[fill] = test_x[fill].fillna(medians).fillna(0)
    else:
        test_x[fill] = test_x[fill].fillna(train_x[fill].median()).fillna(0)

    clf = train_route_classifier(train_x, clf_features, seed)
    return route_proba(clf, test_x, clf_features)[:, MAX_IDX]
