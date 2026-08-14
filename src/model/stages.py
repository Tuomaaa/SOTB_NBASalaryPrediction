r"""Stage 2 and Stage 3 of the prediction pipeline, in one place.

Stage 1 (`train.grabit` / `route_mixture.grabit_latent`) prices a row under
*default parameters* with a two-sided censored loss and emits a LATENT value.
Everything the CBA does to that latent afterwards lives here, so that the suite,
the web export and `predict.py` cannot drift apart — three consumers
re-implementing a clip is how the "$39.68M for Marcus Smart" class of bug
survives a code review.

    latent  ->  push  ->  clip(lo, hi)  ->  signing offset  ->  mechanism cap
              \________  stage 2  _______/  ->  extension clip  ->  re-clip(lo, hi)
                                                \___________ told-route components __________/

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

**Stage 3, mechanism cap clip — the CBA ceiling for Early Bird and Non-Bird
contracts (v8.9x).** After the signing offset fires, a row signed under Early
Bird or Non-Bird still faces a legal ceiling: the CBA's 175%/105% EAS rule for
Early Bird, and the 120%/120% vet-min rule for Non-Bird. The mechanism cap sits
AFTER the signing offset and BEFORE the extension clip, so a positive offset
that pushes a row above its mechanism ceiling is caught before the extension
raise cap, which may be tighter still.

**Stage 3, signing component — the told-mechanism per-type offset (v8.8x).**
The stack carries a systematic residual bias by signing mechanism: Bird Rights
rows are underpriced by $2M, Non-Bird rows overpriced. The correction is the
crudest thing that removes it — ONE CONSTANT PER TYPE, added to the composed
prediction, after which the row goes back through the raise-cap clip and the
[floor, ceiling] clip, because a correction that pushed a row past its ceiling
would be pricing an impossible contract.

Only the four ELIGIBILITY mechanisms are corrected: Bird Rights, Cap Space,
Early Bird, Non-Bird. Every other label — MLE, BAE, Minimum, Rookie Scale,
Other, Unknown — takes a ZERO offset and comes out bit-identical. That is a
LEAKAGE ruling, not a scoring choice, and it was fixed before any score on
this arm was seen. An exception mechanism is *determined by the contract value
itself*: a deal is "the MLE" because of what it pays, so conditioning a
prediction on that label reads the target. The four eligibility mechanisms are
determined by the player's prior contract and the team's books, both settled
before the price is. Sign & Trade is reclassified as Bird Rights (2026-08-07):
unlike MLE/Minimum/BAE, the S&T mechanism does not determine the dollar amount
(contracts range $3.6M–$37.2M), and the CBA requires the originating team to
hold Bird or Early Bird rights. The four stay in however large the excluded
types' biases look, and the excluded types stay out however large theirs look.

Three constants are PRE-REGISTERED and must not be re-tuned:

  TAU = 0.52     chosen 2026-07-26 from the sweep's expected-win-minus-expected-
                 collateral rule, before any score on this arm was seen.
  MARGIN = 1.05  frozen since route-mixture phase 1. Never tuned on a zone
                 metric — Stage 2's clip makes the censored sides one-way
                 valves, so zone MAE is monotone in the margin.
  SIGNING_K = 20 the shrinkage denominator of the signing offset, fixed
                 2026-08-06 before the arm was scored. A k=0 arm was computed
                 REFERENCE-ONLY and decided nothing. Never sweep it.

**Convention (docs/QUEUE.md, 2026-07-26, refined 2026-07-27).** Stage 3 reads
the realized route, so numbers computed with it are TOLD-ROUTE numbers. The
route passes the convention's test — told "he extended", you still have to
compute 1.40 x prior pay to land on Brunson's $34.94M — so it is reported in the
same column as the ex-ante numbers. But v7.1x-v7.13x were computed under the old
"ignore the route" convention, so the ex-ante figure (push only, Stage 3 off) is
reported beside the headline wherever the series has to stay readable. The
signing offset is told-MECHANISM and sits under the same convention for the same
reason: told "he re-signed on Bird Rights", you still have to price him.

Stage 3 never reads the target, and never clamps a ceiling to observed pay
(standing decision, docs/QUEUE.md). `ext_cap_pct` never enters
`max_eligible_pct` or the Stage-1 censor mask: the raise cap binds only
CONDITIONAL on choosing to extend, which is the same "choice, not constraint"
that killed right-censoring good players on minimums. The signing offset is a
fitted parameter and so does read residuals — but never the residuals of the
row it corrects: layer A estimates it leave-fold-out, layer B from seasons
strictly earlier than the one it scores, and the deployed constants below come
from OOF residuals, never in-sample ones.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd

# Pre-registered constants. Re-tuning any of them after seeing a score is the
# failure three experiments died on — do not touch them without a new
# pre-registration.
TAU = 0.52
MARGIN = 1.05
SIGNING_K = 20.0

# The four ELIGIBILITY mechanisms, and the only labels the signing offset may
# correct. The exception mechanisms (MLE, BAE, Minimum) are excluded as
# leakage — a deal is "the MLE" because of what it pays, so the label is
# downstream of the target. Sign & Trade is reclassified as Bird Rights
# (2026-08-07): the mechanism does not determine the dollar amount, and the
# CBA requires the originating team to hold Bird/Early Bird rights.
# See the module docstring.
SIGNING_ELIGIBLE_TYPES = ("Bird Rights", "Cap Space", "Early Bird", "Non-Bird")

# Deployed-form per-type offsets, in cap_pct, for the single-fit consumers.
#
# THESE ARE DATA-DEPENDENT CONSTANTS. They are per-type mean OOF residuals of a
# particular frame, so a training-data rebuild invalidates them exactly the way
# it invalidates a published R2. Regenerate them with every rebuild:
#
#   date    2026-08-07
#   source  scripts/eval_stage3_signing.py — the gated measurement harness
#   frame   the 873-row evaluation frame (18 features), ALL OOF rows, seeds 0-9, k = 20
#   regen   python scripts/eval_stage3_signing.py
#           (the harness writes data/raw/raw_external/signing_offsets.json)
#
# The file-based path is the primary source; the hard-coded fallback below
# exists so a fresh clone without the file still has a reasonable default.
_SIGNING_OFFSETS_FALLBACK = {
    "Bird Rights": 0.013524496140298447,
    "Cap Space": 0.006433982124433493,
    "Early Bird": 0.0071840723138968925,
    "Non-Bird": -0.004729534476553264,
}

_SIGNING_OFFSETS_FILE = (
    Path(__file__).resolve().parent.parent.parent
    / "data" / "raw" / "raw_external" / "signing_offsets.json"
)


def _load_signing_offsets() -> dict:
    if _SIGNING_OFFSETS_FILE.exists():
        with open(_SIGNING_OFFSETS_FILE) as f:
            loaded = json.load(f)
        for t in SIGNING_ELIGIBLE_TYPES:
            if t not in loaded:
                raise ValueError(
                    f"{_SIGNING_OFFSETS_FILE} missing key {t!r}. "
                    "Regenerate with: python scripts/eval_stage3_signing.py")
        return {t: loaded[t] for t in SIGNING_ELIGIBLE_TYPES}
    return dict(_SIGNING_OFFSETS_FALLBACK)


SIGNING_OFFSETS_DEPLOYED = _load_signing_offsets()

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


def _as_signing_cat(signing_type) -> np.ndarray:
    """`signing_cat` as a flat object array. NaN/None simply match no type."""
    return np.asarray(pd.Series(signing_type).astype(object).values, dtype=object)


def signing_offsets(resid, signing_type, pool=None, k: float = SIGNING_K,
                    detail: bool = False) -> dict:
    """Shrunk per-type mean residual, for the eligible signing types alone.

    THE one implementation. Both evaluation layers and the measurement harness
    call it rather than re-deriving `n / (n + k)` — the shrinkage and the
    eligibility list are pre-registered quantities, and a second copy is how one
    of them silently becomes two.

    Args:
        resid: actual - predicted, in cap_pct. NOTE the sign: a positive offset
            RAISES an underpriced type, which is what a NEGATIVE diagnostics
            `bias_$M` (bias there is predicted - actual) asks for.
        signing_type: `signing_cat` per row, aligned with `resid`.
        pool: bool mask of the rows allowed to inform the estimate — the
            leave-fold-out pool in layer A, the training window in layer B.
            None means every row, which is the deployed form.
        k: shrinkage denominator, offset = n / (n + k) x raw mean. PRE-REGISTERED
            at SIGNING_K; exposed only so a harness can state it explicitly.
        detail: return {"n", "raw", "offset"} per type instead of the offset
            alone, for a report that has to show the shrinkage it applied.

    Returns:
        {type: offset} over SIGNING_ELIGIBLE_TYPES, or {type: {...}} under
        `detail`. A type with no rows in the pool gets exactly 0.0, which leaves
        its rows at the uncorrected prediction.
    """
    r = np.asarray(resid, dtype=float)
    cat = _as_signing_cat(signing_type)
    mask = np.ones(len(r), bool) if pool is None else np.asarray(pool, dtype=bool)
    out = {}
    for t in SIGNING_ELIGIBLE_TYPES:
        m = (cat == t) & mask
        n = int(m.sum())
        raw = float(r[m].mean()) if n else 0.0
        off = (n / (n + k)) * raw if n else 0.0
        out[t] = {"n": n, "raw": raw, "offset": off} if detail else off
    return out


def signing_offset_vector(signing_type, offsets) -> np.ndarray:
    """Per-row offset in cap_pct; exactly 0.0 outside the eligible types.

    `offsets` may be the flat {type: float} form or the {type: {"offset": ...}}
    detail form — both come out of `signing_offsets`. An ineligible key raises
    rather than being ignored, so the leakage ruling is enforced by the code and
    not only by the docstring.
    """
    cat = _as_signing_cat(signing_type)
    v = np.zeros(len(cat), dtype=float)
    for t, off in offsets.items():
        if t not in SIGNING_ELIGIBLE_TYPES:
            raise ValueError(
                f"{t!r} is not a correctable signing type. Only "
                f"{SIGNING_ELIGIBLE_TYPES} may carry an offset — the exception "
                "mechanisms are determined by the contract value itself, so "
                "conditioning on them reads the target.")
        v[cat == t] = off["offset"] if isinstance(off, dict) else float(off)
    return v


def stage3_signing(pred, signing_type=None, offsets=None, *, lo=None, hi=None,
                   mech_cap_pct=None,
                   is_extension=None, ext_cap_pct=None) -> np.ndarray:
    """Add the per-type signing offset, then put the row back inside the law.

    The contract, in four clauses, mirroring `stage3`'s:

      1. It ADDS a constant that depends only on the row's signing type, then
         re-applies legality in the chain order `compose` documents:
         signing offset -> mechanism cap clip -> extension clip -> re-clip
         into [lo, hi]. Legality is part of this function precisely so a caller
         cannot forget it; a correction that pushed a row past its ceiling
         would be pricing an impossible contract.
         Pass `lo`/`hi` (and the extension pair where it applies) whenever the
         result is a prediction rather than an intermediate.
      2. It is a NO-OP where `signing_type` is None, `offsets` is None or empty,
         or the row's label is missing or outside SIGNING_ELIGIBLE_TYPES — the
         offset there is exactly 0.0, `x + 0.0` is `x`, and all clips are the
         identity on a value already inside the bound. An unsigned free agent
         has no signing type at all, so `predict.py`'s deployed path is
         bit-identical to the pre-v8.8x pipeline (asserted there).
      3. It never reads the target of the row it corrects. `offsets` is a mean
         over OTHER rows' out-of-fold residuals — leave-fold-out in layer A,
         seasons < T in layer B, and the frame's own OOF for the deployed
         constants in SIGNING_OFFSETS_DEPLOYED.
      4. The mechanism cap clip (`mech_cap_pct`) sits between the signing offset
         and the extension clip, so a positive offset that pushes a row above
         its Early Bird or Non-Bird ceiling is caught before the extension raise
         cap — which may be tighter still — governs last.

    Args:
        pred: the composed prediction to correct, in cap_pct.
        signing_type: `signing_cat` per row, or None to disable.
        offsets: {type: offset} from `signing_offsets`, or None to disable.
        lo, hi: `floor_pct` and `max_eligible_pct`, for the legality re-clip.
        mech_cap_pct: per-row mechanism ceiling in cap_pct, NaN where no cap
            applies. None (default) means no mechanism cap — no-op.
        is_extension, ext_cap_pct: the Stage-3 route inputs, same producer as
            `stage3`, for the raise-cap re-clip.

    Returns:
        The corrected prediction.
    """
    out = np.array(pred, dtype=float, copy=True)
    if signing_type is None or not offsets:
        return out
    out = out + signing_offset_vector(signing_type, offsets)
    # Mechanism cap clip — sits between signing offset and extension clip
    if mech_cap_pct is not None:
        from src.model.mechanism_cap import apply_mechanism_cap
        out = apply_mechanism_cap(out, mech_cap_pct)
    if is_extension is not None and ext_cap_pct is not None:
        out = stage3(out, is_extension=is_extension, ext_cap_pct=ext_cap_pct)
    if lo is not None and hi is not None:
        out = np.clip(out, np.asarray(lo, dtype=float),
                      np.asarray(hi, dtype=float))
    return out


def compose(latent, *, lo, hi, p_max=None, is_extension=None, ext_cap_pct=None,
            signing_type=None, signing_offsets=None, mech_cap_pct=None,
            tau: float = TAU, margin: float = MARGIN) -> np.ndarray:
    """The whole post-Stage-1 chain: push -> clip -> signing -> mech cap -> ext -> re-clip.

    Passing `p_max=None` drops the push; passing `is_extension=None` drops the
    extension clip; passing `signing_type=None` or `signing_offsets=None` drops
    the signing offset; passing `mech_cap_pct=None` drops the mechanism cap
    clip. Each layer is inert by default, so a consumer that has not opted in is
    bit-identical to the composition it had before that layer existed — `p_max`
    and `is_extension` both omitted still reproduces the v7.13x champion
    exactly, which is what makes the ex-ante reference arm free to compute.

    The re-clip into [lo, hi] after each Stage-3 component is a legal, not a
    statistical, step: a contract cannot pay below the league minimum even if a
    raise cap computed lower. It is a no-op on the current frame (verified: zero
    rows have `ext_cap_pct < floor_pct`), and it is the same order both
    measurement harnesses used, so the shipped composition is bit-identical to
    the arms that were gated.
    """
    pred = stage2(latent, lo=lo, hi=hi, p_max=p_max, tau=tau, margin=margin)
    if is_extension is not None and ext_cap_pct is not None:
        pred = stage3(pred, is_extension=is_extension, ext_cap_pct=ext_cap_pct)
        pred = np.clip(pred, np.asarray(lo, dtype=float),
                       np.asarray(hi, dtype=float))
    if signing_type is not None and signing_offsets is not None:
        pred = stage3_signing(pred, signing_type, signing_offsets, lo=lo, hi=hi,
                              mech_cap_pct=mech_cap_pct,
                              is_extension=is_extension,
                              ext_cap_pct=ext_cap_pct)
    elif mech_cap_pct is not None:
        # Mechanism cap without signing offsets — rare but legal
        from src.model.mechanism_cap import apply_mechanism_cap
        pred = apply_mechanism_cap(pred, mech_cap_pct)
        pred = np.clip(pred, np.asarray(lo, dtype=float),
                       np.asarray(hi, dtype=float))
    return pred


def bound_flags(latent, pred, *, lo, hi, p_max=None, is_extension=None,
                ext_cap_pct=None, signing_type=None, signing_offsets=None,
                mech_cap_pct=None,
                tau: float = TAU, margin: float = MARGIN,
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

    The first, second and fourth read the STAGE-2 bound acting on the latent,
    which is what the site labels [MAX] / [MIN] mean. `is_ext_capped` compares
    the final prediction against the same chain with the raise-cap clip removed,
    so the signing offset has to be passed here too when it was composed —
    otherwise a negative offset reads as the raise cap binding. With no offset
    the comparison reduces to `pred < s2` exactly, because `stage2` has already
    clipped into [lo, hi].
    """
    latent = np.asarray(latent, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    s2 = stage2(latent, lo=lo, hi=hi, p_max=p_max, tau=tau, margin=margin)
    no_ext_clip = stage3_signing(s2, signing_type, signing_offsets, lo=lo, hi=hi,
                                 mech_cap_pct=mech_cap_pct)

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
        "is_ext_capped": np.asarray(pred, dtype=float) < no_ext_clip - tol,
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
        _filter_continuations, _compute_floor, _normalize_vetmin_caphold,
    )
    tr = _normalize_vetmin_caphold(_filter_rookie_contracts(
        _filter_continuations(_filter_mislabeled_year1(
            _compute_max_eligible(_filter_prorated(_filter_rookie_scale(
                _filter_year1(df))))))))
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
