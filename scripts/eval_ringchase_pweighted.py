"""Ring-chase discount as a P-weighted proportional pull on the Stage-1 latent.

    latent_new = latent * (1 - delta * P)      where P >= tau, else unchanged
    then push -> clip -> Stage 3, unchanged

## Why this shape, after three failures

Three earlier arms failed and each ruled out one thing:

  * `career_earnings` + `rings` as Stage-1 FEATURES: dSel t = +1.22. The model
    does not lack the information -- `career_earnings` correlates +0.705 with
    `prev_cap_pct` and +0.687 with `age`, both already in it.
  * an additive constant in the Stage-3 signing-offset slot: t = -3.03. Wrong
    slot: after the floor clip, a player who discounted all the way to the
    minimum has residual ~0, so pooling him with Harden's $19.5M cannot
    estimate a common constant.
  * a multiplicative factor on the latent, gated by HARD cell membership:
    t = -1.17. Right slot and right form, wrong gate -- 26 of the cell's 102
    rows were already UNDER-priced and a constant factor pushed them further
    wrong.

This arm keeps the slot and the form and replaces the hard 0/1 cell with a
continuous P. A row the profile does not fit gets P near zero and barely moves,
which is exactly what the cell could not do.

## The target is proportional, not "toward the floor"

`eval_floor_branch.py`'s arm B pulled toward `0.95 * floor_pct`. That target is
meaningless outside the floor zone: Harden's floor is 1.4% of the cap (~$1.7M)
and he signed $33M. A proportional pull needs no zone restriction -- it is
defined for every row, and the existing floor clip absorbs the saturation for
anyone whose discount would land under the minimum.

## P blends two signals that miss opposite halves

  * `P(floor)` from the six-class route classifier ranks -0.611 against the
    over-prediction it exists to fix -- ANTI-ranked, which is what killed arm B.
    It is not useless: it is right about ordinary floor natives and wrong about
    fallen stars, because it reads performance features and a fallen star looks,
    on every one of them, like a well-paid player.
  * `P(ring-chase)` from circumstance only (career earnings, rings, age) ranks
    +0.074 frame-wide and +0.234 inside the floor zone. Weak, but correctly
    signed, and it catches precisely the rows `P(floor)` misses.

Blending is cheap here because a false positive is cheap: a floor native pulled
toward a discount is already at the floor, so the clip returns him unchanged.
The brief's Haslem case is the canonical example -- negative latent, pinned by
the clip, a further pull is a no-op.

## Selection

`delta` and `tau` are BOTH selected inside each fold's pool and never on the
held-out fold. Neither is pre-registered and they cannot be: a six-cell
threshold scan was already run and reported earlier in the session, so the
pre-registration for anything of this kind is spent. Nested selection is the
honest substitute.

PROVISIONAL: the Spotrac migration will move ~15% of the frame's rows, and this
harness's champion reproduction is not bit-identical to the stored
`oof_champion`. Both arms share that reproduction, so the PAIRED delta is
meaningful; the absolute levels are not.

Usage:
    python scripts/eval_ringchase_pweighted.py --seeds 3
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from config import CAP_BY_SEASON, OUTPUTS_DIR, PROCESSED_DIR
from src.model.evaluate_suite import (
    DEFAULT_SEEDS, N_SPLITS, TARGET, _compute_kf_nested, abs_bias_growth,
    load_evaluation_frame, prepare_kf_context,
)
from src.model import route_mixture as rm
from src.model.extension_cap import attach_extension_cap
from src.model.stages import compose
from src.model.train import FEATURE_COLS, _XGB_BASE

LABEL_OVER_M = 2.0                 # the at-floor zone's mean over-prediction
DELTA_GRID = np.round(np.arange(0.05, 0.55, 0.05), 3)
TAU_GRID = np.round(np.arange(0.20, 0.75, 0.05), 2)
P_COLS = ["career_earnings_thru_prev_cap_pct", "rings_thru_prev", "age"]
C2_BAR = 0.30
DSEL_T_BAR = 2.0
OUT = OUTPUTS_DIR / "models"


def _design(sub, p_floor, scaler=None):
    """P's design matrix: circumstance columns plus the route P(floor)."""
    X = np.column_stack([sub[c].values.astype(float) for c in P_COLS]
                        + [np.asarray(p_floor, dtype=float)])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    if scaler is None:
        scaler = StandardScaler().fit(X)
    return scaler.transform(X), scaler


def apply_pull(latent, p, delta, tau):
    """latent * (1 - delta * P) where P >= tau. Below tau the row is untouched."""
    w = np.where(np.asarray(p) >= tau, np.asarray(p), 0.0)
    return np.asarray(latent, float) * (1.0 - delta * w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    seeds = tuple(DEFAULT_SEEDS[:args.seeds])

    df, _ = load_evaluation_frame(verbose=False, allow_missing_computed=True)
    df = attach_extension_cap(df, verbose=False)
    df, clf_features = rm.attach_clf_features(df)
    ce = pd.read_csv(PROCESSED_DIR / "career_earnings.csv")
    rg = pd.read_csv(PROCESSED_DIR / "rings_thru_prev.csv")
    n0 = len(df)
    df = df.merge(ce, on=["player_name_norm", "season"], how="left")
    df = df.merge(rg, on=["player_name_norm", "season"], how="left")
    assert len(df) == n0, "circumstance merge changed the row count"
    df = df.reset_index(drop=True)
    df["career_earnings_thru_prev_cap_pct"] = (
        df["career_earnings_thru_prev_cap_pct"].fillna(0.0))
    df["rings_thru_prev"] = df["rings_thru_prev"].fillna(0)

    y = df[TARGET].values
    cap_m = df["cap"].values / 1e6
    sel = ~df["is_confirmation"].values
    features = list(FEATURE_COLS)
    kf_ctx = prepare_kf_context(df, clf_features)
    folds = list(GroupKFold(n_splits=N_SPLITS).split(
        df, y, df["player_name_norm"].values))
    lab6_all = rm.compute_route6_labels(df)

    champ_acc = np.zeros(len(df))
    cand_acc = np.zeros(len(df))
    p_acc = np.zeros(len(df))
    rows, chosen = [], []

    for si, seed in enumerate(seeds):
        per = []
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

            c6 = rm.train_route6_classifier(tr_aug, clf_features, seed,
                                            labels=lab6_all[tr])
            pf_tr = rm.route_proba(c6, tr_aug, clf_features)[:, rm.R6["floor"]]
            pf_te = rm.route_proba(c6, te_aug, clf_features)[:, rm.R6["floor"]]

            base = XGBRegressor(**{**_XGB_BASE, "random_state": seed})
            base.fit(tr_aug[features], y[tr])
            bp = base.predict(tr_aug[features])
            lab = ((bp - y[tr]) * cap_m[tr]) >= LABEL_OVER_M
            if lab.sum() < 10 or (~lab).sum() < 10:
                p_te = np.zeros(len(va))
            else:
                Xtr, sc = _design(train, pf_tr)
                Xte, _ = _design(test, pf_te, scaler=sc)
                lr = LogisticRegression(max_iter=2000, C=1.0).fit(
                    Xtr, lab.astype(int))
                p_te = lr.predict_proba(Xte)[:, 1]

            per.append({"va": va, "latent": latent, "lo": lo, "hi": hi,
                        "p_max": p_max, "p": p_te,
                        "is_ext": test["is_extension"].values,
                        "ext_cap": test["ext_cap_pct"].values, "champ": champ})
            champ_acc[va] += champ
            p_acc[va] += p_te

        for fi, rec in enumerate(per):
            other = [j for j in range(len(per)) if j != fi]
            cat = lambda k: np.concatenate([per[j][k] for j in other])
            pi = np.concatenate([per[j]["va"] for j in other])
            pl, plo, phi, ppm = cat("latent"), cat("lo"), cat("hi"), cat("p_max")
            pie, pec, pp = cat("is_ext"), cat("ext_cap"), cat("p")
            py = y[pi]

            best = (0.0, 0.0, -np.inf)
            for d in DELTA_GRID:
                for t in TAU_GRID:
                    lat = apply_pull(pl, pp, d, t)
                    pred = compose(lat, lo=plo, hi=phi, p_max=ppm,
                                   is_extension=pie, ext_cap_pct=pec)
                    s = r2_score(py, pred)
                    if s > best[2]:
                        best = (float(d), float(t), s)
            d_sel, t_sel, _ = best
            chosen.append({"seed": seed, "fold": fi, "delta": d_sel, "tau": t_sel})

            lat = apply_pull(rec["latent"], rec["p"], d_sel, t_sel)
            cand = compose(lat, lo=rec["lo"], hi=rec["hi"], p_max=rec["p_max"],
                           is_extension=rec["is_ext"], ext_cap_pct=rec["ext_cap"])
            cand_acc[rec["va"]] += cand
            va, s = rec["va"], sel[rec["va"]]
            rows.append({"seed": seed, "fold": fi,
                         "champ": r2_score(y[va][s], rec["champ"][s]),
                         "cand": r2_score(y[va][s], cand[s]),
                         "moved": int((np.abs(cand - rec["champ"]) > 1e-12).sum())})
        print(f"  seed {seed} done ({si + 1}/{len(seeds)})", flush=True)

    champ_oof = champ_acc / len(seeds)
    cand_oof = cand_acc / len(seeds)
    p_oof = p_acc / len(seeds)
    fr = pd.DataFrame(rows)
    fr["d"] = fr["cand"] - fr["champ"]
    t, p = stats.ttest_1samp(fr["d"], 0)

    ch = pd.DataFrame(chosen)
    print("\nSELECTED IN THE POOL (never on the held-out fold)")
    print(ch.groupby(["delta", "tau"]).size().rename("folds").to_string())

    over = (champ_oof - y) * cap_m
    rho = stats.spearmanr(p_oof, over).statistic
    print(f"\nP vs over-prediction: Spearman {rho:+.3f}  "
          f"(P(floor) alone was -0.611, P(ring-chase) alone +0.074)")
    moved = np.abs(cand_oof - champ_oof) > 1e-12
    e_c, e_d = np.abs(champ_oof - y) * cap_m, np.abs(cand_oof - y) * cap_m
    print(f"rows moved {int(moved.sum())} / {len(df)}")
    print(f"  of those, improved {int((e_d[moved] < e_c[moved]).sum())}, "
          f"worsened {int((e_d[moved] > e_c[moved]).sum())}")
    print(f"frame MAE ${e_c.mean():.3f}M -> ${e_d.mean():.3f}M")
    print(f"A1 {r2_score(y, champ_oof):.4f} -> {r2_score(y, cand_oof):.4f}")
    print(f"\npaired dSel {fr['d'].mean():+.5f}  t = {t:+.2f}  p = {p:.3g}  "
          f"n = {len(fr)}  [{'PASS' if t > DSEL_T_BAR else 'FAIL'}]")

    seg = df["signing_cat"].fillna("Unknown").values
    worst = 0.0
    for c in pd.unique(seg):
        m = seg == c
        if m.sum() >= 8:
            worst = max(worst, float(abs_bias_growth(
                (champ_oof[m] - y[m]) * cap_m[m], (cand_oof[m] - y[m]) * cap_m[m])))
    print(f"C2 worst |bias| growth ${worst:+.3f}M "
          f"[{'PASS' if worst <= C2_BAR else 'FAIL'}]")

    named = [("demar derozan", 2024), ("james harden", 2022), ("chris paul", 2025),
             ("lebron james", 2026), ("al horford", 2023), ("marc gasol", 2020)]
    key = list(zip(df["player_name_norm"], df["season"]))
    print(f"\n{'player':20s}{'P':>7s}{'actual':>9s}{'champ':>9s}{'cand':>9s}{'gain':>8s}")
    for k in named:
        if k not in key:
            continue
        i = key.index(k)
        gain = abs(champ_oof[i] - y[i]) - abs(cand_oof[i] - y[i])
        print(f"{k[0]:20s}{p_oof[i]:7.2f}{y[i]*cap_m[i]:9.1f}"
              f"{champ_oof[i]*cap_m[i]:9.1f}{cand_oof[i]*cap_m[i]:9.1f}"
              f"{gain*cap_m[i]:+8.2f}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ringchase_pweighted_eval.json").write_text(json.dumps({
        "PROVISIONAL": "migration will move ~15% of frame rows",
        "adopted": False, "seeds": list(seeds),
        "delta_grid": DELTA_GRID.tolist(), "tau_grid": TAU_GRID.tolist(),
        "selected": ch.groupby(["delta", "tau"]).size().to_dict().__repr__(),
        "spearman_blended_P": float(rho),
        "dSel": {"delta": float(fr["d"].mean()), "t": float(t), "p": float(p)},
        "A1_champ": float(r2_score(y, champ_oof)),
        "A1_cand": float(r2_score(y, cand_oof)),
        "c2_worst": float(worst),
        "pass": bool(t > DSEL_T_BAR and worst <= C2_BAR),
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT / 'ringchase_pweighted_eval.json'}")


if __name__ == "__main__":
    main()
