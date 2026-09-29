"""Ring-chase discount: hard age gate, P-weighted multiplicative pull on latent.

The fourth and best-motivated shape. Each piece is here because a measured
failure of the previous three put it here.

## The gate is hard, and it is age

A logistic on standardised age is monotone, so a large career-earnings value
compensates for being young: Karl-Anthony Towns at 28 drew the highest P in the
frame (0.333) while signing a $49.2M max the champion already under-priced by
$12.8M. That is not a tuning problem, it is a representation problem — the
model class cannot express a cliff.

A tree can, and was tried. It is worse everywhere (in-zone Spearman +0.170 for
depth 2 vs the logistic's +0.284) and it cuts the wrong way: its depth-2 fit
splits earnings first, then age at 32.5, assigning the HIGHER score to players
UNDER 32.5. The label ("champion over-prices by >= $2M") is dominated frame-wide
by young players on fresh contracts, so information gain sends the tree at the
big group. The logistic only gets the sign right because its form denies it the
freedom to do otherwise.

So the cliff goes in by hand. Measured gain-to-risk inside the gate, where risk
is the existing under-prediction on non-floor rows the pull would deepen:

    age >= 30   1 : 0.67        age >= 35   1 : 0.41
    age >= 33   1 : 0.45        age >= 37   1 : 0.10

35 is where it first turns clearly favourable, and adding `rings == 0` and
`earnings >= P75` leaves 22 rows at roughly 5:1, 14 of them at the floor where
a downward pull has NO downside at all (verified: of 264 at-floor rows, 155 are
over-priced, 109 are exact, and zero are under-priced — the clip guarantees
`pred >= actual` there).

**AGE_MIN = 35 was chosen by scanning, so it is not pre-registered.** It is
selected inside each fold's pool, and the held-out fold never informs its own
threshold.

## The weight is the logistic P, and it is strongest exactly here

In-zone Spearman against the champion's over-prediction:

    frame +0.078   at-floor +0.284   non-floor +0.113   age>=35 +0.351

The 35+ region is where P ranks best, better than the at-floor zone it was
originally aimed at.

## The form is multiplicative, on the latent, before the clip

Additive was measured and rejected: inside the cell the additive discount runs
$1.46M / $4.52M / $4.54M / $23.78M across predicted-value bands while the ratio
stays inside 0.48-0.75 (coefficient of variation 1.24 vs 0.39).

Position matters as much as form. The Stage-3 slot sits AFTER the floor clip,
where a player who discounted all the way to the minimum shows residual ~0 —
not because he needed no correction but because the clip already delivered it.
Pooling that zero with James Harden's $19.5M cannot estimate a common
parameter. On the latent, before push and clip, saturation falls out of the
existing clip instead of corrupting the estimate.

Usage:
    python scripts/eval_ringchase_gated.py --seeds 3 [--gate earnings|retire|continuous]
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
from src.model.stages import SIGNING_K, compose
from src.model.train import FEATURE_COLS, _XGB_BASE

AGE_GRID = (33, 35, 37)          # selected inside the pool, never on the fold
CELL_EARN_Q = 0.75
K_SHRINK = SIGNING_K             # 20, imported, never swept
# Widened from 1.5: the retire gate selected 1.5, the old edge, in 12 of 15
# pools, so the optimum may lie beyond it (see ISSUES #25 on edge optima).
DELTA_GRID = np.round(np.arange(0.0, 3.05, 0.05), 3)
LABEL_OVER_M = 2.0
CIRC = ["career_earnings_thru_prev_cap_pct", "age", "rings_thru_prev"]
C2_BAR = 0.30
DSEL_T_BAR = 2.0
OUT = OUTPUTS_DIR / "models"


# "ringless" is the original cell: no ring yet. "earnings" drops the ring
# condition, because the handoff's own evidence says the ring leg is weak or
# wrong-signed: Marc Gasol took a minimum the summer after his ring, and LeBron
# James holds four. Under "ringless" he can never enter the gate.
GATE_MODE = "ringless"


# "retire" gates on the probability that this is the player's last contract
# (scripts/build_retirement_hazard.py, frozen on careers before 2019) instead
# of absolute age. Grid entries are (window in seasons, threshold).
RETIRE_GRID = ((2, 0.15), (2, 0.25), (2, 0.35), (3, 0.25), (3, 0.35), (3, 0.45))


def gate(sub, g, p75):
    """Rows the pull may touch: near the end (by age or by P(last contract))
    and well paid; under "ringless" also without a ring."""
    rich = sub["career_earnings_thru_prev_cap_pct"].values >= p75
    if GATE_MODE == "retire":
        k, thr = g
        return rich & (np.nan_to_num(sub[f"p_last_{k}y"].values, nan=0.0) >= thr)
    member = rich & (sub["age"].values.astype(float) >= g)
    if GATE_MODE == "ringless":
        member &= np.nan_to_num(sub["rings_thru_prev"].values, nan=-1) == 0
    return member


def best_delta(latent, p, y, member, *, lo, hi, p_max, is_ext, ext_cap):
    """Grid-search the pull strength that minimises SSE on the gated rows.

    A grid because the objective runs through the push and the clip, which is
    the whole point of estimating it here rather than on the post-clip residual.
    """
    if member.sum() == 0:
        return 0.0
    best, best_sse = 0.0, np.inf
    for d in DELTA_GRID:
        lat = np.where(member, latent * (1.0 - p * d), latent)
        pred = compose(lat, lo=lo, hi=hi, p_max=p_max,
                       is_extension=is_ext, ext_cap_pct=ext_cap)
        sse = float(((pred[member] - y[member]) ** 2).sum())
        if sse < best_sse:
            best, best_sse = float(d), sse
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--gate",
                    choices=["ringless", "earnings", "retire", "continuous"],
                    default="ringless")
    args = ap.parse_args()
    global GATE_MODE, AGE_GRID, CIRC
    GATE_MODE = args.gate
    if GATE_MODE in ("retire", "continuous"):
        AGE_GRID = RETIRE_GRID
        CIRC = ["career_earnings_thru_prev_cap_pct", "p_last_2y",
                "rings_thru_prev"]
    seeds = tuple(DEFAULT_SEEDS[:args.seeds])

    df, base_features = load_evaluation_frame(verbose=False,
                                              allow_missing_computed=True)
    df = attach_extension_cap(df, verbose=False)
    df, clf_features = rm.attach_clf_features(df)
    ce = pd.read_csv(PROCESSED_DIR / "career_earnings.csv")
    rg = pd.read_csv(PROCESSED_DIR / "rings_thru_prev.csv")
    n0 = len(df)
    df = df.merge(ce[["player_name_norm", "season",
                      "career_earnings_thru_prev_cap_pct"]],
                  on=["player_name_norm", "season"], how="left")
    df = df.merge(rg[["player_name_norm", "season", "rings_thru_prev"]],
                  on=["player_name_norm", "season"], how="left")
    assert len(df) == n0, "circumstance merge changed the row count"
    df = df.reset_index(drop=True)
    df["career_earnings_thru_prev_cap_pct"] = (
        df["career_earnings_thru_prev_cap_pct"].fillna(0.0))
    df["rings_thru_prev"] = df["rings_thru_prev"].fillna(0)
    rh = pd.read_csv(PROCESSED_DIR / "retirement_hazard.csv")
    df = df.merge(rh[["player_name_norm", "season", "p_last_1y", "p_last_2y",
                      "p_last_3y"]],
                  on=["player_name_norm", "season"], how="left")
    assert len(df) == n0, "hazard merge changed the row count"
    print(f"  p_last coverage: {df['p_last_2y'].notna().mean():.3f}")
    for c in ("p_last_1y", "p_last_2y", "p_last_3y"):
        df[c] = df[c].fillna(0.0)

    y = df[TARGET].values
    cap_m = df["cap"].values / 1e6
    sel = ~df["is_confirmation"].values
    p75 = float(np.nanquantile(df["career_earnings_thru_prev_cap_pct"].values,
                               CELL_EARN_Q))
    for a in AGE_GRID:
        print(f"  gate age>={a}: n = {int(gate(df, a, p75).sum())}")

    # Same KF configuration as the production suite (prehistory anchors on,
    # v8.14x). The function's own default is prehistory=False, which silently
    # scores a pre-v8.14x champion: 231 rows fell back to kf = prev_cap_pct
    # against the suite's 15.
    kf_ctx = prepare_kf_context(df, base_features, prehistory=True,
                                expand_anchors=True)
    features = list(FEATURE_COLS)
    base_feats = [c if c != "kf_market_value" else "prev_cap_pct" for c in features]
    base_feats = [c for c in base_feats if c in df.columns]
    folds = list(GroupKFold(n_splits=N_SPLITS).split(
        df, y, df["player_name_norm"].values))

    champ_acc = np.zeros(len(df))
    cand_acc = np.zeros(len(df))
    p_acc = np.zeros(len(df))
    rows, chosen = [], []

    for si, seed in enumerate(seeds):
        per_fold = []
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
            # circumstance P, fitted inside the training slice
            b = XGBRegressor(**{**_XGB_BASE, "random_state": seed})
            b.fit(train[base_feats], y[tr])
            lab = (((b.predict(train[base_feats]) - y[tr]) * cap_m[tr])
                   >= LABEL_OVER_M).astype(int)
            sc = StandardScaler().fit(train[CIRC].values)
            lr = LogisticRegression(max_iter=2000).fit(sc.transform(train[CIRC].values), lab)
            p_ring = lr.predict_proba(sc.transform(test[CIRC].values))[:, 1]
            per_fold.append({"va": va, "latent": latent, "lo": lo, "hi": hi,
                             "p_max": p_max, "p_ring": p_ring, "champ": champ,
                             "is_ext": test["is_extension"].values,
                             "ext_cap": test["ext_cap_pct"].values})
            champ_acc[va] += champ
            p_acc[va] += p_ring

        for fi, rec in enumerate(per_fold):
            oth = [j for j in range(len(per_fold)) if j != fi]
            cat = lambda k: np.concatenate([per_fold[j][k] for j in oth])
            pool_idx = cat("va")
            pl, plo, phi, ppm = cat("latent"), cat("lo"), cat("hi"), cat("p_max")
            pie, pec, ppr = cat("is_ext"), cat("ext_cap"), cat("p_ring")
            py, pdf = y[pool_idx], df.iloc[pool_idx]

            if GATE_MODE == "continuous":
                # No gate, no threshold, no shrinkage. The player gives up a
                # share of what he is worth ABOVE the floor, in proportion to
                # P(last contract): a near-minimum latent barely moves, so a
                # declining role player the champion already prices low is
                # left alone. delta is chosen on absolute error, because the
                # squared-error choice ran to the grid edge on LeBron's row.
                pp = pdf["p_last_2y"].values
                best_raw, best_mae = 0.0, np.inf
                for d in DELTA_GRID:
                    lat = pl - d * pp * np.maximum(pl - plo, 0.0)
                    pred = compose(lat, lo=plo, hi=phi, p_max=ppm,
                                   is_extension=pie, ext_cap_pct=pec)
                    mae = float(np.abs(pred - py).mean())
                    if mae < best_mae:
                        best_raw, best_mae = float(d), mae
                delta = best_raw
                chosen.append({"seed": seed, "fold": fi, "age": "continuous",
                               "raw": best_raw, "delta": delta,
                               "pool_n": len(py)})
                pv_ = df.iloc[rec["va"]]["p_last_2y"].values
                lat = rec["latent"] - delta * pv_ * np.maximum(
                    rec["latent"] - rec["lo"], 0.0)
                cand = compose(lat, lo=rec["lo"], hi=rec["hi"],
                               p_max=rec["p_max"], is_extension=rec["is_ext"],
                               ext_cap_pct=rec["ext_cap"])
                cand_acc[rec["va"]] += cand
                va, s = rec["va"], sel[rec["va"]]
                rows.append({"champ": r2_score(y[va][s], rec["champ"][s]),
                             "cand": r2_score(y[va][s], cand[s])})
                continue

            best_age, best_score, best_raw = AGE_GRID[0], -np.inf, 0.0
            for a in AGE_GRID:
                mem = gate(pdf, a, p75)
                if mem.sum() < 8:
                    continue
                raw = best_delta(pl, ppr, py, mem, lo=plo, hi=phi, p_max=ppm,
                                 is_ext=pie, ext_cap=pec)
                n = int(mem.sum())
                d = (n / (n + K_SHRINK)) * raw
                lat = np.where(mem, pl * (1.0 - ppr * d), pl)
                s = r2_score(py, compose(lat, lo=plo, hi=phi, p_max=ppm,
                                         is_extension=pie, ext_cap_pct=pec))
                if s > best_score:
                    best_age, best_score, best_raw = a, s, raw

            mp = gate(pdf, best_age, p75)
            n = int(mp.sum())
            delta = (n / (n + K_SHRINK)) * best_raw
            chosen.append({"seed": seed, "fold": fi, "age": best_age,
                           "raw": best_raw, "delta": delta, "pool_n": n})

            mem = gate(df.iloc[rec["va"]], best_age, p75)
            lat = np.where(mem, rec["latent"] * (1.0 - rec["p_ring"] * delta),
                           rec["latent"])
            cand = compose(lat, lo=rec["lo"], hi=rec["hi"], p_max=rec["p_max"],
                           is_extension=rec["is_ext"], ext_cap_pct=rec["ext_cap"])
            cand_acc[rec["va"]] += cand
            va, s = rec["va"], sel[rec["va"]]
            rows.append({"champ": r2_score(y[va][s], rec["champ"][s]),
                         "cand": r2_score(y[va][s], cand[s])})
        print(f"  seed {seed} done ({si+1}/{len(seeds)})", flush=True)

    champ_oof = champ_acc / len(seeds)
    cand_oof = cand_acc / len(seeds)
    p_oof = p_acc / len(seeds)
    fr = pd.DataFrame(rows)
    fr["d"] = fr["cand"] - fr["champ"]
    t, pv = stats.ttest_1samp(fr["d"], 0)
    ch = pd.DataFrame(chosen)

    print("\nSELECTED IN POOL")
    print(ch.groupby("age").agg(folds=("age", "size"), mean_raw=("raw", "mean"),
                                mean_delta=("delta", "mean")).to_string())

    moved = np.abs(cand_oof - champ_oof) > 1e-12
    if GATE_MODE == "continuous":
        at_edge = int((ch["raw"] >= DELTA_GRID[-1] - 1e-9).sum())
        print(f"delta at the grid edge in {at_edge} of {len(ch)} pools")
        mem_all = moved
    else:
        mem_all = gate(df, ch["age"].mode().iloc[0], p75)
    print(f"\nmoved {int(moved.sum())} rows; modal gate n = {int(mem_all.sum())}")
    print(f"bit-identity outside modal gate: "
          f"{'PASS' if not (moved & ~mem_all).any() else 'FAIL'}")

    e_c = np.abs(champ_oof - y) * cap_m
    e_d = np.abs(cand_oof - y) * cap_m
    print(f"gate MAE  ${e_c[mem_all].mean():.3f}M -> ${e_d[mem_all].mean():.3f}M")
    print(f"frame MAE ${e_c.mean():.3f}M -> ${e_d.mean():.3f}M")
    print(f"A1 {r2_score(y, champ_oof):.4f} -> {r2_score(y, cand_oof):.4f}")
    print(f"\npaired dSel {fr['d'].mean():+.5f}  t = {t:+.2f}  p = {pv:.3g}  "
          f"n = {len(fr)}  [{'PASS' if t > DSEL_T_BAR else 'FAIL'}]")

    seg = df["signing_cat"].fillna("Unknown").values
    worst = 0.0
    for c in pd.unique(seg):
        m = seg == c
        if m.sum() < 8:
            continue
        # abs_bias_growth takes the segment's MEAN bias, not the row vector,
        # and the candidate comes first: growth = |cand| - |champ|.
        worst = max(worst, float(abs_bias_growth(
            float(((cand_oof[m] - y[m]) * cap_m[m]).mean()),
            float(((champ_oof[m] - y[m]) * cap_m[m]).mean()))))
    print(f"C2 worst |bias| growth ${worst:+.3f}M "
          f"[{'PASS' if worst <= C2_BAR else 'FAIL'}]")

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "player": df["player_name_norm"], "season": df["season"],
        "age": df["age"], "signing_cat": df["signing_cat"],
        "in_gate": mem_all, "moved": moved, "p_ring": p_oof,
        "actual_m": y*cap_m, "champ_m": champ_oof*cap_m, "cand_m": cand_oof*cap_m,
        "err_champ_m": (champ_oof-y)*cap_m, "err_cand_m": (cand_oof-y)*cap_m,
    }).to_csv(OUT / f"ringchase_gated_{GATE_MODE}_oof.csv", index=False)
    (OUT / f"ringchase_gated_{GATE_MODE}_eval.json").write_text(json.dumps({
        "PROVISIONAL": "migration will move ~15% of frame rows",
        "adopted": False, "gate": GATE_MODE, "age_grid": [list(g) if isinstance(g, tuple) else g for g in AGE_GRID], "k_shrink": K_SHRINK,
        "selected_ages": {str(k): int(v) for k, v in
                          ch["age"].value_counts().items()},
        "mean_delta": float(ch["delta"].mean()),
        "dSel": {"delta": float(fr["d"].mean()), "t": float(t), "p": float(pv)},
        "A1_champ": float(r2_score(y, champ_oof)),
        "A1_cand": float(r2_score(y, cand_oof)),
        "c2_worst": float(worst),
        "pass": bool(t > DSEL_T_BAR and worst <= C2_BAR),
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT / f'ringchase_gated_{GATE_MODE}_eval.json'}")


if __name__ == "__main__":
    main()
