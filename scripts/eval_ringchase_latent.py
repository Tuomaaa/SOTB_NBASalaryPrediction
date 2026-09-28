"""Ring-chase discount as a MULTIPLICATIVE factor on the Stage-1 latent.

Supersedes the additive Stage-3 arm in `eval_ringchase_offset.py`, which
failed (dSel t -0.39 / -0.52 / -3.03). Two things were wrong with it and both
are fixed here.

## 1. Wrong slot in the chain

The additive arm sat where the Stage-3 signing offset sits:

    latent -> push -> clip[lo,hi] -> OFFSET -> mech cap -> ext clip -> re-clip

An offset there sees the residual AFTER the floor clip has already saturated.
A player who discounted all the way to the minimum has residual ~0 — not
because he needed no correction, but because the clip already delivered it.
Pooling his zero with James Harden's $19.5M cannot estimate a common constant:
the two differ in how much the pipeline already did, not in what they need.

So the factor is applied to the LATENT, before push and clip:

    latent * FACTOR -> push -> clip[lo,hi] -> (Stage 3 unchanged)

Saturation then falls out of the existing clip. A member who was floor-pinned
stays floor-pinned and comes out bit-identical; a member with room moves.

## 2. Wrong functional form

Measured inside the pre-registered cell, on the 49 rows the champion
over-prices by more than $0.5M:

    form                 mean   median      sd   coef of variation
    additive ($M)        5.80     3.74    7.17    1.24
    multiplicative       0.57     0.60    0.23    0.39

and by predicted-value band the additive discount runs $1.46M / $4.52M /
$4.54M / $23.78M from the bottom band to the top while the ratio stays inside
0.48-0.75. The behaviour is "takes about 40% less", not "takes $5M less". A
constant in cap_pct cannot be right for a group spanning $2M to $47M.

(That table conditions on being over-priced, so 0.57 itself is biased low as
an estimate. The BAND COMPARISON is what settles the form, and it is robust to
the filter: the same filter applies within every band.)

## What is pre-registered, and what is not

Pre-registered before any arm was scored:
  * the cell rule `rings == 0 & age >= AGE & career_earnings >= frame P75`
  * `K_SHRINK = SIGNING_K = 20`, imported, never swept
  * the gate bars, inherited from the floor-branch and Stage-3 harnesses

NOT pre-registered: the age threshold. A scan of six (age, earnings) cells was
run and reported earlier in the session, so the pre-registration for AGE is
spent — the number cannot be chosen by looking at a score again. It is
therefore selected INSIDE each fold's pool (`AGE_GRID`), and the held-out fold
never informs its own threshold. Same for the factor itself.

Everything here is PROVISIONAL: the Spotrac migration will move ~15% of the
frame's rows, and this harness's champion reproduction differs from the stored
`oof_champion` (A1 0.805 vs 0.827, max abs prediction diff 0.069). Both arms
share that reproduction, so the PAIRED delta is meaningful and the absolute
levels are not.

Usage:
    python scripts/eval_ringchase_latent.py --seeds 3
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

from config import CAP_BY_SEASON, OUTPUTS_DIR, PROCESSED_DIR
from src.model.evaluate_suite import (
    DEFAULT_SEEDS, N_SPLITS, TARGET, _compute_kf_nested, abs_bias_growth,
    load_evaluation_frame, paired_delta, prepare_kf_context,
)
from src.model import route_mixture as rm
from src.model.extension_cap import attach_extension_cap
from src.model.stages import SIGNING_K, compose

CELL_RINGS_MAX = 0
CELL_EARN_Q = 0.75
K_SHRINK = SIGNING_K              # 20. imported, never restated, never swept
AGE_GRID = (30, 32, 34, 36)       # selected inside the pool, never on the fold
FACTOR_GRID = np.round(np.arange(0.70, 1.005, 0.02), 3)

C2_BAR = 0.30                     # $M worst per-segment |bias| growth
DSEL_T_BAR = 2.0
CAP_DISPLAY = CAP_BY_SEASON.get(2026, 153_000_000)
OUT = OUTPUTS_DIR / "models"


def cell_mask(df, age_min, p75):
    earn = df["career_earnings_thru_prev"].values.astype(float)
    rings = np.nan_to_num(df["rings_thru_prev"].values, nan=-1)
    age = df["age"].values.astype(float)
    return (np.isfinite(earn) & (rings == CELL_RINGS_MAX)
            & (age >= age_min) & (earn >= p75))


def shrink_factor(raw, n, k=K_SHRINK):
    """Shrink a raw factor toward 1.0 — the no-op — by n/(n+k).

    Same shrinkage the Stage-3 offset uses, with 1.0 in place of 0.0 as the
    neutral element because the correction is multiplicative here.
    """
    s = n / (n + k) if n else 0.0
    return 1.0 + s * (raw - 1.0), s


def best_factor(latent, y, member, *, lo, hi, p_max, is_ext, ext_cap):
    """Grid-search the factor that minimises SSE on `member` after compose.

    A grid rather than a closed form because the objective runs through the
    push and the clip, which are not differentiable and which are exactly the
    reason the additive arm mis-estimated: the correction has to be judged
    after saturation, not before it.
    """
    if member.sum() == 0:
        return 1.0
    best, best_sse = 1.0, np.inf
    for f in FACTOR_GRID:
        lat = np.where(member, latent * f, latent)
        pred = compose(lat, lo=lo, hi=hi, p_max=p_max,
                       is_extension=is_ext, ext_cap_pct=ext_cap)
        sse = float(((pred[member] - y[member]) ** 2).sum())
        if sse < best_sse:
            best, best_sse = float(f), sse
    return best


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

    y = df[TARGET].values
    cap_m = df["cap"].values / 1e6
    lo_all = df["floor_pct"].values
    hi_all = df["max_eligible_pct"].values
    is_ext = df["is_extension"].values
    ext_cap = df["ext_cap_pct"].values
    sel = ~df["is_confirmation"].values
    p75 = float(np.nanquantile(df["career_earnings_thru_prev"].values.astype(float),
                               CELL_EARN_Q))
    print(f"frame {len(df)} rows; earnings P75 = ${p75/1e6:.1f}M")
    for a in AGE_GRID:
        print(f"  cell age>={a}: n = {int(cell_mask(df, a, p75).sum())}")

    kf_ctx = prepare_kf_context(df, clf_features)
    features = list(rm.FEATURE_COLS) if hasattr(rm, "FEATURE_COLS") else None
    from src.model.train import FEATURE_COLS
    features = list(FEATURE_COLS)
    folds = list(GroupKFold(n_splits=N_SPLITS).split(
        df, y, df["player_name_norm"].values))

    champ_acc = np.zeros(len(df))
    cand_acc = np.zeros(len(df))
    fold_rows = []
    chosen = []
    # Sensitivity curve: a FIXED factor applied to the whole cell, no fitting
    # and no selection. Reported as a curve so "could it be more aggressive?"
    # is answered by the shape rather than by picking the best point.
    SWEEP = np.round(np.arange(0.50, 1.001, 0.05), 2)
    sweep_acc = {f: np.zeros(len(df)) for f in SWEEP}

    for si, seed in enumerate(seeds):
        # pass 1: latent + compose inputs for every fold
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
            mem_sw = cell_mask(test, min(AGE_GRID), p75)
            for fsw in SWEEP:
                lat_sw = np.where(mem_sw, latent * fsw, latent)
                sweep_acc[fsw][va] += compose(
                    lat_sw, lo=lo, hi=hi, p_max=p_max,
                    is_extension=test["is_extension"].values,
                    ext_cap_pct=test["ext_cap_pct"].values)
            per_fold.append({"va": va, "latent": latent, "lo": lo, "hi": hi,
                             "p_max": p_max,
                             "is_ext": test["is_extension"].values,
                             "ext_cap": test["ext_cap_pct"].values,
                             "champ": champ})
            champ_acc[va] += champ

        # pass 2: leave-fold-out age threshold AND factor, applied to fold f
        for fi, rec in enumerate(per_fold):
            pool_idx = np.concatenate([per_fold[j]["va"]
                                       for j in range(len(per_fold)) if j != fi])
            pool_lat = np.concatenate([per_fold[j]["latent"]
                                       for j in range(len(per_fold)) if j != fi])
            pool_lo = np.concatenate([per_fold[j]["lo"] for j in range(len(per_fold)) if j != fi])
            pool_hi = np.concatenate([per_fold[j]["hi"] for j in range(len(per_fold)) if j != fi])
            pool_pm = np.concatenate([per_fold[j]["p_max"] for j in range(len(per_fold)) if j != fi])
            pool_ie = np.concatenate([per_fold[j]["is_ext"] for j in range(len(per_fold)) if j != fi])
            pool_ec = np.concatenate([per_fold[j]["ext_cap"] for j in range(len(per_fold)) if j != fi])
            pool_y = y[pool_idx]
            pool_df = df.iloc[pool_idx]

            # choose the age threshold on the pool only
            best_age, best_score, best_raw = AGE_GRID[0], -np.inf, 1.0
            for a in AGE_GRID:
                mem = cell_mask(pool_df, a, p75)
                if mem.sum() < 10:
                    continue
                raw = best_factor(pool_lat, pool_y, mem, lo=pool_lo, hi=pool_hi,
                                  p_max=pool_pm, is_ext=pool_ie, ext_cap=pool_ec)
                fac, _ = shrink_factor(raw, int(mem.sum()))
                lat = np.where(mem, pool_lat * fac, pool_lat)
                pred = compose(lat, lo=pool_lo, hi=pool_hi, p_max=pool_pm,
                               is_extension=pool_ie, ext_cap_pct=pool_ec)
                score = r2_score(pool_y, pred)
                if score > best_score:
                    best_age, best_score, best_raw = a, score, raw

            mem_pool = cell_mask(pool_df, best_age, p75)
            factor, shr = shrink_factor(best_raw, int(mem_pool.sum()))
            chosen.append({"seed": seed, "fold": fi, "age": best_age,
                           "raw": best_raw, "factor": factor, "shrink": shr,
                           "pool_n": int(mem_pool.sum())})

            mem = cell_mask(df.iloc[rec["va"]], best_age, p75)
            lat = np.where(mem, rec["latent"] * factor, rec["latent"])
            cand = compose(lat, lo=rec["lo"], hi=rec["hi"], p_max=rec["p_max"],
                           is_extension=rec["is_ext"], ext_cap_pct=rec["ext_cap"])
            cand_acc[rec["va"]] += cand
            va = rec["va"]
            s = sel[va]
            fold_rows.append({
                "seed": seed, "fold": fi,
                "champ": r2_score(y[va][s], rec["champ"][s]),
                "cand": r2_score(y[va][s], cand[s]),
                "n_member": int(mem.sum()),
                "moved": int((np.abs(cand - rec["champ"]) > 1e-12).sum()),
            })
        print(f"  seed {seed} done ({si+1}/{len(seeds)})", flush=True)

    champ_oof = champ_acc / len(seeds)
    cand_oof = cand_acc / len(seeds)
    fr = pd.DataFrame(fold_rows)
    fr["d"] = fr["cand"] - fr["champ"]
    from scipy import stats
    t, p = stats.ttest_1samp(fr["d"], 0)

    ch = pd.DataFrame(chosen)
    print("\nSELECTED INSIDE THE POOL (never on the held-out fold)")
    print(ch.groupby("age").agg(folds=("age", "size"),
                                mean_raw=("raw", "mean"),
                                mean_factor=("factor", "mean")).to_string())

    mem_all = cell_mask(df, int(ch["age"].mode().iloc[0]), p75)
    moved = np.abs(cand_oof - champ_oof) > 1e-12
    print(f"\nrows moved: {int(moved.sum())}  (modal cell n = {int(mem_all.sum())})")
    print(f"rows in cell that did NOT move (floor-pinned or clip-absorbed): "
          f"{int((mem_all & ~moved).sum())}")
    print(f"bit-identity outside the modal cell: "
          f"{'PASS' if not (moved & ~mem_all).any() else 'FAIL'}")

    e_c = np.abs(champ_oof - y) * cap_m
    e_d = np.abs(cand_oof - y) * cap_m
    print(f"\ngroup MAE  ${e_c[mem_all].mean():.3f}M -> ${e_d[mem_all].mean():.3f}M "
          f"(win ${e_c[mem_all].mean()-e_d[mem_all].mean():+.3f}M)")
    print(f"frame MAE  ${e_c.mean():.3f}M -> ${e_d.mean():.3f}M")
    print(f"A1 {r2_score(y, champ_oof):.4f} -> {r2_score(y, cand_oof):.4f}")

    print(f"\npaired dSel {fr['d'].mean():+.5f}  t = {t:+.2f}  p = {p:.3g}  "
          f"n = {len(fr)}  [{'PASS' if t > DSEL_T_BAR else 'FAIL'}]")

    seg = df["signing_cat"].fillna("Unknown").values
    worst = 0.0
    for cat in pd.unique(seg):
        m = seg == cat
        if m.sum() < 8:
            continue
        # abs_bias_growth takes the two SEGMENT MEANS, not the residual
        # vectors: |bias_cand| - |bias_champ|, so a bias shrinking toward zero
        # reads as an improvement (ISSUES #20c).
        g = abs_bias_growth(float(((cand_oof[m] - y[m]) * cap_m[m]).mean()),
                            float(((champ_oof[m] - y[m]) * cap_m[m]).mean()))
        worst = max(worst, g)
    print(f"C2 worst |bias| growth ${worst:+.3f}M "
          f"[{'PASS' if worst <= C2_BAR else 'FAIL'}]")

    mem_sweep = cell_mask(df, min(AGE_GRID), p75)
    print(f"\nFIXED-FACTOR SENSITIVITY (cell age>={min(AGE_GRID)}, "
          f"n={int(mem_sweep.sum())}) — a curve, not a selection")
    print(f"  {'factor':>7s}{'moved':>7s}{'group MAE':>11s}{'frame MAE':>11s}"
          f"{'A1':>9s}{'over-pred rows':>15s}")
    for f in SWEEP:
        pv = sweep_acc[f] / len(seeds)
        e = np.abs(pv - y) * cap_m
        mvd = int((np.abs(pv - champ_oof) > 1e-12).sum())
        over = int(((pv - y)[mem_sweep] > 0).sum())
        print(f"  {f:7.2f}{mvd:7d}{e[mem_sweep].mean():11.3f}{e.mean():11.3f}"
              f"{r2_score(y, pv):9.4f}{over:15d}")

    OUT.mkdir(parents=True, exist_ok=True)
    # Per-row OOF, so the arm can be read on the players it was built for
    # rather than only in aggregate.
    pd.DataFrame({
        "player_name_norm": df["player_name_norm"], "season": df["season"],
        "age": df["age"], "signing_cat": df["signing_cat"],
        "career_earnings_thru_prev": df["career_earnings_thru_prev"],
        "rings_thru_prev": df["rings_thru_prev"],
        "in_cell": mem_all, "moved": moved,
        "actual_m": y * cap_m, "champ_m": champ_oof * cap_m,
        "cand_m": cand_oof * cap_m,
        "err_champ_m": (champ_oof - y) * cap_m,
        "err_cand_m": (cand_oof - y) * cap_m,
    }).to_csv(OUT / "ringchase_latent_oof.csv", index=False)
    (OUT / "ringchase_latent_eval.json").write_text(json.dumps({
        "PROVISIONAL": "migration will move ~15% of frame rows; champion "
                       "reproduction differs from stored oof_champion",
        "adopted": False, "seeds": list(seeds), "k_shrink": K_SHRINK,
        "age_grid": list(AGE_GRID), "p75_dollars": p75,
        "selected_ages": ch["age"].value_counts().to_dict(),
        "mean_factor": float(ch["factor"].mean()),
        "dSel": {"delta": float(fr["d"].mean()), "t": float(t), "p": float(p)},
        "A1_champ": float(r2_score(y, champ_oof)),
        "A1_cand": float(r2_score(y, cand_oof)),
        "group_mae_champ_m": float(e_c[mem_all].mean()),
        "group_mae_cand_m": float(e_d[mem_all].mean()),
        "c2_worst": float(worst),
        "pass": bool(t > DSEL_T_BAR and worst <= C2_BAR),
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT / 'ringchase_latent_eval.json'}")


if __name__ == "__main__":
    main()
