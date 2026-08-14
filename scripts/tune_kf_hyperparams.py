"""Efficient KF hyperparameter grid search: P0 x Q_FLOOR.

The inner CV (fitting 4 inner XGBoost models per outer fold) does NOT depend on
P0 or Q_FLOOR, so we run it ONCE and cache the results. Each (P0, Q_FLOOR)
combo then only needs 5 outer model fits (one per fold), reducing runtime from
~20 inner + 5 outer = 25 fits per combo to just 5.

Memory note: XGBoost's C++ backend fragments memory across hundreds of
sequential fits in one process. This script runs each P0 row in a subprocess
(6 Q_FLOOR combos per process) to avoid OOM.

Usage:
    python scripts/tune_kf_hyperparams.py [--seeds 1]
"""

import argparse
import gc
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

from config import OUTPUTS_DIR
from src.model.evaluate_suite import (
    N_SPLITS, load_evaluation_frame, _dollars,
)
from src.model.route_mixture import attach_clf_features
from src.model.stages import (
    signing_offsets, stage3_signing, SIGNING_K,
)
from src.model.train import FEATURE_COLS, TARGET

from scripts.ablation_kf_market_value_full import (
    fit_champion_models, predict_champion,
    prepare_full_frame, build_anchor_map, load_prehistory_anchors,
    estimate_q, kalman_update,
    N_INNER, KF_COL, PREV_COL,
    P0 as DEFAULT_P0, Q_FLOOR as DEFAULT_Q_FLOOR,
)

# Sweep grid (fine P0 sweep at plateau Q values)
P0_GRID = [0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
Q_FLOOR_GRID = [0.02, 0.04]

CACHE_DIR = Path(__file__).resolve().parent.parent / "outputs" / "models"


def _run_inner_cv(seeds, df_eval, features, clf_features, y, groups, folds,
                  needed_idx, full_needed, df_full):
    """Step 1: run inner CV once, cache per-(seed, fold) results."""
    n_seeds = len(seeds)
    inner_oof_cache = [[None] * N_SPLITS for _ in seeds]
    full_pred_cache = [[None] * N_SPLITS for _ in seeds]
    player_inner_cache = [[None] * N_SPLITS for _ in seeds]
    r_var_cache = np.zeros((n_seeds, N_SPLITS))
    estimated_q_cache = np.zeros((n_seeds, N_SPLITS))

    for si, seed in enumerate(seeds):
        t_seed = time.time()
        for fi, (tr, va) in enumerate(folds):
            train_df = df_eval.iloc[tr]
            y_tr, g_tr = y[tr], groups[tr]

            inner_folds = list(GroupKFold(n_splits=N_INNER)
                               .split(train_df, y_tr, g_tr))
            inner_oof = np.full(len(tr), np.nan)
            full_pred = np.zeros((N_INNER, len(needed_idx)))
            player_inner = {}

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
                del model, clf

            resid = y_tr - inner_oof
            r_var = float(np.var(resid, ddof=1))
            est_q, _, _ = estimate_q(df_full, set(g_tr), r_var)

            inner_oof_cache[si][fi] = inner_oof
            full_pred_cache[si][fi] = full_pred
            player_inner_cache[si][fi] = player_inner
            r_var_cache[si, fi] = r_var
            estimated_q_cache[si, fi] = est_q

            inner_r2 = r2_score(y_tr, inner_oof)
            print(f"  seed {seed} fold {fi}: inner R2={inner_r2:.4f}  "
                  f"R={r_var:.5f}  est_Q={est_q:.5f}")

        print(f"  seed {seed} done in {time.time() - t_seed:.0f}s", flush=True)
    gc.collect()
    return (inner_oof_cache, full_pred_cache, player_inner_cache,
            r_var_cache, estimated_q_cache)


def _run_sweep_batch(cache_path, p0_val, q_floor_list, batch_idx):
    """Run one P0 row (all Q_FLOOR combos) in a fresh subprocess.

    Loads cached inner-CV data from pickle, runs the outer model fits,
    returns results as JSON to stdout.
    """
    from src.model.route_mixture import (
        grabit_latent, train_route_classifier, route_proba, MAX_IDX,
    )
    from src.model.stages import compose

    with open(cache_path, "rb") as f:
        cache = pickle.load(f)

    full_pred_cache = cache["full_pred_cache"]
    player_inner_cache = cache["player_inner_cache"]
    r_var_cache = cache["r_var_cache"]
    estimated_q_cache = cache["estimated_q_cache"]
    y = cache["y"]
    groups = cache["groups"]
    sel = cache["sel"]
    prev = cache["prev"]
    recent = cache["recent"]
    anchor = cache["anchor"]
    inter_pos = cache["inter_pos"]
    tier = cache["tier"]
    anchor_kind = cache["anchor_kind"]
    folds = cache["folds"]
    features_swap = cache["features_swap"]
    clf_features = cache["clf_features"]
    cat_all = cache["cat_all"]
    lo_all = cache["lo_all"]
    hi_all = cache["hi_all"]
    is_ext_all = cache["is_ext_all"]
    ext_cap_all = cache["ext_cap_all"]
    mech_all = cache["mech_all"]
    fold_of = cache["fold_of"]
    seeds = cache["seeds"]
    n_seeds = len(seeds)
    df_eval = cache["df_eval"]

    full_mean_cache = {}
    for si in range(n_seeds):
        for fi in range(N_SPLITS):
            full_mean_cache[(si, fi)] = full_pred_cache[si][fi].mean(axis=0)

    # Pre-build fold frame templates
    fold_frames = []
    for fi, (tr, va) in enumerate(folds):
        train_tmpl = df_eval.iloc[tr].copy()
        test_tmpl = df_eval.iloc[va].copy()
        train_tmpl[KF_COL] = 0.0
        test_tmpl[KF_COL] = 0.0
        fold_frames.append((train_tmpl, test_tmpl, tr, va))

    batch_results = []

    for qf_val in q_floor_list:
        t_combo = time.time()

        oof_swap_acc = np.zeros(len(df_eval))
        oof_swap_by_seed = np.zeros((n_seeds, len(df_eval)))

        for si, seed in enumerate(seeds):
            for fi, (train_tmpl, test_tmpl, tr, va) in enumerate(fold_frames):
                r_var = r_var_cache[si, fi]
                q_var = max(estimated_q_cache[si, fi], qf_val)

                fp = full_pred_cache[si][fi]
                fm = full_mean_cache[(si, fi)]
                pi = player_inner_cache[si][fi]
                p0_fc = r_var * 2

                # Compute KF values for train and test rows
                kf_fill = float(np.nanmedian(prev))
                for rows, col_target in ((tr, train_tmpl), (va, test_tmpl)):
                    kf_out = np.full(len(rows), np.nan)
                    for k, idx in enumerate(rows):
                        if not np.isfinite(anchor[idx]):
                            continue
                        pl = inter_pos[idx]
                        if not pl:
                            kf_out[k] = anchor[idx]
                            continue
                        j = pi.get(groups[idx])
                        zs = fp[j, pl] if j is not None else fm[pl]
                        if anchor_kind[idx] == "first_contract":
                            p0_i = p0_fc
                        elif tier[idx] == 2:
                            p0_i = r_var
                        else:
                            p0_i = p0_val
                        kf_out[k] = kalman_update(
                            anchor[idx], zs, q_var, r_var, p0_i)
                    if col_target is train_tmpl:
                        kf_fill = (float(np.nanmedian(kf_out))
                                   if np.isfinite(kf_out).any()
                                   else float(np.nanmedian(prev)))
                    kf_out = np.where(np.isfinite(kf_out), kf_out, kf_fill)
                    col_target[KF_COL] = kf_out

                # Inline champion fitter: grabit + route classifier + compose
                latent, lo, hi = grabit_latent(
                    train_tmpl, test_tmpl, features_swap, seed)
                clf = train_route_classifier(
                    train_tmpl, clf_features, seed)
                p_max = route_proba(clf, test_tmpl, clf_features)[:, MAX_IDX]
                pred = compose(
                    latent, lo=lo, hi=hi, p_max=p_max,
                    is_extension=test_tmpl["is_extension"].values,
                    ext_cap_pct=test_tmpl["ext_cap_pct"].values)
                del clf

                oof_swap_acc[va] += pred
                oof_swap_by_seed[si, va] = pred

        oof_swap = oof_swap_acc / n_seeds

        # Stage 2 metrics
        a1_r2 = float(r2_score(y, oof_swap))
        sel_r2 = float(r2_score(y[sel], oof_swap[sel]))
        a2_r2 = float(r2_score(y[recent], oof_swap[recent]))
        mae_m, bias_m = _dollars(df_eval, oof_swap)

        # Stage 3: signing offset (leave-fold-out)
        resid_oof = y - oof_swap_by_seed.mean(axis=0)
        lfo = {fi_idx: signing_offsets(resid_oof, cat_all,
                                       pool=fold_of != fi_idx,
                                       k=SIGNING_K, detail=True)
               for fi_idx in range(N_SPLITS)}

        oof_s3_by_seed = np.zeros_like(oof_swap_by_seed)
        for fi_idx, (_tr, va) in enumerate(folds):
            for si_idx in range(n_seeds):
                oof_s3_by_seed[si_idx, va] = stage3_signing(
                    oof_swap_by_seed[si_idx, va], cat_all[va],
                    lfo[fi_idx], lo=lo_all[va], hi=hi_all[va],
                    mech_cap_pct=(mech_all[va] if mech_all is not None
                                  else None),
                    is_extension=is_ext_all[va],
                    ext_cap_pct=ext_cap_all[va])
        oof_s3 = oof_s3_by_seed.mean(axis=0)

        s3_a1_r2 = float(r2_score(y, oof_s3))
        s3_sel_r2 = float(r2_score(y[sel], oof_s3[sel]))
        s3_a2_r2 = float(r2_score(y[recent], oof_s3[recent]))
        s3_mae_m, s3_bias_m = _dollars(df_eval, oof_s3)

        elapsed = time.time() - t_combo
        is_default = (p0_val == DEFAULT_P0 and qf_val == DEFAULT_Q_FLOOR)

        batch_results.append({
            "p0": p0_val, "q_floor": qf_val, "is_default": is_default,
            "s2_a1_r2": a1_r2, "s2_sel_r2": sel_r2,
            "s2_a2_r2": a2_r2, "s2_mae_m": mae_m, "s2_bias_m": bias_m,
            "s3_a1_r2": s3_a1_r2, "s3_sel_r2": s3_sel_r2,
            "s3_a2_r2": s3_a2_r2, "s3_mae_m": s3_mae_m,
            "s3_bias_m": s3_bias_m, "elapsed_s": round(elapsed, 1),
        })
        tag = " <-- DEFAULT" if is_default else ""
        print(f"  P0={p0_val:.5f} Q_FLOOR={qf_val:.4f}  "
              f"S2: A1={a1_r2:.4f} MAE=${mae_m:.2f}M  "
              f"S3: A1={s3_a1_r2:.4f} MAE=${s3_mae_m:.2f}M  "
              f"({elapsed:.0f}s){tag}", file=sys.stderr, flush=True)
        gc.collect()

    # Write results to stdout as JSON
    print(json.dumps(batch_results))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=1,
                    help="number of seeds (1 = seed 0 only, for speed)")
    ap.add_argument("--worker", type=str, default=None,
                    help=argparse.SUPPRESS)  # internal: P0 value for worker
    args = ap.parse_args()

    # --- Worker mode: run one P0 row and exit ---
    if args.worker is not None:
        p0_val = float(args.worker)
        cache_path = CACHE_DIR / "_kf_sweep_cache.pkl"
        _run_sweep_batch(cache_path, p0_val, Q_FLOOR_GRID, 0)
        return

    # --- Coordinator mode ---
    seeds = list(range(args.seeds))
    n_seeds = len(seeds)
    t_start = time.time()

    line = "=" * 70
    print(line)
    print("KF HYPERPARAMETER GRID SEARCH: P0 x Q_FLOOR")
    print(f"P0 grid:      {P0_GRID}")
    print(f"Q_FLOOR grid: {Q_FLOOR_GRID}")
    print(f"Combos: {len(P0_GRID) * len(Q_FLOOR_GRID)}")
    print(f"Seeds: {seeds}")
    print(line)

    # --- Load data -----------------------------------------------------------
    df_eval, frame_features = load_evaluation_frame(allow_missing_computed=True)
    df_eval, clf_features = attach_clf_features(df_eval)

    features = [PREV_COL if f == KF_COL else f for f in FEATURE_COLS]
    missing = [f for f in features if f not in df_eval.columns]
    assert not missing, f"baseline features absent from the frame: {missing}"
    assert PREV_COL in features and KF_COL not in features

    unfilled = [f for f in features if f not in frame_features]
    if unfilled:
        df_eval[unfilled] = (df_eval[unfilled]
                             .fillna(df_eval[unfilled].median()).fillna(0))
        print(f"median-filled {unfilled}")

    y = df_eval[TARGET].values
    groups = df_eval["player_name_norm"].values
    sel = ~df_eval["is_confirmation"].values
    prev = df_eval["prev_cap_pct"].values
    seas_all = df_eval["season"].values.astype(int)
    recent = seas_all >= 2024

    features_swap = [KF_COL if f == PREV_COL else f for f in features]

    print(f"\nPreparing the full frame...")
    df_full = prepare_full_frame(df_eval, features)

    print("\nLoading pre-2019 Year-1 anchors (--prehistory):")
    prehistory_events = load_prehistory_anchors()

    print("\nBuilding anchor map (--expand-anchors):")
    (inter_idx, needed_idx, tier, anchor_val,
     anchor_prorated, anchor_kind) = build_anchor_map(
        df_eval, df_full, extra_events=prehistory_events,
        expand_anchors=True)
    anchor = np.where(tier > 0, anchor_val, prev)
    pos_of = {fi: k for k, fi in enumerate(needed_idx)}
    inter_pos = [[pos_of[fi] for fi in lst] for lst in inter_idx]
    full_needed = df_full.iloc[needed_idx]

    n_anchor = int((tier > 0).sum())
    n_inter = sum(1 for lst in inter_idx if lst)
    print(f"eval frame {len(df_eval)} rows; anchored {n_anchor}; "
          f"{n_inter} with intermediates")

    folds = list(GroupKFold(n_splits=N_SPLITS).split(df_eval, y, groups))

    cat_all = df_eval["signing_cat"].values
    lo_all = df_eval["floor_pct"].values
    hi_all = df_eval["max_eligible_pct"].values
    is_ext_all = df_eval["is_extension"].values
    ext_cap_all = df_eval["ext_cap_pct"].values
    mech_all = (df_eval["mech_cap_pct"].values
                if "mech_cap_pct" in df_eval.columns else None)
    fold_of = np.empty(len(df_eval), dtype=int)
    for fi_idx, (_tr, va) in enumerate(folds):
        fold_of[va] = fi_idx

    # =========================================================================
    # STEP 1: Run inner CV ONCE and cache everything
    # =========================================================================
    print(f"\n{line}")
    print("STEP 1: Running inner CV once (caching inner_oof, full_pred, "
          "player_inner)")
    print(line)

    (inner_oof_cache, full_pred_cache, player_inner_cache,
     r_var_cache, estimated_q_cache) = _run_inner_cv(
        seeds, df_eval, features, clf_features, y, groups, folds,
        needed_idx, full_needed, df_full)

    print(f"\nInner CV cached. Total time: {time.time() - t_start:.0f}s")
    print(f"Mean R: {r_var_cache.mean():.5f}  "
          f"Mean estimated Q (pre-floor): {estimated_q_cache.mean():.5f}")

    # =========================================================================
    # STEP 2: Save cache and spawn workers per P0 row
    # =========================================================================
    print(f"\n{line}")
    print("STEP 2: Sweeping P0 x Q_FLOOR grid (one subprocess per P0 row)")
    print(line)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / "_kf_sweep_cache.pkl"
    cache = {
        "full_pred_cache": full_pred_cache,
        "player_inner_cache": player_inner_cache,
        "r_var_cache": r_var_cache,
        "estimated_q_cache": estimated_q_cache,
        "y": y, "groups": groups, "sel": sel, "prev": prev,
        "recent": recent, "anchor": anchor, "inter_pos": inter_pos,
        "tier": tier, "anchor_kind": anchor_kind,
        "folds": folds, "features_swap": features_swap,
        "clf_features": clf_features,
        "cat_all": cat_all, "lo_all": lo_all, "hi_all": hi_all,
        "is_ext_all": is_ext_all, "ext_cap_all": ext_cap_all,
        "mech_all": mech_all, "fold_of": fold_of,
        "seeds": seeds, "df_eval": df_eval,
    }
    with open(cache_path, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    cache_mb = cache_path.stat().st_size / 1e6
    print(f"Cache saved ({cache_mb:.1f} MB)")

    # Free memory before spawning workers
    del (cache, inner_oof_cache, full_pred_cache, player_inner_cache,
         df_full, full_needed, df_eval)
    gc.collect()

    results = []
    for pi, p0_val in enumerate(P0_GRID):
        t_row = time.time()
        print(f"\n  P0 row {pi+1}/{len(P0_GRID)}: P0={p0_val:.5f} "
              f"({len(Q_FLOOR_GRID)} Q_FLOOR combos)...", flush=True)
        cmd = [sys.executable, __file__,
               "--seeds", str(args.seeds),
               "--worker", str(p0_val)]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300,
                              encoding="utf-8")
        if proc.returncode != 0:
            print(f"  ERROR in worker for P0={p0_val}:")
            print(proc.stderr[-500:] if proc.stderr else "(no stderr)")
            continue
        # Worker prints progress to stderr, results JSON to stdout
        if proc.stderr:
            for line_txt in proc.stderr.strip().split("\n"):
                print(f"  {line_txt}")
        batch = json.loads(proc.stdout.strip())
        results.extend(batch)
        print(f"  P0={p0_val:.5f} done in {time.time() - t_row:.0f}s",
              flush=True)

    # Clean up cache file
    cache_path.unlink(missing_ok=True)

    if not results:
        print("ERROR: no results collected")
        return

    # =========================================================================
    # STEP 3: Report
    # =========================================================================
    total_time = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"GRID SEARCH COMPLETE  ({total_time:.0f}s total, "
          f"{total_time / 60:.1f} min)")
    print("=" * 70)

    results.sort(key=lambda r: r["s3_a1_r2"], reverse=True)

    default_matches = [i for i, r in enumerate(results) if r["is_default"]]
    default_rank = default_matches[0] + 1 if default_matches else -1

    print(f"\n{'Rank':>4s}  {'P0':>8s}  {'Q_FLOOR':>8s}  "
          f"{'S3 A1 R2':>9s}  {'S3 MAE':>8s}  "
          f"{'S2 A1 R2':>9s}  {'S2 MAE':>8s}  "
          f"{'S3 sel R2':>10s}  {'S3 A2 R2':>9s}")
    print("-" * 100)

    for i, r in enumerate(results):
        tag = " **" if r.get("is_default") else ""
        print(f"{i+1:4d}  {r['p0']:8.5f}  {r['q_floor']:8.4f}  "
              f"{r['s3_a1_r2']:9.5f}  ${r['s3_mae_m']:6.2f}M  "
              f"{r['s2_a1_r2']:9.5f}  ${r['s2_mae_m']:6.2f}M  "
              f"{r['s3_sel_r2']:10.5f}  {r['s3_a2_r2']:9.5f}{tag}")

    best = results[0]
    default = next((r for r in results if r.get("is_default")), None)
    print(f"\nBest combo:    P0={best['p0']:.5f}  Q_FLOOR={best['q_floor']:.4f}"
          f"  S3 A1 R2={best['s3_a1_r2']:.5f}  MAE=${best['s3_mae_m']:.2f}M")
    if default:
        print(f"Default combo: P0={DEFAULT_P0:.5f}  Q_FLOOR={DEFAULT_Q_FLOOR:.4f}"
              f"  S3 A1 R2={default['s3_a1_r2']:.5f}  "
              f"MAE=${default['s3_mae_m']:.2f}M"
              f"  (rank {default_rank}/{len(results)})")
        delta_r2 = best["s3_a1_r2"] - default["s3_a1_r2"]
        delta_mae = default["s3_mae_m"] - best["s3_mae_m"]
        print(f"Delta:         +{delta_r2:.5f} R2,  -${delta_mae:.2f}M MAE")
    else:
        delta_r2, delta_mae = 0.0, 0.0

    print(f"\n--- TOP 5 ---")
    for i, r in enumerate(results[:5]):
        tag = " <-- DEFAULT" if r.get("is_default") else ""
        print(f"  #{i+1}: P0={r['p0']:.5f}  Q_FLOOR={r['q_floor']:.4f}  "
              f"S3 A1={r['s3_a1_r2']:.5f}  MAE=${r['s3_mae_m']:.2f}M{tag}")

    print(f"\n--- BOTTOM 5 ---")
    for i, r in enumerate(results[-5:]):
        rank = len(results) - 4 + i
        tag = " <-- DEFAULT" if r.get("is_default") else ""
        print(f"  #{rank}: P0={r['p0']:.5f}  Q_FLOOR={r['q_floor']:.4f}  "
              f"S3 A1={r['s3_a1_r2']:.5f}  MAE=${r['s3_mae_m']:.2f}M{tag}")

    # Save JSON
    out_dir = OUTPUTS_DIR / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "description": "KF hyperparameter grid search: P0 x Q_FLOOR",
        "seeds": list(range(args.seeds)),
        "p0_grid": P0_GRID,
        "q_floor_grid": Q_FLOOR_GRID,
        "n_combos": len(results),
        "default_p0": DEFAULT_P0,
        "default_q_floor": DEFAULT_Q_FLOOR,
        "default_rank": default_rank,
        "best": {
            "p0": best["p0"], "q_floor": best["q_floor"],
            "s3_a1_r2": best["s3_a1_r2"], "s3_mae_m": best["s3_mae_m"],
        },
        "default_result": {
            "s3_a1_r2": default["s3_a1_r2"],
            "s3_mae_m": default["s3_mae_m"],
        } if default else None,
        "delta_r2_best_vs_default": delta_r2,
        "delta_mae_best_vs_default_m": delta_mae,
        "results": results,
        "runtime_s": round(total_time, 1),
    }
    out_path = out_dir / "kf_hyperparam_sweep_p0.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
