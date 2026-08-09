"""B1 rolling-forward veto for the kf_market_value SWAP candidate.

Layer B of the evaluation protocol: train on every season < T, score season T,
for T in 2024-2026 — the split geometry `predict.py` actually faces. The
selection-layer result (A) for the SWAP arm sits just under the gate
(dSel +0.00331, t +1.75, with the kf-active segment at +0.0050); this harness
asks whether that effect survives a split the design was never iterated
against. A candidate whose A-gain came from fitting the selection pool shows
nothing here; a real pricing signal should reproduce.

Fold-honesty inside each origin, mirroring the A-layer harness
(scripts/ablation_kf_market_value_full.py):

    for each origin T, seed:
        champion: fit 21 features (prev_cap_pct) on seasons < T, score T
        swap:     4-fold inner GroupKFold (by player) over seasons < T
                    -> fold-honest inner OOF (gives R and Q)
                    -> per-inner-model predictions on the df_full
                       measurement subset
                  kf per row from the inner world only (train rows use the
                  model that excluded their player; rows whose player never
                  entered the window use the 4-model mean), then fit the
                  21-feature SWAP arm on seasons < T and score T.

Every quantity the kf of a season-T row consumes — the anchor, the
measurement seasons, R, Q — is computed from seasons < T. Nothing from
season T enters in any capacity, so this veto is free even of the
single-level-CV channel layer A pays.

The anchor map itself (scripts/ablation_kf_market_value_full.build_anchor_map)
is time-causal per row: anchors and measurement seasons all predate the row's
own season.

Run:  python scripts/eval_kf_forward.py [--seeds 10]
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

from config import OUTPUTS_DIR
from src.model.evaluate_suite import load_evaluation_frame, make_champion_fitter
from src.model.route_mixture import attach_clf_features
from src.model.train import FEATURE_COLS, TARGET
from scripts.ablation_kf_market_value_full import (
    P0, KF_COL, N_INNER, prepare_full_frame, build_anchor_map,
    fit_champion_models, predict_champion, estimate_q, kalman_update,
)

ORIGINS = (2024, 2025, 2026)


def bootstrap_delta_ci(y, pred_a, pred_b, n: int = 4000,
                       seed: int = 0) -> tuple[float, float]:
    """Percentile CI for R2(b) - R2(a), paired by resampled rows."""
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if np.var(y[idx]) > 0:
            draws.append(r2_score(y[idx], pred_b[idx])
                         - r2_score(y[idx], pred_a[idx]))
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()
    seeds = list(range(args.seeds))
    t_start = time.time()

    line = "=" * 70
    print(line)
    print("B1 ROLLING-FORWARD VETO — kf_market_value SWAP candidate")
    print(line)

    df_eval, features = load_evaluation_frame()
    df_eval, clf_features = attach_clf_features(df_eval)
    assert set(features) == set(FEATURE_COLS)
    y = df_eval[TARGET].values
    groups = df_eval["player_name_norm"].values
    prev = df_eval["prev_cap_pct"].values
    season = df_eval["season"].values.astype(int)
    cap_m = df_eval["cap"].values / 1e6

    print("\nPreparing the full frame and the anchor map...")
    df_full = prepare_full_frame(df_eval, features)
    (inter_idx, needed_idx, tier, anchor_val,
     _anchor_pro) = build_anchor_map(df_eval, df_full)
    anchor = np.where(tier > 0, anchor_val, prev)
    pos_of = {fi: k for k, fi in enumerate(needed_idx)}
    inter_pos = [[pos_of[fi] for fi in lst] for lst in inter_idx]
    full_needed = df_full.iloc[needed_idx]

    features_swap = [KF_COL if f == "prev_cap_pct" else f for f in features]
    fitter_21 = make_champion_fitter(clf_features)
    fitter_swap = make_champion_fitter(clf_features)

    fwd_c = np.full(len(df_eval), np.nan)
    fwd_s = np.full(len(df_eval), np.nan)
    detail = {}
    for T in ORIGINS:
        te = np.flatnonzero(season == T)
        tr = np.flatnonzero(season < T)
        if len(te) < 10 or len(tr) < 200:
            continue
        t_orig = time.time()
        train_df, test_df = df_eval.iloc[tr], df_eval.iloc[te]
        g_tr = groups[tr]
        inner_folds = list(GroupKFold(n_splits=N_INNER)
                           .split(train_df, y[tr], g_tr))
        acc_c = np.zeros(len(te))
        acc_s = np.zeros(len(te))
        for seed in seeds:
            acc_c += fitter_21(train_df, test_df, features, seed)

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
            r_var = float(np.var(y[tr] - inner_oof, ddof=1))
            q_var, _, _ = estimate_q(df_full, set(g_tr), r_var)

            def kf_rows(rows):
                out = np.full(len(rows), np.nan)
                for k, i in enumerate(rows):
                    if not np.isfinite(anchor[i]):
                        continue
                    pl = inter_pos[i]
                    if not pl:
                        out[k] = anchor[i]
                        continue
                    j = player_inner.get(groups[i])
                    zs = (full_pred[j, pl] if j is not None
                          else full_mean[pl])
                    p0 = r_var if tier[i] == 2 else P0
                    out[k] = kalman_update(anchor[i], zs, q_var, r_var, p0)
                return out

            kf_tr, kf_te = kf_rows(tr), kf_rows(te)
            fill = (float(np.nanmedian(kf_tr)) if np.isfinite(kf_tr).any()
                    else float(np.nanmedian(prev)))
            tr2, te2 = train_df.copy(), test_df.copy()
            tr2[KF_COL] = np.where(np.isfinite(kf_tr), kf_tr, fill)
            te2[KF_COL] = np.where(np.isfinite(kf_te), kf_te, fill)
            acc_s += fitter_swap(tr2, te2, features_swap, seed)

        fwd_c[te] = acc_c / len(seeds)
        fwd_s[te] = acc_s / len(seeds)
        r2c = r2_score(y[te], fwd_c[te])
        r2s = r2_score(y[te], fwd_s[te])
        mae_c = float(np.abs((fwd_c[te] - y[te]) * cap_m[te]).mean())
        mae_s = float(np.abs((fwd_s[te] - y[te]) * cap_m[te]).mean())
        detail[int(T)] = {"n": int(len(te)), "champ_r2": float(r2c),
                          "swap_r2": float(r2s), "delta": float(r2s - r2c),
                          "champ_mae_m": mae_c, "swap_mae_m": mae_s}
        print(f"  origin {T}: n={len(te):3d}  R2 {r2c:.4f} -> {r2s:.4f} "
              f"({r2s - r2c:+.4f})   MAE ${mae_c:.2f}M -> ${mae_s:.2f}M "
              f"({mae_s - mae_c:+.2f})   {time.time() - t_orig:.0f}s",
              flush=True)

    scored = ~np.isnan(fwd_c)
    r2c_all = r2_score(y[scored], fwd_c[scored])
    r2s_all = r2_score(y[scored], fwd_s[scored])
    delta = r2s_all - r2c_all
    lo, hi = bootstrap_delta_ci(y[scored], fwd_c[scored], fwd_s[scored])
    mae_c_all = float(np.abs((fwd_c[scored] - y[scored]) * cap_m[scored]).mean())
    mae_s_all = float(np.abs((fwd_s[scored] - y[scored]) * cap_m[scored]).mean())

    print(f"\n{line}")
    print("B1 pooled (2024-26)")
    print(line)
    print(f"  champion (prev_cap_pct)  R2={r2c_all:.4f}  MAE=${mae_c_all:.2f}M"
          f"  (n={int(scored.sum())})")
    print(f"  SWAP (kf_market_value)   R2={r2s_all:.4f}  MAE=${mae_s_all:.2f}M")
    print(f"  delta={delta:+.4f}  bootstrap 95% CI [{lo:+.4f}, {hi:+.4f}]")
    origins_pos = sum(1 for d in detail.values() if d["delta"] > 0)
    print(f"  verdict: {'SAME DIRECTION as layer A' if delta > 0 else 'CONTRADICTS layer A'}"
          f" — {origins_pos}/{len(detail)} origins positive")

    out = OUTPUTS_DIR / "models" / "kf_forward_veto.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "seeds": seeds, "by_origin": detail,
        "pooled": {"n": int(scored.sum()), "champ_r2": float(r2c_all),
                   "swap_r2": float(r2s_all), "delta": float(delta),
                   "delta_ci95": [lo, hi],
                   "champ_mae_m": mae_c_all, "swap_mae_m": mae_s_all},
        "runtime_s": round(time.time() - t_start, 1),
    }, indent=2), encoding="utf-8")
    print(f"\nSaved {out}   (total {time.time() - t_start:.0f}s)")


if __name__ == "__main__":
    main()
